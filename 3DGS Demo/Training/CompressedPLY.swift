import Foundation
import simd

/// The compressed PLY format of SuperSplat and PlayCanvas (".compressed.ply"), which the app saves its splats in.
///
/// SPZ, used before, keeps the view-dependent colour (spherical harmonics) in 4 or 5 bits: on a bowl and a bedroom
/// that cost 0.14 and 0.21 dB held out, all of SPZ's loss. Here the splats go in chunks of 256 neighbours (Morton
/// order), each chunk with its own ranges: positions and sizes in 11/10/11 bits of the chunk's range, colour in 8 bits
/// of it, opacity in 8 bits, rotation in 2+10+10+10 bits (the three smallest components), and the spherical harmonics
/// in 8 bits over -4...4. Measured loss 0.01 dB. About 40 bytes a splat with SH 2 (the trainer's float PLY: 248).
nonisolated enum CompressedPLY {
    static let fileSuffix = ".compressed.ply"

    enum FormatError: LocalizedError {
        case notPLY, unsupported(String), truncated

        var errorDescription: String? {
            switch self {
            case .notPLY: "The file is not a PLY file."
            case .unsupported(let what): "Unsupported PLY file: \(what)."
            case .truncated: "The PLY file is shorter than its header says."
            }
        }
    }

    /// Splats decoded for rendering: linear sizes and opacities, rotations real part first, and per splat its
    /// `(degree + 1)²` colour coefficients (base colour first) as RGB triplets.
    struct Splats {
        var count: Int
        var positions: [SIMD3<Float>]
        var scales: [SIMD3<Float>]
        var rotations: [SIMD4<Float>]
        var opacities: [Float]
        var sh: [Float]
        var degree: Int
    }

    private static let shC0: Float = 0.28209479177387814

    // MARK: Header

    private struct Element {
        var name: String
        var count: Int
        var properties: [(type: String, name: String)]
    }

    private static func header(of data: Data) throws -> (elements: [Element], bodyOffset: Int) {
        let marker = Data("end_header\n".utf8)
        guard data.starts(with: Data("ply\n".utf8)),
              let end = data.range(of: marker, in: 0..<min(data.count, 1 << 16)),
              let text = String(data: data[..<end.lowerBound], encoding: .ascii) else { throw FormatError.notPLY }
        var elements: [Element] = []
        for line in text.split(separator: "\n") {
            let words = line.split(separator: " ").map(String.init)
            switch words.first {
            case "format":
                guard words.count >= 2, words[1] == "binary_little_endian" else { throw FormatError.unsupported("not binary little-endian") }
            case "element":
                guard words.count == 3, let count = Int(words[2]) else { throw FormatError.unsupported("element line") }
                elements.append(Element(name: words[1], count: count, properties: []))
            case "property":
                guard words.count == 3, !elements.isEmpty else { throw FormatError.unsupported("property line") }
                elements[elements.count - 1].properties.append((words[1], words[2]))
            default:
                break
            }
        }
        return (elements, end.upperBound)
    }

    /// Whether the file at `url` is in this format (its header has the chunk element and the packed properties).
    static func isCompressed(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        // Only the header is text: the binary data after it would make the whole read fail to decode.
        guard let start = try? handle.read(upToCount: 1 << 16),
              let end = start.range(of: Data("end_header\n".utf8)),
              let text = String(data: start[..<end.lowerBound], encoding: .ascii) else { return false }
        return text.hasPrefix("ply\n") && text.contains("element chunk") && text.contains("packed_position")
    }

    // MARK: Writing

    private static func packUnorm(_ value: Float, _ bits: Int) -> UInt32 {
        let top = Float((1 << bits) - 1)
        guard value.isFinite else { return 0 }
        return UInt32(min(max((value * top + 0.5).rounded(.down), 0), top))
    }

    /// As PlayCanvas's writer: 0 below the range, 1 above it, 0 for a range under 1e-5.
    private static func normalized(_ value: Float, _ low: Float, _ high: Float) -> Float {
        if value <= low { return 0 }
        if value >= high { return 1 }
        return high - low < 1e-5 ? 0 : (value - low) / (high - low)
    }

    private static func spreadBits(_ value: UInt64) -> UInt64 {
        var x = value & 0x3ff
        x = (x | (x << 16)) & 0x30000ff
        x = (x | (x << 8)) & 0x300f00f
        x = (x | (x << 4)) & 0x30c30c3
        x = (x | (x << 2)) & 0x9249249
        return x
    }

    /// Converts the trainer's float PLY (3DGS layout) at `source` to this format at `destination`, keeping the
    /// spherical-harmonic bands up to `shDegree` (the trainer pads to degree 3 with zeros). Returns the splat count.
    @discardableResult
    static func write(from source: URL, to destination: URL, shDegree: Int) throws -> Int {
        let data = try Data(contentsOf: source, options: .alwaysMapped)
        let (elements, bodyOffset) = try header(of: data)
        guard elements.count == 1, let vertex = elements.first, vertex.name == "vertex",
              vertex.properties.allSatisfy({ $0.type == "float" }) else { throw FormatError.unsupported("not a float splat PLY") }
        let names = vertex.properties.map(\.name)
        func column(_ name: String) throws -> Int {
            guard let index = names.firstIndex(of: name) else { throw FormatError.unsupported("no \(name)") }
            return index
        }
        let stride = names.count, n = vertex.count
        guard data.count >= bodyOffset + n * stride * 4 else { throw FormatError.truncated }
        let position = try ["x", "y", "z"].map(column)
        let dc = try ["f_dc_0", "f_dc_1", "f_dc_2"].map(column)
        let scale = try ["scale_0", "scale_1", "scale_2"].map(column)
        let rotation = try ["rot_0", "rot_1", "rot_2", "rot_3"].map(column)   // w, x, y, z
        let opacity = try column("opacity")
        let sourceRest = names.filter { $0.hasPrefix("f_rest_") }.count / 3
        let rest = min(max(0, (shDegree + 1) * (shDegree + 1) - 1), sourceRest)
        var restColumns: [Int] = []   // channel by channel, as the 3DGS layout stores them
        for channel in 0..<3 {
            for k in 0..<rest { restColumns.append(try column("f_rest_\(channel * sourceRest + k)")) }
        }

        var output = Data()
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            func value(_ splat: Int, _ column: Int) -> Float {
                raw.loadUnaligned(fromByteOffset: bodyOffset + (splat * stride + column) * 4, as: Float.self)
            }
            func point(_ i: Int) -> SIMD3<Float> { SIMD3(value(i, position[0]), value(i, position[1]), value(i, position[2])) }

            // Morton order over the bounding box: each chunk of 256 then covers a small region and its ranges are tight.
            var low = SIMD3<Float>(repeating: .infinity), high = SIMD3<Float>(repeating: -.infinity)
            for i in 0..<n {
                let p = point(i)
                if p.x.isFinite, p.y.isFinite, p.z.isFinite { low = simd_min(low, p); high = simd_max(high, p) }
            }
            guard low.x <= high.x else { throw FormatError.unsupported("no finite positions") }
            let extent = simd_max(high - low, SIMD3(repeating: 1e-9))
            var keys = [UInt64](repeating: 0, count: n)
            for i in 0..<n {
                let t = simd_clamp((point(i) - low) / extent, SIMD3(repeating: 0), SIMD3(repeating: 1))
                let q = SIMD3<UInt64>(UInt64(t.x * 1023), UInt64(t.y * 1023), UInt64(t.z * 1023))
                let code = spreadBits(q.x) | (spreadBits(q.y) << 1) | (spreadBits(q.z) << 2)
                keys[i] = (code << 32) | UInt64(i)
            }
            keys.sort()
            let order = keys.map { Int($0 & 0xffff_ffff) }

            let chunks = (n + 255) / 256
            var chunkData = [Float](repeating: 0, count: chunks * 18)
            var packed = [UInt32](repeating: 0, count: n * 4)
            var shBytes = [UInt8](repeating: 0, count: n * rest * 3)
            for chunk in 0..<chunks {
                let members = order[(chunk * 256)..<min(n, chunk * 256 + 256)]
                var pLow = SIMD3<Float>(repeating: .infinity), pHigh = SIMD3<Float>(repeating: -.infinity)
                var sLow = pLow, sHigh = pHigh, cLow = pLow, cHigh = pHigh
                func logScale(_ i: Int) -> SIMD3<Float> {
                    simd_clamp(SIMD3(value(i, scale[0]), value(i, scale[1]), value(i, scale[2])), SIMD3(repeating: -20), SIMD3(repeating: 20))
                }
                func colour(_ i: Int) -> SIMD3<Float> { SIMD3(value(i, dc[0]), value(i, dc[1]), value(i, dc[2])) * shC0 + 0.5 }
                for i in members {
                    let p = point(i), s = logScale(i), c = colour(i)
                    if p.x.isFinite, p.y.isFinite, p.z.isFinite { pLow = simd_min(pLow, p); pHigh = simd_max(pHigh, p) }
                    if s.x.isFinite, s.y.isFinite, s.z.isFinite { sLow = simd_min(sLow, s); sHigh = simd_max(sHigh, s) }
                    if c.x.isFinite, c.y.isFinite, c.z.isFinite { cLow = simd_min(cLow, c); cHigh = simd_max(cHigh, c) }
                }
                // A chunk with no finite values keeps zero ranges.
                if !(pLow.x <= pHigh.x) { pLow = .zero; pHigh = .zero }
                if !(sLow.x <= sHigh.x) { sLow = .zero; sHigh = .zero }
                if !(cLow.x <= cHigh.x) { cLow = .zero; cHigh = .zero }
                let ranges: [Float] = [pLow.x, pLow.y, pLow.z, pHigh.x, pHigh.y, pHigh.z,
                                       sLow.x, sLow.y, sLow.z, sHigh.x, sHigh.y, sHigh.z,
                                       cLow.x, cLow.y, cLow.z, cHigh.x, cHigh.y, cHigh.z]
                for k in 0..<18 { chunkData[chunk * 18 + k] = ranges[k] }

                for (slot, i) in zip((chunk * 256)..<(chunk * 256 + members.count), members) {
                    let p = point(i), s = logScale(i), c = colour(i)
                    packed[slot * 4] = packUnorm(normalized(p.x, pLow.x, pHigh.x), 11) << 21
                        | packUnorm(normalized(p.y, pLow.y, pHigh.y), 10) << 11
                        | packUnorm(normalized(p.z, pLow.z, pHigh.z), 11)

                    // In the PLY's own order (rot_0 is the real part): the format's "largest component" index refers to it.
                    var q = [value(i, rotation[0]), value(i, rotation[1]), value(i, rotation[2]), value(i, rotation[3])]
                    let length = sqrt(q.reduce(0) { $0 + $1 * $1 })
                    q = length.isFinite && length > 0 ? q.map { $0 / length } : [1, 0, 0, 0]
                    var largest = 0
                    for k in 1..<4 where abs(q[k]) > abs(q[largest]) { largest = k }
                    if q[largest] < 0 { q = q.map { -$0 } }
                    var r = UInt32(largest)
                    for k in 0..<4 where k != largest { r = (r << 10) | packUnorm(q[k] * 0.70710678 + 0.5, 10) }
                    packed[slot * 4 + 1] = r

                    packed[slot * 4 + 2] = packUnorm(normalized(s.x, sLow.x, sHigh.x), 11) << 21
                        | packUnorm(normalized(s.y, sLow.y, sHigh.y), 10) << 11
                        | packUnorm(normalized(s.z, sLow.z, sHigh.z), 11)

                    let alpha = 1 / (1 + exp(-value(i, opacity)))
                    packed[slot * 4 + 3] = packUnorm(normalized(c.x, cLow.x, cHigh.x), 8) << 24
                        | packUnorm(normalized(c.y, cLow.y, cHigh.y), 8) << 16
                        | packUnorm(normalized(c.z, cLow.z, cHigh.z), 8) << 8
                        | packUnorm(alpha, 8)

                    for (k, col) in restColumns.enumerated() {
                        let v = value(i, col)
                        shBytes[slot * rest * 3 + k] = v.isFinite ? UInt8(min(max(((v / 8 + 0.5) * 256).rounded(.towardZero), 0), 255)) : 128
                    }
                }
            }

            var text = "ply\nformat binary_little_endian 1.0\ncomment written by Voxelio\nelement chunk \(chunks)\n"
            for name in ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
                         "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
                         "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"] {
                text += "property float \(name)\n"
            }
            text += "element vertex \(n)\n"
            for name in ["packed_position", "packed_rotation", "packed_scale", "packed_color"] { text += "property uint \(name)\n" }
            if rest > 0 {
                text += "element sh \(n)\n"
                for k in 0..<(rest * 3) { text += "property uchar f_rest_\(k)\n" }
            }
            text += "end_header\n"
            output.reserveCapacity(text.utf8.count + chunkData.count * 4 + packed.count * 4 + shBytes.count)
            output.append(Data(text.utf8))
            chunkData.withUnsafeBytes { output.append(contentsOf: $0) }     // little-endian on Apple hardware
            packed.withUnsafeBytes { output.append(contentsOf: $0) }
            output.append(contentsOf: shBytes)
        }
        try output.write(to: destination, options: .atomic)
        return n
    }

    // MARK: Reading

    static func read(from url: URL) throws -> Splats {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let (elements, bodyOffset) = try header(of: data)
        guard let chunk = elements.first(where: { $0.name == "chunk" }),
              let vertex = elements.first(where: { $0.name == "vertex" }) else { throw FormatError.unsupported("not a compressed PLY") }
        let vertexNames = vertex.properties.map(\.name)
        guard vertexNames == ["packed_position", "packed_rotation", "packed_scale", "packed_color"],
              chunk.properties.allSatisfy({ $0.type == "float" }), chunk.properties.count == 12 || chunk.properties.count == 18
        else { throw FormatError.unsupported("unexpected compressed PLY layout") }
        let sh = elements.first(where: { $0.name == "sh" })
        let n = vertex.count, chunkStride = chunk.properties.count
        let restPerChannel = (sh?.properties.count ?? 0) / 3
        let degree = switch restPerChannel { case 3: 1; case 8: 2; case 15: 3; default: 0 }
        let coefficients = (degree + 1) * (degree + 1)

        // Element data follows the header in the order the elements are declared.
        var offsets: [String: Int] = [:]
        var offset = bodyOffset
        for element in elements {
            offsets[element.name] = offset
            let size = element.properties.reduce(0) { total, property in
                total + (property.type == "uchar" || property.type == "uint8" ? 1 : 4)
            }
            offset += element.count * size
        }
        guard data.count >= offset, chunk.count >= (n + 255) / 256 else { throw FormatError.truncated }

        var result = Splats(count: n, positions: .init(repeating: .zero, count: n), scales: .init(repeating: .zero, count: n),
                            rotations: .init(repeating: .zero, count: n), opacities: .init(repeating: 0, count: n),
                            sh: .init(repeating: 0, count: n * coefficients * 3), degree: degree)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let chunkBase = offsets["chunk"]!, vertexBase = offsets["vertex"]!, shBase = offsets["sh"] ?? 0
            func unorm(_ value: UInt32, _ bits: Int) -> Float { Float(value) / Float((1 << bits) - 1) }
            func chunkValue(_ c: Int, _ k: Int) -> Float { raw.loadUnaligned(fromByteOffset: chunkBase + (c * chunkStride + k) * 4, as: Float.self) }
            func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
            let norm: Float = 1 / 0.70710678
            for i in 0..<n {
                let c = i / 256
                func word(_ k: Int) -> UInt32 { raw.loadUnaligned(fromByteOffset: vertexBase + (i * 4 + k) * 4, as: UInt32.self) }
                let p = word(0), r = word(1), s = word(2), col = word(3)
                result.positions[i] = SIMD3(lerp(chunkValue(c, 0), chunkValue(c, 3), unorm(p >> 21, 11)),
                                            lerp(chunkValue(c, 1), chunkValue(c, 4), unorm((p >> 11) & 0x3ff, 10)),
                                            lerp(chunkValue(c, 2), chunkValue(c, 5), unorm(p & 0x7ff, 11)))
                result.scales[i] = SIMD3(exp(lerp(chunkValue(c, 6), chunkValue(c, 9), unorm(s >> 21, 11))),
                                         exp(lerp(chunkValue(c, 7), chunkValue(c, 10), unorm((s >> 11) & 0x3ff, 10))),
                                         exp(lerp(chunkValue(c, 8), chunkValue(c, 11), unorm(s & 0x7ff, 11))))
                let a = (unorm((r >> 20) & 0x3ff, 10) - 0.5) * norm
                let b = (unorm((r >> 10) & 0x3ff, 10) - 0.5) * norm
                let d = (unorm(r & 0x3ff, 10) - 0.5) * norm
                let m = sqrt(max(0, 1 - (a * a + b * b + d * d)))
                // rot_0...rot_3 of the PLY: the real part first, as RealityKit wants it.
                result.rotations[i] = switch r >> 30 {
                case 0: SIMD4(m, a, b, d)
                case 1: SIMD4(a, m, b, d)
                case 2: SIMD4(a, b, m, d)
                default: SIMD4(a, b, d, m)
                }
                var rgb = SIMD3(unorm(col >> 24, 8), unorm((col >> 16) & 0xff, 8), unorm((col >> 8) & 0xff, 8))
                if chunkStride == 18 {
                    rgb = SIMD3(lerp(chunkValue(c, 12), chunkValue(c, 15), rgb.x),
                                lerp(chunkValue(c, 13), chunkValue(c, 16), rgb.y),
                                lerp(chunkValue(c, 14), chunkValue(c, 17), rgb.z))
                }
                result.opacities[i] = unorm(col & 0xff, 8)
                let base = i * coefficients * 3
                let dcValue = (rgb - 0.5) / shC0
                result.sh[base] = dcValue.x; result.sh[base + 1] = dcValue.y; result.sh[base + 2] = dcValue.z
                for k in 1..<coefficients {
                    for channel in 0..<3 {
                        let byte = raw[shBase + i * restPerChannel * 3 + channel * restPerChannel + (k - 1)]
                        let t: Float = byte == 0 ? 0 : byte == 255 ? 1 : (Float(byte) + 0.5) / 256
                        result.sh[base + k * 3 + channel] = (t - 0.5) * 8
                    }
                }
            }
        }
        return result
    }
}
