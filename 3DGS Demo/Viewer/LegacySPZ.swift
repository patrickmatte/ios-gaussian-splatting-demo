import Foundation
import simd

/// Reads the SPZ files that scans trained before the compressed PLY were saved in (SPZ versions 2 and 3,
/// gzip-compressed), so they still open in the viewer. Decodes as spz-swift does, including its conversion back from
/// SPZ's coordinate frame (right, up, back) to the frame the splat was trained in (x, -y, -z of the stored values).
nonisolated enum LegacySPZ {
    enum FormatError: LocalizedError {
        case notSPZ, unsupportedVersion(Int), truncated

        var errorDescription: String? {
            switch self {
            case .notSPZ: "The file is not an SPZ file."
            case .unsupportedVersion(let version): "Unsupported SPZ version \(version)."
            case .truncated: "The SPZ file is shorter than its header says."
            }
        }
    }

    static func read(from url: URL) throws -> CompressedPLY.Splats {
        let raw = try inflate(Data(contentsOf: url))
        guard raw.count >= 16 else { throw FormatError.truncated }
        return try raw.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            guard bytes.loadUnaligned(as: UInt32.self) == 0x5053_474e else { throw FormatError.notSPZ }
            let version = Int(bytes.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
            guard version == 2 || version == 3 else { throw FormatError.unsupportedVersion(version) }
            let n = Int(bytes.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let fileDegree = Int(bytes[12]), fractionalBits = Int(bytes[13])
            guard fileDegree <= 3 else { throw FormatError.notSPZ }
            let rest = (fileDegree + 1) * (fileDegree + 1) - 1
            let rotationSize = version == 3 ? 4 : 3

            let positionBase = 16
            let alphaBase = positionBase + n * 9
            let colourBase = alphaBase + n
            let scaleBase = colourBase + n * 3
            let rotationBase = scaleBase + n * 3
            let shBase = rotationBase + n * rotationSize
            guard bytes.count >= shBase + n * rest * 3 else { throw FormatError.truncated }

            // The app's writer padded to degree 3 with zeros (stored as 128): keep only the bands that carry data.
            var used = 0
            for i in 0..<n {
                for k in stride(from: rest, to: used, by: -1) where (0..<3).contains(where: { bytes[shBase + (i * rest + k - 1) * 3 + $0] != 128 }) {
                    used = k
                    break
                }
                if used == rest { break }
            }
            let degree = switch used { case 0: 0; case 1...3: 1; case 4...8: 2; default: 3 }
            let coefficients = (degree + 1) * (degree + 1)

            // Back from right-up-back: y and z change sign, and with them the rotation's y and z and the SH
            // coefficients that are odd in y or z (spz-swift's coordinateConverter(from: .rub, to: .rdf)).
            let shSigns: [Float] = [-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1]
            let positionScale = 1 / Float(1 << fractionalBits)
            var result = CompressedPLY.Splats(
                count: n, positions: .init(repeating: .zero, count: n), scales: .init(repeating: .zero, count: n),
                rotations: .init(repeating: .zero, count: n), opacities: .init(repeating: 0, count: n),
                sh: .init(repeating: 0, count: n * coefficients * 3), degree: degree)
            for i in 0..<n {
                var p = SIMD3<Float>.zero
                for axis in 0..<3 {
                    let at = positionBase + i * 9 + axis * 3
                    var fixed = Int32(bytes[at]) | Int32(bytes[at + 1]) << 8 | Int32(bytes[at + 2]) << 16
                    if fixed & 0x80_0000 != 0 { fixed |= Int32(bitPattern: 0xff00_0000) }
                    p[axis] = Float(fixed) * positionScale
                }
                result.positions[i] = p * SIMD3(1, -1, -1)
                result.scales[i] = SIMD3(exp(Float(bytes[scaleBase + i * 3]) / 16 - 10),
                                         exp(Float(bytes[scaleBase + i * 3 + 1]) / 16 - 10),
                                         exp(Float(bytes[scaleBase + i * 3 + 2]) / 16 - 10))
                result.opacities[i] = Float(bytes[alphaBase + i]) / 255

                // (x, y, z, w) as stored.
                var q = SIMD4<Float>.zero
                if version == 3 {
                    var packed = bytes.loadUnaligned(fromByteOffset: rotationBase + i * 4, as: UInt32.self)
                    let largest = Int(packed >> 30)
                    var squares: Float = 0
                    for k in stride(from: 3, through: 0, by: -1) where k != largest {
                        let magnitude = Float(packed & 0x1ff) / 511 * 0.70710678
                        q[k] = packed & 0x200 != 0 ? -magnitude : magnitude
                        squares += q[k] * q[k]
                        packed >>= 10
                    }
                    q[largest] = sqrt(max(0, 1 - squares))
                } else {
                    let xyz = SIMD3<Float>(Float(bytes[rotationBase + i * 3]), Float(bytes[rotationBase + i * 3 + 1]),
                                           Float(bytes[rotationBase + i * 3 + 2])) / 127.5 - 1
                    q = SIMD4(xyz, sqrt(max(0, 1 - simd_dot(xyz, xyz))))
                }
                let length = simd_length(q)
                q = length > 0 ? q / length : SIMD4(0, 0, 0, 1)
                result.rotations[i] = SIMD4(q.w, q.x, -q.y, -q.z)

                let base = i * coefficients * 3
                for channel in 0..<3 {
                    result.sh[base + channel] = (Float(bytes[colourBase + i * 3 + channel]) / 255 - 0.5) / 0.15
                }
                for k in 1..<coefficients {
                    for channel in 0..<3 {
                        let byte = bytes[shBase + (i * rest + k - 1) * 3 + channel]
                        result.sh[base + k * 3 + channel] = shSigns[k - 1] * (Float(byte) - 128) / 128
                    }
                }
            }
            return result
        }
    }

    /// The deflate stream inside the gzip wrapper.
    private static func inflate(_ file: Data) throws -> Data {
        let bytes = [UInt8](file)
        guard bytes.count > 18, bytes[0] == 0x1f, bytes[1] == 0x8b else { throw FormatError.notSPZ }
        let flags = bytes[3]
        var offset = 10
        if flags & 0x04 != 0 { offset += 2 + (Int(bytes[offset]) | Int(bytes[offset + 1]) << 8) }   // extra field
        if flags & 0x08 != 0 { while offset < bytes.count, bytes[offset] != 0 { offset += 1 }; offset += 1 }   // name
        if flags & 0x10 != 0 { while offset < bytes.count, bytes[offset] != 0 { offset += 1 }; offset += 1 }   // comment
        if flags & 0x02 != 0 { offset += 2 }   // header checksum
        guard offset < bytes.count - 8 else { throw FormatError.truncated }
        let payload = Data(bytes[offset..<(bytes.count - 8)])
        return try (payload as NSData).decompressed(using: .zlib) as Data
    }
}
