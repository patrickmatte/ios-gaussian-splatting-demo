import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision
import simd

/// Finds the subject of an orbit capture in the training photos, so the finished splat can be cut out.
///
/// Vision finds the foreground objects in each photo; the ones around the point the cameras circle
/// (`SubjectRegion`) are the subject. The masks are small PNGs kept with the scan. After training, msplat
/// drops the splats whose centre falls outside the masks in most views, so the splat of an object
/// comes out without the table, floor and room around it.
///
/// An earlier version painted the background a flat colour and trained on those photos. The splats on
/// the silhouette were pulled towards that colour and the result kept a magenta rim.
nonisolated enum SubjectMasks {
    struct Outcome: Sendable {
        /// The folder holding `<image name>.png` for every photo where the subject was found.
        var folder: URL
        var found: Int
        var total: Int
    }

    enum MaskError: LocalizedError {
        case tooFewMasks(found: Int, total: Int)

        var errorDescription: String? {
            switch self {
            case .tooFewMasks(let found, let total):
                "The subject was found in only \(found) of \(total) photos, so it cannot be cut out. Train as an object without the cut-out instead."
            }
        }
    }

    /// Long edge, in pixels, of the photos Vision looks at and of the saved masks.
    static let maskLongEdge = 960

    /// Photos in which the subject covers more or less than this share of the frame are ignored:
    /// Vision picked something else.
    static let areaRange: ClosedRange<Double> = 0.004...0.7

    /// The folder next to the scan where the masks are kept.
    static func maskFolder(scan: URL) -> URL { scan.appendingPathComponent("masks", isDirectory: true) }

    /// Makes a mask for each of `images` that has none yet. `transformsURL` is the camera list the
    /// images belong to; `scan` is where the masks are kept.
    static func generate(
        images: [URL], transformsURL: URL, scan: URL, region: SubjectRegion.Region,
        status: @escaping @Sendable (String, Double?) -> Void
    ) async throws -> Outcome {
        guard let root = (try JSONSerialization.jsonObject(with: Data(contentsOf: transformsURL))) as? [String: Any],
              let frames = root["frames"] as? [[String: Any]],
              let fx = root["fl_x"] as? Double, let fy = root["fl_y"] as? Double,
              let cx = root["cx"] as? Double, let cy = root["cy"] as? Double else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let width = (root["w"] as? Double) ?? 1920, height = (root["h"] as? Double) ?? 1440

        var poses: [String: [[Double]]] = [:]
        for frame in frames {
            guard let path = frame["file_path"] as? String, let rows = frame["transform_matrix"] as? [[Double]] else { continue }
            poses[(path as NSString).lastPathComponent] = rows
        }

        let folder = maskFolder(scan: scan)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        var found = 0
        for (index, imageURL) in images.enumerated() {
            try Task.checkCancellation()
            // A room or landscape has no subject: give up after a dozen photos instead of scanning them all.
            if index >= 12, found * 4 < index {
                throw MaskError.tooFewMasks(found: found, total: images.count)
            }
            status("Finding the subject in each photo… \(index)/\(images.count)", Double(index) / Double(max(images.count, 1)))

            let maskURL = folder.appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent + ".png")
            if FileManager.default.fileExists(atPath: maskURL.path) {
                found += 1
                continue
            }
            guard let rows = poses[imageURL.lastPathComponent] else { continue }
            autoreleasepool {
                let projected = project(region.center, rows: rows, fx: fx, fy: fy, cx: cx, cy: cy)
                if let mask = findSubject(in: imageURL, projected: projected, sphereRadius: region.radius,
                                          focal: fx, imageSize: (width, height)) {
                    writeMask(mask, to: maskURL)
                    found += 1
                }
            }
        }

        guard found >= max(8, images.count / 2) else { throw MaskError.tooFewMasks(found: found, total: images.count) }
        status("Found the subject in \(found) of \(images.count) photos", 1)
        return Outcome(folder: folder, found: found, total: images.count)
    }

    // MARK: Projection

    /// Where `point` lands in a view (pixels at the full frame size) and its depth, or nil behind the camera.
    private static func project(_ point: SIMD3<Float>, rows m: [[Double]], fx: Double, fy: Double,
                                cx: Double, cy: Double) -> (x: Double, y: Double, depth: Double)? {
        guard m.count >= 3, m[0].count >= 4 else { return nil }
        let d = SIMD3(Double(point.x) - m[0][3], Double(point.y) - m[1][3], Double(point.z) - m[2][3])
        // Camera to world is R; the OpenGL camera looks down -Z with +Y up.
        let x = m[0][0] * d.x + m[1][0] * d.y + m[2][0] * d.z
        let y = m[0][1] * d.x + m[1][1] * d.y + m[2][1] * d.z
        let z = -(m[0][2] * d.x + m[1][2] * d.y + m[2][2] * d.z)
        guard z > 0.01 else { return nil }
        return (fx * x / z + cx, cy - fy * y / z, z)
    }

    // MARK: Vision

    /// A mask of the subject at `maskLongEdge`, 255 where the subject is.
    struct Mask {
        var width: Int
        var height: Int
        var pixels: [UInt8]

        var coverage: Double { Double(pixels.reduce(0) { $0 + ($1 > 127 ? 1 : 0) }) / Double(max(pixels.count, 1)) }
    }

    private static func findSubject(in url: URL, projected: (x: Double, y: Double, depth: Double)?,
                                    sphereRadius: Float, focal: Double, imageSize: (Double, Double)) -> Mask? {
        guard let projected,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              // Unrotated pixels, like the trainer reads them.
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maskLongEdge,
                kCGImageSourceCreateThumbnailWithTransform: false,
              ] as CFDictionary) else { return nil }

        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        guard (try? handler.perform([request])) != nil,
              let observation = request.results?.first,
              !observation.allInstances.isEmpty else { return nil }

        let labels = observation.instanceMask
        CVPixelBufferLockBaseAddress(labels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(labels, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(labels) else { return nil }
        let w = CVPixelBufferGetWidth(labels), h = CVPixelBufferGetHeight(labels)
        let stride = CVPixelBufferGetBytesPerRow(labels)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Subject centre and sphere radius in mask pixels.
        let scale = Double(w) / imageSize.0
        let centreX = projected.x * scale, centreY = projected.y * scale
        let sphere = focal * scale * Double(sphereRadius) / projected.depth
        let near = 0.35 * sphere

        struct Stats { var area = 0, near = 0; var sumX = 0.0, sumY = 0.0 }
        var stats = [UInt8: Stats]()
        for y in 0..<h {
            let row = bytes + y * stride
            for x in 0..<w {
                let label = row[x]
                if label == 0 { continue }
                var s = stats[label] ?? Stats()
                s.area += 1
                s.sumX += Double(x)
                s.sumY += Double(y)
                let dx = Double(x) - centreX, dy = Double(y) - centreY
                if dx * dx + dy * dy <= near * near { s.near += 1 }
                stats[label] = s
            }
        }

        // Keep the objects at the centre of the orbit, and the ones whose middle falls inside the
        // sphere around it (fruit in a bowl, a lid next to a jar).
        var chosen = Set<UInt8>()
        for (label, s) in stats {
            let mx = s.sumX / Double(s.area) - centreX, my = s.sumY / Double(s.area) - centreY
            let inSphere = mx * mx + my * my <= sphere * sphere && Double(s.area) >= 0.002 * Double(w * h)
            if s.near > 0 || inSphere { chosen.insert(label) }
        }
        guard !chosen.isEmpty else { return nil }

        var pixels = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = bytes + y * stride
            for x in 0..<w where chosen.contains(row[x]) { pixels[y * w + x] = 255 }
        }
        let mask = Mask(width: w, height: h, pixels: pixels)
        return areaRange.contains(mask.coverage) ? mask : nil
    }

    // MARK: Mask files

    private static func writeMask(_ mask: Mask, to url: URL) {
        var pixels = mask.pixels
        let image = pixels.withUnsafeMutableBytes { raw -> CGImage? in
            CGContext(data: raw.baseAddress, width: mask.width, height: mask.height, bitsPerComponent: 8,
                      bytesPerRow: mask.width, space: CGColorSpaceCreateDeviceGray(),
                      bitmapInfo: CGImageAlphaInfo.none.rawValue)?.makeImage()
        }
        guard let image,
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}
