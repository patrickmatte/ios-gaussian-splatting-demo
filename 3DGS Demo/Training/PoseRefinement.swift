import Foundation
import Msplat
import os
@preconcurrency import RealityKit
import simd

/// Replaces ARKit's camera poses with ones from Apple's Object Capture (`PhotogrammetrySession`), which
/// solves all the views together. ARKit's poses drift over a long walk; on a 797-view room capture
/// the refined poses lifted held-out PSNR from about 21.5 to 25 dB.
///
/// Object Capture works in its own frame and scale, so the LiDAR seed points are carried across with a
/// similarity transform fitted between the two sets of camera centres. It does not report its focal
/// length, and the poses only fit well with a slightly different one than ARKit's, so a few short
/// trial trainings pick that scale.
nonisolated enum PoseRefinement {
    struct Alignment: Codable, Sendable {
        /// `object capture = scale * rotation * arkit + translation`
        var scale: Double
        var rotation: [Double]      // row-major 3x3
        var translation: [Double]
        var registeredFrames: Int
        var totalFrames: Int
        /// Multiplier on ARKit's focal length, chosen by trial trainings. Nil until chosen.
        var focalScale: Double?
    }

    struct StoredPose: Codable, Sendable {
        var file: String
        var matrix: [Double]        // row-major 4x4 camera-to-world, Object Capture frame
    }

    enum RefinementError: LocalizedError {
        case unsupported
        case tooFewFrames(Int)
        case noPoses
        case stalled

        var errorDescription: String? {
            switch self {
            case .unsupported: "Object Capture is not supported on this device."
            case .tooFewFrames(let n): "Object Capture could place only \(n) views."
            case .noPoses: "Object Capture returned no camera poses."
            case .stalled: "Object Capture stopped responding. Try again, or turn off pose refinement."
            }
        }
    }

    static var isSupported: Bool { PhotogrammetrySession.isSupported }

    // MARK: Files

    static func posesURL(_ folder: URL) -> URL { folder.appendingPathComponent("poses-objectcapture.json") }
    static func alignmentURL(_ folder: URL) -> URL { folder.appendingPathComponent("refinement.json") }
    static func datasetURL(_ folder: URL) -> URL { folder.appendingPathComponent("refined-dataset", isDirectory: true) }

    // MARK: Entry point

    /// Makes sure the scan has refined poses and a chosen focal scale, builds a dataset folder that msplat can
    /// load from them, and returns it.
    static func prepare(scanFolder: URL, status: @escaping @Sendable (String, Double?) -> Void) async throws -> URL {
        guard PhotogrammetrySession.isSupported else { throw RefinementError.unsupported }
        let fileManager = FileManager.default
        let clock = ProcessInfo.processInfo
        let startedAt = clock.systemUptime
        var posesSeconds = 0.0

        var alignment: Alignment
        var poses: [StoredPose]
        if fileManager.fileExists(atPath: posesURL(scanFolder).path),
           fileManager.fileExists(atPath: alignmentURL(scanFolder).path) {
            poses = try JSONDecoder().decode([StoredPose].self, from: Data(contentsOf: posesURL(scanFolder)))
            alignment = try JSONDecoder().decode(Alignment.self, from: Data(contentsOf: alignmentURL(scanFolder)))
        } else {
            status("Refining camera poses…", 0)
            let solveStart = clock.systemUptime
            (poses, alignment) = try await solvePoses(scanFolder: scanFolder, status: status)
            posesSeconds = clock.systemUptime - solveStart
            try JSONEncoder().encode(poses).write(to: posesURL(scanFolder), options: .atomic)
            try JSONEncoder().encode(alignment).write(to: alignmentURL(scanFolder), options: .atomic)
        }

        let dataset = datasetURL(scanFolder)
        try writeDataset(scanFolder: scanFolder, poses: poses, alignment: alignment, focalScale: 1, to: dataset)

        if alignment.focalScale == nil {
            alignment.focalScale = try await chooseFocalScale(
                scanFolder: scanFolder, poses: poses, alignment: alignment, dataset: dataset, status: status
            )
            try JSONEncoder().encode(alignment).write(to: alignmentURL(scanFolder), options: .atomic)
        }

        try writeTransforms(scanFolder: scanFolder, poses: poses, alignment: alignment, focalScale: alignment.focalScale ?? 1, to: dataset)
        print("[refine] \(alignment.registeredFrames) of \(alignment.totalFrames) views placed, focal x\(alignment.focalScale ?? 1)")
        print(String(format: "[time] refinement %.0f s (poses %.0f s, the rest focal choice and dataset)", clock.systemUptime - startedAt, posesSeconds))
        return dataset
    }

    // MARK: Object Capture

    static func solvePoses(
        scanFolder: URL,
        status: @escaping @Sendable (String, Double?) -> Void
    ) async throws -> ([StoredPose], Alignment) {
        let arkit = try readARKitFrames(scanFolder)
        let limit = PhotogrammetrySession.limits.maximumNumberOfInputImages
        print("[refine] Object Capture accepts up to \(limit) images; scan has \(arkit.count)")

        // Over the limit: hand it an evenly spaced subset through a folder of symbolic links.
        var input = scanFolder.appendingPathComponent("images", isDirectory: true)
        let files = arkit.keys.sorted()
        if files.count > limit {
            let subsetFolder = scanFolder.appendingPathComponent("oc-input", isDirectory: true)
            try? FileManager.default.removeItem(at: subsetFolder)
            try FileManager.default.createDirectory(at: subsetFolder, withIntermediateDirectories: true)
            for i in 0..<limit {
                let name = files[i * files.count / limit]
                try FileManager.default.createSymbolicLink(
                    at: subsetFolder.appendingPathComponent(name),
                    withDestinationURL: input.appendingPathComponent(name)
                )
            }
            input = subsetFolder
        }
        defer { try? FileManager.default.removeItem(at: scanFolder.appendingPathComponent("oc-input")) }

        var configuration = PhotogrammetrySession.Configuration()
        configuration.sampleOrdering = .sequential
        configuration.featureSensitivity = .high
        let session = try PhotogrammetrySession(input: input, configuration: configuration)
        // Poses only (a point cloud was requested before and never used). On iOS the session also builds a model:
        // with both requested, the poses arrived after the mesh, textures and optimisation, about 100 of 145 s.
        try session.process(requests: [.poses])

        var result: PhotogrammetrySession.Poses?
        var lastReported = -1
        let started = ProcessInfo.processInfo.systemUptime
        // Stop as soon as the poses arrive. The session can keep working afterwards on things we
        // did not ask for (it builds a model on iOS), which would otherwise leave the app waiting.
        let lastEvent = OSAllocatedUnfairLock(initialState: ProcessInfo.processInfo.systemUptime)
        nonisolated(unsafe) let watched = session
        let watchdog = Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(15))
                let idle = ProcessInfo.processInfo.systemUptime - lastEvent.withLock { $0 }
                if idle > 240 {
                    print("[refine] no output from Object Capture for \(Int(idle)) s; cancelling it")
                    watched.cancel()
                    return
                }
            }
        }
        defer { watchdog.cancel() }
        var lastStage = ""
        outputs: for try await output in session.outputs {
            lastEvent.withLock { $0 = ProcessInfo.processInfo.systemUptime }
            switch output {
            case .requestProgress(_, let fraction):
                let percent = Int(fraction * 100)
                if percent != lastReported {
                    lastReported = percent
                    status("Refining camera poses… \(percent)%", fraction)
                }
            case .requestComplete(_, .poses(let poses)):
                result = poses
                print(String(format: "[refine] Object Capture returned %d poses after %.0f s",
                             poses.posesBySample.count, ProcessInfo.processInfo.systemUptime - started))
                session.cancel()
                break outputs
            case .requestError(_, let error):
                print("[refine] request error: \(error)")
                throw error
            case .requestProgressInfo(_, let info):
                // One line per stage, not one per change of the time estimate.
                guard let stage = info.processingStage.map({ "\($0)" }), stage != lastStage else { break }
                lastStage = stage
                print(String(format: "[refine] Object Capture: %@ at %.0f s%@", stage, ProcessInfo.processInfo.systemUptime - started,
                             info.estimatedRemainingTime.map { String(format: ", about %.0f s left", $0) } ?? ""))
            case .requestComplete(let request, _):
                print("[refine] request complete: \(String(describing: request).prefix(60))")
            case .processingComplete, .processingCancelled, .inputComplete:
                print("[refine] session event: \(output)")
            case .skippedSample(let id):
                print("[refine] skipped sample \(id)")
            case .invalidSample(let id, let reason):
                print("[refine] invalid sample \(id): \(reason)")
            default:
                break
            }
        }
        guard let result else {
            let idle = ProcessInfo.processInfo.systemUptime - lastEvent.withLock { $0 }
            throw idle > 240 ? RefinementError.stalled : RefinementError.noPoses
        }
        guard !result.posesBySample.isEmpty else { throw RefinementError.noPoses }

        var stored: [StoredPose] = []
        let urls = result.urlsBySample
        for (sample, pose) in result.posesBySample {
            guard let name = urls[sample]?.lastPathComponent, arkit[name] != nil else { continue }
            let r = simd_float3x3(pose.rotation)
            let t = pose.translation
            stored.append(StoredPose(file: name, matrix: [
                Double(r.columns.0.x), Double(r.columns.1.x), Double(r.columns.2.x), Double(t.x),
                Double(r.columns.0.y), Double(r.columns.1.y), Double(r.columns.2.y), Double(t.y),
                Double(r.columns.0.z), Double(r.columns.1.z), Double(r.columns.2.z), Double(t.z),
                0, 0, 0, 1,
            ]))
        }
        stored.sort { $0.file < $1.file }
        guard stored.count >= 8 else { throw RefinementError.tooFewFrames(stored.count) }

        // Similarity from ARKit camera centres to Object Capture camera centres.
        let from = stored.map { SIMD3<Double>(arkit[$0.file]![3], arkit[$0.file]![7], arkit[$0.file]![11]) }
        let to = stored.map { SIMD3<Double>($0.matrix[3], $0.matrix[7], $0.matrix[11]) }
        let fit = fitSimilarity(from: from, to: to)
        let alignment = Alignment(
            scale: fit.scale,
            rotation: [fit.rotation.columns.0.x, fit.rotation.columns.1.x, fit.rotation.columns.2.x,
                       fit.rotation.columns.0.y, fit.rotation.columns.1.y, fit.rotation.columns.2.y,
                       fit.rotation.columns.0.z, fit.rotation.columns.1.z, fit.rotation.columns.2.z],
            translation: [fit.translation.x, fit.translation.y, fit.translation.z],
            registeredFrames: stored.count,
            totalFrames: arkit.count,
            focalScale: nil
        )
        return (stored, alignment)
    }

    // MARK: Alignment (Horn's closed-form absolute orientation with scale)

    static func fitSimilarity(from a: [SIMD3<Double>], to b: [SIMD3<Double>])
        -> (scale: Double, rotation: simd_double3x3, translation: SIMD3<Double>) {
        let n = Double(a.count)
        let ma = a.reduce(.zero, +) / n, mb = b.reduce(.zero, +) / n
        var s = [[Double]](repeating: [0, 0, 0], count: 3)
        var varianceA = 0.0
        for (pa, pb) in zip(a, b) {
            let x = pa - ma, y = pb - mb
            varianceA += simd_length_squared(x)
            for j in 0..<3 { for k in 0..<3 { s[j][k] += x[j] * y[k] } }
        }
        let n4: [[Double]] = [
            [s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
            [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
            [s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]],
            [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]],
        ]
        let q = largestEigenvector(symmetric: n4)           // (w, x, y, z)
        let rotation = simd_double3x3(simd_quatd(ix: q[1], iy: q[2], iz: q[3], r: q[0]).normalized)
        var numerator = 0.0
        for (pa, pb) in zip(a, b) { numerator += simd_dot(pb - mb, rotation * (pa - ma)) }
        let scale = numerator / varianceA
        return (scale, rotation, mb - scale * (rotation * ma))
    }

    /// Jacobi eigenvalue iteration for a small symmetric matrix.
    private static func largestEigenvector(symmetric matrix: [[Double]]) -> [Double] {
        let n = matrix.count
        var a = matrix
        var v = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for _ in 0..<100 {
            var p = 0, q = 1, largest = 0.0
            for i in 0..<n { for j in (i + 1)..<n where abs(a[i][j]) > largest { largest = abs(a[i][j]); p = i; q = j } }
            if largest < 1e-14 { break }
            let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot(), s = t * c
            for k in 0..<n {
                let akp = a[k][p], akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }
            for k in 0..<n {
                let apk = a[p][k], aqk = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }
            for k in 0..<n {
                let vkp = v[k][p], vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }
        var best = 0
        for i in 1..<n where a[i][i] > a[best][best] { best = i }
        return (0..<n).map { v[$0][best] }
    }

    // MARK: Dataset for msplat

    static func readARKitFrames(_ scanFolder: URL) throws -> [String: [Double]] {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: scanFolder.appendingPathComponent("transforms.json")))
        guard let root = json as? [String: Any], let frames = root["frames"] as? [[String: Any]] else { return [:] }
        var result: [String: [Double]] = [:]
        for frame in frames {
            guard let path = frame["file_path"] as? String,
                  let rows = frame["transform_matrix"] as? [[Double]] else { continue }
            result[(path as NSString).lastPathComponent] = rows.flatMap { $0 }
        }
        return result
    }

    private static func writeDataset(scanFolder: URL, poses: [StoredPose], alignment: Alignment,
                                     focalScale: Double, to folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeTransforms(scanFolder: scanFolder, poses: poses, alignment: alignment, focalScale: focalScale, to: folder)
        // The refined poses are carried back into ARKit's frame, so the LiDAR seeds are used as captured.
        let seeds = folder.appendingPathComponent("points3D.ply")
        try? FileManager.default.removeItem(at: seeds)
        let original = scanFolder.appendingPathComponent("points3D.ply")
        if FileManager.default.fileExists(atPath: original.path) {
            try FileManager.default.copyItem(at: original, to: seeds)
        }
    }

    /// An Object Capture camera-to-world matrix expressed in ARKit's frame. With
    /// `objectCapture = scale * rotation * arkit + translation`, the inverse is applied to the camera's
    /// position and its orientation. Keeping ARKit's frame keeps the splat upright and in metres.
    static func toARKitFrame(_ m: [Double], alignment: Alignment) -> [Double] {
        let r = alignment.rotation, t = alignment.translation, s = alignment.scale
        // R^T: the transpose of the row-major rotation.
        func rt(_ i: Int, _ j: Int) -> Double { r[j * 3 + i] }
        var out = [Double](repeating: 0, count: 16)
        for i in 0..<3 {
            for j in 0..<3 {
                var v = 0.0
                for k in 0..<3 { v += rt(i, k) * m[k * 4 + j] }
                out[i * 4 + j] = v
            }
            var p = 0.0
            for k in 0..<3 { p += rt(i, k) * (m[k * 4 + 3] - t[k]) }
            out[i * 4 + 3] = p / s
        }
        out[15] = 1
        return out
    }

    /// Camera list in Nerfstudio format, with the refined poses and a scaled focal length. Image paths
    /// are absolute: msplat reads paths starting with "/" as they are.
    static func writeTransforms(scanFolder: URL, poses: [StoredPose], alignment: Alignment, focalScale: Double, to folder: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: scanFolder.appendingPathComponent("transforms.json")))
        guard var root = json as? [String: Any] else { return }
        if let fx = root["fl_x"] as? Double { root["fl_x"] = fx * focalScale }
        if let fy = root["fl_y"] as? Double { root["fl_y"] = fy * focalScale }
        root["ply_file_path"] = "points3D.ply"
        let images = scanFolder.appendingPathComponent("images")
        // Keep whatever else the capture stored for a frame (its own focal length, time, sharpness).
        var stored: [String: [String: Any]] = [:]
        for frame in (root["frames"] as? [[String: Any]]) ?? [] {
            if let path = frame["file_path"] as? String { stored[(path as NSString).lastPathComponent] = frame }
        }
        root["frames"] = poses.map { pose in
            let matrix = toARKitFrame(pose.matrix, alignment: alignment)
            var frame = stored[pose.file] ?? [:]
            frame["file_path"] = images.appendingPathComponent(pose.file).path
            frame["transform_matrix"] = stride(from: 0, to: 16, by: 4).map { Array(matrix[$0..<($0 + 4)]) }
            for key in ["fl_x", "fl_y"] {
                if let value = frame[key] as? Double { frame[key] = value * focalScale }
            }
            return frame
        }
        try JSONSerialization.data(withJSONObject: root).write(to: folder.appendingPathComponent("transforms.json"), options: .atomic)
    }

    // MARK: Focal length

    /// Short trial trainings at a few focal scales; the one with the best held-out PSNR wins.
    private static func chooseFocalScale(
        scanFolder: URL, poses: [StoredPose], alignment: Alignment, dataset: URL,
        status: @escaping @Sendable (String, Double?) -> Void
    ) async throws -> Double {
        let candidates = [0.98, 1.0, 1.02, 1.04, 1.06]
        var best = (scale: 1.0, psnr: -Double.infinity)
        for (index, scale) in candidates.enumerated() {
            try Task.checkCancellation()
            status("Choosing focal length… \(index + 1)/\(candidates.count)", Double(index) / Double(candidates.count))
            try writeTransforms(scanFolder: scanFolder, poses: poses, alignment: alignment, focalScale: scale, to: dataset)

            let psnr: Double = autoreleasepool {
                let data = GaussianDataset(path: dataset.path, downscaleFactor: 4, evalMode: true, testEvery: 8, maxViews: 100)
                guard data.numTrain >= 8, data.numTest > 0 else { return -Double.infinity }
                var config = TrainingConfig()
                config.iterations = 600
                config.shDegree = 0
                config.numDownscales = 1
                config.resolutionSchedule = 300
                config.warmupLength = 100
                config.refineEvery = 100
                config.resetAlphaEvery = 30
                config.maxGaussians = 200_000
                config.bgColor = (0, 0, 0)
                let trainer = GaussianTrainer(dataset: data, config: config)
                for _ in 0..<600 { trainer.step() }
                return Double(trainer.evaluate().psnr)
            }
            print(String(format: "[refine] focal x%.2f -> %.2f dB", scale, psnr))
            if psnr > best.psnr { best = (scale, psnr) }
        }
        return best.scale
    }
}
