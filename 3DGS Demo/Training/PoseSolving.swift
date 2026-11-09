import Foundation
import Msplat
import simd

/// Camera poses solved from the photos (see `PoseSolver`), as a dataset folder msplat can train on.
///
/// ARKit's poses are a centimetre and a degree off. On a bowl scan the solved poses halved the misregistration
/// between a render and its photo (1.9 px to 1.0 px) and lifted held-out PSNR from 24.1 to 25.5 dB, and the
/// splat's edges, wood grain and small print come out visibly sharper. The solved poses stay in ARKit's frame,
/// so the LiDAR seed cloud still lines up and the splat comes out upright and in metres.
nonisolated enum PoseSolving {
    /// The solver's settings are part of the name, so poses solved with weaker settings (600 corners x 5 neighbours
    /// held out 25.44 dB on the bowl scan, against 25.91 dB for 1200 x 14) are solved again. 2400 corners: the poses are
    /// as good as with 1200, but the corners also seed the splat, and with 1200 the thin top arm of a faucet 2.7 m away
    /// got a seed only by chance (1 of 8 solves gave one; 2 of 2 with 2400). The solver takes 25% longer.
    static let solverSettings = (keypoints: 2400, neighbours: 14)
    /// "f": each view's focal length is fitted too (held out +0.28 dB on the HDR bowl, +0.11 on the older one).
    /// "oc": started from poses lined up by Object Capture (see `linedUpStart`).
    private static func settingsName(_ linesUpPasses: Bool) -> String {
        "\(solverSettings.keypoints)x\(solverSettings.neighbours)f\(linesUpPasses ? "-oc" : "")"
    }
    static func transformsURL(_ scanFolder: URL, linesUpPasses: Bool) -> URL {
        scanFolder.appendingPathComponent("transforms-solved-\(settingsName(linesUpPasses)).json")
    }
    /// The seed cloud that goes with the solved poses: the capture's LiDAR points plus the corners the solver triangulated
    /// where the LiDAR has none. The LiDAR reaches a few metres, so the faucet at the far end of the kitchen had no seed;
    /// whether it grew into the splat then came down to luck (it survived in 5 of 9 identical training runs of the HDR
    /// bowl scan, and in all 6 runs started from these points, where the person on the bed also came out steadier).
    static func pointsURL(_ scanFolder: URL, linesUpPasses: Bool) -> URL {
        scanFolder.appendingPathComponent("points-solved-\(settingsName(linesUpPasses)).ply")
    }
    static func datasetURL(_ scanFolder: URL) -> URL { scanFolder.appendingPathComponent("solved-dataset", isDirectory: true) }
    static func startURL(_ scanFolder: URL) -> URL { scanFolder.appendingPathComponent("lined-up-start", isDirectory: true) }

    /// Whether poses are kept from an earlier training of this scan. With lining up on, either kind counts: when ARKit's
    /// passes already lined up, the solver started from ARKit's poses (see `minimumDriftCentimetres`).
    static func hasSolved(_ scanFolder: URL, linesUpPasses: Bool) -> Bool {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: transformsURL(scanFolder, linesUpPasses: false).path) { return true }
        return linesUpPasses && fileManager.fileExists(atPath: transformsURL(scanFolder, linesUpPasses: true).path)
    }

    /// Forgets the solved poses (both kinds), their seed points, Object Capture's poses and the datasets made from them,
    /// so the next training solves again from the photos.
    static func discardSolved(_ scanFolder: URL) {
        var urls = [datasetURL(scanFolder), startURL(scanFolder),
                    PoseRefinement.posesURL(scanFolder), PoseRefinement.alignmentURL(scanFolder)]
        for linesUp in [false, true] {
            urls += [transformsURL(scanFolder, linesUpPasses: linesUp), pointsURL(scanFolder, linesUpPasses: linesUp)]
        }
        for url in urls { try? FileManager.default.removeItem(at: url) }
    }

    /// Solves the poses of the scan (once; the result is kept for later runs, so a resumed or extended training
    /// sees the same cameras), and returns a dataset folder with them.
    /// `skipFocusDriftAbove` (a share, 0.02 = 2%): photos whose reported focal length is that far above the capture's
    /// median, so taken with the lens focused much closer than most, are left out of the dataset.
    /// `linesUpPasses`: start the solver from poses Object Capture lined up (see `linedUpStart`) instead of ARKit's.
    static func prepare(scanFolder: URL, frameWidth: Double = 1920, skipFocusDriftAbove: Double? = nil, linesUpPasses: Bool = true,
                        status: @escaping @Sendable (String, Double?) -> Void) async throws -> URL {
        let fileManager = FileManager.default
        let started = ProcessInfo.processInfo.systemUptime

        // Which start the solver uses: Object Capture's lined-up passes only when ARKit really drifted between them.
        var startsLinedUp = false
        var start = scanFolder
        if linesUpPasses {
            if fileManager.fileExists(atPath: transformsURL(scanFolder, linesUpPasses: true).path),
               fileManager.fileExists(atPath: pointsURL(scanFolder, linesUpPasses: true).path) {
                startsLinedUp = true
            } else {
                do {
                    let lined = try await linedUpStart(scanFolder: scanFolder, status: status)
                    if lined.driftCentimetres >= minimumDriftCentimetres {
                        startsLinedUp = true
                        start = lined.folder
                    } else {
                        print(String(format: "[pose] ARKit's passes already line up (%.1f cm from Object Capture's poses, under %.1f): solving from ARKit's poses",
                                     lined.driftCentimetres, minimumDriftCentimetres))
                    }
                } catch {
                    if Task.isCancelled { throw CancellationError() }
                    print("[pose] Object Capture could not line up the passes (\(error.localizedDescription)); solving from ARKit's poses")
                }
            }
        }
        let solved = transformsURL(scanFolder, linesUpPasses: startsLinedUp)
        let solvedPoints = pointsURL(scanFolder, linesUpPasses: startsLinedUp)

        if !fileManager.fileExists(atPath: solved.path) || !fileManager.fileExists(atPath: solvedPoints.path) {
            status("Solving camera poses…", 0)
            let temporary = scanFolder.appendingPathComponent("transforms-solved.json.writing")
            let temporaryPoints = scanFolder.appendingPathComponent("points-solved.ply.writing")
            try? fileManager.removeItem(at: temporary)
            try? fileManager.removeItem(at: temporaryPoints)
            let result = try PoseSolver.solve(
                datasetDirectory: start.path,
                outputTransforms: temporary.path,
                pointsOutput: temporaryPoints.path,
                // Corners are matched in images about 960 wide, whatever the photo size.
                scale: Float(min(0.5, 960 / max(frameWidth, 1))),
                keypoints: solverSettings.keypoints,
                neighbours: solverSettings.neighbours,
                // Once Object Capture has lined the passes up, 6 of the 14 views each corner is looked for in come from
                // other passes (bedroom +0.1 dB); from ARKit's poses that cost a bowl's fine detail, so none.
                otherPasses: startsLinedUp ? 6 : 0
            ) { stage, fraction in
                status(stage, fraction)
                return !Task.isCancelled
            }
            try? fileManager.removeItem(at: solved)
            try fileManager.moveItem(at: temporary, to: solved)
            try? fileManager.removeItem(at: solvedPoints)
            try fileManager.moveItem(at: temporaryPoints, to: solvedPoints)
            print(String(format: "[pose] solved %d views from %d matched corners (%d observations): median reprojection error %.2f -> %.2f px, cameras moved %.1f cm and %.2f deg on average",
                         result.views, result.points, result.observations, result.errorBefore, result.errorAfter,
                         result.meanShiftCentimetres, result.meanTurnDegrees))
            print("[pose] \(result.newPoints) corners triangulated beyond the LiDAR points were added to the seed cloud")
            print(String(format: "[time] pose solving %.0f s", ProcessInfo.processInfo.systemUptime - started))
        } else {
            print("[pose] using the camera poses solved earlier")
        }

        let dataset = datasetURL(scanFolder)
        try fileManager.createDirectory(at: dataset, withIntermediateDirectories: true)
        let transforms = dataset.appendingPathComponent("transforms.json")
        try? fileManager.removeItem(at: transforms)
        // The photos' paths are made from where the scan is now: the app's data folder moves when the app is
        // reinstalled, so paths stored with the cached poses can point nowhere.
        try writeTransforms(from: solved, scanFolder: scanFolder, to: transforms, skipFocusDriftAbove: skipFocusDriftAbove)
        let seeds = dataset.appendingPathComponent("points3D.ply")
        try? fileManager.removeItem(at: seeds)
        try fileManager.copyItem(at: solvedPoints, to: seeds)
        return dataset
    }

    /// Starting poses for the solver with the passes of the capture lined up, as a dataset folder (absolute photo paths).
    ///
    /// The solver matches each photo with its nearest ones, which are almost always the frames just before and after it,
    /// and it only finds a match within a few pixels of where the starting poses put it. When a room is walked around
    /// several times, ARKit drifts between the passes (by 4 cm on average on a 797-photo bedroom, with the same pot or
    /// shelf a few centimetres apart each time round), the solver cannot link them, and the splat shows objects two or
    /// three times. Object Capture matches every photo with every other, so it closes those loops; but an iPhone gives it
    /// at most 200 photos, and it is less precise locally. So it solves up to 200 evenly spaced photos, the others follow
    /// ARKit's motion between the nearest of those, and the solver then refines every photo from there. On that bedroom,
    /// held out: ARKit 22.0 dB, solver alone 22.3, this 25.0; on views seen only from another pass 14.6, 14.8 and 17.7.
    /// Also returns how far ARKit's camera positions are from Object Capture's (mean, after the best overall fit): the drift.
    static func linedUpStart(scanFolder: URL, status: @escaping @Sendable (String, Double?) -> Void) async throws
        -> (folder: URL, driftCentimetres: Double) {
        guard PoseRefinement.isSupported else { throw PoseRefinement.RefinementError.unsupported }
        let fileManager = FileManager.default
        let started = ProcessInfo.processInfo.systemUptime
        let posesFile = PoseRefinement.posesURL(scanFolder), alignmentFile = PoseRefinement.alignmentURL(scanFolder)
        var poses: [PoseRefinement.StoredPose]
        var alignment: PoseRefinement.Alignment
        if fileManager.fileExists(atPath: posesFile.path), fileManager.fileExists(atPath: alignmentFile.path) {
            poses = try JSONDecoder().decode([PoseRefinement.StoredPose].self, from: Data(contentsOf: posesFile))
            alignment = try JSONDecoder().decode(PoseRefinement.Alignment.self, from: Data(contentsOf: alignmentFile))
            print("[pose] using the passes Object Capture lined up earlier")
        } else {
            status("Lining up the passes (Object Capture)…", 0)
            (poses, alignment) = try await PoseRefinement.solvePoses(scanFolder: scanFolder) { _, fraction in
                status("Lining up the passes (Object Capture)…", fraction)
            }
            try JSONEncoder().encode(poses).write(to: posesFile, options: .atomic)
            try JSONEncoder().encode(alignment).write(to: alignmentFile, options: .atomic)
        }

        let arkit = try PoseRefinement.readARKitFrames(scanFolder)
        func matrix(_ m: [Double]) -> simd_double4x4 {
            simd_double4x4(rows: [SIMD4(m[0], m[1], m[2], m[3]), SIMD4(m[4], m[5], m[6], m[7]),
                                  SIMD4(m[8], m[9], m[10], m[11]), SIMD4(m[12], m[13], m[14], m[15])])
        }
        // Each placed photo's correction from ARKit's pose to the lined-up one, in ARKit's frame: lined up = correction * ARKit.
        var correction: [String: simd_double4x4] = [:]
        var drift = 0.0
        for pose in poses {
            guard let captured = arkit[pose.file] else { continue }
            let lined = PoseRefinement.toARKitFrame(pose.matrix, alignment: alignment)
            correction[pose.file] = matrix(lined) * matrix(captured).inverse
            drift += simd_distance(SIMD3(lined[3], lined[7], lined[11]), SIMD3(captured[3], captured[7], captured[11]))
        }
        drift = 100 * drift / Double(max(correction.count, 1))
        let names = arkit.keys.sorted()                  // capture order
        let placedIndices = names.indices.filter { correction[names[$0]] != nil }
        func blend(_ a: simd_double4x4, _ b: simd_double4x4, _ w: Double) -> simd_double4x4 {
            func rotation(_ m: simd_double4x4) -> simd_double3x3 {
                simd_double3x3(columns: (SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                         SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                         SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
            }
            let r = simd_double3x3(simd_slerp(simd_quatd(rotation(a)), simd_quatd(rotation(b)), w))
            let t = (1 - w) * SIMD3(a.columns.3.x, a.columns.3.y, a.columns.3.z) + w * SIMD3(b.columns.3.x, b.columns.3.y, b.columns.3.z)
            return simd_double4x4(columns: (SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0), SIMD4(t, 1)))
        }
        var lined: [String: simd_double4x4] = [:]
        for (index, name) in names.enumerated() {
            let captured = matrix(arkit[name]!)
            if let c = correction[name] { lined[name] = c * captured; continue }
            // Between two placed photos: their corrections blended by how close each one is in the capture.
            let previous = placedIndices.last { $0 < index }, next = placedIndices.first { $0 > index }
            switch (previous, next) {
            case let (p?, n?):
                let w = Double(index - p) / Double(n - p)
                lined[name] = blend(correction[names[p]]!, correction[names[n]]!, w) * captured
            case let (p?, nil): lined[name] = correction[names[p]]! * captured
            case let (nil, n?): lined[name] = correction[names[n]]! * captured
            default: lined[name] = captured
            }
        }

        // The capture's own camera list with these poses, the photos and the LiDAR seeds where they are.
        guard var root = try JSONSerialization.jsonObject(with: Data(contentsOf: scanFolder.appendingPathComponent("transforms.json"))) as? [String: Any],
              var frames = root["frames"] as? [[String: Any]] else { throw CocoaError(.fileReadCorruptFile) }
        let images = scanFolder.appendingPathComponent("images", isDirectory: true)
        for index in frames.indices {
            guard let path = frames[index]["file_path"] as? String else { continue }
            let name = (path as NSString).lastPathComponent
            frames[index]["file_path"] = images.appendingPathComponent(name).path
            if let m = lined[name] {
                frames[index]["transform_matrix"] = (0..<4).map { row in (0..<4).map { column in m[column][row] } }
            }
        }
        root["frames"] = frames
        root["ply_file_path"] = scanFolder.appendingPathComponent("points3D.ply").path
        let folder = startURL(scanFolder)
        try fileManager.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: root).write(to: folder.appendingPathComponent("transforms.json"), options: .atomic)
        print(String(format: "[pose] Object Capture placed %d of %d photos in %.0f s; ARKit's camera positions are %.1f cm from its poses on average",
                     placedIndices.count, names.count, ProcessInfo.processInfo.systemUptime - started, drift))
        return (folder, drift)
    }

    /// Below this drift between ARKit's passes the solver starts from ARKit's poses even when lining up is on. Object Capture
    /// is less precise locally: on two bowls captured in one orbit (1.2 cm of drift each) its start cost 0.2 dB held out and
    /// part of a thin faucet's top arm; on the bedroom (4.1 cm) it gained 2.7 dB.
    static let minimumDriftCentimetres = 2.5

    /// The focal length ARKit reported for each photo, by file name; nil for captures that do not store one per photo.
    private static func capturedFocalLengths(_ scanFolder: URL) -> [String: Double]? {
        guard let data = try? Data(contentsOf: scanFolder.appendingPathComponent("transforms.json")),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let frames = root["frames"] as? [[String: Any]] else { return nil }
        var result: [String: Double] = [:]
        for frame in frames {
            guard let path = frame["file_path"] as? String, let focal = frame["fl_x"] as? Double else { return nil }
            result[(path as NSString).lastPathComponent] = focal
        }
        return result.isEmpty ? nil : result
    }

    /// The solved camera list with every photo path pointing into this scan's images folder.
    private static func writeTransforms(from solved: URL, scanFolder: URL, to destination: URL, skipFocusDriftAbove: Double?) throws {
        guard var root = try JSONSerialization.jsonObject(with: Data(contentsOf: solved)) as? [String: Any],
              var frames = root["frames"] as? [[String: Any]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let images = scanFolder.appendingPathComponent("images", isDirectory: true)
        for index in frames.indices {
            guard let path = frames[index]["file_path"] as? String else { continue }
            frames[index]["file_path"] = images.appendingPathComponent((path as NSString).lastPathComponent).path
        }
        if let limit = skipFocusDriftAbove, let original = capturedFocalLengths(scanFolder) {
            let name = { (frame: [String: Any]) in ((frame["file_path"] as? String).map { ($0 as NSString).lastPathComponent }) ?? "" }
            let values = frames.compactMap { original[name($0)] }.sorted()
            if values.count == frames.count, values.count >= 60 {
                let ceiling = values[values.count / 2] * (1 + limit)
                let kept = frames.filter { (original[name($0)] ?? 0) <= ceiling }
                // Keep the rule from emptying a capture that was shot with the lens focused all over the place.
                if kept.count >= 60, kept.count * 10 >= frames.count * 6 {
                    if kept.count < frames.count {
                        print("[views] \(frames.count - kept.count) of \(frames.count) photos were taken with the lens focused much closer than in the rest of the capture (focal length above \(Int(ceiling)) px) and are left out")
                    }
                    frames = kept
                } else {
                    print("[views] the focus changes a lot over the capture, so no photo is left out")
                }
            }
        }
        root["frames"] = frames
        root["ply_file_path"] = "points3D.ply"
        try JSONSerialization.data(withJSONObject: root).write(to: destination, options: .atomic)
        // msplat stops the app on a photo it cannot open, so check the first one here.
        if let first = frames.first?["file_path"] as? String, !FileManager.default.fileExists(atPath: first) {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: first])
        }
    }
}
