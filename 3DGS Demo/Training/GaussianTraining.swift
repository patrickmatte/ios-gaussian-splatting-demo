// Copyright 2026 Voxelio.
// Licensed under the PolyForm Noncommercial License 1.0.0.

import Combine
import ImageIO
import Msplat
import SwiftUI
import UIKit

private final class LocalGaussianProgressRelay: @unchecked Sendable {
    nonisolated(unsafe) weak var processor: LocalGaussianProcessor?

    nonisolated init(_ processor: LocalGaussianProcessor) {
        self.processor = processor
    }

    nonisolated func thermal(_ note: String?) {
        Task { @MainActor [weak processor] in
            processor?.thermalNote = note
        }
    }

    nonisolated func status(_ message: String, _ fraction: Double?) {
        Task { @MainActor [weak processor] in
            processor?.statusMessage = message
            processor?.statusFraction = fraction
        }
    }

    nonisolated func publish(iteration: Int, splatCount: Int) {
        Task { @MainActor [weak processor] in
            processor?.acceptProgress(iteration: iteration, splatCount: splatCount)
        }
    }
}

enum LocalGaussianQuality: Int, CaseIterable, Identifiable, Codable, Comparable, Sendable {
    case preview = 1_000
    case balanced = 2_000
    case refined = 3_000
    case high = 4_000
    case maximum = 5_000
    case extended = 7_000
    case thorough = 10_000

    nonisolated var id: Int { rawValue }
    nonisolated var iterations: Int { rawValue }

    nonisolated static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    // 7K: Scantic trains 7,000 to 7,750 steps; thin, far things (a faucet, a lamp) came out better than with 4K or 5K.
    nonisolated static let defaultSelection: Self = .extended

    nonisolated static var maximumIterations: Int {
        Self.thorough.iterations
    }

    nonisolated static func extensionTargets(after completedIterations: Int) -> [Self] {
        allCases.filter { $0.iterations > completedIterations }
    }

    nonisolated var label: String {
        switch self {
        case .preview: return "1K"
        case .balanced: return "2K"
        case .refined: return "3K"
        case .high: return "4K"
        case .maximum: return "5K"
        case .extended: return "7K"
        case .thorough: return "10K"
        }
    }

    nonisolated var detail: String {
        switch self {
        case .preview: return "Quick preview"
        case .balanced: return "Balanced"
        case .refined: return "More detail"
        case .high: return "High detail"
        case .maximum: return "Very high detail"
        case .extended: return "Maximum detail"
        case .thorough: return "Best local result"
        }
    }

    nonisolated var configuration: TrainingConfig {
        var value = TrainingConfig()
        value.iterations = Int32(iterations)
        value.shDegree = 1
        value.shDegreeInterval = 1_000
        value.ssimWeight = self >= .high ? 0.35 : 0.2
        value.downscaleFactor = 2
        value.numDownscales = self == .preview ? 1 : 2
        // Steps at 1/4 and then 1/2 of the final image size are much cheaper than full-size ones (on the phone the cost
        // is mostly pixels: about 120 ms per megapixel and step). A longer ramp costs nothing up to about 6K steps
        // (960x720 images, 4K steps: 24.5 dB with 600-step stages and with 1500-step stages, but 5.1 minutes against
        // 3.0 estimated on the phone); at 10K steps it costs 0.2 dB for 40% less time.
        value.resolutionSchedule = Int32(max(600, iterations * 7 / 20))
        value.refineEvery = 100
        value.warmupLength = 300
        // No periodic opacity reset: after one, every splat restarts from opacity 0.2 and the thin, far, weakly seen ones
        // (the kitchen faucet of the HDR bowl) can fade out for good. Only runs over 6K steps ever had one, at step 3100.
        value.resetAlphaEvery = 1_000
        value.densifyGradThresh = self >= .high ? 0.0002 : 0.0003
        value.densifySizeThresh = 0.01
        // Each densification round reserves buffers for 3x the current count; this keeps that
        // inside the iPhone's memory limit (a 5K run reached 488k splats and was killed at 3.4 GB).
        value.maxGaussians = 500_000
        value.stopScreenSizeAt = 3_000
        value.splitScreenSize = 0.05
        value.bgColor = (0, 0, 0)
        // A random colour behind the splats each step: about +0.3 dB on held-out photos, and fewer splats.
        value.randomBackground = true
        // The saved splat is in the capture's own frame: metres, upright (ARKit's), where the photos were taken. Without
        // it the file was in the trainer's frame, centred on the cameras and scaled to their spread.
        value.keepCrs = true
        // No axis of a splat may pass 1.5 m. The trainer only removes huge splats during densification after step 3000, so
        // a short run never does, and a far splat that gets almost no gradient keeps growing afterwards: Mac runs of
        // the HDR bowl ended with splats of 2.4 to 4.0 m (Scantic's largest is 1.8 m). Costs 0.04 dB.
        value.maxSplatSize = 1.5
        return value
    }
}

@MainActor
final class LocalGaussianProcessor: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var completedIterations = 0
    @Published private(set) var splatCount = 0
    /// What the run is doing before the first training step, such as refining poses.
    @Published fileprivate(set) var statusMessage: String?
    @Published fileprivate(set) var statusFraction: Double?
    /// Set while the phone is hot and training is slowed down or paused.
    @Published fileprivate(set) var thermalNote: String?
    @Published var quality: LocalGaussianQuality = .defaultSelection
    @Published var options = LocalGaussianTrainingOptions.default

    private var job: Task<Void, Never>?

    var isRunning: Bool {
        phase == .running
    }

    func synchronize(with asset: LocalGaussianAsset) {
        guard !isRunning else { return }
        if asset.stage == .failed, let failure = asset.failure {
            phase = .failed(failure)
        } else {
            phase = asset.hasSplat ? .ready : .idle
        }
        completedIterations = asset.completedSteps ?? 0
        options = asset.trainingOptions ?? .default

        if quality.iterations <= completedIterations {
            quality = LocalGaussianQuality.extensionTargets(after: completedIterations).first ?? .maximum
        }
    }

    fileprivate func acceptProgress(iteration: Int, splatCount: Int) {
        completedIterations = iteration
        self.splatCount = splatCount
        statusMessage = nil
        statusFraction = nil
    }

    /// Trains `asset`. With `restartingAt` set, ignores any earlier result and checkpoint and trains
    /// from scratch to that target from the saved camera dataset.
    func process(_ asset: LocalGaussianAsset, library: LocalGaussianLibrary,
                 restartingAt restartQuality: LocalGaussianQuality? = nil) {
        guard !isRunning else { return }
        guard asset.hasSourceDataset else {
            phase = .failed("The original camera dataset is unavailable.")
            return
        }

        if let restartQuality { quality = restartQuality }
        let selectedQuality = restartQuality ?? quality
        let startingIterations = restartQuality == nil ? (asset.completedSteps ?? 0) : 0
        guard selectedQuality.iterations > startingIterations else {
            phase = .failed("Choose a target above the completed iteration count.")
            return
        }
        guard startingIterations == 0 || asset.hasTrainingCheckpoint else {
            phase = .failed("The saved training checkpoint is unavailable, so this result cannot be extended.")
            return
        }

        // Extending keeps the options the checkpoint was trained with (colour detail must match).
        // Object Capture's poses make refining while training pointless (bowl 26.3 vs 26.0 dB, bedroom 25.3 vs 25.0),
        // and it costs a pose gradient read-back every step. Options saved before the two excluded each other
        // can still have both on.
        let usedOptions: LocalGaussianTrainingOptions = {
            var chosen = startingIterations > 0 ? (asset.trainingOptions ?? .default) : options
            chosen.frameWidth = asset.frameSize.width
            chosen.frameHeight = asset.frameSize.height
            if chosen.refinesPoses { chosen.refinesPosesWhileTraining = false }
            // Solved poses are already consistent with the photos; nudging them again while training made a held-out
            // score worse (24.0 against 25.5 dB on the bowl scan) and Object Capture would replace them.
            if chosen.solvesPoses {
                chosen.refinesPoses = false
                chosen.refinesPosesWhileTraining = false
            }
            return chosen
        }()
        let previousOutputURL = asset.hasSplat ? asset.splatURL : nil
        let resumeCheckpointURL = startingIterations > 0 ? asset.checkpointURL : nil
        // Whatever checkpoint the asset has now is replaced when the new result is saved.
        let supersededCheckpointURL = asset.hasTrainingCheckpoint ? asset.checkpointURL : nil
        let outputURL = asset.splatURL(for: selectedQuality.iterations)
        let checkpointURL = asset.checkpointURL(for: selectedQuality.iterations)
        let inputURL = asset.folderURL
        library.beginProcessing(id: asset.id)
        phase = .running
        completedIterations = startingIterations
        splatCount = 0
        statusMessage = "Preparing…"
        statusFraction = nil

        job?.cancel()
        let progressRelay = LocalGaussianProgressRelay(self)
        job = Task { [weak self] in
            let worker = Task.detached(priority: .userInitiated) {
                var trainingURL = inputURL
                if usedOptions.solvesPoses {
                    do {
                        trainingURL = try await PoseSolving.prepare(
                            scanFolder: inputURL, frameWidth: usedOptions.frameWidth,
                            skipFocusDriftAbove: usedOptions.skipsCloseViews ? LocalGaussianTrainingOptions.focusDriftLimit : nil,
                            linesUpPasses: usedOptions.linesUpPasses
                        ) { message, fraction in
                            progressRelay.status(message, fraction)
                        }
                    } catch {
                        if Task.isCancelled { throw CancellationError() }
                        // Train on the poses as captured rather than not at all.
                        print("[pose] could not solve the camera poses (\(error.localizedDescription)); using the poses as captured")
                    }
                } else if usedOptions.refinesPoses {
                    trainingURL = try await PoseRefinement.prepare(scanFolder: inputURL) { message, fraction in
                        progressRelay.status(message, fraction)
                    }
                }
                return try await Self.runTraining(
                    datasetURL: trainingURL,
                    photosURL: inputURL,
                    outputURL: outputURL,
                    resumeCheckpointURL: resumeCheckpointURL,
                    outputCheckpointURL: checkpointURL,
                    quality: selectedQuality,
                    options: usedOptions,
                    viewCount: asset.sampleCount,
                    status: { message, fraction in progressRelay.status(message, fraction) },
                    thermal: { note in progressRelay.thermal(note) }
                ) { iteration, count in
                    progressRelay.publish(iteration: iteration, splatCount: count)
                }
            }
            let result = await withTaskCancellationHandler {
                await worker.result
            } onCancel: {
                worker.cancel()
            }

            guard let self else { return }
            if Task.isCancelled {
                // Stopped by the user: the worker has finished, so it is safe to start another run.
                library.cancelProcessing(id: asset.id)
                phase = asset.hasSplat ? .ready : .idle
                completedIterations = asset.completedSteps ?? 0
                statusMessage = nil
                statusFraction = nil
                thermalNote = nil
                job = nil
                return
            }
            switch result {
            case .success(let outcome):
                completedIterations = outcome.iterations
                splatCount = outcome.splatCount
                phase = .ready
                // The picker only offers targets above what is done; keep its selection one of them.
                if let next = LocalGaussianQuality.extensionTargets(after: outcome.iterations).first {
                    quality = next
                }
                if library.finishProcessing(id: asset.id, steps: outcome.iterations, splatCount: outcome.splatCount, options: usedOptions) {
                    Self.removeSupersededArtifact(previousOutputURL, replacingWith: outputURL)
                    Self.removeSupersededArtifact(supersededCheckpointURL, replacingWith: checkpointURL)
                } else {
                    phase = .failed("Training finished, but the updated result metadata could not be saved.")
                }
            case .failure(let error):
                phase = .failed(error.localizedDescription)
                library.failProcessing(id: asset.id, message: error.localizedDescription)
            }
            job = nil
        }
    }

    /// Asks the running job to stop. It finishes its current step, then the screen returns to idle.
    func stop() {
        guard isRunning else { return }
        statusMessage = "Stopping…"
        statusFraction = nil
        job?.cancel()
    }

    deinit {
        job?.cancel()
    }

    /// Safety limit on the number of views, an evenly spaced subset beyond it.
    nonisolated private static let maximumTrainingViews = 1_500

    nonisolated private static func runTraining(
        datasetURL: URL,
        photosURL: URL,
        outputURL: URL,
        resumeCheckpointURL: URL?,
        outputCheckpointURL: URL,
        quality: LocalGaussianQuality,
        options: LocalGaussianTrainingOptions,
        viewCount: Int,
        status: @escaping @Sendable (String, Double?) -> Void,
        thermal: @escaping @Sendable (String?) -> Void,
        progress: @escaping @Sendable (Int, Int) -> Void
    ) async throws -> (iterations: Int, splatCount: Int) {
        let downscaleFactor = options.downscaleFactor(totalViews: viewCount)
        let views = min(options.viewCount(total: viewCount), Self.maximumTrainingViews)
        let streamsImages = options.streamsImages(totalViews: viewCount)
        print("[train] \(views) of \(viewCount) views, images at 1/\(String(format: "%.2f", downscaleFactor)) size\(streamsImages ? " read from storage" : ""), SH \(options.shDegree), splat limit \(options.maxSplats), pruning \(options.pruning.title)")
        status("Loading images…", 0)
        GaussianDataset.setLoadProgress { done, total in
            status("Loading images… \(done)/\(total)", Double(done) / Double(max(total, 1)))
        }
        let loadStart = ProcessInfo.processInfo.systemUptime
        let dataset = GaussianDataset(
            path: datasetURL.path,
            downscaleFactor: downscaleFactor,
            evalMode: options.measuresQuality,
            testEvery: 8,
            maxViews: views < viewCount ? Int32(views) : 0,
            streamImages: streamsImages
        )
        GaussianDataset.setLoadProgress(nil)
        print(String(format: "[train] images loaded in %.1f s; %d training views, %d held out", ProcessInfo.processInfo.systemUptime - loadStart, dataset.numTrain, dataset.numTest))
        guard dataset.numTrain >= GaussianCapturePolicy.minimumFrames else {
            throw LocalGaussianProcessingError.insufficientViews(dataset.numTrain)
        }

        // Cut-out: find the subject in each training photo now, while memory is still free, and carve
        // the finished splat with those masks.
        var maskFolder: URL?
        if options.subject == .object, options.cutsOutSubject {
            if let region = SubjectRegion.estimate(transformsURL: datasetURL.appendingPathComponent("transforms.json")) {
                let images = (0..<dataset.numTrain).map { URL(fileURLWithPath: dataset.trainImagePath(at: $0)) }
                let outcome = try await SubjectMasks.generate(
                    images: images, transformsURL: datasetURL.appendingPathComponent("transforms.json"),
                    scan: photosURL, region: region, status: status)
                print("[mask] subject found in \(outcome.found) of \(outcome.total) training photos")
                maskFolder = outcome.folder
            } else {
                print("[mask] the cameras do not look at one place, so there is no subject to cut out")
            }
        }

        var config = quality.configuration
        config.downscaleFactor = downscaleFactor
        config.resolutionSchedule = Int32(options.stageLength(iterations: quality.iterations, totalViews: viewCount))
        config.shDegree = Int32(options.shDegree)
        config.maxGaussians = Int32(options.maxSplats)
        config.densifyGradThresh *= options.detail.thresholdMultiplier
        // The gradient that triggers a split shrinks as the images grow: on a 398-view capture the same threshold
        // gave 365k splats at 960 px wide images, 291k at 1440 and 236k at 1920. Halving it at 1920 gave 585k+, so the
        // threshold follows 960 / the width trained at, above 960 px.
        config.densifyGradThresh *= Float(min(1, 960 / (options.frameWidth / Double(downscaleFactor))))
        config.opacityReg = options.pruning.weight
        config.refinePoses = options.refinesPosesWhileTraining
        if options.subject == .object {
            if let region = SubjectRegion.estimate(transformsURL: datasetURL.appendingPathComponent("transforms.json")) {
                // Measured on the Mac: the same quality on the subject with about a third of the splats.
                config.subjectCenter = (region.center.x, region.center.y, region.center.z)
                config.subjectRadius = region.radius
                config.subjectInsideScale = 0.5      // split more readily inside
                config.subjectOutsideScale = 4       // much less outside
                config.subjectOutsideSeedKeep = 0.25 // keep a quarter of the starting points outside
                print(String(format: "[object] subject at (%.2f, %.2f, %.2f), radius %.2f m; cameras %.2f m away, rays miss by %.2f m",
                             region.center.x, region.center.y, region.center.z, region.radius, region.cameraDistance, region.rayMiss))
            } else {
                print("[object] the cameras do not look at one place, so training as a scene")
            }
        }
        // A resumed run starts past the low-resolution stages the schedule above gives a fresh one.
        if resumeCheckpointURL != nil { config.numDownscales = 0 }
        status("Preparing the starting splats…", nil)
        let setupStart = ProcessInfo.processInfo.systemUptime
        let trainer = GaussianTrainer(dataset: dataset, config: config)
        print(String(format: "[train] model ready in %.1f s", ProcessInfo.processInfo.systemUptime - setupStart))
        let startingIteration: Int
        if let resumeCheckpointURL {
            let savedIteration = try checkpointIteration(at: resumeCheckpointURL)
            guard savedIteration <= quality.iterations else {
                throw LocalGaussianProcessingError.checkpointAheadOfTarget(
                    checkpoint: savedIteration,
                    target: quality.iterations
                )
            }
            guard let loadedIteration = trainer.loadCheckpoint(from: resumeCheckpointURL.path),
                  loadedIteration == savedIteration else {
                throw LocalGaussianProcessingError.invalidCheckpoint
            }
            startingIteration = loadedIteration
            if options.refinesPosesWhileTraining {
                let restored = trainer.loadPoses(from: posesURL(for: resumeCheckpointURL).path)
                print("[pose] refined camera poses \(restored ? "restored" : "not found, starting from the captured poses")")
            }
        } else {
            startingIteration = 0
        }

        let trainingStart = ProcessInfo.processInfo.systemUptime
        var thermalState = ProcessInfo.ThermalState.nominal
        var restSeconds = 0.0
        for index in startingIteration..<quality.iterations {
            if Task.isCancelled { throw CancellationError() }

            // Thermal guard. A hot phone is fine until iOS calls it critical; only then does training slow
            // down, resting between steps so the phone can cool. Steps are queued ahead on the GPU, so wait
            // for it to finish before resting, otherwise the pause would not lower the load.
            if index % 10 == 0 {
                let current = ProcessInfo.processInfo.thermalState
                if current != thermalState {
                    print("[thermal] \(thermalState.name) -> \(current.name) at step \(index)")
                    thermalState = current
                    thermal(current == .critical ? "The phone is very hot: training is slowing down to cool it." : nil)
                }
            }
            if thermalState == .critical {
                let restStart = ProcessInfo.processInfo.systemUptime
                msplatSync()
                try await Task.sleep(for: .milliseconds(150))
                restSeconds += ProcessInfo.processInfo.systemUptime - restStart
            }

            let stats = trainer.step()
            if stats.iteration % 100 == 0 {
                print("[mem] step=\(stats.iteration) splats=\(stats.splatCount) footprint=\(currentFootprintMB()) MB")
            }
            if index == 0 || stats.iteration.isMultiple(of: 20) || index + 1 == quality.iterations {
                progress(stats.iteration, stats.splatCount)
            }
        }

        let trainingSeconds = ProcessInfo.processInfo.systemUptime - trainingStart
        thermal(nil)
        print(String(format: "[time] training %.0f s for %d steps (%.1f steps/s), %.0f s of it resting because the phone was hot", trainingSeconds, quality.iterations - startingIteration, Double(quality.iterations - startingIteration) / max(trainingSeconds, 1), restSeconds))
        logColorCheck(trainer: trainer, dataset: dataset, datasetURL: photosURL)
        let temporaryURL = outputURL.appendingPathExtension("writing")
        let temporaryCheckpointURL = outputCheckpointURL.appendingPathExtension("writing")
        let intermediatePLYURL = outputURL.appendingPathExtension("intermediate.ply")
        try removeFileIfPresent(at: outputURL)
        try removeFileIfPresent(at: outputCheckpointURL)
        try? FileManager.default.removeItem(at: temporaryURL)
        try? FileManager.default.removeItem(at: temporaryCheckpointURL)
        try? FileManager.default.removeItem(at: intermediatePLYURL)
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
            try? FileManager.default.removeItem(at: temporaryCheckpointURL)
            try? FileManager.default.removeItem(at: intermediatePLYURL)
        }

        if let maskFolder {
            status("Cutting out the subject…", nil)
            if let kept = trainer.exportPly(to: intermediatePLYURL.path, insideMasksOf: maskFolder.path) {
                print("[mask] kept \(kept) of \(trainer.splatCount) splats inside the subject")
            } else {
                print("[mask] too few masks to cut out the subject; saving all splats")
                trainer.exportPly(to: intermediatePLYURL.path)
            }
        } else {
            trainer.exportPly(to: intermediatePLYURL.path)
        }
        msplatSync()
        let finalSplatCount = try plyVertexCount(at: intermediatePLYURL)
        try CompressedPLY.write(from: intermediatePLYURL, to: temporaryURL, shDegree: options.shDegree)
        guard FileManager.default.fileExists(atPath: temporaryURL.path) else {
            throw LocalGaussianProcessingError.exportFailed
        }

        guard trainer.saveCheckpoint(to: temporaryCheckpointURL.path),
              try fileSize(at: temporaryCheckpointURL) > 0 else {
            throw LocalGaussianProcessingError.checkpointSaveFailed
        }

        try FileManager.default.moveItem(at: temporaryCheckpointURL, to: outputCheckpointURL)
        if options.refinesPosesWhileTraining {
            // The splats were fitted to these poses; a resumed run has to start from them.
            if !trainer.savePoses(to: posesURL(for: outputCheckpointURL).path) {
                print("[pose] could not save the refined camera poses")
            }
        }
        try FileManager.default.moveItem(at: temporaryURL, to: outputURL)

        // Last, because scoring the held-out views with refined cameras nudges their poses and disturbs the
        // optimiser state; everything that matters is saved by now.
        if options.measuresQuality, dataset.numTest > 0 {
            let raw = trainer.evaluate()
            print(String(format: "[eval] held-out views: %d  PSNR %.2f dB  SSIM %.4f  L1 %.4f  splats %d",
                         raw.numTest, raw.psnr, raw.ssim, raw.l1, raw.numGaussians))
            if options.refinesPosesWhileTraining {
                let aligned = trainer.evaluateAligned()
                print(String(format: "[eval] after aligning the held-out poses: PSNR %.2f dB  SSIM %.4f  L1 %.4f",
                             aligned.psnr, aligned.ssim, aligned.l1))
            }
        }
        return (trainer.iteration, finalSplatCount)
    }

    /// Where the refined camera poses of a checkpoint are kept.
    nonisolated private static func posesURL(for checkpoint: URL) -> URL {
        checkpoint.appendingPathExtension("poses")
    }

    nonisolated private static func checkpointIteration(at url: URL) throws -> Int {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let header = try handle.read(upToCount: 12), header.count == 12 else {
            throw LocalGaussianProcessingError.invalidCheckpoint
        }

        func uint32(at offset: Int) -> UInt32 {
            header[offset..<(offset + 4)].enumerated().reduce(into: UInt32(0)) { value, byte in
                value |= UInt32(byte.element) << UInt32(byte.offset * 8)
            }
        }

        guard uint32(at: 0) == 0x4C50_534D,
              uint32(at: 4) == 1 else {
            throw LocalGaussianProcessingError.invalidCheckpoint
        }

        let iteration = Int(uint32(at: 8))
        guard iteration <= LocalGaussianQuality.maximumIterations else {
            throw LocalGaussianProcessingError.invalidCheckpoint
        }
        return iteration
    }

    nonisolated private static func fileSize(at url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    nonisolated private static func removeFileIfPresent(at url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    nonisolated private static func removeSupersededArtifact(_ previousURL: URL?, replacingWith currentURL: URL) {
        guard let previousURL,
              previousURL.standardizedFileURL != currentURL.standardizedFileURL else {
            return
        }
        try? FileManager.default.removeItem(at: previousURL)
        if previousURL.pathExtension == "ckpt" { try? FileManager.default.removeItem(at: posesURL(for: previousURL)) }
    }

    nonisolated private static func plyVertexCount(at url: URL) throws -> Int {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let headerTerminator = Data("end_header\n".utf8)
        guard let data = try handle.read(upToCount: 1_048_576),
              let terminatorRange = data.range(of: headerTerminator),
              let header = String(data: data[..<terminatorRange.upperBound], encoding: .utf8) else {
            throw LocalGaussianProcessingError.exportFailed
        }
        for line in header.split(whereSeparator: \.isNewline) {
            let fields = line.split(separator: " ")
            if fields.count == 3,
               fields[0] == "element",
               fields[1] == "vertex",
               let count = Int(fields[2]) {
                return count
            }
        }
        throw LocalGaussianProcessingError.exportFailed
    }
}

enum GaussianCapturePolicy {
    nonisolated static let minimumFrames = 5
}

private enum LocalGaussianProcessingError: LocalizedError {
    case insufficientViews(Int)
    case exportFailed
    case invalidCheckpoint
    case checkpointAheadOfTarget(checkpoint: Int, target: Int)
    case checkpointSaveFailed

    var errorDescription: String? {
        switch self {
        case .insufficientViews(let count):
            return "At least \(GaussianCapturePolicy.minimumFrames) camera views are required. \(count) were found."
        case .exportFailed:
            return "Local processing finished without producing a splat file."
        case .invalidCheckpoint:
            return "The saved training checkpoint is invalid."
        case .checkpointAheadOfTarget(let checkpoint, let target):
            return "The saved checkpoint is already at iteration \(checkpoint), above the selected \(target) target."
        case .checkpointSaveFailed:
            return "Training finished, but its resume checkpoint could not be saved."
        }
    }
}

nonisolated private func currentFootprintMB() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
}

/// TEMP DEBUG: compares the average colour of the trained splat's renders with that of the
/// captured photos, to tell a training brightness problem from a viewer or capture one.
nonisolated private func logColorCheck(trainer: GaussianTrainer, dataset: GaussianDataset, datasetURL: URL) {
    let count = dataset.numTrain
    guard count > 0 else { return }
    let samples = min(8, count)

    var renderSum = SIMD3<Double>.zero
    var renderPixels = 0.0
    for k in 0..<samples {
        let image = trainer.render(cameraIndex: k * count / samples)
        let n = image.width * image.height
        guard n > 0, image.pixels.count >= n * 3 else { continue }
        for i in 0..<n {
            renderSum += SIMD3(Double(image.pixels[i * 3]), Double(image.pixels[i * 3 + 1]), Double(image.pixels[i * 3 + 2]))
        }
        renderPixels += Double(n)
    }

    let imagesURL = datasetURL.appendingPathComponent("images")
    let files = ((try? FileManager.default.contentsOfDirectory(at: imagesURL, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension.lowercased() == "jpg" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    var photoSum = SIMD3<Double>.zero
    var photoPixels = 0.0
    let step = max(1, files.count / 8)
    for url in stride(from: 0, to: files.count, by: step).map({ files[$0] }) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 64,
                kCGImageSourceCreateThumbnailWithTransform: true,
              ] as CFDictionary) else { continue }
        let w = thumb.width, h = thumb.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let context = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
        context.draw(thumb, in: CGRect(x: 0, y: 0, width: w, height: h))
        for i in 0..<(w * h) {
            photoSum += SIMD3(Double(rgba[i * 4]), Double(rgba[i * 4 + 1]), Double(rgba[i * 4 + 2])) / 255
        }
        photoPixels += Double(w * h)
    }

    guard renderPixels > 0, photoPixels > 0 else { return }
    let render = renderSum / renderPixels
    let photo = photoSum / photoPixels
    print(String(format: "[color] mean RGB  renders: %.3f %.3f %.3f   photos: %.3f %.3f %.3f",
                 render.x, render.y, render.z, photo.x, photo.y, photo.z))
}

extension ProcessInfo.ThermalState {
    nonisolated var name: String {
        switch self {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}
