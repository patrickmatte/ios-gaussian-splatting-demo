import Foundation

/// What a training run is allowed to spend: how many views, how large the images, how much colour
/// detail, how many splats. Chosen before the first training, saved with the scan, and reused when
/// the result is extended (the colour detail has to match the checkpoint).
nonisolated struct LocalGaussianTrainingOptions: Codable, Hashable, Sendable {
    /// Share of the captured views to train on, evenly spaced through the capture.
    var viewFraction: Double = 1
    /// Image size relative to the captured frame. 0 chooses one from the number of views.
    var imageScale: Double = 0
    /// Spherical-harmonics degree: 0 is base colour only, 1 to 3 add view-dependent colour.
    var shDegree: Int = 2
    /// Densification stops adding splats beyond this. 0 means no limit.
    var maxSplats: Int = 500_000
    var detail: Detail = .standard
    /// Holds out every eighth view and reports PSNR and SSIM on them after training. Costs those views.
    var measuresQuality = false
    /// Re-solves the camera poses with Object Capture before training (slow, but fixes ARKit drift).
    var refinesPoses = false
    /// What was captured. Object mode spends the splats on the thing the cameras circle.
    var subject = Subject.scene
    /// How hard splats the photos hardly need are faded out and dropped.
    /// Light: on the HDR bowl capture, judged in SuperSplat against four other opacity settings (Standard, no pruning,
    /// and faster opacity learning), Light with an opacity learning rate of 0.012 looked best by far. Standard pulls
    /// opacities down further and left the bowl more see-through.
    var pruning = Pruning.light
    /// Object mode only: finds the subject in every photo and trains on it alone, with no background.
    var cutsOutSubject = false
    /// Nudges each photo's camera pose while the splat trains. ARKit's positions are off by a few millimetres,
    /// which blurs close-up captures; on a bowl scan this gave about +0.9 dB.
    var refinesPosesWhileTraining = true
    /// Solves the camera poses from the photos before training (corner matching and bundle adjustment, like
    /// photogrammetry software does). ARKit's poses are a centimetre and a degree off; solved ones halved the
    /// render-versus-photo misregistration of a bowl scan and lifted held-out PSNR by 1.4 to 2.4 dB with visibly
    /// sharper edges. Replaces the two refinements above.
    var solvesPoses = true
    /// Before solving, Object Capture lines up the passes of a capture that went round the same place several times
    /// (see `PoseSolving.linedUpStart`), and the solver starts from that when ARKit drifted between the passes by more than
    /// `PoseSolving.minimumDriftCentimetres`. Room captures drift and the splat shows objects two or three times without it.
    /// Needs the solver and a device that runs Object Capture.
    var linesUpPasses = true
    /// Leaves out the photos taken with the lens focused much closer than in most of the capture: ARKit reports a focal
    /// length more than `focusDriftLimit` above the capture's median. Up close the lens refocuses and the field of view
    /// moves (1335 to 1428 px in one capture), hand shake smears the image, and a white bowl captured from 35 cm came
    /// out with a cloud of splats around its rim; without those photos it came out clean. The rule follows the focus,
    /// not a distance, so it does not depend on the size of the subject: a figurine shot entirely up close keeps every
    /// photo. Needs the solved camera poses.
    var skipsCloseViews = true
    static let focusDriftLimit = 0.02
    /// Size of the captured photos: 1920x1440 for video frames, about 4032x3024 for high-resolution photos. Set from
    /// the scan before the options are shown or used.
    var frameWidth = 1920.0
    var frameHeight = 1440.0

    enum Subject: String, Codable, CaseIterable, Identifiable, Sendable {
        case scene, object

        var id: String { rawValue }
        var title: String { self == .scene ? "Scene" : "Object" }
    }

    init(viewFraction: Double = 1, imageScale: Double = 0, shDegree: Int = 2,
         maxSplats: Int = 500_000, detail: Detail = .standard, measuresQuality: Bool = false,
         refinesPoses: Bool = false, subject: Subject = .scene, pruning: Pruning = .light,
         cutsOutSubject: Bool = false, refinesPosesWhileTraining: Bool = true, solvesPoses: Bool = true,
         skipsCloseViews: Bool = true, linesUpPasses: Bool = true) {
        self.viewFraction = viewFraction
        self.imageScale = imageScale
        self.shDegree = shDegree
        self.maxSplats = maxSplats
        self.detail = detail
        self.measuresQuality = measuresQuality
        self.refinesPoses = refinesPoses
        self.subject = subject
        self.pruning = pruning
        self.cutsOutSubject = cutsOutSubject
        self.refinesPosesWhileTraining = refinesPosesWhileTraining
        self.solvesPoses = solvesPoses
        self.skipsCloseViews = skipsCloseViews
        self.linesUpPasses = linesUpPasses
    }

    // Decodes with defaults so options saved before a field existed still load.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        viewFraction = try c.decodeIfPresent(Double.self, forKey: .viewFraction) ?? 1
        imageScale = try c.decodeIfPresent(Double.self, forKey: .imageScale) ?? 0
        shDegree = try c.decodeIfPresent(Int.self, forKey: .shDegree) ?? 1
        maxSplats = try c.decodeIfPresent(Int.self, forKey: .maxSplats) ?? 350_000
        detail = try c.decodeIfPresent(Detail.self, forKey: .detail) ?? .standard
        measuresQuality = try c.decodeIfPresent(Bool.self, forKey: .measuresQuality) ?? false
        refinesPoses = try c.decodeIfPresent(Bool.self, forKey: .refinesPoses) ?? false
        subject = try c.decodeIfPresent(Subject.self, forKey: .subject) ?? .scene
        pruning = try c.decodeIfPresent(Pruning.self, forKey: .pruning) ?? .standard
        cutsOutSubject = try c.decodeIfPresent(Bool.self, forKey: .cutsOutSubject) ?? false
        // Splats trained before this option existed were fitted to the poses as captured.
        refinesPosesWhileTraining = try c.decodeIfPresent(Bool.self, forKey: .refinesPosesWhileTraining) ?? false
        // Splats trained before the solver existed used the poses as captured (or the other refinements).
        solvesPoses = try c.decodeIfPresent(Bool.self, forKey: .solvesPoses) ?? false
        skipsCloseViews = try c.decodeIfPresent(Bool.self, forKey: .skipsCloseViews) ?? false
        // On for scans saved before it existed: it only acts when the solver is on, and it is what a room capture needs.
        linesUpPasses = try c.decodeIfPresent(Bool.self, forKey: .linesUpPasses) ?? true
        frameWidth = try c.decodeIfPresent(Double.self, forKey: .frameWidth) ?? 1920
        frameHeight = try c.decodeIfPresent(Double.self, forKey: .frameHeight) ?? 1440
    }

    /// An L1 pull on every splat's opacity (msplat's `opacityReg`). Splats the photos give almost no
    /// gradient fade out and are culled, while the ones that matter hold their opacity. Measured on
    /// a bowl scan (held-out photos, 10K steps): off 24.9 dB with 800k splats; light the same with
    /// 680k; standard 24.9 with 350k; strong 24.4 with 66k. Cutting splats by raising the split
    /// threshold instead loses 0.6 dB at 150k.
    enum Pruning: String, Codable, CaseIterable, Identifiable, Sendable {
        case off, light, standard, strong

        var id: String { rawValue }

        var title: String {
            switch self {
            case .off: "Off"
            case .light: "Light"
            case .standard: "Standard"
            case .strong: "Strong"
            }
        }

        var weight: Float {
            switch self {
            case .off: 0
            case .light: 0.01
            case .standard: 0.02
            case .strong: 0.05
            }
        }
    }

    enum Detail: String, Codable, CaseIterable, Identifiable, Sendable {
        case fewer, standard, more

        var id: String { rawValue }

        var title: String {
            switch self {
            case .fewer: "Fewer splats"
            case .standard: "Standard"
            case .more: "More splats"
            }
        }

        /// Scales the gradient threshold that decides when a splat is split; lower splits more.
        var thresholdMultiplier: Float {
            switch self {
            case .fewer: 1.5
            case .standard: 1
            case .more: 0.6
            }
        }
    }

    static let `default` = LocalGaussianTrainingOptions()

    /// The everyday recipe, the same for any capture (an object on a stool or a whole room); `TrainingRecipe` picks only
    /// the steps and the image size. SH 2 and a 500k limit: what Scantic's files use (SH 2, 540k to 610k splats, 7,000 to
    /// 7,750 steps), and SH 3 looked no different on a bowl or a bedroom, while its splats take a third more memory (a
    /// bedroom at full size did not fit the phone with SH 3). Light pruning, solved poses with the passes lined up when
    /// ARKit drifted, and the close-up photos left out, as tested.
    static func recipe(frameSize: (width: Double, height: Double), trainingWidth: Double) -> Self {
        var options = Self(viewFraction: 1, imageScale: min(1, trainingWidth / max(frameSize.width, 1)), shDegree: 2,
                           maxSplats: 500_000, detail: .standard, measuresQuality: false, refinesPoses: false,
                           subject: .scene, pruning: .light, cutsOutSubject: false, refinesPosesWhileTraining: false,
                           solvesPoses: true, skipsCloseViews: true, linesUpPasses: true)
        options.frameWidth = frameSize.width
        options.frameHeight = frameSize.height
        return options
    }

    static let viewFractions: [Double] = [1, 0.75, 0.5, 0.25]
    static let imageScales: [Double] = [1, 0.75, 0.5, 1.0 / 3.0, 0.25]
    static let splatLimits: [Int] = [250_000, 350_000, 500_000, 750_000, 1_000_000, 0]

    var frameSize: (width: Double, height: Double) { (frameWidth, frameHeight) }

    /// How many views this run trains on.
    func viewCount(total: Int) -> Int {
        guard total > 0 else { return 0 }
        return min(total, max(1, Int((Double(total) * viewFraction).rounded(.up))))
    }

    /// The divisor msplat applies to the captured frame. Auto is 50%: the photos no longer have to fit in memory
    /// (see `streamsImages`), so a long capture is not dropped to a smaller size.
    func downscaleFactor(totalViews: Int) -> Float {
        if imageScale > 0 { return Float(1 / imageScale) }
        // Video frames (1920 wide) train at 960. Larger photos carry detail the video frames lack, so they train at
        // 1920 wide.
        return frameWidth > 2200 ? Float(frameWidth / 1920) : 2
    }

    /// Steps in each of the two low-resolution stages (images at 1/4, then 1/2 of the size trained at); the rest of the
    /// run is at full size. A full-size step is cheap with 960 px images, so the stages are 7/20 of the run there. With
    /// 1920 px images one costs about 235 ms against 69 ms at half size on the phone, so from 6K steps the full-size stage
    /// is kept to 500 steps (1000 from 10K). On the HDR bowl at 100%, 7K steps: 27.70 dB held out in 8.4 minutes with
    /// 500 full-size steps, 27.74 dB in 10 minutes with 1500, and about 14 minutes with the 2100 the 7/20 rule gives.
    func stageLength(iterations: Int, totalViews: Int) -> Int {
        let ramp = max(600, iterations * 7 / 20)
        guard iterations >= 6_000, frameWidth / Double(downscaleFactor(totalViews: totalViews)) >= 1500 else { return ramp }
        return max(ramp, (iterations - (iterations >= 10_000 ? 1_000 : 500)) / 2)
    }

    /// Photos held in memory as 8-bit RGB take about 4 bytes a pixel counting the lower resolutions. Past this
    /// many pixels in total (about 1.5 GB) they are read from storage instead, one per step.
    nonisolated private static let residentPixelLimit = 380_000_000.0

    /// Whether msplat decodes each photo when a step needs it instead of keeping them all in memory. Always for
    /// images above 50%, which would not fit (460 photos at full size are 3.8 GB), and for captures too large to
    /// hold at the chosen size. Costs a decode per step on another thread, and saves the minutes spent loading.
    func streamsImages(totalViews: Int) -> Bool {
        let factor = Double(downscaleFactor(totalViews: totalViews))
        if factor < 1.99 { return true }
        let pixelsPerView = frameWidth * frameHeight / (factor * factor)
        return Double(viewCount(total: totalViews)) * pixelsPerView > Self.residentPixelLimit
    }

    /// Rough memory held before any splat exists, in megabytes: the training images when they are held in memory
    /// (about 4 bytes a pixel counting the smaller copies) plus the buffers that scale with image size (about
    /// 0.5 KB a pixel, allocated when training reaches full resolution). The splats come on top of this.
    func estimatedImageMegabytes(totalViews: Int) -> Int {
        let factor = Double(downscaleFactor(totalViews: totalViews))
        let pixelsPerView = frameWidth * frameHeight / (factor * factor)
        let images = streamsImages(totalViews: totalViews) ? 0 : Double(viewCount(total: totalViews)) * pixelsPerView * 4
        let buffers = pixelsPerView * 500
        return Int((images + buffers) / 1_000_000)
    }

    /// Rough training time on an iPhone, in seconds. Per step the phone takes about 32 ms per million splats and
    /// 80 ms per million pixels. The pixel cost is fitted to a 100% run, 4K steps, 421k splats: 390 s with the package
    /// compiled optimised and the phone only warm. Earlier runs gave 123 ms (a hot phone, and a Debug build whose
    /// host code ran unoptimised), so a hot phone can take about 50% longer than this.
    func estimatedTrainingSeconds(iterations: Int, totalViews: Int) -> Double {
        let factor = Double(downscaleFactor(totalViews: totalViews))
        let finalMegapixels = frameWidth * frameHeight / (factor * factor) / 1e6
        let stage = stageLength(iterations: iterations, totalViews: totalViews)
        let finalSplats = Double(min(maxSplats == 0 ? 600_000 : maxSplats, 450_000)) / 1e6
        let stageDivisors = [16.0, 4.0, 1.0]
        var milliseconds = 0.0
        for step in 0..<max(iterations, 0) {
            let pixels = finalMegapixels / stageDivisors[min(step / stage, 2)]
            let splats = finalSplats < 0.23 ? finalSplats : 0.23 + (finalSplats - 0.23) * min(1, Double(step) / 1400)
            milliseconds += 32 * splats + 80 * pixels
        }
        return milliseconds / 1000 * (shDegree >= 3 ? 1.1 : 1)
    }

    /// Rough time to solve the camera poses: about 0.25 s a view on an iPhone (103 s for 501 views with 1200 corners a
    /// photo; 2400 take about 25% longer), plus Object Capture on up to 200 photos when it lines up the passes (110 to
    /// 200 s on a phone).
    func estimatedPoseSolvingSeconds(totalViews: Int) -> Double {
        guard solvesPoses else { return 0 }
        return Double(viewCount(total: totalViews)) * 0.25 + (linesUpPasses ? 150 : 0)
    }

    func imageSizeLabel(totalViews: Int) -> String {
        let factor = Double(downscaleFactor(totalViews: totalViews))
        let width = Int(frameWidth / factor), height = Int(frameHeight / factor)
        return "\(width)×\(height)"
    }

    func summary(totalViews: Int) -> String {
        let views = viewCount(total: totalViews)
        return "\(views) views · \(imageSizeLabel(totalViews: totalViews)) · SH \(shDegree)" + (subject == .object && cutsOutSubject ? " · cut out" : "")
    }
}

/// What the user chooses before training, and nothing else (see `LocalGaussianTrainingOptions.recipe`). Phone training
/// time measured on the Mac and scaled to the phone runs, for a 371-photo bowl and a 797-photo bedroom: quick 3.0 to
/// 3.4 min, standard 4.2 to 4.9 min, high quality about 7 to 9 min; the first training of a scan also solves its camera
/// poses (4 to 6 min).
nonisolated enum TrainingRecipe: String, CaseIterable, Identifiable, Codable, Sendable {
    case quick, standard, high

    var id: String { rawValue }
    static let `default`: Self = .standard

    var title: String {
        switch self {
        case .quick: "Quick"
        case .standard: "Standard"
        case .high: "High quality"
        }
    }

    var detail: String {
        switch self {
        case .quick: "Fewer steps: a little less fine detail on thin, far things."
        case .standard: "Recommended."
        case .high: "Full-size images: sharper close up, about twice as long."
        }
    }

    var quality: LocalGaussianQuality { self == .quick ? .maximum : .extended }

    /// Images 960 pixels wide, or 1920 for high quality, whatever size the photos are.
    func options(frameSize: (width: Double, height: Double)) -> LocalGaussianTrainingOptions {
        .recipe(frameSize: frameSize, trainingWidth: self == .high ? 1920 : 960)
    }
}

