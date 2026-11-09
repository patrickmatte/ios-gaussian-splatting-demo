import SwiftUI

/// Says when the camera poses of a scan were solved already (they are reused), and lets the user solve them again.
struct SolvedPosesNote: View {
    let scanFolder: URL
    @Binding var options: LocalGaussianTrainingOptions
    @State private var refresh = 0
    @State private var cleared = false

    var body: some View {
        let _ = refresh
        let solvedEarlier = PoseSolving.hasSolved(scanFolder, linesUpPasses: options.linesUpPasses)
        // Cleared by a tap and not solved since: the message stays until the next training has solved them again.
        let justCleared = cleared && !solvedEarlier
        HStack(alignment: .firstTextBaseline) {
            Group {
                if justCleared {
                    Label("Saved poses removed. They are solved again when training starts.", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else if !options.solvesPoses {
                    Text("Camera poses are not solved: the poses as captured are used.")
                } else if solvedEarlier {
                    Text("Camera poses were solved earlier and will be reused.")
                } else {
                    Text("Camera poses will be solved when training starts.")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            Spacer()
            // Always there. It also turns the solver on, so one tap is enough to get fresh poses at the next training.
            Button {
                PoseSolving.discardSolved(scanFolder)
                options.solvesPoses = true
                options.refinesPoses = false
                options.refinesPosesWhileTraining = false
                cleared = true
                refresh += 1
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } label: {
                Label("Solve again", systemImage: justCleared ? "checkmark" : "arrow.clockwise")
            }
            .font(.caption)
            .buttonStyle(.bordered)
        }
    }
}

/// How long a run will take, roughly, so a choice of image size or steps is not a surprise.
struct TrainingTimeEstimate: View {
    let options: LocalGaussianTrainingOptions
    let totalViews: Int
    let iterations: Int
    /// The scan's poses are kept from an earlier training, so they cost nothing this time.
    var posesSolved = false

    var body: some View {
        let training = options.estimatedTrainingSeconds(iterations: iterations, totalViews: totalViews)
        let poses = posesSolved ? 0 : options.estimatedPoseSolvingSeconds(totalViews: totalViews)
        let minutes = { (seconds: Double) in seconds < 90 ? "\(Int(seconds.rounded())) s" : String(format: "%.0f min", seconds / 60) }
        Text("About \(minutes(training + poses)) in all (training \(minutes(training))\(poses > 0 ? ", camera poses \(minutes(poses))" : "")). A rough guess from earlier runs: longer when the phone is hot.")
            .font(.caption)
            .foregroundStyle(training + poses > 600 ? Color.orange : Color.secondary)
    }
}

/// The choices made before a training run. They trade quality against memory, time and heat.
struct TrainingOptionsForm: View {
    @Binding var options: LocalGaussianTrainingOptions
    let totalViews: Int
    /// The scan's photo size; the sizes shown below and the memory estimate follow it.
    var frameSize: (width: Double, height: Double) = (1920, 1440)

    private var imageMegabytes: Int { options.estimatedImageMegabytes(totalViews: totalViews) }

    private func applyFrameSize() {
        options.frameWidth = frameSize.width
        options.frameHeight = frameSize.height
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Picker("Subject", selection: $options.subject) {
                    ForEach(LocalGaussianTrainingOptions.Subject.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .onChange(of: options.subject) { _, subject in
                    // Measured on a bowl: with the splats focused, SH 3 costs little memory and adds about
                    // 0.8 dB, and SH 3 at 5K steps beats SH 1 at 7K while finishing sooner.
                    // A 398-view room scan scored 24.3 / 24.7 / 24.6 dB at SH 1 / 2 / 3: SH 2 is as good as 3 and cheaper.
                    options.shDegree = subject == .object ? 3 : 2
                    // Light for both: it keeps more splats solid (an object lost about 0.2 dB with Standard), and
                    // on the HDR bowl capture it looked best by far in SuperSplat.
                    options.pruning = .light
                }
                Text(options.subject == .object
                     ? "Puts the splats on the thing the cameras circle and keeps the background coarse. For one object photographed from all sides. Uses SH 3: 5K steps is already better than 7K was."
                     : "Spreads the splats over everything in view. For rooms and landscapes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if options.subject == .object {
                VStack(alignment: .leading, spacing: 2) {
                    Toggle("Cut out the subject", isOn: $options.cutsOutSubject)
                        .font(.subheadline)
                    Text("Trains as usual, then removes everything but the subject: it is found in each photo, and splats that fall outside it in most views are dropped. Adds a minute or two. Needs one subject the cameras circle.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Picker("Views", selection: $options.viewFraction) {
                ForEach(LocalGaussianTrainingOptions.viewFractions, id: \.self) { fraction in
                    let count = LocalGaussianTrainingOptions(viewFraction: fraction).viewCount(total: totalViews)
                    Text("\(Int((fraction * 100).rounded()))% · \(count) views").tag(fraction)
                }
            }

            Picker("Image size", selection: $options.imageScale) {
                let auto = { () -> LocalGaussianTrainingOptions in var o = options; o.imageScale = 0; return o }()
                Text("Auto · \(auto.imageSizeLabel(totalViews: totalViews))").tag(0.0)
                ForEach(LocalGaussianTrainingOptions.imageScales, id: \.self) { scale in
                    let sized = { () -> LocalGaussianTrainingOptions in var o = options; o.imageScale = scale; return o }()
                    Text("\(Int((scale * 100).rounded()))% · \(sized.imageSizeLabel(totalViews: totalViews))").tag(scale)
                }
            }
            .onAppear { applyFrameSize() }
            .onChange(of: frameSize.width) { _, _ in applyFrameSize() }
            if options.streamsImages(totalViews: totalViews) {
                Text("Photos are read from storage while training, so loading is instant and memory stays low. Larger images train slower: roughly 2× the time at 75% and 3× at 100%, compared with 50%.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Picker("Colour detail (SH)", selection: $options.shDegree) {
                    ForEach(0...3, id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                Text("0 is base colour only. 1 to 3 add view-dependent colour and use more memory.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Picker("Splat limit", selection: $options.maxSplats) {
                ForEach(LocalGaussianTrainingOptions.splatLimits, id: \.self) { limit in
                    Text(limit == 0 ? "No limit" : "\(limit / 1000)k").tag(limit)
                }
            }

            Picker("Splitting", selection: $options.detail) {
                ForEach(LocalGaussianTrainingOptions.Detail.allCases) { Text($0.title).tag($0) }
            }

            VStack(alignment: .leading, spacing: 4) {
                Picker("Pruning", selection: $options.pruning) {
                    ForEach(LocalGaussianTrainingOptions.Pruning.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                Text("Fades out splats the photos barely need. Light is the default: it looked best in the viewer. Standard removes more splats, so training is lighter and cooler, but surfaces end up more see-through. Off keeps every splat.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Solve camera poses from the photos", isOn: $options.solvesPoses)
                    .font(.subheadline)
                    .onChange(of: options.solvesPoses) { _, on in
                        if on {
                            options.refinesPoses = false
                            options.refinesPosesWhileTraining = false
                        }
                    }
                Text("Matches corners between neighbouring photos and adjusts every camera together, like photogrammetry software. ARKit's poses are a centimetre and a degree off, which blurs the splat; solved poses halved that error on a bowl scan (+1.4 to +2.4 dB held out, visibly sharper). Takes a minute or two for 400 photos. Recommended.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Line up repeated passes first (Object Capture)", isOn: $options.linesUpPasses)
                    .font(.subheadline)
                    .disabled(!options.solvesPoses || !PoseRefinement.isSupported)
                Text("Walking round a room several times, ARKit drifts between the passes, so the splat shows things two or three times. Object Capture solves up to 200 of the photos together; when it finds ARKit's passes more than 2.5 cm apart, the solver starts from its lined-up poses (a bedroom: 22.3 to 25.0 dB held out, no doubles), otherwise from ARKit's as before (a bowl orbit, 1.2 cm). Adds 2 to 3 minutes the first time.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Leave out photos taken much closer than the rest", isOn: $options.skipsCloseViews)
                    .font(.subheadline)
                    .disabled(!options.solvesPoses)
                Text("When the phone gets much closer than during most of the capture, the lens refocuses, the field of view drifts and hand shake blurs the photo. A bowl photographed from 35 cm came out with a cloud of splats around its rim; without those photos it came out clean. It follows the focus the iPhone reports, not a fixed distance, so a small subject shot entirely up close keeps every photo. Needs the solved poses.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Refine camera poses (Object Capture)", isOn: $options.refinesPoses)
                    .font(.subheadline)
                    // Together they scored no better than either alone (bowl and bedroom), so one switches the other off.
                    .onChange(of: options.refinesPoses) { _, on in
                        if on {
                            options.refinesPosesWhileTraining = false
                            options.solvesPoses = false
                        }
                    }
                Text("Re-solves the camera positions before training. Slow and memory heavy, but removes the drift that blurs long captures: +3 dB on a 782-view room. Use it for big scans; use the option below for close-up objects.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Refine camera poses while training", isOn: $options.refinesPosesWhileTraining)
                    .font(.subheadline)
                    .onChange(of: options.refinesPosesWhileTraining) { _, on in
                        if on {
                            options.refinesPoses = false
                            options.solvesPoses = false
                        }
                    }
                Text("Nudges each photo's camera position as the splat trains, fixing the few-millimetre errors in ARKit's poses that blur close-up captures. About +1 dB on a bowl scan, none on a large room; a little slower. Not combined with Object Capture refinement.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Toggle("Measure quality (holds out ⅛ of the views)", isOn: $options.measuresQuality)
                .font(.subheadline)

            VStack(alignment: .leading, spacing: 2) {
                Text("Images and image buffers ≈ \(imageMegabytes.formatted()) MB")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(imageMegabytes > 1_200 ? Color.red : Color.secondary)
                if imageMegabytes > 1_200 {
                    Label("Little memory is left for splats: training may stop adding splats early, and a run at this size used 3.3 GB, close to where iOS stops the app. Try 75% or 50%.", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Text("More views, larger images and more iterations cost time, heat and battery.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
