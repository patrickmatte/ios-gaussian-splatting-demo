// Copyright 2026 Voxelio.
// Licensed under the PolyForm Noncommercial License 1.0.0.

import ARKit
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import RealityKit
import SwiftUI
import UIKit
import simd

struct GaussianCaptureView: View {
    @Environment(\.dismiss) private var dismiss

    @ObservedObject var library: LocalGaussianLibrary
    let onComplete: (LocalGaussianAsset) -> Void

    @State private var isRecording = false
    @State private var isFinishing = false
    @State private var frameCount = 0
    @State private var pointCount = 0
    @State private var status = "Move around the subject and keep it in view."
    @State private var errorMessage: String?
    @State private var asksToDiscard = false
    @State private var cameraAccess = AVCaptureDevice.authorizationStatus(for: .video)
    /// What the camera session is doing, shown under the title so a black preview can be told apart:
    /// no permission, no frames, interrupted, or an error.
    @State private var cameraState = "Starting the camera…"
    /// Distance from the camera to what is in the middle of the view, from the LiDAR depth, in metres.
    @State private var subjectDistance: Float?
    /// Exposure compensation in stops. Windows and reflections clip to pure white at 0 and the trainer then
    /// paints them as blown-out patches; a negative value keeps their detail. Remembered between captures.
    @AppStorage("captureExposureBias") private var exposureBias = -1.0
    /// Share of the view that is clipped to white, 0...1.
    @State private var blownOutShare: Float = 0
    /// Asks ARKit for HDR video. Experimental: whether the frames come out better is for a test capture to show.
    @AppStorage("captureHDRVideo") private var hdrVideo = false
    /// What the camera is actually delivering after the HDR switch.
    @State private var hdrReport: CaptureHDRReport?
    /// Takes a full-resolution photo (about 12 MP, the camera's still pipeline) for every view instead of using the
    /// 1920x1440 video frame. Development only: on a 579-photo capture the photos were twice as sharp as video frames,
    /// but each shot is tone-mapped on its own and the splat kept only 15% of their fine detail (31% from video), so it
    /// came out no sharper, and Object Capture took 5.4 minutes on the full-size photos. Release builds use video frames.
    @AppStorage("captureHighResPhotos") private var highResPhotos = false
    @State private var highResReport: CaptureHDRReport?

    /// Whether this capture takes photos: only in development builds, whatever was left on during testing.
    private var usesHighResPhotos: Bool {
        #if DEBUG
        highResPhotos
        #else
        false
        #endif
    }

    var body: some View {
        Group {
            if cameraAccess == .denied || cameraAccess == .restricted {
                ContentUnavailableView {
                    Label("Camera access is off", systemImage: "camera.fill")
                } description: {
                    Text("Allow the camera for this app in Settings to capture views.")
                } actions: {
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Close") { dismiss() }
                }
            } else if cameraAccess == .notDetermined {
                ProgressView("Waiting for camera permission…")
                    .tint(.white)
                    .foregroundStyle(.white)
            } else if ARWorldTrackingConfiguration.isSupported {
                camera
            } else {
                ContentUnavailableView(
                    "ARKit unavailable",
                    systemImage: "arkit",
                    description: Text("Run this capture flow on a supported physical iPhone or iPad.")
                )
            }
        }
        .background(Color.black)
        .task {
            // Ask first, so a refusal shows a message instead of a black screen.
            if cameraAccess == .notDetermined {
                _ = await AVCaptureDevice.requestAccess(for: .video)
                cameraAccess = AVCaptureDevice.authorizationStatus(for: .video)
            }
        }
        .alert("Discard capture?", isPresented: $asksToDiscard) {
            Button("Discard", role: .destructive) { dismiss() }
            Button("Keep capturing", role: .cancel) {}
        } message: {
            Text("The captured camera views will be deleted.")
        }
        .alert("Capture failed", isPresented: errorIsPresented) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "The capture could not be saved.")
        }
    }

    private var camera: some View {
        ZStack {
            LocalGaussianCameraSurface(
                shouldRecord: isRecording,
                seedDensity: 100,
                minimumConfidence: 1,
                maximumDepth: 5,
                onProgress: receive,
                onFinish: finish,
                onCameraState: { cameraState = $0 },
                onDistance: { subjectDistance = $0 },
                onClipping: { blownOutShare = $0 },
                onHDR: { report in
                    hdrReport = report
                    if report.fellBack { hdrVideo = false }
                },
                onHighRes: { report in
                    highResReport = report
                    if report.fellBack { highResPhotos = false }
                },
                exposureBias: Float(exposureBias),
                hdr: hdrVideo,
                highRes: usesHighResPhotos
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                captureHeader
                Spacer()
                captureControls
            }
        }
        .persistentSystemOverlays(.hidden)
    }

    private var captureHeader: some View {
        HStack(spacing: 12) {
            Button {
                if isRecording {
                    asksToDiscard = true
                } else if !isFinishing {
                    dismiss()
                }
            } label: {
                Image(systemName: "xmark")
                    .font(.headline)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .disabled(isFinishing)

            VStack(alignment: .leading, spacing: 2) {
                Text("Gaussian capture")
                    .font(.headline)
                Text(captureStateLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if isRecording || isFinishing {
                Label(frameCount.formatted(), systemImage: "photo.stack.fill")
                    .font(.subheadline.monospacedDigit().weight(.semibold))
                    .padding(.horizontal, 12)
                    .frame(height: 38)
                    .background(.ultraThinMaterial, in: Capsule())
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private var captureControls: some View {
        VStack(spacing: 14) {
            if isRecording || isFinishing {
                ProgressView(
                    value: Double(min(frameCount, GaussianCaptureWriter.minimumFrames)),
                    total: Double(GaussianCaptureWriter.minimumFrames)
                )
                .tint(canFinish ? .green : .white)

                HStack {
                    Label("\(frameCount) views", systemImage: "camera.fill")
                    Spacer()
                    Label("\(pointCount) seeds", systemImage: "circle.grid.3x3.fill")
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }

            if let hint = clippingHint {
                Label(hint, systemImage: "sun.max.trianglebadge.exclamationmark")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
            }

            HStack(spacing: 10) {
                Image(systemName: "sun.min")
                Slider(value: $exposureBias, in: -3...1, step: 0.25)
                Image(systemName: "sun.max")
                Text(String(format: "%+.2g EV", exposureBias))
                    .font(.caption.monospacedDigit())
                    .frame(width: 58, alignment: .trailing)
            }
            .foregroundStyle(.secondary)
            .disabled(isFinishing)

            VStack(spacing: 2) {
                Toggle("HDR video (experimental)", isOn: $hdrVideo)
                    .font(.footnote)
                    .disabled(isRecording || isFinishing)
                if let report = hdrReport, hdrVideo || report.fellBack {
                    Text(report.text)
                        .font(.caption2)
                        .foregroundStyle(report.isActive ? Color.secondary : Color.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                #if DEBUG
                Toggle("High-resolution photos (development)", isOn: $highResPhotos)
                    .font(.footnote)
                    .disabled(isRecording || isFinishing)
                if let report = highResReport, highResPhotos || report.fellBack {
                    Text(report.text)
                        .font(.caption2)
                        .foregroundStyle(report.isActive ? Color.secondary : Color.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                #endif
            }

            if let hint = distanceHint {
                Label(hint, systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
            }

            Text(status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button(action: toggleCapture) {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 72, height: 72)

                    if isFinishing {
                        ProgressView()
                            .tint(.white)
                    } else if isRecording {
                        RoundedRectangle(cornerRadius: 7)
                            .fill(.red)
                            .frame(width: 30, height: 30)
                    } else {
                        Circle()
                            .fill(.red)
                            .frame(width: 58, height: 58)
                    }
                }
            }
            .buttonStyle(.plain)
            .disabled(isFinishing || (isRecording && !canFinish))
            .opacity(isRecording && !canFinish ? 0.55 : 1)
            .accessibilityLabel(isRecording ? "Finish capture" : "Start capture")
        }
        .padding(20)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .padding(.horizontal, 16)
        .padding(.bottom, 24)
    }

    /// Close-up captures are the hardest on camera tracking: a few millimetres of position error is several
    /// pixels at 0.4 m and half that at 1 m. Tell the user while they can still change it.
    private var distanceHint: String? {
        guard let distance = subjectDistance, distance > 0.05, distance < 0.8 else { return nil }
        return String(format: "%.0f cm from the middle of the view. Step back to 1 m or more, with the surroundings in view, for a sharper result.", distance * 100)
    }

    /// Light that clips is lost for good: nothing in training can bring back the detail of a window or a shiny
    /// floor that was recorded as pure white.
    private var clippingHint: String? {
        guard blownOutShare > 0.04 else { return nil }
        return String(format: "%.0f%% of the view is blown out to white. Lower the exposure with the slider so windows and reflections keep their detail.", blownOutShare * 100)
    }

    private var canFinish: Bool {
        frameCount >= GaussianCaptureWriter.minimumFrames
    }

    private var captureStateLabel: String {
        if isFinishing { return "Saving dataset" }
        if isRecording { return "Recording views" }
        return cameraState
    }

    private var errorIsPresented: Binding<Bool> {
        Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )
    }

    private func toggleCapture() {
        if isRecording {
            guard canFinish else { return }
            status = "Writing transforms and seed cloud…"
            isFinishing = true
            isRecording = false
        } else {
            frameCount = 0
            pointCount = 0
            status = "Move steadily around the subject."
            isRecording = true
        }
    }

    private func receive(_ progress: GaussianCaptureProgress) {
        frameCount = progress.frameCount
        pointCount = progress.pointCount
        status = progress.message
    }

    private func finish(_ result: Result<LocalGaussianCapturePackage, Error>) {
        isRecording = false
        isFinishing = false

        switch result {
        case .success(let package):
            do {
                let asset = try library.importCapture(package)
                onComplete(asset)
            } catch {
                status = "Ready to try again."
                errorMessage = error.localizedDescription
            }
        case .failure(let error):
            status = "Ready to try again."
            errorMessage = error.localizedDescription
        }
    }
}

private struct LocalGaussianCameraSurface: UIViewRepresentable {
    let shouldRecord: Bool
    let seedDensity: Float
    let minimumConfidence: Int
    let maximumDepth: Float
    let onProgress: (GaussianCaptureProgress) -> Void
    let onFinish: (Result<LocalGaussianCapturePackage, Error>) -> Void
    let onCameraState: (String) -> Void
    let onDistance: (Float?) -> Void
    let onClipping: (Float) -> Void
    let onHDR: (CaptureHDRReport) -> Void
    let onHighRes: (CaptureHDRReport) -> Void
    let exposureBias: Float
    let hdr: Bool
    let highRes: Bool

    func makeCoordinator() -> GaussianARCaptureCoordinator {
        GaussianARCaptureCoordinator(onProgress: onProgress, onFinish: onFinish,
                                     onCameraState: onCameraState, onDistance: onDistance, onClipping: onClipping,
                                     onHDR: onHDR, onHighRes: onHighRes)
    }

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.environment.background = .cameraFeed()
        context.coordinator.attach(to: view.session)
        context.coordinator.applyExposureBias(exposureBias)
        context.coordinator.runSession(hdr: hdr, highRes: highRes)
        return view
    }

    func updateUIView(_ view: ARView, context: Context) {
        context.coordinator.onProgress = onProgress
        context.coordinator.onFinish = onFinish
        context.coordinator.onCameraState = onCameraState
        context.coordinator.onDistance = onDistance
        context.coordinator.onClipping = onClipping
        context.coordinator.onHDR = onHDR
        context.coordinator.onHighRes = onHighRes
        context.coordinator.applyExposureBias(exposureBias)
        context.coordinator.setCaptureMode(hdr: hdr, highRes: highRes)
        context.coordinator.configureSeedCloud(
            density: seedDensity,
            minimumConfidence: minimumConfidence,
            maximumDepth: maximumDepth
        )
        context.coordinator.setRecording(shouldRecord)
    }

    static func dismantleUIView(_ view: ARView, coordinator: GaussianARCaptureCoordinator) {
        coordinator.cancel()
        view.session.pause()
        view.session.delegate = nil
    }
}

/// What the HDR switch did, for the line under it.
nonisolated struct CaptureHDRReport: Sendable {
    var text: String
    /// The camera is delivering HDR frames.
    var isActive: Bool
    /// HDR could not be used and the session went back to the standard camera, so the switch should turn off.
    var fellBack = false
}

private nonisolated struct GaussianCaptureProgress: Sendable {
    let frameCount: Int
    let pointCount: Int
    let message: String
}

private nonisolated final class GaussianARCaptureCoordinator: NSObject, ARSessionDelegate, @unchecked Sendable {
    var onProgress: (GaussianCaptureProgress) -> Void
    var onFinish: (Result<LocalGaussianCapturePackage, Error>) -> Void
    var onCameraState: (String) -> Void
    var onDistance: (Float?) -> Void
    var onClipping: (Float) -> Void
    var onHDR: (CaptureHDRReport) -> Void
    var onHighRes: (CaptureHDRReport) -> Void

    private enum Mode {
        case idle
        case recording
        case finishing
    }

    private let cameraQueue = DispatchQueue(label: "com.voxelio.gaussian.camera", qos: .userInitiated)
    private let writerQueue = DispatchQueue(label: "com.voxelio.gaussian.writer", qos: .userInitiated)
    private let snapshotBuilder = GaussianFrameSnapshotBuilder()

    private var mode: Mode = .idle
    private var writer: GaussianCaptureWriter?
    private var writePending = false
    private var previousPose: simd_float4x4?
    private var previousTimestamp: TimeInterval = 0
    private var acceptedFrames = 0
    private var acceptedPoints = 0
    private var seedDensity: Float = 100
    private var minimumConfidence = 2
    private var maximumDepth: Float = 5

    private var lastStateReport: TimeInterval = 0
    /// Sharpness of recent frames that were otherwise ready to be saved, to tell a blurred one from the usual.
    private var sharpnessHistory: [Float] = []
    private var lastAcceptTime: TimeInterval = 0
    private var skippedForBlur = 0
    private var lastBlurReport: TimeInterval = 0
    private var lastDistanceReport: TimeInterval = 0
    private var lastStateText = ""
    private var watchdog: DispatchWorkItem?
    private var framesSeen = 0
    /// The session whose camera this coordinator configures.
    private weak var session: ARSession?
    /// What the running session was last started with, so a screen update does not restart it; main thread only.
    private var requestedHDR: Bool?
    private var requestedHighRes: Bool?
    /// Camera-queue copies: HDR was asked for and a format that supports it was found, and the time after which the
    /// next frame says what the camera really delivers (frames from before the restart arrive for a moment).
    private var hdrRequested = false
    private var formatReportDue: TimeInterval?
    private var hdrFormatSummary = ""
    /// Camera-queue state of the high-resolution photo mode: it is on, the first photo has been reported, and how
    /// many requests in a row failed (three failures put the capture back on video frames).
    private var highResActive = false
    private var highResReported = false
    private var highResFailures = 0
    private var highResRequestedAt: TimeInterval = 0

    init(
        onProgress: @escaping (GaussianCaptureProgress) -> Void,
        onFinish: @escaping (Result<LocalGaussianCapturePackage, Error>) -> Void,
        onCameraState: @escaping (String) -> Void,
        onDistance: @escaping (Float?) -> Void,
        onClipping: @escaping (Float) -> Void,
        onHDR: @escaping (CaptureHDRReport) -> Void,
        onHighRes: @escaping (CaptureHDRReport) -> Void
    ) {
        self.onProgress = onProgress
        self.onFinish = onFinish
        self.onCameraState = onCameraState
        self.onDistance = onDistance
        self.onClipping = onClipping
        self.onHDR = onHDR
        self.onHighRes = onHighRes
    }

    private var requestedExposureBias: Float?
    private var appliedExposureBias: Float?

    /// Sets the camera's exposure compensation. The automatic exposure stays on; it just aims lower or higher.
    func applyExposureBias(_ stops: Float, force: Bool = false) {
        requestedExposureBias = stops
        guard force || appliedExposureBias != stops else { return }
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            let bias = min(max(stops, device.minExposureTargetBias), device.maxExposureTargetBias)
            device.setExposureTargetBias(bias, completionHandler: nil)
            appliedExposureBias = stops
        } catch {
            print("[capture] could not set the exposure compensation: \(error.localizedDescription)")
        }
    }

    // MARK: Session

    /// (Re)starts the AR session. With `hdr`, the video format is one that supports HDR, as close to the usual
    /// one as possible, because ARKit only captures HDR when the format allows it.
    func runSession(hdr: Bool, highRes: Bool = false) {
        guard let session else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        configuration.isAutoFocusEnabled = true
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }

        // High-resolution photos need ARKit's format for high-resolution frame capture.
        let photoFormat = highRes ? ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing : nil
        if let photoFormat { configuration.videoFormat = photoFormat }

        var hdrFormat: ARConfiguration.VideoFormat?
        var summary = ""
        if hdr {
            if let photoFormat {
                hdrFormat = photoFormat.isVideoHDRSupported ? photoFormat : nil
            } else {
                hdrFormat = Self.hdrVideoFormat(near: configuration.videoFormat)
            }
            if let hdrFormat {
                configuration.videoFormat = hdrFormat
                configuration.videoHDRAllowed = true
                summary = Self.describe(hdrFormat)
            }
        }
        print("[capture] starting the camera, HDR requested: \(hdr), high-resolution photos requested: \(highRes), format: \(Self.describe(configuration.videoFormat)), HDR format found: \(hdrFormat != nil), photo format found: \(photoFormat != nil)")

        let usesHDR = hdrFormat != nil
        let usesPhotos = photoFormat != nil
        requestedHDR = hdr
        requestedHighRes = highRes
        cameraQueue.async { [weak self] in
            self?.hdrRequested = usesHDR
            self?.hdrFormatSummary = summary
            self?.highResActive = usesPhotos
            self?.highResReported = false
            self?.highResFailures = 0
            // Report on the first frame after the camera has had time to restart.
            self?.formatReportDue = (hdr || highRes) ? ProcessInfo.processInfo.systemUptime + 1.5 : nil
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])

        if hdr && !usesHDR {
            let why = highRes ? "HDR is not available with high-resolution photos on this iPhone." : "This iPhone has no HDR video format."
            emit(CaptureHDRReport(text: why + " Using the standard camera.", isActive: false, fellBack: true))
        } else if usesHDR {
            emit(CaptureHDRReport(text: "Starting HDR video (\(summary))…", isActive: false))
        }
        if highRes && !usesPhotos {
            emitHighRes(CaptureHDRReport(text: "This iPhone cannot take high-resolution photos while tracking. Using video frames.",
                                         isActive: false, fellBack: true))
        } else if usesPhotos {
            emitHighRes(CaptureHDRReport(text: "Starting high-resolution photos (camera \(Self.describe(configuration.videoFormat)))…", isActive: false))
        }

        // ARKit may set the camera up again once it starts delivering frames, which drops the exposure bias.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            guard let self, let stops = requestedExposureBias else { return }
            applyExposureBias(stops, force: true)
        }
    }

    /// Tells the screen, after the current view update: the callback changes its state.
    private func emit(_ report: CaptureHDRReport) {
        DispatchQueue.main.async { [weak self] in self?.onHDR(report) }
    }

    private func emitHighRes(_ report: CaptureHDRReport) {
        DispatchQueue.main.async { [weak self] in self?.onHighRes(report) }
    }

    /// Switches HDR and high-resolution photos by restarting the session. Only used between captures.
    func setCaptureMode(hdr: Bool, highRes: Bool) {
        guard requestedHDR != hdr || requestedHighRes != highRes else { return }
        runSession(hdr: hdr, highRes: highRes)
    }

    /// The video format with HDR support closest to `current` in size, then in frame rate, on the same camera.
    private static func hdrVideoFormat(near current: ARConfiguration.VideoFormat) -> ARConfiguration.VideoFormat? {
        func area(_ format: ARConfiguration.VideoFormat) -> CGFloat {
            format.imageResolution.width * format.imageResolution.height
        }
        return ARWorldTrackingConfiguration.supportedVideoFormats
            .filter {
                $0.isVideoHDRSupported && $0.captureDevicePosition == current.captureDevicePosition
                    && $0.captureDeviceType == current.captureDeviceType
            }
            .min { lhs, rhs in
                let l = (abs(area(lhs) - area(current)), abs(lhs.framesPerSecond - current.framesPerSecond))
                let r = (abs(area(rhs) - area(current)), abs(rhs.framesPerSecond - current.framesPerSecond))
                return l < r
            }
    }

    private static func describe(_ format: ARConfiguration.VideoFormat) -> String {
        "\(Int(format.imageResolution.width))×\(Int(format.imageResolution.height)) at \(format.framesPerSecond) fps"
    }

    /// Says what the camera delivers, once per start: the pixel format of the frames and whether the device has
    /// HDR video on. Printed as well, so a log shows it.
    private func reportFormat(of frame: ARFrame) {
        let buffer = frame.capturedImage
        let type = CVPixelBufferGetPixelFormatType(buffer)
        let name = Self.fourCharacterCode(type)
        let transfer = (CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil) as? String) ?? "none"
        let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
        let deviceHDR = device?.isVideoHDREnabled ?? false
        let headroom = CIImage(cvPixelBuffer: buffer).contentHeadroom
        let hdrFrames = GaussianFrameSnapshotBuilder.isHighDynamicRange(buffer)
        print("[capture] frames are \(name) \(CVPixelBufferGetWidth(buffer))×\(CVPixelBufferGetHeight(buffer)), transfer \(transfer), device HDR video: \(deviceHDR), content headroom \(headroom)")

        let report: CaptureHDRReport
        if hdrFrames {
            report = CaptureHDRReport(text: "HDR video on: \(hdrFormatSummary), \(name) frames (\(transfer)), tone-mapped when saved.", isActive: true)
        } else if deviceHDR {
            report = CaptureHDRReport(text: "HDR video on: \(hdrFormatSummary), but ARKit still hands over standard \(name) frames.", isActive: true)
        } else {
            report = CaptureHDRReport(text: "HDR was asked for but the camera reports it off (\(name) frames). Nothing is gained.", isActive: false)
        }
        emit(report)
    }

    static func fourCharacterCode(_ code: OSType) -> String {
        let bytes = [UInt8((code >> 24) & 255), UInt8((code >> 16) & 255), UInt8((code >> 8) & 255), UInt8(code & 255)]
        return String(bytes: bytes, encoding: .ascii) ?? String(code)
    }

    func attach(to session: ARSession) {
        self.session = session
        session.delegate = self
        session.delegateQueue = cameraQueue
        // No frame within a few seconds means the preview will be black: say why it might be.
        let check = DispatchWorkItem { [weak self] in
            guard let self, framesSeen == 0 else { return }
            reportState("No camera image. Check Settings › Privacy › Camera, and close other apps using the camera.")
        }
        watchdog = check
        cameraQueue.asyncAfter(deadline: .now() + 3, execute: check)
    }

    private func reportState(_ text: String) {
        guard text != lastStateText else { return }
        lastStateText = text
        DispatchQueue.main.async { [weak self] in self?.onCameraState(text) }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        reportState("Camera interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        reportState("Camera resumed")
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        reportState(Self.describe(camera.trackingState))
    }

    private static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "Ready"
        case .notAvailable: return "Tracking not available"
        case .limited(let reason):
            switch reason {
            case .initializing: return "Starting tracking… move the phone slowly"
            case .excessiveMotion: return "Moving too fast"
            case .insufficientFeatures: return "Not enough detail in view"
            case .relocalizing: return "Relocalizing…"
            @unknown default: return "Tracking limited"
            }
        }
    }

    func setRecording(_ shouldRecord: Bool) {
        cameraQueue.async { [weak self] in
            guard let self else { return }
            switch (mode, shouldRecord) {
            case (.idle, true):
                start()
            case (.recording, false):
                finish()
            default:
                break
            }
        }
    }

    func configureSeedCloud(density: Float, minimumConfidence: Int, maximumDepth: Float) {
        cameraQueue.async { [weak self] in
            self?.seedDensity = min(max(density, 10), 500)
            self?.minimumConfidence = min(max(minimumConfidence, 0), 2)
            self?.maximumDepth = min(max(maximumDepth, 0.5), 10)
        }
    }

    func cancel() {
        cameraQueue.async { [weak self] in
            guard let self else { return }
            let abandoned = writer
            mode = .idle
            writer = nil
            writerQueue.async {
                abandoned?.discard()
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        autoreleasepool {
            if framesSeen == 0 { reportState(Self.describe(frame.camera.trackingState)) }
            framesSeen += 1
            if let due = formatReportDue, frame.timestamp >= due {
                formatReportDue = nil
                reportFormat(of: frame)
            }
            reportDistance(frame)
            guard mode == .recording,
                  !writePending,
                  frame.camera.trackingState.isCaptureReady,
                  shouldAccept(frame)
            else {
                return
            }

            // Skip a frame that is blurrier than the ones around it: a blurred photo makes the whole splat
            // softer. The next sharp moment follows within a fraction of a second.
            let sharpness = Self.sharpness(of: frame.capturedImage)
            guard isSharpEnough(sharpness, at: frame.timestamp) else {
                skippedForBlur += 1
                if frame.timestamp - lastBlurReport >= 0.6 {
                    lastBlurReport = frame.timestamp
                    report("Moving too fast: slow down for sharper views.")
                }
                return
            }

            if highResActive {
                requestPhoto(during: frame, sharpness: sharpness)
                return
            }

            guard let snapshot = snapshotBuilder.copySnapshot(
                from: frame,
                seedDensity: seedDensity,
                minimumConfidence: minimumConfidence,
                maximumDepth: maximumDepth,
                sharpness: sharpness
            ) else {
                report("Waiting for a stable camera image.")
                return
            }

            // Only value data leaves this delegate callback. Neither ARFrame nor its
            // camera CVPixelBuffers are retained by the writer queue.
            previousPose = frame.camera.transform
            previousTimestamp = frame.timestamp
            lastAcceptTime = frame.timestamp
            commit(snapshot)
        }
    }

    /// Hands a finished view to the writer, one at a time.
    private func commit(_ snapshot: GaussianFrameSnapshot) {
        writePending = true
        let nextIndex = acceptedFrames
        let destination = writer

        writerQueue.async { [weak self] in
            do {
                try destination?.append(snapshot, index: nextIndex)
                self?.cameraQueue.async { [weak self] in
                    guard let self, mode == .recording else { return }
                    acceptedFrames += 1
                    acceptedPoints += snapshot.points.count
                    writePending = false
                    report("View \(acceptedFrames) saved. Keep moving steadily.")
                }
            } catch {
                self?.cameraQueue.async { [weak self] in
                    self?.writePending = false
                    self?.abort(with: error)
                }
            }
        }
    }

    /// Asks the camera for a full-resolution photo now. The LiDAR seed points come from the video frame the request
    /// was made on; the photo brings its own pose and calibration. The next request waits for this one to be written.
    private func requestPhoto(during frame: ARFrame, sharpness: Float) {
        guard let session else { return }
        let points = snapshotBuilder.seedPoints(
            from: frame, seedDensity: seedDensity, minimumConfidence: minimumConfidence, maximumDepth: maximumDepth)
        previousPose = frame.camera.transform
        previousTimestamp = frame.timestamp
        lastAcceptTime = frame.timestamp
        writePending = true
        highResRequestedAt = ProcessInfo.processInfo.systemUptime
        let completion: (ARFrame?, Error?) -> Void = { [weak self] photo, error in
            self?.cameraQueue.async { [weak self] in
                guard let self else { return }
                guard mode == .recording else { writePending = false; return }
                guard let photo, error == nil,
                      let snapshot = snapshotBuilder.copySnapshot(
                        from: photo, seedDensity: seedDensity, minimumConfidence: minimumConfidence,
                        maximumDepth: maximumDepth, sharpness: sharpness, seedPoints: points)
                else {
                    writePending = false
                    highResFailures += 1
                    print("[capture] high-resolution photo failed (\(highResFailures)): \(error?.localizedDescription ?? "no image")")
                    if highResFailures >= 3 {
                        highResActive = false
                        emitHighRes(CaptureHDRReport(text: "High-resolution photos kept failing. Using video frames.", isActive: false, fellBack: true))
                    }
                    return
                }
                highResFailures = 0
                if !highResReported {
                    highResReported = true
                    let ms = (ProcessInfo.processInfo.systemUptime - highResRequestedAt) * 1000
                    print("[capture] first high-resolution photo \(snapshot.width)×\(snapshot.height) after \(Int(ms)) ms, \(snapshot.jpeg.count / 1024) KB")
                    emitHighRes(CaptureHDRReport(text: "High-resolution photos on: \(snapshot.width)×\(snapshot.height), \(Int(ms)) ms to take one.", isActive: true))
                }
                previousPose = photo.camera.transform
                commit(snapshot)
            }
        }
        if Self.canSilenceShutter,
           let format = (session.configuration as? ARWorldTrackingConfiguration)?.videoFormat {
            // Without this every photo plays the camera sound. A new settings object each time: one cannot be reused.
            let settings = AVCapturePhotoSettings(from: format.defaultPhotoSettings)
            settings.isShutterSoundSuppressionEnabled = true
            session.captureHighResolutionFrame(using: settings, completion: completion)
        } else {
            session.captureHighResolutionFrame(completion: completion)
        }
    }

    /// Whether the shutter sound may be turned off here. Some countries require it (the photo output then says no, and
    /// asking for silence anyway would stop the app), so the sound stays on there.
    private static let canSilenceShutter: Bool = {
        let allowed = AVCapturePhotoOutput().isShutterSoundSuppressionSupported
        print("[capture] the shutter sound can be turned off on this iPhone: \(allowed)")
        return allowed
    }()

    func session(_ session: ARSession, didFailWithError error: Error) {
        cameraQueue.async { [weak self] in
            guard let self else { return }
            reportState("Camera error: \(error.localizedDescription)")
            if mode == .recording {
                abort(with: error)
            } else if hdrRequested || highResActive {
                // The HDR or high-resolution format did not start (a LiDAR or tracking mode it cannot run with, say): go back.
                let hadHDR = hdrRequested, hadPhotos = highResActive
                hdrRequested = false
                highResActive = false
                print("[capture] the special camera format failed: \(error.localizedDescription)")
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    runSession(hdr: false, highRes: false)
                    if hadHDR {
                        emit(CaptureHDRReport(text: "HDR video did not start: \(error.localizedDescription) Using the standard camera.",
                                              isActive: false, fellBack: true))
                    }
                    if hadPhotos {
                        emitHighRes(CaptureHDRReport(text: "High-resolution photos did not start: \(error.localizedDescription) Using video frames.",
                                                     isActive: false, fellBack: true))
                    }
                }
            }
        }
    }

    private func start() {
        do {
            writer = try GaussianCaptureWriter(seedDensity: seedDensity)
            previousPose = nil
            previousTimestamp = 0
            acceptedFrames = 0
            acceptedPoints = 0
            writePending = false
            sharpnessHistory.removeAll()
            lastAcceptTime = 0
            skippedForBlur = 0
            mode = .recording
            report("Move steadily around the subject.")
        } catch {
            abort(with: error)
        }
    }

    private func finish() {
        guard let writer else { return }
        mode = .finishing
        self.writer = nil

        writerQueue.async { [weak self] in
            do {
                let package = try writer.finish()
                self?.cameraQueue.async { [weak self] in
                    self?.mode = .idle
                    self?.deliver(.success(package))
                }
            } catch {
                writer.discard()
                self?.cameraQueue.async { [weak self] in
                    self?.mode = .idle
                    self?.deliver(.failure(error))
                }
            }
        }
    }

    private func abort(with error: Error) {
        let abandoned = writer
        writer = nil
        mode = .idle
        writerQueue.async {
            abandoned?.discard()
        }
        deliver(.failure(error))
    }

    /// Accepts a frame unless it is clearly blurrier than the sharp frames of the last few seconds. The
    /// reference is the 75th percentile of recent frames, so a plain wall does not set it; after two
    /// seconds without a saved frame the bar drops, so a dim room still gets captured.
    private func isSharpEnough(_ sharpness: Float, at timestamp: TimeInterval) -> Bool {
        sharpnessHistory.append(sharpness)
        if sharpnessHistory.count > 90 { sharpnessHistory.removeFirst(sharpnessHistory.count - 90) }
        guard sharpnessHistory.count >= 12 else { return true }
        let sorted = sharpnessHistory.sorted()
        let reference = sorted[sorted.count * 3 / 4]
        let waited = lastAcceptTime == 0 ? 0 : timestamp - lastAcceptTime
        let share: Float = waited > 2 ? 0.5 : 0.72
        return sharpness >= reference * share
    }

    /// Mean gradient of the luma plane on a sparse grid (about 170k samples), 0...1. Comparable between
    /// frames of the same scene, not between scenes.
    static func sharpness(of pixelBuffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        // A pixel format this does not know reads as sharp, so nothing is skipped.
        guard let luma = LumaPlane(pixelBuffer) else { return 1 }
        var total = 0, count = 0
        for y in Swift.stride(from: 4, to: luma.height - 4, by: 4) {
            for x in Swift.stride(from: 4, to: luma.width - 4, by: 4) {
                total += abs(luma.value(x: x + 2, y: y) - luma.value(x: x - 2, y: y))
                    + abs(luma.value(x: x, y: y + 2) - luma.value(x: x, y: y - 2))
                count += 1
            }
        }
        return count > 0 ? Float(total) / Float(count) / 255 : 1
    }

    /// Tells the screen how far the middle of the view is, about twice a second.
    private func reportDistance(_ frame: ARFrame) {
        guard frame.timestamp - lastDistanceReport >= 0.5 else { return }
        lastDistanceReport = frame.timestamp
        var distance: Float?
        if let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
            let map = depth.depthMap
            CVPixelBufferLockBaseAddress(map, .readOnly)
            if CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
               let base = CVPixelBufferGetBaseAddress(map)?.assumingMemoryBound(to: Float32.self) {
                let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
                let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float32>.stride
                var values: [Float] = []
                for y in Swift.stride(from: h * 2 / 5, to: h * 3 / 5, by: 3) {
                    for x in Swift.stride(from: w * 2 / 5, to: w * 3 / 5, by: 3) {
                        let z = base[y * stride + x]
                        if z.isFinite, z > 0.05 { values.append(z) }
                    }
                }
                if !values.isEmpty { distance = values.sorted()[values.count / 2] }
            }
            CVPixelBufferUnlockBaseAddress(map, .readOnly)
        }
        let blown = Self.blownOutShare(of: frame.capturedImage)
        DispatchQueue.main.async { [weak self] in
            self?.onDistance(distance)
            self?.onClipping(blown)
        }
    }

    /// Share of the luma plane at or above 250 of 255, on a sparse grid.
    static func blownOutShare(of pixelBuffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let luma = LumaPlane(pixelBuffer) else { return 0 }
        var bright = 0, count = 0
        for y in Swift.stride(from: 0, to: luma.height, by: 8) {
            for x in Swift.stride(from: 0, to: luma.width, by: 8) {
                if luma.value(x: x, y: y) >= 250 { bright += 1 }
                count += 1
            }
        }
        return count > 0 ? Float(bright) / Float(count) : 0
    }

    private func shouldAccept(_ frame: ARFrame) -> Bool {
        guard let previousPose else { return true }
        guard frame.timestamp - previousTimestamp >= 0.32 else { return false }

        let oldPosition = SIMD3<Float>(previousPose.columns.3.x, previousPose.columns.3.y, previousPose.columns.3.z)
        let newPosition = SIMD3<Float>(frame.camera.transform.columns.3.x, frame.camera.transform.columns.3.y, frame.camera.transform.columns.3.z)
        let translation = simd_distance(oldPosition, newPosition)

        let oldForward = -SIMD3<Float>(previousPose.columns.2.x, previousPose.columns.2.y, previousPose.columns.2.z)
        let newForward = -SIMD3<Float>(frame.camera.transform.columns.2.x, frame.camera.transform.columns.2.y, frame.camera.transform.columns.2.z)
        let cosine = simd_clamp(simd_dot(simd_normalize(oldForward), simd_normalize(newForward)), -1, 1)
        let angle = acosf(cosine)
        return translation >= 0.055 || angle >= 0.09
    }

    private func report(_ message: String) {
        let progress = GaussianCaptureProgress(
            frameCount: acceptedFrames,
            pointCount: acceptedPoints,
            message: message
        )
        DispatchQueue.main.async { [weak self] in
            self?.onProgress(progress)
        }
    }

    private func deliver(_ result: Result<LocalGaussianCapturePackage, Error>) {
        DispatchQueue.main.async { [weak self] in
            self?.onFinish(result)
        }
    }
}

/// The luma plane of a camera frame, read as 0...255 whatever the bit depth: the standard 8-bit frames and the
/// 10-bit ones HDR video may use. The buffer must stay locked while this is used.
private nonisolated struct LumaPlane {
    let width: Int
    let height: Int
    private let bytes: UnsafePointer<UInt8>?
    private let words: UnsafePointer<UInt16>?
    private let stride: Int
    private let isVideoRange: Bool

    init?(_ buffer: CVPixelBuffer) {
        guard CVPixelBufferGetPlaneCount(buffer) > 0,
              let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
        switch CVPixelBufferGetPixelFormatType(buffer) {
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            bytes = UnsafePointer(base.assumingMemoryBound(to: UInt8.self))
            words = nil
            stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange, kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            bytes = nil
            words = UnsafePointer(base.assumingMemoryBound(to: UInt16.self))
            stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) / 2
        default:
            return nil
        }
        isVideoRange = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange]
            .contains(CVPixelBufferGetPixelFormatType(buffer))
        width = CVPixelBufferGetWidthOfPlane(buffer, 0)
        height = CVPixelBufferGetHeightOfPlane(buffer, 0)
    }

    func value(x: Int, y: Int) -> Int {
        if let words {
            // 10 bits, stored in the top of a 16-bit word. Video range is 64...940 of 1023.
            let word = Int(words[y * stride + x])
            return isVideoRange ? min(max((word - 4096) * 255 / 56064, 0), 255) : word >> 8
        }
        let byte = Int(bytes![y * stride + x])
        return isVideoRange ? min(max((byte - 16) * 255 / 219, 0), 255) : byte
    }
}

private nonisolated struct GaussianFrameSnapshot: Sendable {
    let jpeg: Data
    let width: Int
    let height: Int
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let cameraToWorldRows: [[Float]]
    let points: [GaussianSeedPoint]
    /// Frame time on the capture clock, and what the camera did for this frame.
    let timestamp: TimeInterval
    let exposureDuration: TimeInterval
    let exposureOffset: Float
    let sharpness: Float
    /// Pixel format of the camera frame ("420f" for the usual 8-bit one) and whether it was HDR and tone-mapped.
    let pixelFormat: String
    let toneMapped: Bool
}

private nonisolated struct GaussianSeedPoint: Sendable {
    let position: SIMD3<Float>
    let luminance: UInt8
}

private nonisolated final class GaussianFrameSnapshotBuilder: @unchecked Sendable {
    private let context = CIContext(options: [.cacheIntermediates: false])
    /// HDR frames go through half-float intermediates so values above white survive until the tone mapping.
    private let hdrContext = CIContext(options: [
        .cacheIntermediates: false,
        .workingFormat: Int(CIFormat.RGBAh.rawValue),
    ])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    /// How far above SDR white the camera's HDR signal reaches: HLG peaks at 1000 nits against a reference white of
    /// 203, the 4.93 Core Image reports for an HLG buffer. Set explicitly, because the filter clips early without it,
    /// and fixed, so every view of a capture is mapped the same way.
    private static let sourceHeadroom: Float = 4.93

    /// True for frames that carry more than SDR range: an HLG or PQ transfer function, or a 10-bit format with none.
    static func isHighDynamicRange(_ buffer: CVPixelBuffer) -> Bool {
        let transfer = CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil) as? String
        if let transfer {
            return transfer == kCVImageBufferTransferFunction_ITU_R_2100_HLG as String
                || transfer == kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String
        }
        let type = CVPixelBufferGetPixelFormatType(buffer)
        return type == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
            || type == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
    }

    /// The LiDAR seed points of a video frame.
    func seedPoints(from frame: ARFrame, seedDensity: Float, minimumConfidence: Int, maximumDepth: Float) -> [GaussianSeedPoint] {
        copySeedPoints(
            depth: frame.smoothedSceneDepth ?? frame.sceneDepth,
            image: frame.capturedImage,
            intrinsics: frame.camera.intrinsics,
            cameraToWorld: frame.camera.transform,
            seedDensity: seedDensity,
            minimumConfidence: minimumConfidence,
            maximumDepth: maximumDepth
        )
    }

    /// The view for `frame`. `seedPoints` replaces the frame's own LiDAR points: a high-resolution photo carries no
    /// depth, so its points come from the video frame the photo was requested on.
    func copySnapshot(
        from frame: ARFrame,
        seedDensity: Float,
        minimumConfidence: Int,
        maximumDepth: Float,
        sharpness: Float,
        seedPoints: [GaussianSeedPoint]? = nil
    ) -> GaussianFrameSnapshot? {
        let pixelBuffer = frame.capturedImage
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        // An 8-bit JPEG cannot hold HDR: roll the highlights into SDR range instead of letting them clip.
        let toneMapped = Self.isHighDynamicRange(pixelBuffer)
        if toneMapped {
            let filter = CIFilter.toneMapHeadroom()
            filter.inputImage = image
            filter.sourceHeadroom = Self.sourceHeadroom
            filter.targetHeadroom = 1
            image = filter.outputImage ?? image
        }
        guard let jpeg = (toneMapped ? hdrContext : context).jpegRepresentation(
            of: image,
            colorSpace: colorSpace,
            options: [
                CIImageRepresentationOption(
                    rawValue: kCGImageDestinationLossyCompressionQuality as String
                ): 0.88
            ]
        ) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let intrinsics = frame.camera.intrinsics
        let pose = frame.camera.transform
        // A camera without a usable pose or calibration cannot be trained on (and would stop the camera list being saved).
        let calibration = [intrinsics.columns.0.x, intrinsics.columns.1.y, intrinsics.columns.2.x, intrinsics.columns.2.y]
        guard calibration.allSatisfy({ $0.isFinite && $0 > 0 }), pose.rowMajorValues.joined().allSatisfy(\.isFinite) else {
            print("[capture] a frame came without a valid camera pose or focal length; skipped")
            return nil
        }
        return GaussianFrameSnapshot(
            jpeg: jpeg,
            width: width,
            height: height,
            fx: intrinsics.columns.0.x,
            fy: intrinsics.columns.1.y,
            cx: intrinsics.columns.2.x,
            cy: intrinsics.columns.2.y,
            cameraToWorldRows: pose.rowMajorValues,
            points: seedPoints ?? copySeedPoints(
                depth: frame.smoothedSceneDepth ?? frame.sceneDepth,
                image: pixelBuffer,
                intrinsics: intrinsics,
                cameraToWorld: pose,
                seedDensity: seedDensity,
                minimumConfidence: minimumConfidence,
                maximumDepth: maximumDepth
            ),
            timestamp: frame.timestamp,
            exposureDuration: frame.camera.exposureDuration,
            exposureOffset: frame.camera.exposureOffset,
            sharpness: sharpness,
            pixelFormat: GaussianARCaptureCoordinator.fourCharacterCode(CVPixelBufferGetPixelFormatType(pixelBuffer)),
            toneMapped: toneMapped
        )
    }

    private func copySeedPoints(
        depth: ARDepthData?,
        image: CVPixelBuffer,
        intrinsics: simd_float3x3,
        cameraToWorld: simd_float4x4,
        seedDensity: Float,
        minimumConfidence: Int,
        maximumDepth: Float
    ) -> [GaussianSeedPoint] {
        guard let depth else { return [] }
        let map = depth.depthMap
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else {
            return []
        }

        CVPixelBufferLockBaseAddress(map, .readOnly)
        CVPixelBufferLockBaseAddress(image, .readOnly)
        if let confidence = depth.confidenceMap {
            CVPixelBufferLockBaseAddress(confidence, .readOnly)
        }
        defer {
            if let confidence = depth.confidenceMap {
                CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
            }
            CVPixelBufferUnlockBaseAddress(image, .readOnly)
            CVPixelBufferUnlockBaseAddress(map, .readOnly)
        }

        guard let depthBase = CVPixelBufferGetBaseAddress(map)?.assumingMemoryBound(to: Float32.self) else {
            return []
        }

        let depthWidth = CVPixelBufferGetWidth(map)
        let depthHeight = CVPixelBufferGetHeight(map)
        let depthStride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float32>.stride
        let imageWidth = CVPixelBufferGetWidth(image)
        let imageHeight = CVPixelBufferGetHeight(image)
        let densityScale = Double(min(max(seedDensity, 10), 500)) / 100
        let targetSamples = min(max(Int(900 * densityScale), 225), 4_500)
        let step = max(2, Int(sqrt(Double(depthWidth * depthHeight) / Double(targetSamples))))
        let confidenceThreshold = UInt8(min(max(minimumConfidence, 0), 2))
        let depthLimit = min(max(maximumDepth, 0.5), 10)

        let confidenceBase = depth.confidenceMap.flatMap {
            CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self)
        }
        let confidenceStride = depth.confidenceMap.map(CVPixelBufferGetBytesPerRow) ?? 0

        let luma = LumaPlane(image)

        var result: [GaussianSeedPoint] = []
        result.reserveCapacity(targetSamples)
        for y in stride(from: step / 2, to: depthHeight, by: step) {
            for x in stride(from: step / 2, to: depthWidth, by: step) {
                if let confidenceBase,
                   confidenceBase[y * confidenceStride + x] < confidenceThreshold {
                    continue
                }
                let z = depthBase[y * depthStride + x]
                guard z.isFinite, z >= 0.18, z <= depthLimit else { continue }

                let imageX = Float(x) * Float(imageWidth) / Float(depthWidth)
                let imageY = Float(y) * Float(imageHeight) / Float(depthHeight)
                let cameraX = (imageX - intrinsics.columns.2.x) * z / intrinsics.columns.0.x
                let cameraY = (intrinsics.columns.2.y - imageY) * z / intrinsics.columns.1.y
                let world = cameraToWorld * SIMD4<Float>(cameraX, cameraY, -z, 1)

                let sampleX = min(max(Int(imageX), 0), imageWidth - 1)
                let sampleY = min(max(Int(imageY), 0), imageHeight - 1)
                let luminance = luma.map { UInt8($0.value(x: sampleX, y: sampleY)) } ?? 160
                result.append(
                    GaussianSeedPoint(
                        position: SIMD3<Float>(world.x, world.y, world.z),
                        luminance: luminance
                    )
                )
            }
        }
        return result
    }
}

private nonisolated final class GaussianCaptureWriter: @unchecked Sendable {
    nonisolated static let minimumFrames = 5

    private struct Voxel: Hashable {
        let x: Int
        let y: Int
        let z: Int

        init(_ point: SIMD3<Float>, density: Float) {
            let scale = min(max(density, 10), 500)
            x = Int(floor(point.x * scale))
            y = Int(floor(point.y * scale))
            z = Int(floor(point.z * scale))
        }
    }

    private struct DatasetFrame: Encodable {
        let filePath: String
        let transformMatrix: [[Float]]
        /// This frame's own camera calibration: the focal length moves a little as the camera refocuses.
        let flX: Float
        let flY: Float
        let cx: Float
        let cy: Float
        let width: Int
        let height: Int
        let timestamp: Double
        /// Left out when ARKit gives no number for it (as it can for high-resolution photos): JSON has no NaN, and one
        /// would stop the whole camera list being saved ("isn't in the correct format").
        let exposureDuration: Double?
        let exposureOffset: Float?
        let sharpness: Float?

        enum CodingKeys: String, CodingKey {
            case filePath = "file_path"
            case transformMatrix = "transform_matrix"
            case flX = "fl_x"
            case flY = "fl_y"
            case cx
            case cy
            case width = "w"
            case height = "h"
            case timestamp
            case exposureDuration = "exposure_duration"
            case exposureOffset = "exposure_offset"
            case sharpness
        }
    }

    private struct DatasetManifest: Encodable {
        let cameraModel = "OPENCV"
        let flX: Float
        let flY: Float
        let cx: Float
        let cy: Float
        let width: Int
        let height: Int
        let plyFilePath: String?
        /// Pixel format of the camera frames and whether HDR frames were tone-mapped into the photos.
        let pixelFormat: String
        let hdrToneMapped: Bool
        let frames: [DatasetFrame]

        enum CodingKeys: String, CodingKey {
            case cameraModel = "camera_model"
            case flX = "fl_x"
            case flY = "fl_y"
            case cx
            case cy
            case width = "w"
            case height = "h"
            case plyFilePath = "ply_file_path"
            case pixelFormat = "pixel_format"
            case hdrToneMapped = "hdr_tone_mapped"
            case frames
        }
    }

    private let rootURL: URL
    private let imagesURL: URL
    private let seedDensity: Float
    private var frames: [DatasetFrame] = []
    private var seedPoints: [Voxel: GaussianSeedPoint] = [:]
    private var reference: GaussianFrameSnapshot?
    private var reportedMissingValues = false

    init(seedDensity: Float) throws {
        self.seedDensity = min(max(seedDensity, 10), 500)
        rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("Voxelio-Gaussian-\(UUID().uuidString)", isDirectory: true)
        imagesURL = rootURL.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imagesURL, withIntermediateDirectories: true)
    }

    func append(_ snapshot: GaussianFrameSnapshot, index: Int) throws {
        let name = String(format: "frame_%05d.jpg", index)
        try snapshot.jpeg.write(to: imagesURL.appendingPathComponent(name), options: .atomic)
        if reference == nil {
            reference = snapshot
            try snapshot.jpeg.write(to: rootURL.appendingPathComponent("thumbnail.jpg"), options: .atomic)
        }
        frames.append(
            DatasetFrame(
                filePath: "images/\(name)",
                transformMatrix: snapshot.cameraToWorldRows,
                flX: snapshot.fx, flY: snapshot.fy, cx: snapshot.cx, cy: snapshot.cy,
                width: snapshot.width, height: snapshot.height,
                timestamp: snapshot.timestamp,
                exposureDuration: snapshot.exposureDuration.isFinite ? snapshot.exposureDuration : nil,
                exposureOffset: snapshot.exposureOffset.isFinite ? snapshot.exposureOffset : nil,
                sharpness: snapshot.sharpness.isFinite ? snapshot.sharpness : nil
            )
        )
        if !reportedMissingValues, !(snapshot.exposureDuration.isFinite && snapshot.exposureOffset.isFinite && snapshot.sharpness.isFinite) {
            reportedMissingValues = true
            print("[capture] \(name): no number for exposure duration \(snapshot.exposureDuration), exposure offset \(snapshot.exposureOffset) or sharpness \(snapshot.sharpness); left out of the camera list")
        }
        for point in snapshot.points {
            seedPoints[Voxel(point.position, density: seedDensity)] = point
        }
    }

    func finish() throws -> LocalGaussianCapturePackage {
        guard frames.count >= Self.minimumFrames else {
            throw GaussianCaptureError.notEnoughFrames(captured: frames.count)
        }
        guard let reference else { throw GaussianCaptureError.emptyCapture }

        let hasSeeds = !seedPoints.isEmpty
        if hasSeeds {
            try writePLY(Array(seedPoints.values), to: rootURL.appendingPathComponent("points3D.ply"))
        }
        let manifest = DatasetManifest(
            flX: reference.fx,
            flY: reference.fy,
            cx: reference.cx,
            cy: reference.cy,
            width: reference.width,
            height: reference.height,
            plyFilePath: hasSeeds ? "points3D.ply" : nil,
            pixelFormat: reference.pixelFormat,
            hdrToneMapped: reference.toneMapped,
            frames: frames
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: rootURL.appendingPathComponent("transforms.json"),
            options: .atomic
        )
        return LocalGaussianCapturePackage(
            temporaryURL: rootURL,
            sampleCount: frames.count,
            seedPointCount: seedPoints.count
        )
    }

    func discard() {
        try? FileManager.default.removeItem(at: rootURL)
    }

    private func writePLY(_ points: [GaussianSeedPoint], to url: URL) throws {
        var text = "ply\nformat ascii 1.0\nelement vertex \(points.count)\n"
        text += "property float x\nproperty float y\nproperty float z\n"
        text += "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        text.reserveCapacity(text.count + points.count * 52)
        for point in points {
            let p = point.position
            let c = point.luminance
            text += "\(p.x) \(p.y) \(p.z) \(c) \(c) \(c)\n"
        }
        guard let data = text.data(using: .utf8) else { throw GaussianCaptureError.cannotEncodePointCloud }
        try data.write(to: url, options: .atomic)
    }
}

private nonisolated enum GaussianCaptureError: LocalizedError {
    case emptyCapture
    case notEnoughFrames(captured: Int)
    case cannotEncodePointCloud

    var errorDescription: String? {
        switch self {
        case .emptyCapture:
            return "No camera views were saved."
        case .notEnoughFrames(let captured):
            return "Capture at least \(GaussianCaptureWriter.minimumFrames) views before finishing. \(captured) captured."
        case .cannotEncodePointCloud:
            return "The LiDAR seed cloud could not be encoded."
        }
    }
}

private extension ARCamera.TrackingState {
    nonisolated var isCaptureReady: Bool {
        if case .normal = self { return true }
        return false
    }
}

private extension simd_float4x4 {
    nonisolated var rowMajorValues: [[Float]] {
        (0..<4).map { row in
            (0..<4).map { column in self[column][row] }
        }
    }
}
