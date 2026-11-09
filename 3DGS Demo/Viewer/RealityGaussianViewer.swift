import Metal
import Observation
import RealityKit
import SwiftUI
import simd

/// A splat viewer built on RealityKit 27's `GaussianSplatResource` and `GaussianSplatComponent`,
/// driven by the three.js-style `OrbitCameraController`: one finger orbits, two fingers pan the
/// look-at target, pinch zooms.
@available(iOS 27.0, *)
struct RealityGaussianViewer: View {
    let fileURL: URL
    @State private var model = RealityGaussianViewerModel()

    var body: some View {
        ZStack {
            RealityView { content in
                content.camera = .virtual
                content.add(model.root)
                _ = content.subscribe(to: SceneEvents.Update.self) { event in
                    model.update(deltaTime: event.deltaTime)
                }
            }
            .background(Color.black)

            OrbitCameraInput(controller: model.orbit)

            overlay
        }
        .task(id: fileURL) { await model.load(fileURL) }
    }

    @ViewBuilder
    private var overlay: some View {
        switch model.state {
        case .loading:
            ProgressView("Loading splat…")
                .padding()
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .padding()
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        case .ready(let count):
            VStack {
                Spacer()
                HStack {
                    Text("\(count.formatted()) splats")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                    Spacer()
                    Picker("Colour space", selection: $model.colorSpace) {
                        ForEach(RealityGaussianViewerModel.ColorSpaceChoice.allCases) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.menu)
                    .tint(.white)
                    Button("Reset view", systemImage: "arrow.counterclockwise") {
                        model.resetCamera()
                    }
                    .buttonStyle(.bordered)
                    .tint(.white)
                }
                .padding()
            }
        }
    }
}

@available(iOS 27.0, *)
@MainActor
@Observable
final class RealityGaussianViewerModel {
    enum State: Equatable {
        case loading
        case ready(splats: Int)
        case failed(String)
    }

    private(set) var state = State.loading

    /// How RealityKit is told the splat colours are encoded. sRGB is the 3DGS convention; the
    /// others exist to tell a colour-space problem apart from a training one.
    enum ColorSpaceChoice: String, CaseIterable, Identifiable {
        case sRGB = "sRGB"
        case linear = "Linear sRGB"
        case extendedLinear = "Extended linear sRGB"
        case displayP3 = "Display P3"

        var id: String { rawValue }

        var cgColorSpace: CGColorSpace {
            let name: CFString = switch self {
            case .sRGB: CGColorSpace.sRGB
            case .linear: CGColorSpace.linearSRGB
            case .extendedLinear: CGColorSpace.extendedLinearSRGB
            case .displayP3: CGColorSpace.displayP3
            }
            return CGColorSpace(name: name) ?? CGColorSpaceCreateDeviceRGB()
        }
    }

    var colorSpace = ColorSpaceChoice.sRGB {
        didSet { resource?.colorSpace = colorSpace.cgColorSpace }
    }
    @ObservationIgnored private var resource: GaussianSplatResource?

    /// Not observed: updated every frame.
    @ObservationIgnored let root = Entity()
    @ObservationIgnored let orbit = OrbitCameraController()
    @ObservationIgnored private let camera = PerspectiveCamera()
    @ObservationIgnored private let splatEntity = Entity()

    @ObservationIgnored private var home = (target: SIMD3<Float>.zero, distance: Float(2))
    @ObservationIgnored private var loadedURL: URL?

    private let fieldOfView: Float = 60
    private let homeTheta: Float = 0.35
    private let homePhi: Float = 1.35

    init() {
        root.addChild(camera)
        root.addChild(splatEntity)
        orbit.fieldOfViewDegrees = fieldOfView
        orbit.dampingFactor = 0.2
    }

    func update(deltaTime: TimeInterval) {
        guard case .ready = state else { return }
        orbit.update(camera: camera, deltaTime: deltaTime)
    }

    func resetCamera() {
        orbit.setPose(camera: camera, target: home.target, distance: home.distance,
                      theta: homeTheta, phi: homePhi)
    }

    func load(_ url: URL) async {
        guard loadedURL != url else { return }
        loadedURL = url
        state = .loading
        do {
            let data = try await Task.detached(priority: .userInitiated) {
                try SplatArrays.read(from: url)
            }.value
            let resource = try Self.makeResource(from: data)
            resource.colorSpace = colorSpace.cgColorSpace
            self.resource = resource
            splatEntity.components.set(GaussianSplatComponent(resource))
            frame(data)
            state = .ready(splats: data.count)
        } catch {
            loadedURL = nil
            state = .failed(error.localizedDescription)
        }
    }

    /// Aims the camera at the middle of the cloud, far enough back to see all of it, and sets the
    /// zoom limits from its size.
    private func frame(_ data: SplatArrays) {
        let radius = max(data.radius, 0.01)
        let distance = radius / tan(fieldOfView * .pi / 360) * 1.1

        camera.components.set(PerspectiveCameraComponent(
            near: radius * 0.001,
            far: distance + radius * 20,
            fieldOfViewInDegrees: fieldOfView
        ))
        orbit.minDistance = radius * 0.002
        orbit.maxDistance = distance * 6
        home = (data.center, distance)
        resetCamera()
    }

    // MARK: Resource

    private enum BuildError: LocalizedError {
        case bufferFailed(String)

        var errorDescription: String? {
            switch self {
            case .bufferFailed(let name): "Could not allocate the \(name) buffer."
            }
        }
    }

    private static func makeResource(from data: SplatArrays) throws -> GaussianSplatResource {
        func buffer<T>(_ values: [T], _ name: String) throws -> LowLevelBuffer {
            let bytes = values.count * MemoryLayout<T>.stride
            // LowLevelBuffer capacities must be a multiple of 16 bytes.
            let capacity = max(16, (bytes + 15) / 16 * 16)
            do {
                let buffer = try LowLevelBuffer(descriptor: .init(capacity: capacity))
                buffer.withUnsafeMutableBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    values.withUnsafeBytes { base.copyMemory(from: $0.baseAddress!, byteCount: bytes) }
                }
                buffer.bytesUsed = capacity
                return buffer
            } catch {
                throw BuildError.bufferFailed(name)
            }
        }

        func descriptor(_ buffer: LowLevelBuffer, _ format: MTLAttributeFormat,
                        _ stride: Int) -> GaussianSplatResource.BufferDescriptor {
            .init(buffer: buffer, format: format, stride: stride, offset: 0)
        }

        let coefficients = (data.degree + 1) * (data.degree + 1)
        let degree: GaussianSplatResource.SphericalHarmonicDegree = switch data.degree {
        case 1: .first
        case 2: .second
        case 3: .third
        default: .zero
        }

        let buffers = try GaussianSplatResource.BufferResource(
            count: data.count,
            position: descriptor(try buffer(data.positions, "position"),
                                 .float3, MemoryLayout<SIMD3<Float>>.stride),
            scale: descriptor(try buffer(data.scales, "scale"),
                              .float3, MemoryLayout<SIMD3<Float>>.stride),
            // RealityKit reads the rotation with the real part first.
            rotation: descriptor(try buffer(data.rotations, "rotation"),
                                 .float4, MemoryLayout<SIMD4<Float>>.stride),
            opacity: descriptor(try buffer(data.opacities, "opacity"),
                                .float, MemoryLayout<Float>.stride),
            sphericalHarmonics: (descriptor(try buffer(data.sh, "spherical harmonics"),
                                            .float3, coefficients * 3 * MemoryLayout<Float>.stride),
                                 degree)
        )

        let resource = GaussianSplatResource(buffers)
        // Scales and opacity were expanded to linear values while reading.
        resource.scaleActivation = .identity
        resource.opacityActivation = .identity
        return resource
    }
}

/// A decoded splat file laid out the way RealityKit's buffers want it. Not tied to the main actor, so decoding runs
/// on the loading task's thread.
nonisolated private struct SplatArrays: Sendable {
    var count: Int
    var positions: [SIMD3<Float>]
    var scales: [SIMD3<Float>]
    /// Real part first.
    var rotations: [SIMD4<Float>]
    var opacities: [Float]
    /// `count × (degree + 1)²` RGB triplets, each splat's coefficients one after another.
    var sh: [Float]
    var degree: Int
    var center: SIMD3<Float>
    var radius: Float

    static func read(from url: URL) throws -> SplatArrays {
        let splats: CompressedPLY.Splats
        if CompressedPLY.isCompressed(url) {
            splats = try CompressedPLY.read(from: url)
        } else if url.pathExtension.lowercased() == "spz" {
            // Scans trained before the compressed PLY.
            splats = try LegacySPZ.read(from: url)
        } else {
            throw ReadError.unsupported
        }
        guard splats.count > 0 else { throw ReadError.empty }
        let (center, radius) = extent(of: splats.positions)
        return SplatArrays(count: splats.count, positions: splats.positions, scales: splats.scales,
                           rotations: splats.rotations, opacities: splats.opacities, sh: splats.sh,
                           degree: splats.degree, center: center, radius: radius)
    }

    /// The middle of the cloud and the RMS distance from it, which is robust to the stray far-away splats that a
    /// bounding box would be stretched by.
    private static func extent(of positions: [SIMD3<Float>]) -> (center: SIMD3<Float>, radius: Float) {
        var sum = SIMD3<Double>.zero
        for p in positions { sum += SIMD3<Double>(p) }
        let center = SIMD3<Float>(sum / Double(positions.count))
        var squares: Double = 0
        for p in positions { squares += Double(simd_length_squared(p - center)) }
        return (center, Float((squares / Double(positions.count)).squareRoot()))
    }

    private enum ReadError: LocalizedError {
        case empty, unsupported
        var errorDescription: String? {
            switch self {
            case .empty: "The splat file contains no points."
            case .unsupported: "The splat file is neither a compressed PLY nor an SPZ."
            }
        }
    }
}
