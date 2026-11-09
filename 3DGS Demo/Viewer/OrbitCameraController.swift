import Foundation
import RealityKit
import simd

/// A port of three.js `OrbitControls` (examples/jsm/controls/OrbitControls.js) onto a
/// RealityKit camera entity.
///
/// The formulas follow the original: rotation is `2π · pixels / viewHeight` (height for
/// both axes, so aspect ratio does not skew the speed), pan converts pixels to world
/// units through `tan(fov/2)` at the target distance, and zoom is an exponential
/// `0.95 ^ (zoomSpeed · |delta · 0.01|)`. Damping decays the accumulated deltas by
/// `1 - dampingFactor` each frame rather than applying them outright.
///
/// One simplification: three.js rotates the offset into "Y-axis-is-up" space via
/// `_quat`/`_quatInverse` to support arbitrary camera up vectors. RealityKit's world is
/// already Y-up and this viewer never changes that, so those rotations are the identity
/// and are omitted.
///
/// Deliberately not `@Observable`: `update(camera:deltaTime:)` runs every frame, and
/// publishing that would invalidate the SwiftUI hierarchy 60+ times a second.
final class OrbitCameraController: OrbitInputTarget {

    /// Spherical coordinates in three.js's convention: `phi` is measured from +Y,
    /// `theta` is the azimuth about +Y measured from +Z.
    struct Spherical {
        var radius: Float
        var theta: Float
        var phi: Float

        static let epsilon: Float = 0.000001

        init(radius: Float, theta: Float, phi: Float) {
            self.radius = radius
            self.theta = theta
            self.phi = phi
        }

        init(from v: SIMD3<Float>) {
            radius = simd_length(v)
            if radius == 0 {
                theta = 0
                phi = 0
            } else {
                theta = atan2(v.x, v.z)
                phi = acos(max(-1, min(1, v.y / radius)))
            }
        }

        var cartesian: SIMD3<Float> {
            let sinPhiRadius = radius * sin(phi)
            return SIMD3(sinPhiRadius * sin(theta),
                         radius * cos(phi),
                         sinPhiRadius * cos(theta))
        }

        /// Keeps `phi` off the poles, where the look-at basis would be degenerate.
        mutating func makeSafe() {
            phi = max(Self.epsilon, min(.pi - Self.epsilon, phi))
        }
    }

    // MARK: Configuration

    var target = SIMD3<Float>.zero
    var minDistance: Float = 0
    var maxDistance: Float = .infinity
    var minPolarAngle: Float = 0
    var maxPolarAngle: Float = .pi
    var minAzimuthAngle: Float = -.infinity
    var maxAzimuthAngle: Float = .infinity

    var enableRotate = true
    var enablePan = true
    var enableZoom = true

    var enableDamping = true
    /// three.js defaults this to 0.05; 0.5 tracks the pointer far more closely, at the
    /// cost of most of the glide after release.
    var dampingFactor: Float = 0.5

    var rotateSpeed: Float = 1
    var zoomSpeed: Float = 1
    var panSpeed: Float = 1

    /// True pans along the camera's own up vector; false pans parallel to the ground.
    var screenSpacePanning = true

    /// Vertical field of view, needed to convert pan pixels into world units.
    var fieldOfViewDegrees: Float = 60

    /// View size in points. Only the height is used, matching three.js.
    var viewSize = CGSize(width: 1, height: 1)

    // MARK: State

    private var spherical = Spherical(radius: 1, theta: 0, phi: .pi / 2)
    private var sphericalDelta = SIMD2<Float>.zero   // (theta, phi)
    private var panOffset = SIMD3<Float>.zero
    private var scale: Float = 1

    /// Camera basis from the previous update, used by pan exactly as three.js uses
    /// `object.matrix` — i.e. the orientation before this frame's changes.
    private var cameraRight = SIMD3<Float>(1, 0, 0)
    private var cameraUp = SIMD3<Float>(0, 1, 0)
    private var cameraPosition = SIMD3<Float>(0, 0, 1)

    var distance: Float { spherical.radius }

    // MARK: Input

    /// Left-drag. `delta` is in pixels with **down and right positive** (DOM convention).
    /// Nothing to do: damping already carries the motion out after a release.
    func endRotate() {}

    func rotate(byPixels delta: CGSize) {
        guard enableRotate else { return }
        let height = Float(max(viewSize.height, 1))
        sphericalDelta.x -= 2 * .pi * (Float(delta.width) * rotateSpeed) / height
        sphericalDelta.y -= 2 * .pi * (Float(delta.height) * rotateSpeed) / height
    }

    /// Right-drag (or shift-drag). `delta` is in pixels, down and right positive.
    func pan(byPixels delta: CGSize) {
        guard enablePan else { return }
        let height = Float(max(viewSize.height, 1))

        // Half the vertical fov spans centre to top of screen at the target distance.
        var targetDistance = simd_length(cameraPosition - target)
        targetDistance *= tan((fieldOfViewDegrees / 2) * .pi / 180)

        let dx = 2 * (Float(delta.width) * panSpeed) * targetDistance / height
        let dy = 2 * (Float(delta.height) * panSpeed) * targetDistance / height

        panLeft(dx)
        panUp(dy)
    }

    private func panLeft(_ distance: Float) {
        panOffset += cameraRight * -distance
    }

    private func panUp(_ distance: Float) {
        let axis = screenSpacePanning
            ? cameraUp
            : simd_cross(SIMD3<Float>(0, 1, 0), cameraRight)
        panOffset += axis * distance
    }

    /// Scroll wheel. `deltaY` follows the DOM sign convention: negative scrolls up and
    /// zooms in. Callers converting from AppKit must negate `scrollingDeltaY`.
    func zoom(wheelDeltaY deltaY: CGFloat) {
        guard enableZoom, deltaY != 0 else { return }
        let factor = zoomScale(for: Float(deltaY))
        if deltaY < 0 {
            scale *= factor          // dollyIn
        } else {
            scale /= factor          // dollyOut
        }
    }

    /// Middle-drag dolly, driven by vertical pixels.
    func dolly(byPixels deltaY: CGFloat) {
        zoom(wheelDeltaY: deltaY)
    }

    /// Trackpad pinch. `magnification` is NSEvent's fractional delta.
    func magnify(by magnification: CGFloat) {
        guard enableZoom, magnification != 0 else { return }
        scale /= Float(1 + magnification)
    }

    private func zoomScale(for delta: Float) -> Float {
        pow(0.95, zoomSpeed * abs(delta * 0.01))
    }

    // MARK: Pose

    /// Places the camera directly, clearing any in-flight momentum. Used for the initial
    /// framing of a newly loaded cloud and for the Reset button.
    func setPose(camera: Entity,
                 target: SIMD3<Float>,
                 distance: Float,
                 theta: Float = 0,
                 phi: Float = .pi / 2) {
        self.target = target
        sphericalDelta = .zero
        panOffset = .zero
        scale = 1

        var pose = Spherical(radius: distance, theta: theta, phi: phi)
        pose.makeSafe()
        spherical = pose

        apply(position: target + pose.cartesian, to: camera)
    }

    // MARK: Per-frame update

    func update(camera: Entity, deltaTime: TimeInterval) {
        var offset = camera.position(relativeTo: nil) - target
        spherical = Spherical(from: offset)

        if enableDamping {
            spherical.theta += sphericalDelta.x * dampingFactor
            spherical.phi += sphericalDelta.y * dampingFactor
        } else {
            spherical.theta += sphericalDelta.x
            spherical.phi += sphericalDelta.y
        }

        clampAzimuth()
        spherical.phi = max(minPolarAngle, min(maxPolarAngle, spherical.phi))
        spherical.makeSafe()

        if enableDamping {
            target += panOffset * dampingFactor
        } else {
            target += panOffset
        }

        spherical.radius = max(minDistance, min(maxDistance, spherical.radius * scale))

        offset = spherical.cartesian
        apply(position: target + offset, to: camera)

        if enableDamping {
            sphericalDelta *= (1 - dampingFactor)
            panOffset *= (1 - dampingFactor)
        } else {
            sphericalDelta = .zero
            panOffset = .zero
        }
        scale = 1
    }

    private func clampAzimuth() {
        var min = minAzimuthAngle
        var max = maxAzimuthAngle
        guard min.isFinite, max.isFinite else { return }

        let twoPI = 2 * Float.pi
        if min < -.pi { min += twoPI } else if min > .pi { min -= twoPI }
        if max < -.pi { max += twoPI } else if max > .pi { max -= twoPI }

        if min <= max {
            spherical.theta = Swift.max(min, Swift.min(max, spherical.theta))
        } else {
            spherical.theta = spherical.theta > (min + max) / 2
                ? Swift.max(min, spherical.theta)
                : Swift.min(max, spherical.theta)
        }
    }

    /// Positions the camera and aims it at the target, caching the resulting basis for
    /// the next pan.
    private func apply(position: SIMD3<Float>, to camera: Entity) {
        let forward = simd_normalize(target - position)
        let zAxis = -forward                                    // camera looks down -Z

        var xAxis = simd_cross(SIMD3<Float>(0, 1, 0), zAxis)
        let xLength = simd_length(xAxis)
        xAxis = xLength < 1e-6 ? SIMD3<Float>(1, 0, 0) : xAxis / xLength

        let yAxis = simd_cross(zAxis, xAxis)

        camera.setPosition(position, relativeTo: nil)
        camera.setOrientation(simd_quatf(simd_float3x3(xAxis, yAxis, zAxis)), relativeTo: nil)

        cameraPosition = position
        cameraRight = xAxis
        cameraUp = yAxis
    }
}

/// What the orbit input view drives. Extracted from the GaussianTest project, where a second
/// controller (a rig's look) shares the same input view.
@MainActor
protocol OrbitInputTarget: AnyObject {
    /// Set by the input view on layout. The orbit camera converts pixels to radians through it.
    var viewSize: CGSize { get set }
    func rotate(byPixels delta: CGSize)
    func pan(byPixels delta: CGSize)
    func dolly(byPixels deltaY: CGFloat)
    func zoom(wheelDeltaY deltaY: CGFloat)
    func magnify(by magnification: CGFloat)
    /// Called when a rotate drag ends. The orbit camera has nothing to do here.
    func endRotate()
}
