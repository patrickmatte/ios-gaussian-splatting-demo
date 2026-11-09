import Foundation
import simd

/// Finds the subject of an orbit capture from the camera poses alone: the point closest to all the
/// cameras' optical axes. Object mode concentrates splats inside a sphere around it.
nonisolated enum SubjectRegion {
    struct Region: Sendable {
        var center: SIMD3<Float>
        var radius: Float
        /// Median distance from the cameras to the centre.
        var cameraDistance: Float
        /// Median distance by which an optical axis misses the centre. Small means a real orbit.
        var rayMiss: Float
    }

    /// The sphere radius, as a share of the cameras' median distance to the subject. Measured on a bowl
    /// captured from 0.35 to 0.72 m: the dense core of the starting points lay within about 0.2 m,
    /// against 0.26 m from this share.
    static let radiusShare: Float = 0.5

    /// Nil when the cameras do not look at one place, so a room or landscape keeps normal training.
    static func estimate(transformsURL: URL) -> Region? {
        guard let data = try? Data(contentsOf: transformsURL),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let frames = root["frames"] as? [[String: Any]] else { return nil }

        var centers: [SIMD3<Double>] = []
        var axes: [SIMD3<Double>] = []
        for frame in frames {
            guard let m = frame["transform_matrix"] as? [[Double]], m.count >= 3, m[0].count >= 4 else { continue }
            centers.append(SIMD3(m[0][3], m[1][3], m[2][3]))
            // OpenGL camera: looks down its own -Z axis.
            let forward = -SIMD3(m[0][2], m[1][2], m[2][2])
            let length = simd_length(forward)
            guard length > 1e-9 else { centers.removeLast(); continue }
            axes.append(forward / length)
        }
        guard centers.count >= 8 else { return nil }

        // Least squares: minimise the summed squared distance from p to every optical axis.
        var a = simd_double3x3(0)
        var b = SIMD3<Double>(0, 0, 0)
        for (c, d) in zip(centers, axes) {
            let projector = matrix_identity_double3x3 - simd_double3x3(rows: [d.x * d, d.y * d, d.z * d])
            a += projector
            b += projector * c
        }
        guard abs(a.determinant) > 1e-9 else { return nil }
        let p = a.inverse * b

        var distances: [Double] = []
        var misses: [Double] = []
        for (c, d) in zip(centers, axes) {
            distances.append(simd_length(p - c))
            let toPoint = p - c
            misses.append(simd_length(toPoint - simd_dot(toPoint, d) * d))
        }
        let distance = median(distances), miss = median(misses)
        guard distance > 1e-6, miss < 0.5 * distance else { return nil }

        return Region(center: SIMD3<Float>(p), radius: Float(distance) * radiusShare,
                      cameraDistance: Float(distance), rayMiss: Float(miss))
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }
}
