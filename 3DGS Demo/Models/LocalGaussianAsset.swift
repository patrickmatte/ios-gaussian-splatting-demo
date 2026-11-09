// Copyright 2026 Voxelio.
// Licensed under the PolyForm Noncommercial License 1.0.0.

import Combine
import Foundation

nonisolated enum LocalGaussianStage: String, Codable, Hashable, Sendable {
    case captured
    case processing
    case ready
    case failed

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        switch value {
        case "captured": self = .captured
        case "processing", "training": self = .processing
        case "ready", "trained": self = .ready
        case "failed": self = .failed
        default:
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unknown Gaussian stage: \(value)")
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

nonisolated struct LocalGaussianAsset: Identifiable, Codable, Hashable, Sendable {
    var id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    var sampleCount: Int
    var seedPointCount: Int
    var stage: LocalGaussianStage
    var completedSteps: Int?
    var splatCount: Int?
    var trainingOptions: LocalGaussianTrainingOptions?
    var failure: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case title
        case createdAt
        case updatedAt
        case sampleCount = "keyframeCount"
        case seedPointCount = "pointCount"
        case stage = "status"
        case completedSteps = "trainingIterations"
        case splatCount
        case trainingOptions
        case failure = "errorMessage"
    }

    var folderURL: URL {
        LocalGaussianLibrary.rootURL.appendingPathComponent(
            "\(id.uuidString).\(LocalGaussianLibrary.packageExtension)",
            isDirectory: true
        )
    }

    /// Size of the captured photos, from the camera list (1920x1440 for video frames, about 4032x3024 for
    /// high-resolution photos). Read once per scan.
    var frameSize: (width: Double, height: Double) {
        let key = transformsURL.path as NSString
        if let cached = LocalGaussianAsset.frameSizeCache.object(forKey: key) {
            return (cached.width, cached.height)
        }
        var size = (width: 1920.0, height: 1440.0)
        if let data = try? Data(contentsOf: transformsURL),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let w = root["w"] as? Double, let h = root["h"] as? Double, w > 0, h > 0 {
            size = (w, h)
            LocalGaussianAsset.frameSizeCache.setObject(FrameSizeBox(width: w, height: h), forKey: key)
        }
        return size
    }

    private final class FrameSizeBox {
        let width: Double, height: Double
        init(width: Double, height: Double) { self.width = width; self.height = height }
    }
    nonisolated(unsafe) private static let frameSizeCache = NSCache<NSString, FrameSizeBox>()

    var metadataURL: URL { folderURL.appendingPathComponent("scan.json") }
    var imagesURL: URL { folderURL.appendingPathComponent("images", isDirectory: true) }
    var transformsURL: URL { folderURL.appendingPathComponent("transforms.json") }
    var seedCloudURL: URL { folderURL.appendingPathComponent("points3D.ply") }
    var thumbnailURL: URL { folderURL.appendingPathComponent("thumbnail.jpg") }
    var legacySplatURL: URL { folderURL.appendingPathComponent("trained.spz") }
    var legacyCheckpointURL: URL { folderURL.appendingPathComponent("training.ckpt") }

    /// The trained splat: a compressed PLY (`CompressedPLY`), or an SPZ for scans trained before.
    var splatURL: URL {
        guard let completedSteps else { return legacySplatURL }
        let versionedURL = splatURL(for: completedSteps)
        if FileManager.default.fileExists(atPath: versionedURL.path) { return versionedURL }
        let olderURL = folderURL.appendingPathComponent("trained-\(completedSteps).spz")
        return FileManager.default.fileExists(atPath: olderURL.path) ? olderURL : legacySplatURL
    }

    var checkpointURL: URL {
        guard let completedSteps else { return legacyCheckpointURL }
        let versionedURL = checkpointURL(for: completedSteps)
        return FileManager.default.fileExists(atPath: versionedURL.path) ? versionedURL : legacyCheckpointURL
    }

    func splatURL(for iterations: Int) -> URL {
        folderURL.appendingPathComponent("trained-\(iterations)\(CompressedPLY.fileSuffix)")
    }

    func checkpointURL(for iterations: Int) -> URL {
        folderURL.appendingPathComponent("training-\(iterations).ckpt")
    }

    var hasSplat: Bool {
        FileManager.default.fileExists(atPath: splatURL.path)
    }

    var hasTrainingCheckpoint: Bool {
        FileManager.default.fileExists(atPath: checkpointURL.path)
    }

    var hasSourceDataset: Bool {
        FileManager.default.fileExists(atPath: transformsURL.path) &&
            FileManager.default.fileExists(atPath: imagesURL.path)
    }

    var subtitle: String {
        switch stage {
        case .captured:
            return "Gaussian capture · \(sampleCount) frames"
        case .processing:
            return "Processing on this device"
        case .ready:
            return "3D Gaussian · processed locally"
        case .failed:
            return "Local processing needs attention"
        }
    }
}

nonisolated struct LocalGaussianCapturePackage: Sendable {
    let temporaryURL: URL
    let sampleCount: Int
    let seedPointCount: Int
}

@MainActor
final class LocalGaussianLibrary: ObservableObject {
    @Published private(set) var assets: [LocalGaussianAsset] = []

    nonisolated static let packageExtension = "gaussiansplat"

    nonisolated static var rootURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GaussianSplattingDemo", isDirectory: true)
    }

    private let files = FileManager.default
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        self.encoder = encoder

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder

        refresh()
    }

    func refresh() {
        do {
            try files.createDirectory(at: Self.rootURL, withIntermediateDirectories: true)
            let folders = try files.contentsOfDirectory(
                at: Self.rootURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            assets = folders
                .filter { $0.pathExtension == Self.packageExtension }
                .compactMap(readAsset)
                .map(recoverInterruptedProcessing)
                .sorted { $0.updatedAt > $1.updatedAt }
        } catch {
            assets = []
        }
    }

    @discardableResult
    func importCapture(_ capture: LocalGaussianCapturePackage) throws -> LocalGaussianAsset {
        try files.createDirectory(at: Self.rootURL, withIntermediateDirectories: true)

        let id = UUID()
        let now = Date()
        let destination = Self.rootURL.appendingPathComponent(
            "\(id.uuidString).\(Self.packageExtension)",
            isDirectory: true
        )
        if files.fileExists(atPath: destination.path) {
            try files.removeItem(at: destination)
        }
        try files.moveItem(at: capture.temporaryURL, to: destination)

        let asset = LocalGaussianAsset(
            id: id,
            title: Self.titleFormatter.string(from: now),
            createdAt: now,
            updatedAt: now,
            sampleCount: capture.sampleCount,
            seedPointCount: capture.seedPointCount,
            stage: .captured,
            completedSteps: nil,
            splatCount: nil,
            trainingOptions: nil,
            failure: nil
        )
        try persist(asset)
        assets.insert(asset, at: 0)
        return asset
    }

    func asset(id: UUID) -> LocalGaussianAsset? {
        assets.first { $0.id == id }
    }

    func beginProcessing(id: UUID) {
        mutate(id: id) {
            $0.stage = .processing
            $0.failure = nil
        }
    }

    @discardableResult
    func finishProcessing(id: UUID, steps: Int, splatCount: Int, options: LocalGaussianTrainingOptions) -> Bool {
        mutate(id: id) {
            $0.stage = .ready
            $0.completedSteps = steps
            $0.splatCount = splatCount
            $0.trainingOptions = options
            $0.failure = nil
        }
    }

    /// The user stopped a run: back to what the scan was before it started.
    func cancelProcessing(id: UUID) {
        mutate(id: id) {
            $0.stage = $0.hasSplat ? .ready : .captured
            $0.failure = nil
        }
    }

    func failProcessing(id: UUID, message: String) {
        mutate(id: id) {
            $0.stage = .failed
            $0.failure = message
        }
    }

    func remove(_ asset: LocalGaussianAsset) {
        do {
            if files.fileExists(atPath: asset.folderURL.path) {
                try files.removeItem(at: asset.folderURL)
            }
            assets.removeAll { $0.id == asset.id }
        } catch {
            // Keep the row when deletion fails so the user can retry.
        }
    }

    @discardableResult
    private func mutate(id: UUID, edit: (inout LocalGaussianAsset) -> Void) -> Bool {
        guard let index = assets.firstIndex(where: { $0.id == id }) else { return false }
        var value = assets[index]
        edit(&value)
        value.updatedAt = Date()
        do {
            try persist(value)
            assets[index] = value
            assets.sort { $0.updatedAt > $1.updatedAt }
            return true
        } catch {
            // Metadata on disk remains authoritative if an atomic write fails.
            return false
        }
    }

    private func persist(_ asset: LocalGaussianAsset) throws {
        try files.createDirectory(at: asset.folderURL, withIntermediateDirectories: true)
        try encoder.encode(asset).write(to: asset.metadataURL, options: .atomic)
    }

    private func readAsset(folderURL: URL) -> LocalGaussianAsset? {
        let metadata = folderURL.appendingPathComponent("scan.json")
        guard let data = try? Data(contentsOf: metadata),
              var asset = try? decoder.decode(LocalGaussianAsset.self, from: data)
        else {
            return nil
        }

        // The folder name is the stable identity. This also repairs metadata copied
        // between packages by external file tools.
        if let folderID = UUID(uuidString: folderURL.deletingPathExtension().lastPathComponent) {
            asset.id = folderID
        }
        return asset
    }

    private func recoverInterruptedProcessing(_ asset: LocalGaussianAsset) -> LocalGaussianAsset {
        guard asset.stage == .processing else { return asset }
        var repaired = asset
        repaired.stage = asset.hasSplat ? .ready : .captured
        repaired.failure = nil
        try? persist(repaired)
        return repaired
    }

    private static let titleFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()
}
