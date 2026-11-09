// Copyright 2026 Voxelio.
// Licensed under the PolyForm Noncommercial License 1.0.0.

import SwiftUI
import UIKit

struct GaussianAssetDetailView: View {
    let assetID: UUID
    @ObservedObject var library: LocalGaussianLibrary

    @Environment(\.dismiss) private var dismiss
    @StateObject private var processor = LocalGaussianProcessor()
    @State private var showsViewer = false
    @State private var asksToDelete = false
    @State private var asksToRetrain = false
    @State private var retrainQuality = LocalGaussianQuality.defaultSelection
    @State private var recipe = TrainingRecipe.default
    /// Development only: the full set of training options (image size, SH, splat limit, pose choices, extending a result).
    /// Release builds offer just the three recipes.
    @AppStorage("showsAdvancedTrainingOptions") private var showsAdvanced = false

    private var advanced: Bool {
        #if DEBUG
        showsAdvanced
        #else
        false
        #endif
    }

    private var asset: LocalGaussianAsset? {
        library.asset(id: assetID)
    }

    var body: some View {
        Group {
            if let asset {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        preview(asset)
                        captureSummary(asset)

                        if processor.isRunning || !asset.hasSplat || (advanced && !availableQualities(for: asset).isEmpty) {
                            trainingPanel(asset)
                        }

                        if asset.hasSplat {
                            resultActions(asset)
                        }

                        if asset.hasSplat, asset.hasSourceDataset, !processor.isRunning {
                            retrainPanel(asset)
                        }

                        #if DEBUG
                        Toggle("All training options (development)", isOn: $showsAdvanced)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        #endif

                        Button("Delete project", systemImage: "trash", role: .destructive) {
                            asksToDelete = true
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 12)
                    }
                    .padding()
                }
                .navigationTitle(asset.title)
                .navigationBarTitleDisplayMode(.inline)
                .onAppear { processor.synchronize(with: asset) }
                .onChange(of: asset.updatedAt) { _, _ in
                    processor.synchronize(with: asset)
                }
                .fullScreenCover(isPresented: $showsViewer) {
                    GaussianFullScreenViewer(asset: asset)
                }
                .alert("Delete this project?", isPresented: $asksToDelete) {
                    Button("Delete", role: .destructive) {
                        library.remove(asset)
                        dismiss()
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("The source images, checkpoint, and trained splat will be removed from this device.")
                }
            } else {
                ContentUnavailableView(
                    "Project unavailable",
                    systemImage: "questionmark.folder",
                    description: Text("The local project could not be loaded.")
                )
            }
        }
    }

    @ViewBuilder
    private func preview(_ asset: LocalGaussianAsset) -> some View {
        ZStack {
            Color.black

            // Only the captured thumbnail: a splat loaded here costs a lot of GPU time and memory just to sit at the top of
            // the page, and slows a training that is running. The splat is shown by "Open viewer".
            if let image = UIImage(contentsOfFile: asset.thumbnailURL.path) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
            } else if processor.isRunning {
                ProgressView()
                    .tint(.white)
            } else {
                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 44, weight: .light))
                    .foregroundStyle(.white.opacity(0.65))
            }

            if asset.hasSplat, !processor.isRunning {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Label("Open viewer", systemImage: "arrow.up.left.and.arrow.down.right")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .frame(height: 38)
                            .background(.ultraThinMaterial, in: Capsule())
                            .padding(12)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 300)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .contentShape(Rectangle())
        .onTapGesture {
            guard asset.hasSplat, !processor.isRunning else { return }
            showsViewer = true
        }
    }

    private func captureSummary(_ asset: LocalGaussianAsset) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Capture")
                .font(.title3.bold())

            LabeledContent("Views", value: asset.sampleCount.formatted())
            LabeledContent("Seed points", value: asset.seedPointCount.formatted())
            if let splats = asset.splatCount {
                LabeledContent("Splats", value: splats.formatted())
            }
            LabeledContent("Status", value: stageLabel(asset.stage))
        }
        .padding()
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func trainingPanel(_ asset: LocalGaussianAsset) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(asset.hasSplat && advanced ? "Extend training" : "On-device training")
                .font(.title3.bold())

            if processor.isRunning {
                if let status = processor.statusMessage {
                    Label(status, systemImage: "scope")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                if processor.statusMessage != nil {
                    if let fraction = processor.statusFraction {
                        ProgressView(value: min(max(fraction, 0), 1))
                    } else {
                        ProgressView()   // indeterminate: nothing to measure yet
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    ProgressView(
                        value: Double(processor.completedIterations),
                        total: Double(processor.quality.iterations)
                    )
                }

                HStack {
                    Text("Iteration \(processor.completedIterations) / \(processor.quality.iterations)")
                    Spacer()
                    if processor.splatCount > 0 {
                        Text("\(processor.splatCount) splats")
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

                if let note = processor.thermalNote {
                    Label(note, systemImage: "thermometer.high")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }

                Button("Stop training", systemImage: "stop.circle", role: .destructive) {
                    processor.stop()
                }
                .buttonStyle(.bordered)
            } else if !advanced {
                recipeChoice(asset)
                Button {
                    train(asset, restarting: false)
                } label: {
                    Label("Start training", systemImage: "cpu")
                        .frame(maxWidth: .infinity, minHeight: 44)
                }
                .buttonStyle(.borderedProminent)
            } else {
                if asset.hasSplat {
                    Text("Trained with \((asset.trainingOptions ?? .default).summary(totalViews: asset.sampleCount))")
                        .font(.footnote.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else {
                    TrainingOptionsForm(options: $processor.options, totalViews: asset.sampleCount, frameSize: asset.frameSize)
                    SolvedPosesNote(scanFolder: asset.folderURL, options: $processor.options)
                    Divider()
                }

                Picker("Quality", selection: $processor.quality) {
                    ForEach(availableQualities(for: asset)) { quality in
                        Text("\(quality.label) · \(quality.detail)").tag(quality)
                    }
                }
                if !asset.hasSplat {
                    TrainingTimeEstimate(options: processor.options, totalViews: asset.sampleCount, iterations: processor.quality.iterations)
                }

                Button {
                    processor.process(asset, library: library)
                } label: {
                    Label(
                        asset.hasSplat ? "Continue to \(processor.quality.label)" : "Start training",
                        systemImage: "cpu"
                    )
                    .frame(maxWidth: .infinity, minHeight: 44)
                }
                .buttonStyle(.borderedProminent)
                .disabled(availableQualities(for: asset).isEmpty)
            }

            if case .failed(let message) = processor.phase {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            Text("Training runs locally with msplat. Keep the app open until the selected target completes.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    /// The only choice in release builds: how long to train, with the same recipe for every capture.
    @ViewBuilder
    private func recipeChoice(_ asset: LocalGaussianAsset) -> some View {
        Picker("Training", selection: $recipe) {
            ForEach(TrainingRecipe.allCases) { Text($0.title).tag($0) }
        }
        .pickerStyle(.segmented)
        let options = recipe.options(frameSize: asset.frameSize)
        Text(recipe.detail)
            .font(.caption)
            .foregroundStyle(.secondary)
        TrainingTimeEstimate(options: options, totalViews: asset.sampleCount, iterations: recipe.quality.iterations,
                             posesSolved: PoseSolving.hasSolved(asset.folderURL, linesUpPasses: options.linesUpPasses))
    }

    /// Starts a training with the chosen recipe; `restarting` replaces the current result.
    private func train(_ asset: LocalGaussianAsset, restarting: Bool) {
        processor.options = recipe.options(frameSize: asset.frameSize)
        processor.quality = recipe.quality
        processor.process(asset, library: library, restartingAt: restarting ? recipe.quality : nil)
    }

    /// Throws the current result away and trains again from the saved camera views.
    private func retrainPanel(_ asset: LocalGaussianAsset) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(advanced ? "Retrain from scratch" : "Train again")
                .font(.title3.bold())

            if advanced {
                TrainingOptionsForm(options: $processor.options, totalViews: asset.sampleCount, frameSize: asset.frameSize)
                SolvedPosesNote(scanFolder: asset.folderURL, options: $processor.options)
                Divider()

                Picker("Quality", selection: $retrainQuality) {
                    ForEach(LocalGaussianQuality.allCases) { quality in
                        Text("\(quality.label) · \(quality.detail)").tag(quality)
                    }
                }
                TrainingTimeEstimate(options: processor.options, totalViews: asset.sampleCount, iterations: retrainQuality.iterations)
            } else {
                recipeChoice(asset)
            }

            Button(advanced ? "Retrain from scratch" : "Train again", systemImage: "arrow.triangle.2.circlepath") {
                asksToRetrain = true
            }
            .buttonStyle(.bordered)

            Text("Starts over from the captured photos and replaces the current splat once it finishes.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .confirmationDialog(
            advanced ? "Retrain from scratch to \(retrainQuality.label)?" : "Train again (\(recipe.title))?",
            isPresented: $asksToRetrain,
            titleVisibility: .visible
        ) {
            Button("Retrain", role: .destructive) {
                if advanced {
                    processor.process(asset, library: library, restartingAt: retrainQuality)
                } else {
                    train(asset, restarting: true)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The current result is replaced when the new one finishes. Keep the app open until then.")
        }
    }

    private func resultActions(_ asset: LocalGaussianAsset) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Result")
                .font(.title3.bold())

            Button("Open interactive viewer", systemImage: "view.3d") {
                showsViewer = true
            }
            .buttonStyle(.borderedProminent)

            ShareLink(item: asset.splatURL) {
                Label("Share splat", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)

            if let steps = asset.completedSteps {
                Text("\(asset.splatURL.pathExtension == "spz" ? "SPZ" : "Compressed PLY") · \(steps.formatted()) training iterations" + (asset.splatCount.map { " · \($0.formatted()) splats" } ?? ""))
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func availableQualities(for asset: LocalGaussianAsset) -> [LocalGaussianQuality] {
        LocalGaussianQuality.extensionTargets(after: asset.completedSteps ?? 0)
    }

    private func stageLabel(_ stage: LocalGaussianStage) -> String {
        switch stage {
        case .captured: "Captured"
        case .processing: "Training"
        case .ready: "Ready"
        case .failed: "Needs attention"
        }
    }
}

private struct GaussianFullScreenViewer: View {
    let asset: LocalGaussianAsset
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()
            RealityGaussianViewer(fileURL: asset.splatURL)
                .ignoresSafeArea()

            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .padding()
        }
    }
}
