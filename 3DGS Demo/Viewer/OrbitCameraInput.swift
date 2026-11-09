#if os(iOS)
import SwiftUI
import UIKit

/// Touch and trackpad input for `OrbitCameraController` — the iOS counterpart to the
/// AppKit `OrbitCameraInput` in `OrbitCameraInput.swift`, sharing its name so
/// `ContentView` needs no platform branch at the call site.
///
/// A `UIViewRepresentable` rather than SwiftUI gestures for much the same reason as on
/// macOS: the mapping depends on how many fingers are down, and on receiving real scroll
/// events from a trackpad or mouse, neither of which `DragGesture` reports.
struct OrbitCameraInput: UIViewRepresentable {
    let controller: any OrbitInputTarget

    func makeUIView(context: Context) -> OrbitInputView {
        let view = OrbitInputView()
        view.controller = controller
        return view
    }

    func updateUIView(_ uiView: OrbitInputView, context: Context) {
        uiView.controller = controller
    }
}

/// One finger rotates, two fingers pan, pinch zooms. Deliberately close to the macOS
/// mapping: left-drag rotates there too, and the middle-drag dolly has no touch
/// equivalent because pinch covers it.
final class OrbitInputView: UIView {

    /// Where the input goes, which changes with the camera looked through. Each gets the view's size as
    /// it takes over, not only at layout: a target that missed the last layout would otherwise turn by
    /// pixels over a height of 1 point — many turns for the smallest drag.
    var controller: (any OrbitInputTarget)? {
        didSet {
            if bounds.height > 0 { controller?.viewSize = bounds.size }
        }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        // Off by default on UIView, and without it the two-finger recognizers never
        // see their second touch.
        isMultipleTouchEnabled = true
        addRecognizers()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("OrbitInputView is created in code only") }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        controller?.viewSize = bounds.size
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        controller?.viewSize = bounds.size
    }

    // MARK: Recognizers

    private func addRecognizers() {
        let rotate = UIPanGestureRecognizer(target: self, action: #selector(handleRotate))
        rotate.minimumNumberOfTouches = 1
        rotate.maximumNumberOfTouches = 1
        addGestureRecognizer(rotate)

        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan))
        pan.minimumNumberOfTouches = 2
        pan.maximumNumberOfTouches = 2
        pan.delegate = self
        addGestureRecognizer(pan)

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch))
        pinch.delegate = self
        addGestureRecognizer(pinch)

        // Trackpad and mouse wheel. `maximumNumberOfTouches = 0` is the documented way
        // to build a recognizer that fires only for indirect scroll events, leaving the
        // touch gestures above untouched.
        let scroll = UIPanGestureRecognizer(target: self, action: #selector(handleScroll))
        scroll.allowedScrollTypesMask = .all
        scroll.maximumNumberOfTouches = 0
        addGestureRecognizer(scroll)
    }

    /// Per-callback delta. UIKit's Y axis already points down, which is the DOM
    /// convention `OrbitCameraController` expects, so — unlike the AppKit view — nothing
    /// is inverted here.
    private func consumeTranslation(_ recognizer: UIPanGestureRecognizer) -> CGSize {
        let translation = recognizer.translation(in: self)
        recognizer.setTranslation(.zero, in: self)
        return CGSize(width: translation.x, height: translation.y)
    }

    @objc private func handleRotate(_ recognizer: UIPanGestureRecognizer) {
        guard let controller else { return }
        switch recognizer.state {
        case .began:
            // Discard the translation accumulated before the gesture was recognized,
            // which would otherwise land as one jump.
            recognizer.setTranslation(.zero, in: self)
        case .changed:
            controller.rotate(byPixels: consumeTranslation(recognizer))
        case .ended, .cancelled, .failed:
            controller.endRotate()
        default:
            break
        }
    }

    @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
        guard let controller else { return }
        switch recognizer.state {
        case .began:
            recognizer.setTranslation(.zero, in: self)
        case .changed:
            controller.pan(byPixels: consumeTranslation(recognizer))
        default:
            break
        }
    }

    @objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
        guard let controller, recognizer.state == .changed else {
            recognizer.scale = 1
            return
        }
        // `magnify(by:)` takes NSEvent's fractional per-event delta, so the cumulative
        // scale is converted to one by resetting it each time.
        controller.magnify(by: recognizer.scale - 1)
        recognizer.scale = 1
    }

    @objc private func handleScroll(_ recognizer: UIPanGestureRecognizer) {
        guard let controller else { return }
        switch recognizer.state {
        case .began:
            recognizer.setTranslation(.zero, in: self)
        case .changed:
            // Scroll translation follows the drag-the-content convention — fingers up
            // gives a negative dy — which is already the DOM wheel sign, so no negation
            // here (the AppKit path needs one because NSEvent reports the opposite).
            // Scaled like the macOS precise-delta branch. Negate to invert zoom.
            controller.zoom(wheelDeltaY: consumeTranslation(recognizer).height * 3)
        default:
            break
        }
    }
}

extension OrbitInputView: UIGestureRecognizerDelegate {
    /// Two fingers dragging while pinching reads as a single gesture to the user, so the
    /// two-finger pan and the pinch run together — the way Maps and Photos behave.
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        true
    }
}
#endif
