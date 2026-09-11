# Gesture Mac: researched build plan

## Product goal

Create a local-first macOS input utility that lets a user point, click, drag, scroll, and trigger a small number of shortcuts using one hand and a normal webcam. It should complement a mouse/trackpad, not trap the user in a gesture-only mode.

The hard problem is not landmark detection. It is avoiding accidental actions while keeping latency low enough that the pointer feels attached to the hand. This prototype therefore validates the gesture/state model before investing in a signed Swift menu-bar application.

## Evidence behind the architecture

- MediaPipe Hand Landmarker returns handedness plus 21 normalized image landmarks and 21 world-space landmarks. Its video/live modes reuse tracking between frames instead of running palm detection every frame, reducing latency. The live API may drop frames when busy, which is desirable later for low-latency control rather than building a queue. [Google's Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- The official full hand model combines palm detection and hand landmark estimation and was trained on roughly 30,000 real-world images plus synthetic hands. Google's published Pixel 6 benchmark is 17.12 ms on CPU; Mac performance still needs measurement on the target machine. [Hand Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index)
- macOS requires explicit Camera permission for camera capture and explicit Accessibility permission for an app that controls the Mac. These are separate user approvals. [Apple camera permissions](https://support.apple.com/guide/mac-help/control-access-to-your-camera-mchlf6d108da/mac), [Apple accessibility permissions](https://support.apple.com/guide/mac-help/allow-accessibility-apps-to-access-your-mac-mh43185/mac)
- Native Quartz events can represent mouse motion, button events, and pixel scroll-wheel events, and can be posted into the system event stream. That makes Quartz a better macOS backend than browser automation or app-specific integrations; Chrome and other apps receive ordinary input. [Apple `CGEvent` documentation](https://developer.apple.com/documentation/coregraphics/cgevent)
- Apple exposes `AXIsProcessTrustedWithOptions` so the app can check Accessibility trust and optionally show the system prompt. [Apple accessibility API](https://developer.apple.com/documentation/applicationservices/1459186-axisprocesstrustedwithoptions)

## Interaction model

The design uses **pose as a clutch** and **motion as a value**:

1. A distinct pose selects a mode: point, pinch, two-finger scroll, or neutral.
2. Motion only has an effect while that deliberate pose remains stable.
3. Pose changes must persist briefly before a state transition.
4. Discrete actions fire once per transition, never once per video frame.
5. Losing the hand releases any held mouse button and stops output.

This prevents the most dangerous failure mode: a classifier flickering between labels and generating repeated clicks or endless scroll events.

### MVP gesture vocabulary

| Pose | Mapping | Why |
|---|---|---|
| Index only | Absolute pointer | Easy to learn; active-area mapping reaches the whole screen |
| Thumb–index pinch | Click / hold-to-drag | Continuous pinch distance gives hysteresis; hold differentiates drag |
| Index + middle | Hand-motion scroll | Deliberate clutch prevents scrolling while pointing |
| Thumb–middle pinch | Right click | Related mnemonic, spatially separable from primary pinch |
| Fist hold | Pause/resume | Requires dwell, so it is hard to trigger by accident |
| Open palm | No action | A discoverable safe/rest pose |

Later gestures should be user-assignable rather than hard-coded. Candidates: three-finger horizontal swipe for browser back/forward or desktop switching, palm push for Mission Control, and media volume/play controls. Each must pass false-positive testing before becoming a default.

## Architecture

```text
Camera → latest-frame capture → MediaPipe landmarks
       → normalized geometry → gesture state machine
       → smoothing / hysteresis / dwell / cooldown
       → semantic actions → Quartz macOS event backend
                           → on-screen preview + telemetry
```

- `gesture_engine.py`: portable, I/O-free classifier and state machine. This is the prototype's useful core.
- `controller.py`: macOS Quartz adapter. It knows about displays and system events, not gestures.
- `app.py`: disposable camera/preview shell and CLI.

No frames or landmarks leave the machine, and nothing is recorded.

Implementation note: MediaPipe 1.0.1 was tested first but reproducibly aborted in `DrishtiMetalHelper` on the target macOS 26.6 / Apple M2 host, including with the CPU delegate forced. The same pinned model and blank-frame inference pass on 0.10.35, so the prototype pins 0.10.35 and exposes `gesture-mac --check` as a process-level compatibility gate. Revisit this pin when upstream fixes the regression.

## Delivery phases

### Phase 0 — feasibility prototype (implemented now)

- Camera preview with skeleton, active area, current gesture, and FPS.
- Main-display cursor movement with adaptive One Euro smoothing.
- Pinch click, hold-to-drag, two-finger 2D scroll, right click.
- Gesture stability, pinch hysteresis, cooldowns, no-hand drag release.
- Preview-only default, explicit `--control`, fist/keyboard pause, clean shutdown.
- Camera-free model/runtime compatibility check.

Exit criteria: 10 minutes of mixed Chrome/Finder use with no stuck mouse button, median end-to-end response subjectively below 100 ms, at least 95% intended click recognition, and fewer than one unintended discrete action per 10 minutes.

### Phase 1 — measure and calibrate

- Add a 30-second onboarding flow that learns neutral hand scale, pinch-open/closed distances, dominant hand, comfortable active area, and scroll direction.
- Record only anonymous counters/timings in memory: frame latency, classification changes, missed/extra actions. Offer an explicit export; never save camera frames.
- Move capture and inference off the UI thread; switch to MediaPipe live-stream/latest-frame semantics if profiling shows blocking.
- Add multi-monitor selection and per-display calibration.

Exit criteria: p95 processing latency below 50 ms on target Apple-silicon Macs; pointer target acquisition within 1.5× trackpad time for medium targets; false clicks below 0.1/minute.

### Phase 2 — native macOS shell

- Swift/SwiftUI menu-bar app with status icon, live enable/disable, camera picker, launch at login, and permission onboarding.
- Bundle the model; sign, notarize, add camera usage text, and preserve all-local processing.
- Use native AVFoundation capture and either MediaPipe Tasks for iOS/macOS or a small local inference service selected after benchmark comparison.
- Global emergency disable shortcut and automatic pause when the camera is obscured, the hand is lost, or tracking confidence degrades.

### Phase 3 — customization and accessibility

- Gesture-to-action editor with conflict detection and per-app profiles.
- Left-handed/mobility-limited alternatives, dwell click, sensitivity curves, tremor filtering, and reduced-motion preview.
- Optional shortcuts: browser back/forward, Mission Control, app switcher, media control, screenshots, and presentation navigation.
- Accessibility review: every gesture has a keyboard/menu equivalent; no gesture is the only way to exit.

### Phase 4 — reliability and release

- Test matrix across lighting, skin tones, sleeves/backgrounds, hand sizes, camera positions, external displays, and supported macOS versions.
- Automated state-machine replay from synthetic landmark traces plus opt-in, consented real traces containing landmarks only.
- Threat/privacy review, signed updates, crash reporting that excludes images/landmarks by default, and a clear camera indicator/status.

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Accidental clicks/scrolls | Distinct clutch poses, stability windows, hysteresis, cooldowns, safe open-palm pose |
| Cursor jitter/fatigue | Active-area mapping, One Euro filter, adjustable gain, rest pose, short sessions |
| Stuck drag | Release on pinch-up, hand loss, pause, quit, and exception cleanup |
| Perceived lag | Latest-frame pipeline; never queue frames; profile capture/inference/event timing separately |
| Permission confusion | Guided Camera + Accessibility onboarding and trust checks |
| Privacy concern | On-device inference, no recording, no network after model is bundled |
| Gesture exclusion | Per-user calibration and alternative gestures/dwell controls |

## Immediate validation script

1. Run preview for two minutes and verify each pose is stable under normal lighting.
2. Enable control and acquire targets at each screen corner and center.
3. Open a long page in Chrome; alternate slow and fast two-finger vertical movement.
4. Click ten small toolbar targets, drag a Finder item without dropping it, and right-click three times.
5. Remove the hand during a drag and confirm the button releases.
6. Work normally for ten minutes with the app enabled and count unintended clicks/scrolls.

Write the result in `NOTES.md`. That verdict decides whether to tune the state machine, change the gesture set, or proceed to the native shell.
