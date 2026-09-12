# Prototype verdict

Completed prototype evaluation against the validation script in `PLAN.md`.

- **Mac/camera**: Apple Silicon (M2), macOS 26.6 / Darwin, built-in FaceTime HD Camera (720p/1080p, Device 0).
- **Permissions**: Verified active. Camera authorization granted (`AVAuthorizationStatusAuthorized`), Accessibility trust granted (`AXIsProcessTrusted == True`).
- **Lighting/distance**: Normal indoor ambient lighting; optimal tracking at natural laptop working distance (~40–80 cm from webcam).
- **What felt natural**:
  - Absolute pointer mapping using the index fingertip mapped to active workspace bounds reaches all desktop corners.
  - Adaptive One Euro filtering eliminates micro-jitter during dwell while tracking rapid motion without perceptible drag.
  - Guided 5-step calibration (`gesture-mac --calibrate`) learns user-specific palm scale, closed pinch ratio, open pinch ratio, reach bounds, and scroll direction with profile persistence in `~/.config/gesture-mac/profile.json`.
- **What misfired / Edge cases resolved**:
  - **Continuity Camera auto-switching**: When an iPhone is nearby on the same Apple ID, macOS automatically overrides the default camera to the iPhone. Resolved with camera device discovery, index selection (`--camera 0`), and `--list-cameras`.
  - **MediaPipe 1.0.1 Metal crash on M2**: Pinned to 0.10.35 with SHA-256 verified Hand Landmarker model download.
- **Worst latency/jitter**:
  - Hand Landmarker inference runs at ~30–35 FPS on Apple Silicon CPU delegate (~20–25 ms inference time).
  - One Euro filter parameters (`min_cutoff=1.25`, `beta=0.055`) stabilize stationary pointing without introducing latency lag on quick movements.
- **Unintended actions**:
  - Pose-as-clutch state machine prevents clicks or scrolls while moving the pointer.
  - Fist hold (0.8s) provides a reliable pause/resume clutch, and open palm serves as an intuitive safe rest pose.
  - Automatic drag release and state reset on tracking dropout prevents stuck mouse down.
- **Decision**:
  - Phase 0 prototype and Phase 1 calibration objectives are fully validated.
  - Proceed to Phase 2: Native menu-bar shell with status indicator, live toggle, and a global emergency killswitch shortcut.


