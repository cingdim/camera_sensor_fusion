# C920s + V4L2 ArUco Pipeline  
*(Facade + Strategy Design Patterns)*

This repository captures frames from a Logitech C920 (or any V4L2 USB camera), **undistorts** them using a calibration file, detects **ArUco markers**, estimates pose using PnP, and saves results to a per-session folder.  

The design applies two object-oriented design patterns:  
- **Facade** a single entrypoint to run the full pipeline  
- **Strategy** pluggable steps for capture, preprocessing, undistortion, detection, and localization  

---

## Quick Start

### 1. Find your camera index
```bash
v4l2-ctl --list-devices
```
Look for the *HD Pro Webcam C920* block (e.g. `/dev/video8`).

---

### 2. Activate environment
```bash
source .venv/bin/activate
```

---

### 3. Run a session
Example (USB C920 on `/dev/video8`, 1080p @ 15fps, ArUco 4x4_50, marker side = 14.45 cm):

```bash
python3 cli.py --device 8 --fps 15 --duration 10   --calib calib/c920s_1920x1080_simple.yml   --out data/sessions   --dict 4x4_50 --marker-length-m 0.1445
```

After it finishes, open the newest folder under `data/sessions/`.

---

## Multi-camera service (new)

This repo now includes a clean, multi-camera runner in the `camera_fusion` package.
Each camera runs in its own process with an isolated config and output folder prefix.

### Setup

Configs are JSON by default. YAML works if you install PyYAML:

```bash
pip install pyyaml
```

### One camera

```bash
python -m camera_fusion.run --config configs/cam1.json
```

Override any config values from the CLI (no code edits needed):

```bash
python -m camera_fusion.run --config configs/cam1.json --device 8 --fps 10 --target-ids 1 2 3
```

### Multiple cameras (separate terminals)

```bash
python -m camera_fusion.run --config configs/cam1.json
python -m camera_fusion.run --config configs/cam2.json
```

### Multiple cameras (single launcher)

```bash
python -m camera_fusion.launch configs/cam1.json configs/cam2.json
```

### Dry run (no physical camera)

```bash
python -m camera_fusion.run --config configs/cam1.json --dry-run --max-frames 5 --no-detect --no-save-frames
```

### Troubleshooting

- If the camera fails to open, double-check the `device` index/path and that no other process is using it.
- If ArUco detection fails, ensure `opencv-contrib-python` is installed.
- For YAML configs, install PyYAML as noted above.

## Outputs

```
data/sessions/<aruco_session_YYYYMMDD_HHMMSS>/
  frames/        # undistorted originals (no drawings)
  annotated/     # frames with detections (boxes + axes + text)
  detections.csv # ts_iso, frame_idx, marker_id, rvec_*, tvec_*, img_path
  logs/session.log
  config.json
```

---

## Flags

| Flag | Description |
|------|-------------|
| `--device` | V4L2 index (see `v4l2-ctl --list-devices`). |
| `--fps` / `--duration` | Capture rate & session length. |
| `--calib` | Camera calibration `.yml` (must match resolution). |
| `--out` | Root folder for session outputs. |
| `--dict` | ArUco dictionary (e.g., `4x4_50`, `4x4_100`, `5x5_50`, …). |
| `--marker-length-m` | Marker side length in meters (needed for pose & axes). Use `0` to skip pose estimation (IDs only). |

