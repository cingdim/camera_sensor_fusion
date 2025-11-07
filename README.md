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
| `--grayscale` *(optional)* | Preprocess frames in grayscale before detection. |

