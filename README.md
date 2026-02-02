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

### Quick Start (new system)

1. **Find your camera device:**
   ```bash
   v4l2-ctl --list-devices
   ```
   Note the device path (e.g., `/dev/video8`) or index (e.g., `8`).

2. **Edit a config file** (e.g., `configs/cam1.json`):
   ```json
   {
     "camera_name": "cam1",
     "device": 8,
     "target_ids": [1, 2, 3]
   }
   ```

3. **Run it:**
   ```bash
   python -m camera_fusion.run --config configs/cam1.json
   ```

4. **Check outputs:**
   ```bash
   ls data/sessions/cam1_session_*/
   ```

### One camera

Run with a config file:
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

Press `Ctrl+C` once to stop all cameras cleanly.

### Optional: YAML configs

Configs are JSON by default. To use YAML instead:

```bash
pip install pyyaml
```

Then create `configs/cam1.yaml`:
```yaml
camera_name: cam1
device: 8
fps: 15
target_ids: [1, 2, 3]
reference_id: 0  # Floor/world marker
```

### Reference frame / world coordinate system

To compute robot marker poses relative to a fixed world frame (e.g., a floor marker):

1. **Place a reference ArUco marker** on the floor or fixed location
2. **Set `reference_id`** to that marker's ID in your config:
   ```json
   {
     "reference_id": 0,
     "target_ids": [0, 1, 2]
   }
   ```
3. **Include reference ID in `target_ids`** so it gets detected

When the reference marker is visible, the CSV will include:
- Camera-relative poses (`rvec_x/y/z`, `tvec_x/y/z`)
- Reference-relative poses (`ref_rvec_x/y/z`, `ref_tvec_x/y/z`)
- `ref_visible` flag (1 if reference seen, 0 otherwise)

When reference marker is NOT visible in a frame:
- `ref_visible = 0`
- Reference-relative fields are NaN
- Camera-relative poses still logged

**Use case:** Track robot joint markers relative to floor coordinate system instead of camera.

### Dry run (no physical camera)

```bash
python -m camera_fusion.run --config configs/cam1.json --dry-run --max-frames 5 --no-detect --no-save-frames
```

### Config file options

All config files (JSON or YAML) support these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_name` | string | `"cam"` | Unique identifier for logging and output folder prefix |
| `device` | int or string | `0` | Camera index (e.g., `8`) or path (e.g., `"/dev/video8"`) |
| `fps` | int | `15` | Target frame rate |
| `width` | int | `1920` | Frame width |
| `height` | int | `1080` | Frame height |
| `calibration_path` | string | `"calib/c920s_1920x1080_simple.yml"` | Path to calibration file |
| `session_root` | string | `"data/sessions"` | Root directory for session outputs |
| `duration_sec` | float | `30.0` | Max session duration (seconds) |
| `aruco_dict` | string | `"4x4_50"` | ArUco dictionary (accepts `4x4_50`, `DICT_4X4_50`, `4X4_50`, etc.) |
| `marker_length_m` | float | `0.035` | Marker side length in meters - measure the **outer black square** edge-to-edge (NOT diagonal, NOT inner pattern) |
| `target_ids` | list of int | `null` | Filter to specific marker IDs (e.g., `[1, 2, 3]`); `null` = all |
| `reference_id` | int or null | `null` | ID of reference/world marker (e.g., floor marker). When set, computes poses of other markers relative to this reference frame |
| `marker_lengths_m` | map | `null` | Per-marker lengths override (e.g., `{ "0": 0.150, "1": 0.050 }`). Keys may be strings in JSON |
| `no_detect` | bool | `false` | Skip detection (capture only) |
| `dry_run` | bool | `false` | Use synthetic frames (no physical camera) |
| `max_frames` | int or null | `null` | Stop after N frames |
| `save_annotated` | bool | `true` | Save frames with ArUco markers drawn |
| `save_frames` | bool | `true` | Save undistorted original frames |

### CLI overrides

Any config option can be overridden from the command line:

```bash
python -m camera_fusion.run --config configs/cam1.json \
  --camera-name cam_left \
  --device /dev/video8 \
  --fps 20 \
  --width 1280 \
  --height 720 \
  --calib calib/other.yml \
  --out data/custom_sessions \
  --duration 60.0 \
  --dict 5x5_100 \
  --marker-length-m 0.05 \
  --target-ids 10 11 12 \
  --no-detect \
  --dry-run \
  --max-frames 100 \
  --no-save-frames \
  --no-save-annotated
```

### Graceful shutdown

Press `Ctrl+C` (SIGINT) to stop cleanly. The worker will:
- Release the camera
- Close output files
- Write session summary to logs

When using the multi-camera launcher, `Ctrl+C` stops all processes.

### Troubleshooting

- **Camera fails to open**: Check `device` index/path and ensure no other process is using it.
- **ArUco detection fails**: Install `opencv-contrib-python` (not just `opencv-python`).
- **YAML config errors**: Install PyYAML: `pip install pyyaml`.
- **Device string `/dev/videoX`**: Automatically parsed to integer index `X` with V4L2 backend.
- **Logs not appearing**: Check `session_root/<camera_name>_session_*/logs/session.log`.
- **Multiple instances conflict**: Ensure each camera has a unique `camera_name` and `device`.
- **Inaccurate pose estimation**: Measure `marker_length_m` precisely. Use a ruler to measure the outer black square edge-to-edge (e.g., if marker is 35mm wide, use `0.035`). Do NOT measure diagonally or just the inner pattern.

## Outputs

### Multi-camera service outputs

Each camera worker creates a unique session folder:

```
data/sessions/<camera_name>_session_YYYYMMDD_HHMMSS/
  frames/             # undistorted originals (no drawings)
  annotated/          # frames with ArUco markers + axes drawn
  detections.csv      # recorded_at, frame_idx, marker_id, rvec_*, tvec_*, [ref_visible, ref_rvec_*, ref_tvec_*], length_m, image_path
  logs/session.log    # per-camera logs with [camera_name] prefix
  config.json         # snapshot of config used for this session
```

**CSV columns:**
- Without reference: `recorded_at, frame_idx, marker_id, rvec_x/y/z, tvec_x/y/z, length_m, image_path`
- With reference: adds `ref_visible, ref_rvec_x/y/z, ref_tvec_x/y/z` (poses relative to reference marker)

Example with two cameras:
```
data/sessions/
  cam1_session_20260202_143012/
  cam2_session_20260202_143012/
```

### Legacy CLI outputs

```
data/sessions/<aruco_session_YYYYMMDD_HHMMSS>/
  frames/        # undistorted originals (no drawings)
  annotated/     # frames with detections (boxes + axes + text)
  detections.csv # ts_iso, frame_idx, marker_id, rvec_*, tvec_*, img_path
  logs/session.log
  config.json
```

---

## Legacy CLI Flags

| Flag | Description |
|------|-------------|
| `--device` | V4L2 index (see `v4l2-ctl --list-devices`). |
| `--fps` / `--duration` | Capture rate & session length. |
| `--calib` | Camera calibration `.yml` (must match resolution). |
| `--out` | Root folder for session outputs. |
| `--dict` | ArUco dictionary (e.g., `4x4_50`, `4x4_100`, `5x5_50`, …). |
| `--marker-length-m` | Marker side length in meters (needed for pose & axes). Use `0` to skip pose estimation (IDs only). |

