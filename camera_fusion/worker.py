from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from iiot_pipeline.ip_types import Frame, Detection
from iiot_pipeline.services.calib import load_calib
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.strategies.preprocess import ColorFrame
from iiot_pipeline.strategies.undistort_simple import SimpleUndistort
from iiot_pipeline.strategies.localize_pnp import PnPLocalize

from .capture import BaseCapture, SyntheticCapture, USBOpenCVCapture
from .config import CameraConfig
from .detect import build_detector, detect_markers
from .logging_utils import add_file_handler, setup_logger
from .output import CsvOutput, OutputSink
from .transforms import compute_relative_pose


@dataclass
class SessionSummary:
    session_path: str
    frames_processed: int
    csv_path: str
    log_path: str
    avg_fps: float
    errors: int


class NoUndistort:
    def apply(self, f: Frame) -> Frame:
        return f


class NoLocalize:
    def estimate(self, detections) -> list:
        return []


class CameraWorker:
    def __init__(
        self,
        config: CameraConfig,
        logger=None,
        outputs: Optional[list[OutputSink]] = None,
        capture: Optional[BaseCapture] = None,
    ):
        self.config = config
        self.logger = logger or setup_logger(config.camera_name)
        
        # Default output with reference support if reference_id is set
        if outputs is None:
            use_ref = config.reference_id is not None
            outputs = [CsvOutput(use_reference=use_ref)]
        
        self.outputs = outputs
        self.capture = capture
        self._stop_event = threading.Event()

        if config.target_ids is None:
            self.target_ids = None
        else:
            self.target_ids = {int(t) for t in config.target_ids}

    def stop(self) -> None:
        self._stop_event.set()

    def _build_capture(self) -> BaseCapture:
        if self.capture is not None:
            return self.capture
        if self.config.dry_run:
            return SyntheticCapture(self.config.fps, self.config.width, self.config.height)
        return USBOpenCVCapture(
            self.config.device,
            self.config.fps,
            self.config.width,
            self.config.height,
        )

    def run(self) -> SessionSummary:
        storage = SessionStorage(self.config.session_root, name=f"{self.config.camera_name}_session")
        session_path = storage.begin()
        storage.write_manifest(self.config.as_dict())

        log_file = str(Path(storage.logs_dir) / "session.log")
        add_file_handler(self.logger, self.config.camera_name, log_file)

        for out in self.outputs:
            out.open(Path(storage.session_dir))

        cap = self._build_capture()
        pre = ColorFrame()

        if self.config.dry_run:
            und = NoUndistort()
            loc = NoLocalize()
        else:
            und = SimpleUndistort(self.config.calibration_path)
            K, dist, _ = load_calib(self.config.calibration_path)
            loc = PnPLocalize(K, dist, self.config.marker_length_m)

        detector_state = build_detector(self.config.aruco_dict)
        
        # Initialize LightGlue fallback if enabled
        lightglue_fallback = None
        if self.config.lightglue is not None and self.config.lightglue.enabled:
            try:
                from .fallback import LightGlueFallback
                lightglue_fallback = LightGlueFallback(
                    self.config.lightglue,
                    self.config.aruco_dict
                )
                self.logger.info("LightGlue fallback initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LightGlue fallback: {e}")
                lightglue_fallback = None

        self.logger.info("session started: %s", session_path)
        self.logger.info("config: %s", self.config.as_dict())

        cap.start()
        t0 = time.time()
        frames = 0
        errors = 0

        try:
            while True:
                if self._stop_event.is_set():
                    break
                if self.config.duration_sec and (time.time() - t0) >= self.config.duration_sec:
                    break
                if self.config.max_frames and frames >= self.config.max_frames:
                    break

                f = cap.next_frame()
                if f is None:
                    errors += 1
                    continue

                f = pre.apply(f)
                f = und.apply(f)

                dets = []
                poses = []
                pose_map = {}
                debug_info_list = []
                
                if not self.config.no_detect and not self.config.dry_run:
                    # Step 1: Run ArUco detection
                    dets = detect_markers(f.image, detector_state)
                    
                    # Step 2: Build detected_dict for fallback
                    detected_dict = {d.marker_id: d.corners for d in dets}
                    
                    # Step 3: Attempt LightGlue recovery for missing markers
                    if lightglue_fallback is not None and self.target_ids is not None:
                        detected_dict, debug_info_list = lightglue_fallback.recover_missing(
                            f.image,
                            detected_dict,
                            self.target_ids
                        )
                        
                        # Rebuild dets list from updated detected_dict
                        dets = [
                            Detection(mid, corners, None, None)
                            for mid, corners in detected_dict.items()
                        ]
                    
                    # Step 4: Continue with existing pose estimation
                    if self.config.marker_lengths_m:
                        poses = loc.estimate_with_lengths(
                            dets,
                            self.config.marker_lengths_m,
                            default_length=self.config.marker_length_m,
                        )
                    else:
                        poses = loc.estimate(dets)

                    pose_map = {
                        d.marker_id: poses[i]
                        for i, d in enumerate(dets)
                        if i < len(poses) and poses[i] is not None
                    }

                if dets and self.config.save_annotated:
                    draw = f.image.copy()
                    h, w = draw.shape[:2]
                    txt = f"#{f.idx} {f.ts_iso} {w}x{h}"
                    cv2.putText(
                        draw,
                        txt,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Build debug info lookup
                    debug_sources = {}
                    for info in debug_info_list:
                        debug_sources[info.marker_id] = info.source
                    
                    # Draw detected markers with color coding by source
                    for det in dets:
                        source = debug_sources.get(det.marker_id, "aruco")
                        
                        # Color code: green=aruco, cyan=lg_track, magenta=lg_reacquire
                        if source == "aruco":
                            color = (0, 255, 0)  # Green
                        elif source == "lg_track":
                            color = (255, 255, 0)  # Cyan
                        else:  # lg_reacquire
                            color = (255, 0, 255)  # Magenta
                        
                        # Draw marker corners
                        corners_int = det.corners.astype(np.int32)
                        cv2.polylines(draw, [corners_int], True, color, 2)
                        
                        # Draw marker ID
                        # Draw marker ID
                        center = corners_int.reshape(-1, 2).mean(axis=0)
                        org = (int(center[0]), int(center[1]))

                        label = f"{det.marker_id}"
                        if source != "aruco":
                            label += f" ({source})"

                        cv2.putText(
                            draw,
                            label,
                            org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )


                    # Draw pose axes
                    for det in dets:
                        pose = pose_map.get(det.marker_id)
                        if pose is None:
                            continue
                        try:
                            cv2.drawFrameAxes(
                                draw,
                                loc.K,
                                loc.dist,
                                pose.rvec,
                                pose.tvec,
                                max(0.01, self.config.marker_length_m * 0.5),
                            )
                        except Exception:
                            pass

                    storage.save_annotated(f.idx, draw)
                    
                    # Save debug frame if LightGlue debug enabled
                    if lightglue_fallback is not None and self.config.lightglue.debug_save:
                        debug_path = Path(storage.session_dir) / "debug_lightglue"
                        debug_path.mkdir(exist_ok=True)
                        debug_file = debug_path / f"frame_{f.idx:06d}.png"
                        cv2.imwrite(str(debug_file), draw)

                if self.config.save_frames:
                    storage.save_frame(f)

                # Compute reference-relative poses if reference marker is present
                ref_visible = False
                ref_pose_map = {}
                
                if self.config.reference_id is not None and self.config.reference_id in pose_map:
                    ref_pose = pose_map[self.config.reference_id]
                    if ref_pose is not None and ref_pose.rvec is not None and ref_pose.tvec is not None:
                        ref_visible = True
                    else:
                        ref_visible = False
                    
                    for marker_id, pose in pose_map.items():
                        if marker_id == self.config.reference_id:
                            continue
                        if pose is None or pose.rvec is None or pose.tvec is None:
                            continue
                        try:
                            ref_rvec, ref_tvec = compute_relative_pose(
                                ref_pose.rvec, ref_pose.tvec,
                                pose.rvec, pose.tvec
                            )
                            ref_pose_map[marker_id] = (ref_rvec, ref_tvec)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to compute relative pose for marker %d: %s",
                                marker_id, e
                            )

                for det in dets:
                    if self.target_ids is not None and det.marker_id not in self.target_ids:
                        continue
                    pose = pose_map.get(det.marker_id)
                    rvec = pose.rvec if pose is not None else None
                    tvec = pose.tvec if pose is not None else None
                    ts_unix = time.time()

                    length_m = self.config.marker_length_m
                    if self.config.marker_lengths_m and det.marker_id in self.config.marker_lengths_m:
                        length_m = self.config.marker_lengths_m[det.marker_id]
                    
                    # Get reference-relative pose if available
                    ref_rvec, ref_tvec = None, None
                    if det.marker_id in ref_pose_map:
                        ref_rvec, ref_tvec = ref_pose_map[det.marker_id]

                    for out in self.outputs:
                        out.write_detection(
                            ts_unix,
                            f.idx,
                            det.marker_id,
                            rvec,
                            tvec,
                            storage.last_path,
                            ref_visible=ref_visible,
                            ref_rvec=ref_rvec,
                            ref_tvec=ref_tvec,
                            length_m=length_m,
                        )

                self.logger.info(
                    "frame=%d saved=%s dets=%d",
                    f.idx,
                    storage.last_path,
                    len(dets),
                )
                frames += 1

        finally:
            try:
                cap.stop()
            except Exception:
                pass

            for out in self.outputs:
                try:
                    out.close()
                except Exception:
                    pass

        avg = frames / max(1e-6, (time.time() - t0))
        self.logger.info("summary frames=%d avg_fps=%.2f errors=%d", frames, avg, errors)

        csv_path = str(Path(storage.session_dir) / "detections.csv")
        return SessionSummary(
            str(session_path),
            frames,
            csv_path,
            log_file,
            avg,
            errors,
        )
