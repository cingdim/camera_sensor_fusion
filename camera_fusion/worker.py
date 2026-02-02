from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from iiot_pipeline.ip_types import Frame
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
                if not self.config.no_detect and not self.config.dry_run:
                    dets = detect_markers(f.image, detector_state)
                    poses = loc.estimate(dets)
                    pose_map = {d.marker_id: poses[i] for i, d in enumerate(dets) if i < len(poses)}

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

                    ids = np.array([d.marker_id for d in dets], dtype=np.int32).reshape(-1, 1)
                    corners = [d.corners for d in dets]
                    try:
                        cv2.aruco.drawDetectedMarkers(draw, corners, ids)
                    except Exception:
                        pass

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

                if self.config.save_frames:
                    storage.save_frame(f)

                # Compute reference-relative poses if reference marker is present
                ref_visible = False
                ref_pose_map = {}
                
                if self.config.reference_id is not None and self.config.reference_id in pose_map:
                    ref_visible = True
                    ref_pose = pose_map[self.config.reference_id]
                    
                    for marker_id, pose in pose_map.items():
                        if marker_id == self.config.reference_id:
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
