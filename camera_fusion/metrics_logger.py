from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class _OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def avg(self) -> float:
        return self.mean if self.count > 0 else 0.0

    def std(self) -> float:
        if self.count <= 0:
            return 0.0
        variance = self.m2 / self.count
        return variance ** 0.5


class CameraMetricsLogger:
    """Per-camera metrics logger for ArUco baseline measurements."""

    FIELDNAMES = [
        "timestamp",
        "frame_index",
        "detected_marker_count",
        "expected_marker_count",
        "success_bool",
        "partial_ratio",
        "marker_ids_detected",
        "pose_available_bool",
        "tx",
        "ty",
        "tz",
        "rx",
        "ry",
        "rz",
        "aruco_detect_ms",
        "total_frame_ms",
    ]

    def __init__(
        self,
        session_dir: str | Path,
        expected_marker_count: int = 3,
        flush_every_n_frames: int = 30,
        primary_marker_strategy: str = "min_id",
    ):
        self.session_dir = Path(session_dir)
        self.expected_marker_count = max(1, int(expected_marker_count))
        self.flush_every_n_frames = max(1, int(flush_every_n_frames))
        self.primary_marker_strategy = primary_marker_strategy

        self.csv_path = self.session_dir / "metrics_frames.csv"
        self.summary_path = self.session_dir / "metrics_summary.json"

        self._csv_fp = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_fp, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        self._rows_since_flush = 0
        self._finalized = False

        self.total_frames = 0
        self.frames_with_any_marker = 0
        self.frames_success_all3 = 0

        self.partial_ratio_stats = _OnlineStats()
        self.aruco_detect_ms_stats = _OnlineStats()
        self.total_frame_ms_stats = _OnlineStats()

        self.tx_stats = _OnlineStats()
        self.ty_stats = _OnlineStats()
        self.tz_stats = _OnlineStats()
        self.rx_stats = _OnlineStats()
        self.ry_stats = _OnlineStats()
        self.rz_stats = _OnlineStats()

        self.counts_by_marker_id: dict[int, int] = defaultdict(int)

    def _select_primary_marker_id(self, marker_ids: list[int]) -> Optional[int]:
        if not marker_ids:
            return None
        if self.primary_marker_strategy == "min_id":
            return min(marker_ids)
        return min(marker_ids)

    @staticmethod
    def _vec3(value: Any) -> Optional[tuple[float, float, float]]:
        if value is None:
            return None
        try:
            flat = value.reshape(-1)
            if flat.shape[0] < 3:
                return None
            return float(flat[0]), float(flat[1]), float(flat[2])
        except Exception:
            try:
                vals = list(value)
                if len(vals) < 3:
                    return None
                return float(vals[0]), float(vals[1]), float(vals[2])
            except Exception:
                return None

    def log_frame(
        self,
        *,
        timestamp: str,
        frame_index: int,
        detected_marker_ids: list[int],
        pose_map: dict[int, Any],
        aruco_detect_ms: float,
        total_frame_ms: float,
    ) -> None:
        marker_ids = sorted(int(mid) for mid in detected_marker_ids)
        detected_marker_count = len(marker_ids)
        success_bool = 1 if detected_marker_count == self.expected_marker_count else 0
        partial_ratio = detected_marker_count / float(self.expected_marker_count)

        primary_marker_id = self._select_primary_marker_id(marker_ids)
        pose = pose_map.get(primary_marker_id) if primary_marker_id is not None else None

        rvec = None
        tvec = None
        if pose is not None:
            rvec = self._vec3(getattr(pose, "rvec", None))
            tvec = self._vec3(getattr(pose, "tvec", None))

        pose_available = 1 if (rvec is not None and tvec is not None) else 0

        tx, ty, tz = (None, None, None)
        rx, ry, rz = (None, None, None)
        if pose_available:
            tx, ty, tz = tvec
            rx, ry, rz = rvec

            self.tx_stats.add(tx)
            self.ty_stats.add(ty)
            self.tz_stats.add(tz)
            self.rx_stats.add(rx)
            self.ry_stats.add(ry)
            self.rz_stats.add(rz)

        self.total_frames += 1
        if detected_marker_count > 0:
            self.frames_with_any_marker += 1
        if success_bool == 1:
            self.frames_success_all3 += 1

        self.partial_ratio_stats.add(partial_ratio)
        self.aruco_detect_ms_stats.add(float(aruco_detect_ms))
        self.total_frame_ms_stats.add(float(total_frame_ms))

        for marker_id in marker_ids:
            self.counts_by_marker_id[marker_id] += 1

        self._writer.writerow(
            {
                "timestamp": timestamp,
                "frame_index": int(frame_index),
                "detected_marker_count": detected_marker_count,
                "expected_marker_count": self.expected_marker_count,
                "success_bool": success_bool,
                "partial_ratio": partial_ratio,
                "marker_ids_detected": json.dumps(marker_ids),
                "pose_available_bool": pose_available,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "aruco_detect_ms": float(aruco_detect_ms),
                "total_frame_ms": float(total_frame_ms),
            }
        )

        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n_frames:
            self._csv_fp.flush()
            self._rows_since_flush = 0

    def _build_summary(self) -> dict[str, Any]:
        total = self.total_frames
        success_rate_all3 = (self.frames_success_all3 / total) if total > 0 else 0.0

        return {
            "total_frames": total,
            "frames_with_any_marker": self.frames_with_any_marker,
            "frames_success_all3": self.frames_success_all3,
            "success_rate_all3": success_rate_all3,
            "avg_partial_ratio": self.partial_ratio_stats.avg(),
            "std_partial_ratio": self.partial_ratio_stats.std(),
            "avg_aruco_detect_ms": self.aruco_detect_ms_stats.avg(),
            "std_aruco_detect_ms": self.aruco_detect_ms_stats.std(),
            "avg_total_frame_ms": self.total_frame_ms_stats.avg(),
            "std_total_frame_ms": self.total_frame_ms_stats.std(),
            "pose_stability": {
                "std_tx": self.tx_stats.std(),
                "std_ty": self.ty_stats.std(),
                "std_tz": self.tz_stats.std(),
                "std_rx": self.rx_stats.std(),
                "std_ry": self.ry_stats.std(),
                "std_rz": self.rz_stats.std(),
            },
            "counts_by_marker_id": {
                str(k): v for k, v in sorted(self.counts_by_marker_id.items(), key=lambda item: item[0])
            },
        }

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            self._csv_fp.flush()
        finally:
            self._csv_fp.close()

        summary = self._build_summary()
        with self.summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
