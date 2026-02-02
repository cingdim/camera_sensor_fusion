from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class CameraConfig:
    camera_name: str = "cam"
    device: int | str = 0
    fps: int = 15
    width: int = 1920
    height: int = 1080
    calibration_path: str = "calib/c920s_1920x1080_simple.yml"
    session_root: str = "data/sessions"
    duration_sec: float = 30.0
    aruco_dict: str = "4x4_50"
    marker_length_m: float = 0.035
    target_ids: Optional[list[int]] = None
    reference_id: Optional[int] = None
    marker_lengths_m: Optional[dict[int, float]] = None
    no_detect: bool = False
    dry_run: bool = False
    max_frames: Optional[int] = None
    save_annotated: bool = True
    save_frames: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_overrides(self, **kwargs: Any) -> "CameraConfig":
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self


def _normalize_target_ids(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, (int, float)):
        return [int(value)]
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping")
    return data


def load_config(path: str | Path) -> CameraConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    if p.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(p)
    else:
        with p.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON/YAML object")

    cfg = CameraConfig()
    cfg.camera_name = str(raw.get("camera_name", cfg.camera_name))
    cfg.device = raw.get("device", cfg.device)
    cfg.fps = int(raw.get("fps", cfg.fps))
    cfg.width = int(raw.get("width", cfg.width))
    cfg.height = int(raw.get("height", cfg.height))
    cfg.calibration_path = str(raw.get("calibration_path", cfg.calibration_path))
    cfg.session_root = str(raw.get("session_root", cfg.session_root))
    cfg.duration_sec = float(raw.get("duration_sec", cfg.duration_sec))
    cfg.aruco_dict = str(raw.get("aruco_dict", cfg.aruco_dict))
    cfg.marker_length_m = float(raw.get("marker_length_m", cfg.marker_length_m))
    cfg.target_ids = _normalize_target_ids(raw.get("target_ids", cfg.target_ids))
    cfg.reference_id = raw.get("reference_id", cfg.reference_id)
    if cfg.reference_id is not None:
        cfg.reference_id = int(cfg.reference_id)
    lengths_raw = raw.get("marker_lengths_m", cfg.marker_lengths_m)
    if lengths_raw is not None:
        if not isinstance(lengths_raw, dict):
            raise ValueError("marker_lengths_m must be a mapping of marker_id -> length_m")
        cfg.marker_lengths_m = {int(k): float(v) for k, v in lengths_raw.items()}
    cfg.no_detect = bool(raw.get("no_detect", cfg.no_detect))
    cfg.dry_run = bool(raw.get("dry_run", cfg.dry_run))
    cfg.max_frames = raw.get("max_frames", cfg.max_frames)
    if cfg.max_frames is not None:
        cfg.max_frames = int(cfg.max_frames)
    cfg.save_annotated = bool(raw.get("save_annotated", cfg.save_annotated))
    cfg.save_frames = bool(raw.get("save_frames", cfg.save_frames))
    return cfg
