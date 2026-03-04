#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _latest_session(session_root: Path, camera_name: str) -> Path | None:
    candidates = sorted(
        session_root.glob(f"{camera_name}_session_*"),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _print_camera_metrics(session_root: Path, camera_name: str, show_summary: bool) -> None:
    session = _latest_session(session_root, camera_name)
    if session is None:
        print(f"[{camera_name}] no session folders found under: {session_root}")
        return

    frames_csv = session / "metrics_frames.csv"
    summary_json = session / "metrics_summary.json"
    log_file = session / "logs" / "session.log"

    print(f"[{camera_name}] latest session: {session}")
    print(f"  metrics_frames.csv: {frames_csv} ({'OK' if frames_csv.exists() else 'MISSING'})")
    print(f"  metrics_summary.json: {summary_json} ({'OK' if summary_json.exists() else 'MISSING'})")
    print(f"  session.log: {log_file} ({'OK' if log_file.exists() else 'MISSING'})")

    if show_summary and summary_json.exists():
        try:
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            print("  summary:")
            print(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"  failed to read summary json: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show latest metrics files for camera sessions"
    )
    parser.add_argument(
        "--session-root",
        default="data/sessions",
        help="Session root directory (default: data/sessions)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam1", "cam2"],
        help="Camera names to check (default: cam1 cam2)",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print the full metrics_summary.json content",
    )
    args = parser.parse_args()

    session_root = Path(args.session_root)
    if not session_root.exists():
        print(f"Session root does not exist: {session_root}")
        return 1

    for camera_name in args.cameras:
        _print_camera_metrics(session_root, camera_name, args.show_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
