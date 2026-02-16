import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Launch multiple camera workers concurrently",
        epilog="Example: python -m camera_fusion.launch configs/cam1_stream.json configs/cam2_stream.json"
    )
    ap.add_argument("configs", nargs="+", help="List of config files (e.g., cam1.json cam2.json)")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between spawns (sec)")

    args = ap.parse_args()

    procs: list[subprocess.Popen] = []

    def _terminate_all():
        """Gracefully terminate all worker processes."""
        for p in procs:
            if p.poll() is None:
                p.terminate()

    def _handle_signal(_sig, _frame):
        """Signal handler for SIGINT and SIGTERM."""
        print("\n[launch.py] Received stop signal, terminating all workers...")
        _terminate_all()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    # Start all workers
    for i, cfg in enumerate(args.configs, 1):
        cfg_path = Path(cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")
        cmd = [sys.executable, "-m", "camera_fusion.run", "--config", cfg]
        print(f"[launch.py] Starting worker {i}/{len(args.configs)}: {cfg_path.name}")
        procs.append(subprocess.Popen(cmd))
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"[launch.py] All {len(procs)} worker(s) started. Press Ctrl+C to stop.")

    try:
        exit_codes = [p.wait() for p in procs]
        max_code = max(exit_codes) if exit_codes else 0
        if max_code != 0:
            print(f"[launch.py] One or more workers exited with error code {max_code}")
        return max_code
    except KeyboardInterrupt:
        _terminate_all()
        return 130


if __name__ == "__main__":
    sys.exit(main())
