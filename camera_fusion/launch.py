import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch multiple camera workers")
    ap.add_argument("configs", nargs="+", help="List of config files")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between spawns (sec)")

    args = ap.parse_args()

    procs: list[subprocess.Popen] = []

    def _terminate_all():
        for p in procs:
            if p.poll() is None:
                p.terminate()

    def _handle_signal(_sig, _frame):
        _terminate_all()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    for cfg in args.configs:
        if not Path(cfg).exists():
            raise FileNotFoundError(f"Config not found: {cfg}")
        cmd = [sys.executable, "-m", "camera_fusion.run", "--config", cfg]
        procs.append(subprocess.Popen(cmd))
        if args.delay > 0:
            time.sleep(args.delay)

    try:
        return max(p.wait() for p in procs) if procs else 0
    except KeyboardInterrupt:
        _terminate_all()
        return 130


if __name__ == "__main__":
    sys.exit(main())
