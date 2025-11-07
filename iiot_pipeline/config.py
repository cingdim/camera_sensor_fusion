from dataclasses import dataclass

@dataclass
class RunConfig:
    fps: int = 15
    grayscale: bool = False
    cameraType: str = "usb"
    sessionRoot: str = "data/sessions"
    calibrationPath: str = "calib/c920s_1920x1080_simple.yml"
    durationSec: float = 30.0
    arucoDict: str = "4x4_50"
    markerLengthM: float = 0.035
    device: int = 8
    width: int = 1920
    height: int = 1080
    noDetect: bool = False

@dataclass
class SessionSummary:
    sessionPath: str
    framesProcessed: int
    csvPath: str
    logPath: str
    avgFps: float
    errors: int

