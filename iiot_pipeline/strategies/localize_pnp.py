import cv2, numpy as np
from ..ip_types import Detection, Pose

class PnPLocalize:
    def __init__(self, K, dist, marker_length_m: float):
        self.K, self.dist, self.L = K, dist, marker_length_m

    def estimate(self, detections: list[Detection]) -> list[Pose]:
        poses = []
        if self.L <= 0 or not detections: return poses
        # Estimate per marker (you can aggregate later)
        for det in detections:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([det.corners], self.L, self.K, self.dist)
            poses.append(Pose(rvec[0], tvec[0]))
        return poses

