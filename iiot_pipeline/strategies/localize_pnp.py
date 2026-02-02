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

    def estimate_with_lengths(
        self,
        detections: list[Detection],
        length_map: dict[int, float] | None,
        default_length: float | None = None,
    ) -> list[Pose]:
        poses = []
        if not detections:
            return poses
        default_len = self.L if default_length is None else default_length
        if default_len <= 0 and not length_map:
            return poses

        for det in detections:
            length = default_len
            if length_map and det.marker_id in length_map:
                length = length_map[det.marker_id]
            if length <= 0:
                poses.append(Pose(None, None))
                continue
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [det.corners], length, self.K, self.dist
            )
            poses.append(Pose(rvec[0], tvec[0]))
        return poses

