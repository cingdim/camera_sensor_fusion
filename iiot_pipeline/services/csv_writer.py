import csv
import numpy as np
import io

class CsvWriter:
    # Match mock.py header order
    HEADER = [
        "recorded_at",
        "frame_idx", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "image_path"
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._opened = False
        self._fh = None
        self._w = None

    def open(self):
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.HEADER)
        self._opened = True

    def _vec3(self, vec):
        if vec is None:
            return [float("nan")] * 3
        a = np.array(vec).reshape(-1).tolist()
        if len(a) < 3:
            a += [float("nan")] * (3 - len(a))
        return a[:3]

    def append(self, ts_unix, frame_idx, marker_id, rvec, tvec, img_path):
        r = self._vec3(rvec)
        t = self._vec3(tvec)
        self._w.writerow([
            f"{ts_unix:.6f}",
            frame_idx, marker_id,
            *r, *t,
            img_path
        ])

    @classmethod
    def to_csv_line(cls, ts_unix, frame_idx, marker_id, rvec, tvec, img_path):
        def _vec3_local(vec):
            if vec is None:
                return [float("nan"), float("nan"), float("nan")]
            a = np.array(vec).reshape(-1).tolist()
            if len(a) < 3:
                a += [float("nan")] * (3 - len(a))
            return a[:3]

        r = _vec3_local(rvec)
        t = _vec3_local(tvec)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            f"{ts_unix:.6f}",
            frame_idx, marker_id,
            *r, *t,
            img_path
        ])
        return buf.getvalue().strip()

    def close(self):
        if self._opened and self._fh:
            self._fh.close()
            self._opened = False
            self._fh = None
            self._w = None
