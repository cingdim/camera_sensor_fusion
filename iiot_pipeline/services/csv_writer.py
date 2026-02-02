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

    HEADER_WITH_REF = [
        "recorded_at",
        "frame_idx", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "ref_visible",
        "ref_rvec_x", "ref_rvec_y", "ref_rvec_z",
        "ref_tvec_x", "ref_tvec_y", "ref_tvec_z",
        "image_path"
    ]

    def __init__(self, csv_path: str, use_reference: bool = False):
        self.csv_path = csv_path
        self.use_reference = use_reference
        self._opened = False
        self._fh = None
        self._w = None

    def open(self):
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        header = self.HEADER_WITH_REF if self.use_reference else self.HEADER
        self._w.writerow(header)
        self._opened = True

    def _vec3(self, vec):
        if vec is None:
            return [float("nan")] * 3
        a = np.array(vec).reshape(-1).tolist()
        if len(a) < 3:
            a += [float("nan")] * (3 - len(a))
        return a[:3]

    def append(self, ts_unix, frame_idx, marker_id, rvec, tvec, img_path,
               ref_visible=None, ref_rvec=None, ref_tvec=None):
        r = self._vec3(rvec)
        t = self._vec3(tvec)
        
        if self.use_reference:
            ref_r = self._vec3(ref_rvec)
            ref_t = self._vec3(ref_tvec)
            ref_vis = 1 if ref_visible else 0
            self._w.writerow([
                f"{ts_unix:.6f}",
                frame_idx, marker_id,
                *r, *t,
                ref_vis,
                *ref_r, *ref_t,
                img_path
            ])
        else:
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
