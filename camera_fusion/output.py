from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from iiot_pipeline.services.csv_writer import CsvWriter


class OutputSink(ABC):
    @abstractmethod
    def open(self, session_dir: Path) -> None: ...

    @abstractmethod
    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class CsvOutput(OutputSink):
    def __init__(self, filename: str = "detections.csv"):
        self.filename = filename
        self._writer: Optional[CsvWriter] = None

    def open(self, session_dir: Path) -> None:
        path = session_dir / self.filename
        self._writer = CsvWriter(str(path))
        self._writer.open()

    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
    ) -> None:
        if self._writer is None:
            return
        self._writer.append(ts_unix, frame_idx, marker_id, rvec, tvec, image_path)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class NullOutput(OutputSink):
    def open(self, session_dir: Path) -> None:
        return None

    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
    ) -> None:
        return None

    def close(self) -> None:
        return None
