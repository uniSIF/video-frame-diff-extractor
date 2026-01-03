"""Video file loading and frame iteration."""

from __future__ import annotations

import os
from typing import Iterator, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoLoader:
    """Load video files and provide frame iteration."""

    SUPPORTED_FORMATS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, file_path: str) -> None:
        """
        Open a video file for reading.

        Args:
            file_path: Path to the video file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            formats_str = ", ".join(sorted(f.upper() for f in self.SUPPORTED_FORMATS))
            raise ValueError(
                f"サポートされていない形式です: {ext}\n"
                f"対応形式: {formats_str}"
            )

        self._file_path = file_path
        self._cap = cv2.VideoCapture(file_path)

        if not self._cap.isOpened():
            raise ValueError(f"映像ファイルを開けませんでした: {file_path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        """Return total number of frames."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Return frames per second."""
        return self._fps

    def frames(self) -> Iterator[Tuple[int, NDArray[np.uint8]]]:
        """
        Yield (frame_number, frame_image) tuples.

        Yields:
            Tuple of (frame_number, frame_image) where frame_image is a BGR numpy array
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_num, frame
            frame_num += 1

    def close(self) -> None:
        """Release video resources."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoLoader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.close()
