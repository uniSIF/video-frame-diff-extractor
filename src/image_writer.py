"""Image file saving functionality for video frame extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageWriter:
    """Save extracted frames as image files."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """
        Initialize ImageWriter with output directory.

        Creates the output directory if it doesn't exist.

        Args:
            output_dir: Path to the output directory.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        image: NDArray[np.uint8],
        frame_number: int,
        fps: float,
    ) -> Path:
        """
        Save an image to the output directory.

        Args:
            image: Image array in BGR format.
            frame_number: Frame number for filename.
            fps: Frames per second for timestamp calculation.

        Returns:
            Path to the saved image file.
        """
        timestamp = self._format_timestamp(frame_number, fps)
        filename = f"frame_{frame_number:06d}_{timestamp}.png"
        output_path = self.output_dir / filename

        cv2.imwrite(str(output_path), image)

        return output_path

    def _format_timestamp(self, frame_number: int, fps: float) -> str:
        """
        Format timestamp from frame number and fps.

        Args:
            frame_number: Frame number.
            fps: Frames per second.

        Returns:
            Formatted timestamp string (e.g., "01m30s").
        """
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}m{seconds:02d}s"
