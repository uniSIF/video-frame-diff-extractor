"""Frame difference detection for video processing."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class DiffDetector:
    """Detect differences between consecutive video frames."""

    def __init__(self, threshold: float = 0.05) -> None:
        """
        Initialize the difference detector.

        Args:
            threshold: Difference detection threshold (0.0-1.0).
                      Represents the ratio of changed pixels required
                      to consider frames as different.
        """
        self.threshold = threshold

    def detect(
        self,
        prev_frame: NDArray[np.uint8],
        curr_frame: NDArray[np.uint8],
    ) -> bool:
        """
        Detect if there is a significant difference between two frames.

        Args:
            prev_frame: Previous frame as BGR image array.
            curr_frame: Current frame as BGR image array.

        Returns:
            True if the difference exceeds the threshold, False otherwise.
        """
        # Convert to grayscale for comparison
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)

        # Threshold to get binary mask of changed pixels
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        # Calculate ratio of changed pixels
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = np.count_nonzero(thresh)
        change_ratio = changed_pixels / total_pixels

        return change_ratio >= self.threshold

    def reset(self) -> None:
        """Reset internal state (placeholder for future stateful operations)."""
        pass
