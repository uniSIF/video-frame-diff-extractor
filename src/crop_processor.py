"""Condition image-based cropping functionality using template matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class CropResult:
    """Result of crop operation."""

    image: NDArray[np.uint8]
    start_y: int
    end_y: int
    match_count: int
    warning: Optional[str] = None


class CropProcessor:
    """Process images by cropping based on condition image template matching."""

    def __init__(
        self,
        condition_image: NDArray[np.uint8],
        scan_area: Optional[Tuple[int, int, int, int]] = None,
        match_threshold: float = 0.8,
    ) -> None:
        """
        Initialize CropProcessor.

        Args:
            condition_image: Template image to match for cropping boundaries.
            scan_area: Optional (x, y, width, height) to limit search area.
            match_threshold: Minimum match score (0.0-1.0) to consider a match.
        """
        self.condition_image = condition_image
        self.scan_area = scan_area
        self.match_threshold = match_threshold

    def find_matches(self, image: NDArray[np.uint8]) -> List[int]:
        """
        Find all positions where the condition image matches.

        Args:
            image: Image to search for matches.

        Returns:
            List of y-coordinates where matches were found, sorted ascending.
        """
        # Determine search region
        if self.scan_area is not None:
            x, y, w, h = self.scan_area
            search_region = image[y:y + h, x:x + w]
            y_offset = y
        else:
            search_region = image
            y_offset = 0

        # Check if template is larger than search region
        template_h, template_w = self.condition_image.shape[:2]
        region_h, region_w = search_region.shape[:2]

        if template_h > region_h or template_w > region_w:
            return []

        # Perform template matching
        result = cv2.matchTemplate(
            search_region,
            self.condition_image,
            cv2.TM_CCOEFF_NORMED,
        )

        # Find all locations above threshold
        locations = np.where(result >= self.match_threshold)
        y_positions = locations[0]

        if len(y_positions) == 0:
            return []

        # Group nearby matches (within template height) to avoid duplicates
        y_positions_sorted = sorted(set(y_positions))
        grouped_matches: List[int] = []

        for y_pos in y_positions_sorted:
            # Add y_offset to convert back to original image coordinates
            absolute_y = y_pos + y_offset

            # Check if this is a new match (not too close to previous)
            if not grouped_matches or absolute_y - grouped_matches[-1] >= template_h:
                grouped_matches.append(absolute_y)

        return grouped_matches

    def crop(self, image: NDArray[np.uint8]) -> CropResult:
        """
        Crop image based on condition image match positions.

        The cropping uses the y-coordinates of matched positions:
        - start_y: Top of first match
        - end_y: Top of second match (or bottom of image if only one match)

        Args:
            image: Image to crop.

        Returns:
            CropResult with cropped image and metadata.
        """
        matches = self.find_matches(image)
        match_count = len(matches)
        height = image.shape[0]
        warning: Optional[str] = None

        if match_count == 0:
            # No match: return original image with warning
            warning = "条件画像が見つかりませんでした。元画像を保存します。"
            return CropResult(
                image=image,
                start_y=0,
                end_y=height,
                match_count=0,
                warning=warning,
            )
        elif match_count == 1:
            # Single match: crop from match to bottom
            warning = "条件画像が1箇所のみ見つかりました。下端までクロップします。"
            start_y = matches[0]
            return CropResult(
                image=image[start_y:, :],
                start_y=start_y,
                end_y=height,
                match_count=1,
                warning=warning,
            )
        else:
            # Two or more matches: crop between first and second
            start_y = matches[0]
            end_y = matches[1]
            return CropResult(
                image=image[start_y:end_y, :],
                start_y=start_y,
                end_y=end_y,
                match_count=match_count,
                warning=None,
            )
