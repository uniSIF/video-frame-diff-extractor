"""Tests for CropProcessor template matching and cropping functionality."""

import numpy as np
import pytest
from typing import Tuple

from src.crop_processor import CropProcessor, CropResult


def create_test_image_with_markers(
    height: int = 300,
    width: int = 200,
    marker_positions: list[int] = None,
    marker_size: Tuple[int, int] = (20, 20),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a test image with markers at specified y positions.

    Returns:
        Tuple of (image, marker_template)
    """
    if marker_positions is None:
        marker_positions = [50, 150]

    # Create base image (gray background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 128

    # Create marker (white rectangle with black border)
    marker_h, marker_w = marker_size
    marker = np.zeros((marker_h, marker_w, 3), dtype=np.uint8)
    marker[2:-2, 2:-2] = 255  # White center with black border

    # Place markers at specified y positions (centered horizontally)
    x_pos = (width - marker_w) // 2
    for y_pos in marker_positions:
        if y_pos + marker_h <= height:
            image[y_pos:y_pos + marker_h, x_pos:x_pos + marker_w] = marker

    return image, marker


class TestCropProcessorInit:
    """Tests for CropProcessor initialization."""

    def test_init_with_condition_image(self) -> None:
        """CropProcessor should accept condition image."""
        condition = np.zeros((20, 20, 3), dtype=np.uint8)
        processor = CropProcessor(condition)
        assert processor.condition_image is not None

    def test_init_default_match_threshold(self) -> None:
        """CropProcessor should use default match threshold of 0.8."""
        condition = np.zeros((20, 20, 3), dtype=np.uint8)
        processor = CropProcessor(condition)
        assert processor.match_threshold == 0.8

    def test_init_custom_match_threshold(self) -> None:
        """CropProcessor should accept custom match threshold."""
        condition = np.zeros((20, 20, 3), dtype=np.uint8)
        processor = CropProcessor(condition, match_threshold=0.9)
        assert processor.match_threshold == 0.9

    def test_init_with_scan_area(self) -> None:
        """CropProcessor should accept scan area."""
        condition = np.zeros((20, 20, 3), dtype=np.uint8)
        scan_area = (10, 20, 100, 200)
        processor = CropProcessor(condition, scan_area=scan_area)
        assert processor.scan_area == scan_area

    def test_init_without_scan_area(self) -> None:
        """CropProcessor should default to None for scan area."""
        condition = np.zeros((20, 20, 3), dtype=np.uint8)
        processor = CropProcessor(condition)
        assert processor.scan_area is None


class TestCropProcessorFindMatches:
    """Tests for CropProcessor template matching functionality."""

    def test_find_matches_detects_single_marker(self) -> None:
        """Should detect a single marker in the image."""
        image, marker = create_test_image_with_markers(marker_positions=[100])
        processor = CropProcessor(marker)
        matches = processor.find_matches(image)
        assert len(matches) == 1
        # Y position should be close to 100
        assert abs(matches[0] - 100) < 5

    def test_find_matches_detects_multiple_markers(self) -> None:
        """Should detect multiple markers in the image."""
        image, marker = create_test_image_with_markers(marker_positions=[50, 150, 250])
        processor = CropProcessor(marker)
        matches = processor.find_matches(image)
        assert len(matches) == 3

    def test_find_matches_returns_sorted_by_y(self) -> None:
        """Matches should be sorted by y coordinate (top to bottom)."""
        image, marker = create_test_image_with_markers(marker_positions=[200, 50, 150])
        processor = CropProcessor(marker)
        matches = processor.find_matches(image)
        assert matches == sorted(matches)

    def test_find_matches_no_match_returns_empty(self) -> None:
        """Should return empty list when no match is found."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        # Create a completely different marker
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255  # Blue only
        processor = CropProcessor(marker, match_threshold=0.95)
        matches = processor.find_matches(image)
        assert len(matches) == 0

    def test_find_matches_respects_scan_area(self) -> None:
        """Should only search within the specified scan area."""
        # Place markers at y=50 and y=200
        image, marker = create_test_image_with_markers(marker_positions=[50, 200])
        # Scan area only covers y=100-250, so only the marker at y=200 should be found
        scan_area = (0, 100, 200, 150)  # x, y, width, height
        processor = CropProcessor(marker, scan_area=scan_area)
        matches = processor.find_matches(image)
        # Should only find the marker at y=200 (within scan area)
        assert len(matches) == 1
        assert abs(matches[0] - 200) < 5

    def test_find_matches_respects_threshold(self) -> None:
        """Should respect match threshold for detection."""
        image, marker = create_test_image_with_markers(marker_positions=[100])
        # Add noise to reduce match quality
        noisy_image = image.copy()
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # With high threshold, might not find match
        processor_high = CropProcessor(marker, match_threshold=0.99)
        matches_high = processor_high.find_matches(noisy_image)

        # With lower threshold, should find match
        processor_low = CropProcessor(marker, match_threshold=0.5)
        matches_low = processor_low.find_matches(noisy_image)

        # Lower threshold should find at least as many matches
        assert len(matches_low) >= len(matches_high)


class TestCropProcessorCrop:
    """Tests for CropProcessor.crop method (basic functionality for 7.1)."""

    def test_crop_returns_crop_result(self) -> None:
        """crop should return a CropResult object."""
        image, marker = create_test_image_with_markers(marker_positions=[50, 150])
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert isinstance(result, CropResult)

    def test_crop_result_has_required_fields(self) -> None:
        """CropResult should have image, start_y, end_y, match_count, should_save fields."""
        image, marker = create_test_image_with_markers(marker_positions=[50, 150])
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert hasattr(result, 'image')
        assert hasattr(result, 'start_y')
        assert hasattr(result, 'end_y')
        assert hasattr(result, 'match_count')
        assert hasattr(result, 'should_save')
        assert hasattr(result, 'warning')

    def test_crop_match_count_reflects_found_markers(self) -> None:
        """match_count should reflect number of markers found."""
        image, marker = create_test_image_with_markers(marker_positions=[50, 150])
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.match_count == 2


class TestCropProcessorCropEdgeCases:
    """Tests for CropProcessor.crop edge cases (Task 7.2)."""

    def test_crop_uses_first_match_as_start_y(self) -> None:
        """start_y should be the y-coordinate of the first match."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        # start_y should be close to 50 (first marker position)
        assert abs(result.start_y - 50) < 5

    def test_crop_uses_second_match_as_end_y(self) -> None:
        """end_y should be the y-coordinate of the second match."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        # end_y should be close to 150 (second marker position)
        assert abs(result.end_y - 150) < 5

    def test_crop_no_match_returns_original_image(self) -> None:
        """When no match found, should return original image."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        # Create a marker that won't match
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255  # Blue only
        processor = CropProcessor(marker, match_threshold=0.95)
        result = processor.crop(image)
        # Should return original image
        assert result.image.shape == image.shape
        np.testing.assert_array_equal(result.image, image)

    def test_crop_no_match_has_warning(self) -> None:
        """When no match found, should have warning message."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255
        processor = CropProcessor(marker, match_threshold=0.95)
        result = processor.crop(image)
        assert result.warning is not None
        assert len(result.warning) > 0

    def test_crop_no_match_should_save_is_false(self) -> None:
        """When no match found, should_save should be False."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255
        processor = CropProcessor(marker, match_threshold=0.95)
        result = processor.crop(image)
        assert result.should_save is False

    def test_crop_no_match_start_y_is_zero(self) -> None:
        """When no match found, start_y should be 0."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255
        processor = CropProcessor(marker, match_threshold=0.95)
        result = processor.crop(image)
        assert result.start_y == 0

    def test_crop_no_match_end_y_is_image_height(self) -> None:
        """When no match found, end_y should be image height."""
        image = np.ones((300, 200, 3), dtype=np.uint8) * 128
        marker = np.zeros((20, 20, 3), dtype=np.uint8)
        marker[:, :, 0] = 255
        processor = CropProcessor(marker, match_threshold=0.95)
        result = processor.crop(image)
        assert result.end_y == 300

    def test_crop_single_match_crops_to_bottom(self) -> None:
        """When single match, should crop from match to bottom."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[100]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        # Should crop from ~100 to 300 (bottom)
        expected_height = 300 - result.start_y
        assert result.image.shape[0] == expected_height
        assert result.end_y == 300

    def test_crop_single_match_has_warning(self) -> None:
        """When single match, should have warning message."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[100]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.warning is not None
        assert len(result.warning) > 0

    def test_crop_single_match_count_is_one(self) -> None:
        """When single match, match_count should be 1."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[100]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.match_count == 1

    def test_crop_single_match_should_save_is_true(self) -> None:
        """When single match, should_save should be True."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[100]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.should_save is True

    def test_crop_two_matches_should_save_is_true(self) -> None:
        """When two matches, should_save should be True."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.should_save is True

    def test_crop_maintains_full_width(self) -> None:
        """Cropped image should maintain original width."""
        image, marker = create_test_image_with_markers(
            height=300, width=200, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        # Width should be same as original
        assert result.image.shape[1] == 200

    def test_crop_two_matches_no_warning(self) -> None:
        """When two or more matches, should have no warning."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        assert result.warning is None

    def test_crop_correct_height_with_two_matches(self) -> None:
        """Cropped height should be end_y - start_y."""
        image, marker = create_test_image_with_markers(
            height=300, marker_positions=[50, 150]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        expected_height = result.end_y - result.start_y
        assert result.image.shape[0] == expected_height

    def test_crop_three_matches_uses_first_two(self) -> None:
        """With three matches, should use first two for cropping."""
        image, marker = create_test_image_with_markers(
            height=400, marker_positions=[50, 150, 250]
        )
        processor = CropProcessor(marker)
        result = processor.crop(image)
        # Should use first (50) and second (150) matches
        assert abs(result.start_y - 50) < 5
        assert abs(result.end_y - 150) < 5
        # match_count should be 3 (all found)
        assert result.match_count == 3

    def test_crop_preserves_image_content(self) -> None:
        """Cropped region should contain correct image content."""
        image, marker = create_test_image_with_markers(
            height=300, width=200, marker_positions=[50, 150]
        )
        # Add unique color at a specific location
        image[100, 100] = [255, 0, 0]  # Red pixel at y=100

        processor = CropProcessor(marker)
        result = processor.crop(image)

        # The red pixel should be in the cropped image
        # at y = 100 - start_y (relative position)
        relative_y = 100 - result.start_y
        if 0 <= relative_y < result.image.shape[0]:
            np.testing.assert_array_equal(result.image[relative_y, 100], [255, 0, 0])
