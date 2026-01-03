"""Integration tests for complete video processing pipeline.

These tests verify end-to-end functionality using real video files
generated in-memory for testing purposes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.main import main, process_video


def create_test_video(
    output_path: Path,
    frames: List[NDArray[np.uint8]],
    fps: float = 30.0,
) -> None:
    """
    Create a test video file from a list of frames.

    Args:
        output_path: Path to save the video file.
        frames: List of BGR image arrays.
        fps: Frames per second for the output video.
    """
    if not frames:
        raise ValueError("At least one frame is required")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class TestVideoToImagePipeline:
    """Integration tests for complete video-to-image extraction pipeline."""

    def test_extracts_frames_with_differences(self, tmp_path: Path) -> None:
        """Test complete flow: video file → difference detection → image output."""
        # Create test frames with clear differences
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black frame
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black frame (same)
        frame3 = np.full((100, 100, 3), 255, dtype=np.uint8)  # White frame (different)
        frame4 = np.full((100, 100, 3), 255, dtype=np.uint8)  # White frame (same)
        frame5 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black frame (different)

        frames = [frame1, frame2, frame3, frame4, frame5]

        # Create test video
        video_path = tmp_path / "test_video.mp4"
        create_test_video(video_path, frames)

        # Set up output directory
        output_dir = tmp_path / "output"

        # Run processing
        detected_count = process_video(
            input_file=str(video_path),
            output_dir=str(output_dir),
            threshold=0.5,  # 50% change threshold
            crop_image=None,
            scan_area=None,
        )

        # Verify results
        assert detected_count == 2  # frame3 and frame5 should be detected
        assert output_dir.exists()

        # Check output images exist
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 2

    def test_no_frames_extracted_when_no_differences(self, tmp_path: Path) -> None:
        """Test that no images are saved when video has no significant changes."""
        # Create identical frames
        identical_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frames = [identical_frame.copy() for _ in range(5)]

        # Create test video
        video_path = tmp_path / "identical_frames.mp4"
        create_test_video(video_path, frames)

        output_dir = tmp_path / "output"

        detected_count = process_video(
            input_file=str(video_path),
            output_dir=str(output_dir),
            threshold=0.05,
            crop_image=None,
            scan_area=None,
        )

        # Verify no frames were detected
        assert detected_count == 0

    def test_threshold_affects_detection_sensitivity(self, tmp_path: Path) -> None:
        """Test that threshold parameter controls detection sensitivity."""
        # Create frames with partial change (50% of pixels)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[50:, :, :] = 255  # Bottom half is white

        frames = [frame1, frame2]

        video_path = tmp_path / "partial_change.mp4"
        create_test_video(video_path, frames)

        # High threshold - should not detect
        output_dir_high = tmp_path / "output_high"
        detected_high = process_video(
            input_file=str(video_path),
            output_dir=str(output_dir_high),
            threshold=0.6,  # Require 60% change
            crop_image=None,
            scan_area=None,
        )

        # Low threshold - should detect
        output_dir_low = tmp_path / "output_low"
        detected_low = process_video(
            input_file=str(video_path),
            output_dir=str(output_dir_low),
            threshold=0.3,  # Require 30% change
            crop_image=None,
            scan_area=None,
        )

        assert detected_high == 0
        assert detected_low == 1

    def test_output_directory_created_automatically(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        # Nested directory that doesn't exist
        output_dir = tmp_path / "nested" / "deep" / "output"
        assert not output_dir.exists()

        process_video(
            input_file=str(video_path),
            output_dir=str(output_dir),
            threshold=0.05,
            crop_image=None,
            scan_area=None,
        )

        assert output_dir.exists()

    def test_output_filename_contains_frame_number(self, tmp_path: Path) -> None:
        """Test that output filenames include frame number information."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        output_dir = tmp_path / "output"

        process_video(
            input_file=str(video_path),
            output_dir=str(output_dir),
            threshold=0.05,
            crop_image=None,
            scan_area=None,
        )

        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 1

        # Filename should contain frame number
        filename = output_images[0].name
        assert "frame_" in filename


class TestCropProcessingFlow:
    """Integration tests for cropping functionality with real videos.

    Note: Video encoding introduces compression artifacts that affect template matching.
    These tests use lower match thresholds and mock the CropProcessor initialization
    to ensure consistent matching behavior in integration tests.
    Template matching accuracy is thoroughly tested in test_crop_processor.py.
    """

    def test_cropping_pipeline_runs_with_condition_image(self, tmp_path: Path) -> None:
        """Test that cropping pipeline executes correctly with condition image."""
        from unittest.mock import patch, MagicMock
        from src.crop_processor import CropResult

        frame1 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame2 = np.full((200, 100, 3), 128, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        # Create any marker image (won't be used directly due to mock)
        marker = np.full((10, 20, 3), 255, dtype=np.uint8)
        condition_path = tmp_path / "marker.png"
        cv2.imwrite(str(condition_path), marker)

        output_dir = tmp_path / "output"

        # Mock the CropProcessor to return predictable crop results
        mock_crop_result = CropResult(
            image=frame2[50:150, :],  # Cropped image
            start_y=50,
            end_y=150,
            match_count=2,
            warning=None,
        )

        with patch("src.main.CropProcessor") as MockCropProcessor:
            mock_instance = MagicMock()
            mock_instance.crop.return_value = mock_crop_result
            MockCropProcessor.return_value = mock_instance

            detected_count = process_video(
                input_file=str(video_path),
                output_dir=str(output_dir),
                threshold=0.05,
                crop_image=str(condition_path),
                scan_area=None,
            )

        assert detected_count == 1
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 1

        # Verify CropProcessor was called
        MockCropProcessor.assert_called_once()
        mock_instance.crop.assert_called()

        # Check output image is cropped (height should be 100, not 200)
        cropped_img = cv2.imread(str(output_images[0]))
        assert cropped_img is not None
        assert cropped_img.shape[0] == 100
        assert cropped_img.shape[1] == 100

    def test_cropping_pipeline_with_scan_area(self, tmp_path: Path) -> None:
        """Test that scan_area parameter is passed correctly to CropProcessor."""
        from unittest.mock import patch, MagicMock
        from src.crop_processor import CropResult

        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame2 = np.full((200, 200, 3), 128, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        marker = np.full((10, 10, 3), 255, dtype=np.uint8)
        condition_path = tmp_path / "marker.png"
        cv2.imwrite(str(condition_path), marker)

        output_dir = tmp_path / "output"

        mock_crop_result = CropResult(
            image=frame2[50:150, :],
            start_y=50,
            end_y=150,
            match_count=2,
            warning=None,
        )

        with patch("src.main.CropProcessor") as MockCropProcessor:
            mock_instance = MagicMock()
            mock_instance.crop.return_value = mock_crop_result
            MockCropProcessor.return_value = mock_instance

            detected_count = process_video(
                input_file=str(video_path),
                output_dir=str(output_dir),
                threshold=0.05,
                crop_image=str(condition_path),
                scan_area=(90, 0, 110, 200),
            )

        assert detected_count == 1

        # Verify scan_area was passed to CropProcessor
        call_kwargs = MockCropProcessor.call_args[1]
        assert call_kwargs["scan_area"] == (90, 0, 110, 200)

    def test_crop_fallback_when_no_marker_found(self, tmp_path: Path) -> None:
        """Test that original image is saved when condition image is not found."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 128, dtype=np.uint8)  # No markers

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        # Create marker that won't be found
        marker = np.full((10, 10, 3), 255, dtype=np.uint8)
        condition_path = tmp_path / "marker.png"
        cv2.imwrite(str(condition_path), marker)

        output_dir = tmp_path / "output"

        detected_count = process_video(
            input_file=str(video_path),
            output_dir=str(output_dir),
            threshold=0.05,
            crop_image=str(condition_path),
            scan_area=None,
        )

        assert detected_count == 1

        # Original image should be saved (not cropped)
        output_images = list(output_dir.glob("*.png"))
        saved_img = cv2.imread(str(output_images[0]))
        assert saved_img.shape[0] == 100  # Original height

    def test_crop_warning_displayed_on_single_marker(self, tmp_path: Path) -> None:
        """Test that warning is displayed when only one marker is found."""
        from unittest.mock import patch, MagicMock
        from src.crop_processor import CropResult

        frame1 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame2 = np.full((200, 100, 3), 128, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        marker = np.full((10, 20, 3), 255, dtype=np.uint8)
        condition_path = tmp_path / "marker.png"
        cv2.imwrite(str(condition_path), marker)

        output_dir = tmp_path / "output"

        # Simulate single marker found
        mock_crop_result = CropResult(
            image=frame2[50:, :],
            start_y=50,
            end_y=200,
            match_count=1,
            warning="条件画像が1箇所のみ見つかりました。下端までクロップします。",
        )

        with patch("src.main.CropProcessor") as MockCropProcessor:
            mock_instance = MagicMock()
            mock_instance.crop.return_value = mock_crop_result
            MockCropProcessor.return_value = mock_instance

            with patch("src.main.ProgressReporter") as MockReporter:
                mock_reporter = MagicMock()
                MockReporter.return_value = mock_reporter

                detected_count = process_video(
                    input_file=str(video_path),
                    output_dir=str(output_dir),
                    threshold=0.05,
                    crop_image=str(condition_path),
                    scan_area=None,
                )

        assert detected_count == 1

        # Verify warning was displayed
        mock_reporter.error.assert_called_with(
            "条件画像が1箇所のみ見つかりました。下端までクロップします。"
        )

        # Verify cropped image was saved (height = 150)
        output_images = list(output_dir.glob("*.png"))
        cropped_img = cv2.imread(str(output_images[0]))
        assert cropped_img.shape[0] == 150


class TestErrorHandling:
    """Integration tests for error handling scenarios."""

    def test_graceful_exit_on_file_not_found(self, tmp_path: Path) -> None:
        """Test that proper error is raised for non-existent video file."""
        with pytest.raises(FileNotFoundError, match="ファイルが存在しません"):
            process_video(
                input_file=str(tmp_path / "nonexistent.mp4"),
                output_dir=str(tmp_path / "output"),
                threshold=0.05,
                crop_image=None,
                scan_area=None,
            )

    def test_graceful_exit_on_unsupported_format(self, tmp_path: Path) -> None:
        """Test that proper error is raised for unsupported file format."""
        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("not a video")

        with pytest.raises(ValueError, match="サポートされていない形式"):
            process_video(
                input_file=str(unsupported_file),
                output_dir=str(tmp_path / "output"),
                threshold=0.05,
                crop_image=None,
                scan_area=None,
            )

    def test_graceful_exit_on_invalid_crop_image(self, tmp_path: Path) -> None:
        """Test error handling when crop condition image is invalid."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        with pytest.raises(FileNotFoundError, match="クロップ条件画像"):
            process_video(
                input_file=str(video_path),
                output_dir=str(tmp_path / "output"),
                threshold=0.05,
                crop_image=str(tmp_path / "nonexistent_marker.png"),
                scan_area=None,
            )


class TestCLIIntegration:
    """Integration tests for CLI entry point."""

    def test_main_success_returns_zero(self, tmp_path: Path) -> None:
        """Test that main returns 0 on successful processing."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        output_dir = tmp_path / "output"

        exit_code = main([
            str(video_path),
            "-o", str(output_dir),
            "-t", "0.05",
        ])

        assert exit_code == 0
        assert output_dir.exists()

    def test_main_file_not_found_returns_one(self, tmp_path: Path) -> None:
        """Test that main returns 1 when file is not found."""
        exit_code = main([
            str(tmp_path / "nonexistent.mp4"),
            "-o", str(tmp_path / "output"),
        ])

        assert exit_code == 1

    def test_main_invalid_format_returns_one(self, tmp_path: Path) -> None:
        """Test that main returns 1 for unsupported format."""
        invalid_file = tmp_path / "test.xyz"
        invalid_file.write_text("not a video")

        exit_code = main([
            str(invalid_file),
            "-o", str(tmp_path / "output"),
        ])

        assert exit_code == 1

    def test_main_with_all_options(self, tmp_path: Path) -> None:
        """Test main with all CLI options specified (including crop options)."""
        from unittest.mock import patch, MagicMock
        from src.crop_processor import CropResult

        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame2 = np.full((200, 200, 3), 128, dtype=np.uint8)

        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, [frame1, frame2])

        marker = np.full((10, 10, 3), 255, dtype=np.uint8)
        condition_path = tmp_path / "marker.png"
        cv2.imwrite(str(condition_path), marker)

        output_dir = tmp_path / "output"

        # Mock CropProcessor to ensure predictable behavior
        mock_crop_result = CropResult(
            image=frame2[50:150, :],
            start_y=50,
            end_y=150,
            match_count=2,
            warning=None,
        )

        with patch("src.main.CropProcessor") as MockCropProcessor:
            mock_instance = MagicMock()
            mock_instance.crop.return_value = mock_crop_result
            MockCropProcessor.return_value = mock_instance

            exit_code = main([
                str(video_path),
                "-o", str(output_dir),
                "-t", "0.05",
                "--crop-image", str(condition_path),
                "--scan-area", "90,0,120,200",
            ])

        assert exit_code == 0

        # Verify CropProcessor was initialized with scan_area
        call_kwargs = MockCropProcessor.call_args[1]
        assert call_kwargs["scan_area"] == (90, 0, 120, 200)

        # Verify output
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 1

        cropped_img = cv2.imread(str(output_images[0]))
        assert cropped_img.shape[0] == 100  # Cropped height
