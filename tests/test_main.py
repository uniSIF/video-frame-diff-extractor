"""Tests for main processing pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator, Tuple
from unittest.mock import MagicMock, Mock, patch, call

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.main import main, process_video


class TestProcessVideo:
    """Tests for process_video function."""

    def test_process_video_detects_and_saves_diff_frames(self, tmp_path: Path) -> None:
        """Test that process_video detects different frames and saves them."""
        # Create mock video frames (3 frames: 1st and 3rd are different)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame3 = np.full((100, 100, 3), 255, dtype=np.uint8)  # Different

        mock_loader = MagicMock()
        mock_loader.frame_count = 3
        mock_loader.fps = 30.0
        mock_loader.frames.return_value = iter([
            (0, frame1),
            (1, frame2),
            (2, frame3),
        ])

        output_dir = tmp_path / "output"

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.DiffDetector") as mock_detector_cls:
                mock_detector = MagicMock()
                # Frames 0->1: no diff, Frames 1->2: diff detected
                mock_detector.detect.side_effect = [False, True]
                mock_detector_cls.return_value = mock_detector

                with patch("src.main.ImageWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.save.return_value = output_dir / "frame_000002.png"
                    mock_writer_cls.return_value = mock_writer

                    with patch("src.main.ProgressReporter") as mock_reporter_cls:
                        mock_reporter = MagicMock()
                        mock_reporter_cls.return_value = mock_reporter

                        detected_count = process_video(
                            input_file="test.mp4",
                            output_dir=str(output_dir),
                            threshold=0.05,
                            crop_image=None,
                            scan_area=None,
                        )

        assert detected_count == 1
        mock_writer.save.assert_called_once()
        mock_reporter.complete.assert_called_once()

    def test_process_video_with_crop_enabled(self, tmp_path: Path) -> None:
        """Test that process_video applies cropping when crop_image is provided."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        mock_loader = MagicMock()
        mock_loader.frame_count = 2
        mock_loader.fps = 30.0
        mock_loader.frames.return_value = iter([
            (0, frame1),
            (1, frame2),
        ])

        condition_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cropped_img = np.zeros((50, 100, 3), dtype=np.uint8)

        output_dir = tmp_path / "output"

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.DiffDetector") as mock_detector_cls:
                mock_detector = MagicMock()
                mock_detector.detect.return_value = True
                mock_detector_cls.return_value = mock_detector

                with patch("src.main.cv2.imread", return_value=condition_img):
                    with patch("src.main.CropProcessor") as mock_crop_cls:
                        mock_crop = MagicMock()
                        mock_crop.crop.return_value = MagicMock(
                            image=cropped_img,
                            warning=None,
                        )
                        mock_crop_cls.return_value = mock_crop

                        with patch("src.main.ImageWriter") as mock_writer_cls:
                            mock_writer = MagicMock()
                            mock_writer_cls.return_value = mock_writer

                            with patch("src.main.ProgressReporter"):
                                detected_count = process_video(
                                    input_file="test.mp4",
                                    output_dir=str(output_dir),
                                    threshold=0.05,
                                    crop_image="condition.png",
                                    scan_area=None,
                                )

        assert detected_count == 1
        mock_crop.crop.assert_called_once()
        # Verify cropped image is saved
        saved_image = mock_writer.save.call_args[0][0]
        np.testing.assert_array_equal(saved_image, cropped_img)

    def test_process_video_displays_crop_warning(self, tmp_path: Path) -> None:
        """Test that process_video displays warning when crop has issues."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        mock_loader = MagicMock()
        mock_loader.frame_count = 2
        mock_loader.fps = 30.0
        mock_loader.frames.return_value = iter([
            (0, frame1),
            (1, frame2),
        ])

        condition_img = np.zeros((10, 10, 3), dtype=np.uint8)
        output_dir = tmp_path / "output"

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.DiffDetector") as mock_detector_cls:
                mock_detector = MagicMock()
                mock_detector.detect.return_value = True
                mock_detector_cls.return_value = mock_detector

                with patch("src.main.cv2.imread", return_value=condition_img):
                    with patch("src.main.CropProcessor") as mock_crop_cls:
                        mock_crop = MagicMock()
                        mock_crop.crop.return_value = MagicMock(
                            image=frame2,
                            warning="条件画像が見つかりませんでした。元画像を保存します。",
                        )
                        mock_crop_cls.return_value = mock_crop

                        with patch("src.main.ImageWriter") as mock_writer_cls:
                            mock_writer = MagicMock()
                            mock_writer_cls.return_value = mock_writer

                            with patch("src.main.ProgressReporter") as mock_reporter_cls:
                                mock_reporter = MagicMock()
                                mock_reporter_cls.return_value = mock_reporter

                                detected_count = process_video(
                                    input_file="test.mp4",
                                    output_dir=str(output_dir),
                                    threshold=0.05,
                                    crop_image="condition.png",
                                    scan_area=None,
                                )

        # Should still save the image even with warning
        assert detected_count == 1
        mock_writer.save.assert_called_once()

    def test_process_video_updates_progress(self, tmp_path: Path) -> None:
        """Test that process_video updates progress for each frame."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        mock_loader = MagicMock()
        mock_loader.frame_count = 5
        mock_loader.fps = 30.0
        mock_loader.frames.return_value = iter(enumerate(frames))

        output_dir = tmp_path / "output"

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.DiffDetector") as mock_detector_cls:
                mock_detector = MagicMock()
                mock_detector.detect.return_value = False
                mock_detector_cls.return_value = mock_detector

                with patch("src.main.ImageWriter"):
                    with patch("src.main.ProgressReporter") as mock_reporter_cls:
                        mock_reporter = MagicMock()
                        mock_reporter_cls.return_value = mock_reporter

                        process_video(
                            input_file="test.mp4",
                            output_dir=str(output_dir),
                            threshold=0.05,
                            crop_image=None,
                            scan_area=None,
                        )

        # Progress should be updated for each frame
        assert mock_reporter.update.call_count == 5

    def test_process_video_handles_file_not_found(self, tmp_path: Path) -> None:
        """Test that process_video handles non-existent files gracefully."""
        with patch("src.main.VideoLoader") as mock_loader_cls:
            mock_loader_cls.side_effect = FileNotFoundError("ファイルが存在しません")

            with patch("src.main.ProgressReporter") as mock_reporter_cls:
                mock_reporter = MagicMock()
                mock_reporter_cls.return_value = mock_reporter

                with pytest.raises(FileNotFoundError):
                    process_video(
                        input_file="nonexistent.mp4",
                        output_dir=str(tmp_path / "output"),
                        threshold=0.05,
                        crop_image=None,
                        scan_area=None,
                    )

    def test_process_video_handles_invalid_format(self, tmp_path: Path) -> None:
        """Test that process_video handles unsupported formats gracefully."""
        with patch("src.main.VideoLoader") as mock_loader_cls:
            mock_loader_cls.side_effect = ValueError("サポートされていない形式です")

            with pytest.raises(ValueError):
                process_video(
                    input_file="test.txt",
                    output_dir=str(tmp_path / "output"),
                    threshold=0.05,
                    crop_image=None,
                    scan_area=None,
                )

    def test_process_video_releases_resources_on_error(self, tmp_path: Path) -> None:
        """Test that resources are released even when an error occurs."""
        mock_loader = MagicMock()
        mock_loader.frame_count = 3
        mock_loader.fps = 30.0
        mock_loader.frames.side_effect = RuntimeError("Unexpected error")

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.DiffDetector"):
                with patch("src.main.ImageWriter"):
                    with patch("src.main.ProgressReporter"):
                        with pytest.raises(RuntimeError):
                            process_video(
                                input_file="test.mp4",
                                output_dir=str(tmp_path / "output"),
                                threshold=0.05,
                                crop_image=None,
                                scan_area=None,
                            )

        mock_loader.close.assert_called_once()

    def test_process_video_invalid_crop_image_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid crop image path raises FileNotFoundError."""
        mock_loader = MagicMock()
        mock_loader.frame_count = 2
        mock_loader.fps = 30.0

        with patch("src.main.VideoLoader", return_value=mock_loader):
            with patch("src.main.cv2.imread", return_value=None):
                with pytest.raises(FileNotFoundError, match="クロップ条件画像"):
                    process_video(
                        input_file="test.mp4",
                        output_dir=str(tmp_path / "output"),
                        threshold=0.05,
                        crop_image="nonexistent_condition.png",
                        scan_area=None,
                    )

        mock_loader.close.assert_called_once()


class TestMain:
    """Tests for main function (CLI entry point)."""

    def test_main_parses_args_and_calls_process_video(self) -> None:
        """Test that main parses arguments and calls process_video."""
        with patch("src.main.parse_args") as mock_parse:
            mock_config = MagicMock()
            mock_config.input_file = "test.mp4"
            mock_config.output_dir = "./output"
            mock_config.threshold = 0.05
            mock_config.crop_image = None
            mock_config.scan_area = None
            mock_parse.return_value = mock_config

            with patch("src.main.process_video", return_value=5) as mock_process:
                result = main(["test.mp4"])

        assert result == 0
        mock_process.assert_called_once_with(
            input_file="test.mp4",
            output_dir="./output",
            threshold=0.05,
            crop_image=None,
            scan_area=None,
        )

    def test_main_returns_1_on_file_not_found(self) -> None:
        """Test that main returns 1 when file is not found."""
        with patch("src.main.parse_args") as mock_parse:
            mock_config = MagicMock()
            mock_config.input_file = "nonexistent.mp4"
            mock_config.output_dir = "./output"
            mock_config.threshold = 0.05
            mock_config.crop_image = None
            mock_config.scan_area = None
            mock_parse.return_value = mock_config

            with patch("src.main.process_video") as mock_process:
                mock_process.side_effect = FileNotFoundError("ファイルが存在しません")

                with patch("sys.stderr"):
                    result = main(["nonexistent.mp4"])

        assert result == 1

    def test_main_returns_1_on_value_error(self) -> None:
        """Test that main returns 1 on ValueError (e.g., invalid format)."""
        with patch("src.main.parse_args") as mock_parse:
            mock_config = MagicMock()
            mock_config.input_file = "test.txt"
            mock_config.output_dir = "./output"
            mock_config.threshold = 0.05
            mock_config.crop_image = None
            mock_config.scan_area = None
            mock_parse.return_value = mock_config

            with patch("src.main.process_video") as mock_process:
                mock_process.side_effect = ValueError("サポートされていない形式")

                with patch("sys.stderr"):
                    result = main(["test.txt"])

        assert result == 1

    def test_main_returns_1_on_keyboard_interrupt(self) -> None:
        """Test that main handles keyboard interrupt gracefully."""
        with patch("src.main.parse_args") as mock_parse:
            mock_config = MagicMock()
            mock_config.input_file = "test.mp4"
            mock_config.output_dir = "./output"
            mock_config.threshold = 0.05
            mock_config.crop_image = None
            mock_config.scan_area = None
            mock_parse.return_value = mock_config

            with patch("src.main.process_video") as mock_process:
                mock_process.side_effect = KeyboardInterrupt()

                with patch("sys.stderr"):
                    result = main(["test.mp4"])

        assert result == 130
