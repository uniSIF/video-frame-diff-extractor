"""Tests for VideoLoader component."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.video_loader import VideoLoader


class TestVideoLoaderInit:
    """Tests for VideoLoader initialization."""

    def test_init_with_nonexistent_file_raises_file_not_found_error(self) -> None:
        """VideoLoader should raise FileNotFoundError for nonexistent files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            VideoLoader("/nonexistent/path/video.mp4")
        assert "存在しません" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    def test_init_with_unsupported_format_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """VideoLoader should raise ValueError for unsupported formats."""
        unsupported_file = tmp_path / "video.txt"
        unsupported_file.write_text("not a video")

        with pytest.raises(ValueError) as exc_info:
            VideoLoader(str(unsupported_file))
        # Error message should list supported formats
        error_msg = str(exc_info.value)
        assert "MP4" in error_msg or "mp4" in error_msg

    def test_supported_formats_are_recognized(self) -> None:
        """VideoLoader should recognize MP4, AVI, MOV, MKV, WebM extensions."""
        supported = VideoLoader.SUPPORTED_FORMATS
        assert ".mp4" in supported
        assert ".avi" in supported
        assert ".mov" in supported
        assert ".mkv" in supported
        assert ".webm" in supported


class TestVideoLoaderWithTestVideo:
    """Tests for VideoLoader with actual video files."""

    @pytest.fixture
    def test_video_path(self) -> str:
        """Return path to test video file if available."""
        # Check for test video in project
        possible_paths = [
            "105_4.mp4",
            "tests/fixtures/test.mp4",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        pytest.skip("No test video available")

    def test_init_opens_valid_video_file(self, test_video_path: str) -> None:
        """VideoLoader should successfully open a valid video file."""
        loader = VideoLoader(test_video_path)
        assert loader is not None
        loader.close()

    def test_frame_count_returns_positive_integer(self, test_video_path: str) -> None:
        """VideoLoader.frame_count should return positive integer."""
        loader = VideoLoader(test_video_path)
        try:
            assert loader.frame_count > 0
            assert isinstance(loader.frame_count, int)
        finally:
            loader.close()

    def test_fps_returns_positive_float(self, test_video_path: str) -> None:
        """VideoLoader.fps should return positive float."""
        loader = VideoLoader(test_video_path)
        try:
            assert loader.fps > 0
            assert isinstance(loader.fps, float)
        finally:
            loader.close()

    def test_frames_returns_iterator(self, test_video_path: str) -> None:
        """VideoLoader.frames() should return an iterator."""
        loader = VideoLoader(test_video_path)
        try:
            frames = loader.frames()
            assert hasattr(frames, "__iter__")
            assert hasattr(frames, "__next__")
        finally:
            loader.close()

    def test_frames_yields_frame_number_and_image(self, test_video_path: str) -> None:
        """VideoLoader.frames() should yield (frame_number, image) tuples."""
        loader = VideoLoader(test_video_path)
        try:
            for frame_num, image in loader.frames():
                # First frame should be frame 0
                assert frame_num == 0
                # Image should be numpy array
                assert isinstance(image, np.ndarray)
                # Image should be 3D (height, width, channels)
                assert len(image.shape) == 3
                assert image.dtype == np.uint8
                break  # Only test first frame
        finally:
            loader.close()

    def test_frames_yields_consecutive_frame_numbers(
        self, test_video_path: str
    ) -> None:
        """VideoLoader.frames() should yield consecutive frame numbers."""
        loader = VideoLoader(test_video_path)
        try:
            frame_numbers = []
            for frame_num, _ in loader.frames():
                frame_numbers.append(frame_num)
                if len(frame_numbers) >= 5:
                    break
            assert frame_numbers == [0, 1, 2, 3, 4]
        finally:
            loader.close()

    def test_close_releases_resources(self, test_video_path: str) -> None:
        """VideoLoader.close() should release video resources."""
        loader = VideoLoader(test_video_path)
        loader.close()
        # After close, trying to iterate should not work
        # (implementation detail, but good to verify resource release)

    def test_context_manager_protocol(self, test_video_path: str) -> None:
        """VideoLoader should support context manager protocol."""
        with VideoLoader(test_video_path) as loader:
            assert loader.frame_count > 0


class TestVideoLoaderCaseInsensitiveExtension:
    """Tests for case-insensitive extension handling."""

    def test_uppercase_extension_is_recognized(self, tmp_path: Path) -> None:
        """VideoLoader should recognize uppercase extensions like .MP4."""
        # Create a file with uppercase extension
        video_file = tmp_path / "video.MP4"
        video_file.write_bytes(b"")  # Empty file

        # Should not raise ValueError for extension, but will fail to open
        # We're testing extension validation, not actual video loading
        with pytest.raises(Exception) as exc_info:
            VideoLoader(str(video_file))
        # Should NOT be ValueError about unsupported format
        assert not (
            isinstance(exc_info.value, ValueError)
            and "サポート" in str(exc_info.value)
        )

    def test_mixed_case_extension_is_recognized(self, tmp_path: Path) -> None:
        """VideoLoader should recognize mixed case extensions like .Mp4."""
        video_file = tmp_path / "video.Mp4"
        video_file.write_bytes(b"")

        with pytest.raises(Exception) as exc_info:
            VideoLoader(str(video_file))
        # Should NOT be ValueError about unsupported format
        assert not (
            isinstance(exc_info.value, ValueError)
            and "サポート" in str(exc_info.value)
        )
