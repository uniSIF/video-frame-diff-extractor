"""Tests for ImageWriter image saving functionality."""

import numpy as np
import pytest
from pathlib import Path

from src.image_writer import ImageWriter


class TestImageWriterInit:
    """Tests for ImageWriter initialization."""

    def test_init_creates_output_directory(self, tmp_path: Path) -> None:
        """ImageWriter should create output directory if it doesn't exist."""
        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()
        ImageWriter(output_dir)
        assert output_dir.exists()

    def test_init_accepts_existing_directory(self, tmp_path: Path) -> None:
        """ImageWriter should accept existing directory."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        writer = ImageWriter(output_dir)
        assert writer.output_dir == output_dir

    def test_init_accepts_string_path(self, tmp_path: Path) -> None:
        """ImageWriter should accept string path."""
        output_dir = str(tmp_path / "string_path")
        writer = ImageWriter(output_dir)
        assert writer.output_dir == Path(output_dir)

    def test_init_creates_nested_directories(self, tmp_path: Path) -> None:
        """ImageWriter should create nested directories."""
        output_dir = tmp_path / "level1" / "level2" / "level3"
        ImageWriter(output_dir)
        assert output_dir.exists()


class TestImageWriterSave:
    """Tests for ImageWriter.save method."""

    def test_save_creates_png_file(self, tmp_path: Path) -> None:
        """save should create a PNG file."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = writer.save(image, frame_number=0, fps=30.0)
        assert result.exists()
        assert result.suffix == ".png"

    def test_save_returns_path_object(self, tmp_path: Path) -> None:
        """save should return a Path object."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = writer.save(image, frame_number=0, fps=30.0)
        assert isinstance(result, Path)

    def test_save_filename_contains_frame_number(self, tmp_path: Path) -> None:
        """save should include frame number in filename."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = writer.save(image, frame_number=42, fps=30.0)
        assert "000042" in result.name

    def test_save_filename_contains_timestamp(self, tmp_path: Path) -> None:
        """save should include timestamp in filename."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Frame 90 at 30fps = 3.0 seconds
        result = writer.save(image, frame_number=90, fps=30.0)
        assert "00m03s" in result.name or "3s" in result.name or "003" in result.name

    def test_save_filename_format(self, tmp_path: Path) -> None:
        """save should use format: frame_{number:06d}_{timestamp}.png"""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = writer.save(image, frame_number=123, fps=30.0)
        # Should start with "frame_"
        assert result.name.startswith("frame_")
        # Should contain 6-digit frame number
        assert "000123" in result.name

    def test_save_multiple_images(self, tmp_path: Path) -> None:
        """save should handle multiple images without overwriting."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result1 = writer.save(image, frame_number=1, fps=30.0)
        result2 = writer.save(image, frame_number=2, fps=30.0)

        assert result1 != result2
        assert result1.exists()
        assert result2.exists()

    def test_save_preserves_image_content(self, tmp_path: Path) -> None:
        """save should preserve image content correctly."""
        import cv2

        writer = ImageWriter(tmp_path)
        # Create image with specific pattern
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [255, 128, 64]  # BGR

        result = writer.save(image, frame_number=0, fps=30.0)

        # Read back and verify
        loaded = cv2.imread(str(result))
        assert loaded is not None
        assert loaded.shape == image.shape
        # Check center pixel
        np.testing.assert_array_equal(loaded[50, 50], [255, 128, 64])

    def test_save_handles_large_frame_number(self, tmp_path: Path) -> None:
        """save should handle large frame numbers."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = writer.save(image, frame_number=999999, fps=30.0)
        assert "999999" in result.name
        assert result.exists()

    def test_save_calculates_correct_timestamp(self, tmp_path: Path) -> None:
        """save should calculate timestamp from frame number and fps."""
        writer = ImageWriter(tmp_path)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Frame 1800 at 30fps = 60 seconds = 1 minute
        result = writer.save(image, frame_number=1800, fps=30.0)
        # Should contain 01m00s or similar
        assert "01m" in result.name or "60" in result.name
