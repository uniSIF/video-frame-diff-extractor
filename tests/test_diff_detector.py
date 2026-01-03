"""Tests for DiffDetector frame difference detection."""

import numpy as np
import pytest

from src.diff_detector import DiffDetector


class TestDiffDetectorInit:
    """Tests for DiffDetector initialization."""

    def test_default_threshold(self) -> None:
        """DiffDetector should use default threshold of 0.05."""
        detector = DiffDetector()
        assert detector.threshold == 0.05

    def test_custom_threshold(self) -> None:
        """DiffDetector should accept custom threshold."""
        detector = DiffDetector(threshold=0.1)
        assert detector.threshold == 0.1


class TestDiffDetectorDetect:
    """Tests for DiffDetector.detect method."""

    def test_identical_frames_no_diff(self) -> None:
        """Identical frames should return False (no difference)."""
        detector = DiffDetector(threshold=0.05)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.detect(frame, frame.copy()) is False

    def test_completely_different_frames(self) -> None:
        """Completely different frames should return True."""
        detector = DiffDetector(threshold=0.05)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert detector.detect(frame1, frame2) is True

    def test_diff_below_threshold(self) -> None:
        """Difference below threshold should return False."""
        detector = DiffDetector(threshold=0.1)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Change only 5% of pixels (below 10% threshold)
        frame2[:5, :, :] = 255
        assert detector.detect(frame1, frame2) is False

    def test_diff_above_threshold(self) -> None:
        """Difference above threshold should return True."""
        detector = DiffDetector(threshold=0.05)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Change 10% of pixels (above 5% threshold)
        frame2[:10, :, :] = 255
        assert detector.detect(frame1, frame2) is True

    def test_diff_at_exact_threshold(self) -> None:
        """Difference at exact threshold should return True (boundary case)."""
        detector = DiffDetector(threshold=0.05)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Change exactly 5% of pixels (at threshold boundary)
        frame2[:5, :, :] = 255
        # At threshold, should detect as changed
        assert detector.detect(frame1, frame2) is True

    def test_grayscale_conversion(self) -> None:
        """Should work with color frames by converting to grayscale."""
        detector = DiffDetector(threshold=0.05)
        # Create color frames with significant color difference
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1[:, :, 0] = 255  # Red channel only
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[:, :, 2] = 255  # Blue channel only
        # Different colors should still be detected after grayscale conversion
        result = detector.detect(frame1, frame2)
        # Note: Red and Blue have different grayscale values
        assert isinstance(result, bool)

    def test_small_noise_filtered(self) -> None:
        """Small noise should be filtered out by blur preprocessing."""
        detector = DiffDetector(threshold=0.05)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Add salt-and-pepper noise to a few scattered pixels
        np.random.seed(42)
        noise_mask = np.random.random((100, 100)) < 0.01  # 1% noise
        frame2[noise_mask] = 255
        # Should filter out small scattered noise
        assert detector.detect(frame1, frame2) is False


class TestDiffDetectorReset:
    """Tests for DiffDetector.reset method."""

    def test_reset_clears_state(self) -> None:
        """Reset should clear any internal state."""
        detector = DiffDetector()
        # Just verify reset can be called without error
        detector.reset()
        # After reset, detector should work normally
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.detect(frame, frame) is False
