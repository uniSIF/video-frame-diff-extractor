"""Tests for ProgressReporter component."""

import io
import sys
from unittest.mock import patch

import pytest

from src.progress_reporter import ProgressReporter


class TestProgressReporterInit:
    """Tests for ProgressReporter initialization."""

    def test_init_with_total_frames(self) -> None:
        """ProgressReporter should accept total_frames in constructor."""
        reporter = ProgressReporter(total_frames=100)
        assert reporter is not None

    def test_init_with_zero_frames(self) -> None:
        """ProgressReporter should handle zero total frames."""
        reporter = ProgressReporter(total_frames=0)
        assert reporter is not None


class TestProgressReporterUpdate:
    """Tests for ProgressReporter.update() method."""

    def test_update_outputs_progress(self) -> None:
        """update() should output progress information."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=50, detected_count=5)
            output = mock_stderr.getvalue()
            # Should contain progress percentage
            assert "50" in output or "50%" in output

    def test_update_shows_detected_count(self) -> None:
        """update() should show number of detected frames."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=50, detected_count=10)
            output = mock_stderr.getvalue()
            # Should contain detected count
            assert "10" in output

    def test_update_uses_carriage_return_for_same_line(self) -> None:
        """update() should use carriage return for same-line update."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=50, detected_count=5)
            output = mock_stderr.getvalue()
            # Should use \r for same-line update (no newline)
            assert "\r" in output
            assert output.count("\n") == 0

    def test_update_calculates_percentage(self) -> None:
        """update() should calculate and display correct percentage."""
        reporter = ProgressReporter(total_frames=200)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=100, detected_count=0)
            output = mock_stderr.getvalue()
            # 100/200 = 50%
            assert "50" in output

    def test_update_handles_100_percent(self) -> None:
        """update() should handle 100% progress correctly."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=100, detected_count=10)
            output = mock_stderr.getvalue()
            assert "100" in output


class TestProgressReporterComplete:
    """Tests for ProgressReporter.complete() method."""

    def test_complete_outputs_summary(self) -> None:
        """complete() should output completion summary."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.complete(detected_count=15, output_dir="./output")
            output = mock_stderr.getvalue()
            # Should contain detected count and output directory
            assert "15" in output
            assert "output" in output

    def test_complete_outputs_newline(self) -> None:
        """complete() should output on new line."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.complete(detected_count=15, output_dir="./output")
            output = mock_stderr.getvalue()
            # Should end with newline
            assert output.endswith("\n")

    def test_complete_clears_progress_line(self) -> None:
        """complete() should clear/move past progress line first."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.complete(detected_count=5, output_dir="./frames")
            output = mock_stderr.getvalue()
            # Should start with newline to clear progress line
            assert output.startswith("\n")


class TestProgressReporterError:
    """Tests for ProgressReporter.error() method."""

    def test_error_outputs_message(self) -> None:
        """error() should output error message."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.error("Something went wrong")
            output = mock_stderr.getvalue()
            assert "Something went wrong" in output

    def test_error_outputs_on_new_line(self) -> None:
        """error() should output on new line."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.error("Test error")
            output = mock_stderr.getvalue()
            # Should start with newline to clear progress line
            assert output.startswith("\n")
            # Should end with newline
            assert output.endswith("\n")


class TestProgressReporterIntegration:
    """Integration tests for ProgressReporter workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: updates followed by completion."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=25, detected_count=2)
            reporter.update(current_frame=50, detected_count=5)
            reporter.update(current_frame=75, detected_count=8)
            reporter.update(current_frame=100, detected_count=10)
            reporter.complete(detected_count=10, output_dir="./output")

            output = mock_stderr.getvalue()
            # Should have final completion message
            assert "10" in output
            assert "output" in output

    def test_workflow_with_error(self) -> None:
        """Test workflow that ends with error."""
        reporter = ProgressReporter(total_frames=100)
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            reporter.update(current_frame=50, detected_count=5)
            reporter.error("Processing failed")

            output = mock_stderr.getvalue()
            assert "Processing failed" in output
