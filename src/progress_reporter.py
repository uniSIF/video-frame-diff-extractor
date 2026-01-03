"""Progress reporting for video frame processing."""

from __future__ import annotations

import sys


class ProgressReporter:
    """Report processing progress to stderr."""

    def __init__(self, total_frames: int) -> None:
        """
        Initialize progress reporter.

        Args:
            total_frames: Total number of frames to process
        """
        self._total_frames = total_frames

    def update(self, current_frame: int, detected_count: int) -> None:
        """
        Update and display progress on the same line.

        Args:
            current_frame: Current frame number being processed
            detected_count: Number of frames detected so far
        """
        if self._total_frames > 0:
            percentage = (current_frame / self._total_frames) * 100
        else:
            percentage = 100.0

        message = (
            f"\r処理中: {current_frame}/{self._total_frames} "
            f"({percentage:.1f}%) | 検出: {detected_count}フレーム"
        )
        sys.stderr.write(message)
        sys.stderr.flush()

    def complete(self, detected_count: int, output_dir: str) -> None:
        """
        Display completion message.

        Args:
            detected_count: Total number of detected frames
            output_dir: Output directory path
        """
        message = (
            f"\n完了: {detected_count}フレームを検出しました。"
            f" 出力先: {output_dir}\n"
        )
        sys.stderr.write(message)
        sys.stderr.flush()

    def error(self, message: str) -> None:
        """
        Display error message.

        Args:
            message: Error message to display
        """
        sys.stderr.write(f"\nエラー: {message}\n")
        sys.stderr.flush()
