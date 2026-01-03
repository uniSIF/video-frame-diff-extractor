"""Main processing pipeline for video frame diff extraction."""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import cv2

from src.cli import parse_args
from src.crop_processor import CropProcessor
from src.diff_detector import DiffDetector
from src.image_writer import ImageWriter
from src.progress_reporter import ProgressReporter
from src.video_loader import VideoLoader


def process_video(
    input_file: str,
    output_dir: str,
    threshold: float,
    crop_image: Optional[str],
    scan_area: Optional[Tuple[int, int, int, int]],
) -> int:
    """
    Process video file to extract frames with detected differences.

    Args:
        input_file: Path to the input video file.
        output_dir: Directory to save extracted images.
        threshold: Difference detection threshold (0.0-1.0).
        crop_image: Optional path to condition image for cropping.
        scan_area: Optional scan area (x, y, width, height) for cropping.

    Returns:
        Number of frames detected and saved.

    Raises:
        FileNotFoundError: If input file or crop image doesn't exist.
        ValueError: If input file format is not supported.
    """
    loader = VideoLoader(input_file)

    try:
        # Initialize components
        detector = DiffDetector(threshold=threshold)
        writer = ImageWriter(output_dir)
        reporter = ProgressReporter(loader.frame_count)

        # Initialize crop processor if enabled
        crop_processor: Optional[CropProcessor] = None
        if crop_image is not None:
            condition_img = cv2.imread(crop_image)
            if condition_img is None:
                raise FileNotFoundError(
                    f"クロップ条件画像を読み込めませんでした: {crop_image}"
                )
            crop_processor = CropProcessor(
                condition_image=condition_img,
                scan_area=scan_area,
            )

        detected_count = 0
        prev_frame = None

        for frame_num, frame in loader.frames():
            if prev_frame is not None:
                if detector.detect(prev_frame, frame):
                    # Apply cropping if enabled
                    output_image = frame
                    if crop_processor is not None:
                        crop_result = crop_processor.crop(frame)
                        output_image = crop_result.image
                        if crop_result.warning:
                            reporter.error(crop_result.warning)

                    # Save the image
                    writer.save(output_image, frame_num, loader.fps)
                    detected_count += 1

            prev_frame = frame
            reporter.update(frame_num + 1, detected_count)

        reporter.complete(detected_count, output_dir)
        return detected_count

    finally:
        loader.close()


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:] if None).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        config = parse_args(args)
        process_video(
            input_file=config.input_file,
            output_dir=config.output_dir,
            threshold=config.threshold,
            crop_image=config.crop_image,
            scan_area=config.scan_area,
        )
        return 0

    except FileNotFoundError as e:
        sys.stderr.write(f"\nエラー: {e}\n")
        return 1

    except ValueError as e:
        sys.stderr.write(f"\nエラー: {e}\n")
        return 1

    except KeyboardInterrupt:
        sys.stderr.write("\n処理が中断されました。\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
