"""Command-line interface for video frame diff extractor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Config:
    """Configuration object for video frame extraction."""

    input_file: str
    output_dir: str
    threshold: float
    crop_image: Optional[str]
    scan_area: Optional[Tuple[int, int, int, int]]


def _parse_scan_area(value: str) -> Tuple[int, int, int, int]:
    """Parse scan area string in x,y,width,height format."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"scan-area must be in x,y,width,height format, got: {value}"
        )
    try:
        return tuple(int(p) for p in parts)  # type: ignore
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"scan-area values must be integers, got: {value}"
        )


def parse_args(args: Optional[List[str]] = None) -> Config:
    """
    Parse command-line arguments and return a Config object.

    Args:
        args: List of arguments (defaults to sys.argv[1:] if None)

    Returns:
        Config object with parsed values

    Raises:
        SystemExit: On argument error or --help
    """
    parser = argparse.ArgumentParser(
        prog="frame-extractor",
        description="映像ファイルからフレーム差分を検出し、変化フレームを画像として抽出します。",
    )

    parser.add_argument(
        "input_file",
        help="入力映像ファイルのパス（MP4, AVI, MOV, MKV, WebM対応）",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default="./output",
        help="出力ディレクトリのパス（デフォルト: ./output）",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.05,
        help="差分検出閾値（0.0-1.0、デフォルト: 0.05）",
    )

    parser.add_argument(
        "--crop-image",
        help="クロップ判定条件画像のパス",
    )

    parser.add_argument(
        "--scan-area",
        type=_parse_scan_area,
        help="走査エリア（x,y,width,height形式）",
    )

    parsed = parser.parse_args(args)

    return Config(
        input_file=parsed.input_file,
        output_dir=parsed.output_dir,
        threshold=parsed.threshold,
        crop_image=parsed.crop_image,
        scan_area=parsed.scan_area,
    )
