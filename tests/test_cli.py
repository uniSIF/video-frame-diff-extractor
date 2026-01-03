"""Tests for CLI argument parsing."""

import pytest
from src.cli import Config, parse_args


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_required_fields(self) -> None:
        """Config should have input_file, output_dir, threshold fields."""
        config = Config(
            input_file="video.mp4",
            output_dir="./output",
            threshold=0.05,
            crop_image=None,
            scan_area=None,
        )
        assert config.input_file == "video.mp4"
        assert config.output_dir == "./output"
        assert config.threshold == 0.05

    def test_config_optional_crop_fields(self) -> None:
        """Config should have optional crop_image and scan_area fields."""
        config = Config(
            input_file="video.mp4",
            output_dir="./output",
            threshold=0.05,
            crop_image="condition.png",
            scan_area=(100, 200, 300, 400),
        )
        assert config.crop_image == "condition.png"
        assert config.scan_area == (100, 200, 300, 400)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_required_input_file(self) -> None:
        """parse_args should accept input file as positional argument."""
        config = parse_args(["video.mp4"])
        assert config.input_file == "video.mp4"

    def test_parse_output_dir_option(self) -> None:
        """parse_args should accept --output-dir option."""
        config = parse_args(["video.mp4", "--output-dir", "./frames"])
        assert config.output_dir == "./frames"

    def test_parse_default_output_dir(self) -> None:
        """parse_args should use default output directory when not specified."""
        config = parse_args(["video.mp4"])
        assert config.output_dir == "./output"

    def test_parse_threshold_option(self) -> None:
        """parse_args should accept --threshold option."""
        config = parse_args(["video.mp4", "--threshold", "0.1"])
        assert config.threshold == 0.1

    def test_parse_default_threshold(self) -> None:
        """parse_args should use default threshold 0.05 when not specified."""
        config = parse_args(["video.mp4"])
        assert config.threshold == 0.05

    def test_parse_crop_image_option(self) -> None:
        """parse_args should accept --crop-image option."""
        config = parse_args(["video.mp4", "--crop-image", "condition.png"])
        assert config.crop_image == "condition.png"

    def test_parse_scan_area_option(self) -> None:
        """parse_args should accept --scan-area option with x,y,width,height format."""
        config = parse_args(["video.mp4", "--scan-area", "100,200,300,400"])
        assert config.scan_area == (100, 200, 300, 400)

    def test_parse_no_crop_image_by_default(self) -> None:
        """parse_args should return None for crop_image when not specified."""
        config = parse_args(["video.mp4"])
        assert config.crop_image is None

    def test_parse_no_scan_area_by_default(self) -> None:
        """parse_args should return None for scan_area when not specified."""
        config = parse_args(["video.mp4"])
        assert config.scan_area is None

    def test_parse_missing_input_file_raises_error(self) -> None:
        """parse_args should raise SystemExit when input file is missing."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_help_option(self) -> None:
        """parse_args should raise SystemExit for --help option."""
        with pytest.raises(SystemExit):
            parse_args(["--help"])

    def test_parse_short_options(self) -> None:
        """parse_args should accept short options -o, -t."""
        config = parse_args(["video.mp4", "-o", "./frames", "-t", "0.2"])
        assert config.output_dir == "./frames"
        assert config.threshold == 0.2

    def test_parse_all_options_combined(self) -> None:
        """parse_args should correctly parse all options together."""
        config = parse_args([
            "video.mp4",
            "--output-dir", "./frames",
            "--threshold", "0.15",
            "--crop-image", "cond.png",
            "--scan-area", "10,20,100,200",
        ])
        assert config.input_file == "video.mp4"
        assert config.output_dir == "./frames"
        assert config.threshold == 0.15
        assert config.crop_image == "cond.png"
        assert config.scan_area == (10, 20, 100, 200)
