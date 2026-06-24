"""Common command-line helpers for benchmark scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_output_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root for generated benchmark outputs. Overrides "
            "MPPI_BENCH_OUTPUT_ROOT and local data defaults."
        ),
    )


def add_sweep_config_args(
    parser: argparse.ArgumentParser,
    *,
    parallel: bool = True,
    freq_calibration: bool = True,
    record_video: bool = True,
) -> None:
    add_output_root_arg(parser)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)

    if parallel:
        parser.add_argument(
            "--parallel",
            choices=("sequential", "controllers", "axis", "all"),
            default=None,
        )
        parser.add_argument("--max-workers", default=None)
        parser.add_argument("--num-gpus", type=int, default=None)

    if freq_calibration:
        parser.add_argument("--freq-calibration-iters", type=int, default=None)

    if record_video:
        parser.add_argument(
            "--record-video",
            dest="record_video",
            action="store_true",
            default=None,
        )
        parser.add_argument("--no-record-video", dest="record_video", action="store_false")


def apply_sweep_config_overrides(config, args: argparse.Namespace) -> None:
    for attr in (
        "num_trials",
        "max_iterations",
        "parallel",
        "max_workers",
        "num_gpus",
        "freq_calibration_iters",
        "record_video",
    ):
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                setattr(config, attr, value)

    if hasattr(args, "output_root") and args.output_root is not None:
        config.output_root = str(args.output_root)
