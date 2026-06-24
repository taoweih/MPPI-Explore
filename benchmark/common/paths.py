"""Benchmark output path helpers.

Local runs write under ``benchmark/<category>/data``.  Cluster runs can set
``MPPI_BENCH_OUTPUT_ROOT`` or pass ``--output-root`` to keep generated data on
scratch storage without changing source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


OUTPUT_ROOT_ENV = "MPPI_BENCH_OUTPUT_ROOT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _override_root(output_root: Optional[str | Path] = None) -> Optional[Path]:
    value = output_root if output_root is not None else os.environ.get(OUTPUT_ROOT_ENV)
    if value is None or str(value) == "":
        return None
    return Path(value).expanduser().resolve()


def category_data_dir(category: str, output_root: Optional[str | Path] = None) -> Path:
    root = _override_root(output_root)
    if root is not None:
        return root / category
    return repo_root() / "benchmark" / category / "data"


def runs_dir(category: str, output_root: Optional[str | Path] = None) -> Path:
    path = category_data_dir(category, output_root) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir(category: str = "senior_thesis", output_root: Optional[str | Path] = None) -> Path:
    path = category_data_dir(category, output_root) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def senior_thesis_weights_dir(output_root: Optional[str | Path] = None) -> Path:
    path = category_data_dir("senior_thesis", output_root) / "saved_pretrain_weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archived_visualize_dir(task_name: str) -> Path:
    return repo_root() / "archive" / "visualize" / task_name

