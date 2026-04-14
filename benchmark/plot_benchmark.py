"""Re-plot benchmark CSV data for thesis/paper figures.

Usage:
    python benchmark/plot_benchmark.py <benchmark_dir> [options]

Examples:
    # All controllers, all metrics, both sweeps (default)
    python benchmark/plot_benchmark.py benchmark/benchmark_data/u_point_mass_benchmark_thesis_2026_04_12_20_29_25

    # Only MPPI and Density, only horizon sweep, only frequency
    python benchmark/plot_benchmark.py benchmark/benchmark_data/u_point_mass_benchmark_thesis_2026_04_12_20_29_25 \
        --controllers "MPPI" "Density-Guided MPPI" --sweep horizon --metrics frequency

    # Multiple metrics
    python benchmark/plot_benchmark.py <dir> --metrics success_rate frequency

Output: benchmark/benchmark_data/thesis_plots/<task>_<ctrl_tag>_<sweep>_<metric>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

THESIS_PLOTS_DIR = Path(__file__).resolve().parent / "benchmark_data" / "thesis_plots"

# ── Thesis style ──────────────────────────────────────────────────────────
# Matplotlib tab10 — standard in robotics/ML papers
COLORS = [
    "#1F77B4",  # blue
    "#FF7F0E",  # orange
    "#2CA02C",  # green
    "#D62728",  # red
    "#9467BD",  # purple
    "#8C564B",  # brown
]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]

TASK_DISPLAY_NAMES = {
    "u_point_mass": "U-Shaped Point Mass",
    "ur5e": "UR5e Reach",
    "ant": "Ant Locomotion",
}

# Short task prefix for filenames
TASK_SHORT_NAMES = {
    "u_point_mass": "u",
    "ur5e": "ur5e",
    "ant": "ant",
}

# Short controller tag for filenames — maps a frozenset of selected names
# to a compact label.  "complete" = all controllers included.
CONTROLLER_FILE_TAGS = {
    frozenset(["MPPI"]): "mppi",
    frozenset(["MPPI", "Density-Guided MPPI"]): "density",
    frozenset(["MPPI", "Value-Guided MPPI"]): "value",
    frozenset(["MPPI", "Density-Guided MPPI", "Value-Guided MPPI"]): "density_value",
    frozenset(["MPPI", "Density-Guided MPPI", "Value-Guided MPPI", "Density + Value-Guided MPPI"]): "complete",
}

METRIC_FILE_NAMES = {
    "success_rate": "success_rate",
    "success_time": "success_time",
    "frequency": "control_freq",
}

CONTROLLER_DISPLAY_NAMES = {
    "MPPI": "MPPI",
    "Density-Guided MPPI": "Density-Guided MPPI",
    "Value-Guided MPPI": "Value-Guided MPPI",
    "Density + Value-Guided MPPI": "Density + Value-Guided MPPI",
}


def _setup_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        # Axes
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Legend
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ── Data loading ──────────────────────────────────────────────────────────

def load_benchmark(bench_dir: Path) -> dict:
    with open(bench_dir / "metadata.json") as f:
        meta = json.load(f)
    csvs = {}
    for csv_file in bench_dir.glob("*.csv"):
        csvs[csv_file.stem] = np.loadtxt(csv_file, delimiter=",")
    return {
        "meta": meta,
        "controller_names": meta["controller_names"],
        "horizons": np.array(meta["horizons"]),
        "num_samples_list": np.array(meta["num_samples_list"]),
        "csvs": csvs,
    }


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_matrix(
    x_values: np.ndarray,
    values: np.ndarray,
    controller_names: list[str],
    ctrl_indices: list[int],
    *,
    std: np.ndarray | None = None,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    skip_zeros: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for ci, idx in enumerate(ctrl_indices):
        color = COLORS[ci % len(COLORS)]
        marker = MARKERS[ci % len(MARKERS)]
        ls = LINESTYLES[ci % len(LINESTYLES)]
        display = CONTROLLER_DISPLAY_NAMES.get(controller_names[idx], controller_names[idx])

        if skip_zeros:
            mask = values[idx] != 0
            if not np.any(mask):
                continue
            ax.plot(
                x_values[mask], values[idx][mask],
                marker=marker, color=color, linestyle=ls,
                label=display,
            )
            if std is not None:
                ax.fill_between(
                    x_values[mask],
                    values[idx][mask] - std[idx][mask],
                    values[idx][mask] + std[idx][mask],
                    alpha=0.15, color=color, linewidth=0,
                )
        else:
            ax.plot(
                x_values, values[idx],
                marker=marker, color=color, linestyle=ls,
                label=display,
            )
            if std is not None:
                ax.fill_between(
                    x_values,
                    values[idx] - std[idx],
                    values[idx] + std[idx],
                    alpha=0.15, color=color, linewidth=0,
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Title omitted — use LaTeX figure caption instead
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Sweep / metric definitions ────────────────────────────────────────────

SWEEPS = {
    "horizon": {
        "x_key": "horizons",
        "axis_label": "Horizon (s)",
        "metrics": [
            {
                "csv": "horizon_success_rate",
                "std_csv": None,
                "ylabel": "Success Rate (%)",
                "title_fmt": "Success Rate vs Horizon",
                "fname_metric": "success_rate",
                "skip_zeros": False,
            },
            {
                "csv": "horizon_success_time_mean",
                "std_csv": "horizon_success_time_std",
                "ylabel": "Time to Success (s)",
                "title_fmt": "Time to Success vs Horizon",
                "fname_metric": "success_time",
                "skip_zeros": True,
            },
            {
                "csv": "horizon_frequency_mean",
                "std_csv": "horizon_frequency_std",
                "ylabel": "Control Frequency (Hz)",
                "title_fmt": "Control Frequency vs Horizon",
                "fname_metric": "frequency",
                "skip_zeros": False,
            },
        ],
    },
    "samples": {
        "x_key": "num_samples_list",
        "axis_label": "Number of Samples",
        "metrics": [
            {
                "csv": "num_samples_success_rate",
                "std_csv": None,
                "ylabel": "Success Rate (%)",
                "title_fmt": "Success Rate vs Number of Samples",
                "fname_metric": "success_rate",
                "skip_zeros": False,
            },
            {
                "csv": "num_samples_success_time_mean",
                "std_csv": "num_samples_success_time_std",
                "ylabel": "Time to Success (s)",
                "title_fmt": "Time to Success vs Number of Samples",
                "fname_metric": "success_time",
                "skip_zeros": True,
            },
            {
                "csv": "num_samples_frequency_mean",
                "std_csv": "num_samples_frequency_std",
                "ylabel": "Control Frequency (Hz)",
                "title_fmt": "Control Frequency vs Number of Samples",
                "fname_metric": "frequency",
                "skip_zeros": False,
            },
        ],
    },
}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results (thesis quality)")
    parser.add_argument("bench_dir", type=Path, help="Benchmark data directory")
    parser.add_argument(
        "--controllers", nargs="+", default=None,
        help='Controller names to plot (default: all). e.g. "MPPI" "Density-Guided MPPI"',
    )
    parser.add_argument(
        "--sweep", choices=["horizon", "samples", "both"], default="both",
        help="Which sweep to plot (default: both)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory (default: {THESIS_PLOTS_DIR})",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        choices=["success_rate", "success_time", "frequency"],
        help="Which metrics to plot (default: all). e.g. success_rate frequency",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_controllers",
        help="List available controllers and exit",
    )
    args = parser.parse_args()

    _setup_style()
    bench = load_benchmark(args.bench_dir)

    if args.list_controllers:
        print("Available controllers:")
        for name in bench["controller_names"]:
            print(f"  - {name}")
        sys.exit(0)

    selected = args.controllers if args.controllers else bench["controller_names"]
    missing = [s for s in selected if s not in bench["controller_names"]]
    if missing:
        print(f"ERROR: controllers not found: {missing}")
        print(f"Available: {bench['controller_names']}")
        sys.exit(1)

    ctrl_indices = [bench["controller_names"].index(n) for n in selected]
    out_dir = args.output_dir if args.output_dir else THESIS_PLOTS_DIR
    task_name = bench["meta"]["task_name"]
    display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
    display_selected = [CONTROLLER_DISPLAY_NAMES.get(n, n) for n in selected]
    ctrl_tag = ", ".join(display_selected)

    # Build compact filename prefix
    task_short = TASK_SHORT_NAMES.get(task_name, task_name)
    ctrl_set = frozenset(selected)
    ctrl_file_tag = CONTROLLER_FILE_TAGS.get(ctrl_set, "_".join(
        s.lower().replace(" ", "_") for s in selected
    ))

    sweep_keys = ["horizon", "samples"] if args.sweep == "both" else [args.sweep]

    print(f"Task: {display_name}")
    print(f"Controllers: {ctrl_tag}")
    print()

    selected_metrics = args.metrics  # None means all

    for sweep_key in sweep_keys:
        sweep = SWEEPS[sweep_key]
        x_values = bench[sweep["x_key"]]

        for m in sweep["metrics"]:
            if selected_metrics and m["fname_metric"] not in selected_metrics:
                continue
            data = bench["csvs"].get(m["csv"])
            if data is None:
                continue
            std_data = bench["csvs"].get(m["std_csv"]) if m["std_csv"] else None

            metric_file = METRIC_FILE_NAMES.get(m["fname_metric"], m["fname_metric"])
            fname = f"{task_short}_{ctrl_file_tag}_{sweep_key}_{metric_file}.png"
            plot_matrix(
                x_values, data, bench["controller_names"], ctrl_indices,
                std=std_data,
                xlabel=sweep["axis_label"],
                ylabel=m["ylabel"],
                title=m["title_fmt"],
                out_path=out_dir / fname,
                skip_zeros=m["skip_zeros"],
            )

    print(f"\nDone — {out_dir}")


if __name__ == "__main__":
    main()

