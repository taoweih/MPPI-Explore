"""Crop U point mass memory heatmaps for thesis figures.

Extracts the heatmap region from visualize/u_point_mass/memory_step_*.png,
crops to the U obstacle + goal area, strips axes/colorbar/title, adds a
time label, and saves clean panels to thesis_plots.

Usage:
    python benchmark/u_point_vis_util.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────

VIS_DIR = Path(__file__).resolve().parents[1] / "visualize" / "u_point_mass"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_data" / "thesis_plots"

# Steps to extract (0 to 750, 4 evenly spaced)
STEPS = [0, 250, 500, 750]

# Frequency (steps -> seconds)
FREQUENCY = 50.0  # Hz

# Original heatmap data extent (from visualize_memory: extent=[-1, 1, -1, 1])
DATA_EXTENT = (-1.0, 1.0, -1.0, 1.0)  # (x0, x1, y0, y1)

# Original figure layout (from visualize_memory: figsize=(8,7), dpi=150)
# Axes bounding box in figure-fraction coords (computed from matplotlib layout)
AX_FRAC = (0.1149, 0.1276, 0.7932, 0.9028)  # (x0, y0, x1, y1)

# Crop region in data coordinates — covers U obstacle + goal with margin
CROP_XLIM = (-0.32, 0.32)
CROP_YLIM = (-0.28, 0.88)

# U obstacle geometry from scene.xml
U_WALLS = [
    ((-0.2, 0.19), 0.4, 0.02),   # front wall
    ((0.20, -0.19), 0.02, 0.4),  # right wall
    ((-0.22, -0.19), 0.02, 0.4), # left wall
]

# Goal from scene.xml
GOAL_XY = (0.025, 0.775)
GOAL_RADIUS = 0.01

# Point mass radius from point_mass.xml
PM_RADIUS = 0.01


def _extract_heatmap(img: np.ndarray) -> np.ndarray:
    """Extract just the heatmap axes region from the full figure image."""
    h, w = img.shape[:2]
    x0f, y0f, x1f, y1f = AX_FRAC
    # Image y is flipped (row 0 = top)
    px0 = int(x0f * w)
    px1 = int(x1f * w)
    py0 = int((1 - y1f) * h)
    py1 = int((1 - y0f) * h)
    return img[py0:py1, px0:px1]


def _data_to_pixel(heatmap_h: int, heatmap_w: int,
                   data_x: float, data_y: float) -> tuple[int, int]:
    """Convert data coordinates to pixel coordinates in the cropped heatmap."""
    dx0, dx1, dy0, dy1 = DATA_EXTENT
    px = int((data_x - dx0) / (dx1 - dx0) * heatmap_w)
    # Image y is flipped
    py = int((1 - (data_y - dy0) / (dy1 - dy0)) * heatmap_h)
    return px, py


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step_dt = 1.0 / FREQUENCY

    for step in STEPS:
        src = VIS_DIR / f"memory_step_{step:05d}.png"
        if not src.exists():
            print(f"WARNING: {src} not found, skipping step {step}")
            continue

        # Load and extract heatmap region
        img = plt.imread(str(src))
        heatmap = _extract_heatmap(img)
        hm_h, hm_w = heatmap.shape[:2]

        # Compute pixel crop for the data region
        dx0, dx1, dy0, dy1 = DATA_EXTENT
        cx0, cx1 = CROP_XLIM
        cy0, cy1 = CROP_YLIM

        col0 = int((cx0 - dx0) / (dx1 - dx0) * hm_w)
        col1 = int((cx1 - dx0) / (dx1 - dx0) * hm_w)
        row0 = int((1 - (cy1 - dy0) / (dy1 - dy0)) * hm_h)  # y flipped
        row1 = int((1 - (cy0 - dy0) / (dy1 - dy0)) * hm_h)

        col0 = max(0, col0)
        col1 = min(hm_w, col1)
        row0 = max(0, row0)
        row1 = min(hm_h, row1)

        cropped = heatmap[row0:row1, col0:col1]

        # Plot clean figure
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(cropped, extent=[cx0, cx1, cy0, cy1], origin="upper", aspect="equal")

        ax.set_xlim(*CROP_XLIM)
        ax.set_ylim(*CROP_YLIM)
        ax.set_axis_off()

        # Time label
        t_sec = step * step_dt
        ax.text(
            0.5, 0.03, f"t = {t_sec:.0f}s",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=14, fontfamily="serif",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

        fig.tight_layout(pad=0)
        out_path = OUT_DIR / f"u_point_mass_memory_t{t_sec:.0f}s.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone — {OUT_DIR}")


if __name__ == "__main__":
    main()
