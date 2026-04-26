"""Crop U point mass rollout plots for thesis figures.

Extracts the rollouts axes region from visualize/u_point_mass/rollouts_step_*.png,
crops to the same U obstacle + goal area as the learned-value thesis plots
(see ``u_point_vis_util.py``), strips the spine border, adds a time label,
and saves clean panels to ``benchmark_data/thesis_plots``.

Usage:
    python benchmark/u_point_rollouts_vis_util.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────

VIS_DIR = Path(__file__).resolve().parents[1] / "visualize" / "u_point_mass"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_data" / "thesis_plots"

# Same time stamps as the learned-value thesis panels (u_point_vis_util.py)
STEPS = [0, 200, 400, 600, 640, 660, 680, 720]

# Frequency (steps -> seconds)
FREQUENCY = 50.0  # Hz

# Original axes data extent (visualize_rollouts xlim/ylim = (-1, 1))
DATA_EXTENT = (-1.0, 1.0, -1.0, 1.0)

# Crop region in data coordinates — matches u_point_vis_util.py learned-value crops
CROP_XLIM = (-0.32, 0.32)
CROP_YLIM = (-0.28, 0.88)


def _find_axes_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """Detect axes spine bounding box (px0, py0, px1, py1) in pixel coords.

    Spine rows/columns are recognised as those whose dark-pixel count spans
    a large fraction of the image — the trajectories don't span half the
    width/height, so this isolates the four spine lines cleanly.
    """
    if img.ndim == 3:
        gray = img[..., :3].mean(axis=-1)
    else:
        gray = img
    thresh = 0.3 if gray.max() <= 1.0 else 80
    dark = gray < thresh
    h, w = dark.shape
    dark_rows = np.where(dark.sum(axis=1) > 0.5 * w)[0]
    dark_cols = np.where(dark.sum(axis=0) > 0.5 * h)[0]
    if dark_rows.size == 0 or dark_cols.size == 0:
        raise RuntimeError("Could not detect axes spine.")
    return (
        int(dark_cols.min()),
        int(dark_rows.min()),
        int(dark_cols.max()),
        int(dark_rows.max()),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step_dt = 1.0 / FREQUENCY

    for step in STEPS:
        src = VIS_DIR / f"rollouts_step_{step:05d}.png"
        if not src.exists():
            print(f"WARNING: {src} not found, skipping step {step}")
            continue

        img = plt.imread(str(src))
        px0, py0, px1, py1 = _find_axes_box(img)
        axes_img = img[py0 : py1 + 1, px0 : px1 + 1]
        ah, aw = axes_img.shape[:2]

        # Pixel crop within the detected axes region for the data crop window.
        dx0, dx1, dy0, dy1 = DATA_EXTENT
        cx0, cx1 = CROP_XLIM
        cy0, cy1 = CROP_YLIM

        col0 = int((cx0 - dx0) / (dx1 - dx0) * aw)
        col1 = int((cx1 - dx0) / (dx1 - dx0) * aw)
        row0 = int((1 - (cy1 - dy0) / (dy1 - dy0)) * ah)  # y flipped
        row1 = int((1 - (cy0 - dy0) / (dy1 - dy0)) * ah)

        col0 = max(0, col0)
        col1 = min(aw, col1)
        row0 = max(0, row0)
        row1 = min(ah, row1)

        cropped = axes_img[row0:row1, col0:col1]

        fig, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(cropped, extent=[cx0, cx1, cy0, cy1], origin="upper", aspect="equal")

        ax.set_xlim(*CROP_XLIM)
        ax.set_ylim(*CROP_YLIM)
        ax.set_axis_off()

        t_sec = step * step_dt
        ax.text(
            0.5, 0.03, f"t = {t_sec:.1f}s",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=14, fontfamily="serif",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

        fig.tight_layout(pad=0)
        out_path = OUT_DIR / f"u_point_mass_rollouts_t{t_sec:.1f}s.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone — {OUT_DIR}")


if __name__ == "__main__":
    main()
