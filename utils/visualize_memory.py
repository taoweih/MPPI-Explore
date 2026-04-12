"""Visualization of the hash-grid value function as 2D heatmaps and 3D scatter."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def visualize_memory(
    controller,
    step: int,
    *,
    resolution: int = 500,
    output_dir: str | Path,
    plot_overlay: Optional[Callable[[plt.Axes], None]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Render the controller's value memory as a 2D heatmap and save to disk.

    Parameters
    ----------
    controller:
        An ``MPPIMemoryContinuous`` instance (must have ``.memory``).
    step:
        Current simulation step (used in filename and title).
    resolution:
        Number of grid points per axis.
    output_dir:
        Directory to save figures.
    plot_overlay:
        Optional callback ``fn(ax)`` that draws task-specific overlays
        (obstacles, goals, current position, etc.) on the axes.
    vmin, vmax:
        Color scale limits for the heatmap.
    xlim, ylim:
        Optional (min, max) tuples to crop the view. Defaults to full grid.
    """
    memory = controller.memory
    x0, x1 = xlim if xlim is not None else (memory.grid_min, memory.grid_max)
    y0, y1 = ylim if ylim is not None else (memory.grid_min, memory.grid_max)

    xs = np.linspace(x0, x1, resolution, dtype=np.float32)
    ys = np.linspace(y0, y1, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    states = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (res*res, 2)

    values = memory.predict(states).reshape(resolution, resolution)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        values,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Value")

    if plot_overlay is not None:
        plot_overlay(ax)

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_title(f"Value Memory — step {step}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"memory_step_{step:05d}.png", dpi=150)
    plt.close(fig)


def visualize_memory_3d(
    controller,
    step: int,
    *,
    slice_dim: int = 2,
    slice_value: float = 0.0,
    resolution: int = 500,
    output_dir: str | Path,
    plot_overlay: Optional[Callable[[plt.Axes], None]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Render a 2D slice of a 3D value memory as a heatmap.

    Parameters
    ----------
    slice_dim:
        Which dimension to hold fixed (0=X, 1=Y, 2=Z).
    slice_value:
        The value at which to slice the held-fixed dimension.
    """
    memory = controller.memory
    grid_min = memory.grid_min
    grid_max = memory.grid_max

    # The two free axes
    free_dims = [d for d in range(3) if d != slice_dim]
    axis_labels = ["X", "Y", "Z"]

    coords = np.linspace(grid_min, grid_max, resolution, dtype=np.float32)
    aa, bb = np.meshgrid(coords, coords)
    flat_a = aa.ravel()
    flat_b = bb.ravel()

    states = np.empty((resolution * resolution, 3), dtype=np.float32)
    states[:, free_dims[0]] = flat_a
    states[:, free_dims[1]] = flat_b
    states[:, slice_dim] = slice_value

    values = memory.predict(states).reshape(resolution, resolution)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        values,
        origin="lower",
        extent=[grid_min, grid_max, grid_min, grid_max],
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Value")

    if plot_overlay is not None:
        plot_overlay(ax)

    ax.set_xlabel(axis_labels[free_dims[0]], fontsize=14)
    ax.set_ylabel(axis_labels[free_dims[1]], fontsize=14)
    ax.set_title(
        f"Value Memory — step {step} "
        f"({axis_labels[slice_dim]}={slice_value:.2f})"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"memory_step_{step:05d}.png", dpi=150)
    plt.close(fig)


# ── 3D volumetric scatter visualization ──────────────────────────────


def _draw_cylinder(
    ax,
    center: Tuple[float, float, float],
    radius: float,
    half_height: float,
    n_sides: int = 16,
    color: str = "gray",
    alpha: float = 0.3,
) -> None:
    """Draw a vertical cylinder as a polygonal prism."""
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    x_circle = cx + radius * np.cos(theta)
    y_circle = cy + radius * np.sin(theta)
    z_bot = cz - half_height
    z_top = cz + half_height

    for i in range(n_sides):
        verts = [
            [x_circle[i], y_circle[i], z_bot],
            [x_circle[i + 1], y_circle[i + 1], z_bot],
            [x_circle[i + 1], y_circle[i + 1], z_top],
            [x_circle[i], y_circle[i], z_top],
        ]
        ax.add_collection3d(
            Poly3DCollection([verts], alpha=alpha, facecolor=color, edgecolor="none")
        )

    top_cap = [[x_circle[i], y_circle[i], z_top] for i in range(n_sides)]
    bot_cap = [[x_circle[i], y_circle[i], z_bot] for i in range(n_sides)]
    ax.add_collection3d(Poly3DCollection([top_cap], alpha=alpha, facecolor=color, edgecolor="none"))
    ax.add_collection3d(Poly3DCollection([bot_cap], alpha=alpha, facecolor=color, edgecolor="none"))


def visualize_memory_3d_scatter(
    controller,
    step: int,
    *,
    resolution: int = 30,
    output_dir: str | Path,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    goal_xyz: Optional[np.ndarray] = None,
    ee_xyz: Optional[np.ndarray] = None,
    cylinders: Optional[Sequence[Tuple[Tuple[float, float, float], float, float]]] = None,
    elev: float = 25.0,
    azim: float = -60.0,
    point_size: float = 8.0,
    alpha: float = 0.4,
    value_threshold: Optional[float] = None,
) -> None:
    """Render the 3D value memory as a volumetric scatter plot.

    Parameters
    ----------
    controller:
        An ``MPPIMemoryContinuous`` instance.
    step:
        Current simulation step.
    resolution:
        Number of grid points per axis (total points = resolution^3).
        Keep low (20-40) for readable plots.
    output_dir:
        Directory to save figures.
    xlim, ylim, zlim:
        Axis bounds. Defaults to grid bounds.
    vmin, vmax:
        Color scale limits.
    goal_xyz:
        Goal position (3,) to mark with a red star.
    ee_xyz:
        Current end-effector position (3,) to mark with a green dot.
    cylinders:
        List of ((cx, cy, cz), radius, half_height) obstacle cylinders to draw.
    elev, azim:
        Camera elevation and azimuth angles.
    point_size:
        Scatter point size.
    alpha:
        Scatter point transparency.
    value_threshold:
        If set, only show points with value below this threshold.
        Helps declutter the plot by hiding high-cost regions.
    """
    memory = controller.memory
    x0, x1 = xlim if xlim is not None else (memory.grid_min, memory.grid_max)
    y0, y1 = ylim if ylim is not None else (memory.grid_min, memory.grid_max)
    z0, z1 = zlim if zlim is not None else (memory.grid_min, memory.grid_max)

    xs = np.linspace(x0, x1, resolution, dtype=np.float32)
    ys = np.linspace(y0, y1, resolution, dtype=np.float32)
    zs = np.linspace(z0, z1, resolution, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")

    states = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    values = memory.predict(states)
    coords = states

    # Filter by threshold to reduce clutter
    if value_threshold is not None:
        mask = values < value_threshold
        coords = coords[mask]
        values = values[mask]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    if len(values) > 0:
        v_lo = vmin if vmin is not None else float(values.min())
        v_hi = vmax if vmax is not None else float(values.max())
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=values, cmap="Blues_r", vmin=v_lo, vmax=v_hi,
            s=point_size, alpha=alpha, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="Value", shrink=0.6, pad=0.1)

    # Draw obstacles
    if cylinders is not None:
        for ctr, r, hh in cylinders:
            _draw_cylinder(ax, ctr, r, hh)

    # Draw goal
    if goal_xyz is not None:
        ax.scatter(
            goal_xyz[0], goal_xyz[1], goal_xyz[2],
            color="red", s=200, marker="*", label="Goal", zorder=5,
        )

    # Draw current EE
    if ee_xyz is not None:
        ax.scatter(
            ee_xyz[0], ee_xyz[1], ee_xyz[2],
            color="lime", s=120, marker="o", label="EE", zorder=5,
            edgecolors="black", linewidths=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Value Memory — step {step}", fontsize=14)

    if goal_xyz is not None or ee_xyz is not None:
        ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / f"memory_step_{step:05d}.png", dpi=150)
    plt.close(fig)
