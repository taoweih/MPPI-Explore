"""Ant locomotion example using mujoco_warp MPPI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patches as patches
import mujoco
import numpy as np

from algs import (
    MPPI,
    DensityGuidedMPPI,
    ValueGuidedMPPI,
    ValuePretrainConfig,
)
from simulation.deterministic import run_interactive
from tasks.ant import Ant
from utils.visualize_learned_value import visualize_learned_value, visualize_rollouts

WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "senior_thesis_benchmarks"
    / "saved_pretrain_weights"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "controller",
        nargs="?",
        default="mppi",
        choices=("mppi", "density", "learned_value", "density_learned_value"),
    )
    args = parser.parse_args()

    # ── Shared MPPI parameters ──────────────────────────────────────────
    num_samples = 1024
    noise_level = 0.3
    temperature = 0.00001
    plan_horizon = 2.0
    spline_type = "zero"
    num_knots = 16
    iterations = 1
    seed = 0

    # ── Density-guided parameters ────────────────────────────────────────
    num_knots_per_stage = 4
    kde_bandwidth = 0.10
    inverse_density_power = 0.5

    # ── Learned value parameters ───────────────────────────────────────
    value_grid_min = -10.0
    value_grid_max = 10.0
    hashgrid_num_levels = 8
    hashgrid_table_size = 4096
    hashgrid_min_resolution = 40.0
    hashgrid_max_resolution = 80.0
    online_learning_rate = 1e-3
    online_update_epochs = 1
    online_batch_size = 2
    online_anchor_samples = 0
    online_new_state_weight = 1.0
    goal_value = 0.0
    goal_weight = 2000.0
    weights_key = "ant_learned_value"
    pretrain_mode = "load"  # "load" or "train"

    # ── Pretraining parameters ──────────────────────────────────────────
    pretrain_sample_count = 100000
    pretrain_epochs = 101
    pretrain_batch_size = 512
    pretrain_learning_rate = 1e-2
    pretrain_print_every = 50
    pretrain_target_scale = 1.0

    # ── Visualization parameters ───────────────────────────────────────
    visualize = False
    visualize_every = 500

    # ── Simulation parameters ───────────────────────────────────────────
    frequency = 50.0
    max_steps = 5001
    show_traces = False
    record_video = False

    # ── Video quality (only used when record_video=True) ────────────────
    video_width = 1080
    video_height = 1080
    video_crf = 18       # 0 = lossless, 18 ≈ visually lossless, 23 = default H.264, 28 = small file
    video_preset = "slow"  # ultrafast, fast, medium, slow, veryslow

    # ── Camera parameters (third-person, behind ant looking toward goal) ─
    camera_lookat = (3.354, 2.566, -0.515)
    camera_distance = 16.108
    camera_azimuth = 45.0
    camera_elevation = -49.3

    # ── Screenshot parameters ──────────────────────────────────────────
    take_screenshot = False
    screenshot_path = str(
        Path(__file__).resolve().parents[1] / "visualize" / "ant" / "ant.png"
    )
    screenshot_dpi = 600
    screenshot_step = 20  # delay capture so the ant settles onto the ground
    screenshot_every = 50  # also capture every N steps as <task>_step_<N>.png

    # ── Task setup ──────────────────────────────────────────────────────
    task = Ant()
    mj_model = task.mj_model
    mj_model.opt.timestep = task.sim_dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Goal position (needed for learned value variants)
    _tmp_data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, _tmp_data)
    goal_xy = np.asarray(_tmp_data.xpos[task.goal_pos_id, :2], dtype=np.float32)

    # ── Build controller ────────────────────────────────────────────────
    shared = dict(
        task=task,
        num_samples=num_samples,
        noise_level=noise_level,
        temperature=temperature,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        iterations=iterations,
        seed=seed,
    )

    if args.controller == "density":
        controller = DensityGuidedMPPI(
            **shared,
            num_knots_per_stage=num_knots_per_stage,
            kde_bandwidth=kde_bandwidth,
            inverse_density_power=inverse_density_power,
        )

    elif args.controller in ("learned_value", "density_learned_value"):
        use_density = args.controller == "density_learned_value"
        controller = ValueGuidedMPPI(
            **shared,
            value_grid_min=value_grid_min,
            value_grid_max=value_grid_max,
            hashgrid_num_levels=hashgrid_num_levels,
            hashgrid_table_size=hashgrid_table_size,
            hashgrid_min_resolution=hashgrid_min_resolution,
            hashgrid_max_resolution=hashgrid_max_resolution,
            online_learning_rate=online_learning_rate,
            online_update_epochs=online_update_epochs,
            online_batch_size=online_batch_size,
            online_anchor_samples=online_anchor_samples,
            online_new_state_weight=online_new_state_weight,
            goal_state=goal_xy[None, :],
            goal_value=goal_value,
            goal_weight=goal_weight,
            use_density_guided=use_density,
            num_knots_per_stage=num_knots_per_stage,
            kde_bandwidth=kde_bandwidth,
            inverse_density_power=inverse_density_power,
            disable_replay_anchors=True,
        )

        def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(value_grid_min, value_grid_max, size=(n, task.state_dim)).astype(np.float32)

        def target_function(states: np.ndarray) -> np.ndarray:
            diff = states - goal_xy[None, :]
            return pretrain_target_scale * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

        controller.configure_value_pretraining(
            state_sampler=state_sampler,
            target_function=target_function,
            config=ValuePretrainConfig(
                sample_count=pretrain_sample_count,
                epochs=pretrain_epochs,
                batch_size=pretrain_batch_size,
                learning_rate=pretrain_learning_rate,
                print_every=pretrain_print_every,
            ),
        )
        controller.pretrained_value_key = weights_key

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = WEIGHTS_DIR / f"{weights_key}.pt"

        if pretrain_mode == "load":
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found at {weights_path}. "
                    "Run with --pretrain-mode train to generate them."
                )
            controller.load_pretrained_value_weights(weights_path)
            print(f"Loaded pretrained learned value weights from {weights_path}")
        else:  # train
            if weights_path.exists():
                print(
                    f"WARNING: Pretrained weights already exist at {weights_path}. "
                    "Training from scratch — existing file will NOT be overwritten. "
                    "Delete it manually to save new weights."
                )
            if not controller.pretrain_learned_value(verbose=True):
                raise RuntimeError("Value pretraining callbacks were not configured.")
            if not weights_path.exists():
                controller.save_pretrained_value_weights(weights_path)
                print(f"Saved pretrained learned value weights to {weights_path}")

    else:  # mppi
        controller = MPPI(**shared)

    # ── Visualization setup ──────────────────────────────────────────────
    vis_fn = None
    if visualize:
        vis_dir = Path(__file__).resolve().parents[1] / "visualize" / "ant"

        def _plot_overlay(ax):
            # Capsule obstacles from scene.xml (radius=0.3)
            obstacle_positions = [
                (0, 2), (2, 0), (3, 4), (2, 6),
                (1, 3), (6, 1), (4, 3), (2, 5),
            ]
            for ox, oy in obstacle_positions:
                ax.add_patch(patches.Circle(
                    (ox, oy), 0.4,
                    edgecolor="black", facecolor="gray", alpha=0.8,
                ))
            # Goal (radius 0.3 from scene.xml)
            ax.add_patch(patches.Circle(
                (goal_xy[0], goal_xy[1]), 0.3,
                edgecolor="black", facecolor="red", alpha=0.5, linewidth=0.5, label="Goal",
            ))
            # Current ant position (torso radius 0.25 from ant.xml)
            ant_pos = mj_data.xpos[task.end_effector_pos_id, :2]
            ax.add_patch(patches.Circle(
                (ant_pos[0], ant_pos[1]), 0.25,
                edgecolor="black", facecolor="lime", linewidth=0.5, label="Ant",
            ))

        if args.controller in ("learned_value", "density_learned_value"):
            def _vis_fn(ctrl, step):
                visualize_learned_value(
                    ctrl, step,
                    output_dir=vis_dir,
                    plot_overlay=_plot_overlay,
                    vmin=0, vmax=10,
                    xlim=(-1, 8),
                    ylim=(-1, 8),
                )
        else:
            def _vis_fn(ctrl, step):
                visualize_rollouts(
                    ctrl, step,
                    mj_model=mj_model,
                    mj_data=mj_data,
                    output_dir=vis_dir,
                    plot_overlay=_plot_overlay,
                    xlim=(-1, 8),
                    ylim=(-1, 8),
                )

        vis_fn = _vis_fn

    # ── Run ─────────────────────────────────────────────────────────────
    run_interactive(
        controller=controller,
        mj_model=mj_model,
        mj_data=mj_data,
        frequency=frequency,
        show_traces=show_traces,
        record_video=record_video,
        video_width=video_width,
        video_height=video_height,
        video_crf=video_crf,
        video_preset=video_preset,
        max_steps=max_steps,
        camera_lookat=camera_lookat,
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        take_screenshot=take_screenshot,
        screenshot_path=screenshot_path,
        screenshot_dpi=screenshot_dpi,
        screenshot_step=screenshot_step,
        screenshot_every=screenshot_every,
        visualize_fn=vis_fn,
        visualize_every=visualize_every,
    )


if __name__ == "__main__":
    main()
