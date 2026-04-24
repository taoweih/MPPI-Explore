"""UR5e reach example using mujoco_warp MPPI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from algs import (
    MPPI,
    DensityGuidedMPPI,
    ValueGuidedMPPI,
    ValuePretrainConfig,
)
from simulation.deterministic import run_interactive
from tasks.ur5e import UR5e
from utils.visualize_learned_value import (
    visualize_learned_value_3d_scatter,
    visualize_rollouts_3d,
)

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
    num_samples = 512
    noise_level = 3.0
    temperature = 0.01
    plan_horizon = 0.1
    spline_type = "zero"
    num_knots = 8
    iterations = 1
    seed = 5

    # ── Density-guided parameters ───────────────────────────────────────
    num_knots_per_stage = 2
    kde_bandwidth = 0.30
    inverse_density_power = 0.5
    state_dim = 3
    state_source_field = "site_xpos"

    # ── Learned value parameters ───────────────────────────────────────────────
    value_grid_min = -2.0
    value_grid_max = 2.0
    hashgrid_num_levels = 8
    hashgrid_table_size = 262144
    hashgrid_min_resolution = 65.0
    hashgrid_max_resolution = 130.0
    online_learning_rate = 1e-3
    online_update_epochs = 2
    online_batch_size = 2
    online_anchor_samples = 0
    online_new_state_weight = 10.0
    goal_value = 0.0
    goal_weight = 1000.0
    weights_key = "ur5e_learned_value"
    pretrain_mode = "load"  # "load" or "train"

    # ── Pretraining parameters ──────────────────────────────────────────
    pretrain_sample_count = 10000000
    pretrain_epochs = 101
    pretrain_batch_size = 512*100
    pretrain_learning_rate = 1e-3
    pretrain_print_every = 50
    pretrain_target_scale = 1.0

    # ── Visualization parameters ───────────────────────────────────────
    visualize = False
    visualize_every = 50

    # ── Simulation parameters ───────────────────────────────────────────
    frequency = 50.0
    max_steps = 1000
    show_traces = False
    record_video = True

    # ── Video quality (only used when record_video=True) ────────────────
    video_width = 1080
    video_height = 1080
    video_crf = 18       # 0 = lossless, 18 ≈ visually lossless, 23 = default H.264, 28 = small file
    video_preset = "slow"  # ultrafast, fast, medium, slow, veryslow

    # ── Camera parameters (matches scene.xml <global> + <statistic>) ────
    camera_lookat = (-0.093, 0.274, 0.251)
    camera_distance = 2.480
    camera_azimuth = 55.4
    camera_elevation = -52.7

    # ── Screenshot parameters ──────────────────────────────────────────
    take_screenshot = True
    screenshot_path = str(
        Path(__file__).resolve().parents[1] / "visualize" / "ur5e" / "ur5e.png"
    )
    screenshot_dpi = 600
    screenshot_every = 10  # also capture every N steps as <task>_step_<N>.png

    # ── Task setup ──────────────────────────────────────────────────────
    task = UR5e()
    mj_model = task.mj_model
    mj_model.opt.timestep = task.sim_dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    ee_site_id = task.end_effector_pos_id

    # Goal position (needed for learned value variants)
    _tmp_data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, _tmp_data)
    goal_xyz = np.asarray(_tmp_data.xpos[task.goal_pos_id], dtype=np.float32)

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
            state_dim=state_dim,
            state_source_field=state_source_field,
            state_source_body_id=ee_site_id,
        )

    elif args.controller in ("learned_value", "density_learned_value"):
        use_density = args.controller == "density_learned_value"
        controller = ValueGuidedMPPI(
            **shared,
            state_dim=state_dim,
            state_source_field=state_source_field,
            state_source_body_id=ee_site_id,
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
            goal_state=goal_xyz[None, :],
            goal_value=goal_value,
            goal_weight=goal_weight,
            use_density_guided=use_density,
            num_knots_per_stage=num_knots_per_stage,
            kde_bandwidth=kde_bandwidth,
            inverse_density_power=inverse_density_power,
            disable_replay_anchors=True,
        )

        def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(value_grid_min, value_grid_max, size=(n, state_dim)).astype(np.float32)

        def target_function(states: np.ndarray) -> np.ndarray:
            diff = states - goal_xyz[None, :]
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

    # ── Visualization setup ───────────────────────────────────────────────────
    vis_fn = None
    if visualize:
        vis_dir = Path(__file__).resolve().parents[1] / "visualize" / "ur5e"

        # Obstacle cylinders from scene.xml: (center, radius, half_height)
        ur5e_cylinders = [
            ((0.0, 0.3, 0.20), 0.085, 0.20),
            ((0.2, 0.3, 0.20), 0.085, 0.20),
            ((-0.2, 0.3, 0.20), 0.085, 0.20),
            ((-0.2, 0.5, 0.20), 0.085, 0.20),
            ((-0.2, 0.7, 0.20), 0.085, 0.20),
        ]

        if args.controller in ("learned_value", "density_learned_value"):
            def _vis_fn(ctrl, step):
                ee_pos = mj_data.site_xpos[ee_site_id].copy()
                visualize_learned_value_3d_scatter(
                    ctrl, step,
                    resolution=30,
                    output_dir=vis_dir,
                    xlim=(-0.6, 0.6),
                    ylim=(-0.2, 1.0),
                    zlim=(-0.1, 0.7),
                    goal_xyz=goal_xyz,
                    ee_xyz=ee_pos,
                    cylinders=ur5e_cylinders,
                    elev=25.0,
                    azim=-60.0,
                )
        else:
            def _vis_fn(ctrl, step):
                ee_pos = mj_data.site_xpos[ee_site_id].copy()
                visualize_rollouts_3d(
                    ctrl, step,
                    mj_model=mj_model,
                    mj_data=mj_data,
                    output_dir=vis_dir,
                    xlim=(-0.6, 0.6),
                    ylim=(-0.2, 1.0),
                    zlim=(-0.1, 0.7),
                    goal_xyz=goal_xyz,
                    ee_xyz=ee_pos,
                    cylinders=ur5e_cylinders,
                    elev=25.0,
                    azim=-60.0,
                    num_rollout_samples=50,
                    rollout_state_indices=(0, 1, 2),
                    rollout_site_id=ee_site_id,
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
        screenshot_every=screenshot_every,
        visualize_fn=vis_fn,
        visualize_every=visualize_every,
    )


if __name__ == "__main__":
    main()
