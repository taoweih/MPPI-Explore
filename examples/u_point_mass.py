"""U-shaped point mass example using mujoco_warp MPPI."""

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
from tasks.u_point_mass import UPointMass
from utils.visualize_learned_value import visualize_learned_value

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
    temperature = 0.001
    plan_horizon = 0.2
    spline_type = "zero"
    num_knots = 16
    iterations = 1
    seed = 0

    # ── Density-guided parameters ───────────────────────────────────────
    num_knots_per_stage = 4
    kde_bandwidth = 0.15
    inverse_density_power = 0.5
    state_dim = 2
    state_source_field = "qpos"

    # ── Learned value parameters ───────────────────────────────────────────────
    value_grid_min = -1.0
    value_grid_max = 1.0
    hashgrid_num_levels = 4
    hashgrid_table_size = 4096
    hashgrid_min_resolution = 50.0
    hashgrid_max_resolution = 200.0
    online_learning_rate = 5e-3
    online_update_epochs = 1
    online_batch_size = 1
    online_anchor_samples = 0
    online_new_state_weight = 10.0
    goal_value = 0.0
    goal_weight = 2000.0
    weights_key = "u_point_mass_learned_value"
    pretrain_mode = "load"  # "load" or "train"

    # ── Pretraining parameters ──────────────────────────────────────────
    pretrain_sample_count = 100000
    pretrain_epochs = 100
    pretrain_batch_size = 512
    pretrain_learning_rate = 1e-3
    pretrain_print_every = 50
    pretrain_target_scale = 1.0

    # ── Visualization parameters ───────────────────────────────────────
    visualize = True
    visualize_every = 50

    # ── Simulation parameters ───────────────────────────────────────────
    frequency = 50.0
    max_steps = 1001
    show_traces = True
    record_video = False

    # ── Task setup ──────────────────────────────────────────────────────
    task = UPointMass()
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
            state_dim=state_dim,
            state_source_field=state_source_field,
        )

    elif args.controller in ("learned_value", "density_learned_value"):
        use_density = args.controller == "density_learned_value"
        controller = ValueGuidedMPPI(
            **shared,
            state_dim=state_dim,
            state_source_field=state_source_field,
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
        )

        def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(value_grid_min, value_grid_max, size=(n, state_dim)).astype(np.float32)

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

    # ── Learned value visualization setup ─────────────────────────────────────
    vis_fn = None
    if visualize and args.controller in ("learned_value", "density_learned_value"):
        vis_dir = Path(__file__).resolve().parents[1] / "visualize" / "u_point_mass"

        def _plot_overlay(ax):
            # U-shaped walls from scene.xml
            ax.add_patch(patches.Rectangle((-0.2, 0.19), 0.4, 0.02,
                         edgecolor="black", facecolor="gray"))
            ax.add_patch(patches.Rectangle((0.20, -0.19), 0.02, 0.4,
                         edgecolor="black", facecolor="gray"))
            ax.add_patch(patches.Rectangle((-0.22, -0.19), 0.02, 0.4,
                         edgecolor="black", facecolor="gray"))
            # Goal (radius 0.01 from scene.xml)
            ax.add_patch(patches.Circle(
                (goal_xy[0], goal_xy[1]), 0.01,
                edgecolor="black", facecolor="red", linewidth=0.5, label="Goal",
                zorder=5,
            ))
            # Current point mass position (radius 0.01 from point_mass.xml)
            pm_pos = mj_data.xpos[task.end_effector_pos_id, :2]
            ax.add_patch(patches.Circle(
                (pm_pos[0], pm_pos[1]), 0.01,
                edgecolor="black", facecolor="lime", linewidth=0.5, label="Point Mass",
                zorder=6,
            ))

        def _vis_fn(ctrl, step):
            visualize_learned_value(
                ctrl, step,
                output_dir=vis_dir,
                plot_overlay=_plot_overlay,
                vmin=0, vmax=2,
                mj_model=mj_model,
                mj_data=mj_data,
                num_rollout_samples=50,
                rollout_color="black",
                rollout_alpha=0.12,
                best_rollout_color="purple",
                best_rollout_linewidth=1.5,
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
        max_steps=max_steps,
        visualize_fn=vis_fn,
        visualize_every=visualize_every,
    )


if __name__ == "__main__":
    main()
