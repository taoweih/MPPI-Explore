"""Unitree H1 push-crate example using mujoco_warp MPPI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco

from algs import (
    DIALMPC,
    MPPI,
    DensityGuidedMPPI,
    KDEDensityModel,
    KNNDensityModel,
)
from simulation import run_interactive, run_interactive_async
from tasks.h1_push_crate import H1PushCrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "controller",
        nargs="?",
        default="mppi",
        choices=("mppi", "density", "dial"),
    )
    parser.add_argument(
        "--simulation",
        "--sim",
        default="deterministic",
        choices=("deterministic", "async"),
        help="CPU simulation loop to use.",
    )
    args = parser.parse_args()

    # -- Algorithm parameters: base MPPI ---------------------------------
    planning_dt = 0.02
    sample_horizon_steps = 24
    control_node_count = 8
    num_samples = 4096
    noise_level = 1.0
    temperature = 0.01
    iterations = 1
    plan_horizon = sample_horizon_steps * planning_dt
    spline_type = "zero"
    num_knots = control_node_count + 1
    seed = 0

    # -- Algorithm parameters: density-guided MPPI -----------------------
    density_model_type = "knn"  # "kde" or "knn"
    density_spline_type = spline_type
    num_knots_per_stage = 2
    inverse_density_power = 1.0

    # KDE density model
    kde_bandwidth = 0.20

    # KNN density model
    knn_k = 5
    knn_position_weight = 1.0
    knn_angle_weight = 1.0
    knn_linear_velocity_weight = 1.0
    knn_angular_velocity_weight = 1.0

    # -- Algorithm parameters: DIAL-MPC ----------------------------------
    # Defaults from tmp/dial-mpc/dial_mpc/examples/unitree_h1_push_crate.yaml.
    dial_num_samples = 2048
    dial_horizon_steps = 24
    dial_control_node_count = 6
    dial_spline_type = "quadratic"
    dial_num_diffusion_steps = 4
    dial_num_initial_diffusion_steps = 10
    dial_temp_sample = 0.05
    dial_sigma_scale = 1.0
    dial_horizon_diffuse_factor = 0.9
    dial_traj_diffuse_factor = 0.5
    dial_seed = 0

    # -- Simulation parameters -------------------------------------------
    frequency = 1.0 / planning_dt
    max_steps = 300
    show_traces = False
    record_video = False

    # Video quality (only used when record_video=True)
    video_width = 1080
    video_height = 1080
    video_crf = 18
    video_preset = "slow"

    # Screenshot parameters
    take_screenshot = False
    screenshot_path = str(
        Path(__file__).resolve().parents[1]
        / "visualize"
        / "h1_push_crate"
        / "h1_push_crate.png"
    )
    screenshot_dpi = 600
    screenshot_every = 10

    # Camera parameters
    camera_lookat = (1.0, 0.0, 0.85)
    camera_distance = 4.0
    camera_azimuth = 145.0
    camera_elevation = -18.0

    # Task setup
    task = H1PushCrate(
        planning_dt=planning_dt,
        sim_dt=planning_dt,
        target_vx=0.8,
        target_vy=0.0,
        target_vyaw=0.0,
        gait="slow_walk",
    )
    mj_model = task.mj_model
    mj_model.opt.timestep = task.sim_dt
    mj_data = mujoco.MjData(mj_model)
    task.reset_to_home(mj_data)

    # Controller setup
    base_mppi_kwargs = dict(
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
    density_mppi_kwargs = dict(base_mppi_kwargs)
    density_mppi_kwargs["spline_type"] = density_spline_type

    def build_density_model():
        if density_model_type == "kde":
            return KDEDensityModel(
                bandwidth=kde_bandwidth,
                alpha=inverse_density_power,
            )
        if density_model_type == "knn":
            return KNNDensityModel(
                k=knn_k,
                alpha=inverse_density_power,
                position_weight=knn_position_weight,
                angle_weight=knn_angle_weight,
                linear_velocity_weight=knn_linear_velocity_weight,
                angular_velocity_weight=knn_angular_velocity_weight,
            )
        raise ValueError(
            f"Unknown density_model_type={density_model_type!r}; expected 'kde' or 'knn'."
        )

    if args.controller == "dial":
        controller = DIALMPC(
            task=task,
            num_samples=dial_num_samples,
            sigma_scale=dial_sigma_scale,
            temp_sample=dial_temp_sample,
            plan_horizon=dial_horizon_steps * planning_dt,
            spline_type=dial_spline_type,
            num_knots=dial_control_node_count + 1,
            num_diffusion_steps=dial_num_diffusion_steps,
            num_initial_diffusion_steps=dial_num_initial_diffusion_steps,
            horizon_diffuse_factor=dial_horizon_diffuse_factor,
            traj_diffuse_factor=dial_traj_diffuse_factor,
            seed=dial_seed,
        )
    elif args.controller == "density":
        controller = DensityGuidedMPPI(
            **density_mppi_kwargs,
            density_model=build_density_model(),
            num_knots_per_stage=num_knots_per_stage,
        )
    else:
        controller = MPPI(**base_mppi_kwargs)

    # Run
    simulation_runner = (
        run_interactive_async if args.simulation == "async" else run_interactive
    )
    simulation_runner(
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
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_lookat=camera_lookat,
        take_screenshot=take_screenshot,
        screenshot_path=screenshot_path,
        screenshot_dpi=screenshot_dpi,
        screenshot_every=screenshot_every,
    )


if __name__ == "__main__":
    main()
