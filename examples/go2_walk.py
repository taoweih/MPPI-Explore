"""Unitree Go2 walk example using mujoco_warp MPPI."""

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
from simulation.deterministic import run_interactive
from tasks.go2_walk import Go2Walk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "controller",
        nargs="?",
        default="mppi",
        choices=("mppi", "dial", "density"),
    )
    args = parser.parse_args()

    # -- Algorithm parameters: base MPPI ---------------------------------
    planning_dt = 0.02
    num_samples = 2048
    noise_level = 1.0
    temperature = 0.05
    plan_horizon = 16 * planning_dt
    spline_type = "zero"
    num_knots = 5
    iterations = 2
    seed = 0

    # -- Algorithm parameters: density-guided MPPI -----------------------
    density_model_type = "knn"  # "kde" or "knn"
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
    # Defaults from tmp/dial-mpc/dial_mpc/examples/unitree_go2_trot.yaml.
    dial_num_samples = 2048
    dial_horizon_steps = 16
    dial_control_node_count = 4
    dial_spline_type = "zero"
    dial_num_diffusion_steps = 2
    dial_num_initial_diffusion_steps = 10
    dial_temp_sample = 0.05
    dial_sigma_scale = 1.0
    dial_horizon_diffuse_factor = 0.9
    dial_traj_diffuse_factor = 0.5
    dial_seed = 0

    # -- Simulation parameters -------------------------------------------
    frequency = 1.0 / planning_dt
    max_steps = 400
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
        / "go2_walk"
        / "go2_walk.png"
    )
    screenshot_dpi = 600
    screenshot_every = 10

    # Task setup
    task = Go2Walk(
        planning_dt=planning_dt,
        sim_dt=planning_dt,
        target_vx=0.8,
        target_vy=0.0,
        target_vyaw=0.0,
        gait="trot",
    )
    mj_model = task.mj_model
    mj_model.opt.timestep = task.sim_dt
    mj_data = mujoco.MjData(mj_model)
    task.reset_to_home(mj_data)

    fixed_camera_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
    )
    if fixed_camera_id < 0:
        fixed_camera_id = None

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

    if args.controller == "density":
        controller = DensityGuidedMPPI(
            **base_mppi_kwargs,
            density_model=build_density_model(),
            num_knots_per_stage=num_knots_per_stage,
        )
    elif args.controller == "dial":
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
    else:
        controller = MPPI(**base_mppi_kwargs)

    # Run
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
        fixed_camera_id=fixed_camera_id,
        take_screenshot=take_screenshot,
        screenshot_path=screenshot_path,
        screenshot_dpi=screenshot_dpi,
        screenshot_every=screenshot_every,
    )


if __name__ == "__main__":
    main()
