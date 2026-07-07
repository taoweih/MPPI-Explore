"""UR5e KNN-k sensitivity benchmark for paper experiments."""

from __future__ import annotations

import argparse

import numpy as np

from algs import DIALMPC, DensityGuidedMPPI, KNNDensityModel, MPPI
from benchmark.paper.scripts.sensitivity._success_knn_k import (
    ControllerSpec,
    SuccessKConfig,
    add_success_k_sensitivity_args,
    apply_success_k_overrides,
    run_success_k_sensitivity,
)
from tasks import UR5e


K_VALUES = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 50], dtype=np.int32)

FIXED_HORIZON = 0.5
NUM_SAMPLES = 512

NUM_TRIALS = 50
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 0.4
FREQUENCY = 50.0

RECORD_VIDEO = False
OUTPUT_NAME = "ur5e_knn_k_sensitivity"

PARALLEL = "all"
MAX_WORKERS = "50"
NUM_GPUS = 2
FREQ_CALIBRATION_ITERS = 50

NOISE_LEVEL = 3.0
TEMPERATURE = 0.01
NUM_KNOTS = 8
ITERATIONS = 1
SEED = 5

DIAL_NUM_INITIAL_DIFFUSION_STEPS = 10
DIAL_HORIZON_DIFFUSE_FACTOR = 0.9
DIAL_TRAJ_DIFFUSE_FACTOR = 0.5
DIAL_INCLUDE_MEAN_SAMPLE = True
DIAL_FIX_FIRST_KNOT = True
DIAL_NORMALIZE_COSTS = True
DIAL_COST_NORMALIZATION_EPS = 1.0e-6

NUM_KNOTS_PER_STAGE = 2
INVERSE_DENSITY_POWER = 1.0
RESAMPLE_COST_WEIGHT = 1.0
RESAMPLE_COST_TEMPERATURE = None

KNN_POSITION_WEIGHT = 1.0
KNN_ANGLE_WEIGHT = 1.0
KNN_LINEAR_VELOCITY_WEIGHT = 1.0
KNN_ANGULAR_VELOCITY_WEIGHT = 1.0
KNN_TASK_STATE_WEIGHT = 1.0


def _base_mppi_kwargs(task: UR5e) -> dict:
    return dict(
        task=task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=FIXED_HORIZON,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=ITERATIONS,
        seed=SEED,
    )


def _mppi_factory(task: UR5e) -> MPPI:
    return MPPI(**_base_mppi_kwargs(task))


def _dial_factory(task: UR5e) -> DIALMPC:
    return DIALMPC(
        **_base_mppi_kwargs(task),
        num_initial_diffusion_steps=DIAL_NUM_INITIAL_DIFFUSION_STEPS,
        horizon_diffuse_factor=DIAL_HORIZON_DIFFUSE_FACTOR,
        traj_diffuse_factor=DIAL_TRAJ_DIFFUSE_FACTOR,
        include_mean_sample=DIAL_INCLUDE_MEAN_SAMPLE,
        fix_first_knot=DIAL_FIX_FIRST_KNOT,
        normalize_costs=DIAL_NORMALIZE_COSTS,
        cost_normalization_eps=DIAL_COST_NORMALIZATION_EPS,
    )


def _density_knn_factory(
    task: UR5e,
    *,
    k: int,
    include_task_state: bool,
) -> DensityGuidedMPPI:
    return DensityGuidedMPPI(
        **_base_mppi_kwargs(task),
        density_model=KNNDensityModel(
            k=k,
            alpha=INVERSE_DENSITY_POWER,
            position_weight=KNN_POSITION_WEIGHT,
            angle_weight=KNN_ANGLE_WEIGHT,
            linear_velocity_weight=KNN_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=KNN_ANGULAR_VELOCITY_WEIGHT,
            include_task_state=include_task_state,
            task_state_weight=KNN_TASK_STATE_WEIGHT,
        ),
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        resample_cost_weight=RESAMPLE_COST_WEIGHT,
        resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
    )


def build_controller_specs(k: int) -> list[ControllerSpec]:
    return [
        ControllerSpec(name="MPPI", factory=_mppi_factory, sweeps_k=False),
        ControllerSpec(name="DIAL-MPC", factory=_dial_factory, sweeps_k=False),
        ControllerSpec(
            name="Density-Guided MPPI (KNN qpos+qvel)",
            factory=lambda task: _density_knn_factory(
                task,
                k=k,
                include_task_state=False,
            ),
        ),
        ControllerSpec(
            name="Density-Guided MPPI (KNN qpos+qvel+state)",
            factory=lambda task: _density_knn_factory(
                task,
                k=k,
                include_task_state=True,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_success_k_sensitivity_args(parser)
    args, _ = parser.parse_known_args()

    config = SuccessKConfig(
        task_name="ur5e",
        output_name=OUTPUT_NAME,
        k_values=K_VALUES,
        horizon=FIXED_HORIZON,
        num_samples=NUM_SAMPLES,
        num_trials=NUM_TRIALS,
        frequency=FREQUENCY,
        goal_threshold=GOAL_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        record_video=RECORD_VIDEO,
        parallel=PARALLEL,
        max_workers=MAX_WORKERS,
        num_gpus=NUM_GPUS,
        freq_calibration_iters=FREQ_CALIBRATION_ITERS,
        controller_params={
            "shared": {
                "noise_level": NOISE_LEVEL,
                "temperature": TEMPERATURE,
                "plan_horizon": FIXED_HORIZON,
                "num_samples": NUM_SAMPLES,
                "num_knots": NUM_KNOTS,
                "iterations": ITERATIONS,
                "seed": SEED,
            },
            "density": {
                "num_knots_per_stage": NUM_KNOTS_PER_STAGE,
                "inverse_density_power": INVERSE_DENSITY_POWER,
                "resample_cost_weight": RESAMPLE_COST_WEIGHT,
                "resample_cost_temperature": RESAMPLE_COST_TEMPERATURE,
            },
            "knn_density": {
                "knn_k_values": K_VALUES,
                "position_weight": KNN_POSITION_WEIGHT,
                "angle_weight": KNN_ANGLE_WEIGHT,
                "linear_velocity_weight": KNN_LINEAR_VELOCITY_WEIGHT,
                "angular_velocity_weight": KNN_ANGULAR_VELOCITY_WEIGHT,
            },
            "knn_task_state_density": {
                "knn_k_values": K_VALUES,
                "include_task_state": True,
                "task_state_weight": KNN_TASK_STATE_WEIGHT,
                "task_state_dim": UR5e.state_dim,
            },
            "dial_mpc": {
                "num_initial_diffusion_steps": DIAL_NUM_INITIAL_DIFFUSION_STEPS,
                "horizon_diffuse_factor": DIAL_HORIZON_DIFFUSE_FACTOR,
                "traj_diffuse_factor": DIAL_TRAJ_DIFFUSE_FACTOR,
                "include_mean_sample": DIAL_INCLUDE_MEAN_SAMPLE,
                "fix_first_knot": DIAL_FIX_FIRST_KNOT,
                "normalize_costs": DIAL_NORMALIZE_COSTS,
                "cost_normalization_eps": DIAL_COST_NORMALIZATION_EPS,
            },
        },
    )
    apply_success_k_overrides(config, args)

    run_success_k_sensitivity(
        task_factory=UR5e,
        build_controller_specs=build_controller_specs,
        config=config,
        module_name=__name__,
    )


if __name__ == "__main__":
    main()
