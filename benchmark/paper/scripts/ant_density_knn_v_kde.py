"""Ant benchmark comparing MPPI and density-estimator variants.

This focused test script reuses the senior-thesis benchmark runner so success
rate, time-to-success, and frequency are computed and saved the same way as the
main benchmark and sensitivity scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from algs import (
    MPPI,
    DIALMPC,
    DensityGuidedMPPI,
    KDEDensityModel,
    KNNDensityModel,
)
from benchmark.common.cli import add_sweep_config_args, apply_sweep_config_overrides
from benchmark.common.paths import runs_dir
from benchmark.senior_thesis.scripts.benchmark_suite import (
    ControllerSpec,
    SeniorThesisBenchmarkSuite,
    SweepConfig,
    SweepResult,
)
from tasks import Ant


# Horizon sweep
HORIZONS = np.linspace(0.5, 8.0, 20)
NUM_SAMPLES_FOR_HORIZON_SWEEP = 1024

# Sample-count sweep
NUM_SAMPLES_LIST = np.linspace(128, 2048, 20, dtype=int).tolist()
HORIZON_FOR_SAMPLE_SWEEP = 4.0

# Trial settings
NUM_TRIALS = 20
MAX_ITERATIONS = 10000
GOAL_THRESHOLD = 0.5
FREQUENCY = 50.0

# Output
RECORD_VIDEO = False
OUTPUT_NAME = "ant_mppi_vs_density_kde_vs_knn_vs_knn_task_state_vs_dial"

# Parallelism: "sequential", "controllers", "axis", or "all"
PARALLEL = "all"
MAX_WORKERS = "20"  # int or "auto" (= total jobs in batch)
NUM_GPUS = 2
FREQ_CALIBRATION_ITERS = 50

# Shared MPPI parameters
NOISE_LEVEL = 0.3
TEMPERATURE = 0.001
NUM_KNOTS = 16
ITERATIONS = 1
SEED = 0

# DIAL-MPC parameters. The shared MPPI parameters above intentionally define
# DIAL's sample count, horizon, noise scale, temperature, knots, and diffusion
# updates; these are the DIAL-only defaults from the paper-style controller.
DIAL_NUM_INITIAL_DIFFUSION_STEPS = 10
DIAL_HORIZON_DIFFUSE_FACTOR = 0.9
DIAL_TRAJ_DIFFUSE_FACTOR = 0.5
DIAL_INCLUDE_MEAN_SAMPLE = True
DIAL_FIX_FIRST_KNOT = True
DIAL_NORMALIZE_COSTS = True
DIAL_COST_NORMALIZATION_EPS = 1.0e-6

# Density-guided parameters
NUM_KNOTS_PER_STAGE = 4
INVERSE_DENSITY_POWER = 1.0
RESAMPLE_COST_WEIGHT = 1.0
RESAMPLE_COST_TEMPERATURE = None  # None uses unit temperature for normalized cost scores.

# KDE density model
KDE_BANDWIDTH = 0.10

# KNN density model
KNN_K = 5
KNN_POSITION_WEIGHT = 1.0
KNN_ANGLE_WEIGHT = 1.0
KNN_LINEAR_VELOCITY_WEIGHT = 1.0
KNN_ANGULAR_VELOCITY_WEIGHT = 1.0
KNN_TASK_STATE_WEIGHT = 1.0


def _num_samples(num_samples: Optional[int]) -> int:
    return (
        NUM_SAMPLES_FOR_HORIZON_SWEEP
        if num_samples is None
        else int(num_samples)
    )


def _base_mppi_kwargs(
    task: Ant,
    horizon: float,
    num_samples: Optional[int],
) -> dict:
    return dict(
        task=task,
        num_samples=_num_samples(num_samples),
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=float(horizon),
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=ITERATIONS,
        seed=SEED,
    )


def _mppi_factory(
    task: Ant,
    horizon: float,
    num_samples: Optional[int] = None,
) -> MPPI:
    return MPPI(**_base_mppi_kwargs(task, horizon, num_samples))


def _dial_factory(
    task: Ant,
    horizon: float,
    num_samples: Optional[int] = None,
) -> DIALMPC:
    return DIALMPC(
        **_base_mppi_kwargs(task, horizon, num_samples),
        num_initial_diffusion_steps=DIAL_NUM_INITIAL_DIFFUSION_STEPS,
        horizon_diffuse_factor=DIAL_HORIZON_DIFFUSE_FACTOR,
        traj_diffuse_factor=DIAL_TRAJ_DIFFUSE_FACTOR,
        include_mean_sample=DIAL_INCLUDE_MEAN_SAMPLE,
        fix_first_knot=DIAL_FIX_FIRST_KNOT,
        normalize_costs=DIAL_NORMALIZE_COSTS,
        cost_normalization_eps=DIAL_COST_NORMALIZATION_EPS,
    )


def _density_kde_factory(
    task: Ant,
    horizon: float,
    num_samples: Optional[int] = None,
) -> DensityGuidedMPPI:
    return DensityGuidedMPPI(
        **_base_mppi_kwargs(task, horizon, num_samples),
        density_model=KDEDensityModel(
            bandwidth=KDE_BANDWIDTH,
            alpha=INVERSE_DENSITY_POWER,
        ),
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        resample_cost_weight=RESAMPLE_COST_WEIGHT,
        resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
    )


def _density_knn_factory(
    task: Ant,
    horizon: float,
    num_samples: Optional[int] = None,
) -> DensityGuidedMPPI:
    return DensityGuidedMPPI(
        **_base_mppi_kwargs(task, horizon, num_samples),
        density_model=KNNDensityModel(
            k=KNN_K,
            alpha=INVERSE_DENSITY_POWER,
            position_weight=KNN_POSITION_WEIGHT,
            angle_weight=KNN_ANGLE_WEIGHT,
            linear_velocity_weight=KNN_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=KNN_ANGULAR_VELOCITY_WEIGHT,
            include_task_state=False,
        ),
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        resample_cost_weight=RESAMPLE_COST_WEIGHT,
        resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
    )


def _density_knn_task_state_factory(
    task: Ant,
    horizon: float,
    num_samples: Optional[int] = None,
) -> DensityGuidedMPPI:
    return DensityGuidedMPPI(
        **_base_mppi_kwargs(task, horizon, num_samples),
        density_model=KNNDensityModel(
            k=KNN_K,
            alpha=INVERSE_DENSITY_POWER,
            position_weight=KNN_POSITION_WEIGHT,
            angle_weight=KNN_ANGLE_WEIGHT,
            linear_velocity_weight=KNN_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=KNN_ANGULAR_VELOCITY_WEIGHT,
            include_task_state=True,
            task_state_weight=KNN_TASK_STATE_WEIGHT,
        ),
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        resample_cost_weight=RESAMPLE_COST_WEIGHT,
        resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
    )


def build_controller_specs() -> list[ControllerSpec]:
    return [
        ControllerSpec(name="MPPI", factory=_mppi_factory),
        ControllerSpec(name="Density-Guided MPPI (KDE)", factory=_density_kde_factory),
        ControllerSpec(
            name="Density-Guided MPPI (KNN qpos+qvel)",
            factory=_density_knn_factory,
        ),
        ControllerSpec(
            name="Density-Guided MPPI (KNN qpos+qvel+state)",
            factory=_density_knn_task_state_factory,
        ),
        ControllerSpec(name="DIAL-MPC", factory=_dial_factory),
    ]


class PaperBenchmarkSuite(SeniorThesisBenchmarkSuite):
    """Paper benchmark suite with paper-category output paths."""

    def _save_results(
        self,
        *,
        horizon_result: SweepResult,
        sample_result: Optional[SweepResult],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = runs_dir("paper", output_root=self.config.output_root) / (
            f"{OUTPUT_NAME}_{timestamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_sweep(out_dir, horizon_result)
        if sample_result is not None:
            self._save_sweep(out_dir, sample_result)

        summary_payload: dict[str, Any] = {
            "controller_names": np.array(
                [spec.name for spec in self.controller_specs],
                dtype=object,
            ),
            "horizon_values": horizon_result.axis_values,
            "horizon_success": horizon_result.success,
            "horizon_success_time": horizon_result.success_time,
            "horizon_success_time_std": horizon_result.success_time_std,
            "horizon_frequency_mean": horizon_result.frequency_mean,
            "horizon_frequency_std": horizon_result.frequency_std,
        }
        if sample_result is not None:
            summary_payload.update(
                {
                    "num_samples_values": sample_result.axis_values,
                    "num_samples_success": sample_result.success,
                    "num_samples_success_time": sample_result.success_time,
                    "num_samples_success_time_std": sample_result.success_time_std,
                    "num_samples_frequency_mean": sample_result.frequency_mean,
                    "num_samples_frequency_std": sample_result.frequency_std,
                }
            )
        np.savez(out_dir / "summary.npz", **summary_payload)

        metadata = {
            "task_name": self.task_name,
            "controller_names": [spec.name for spec in self.controller_specs],
            "config": self._jsonify(asdict(self.config)),
            "horizons": horizon_result.axis_values.tolist(),
        }
        if sample_result is not None:
            metadata["num_samples_list"] = sample_result.axis_values.tolist()
        if self.config.controller_params is not None:
            metadata["controller_params"] = self._jsonify(self.config.controller_params)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    add_sweep_config_args(parser)
    parser.add_argument(
        "--simulation",
        "--sim",
        dest="simulation_mode",
        default="deterministic",
        choices=("deterministic", "async"),
        help="CPU simulation loop to use.",
    )
    args, _ = parser.parse_known_args()

    config = SweepConfig(
        horizons=HORIZONS,
        num_samples_list=NUM_SAMPLES_LIST,
        sweep_horizon_for_samples=HORIZON_FOR_SAMPLE_SWEEP,
        num_trials=NUM_TRIALS,
        frequency=FREQUENCY,
        goal_threshold=GOAL_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        record_video=RECORD_VIDEO,
        output_tag=OUTPUT_NAME,
        output_root=str(args.output_root) if args.output_root is not None else None,
        parallel=PARALLEL,
        max_workers=MAX_WORKERS,
        num_gpus=NUM_GPUS,
        simulation_mode=args.simulation_mode,
        freq_calibration_iters=FREQ_CALIBRATION_ITERS,
        controller_params={
            "shared": {
                "noise_level": NOISE_LEVEL,
                "temperature": TEMPERATURE,
                "num_knots": NUM_KNOTS,
                "iterations": ITERATIONS,
                "seed": SEED,
                "num_samples_default": NUM_SAMPLES_FOR_HORIZON_SWEEP,
            },
            "density": {
                "num_knots_per_stage": NUM_KNOTS_PER_STAGE,
                "inverse_density_power": INVERSE_DENSITY_POWER,
                "resample_cost_weight": RESAMPLE_COST_WEIGHT,
                "resample_cost_temperature": RESAMPLE_COST_TEMPERATURE,
                "resample_score_normalization": "zscore(-log_density) + zscore(-cost)",
                "resample_cost_temperature_default": 1.0,
            },
            "kde_density": {
                "kde_bandwidth": KDE_BANDWIDTH,
            },
            "knn_density": {
                "knn_k": KNN_K,
                "include_task_state": False,
                "position_weight": KNN_POSITION_WEIGHT,
                "angle_weight": KNN_ANGLE_WEIGHT,
                "linear_velocity_weight": KNN_LINEAR_VELOCITY_WEIGHT,
                "angular_velocity_weight": KNN_ANGULAR_VELOCITY_WEIGHT,
            },
            "knn_task_state_density": {
                "knn_k": KNN_K,
                "include_task_state": True,
                "task_state_weight": KNN_TASK_STATE_WEIGHT,
                "task_state_dim": Ant.state_dim,
                "position_weight": KNN_POSITION_WEIGHT,
                "angle_weight": KNN_ANGLE_WEIGHT,
                "linear_velocity_weight": KNN_LINEAR_VELOCITY_WEIGHT,
                "angular_velocity_weight": KNN_ANGULAR_VELOCITY_WEIGHT,
            },
            "dial_mpc": {
                "num_samples": "same as shared/sweep value",
                "noise_level": "same as shared",
                "temperature": "same as shared",
                "plan_horizon": "same as sweep value",
                "spline_type": "zero",
                "num_knots": "same as shared",
                "iterations": "same as shared",
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
    apply_sweep_config_overrides(config, args)

    suite = PaperBenchmarkSuite(
        task_name="ant",
        task_factory=Ant,
        controller_specs=build_controller_specs(),
        config=config,
        module_name=__name__,
    )
    suite.run()


if __name__ == "__main__":
    main()
