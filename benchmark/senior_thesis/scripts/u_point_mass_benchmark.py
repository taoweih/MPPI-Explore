"""Senior thesis benchmark sweep for UPointMass in mppi_mjwarp."""

from __future__ import annotations

import argparse
from typing import Optional

import mujoco
import numpy as np

from algs import (
    MPPI,
    DensityGuidedMPPI,
    ValueGuidedMPPI,
    ValueDensityGuidedMPPI,
    HashGridValueModel,
    KDEDensityModel,
)
from benchmark.common.cli import add_sweep_config_args, apply_sweep_config_overrides
from benchmark.common.paths import senior_thesis_weights_dir
from benchmark.senior_thesis.scripts.benchmark_suite import (
    ControllerSpec,
    SeniorThesisBenchmarkSuite,
    SweepConfig,
)
from tasks import UPointMass

# ════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION — change these to control what the benchmark does
# ════════════════════════════════════════════════════════════════════════

# Horizon sweep
HORIZONS = np.linspace(0.2, 3.0, 20)
NUM_SAMPLES_FOR_HORIZON_SWEEP = 512

# Sample sweep
NUM_SAMPLES_LIST = np.linspace(32, 1024, 20, dtype=int).tolist()
HORIZON_FOR_SAMPLE_SWEEP = 1.5

# Trial settings
NUM_TRIALS = 50
MAX_ITERATIONS = 2000
GOAL_THRESHOLD = 0.4
FREQUENCY = 50.0

# Output
RECORD_VIDEO = False
OUTPUT_TAG = "thesis"

# Parallelism — "sequential", "controllers", "axis", or "all"
PARALLEL = "all"
MAX_WORKERS = "20"  # int or "auto" (= total jobs in batch)
NUM_GPUS = 1  # number of GPUs for round-robin assignment (1 = single GPU)

# ════════════════════════════════════════════════════════════════════════
# CONTROLLER PARAMETERS — must match examples/u_point_mass.py
# ════════════════════════════════════════════════════════════════════════

# Shared MPPI
NOISE_LEVEL = 3.0
TEMPERATURE = 0.001
NUM_KNOTS = 16

# Staged rollout
NUM_KNOTS_PER_STAGE = 4
KDE_BANDWIDTH = 0.15
INVERSE_DENSITY_POWER = 0.5

# Learned value
VALUE_GRID_MIN = -1.0
VALUE_GRID_MAX = 1.0
HASHGRID_NUM_LEVELS = 4
HASHGRID_TABLE_SIZE = 4096
HASHGRID_MIN_RESOLUTION = 50.0
HASHGRID_MAX_RESOLUTION = 200.0
ONLINE_LEARNING_RATE = 5e-3
ONLINE_UPDATE_EPOCHS = 1
ONLINE_BATCH_SIZE = 1
ONLINE_ANCHOR_SAMPLES = 0
ONLINE_NEW_STATE_WEIGHT = 10.0
GOAL_VALUE = 0.0
GOAL_WEIGHT = 2000.0

# Pretraining
PRETRAIN_SAMPLE_COUNT = 100000
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_TARGET_SCALE = 1.0

# ════════════════════════════════════════════════════════════════════════


def _goal_xy(task: UPointMass) -> np.ndarray:
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id, :2], dtype=np.float32)


WEIGHTS_DIR = senior_thesis_weights_dir()
WEIGHTS_KEY = "u_point_mass_learned_value"


def _learned_value_controller(
    task: UPointMass,
    horizon: float,
    *,
    use_density_guided: bool,
    num_samples: Optional[int] = None,
) -> ValueGuidedMPPI:
    goal_xy = _goal_xy(task)
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)

    value_model = HashGridValueModel(
        state_dim=task.state_dim,
        grid_min=VALUE_GRID_MIN, grid_max=VALUE_GRID_MAX,
        num_levels=HASHGRID_NUM_LEVELS, table_size=HASHGRID_TABLE_SIZE,
        min_resolution=HASHGRID_MIN_RESOLUTION,
        max_resolution=HASHGRID_MAX_RESOLUTION,
        seed=0,
    )
    controller_kwargs = dict(
        task=task, num_samples=n, noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE, plan_horizon=horizon, spline_type="zero",
        num_knots=NUM_KNOTS, iterations=1, seed=0,
        value_model=value_model,
        online_learning_rate=ONLINE_LEARNING_RATE,
        online_update_epochs=ONLINE_UPDATE_EPOCHS,
        online_batch_size=ONLINE_BATCH_SIZE,
    )
    if use_density_guided:
        controller = ValueDensityGuidedMPPI(
            **controller_kwargs,
            density_model=KDEDensityModel(
                bandwidth=KDE_BANDWIDTH, alpha=INVERSE_DENSITY_POWER,
            ),
            num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        )
    else:
        controller = ValueGuidedMPPI(**controller_kwargs)

    def state_sampler(rng: np.random.Generator, m: int) -> np.ndarray:
        return rng.uniform(VALUE_GRID_MIN, VALUE_GRID_MAX, size=(m, 2)).astype(np.float32)

    def target_function(states: np.ndarray) -> np.ndarray:
        diff = states - goal_xy[None, :]
        return PRETRAIN_TARGET_SCALE * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / f"{WEIGHTS_KEY}.pt"
    if weights_path.exists():
        controller.load_pretrained_weights(weights_path)
    else:
        controller.pretrain(
            state_sampler=state_sampler,
            target_function=target_function,
            sample_count=PRETRAIN_SAMPLE_COUNT,
            epochs=PRETRAIN_EPOCHS,
            batch_size=PRETRAIN_BATCH_SIZE,
            learning_rate=PRETRAIN_LEARNING_RATE,
            verbose=True,
            print_every=50,
        )
        controller.save_pretrained_weights(weights_path)
    return controller


def _mppi_factory(task, h, num_samples=None):
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)
    return MPPI(
        task=task, num_samples=n, noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE, plan_horizon=h, spline_type="zero",
        num_knots=NUM_KNOTS, iterations=1, seed=0,
    )


def _density_factory(task, h, num_samples=None):
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)
    return DensityGuidedMPPI(
        task=task, num_samples=n, noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE, plan_horizon=h, spline_type="zero",
        num_knots=NUM_KNOTS, iterations=1, seed=0,
        density_model=KDEDensityModel(
            bandwidth=KDE_BANDWIDTH, alpha=INVERSE_DENSITY_POWER,
        ),
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
    )


def _learned_value_factory(task, h, num_samples=None):
    return _learned_value_controller(task, h, use_density_guided=False, num_samples=num_samples)


def _density_learned_value_factory(task, h, num_samples=None):
    return _learned_value_controller(task, h, use_density_guided=True, num_samples=num_samples)


def build_controller_specs() -> list[ControllerSpec]:
    return [
        ControllerSpec(name="MPPI", factory=_mppi_factory),
        ControllerSpec(name="Density-Guided MPPI", factory=_density_factory),
        ControllerSpec(name="Value-Guided MPPI", factory=_learned_value_factory),
        ControllerSpec(name="Density + Value-Guided MPPI", factory=_density_learned_value_factory),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_sweep_config_args(parser)
    args, _ = parser.parse_known_args()

    global WEIGHTS_DIR
    WEIGHTS_DIR = senior_thesis_weights_dir(args.output_root)

    config = SweepConfig(
        horizons=HORIZONS,
        num_samples_list=NUM_SAMPLES_LIST,
        sweep_horizon_for_samples=HORIZON_FOR_SAMPLE_SWEEP,
        num_trials=NUM_TRIALS,
        frequency=FREQUENCY,
        goal_threshold=GOAL_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        record_video=RECORD_VIDEO,
        output_tag=OUTPUT_TAG,
        parallel=PARALLEL,
        max_workers=MAX_WORKERS,
        num_gpus=NUM_GPUS,
        pretrain_weights_dir=str(WEIGHTS_DIR),
        output_root=str(args.output_root) if args.output_root is not None else None,
        controller_params={
            "shared": {
                "noise_level": NOISE_LEVEL,
                "temperature": TEMPERATURE,
                "num_knots": NUM_KNOTS,
                "num_samples_default": NUM_SAMPLES_FOR_HORIZON_SWEEP,
            },
            "density": {
                "num_knots_per_stage": NUM_KNOTS_PER_STAGE,
                "kde_bandwidth": KDE_BANDWIDTH,
                "inverse_density_power": INVERSE_DENSITY_POWER,
            },
            "learned_value": {
                "value_grid_min": VALUE_GRID_MIN,
                "value_grid_max": VALUE_GRID_MAX,
                "hashgrid_num_levels": HASHGRID_NUM_LEVELS,
                "hashgrid_table_size": HASHGRID_TABLE_SIZE,
                "hashgrid_min_resolution": HASHGRID_MIN_RESOLUTION,
                "hashgrid_max_resolution": HASHGRID_MAX_RESOLUTION,
                "online_learning_rate": ONLINE_LEARNING_RATE,
                "online_update_epochs": ONLINE_UPDATE_EPOCHS,
                "online_batch_size": ONLINE_BATCH_SIZE,
                "online_anchor_samples": ONLINE_ANCHOR_SAMPLES,
                "online_new_state_weight": ONLINE_NEW_STATE_WEIGHT,
                "goal_value": GOAL_VALUE,
                "goal_weight": GOAL_WEIGHT,
            },
            "pretraining": {
                "pretrain_sample_count": PRETRAIN_SAMPLE_COUNT,
                "pretrain_epochs": PRETRAIN_EPOCHS,
                "pretrain_batch_size": PRETRAIN_BATCH_SIZE,
                "pretrain_learning_rate": PRETRAIN_LEARNING_RATE,
                "pretrain_target_scale": PRETRAIN_TARGET_SCALE,
            },
        },
    )
    apply_sweep_config_overrides(config, args)

    suite = SeniorThesisBenchmarkSuite(
        task_name="u_point_mass",
        task_factory=UPointMass,
        controller_specs=build_controller_specs(),
        config=config,
        module_name=__name__,
    )
    suite.run()


if __name__ == "__main__":
    main()
