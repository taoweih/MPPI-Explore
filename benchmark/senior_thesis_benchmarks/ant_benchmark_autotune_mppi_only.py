"""Senior thesis benchmark sweep for Ant in mppi_mjwarp."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algs import (  # noqa: E402
    MPPI,
    DensityGuidedMPPI,
    ValueGuidedMPPI,
    ValuePretrainConfig,
)
from benchmark.senior_thesis_benchmarks.benchmark_suite import (  # noqa: E402
    ControllerSpec,
    SeniorThesisBenchmarkSuite,
    SweepConfig,
)
from tasks import Ant  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# Horizon sweep
HORIZONS = np.linspace(0.5, 4.0, 10)
NUM_SAMPLES_FOR_HORIZON_SWEEP = 1024

# Sample sweep
NUM_SAMPLES_LIST = np.linspace(128, 2048, 10, dtype=int).tolist()
HORIZON_FOR_SAMPLE_SWEEP = 2.0

# Trial settings
NUM_TRIALS = 30
MAX_ITERATIONS = 10000
GOAL_THRESHOLD = 0.5
FREQUENCY = 50.0

# Output
RECORD_VIDEO = False
OUTPUT_TAG = "thesis"

# Parallelism — "sequential", "controllers", "axis", or "all"
PARALLEL = "all"
MAX_WORKERS = "20"  # int or "auto" (= total jobs in batch)
NUM_GPUS = 2  # number of GPUs for round-robin assignment (1 = single GPU)

# ════════════════════════════════════════════════════════════════════════
# CONTROLLER PARAMETERS — must match examples/ant.py
# ════════════════════════════════════════════════════════════════════════

# Shared MPPI
NOISE_LEVEL = 0.3
TEMPERATURE = 0.00001
NUM_KNOTS = 16

# Staged rollout
NUM_KNOTS_PER_STAGE = 4
KDE_BANDWIDTH = 0.10
INVERSE_DENSITY_POWER = 0.5

# Learned value
VALUE_GRID_MIN = -10.0
VALUE_GRID_MAX = 10.0
HASHGRID_NUM_LEVELS = 8
HASHGRID_TABLE_SIZE = 4096
HASHGRID_MIN_RESOLUTION = 40.0
HASHGRID_MAX_RESOLUTION = 80.0
ONLINE_LEARNING_RATE = 1e-3
ONLINE_UPDATE_EPOCHS = 1
ONLINE_BATCH_SIZE = 2
ONLINE_ANCHOR_SAMPLES = 0
ONLINE_NEW_STATE_WEIGHT = 1.0
GOAL_VALUE = 0.0
GOAL_WEIGHT = 2000.0

# Pretraining
PRETRAIN_SAMPLE_COUNT = 100000
PRETRAIN_EPOCHS = 101
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_LEARNING_RATE = 1e-2
PRETRAIN_TARGET_SCALE = 1.0

# ════════════════════════════════════════════════════════════════════════


def _goal_xy(task: Ant) -> np.ndarray:
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id, :2], dtype=np.float32)


def _learned_value_controller(
    task: Ant,
    horizon: float,
    *,
    use_density_guided: bool,
    num_samples: Optional[int] = None,
) -> ValueGuidedMPPI:
    goal_xy = _goal_xy(task)
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)

    controller = ValueGuidedMPPI(
        task=task,
        num_samples=n,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=horizon,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=1,
        seed=0,
        state_dim=2,
        state_source_field="qpos",
        value_grid_min=VALUE_GRID_MIN,
        value_grid_max=VALUE_GRID_MAX,
        hashgrid_num_levels=HASHGRID_NUM_LEVELS,
        hashgrid_table_size=HASHGRID_TABLE_SIZE,
        hashgrid_min_resolution=HASHGRID_MIN_RESOLUTION,
        hashgrid_max_resolution=HASHGRID_MAX_RESOLUTION,
        online_learning_rate=ONLINE_LEARNING_RATE,
        online_update_epochs=ONLINE_UPDATE_EPOCHS,
        online_batch_size=ONLINE_BATCH_SIZE,
        online_anchor_samples=ONLINE_ANCHOR_SAMPLES,
        online_new_state_weight=ONLINE_NEW_STATE_WEIGHT,
        goal_state=goal_xy[None, :],
        goal_value=GOAL_VALUE,
        goal_weight=GOAL_WEIGHT,
        use_density_guided=use_density_guided,
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        kde_bandwidth=KDE_BANDWIDTH,
        inverse_density_power=INVERSE_DENSITY_POWER,
    )

    def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(VALUE_GRID_MIN, VALUE_GRID_MAX, size=(n, 2)).astype(np.float32)

    def target_function(states: np.ndarray) -> np.ndarray:
        diff = states - goal_xy[None, :]
        return PRETRAIN_TARGET_SCALE * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    controller.configure_value_pretraining(
        state_sampler=state_sampler,
        target_function=target_function,
        config=ValuePretrainConfig(
            sample_count=PRETRAIN_SAMPLE_COUNT,
            epochs=PRETRAIN_EPOCHS,
            batch_size=PRETRAIN_BATCH_SIZE,
            learning_rate=PRETRAIN_LEARNING_RATE,
            print_every=50,
        ),
    )
    controller.pretrained_value_key = "ant_learned_value"
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
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        kde_bandwidth=KDE_BANDWIDTH,
        inverse_density_power=INVERSE_DENSITY_POWER,
        state_dim=2,
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
    suite = SeniorThesisBenchmarkSuite(
        task_name="ant",
        task_factory=Ant,
        controller_specs=build_controller_specs(),
        config=SweepConfig(
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
            pretrain_weights_dir=str(Path(__file__).resolve().parent / "saved_pretrain_weights"),
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
                    "state_dim": 2,
                    "state_source_field": "qpos",
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
        ),
    )
    suite.run()


if __name__ == "__main__":
    main()
