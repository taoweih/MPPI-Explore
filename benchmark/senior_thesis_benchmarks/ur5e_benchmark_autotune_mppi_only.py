"""Senior thesis benchmark sweep for UR5e in mppi_mjwarp."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algs import (  # noqa: E402
    MPPI,
    MPPIStagedRollout,
    MPPIMemoryContinuous,
    MemoryPretrainConfig,
)
from benchmark.senior_thesis_benchmarks.benchmark_suite import (  # noqa: E402
    ControllerSpec,
    SeniorThesisBenchmarkSuite,
    SweepConfig,
)
from tasks import UR5e  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# Horizon sweep
HORIZONS = np.linspace(0.1, 0.4, 4)
NUM_SAMPLES_FOR_HORIZON_SWEEP = 512

# Sample sweep
NUM_SAMPLES_LIST = np.linspace(32, 2048, 4, dtype=int).tolist()
HORIZON_FOR_SAMPLE_SWEEP = 0.2

# Trial settings
NUM_TRIALS = 20
MAX_ITERATIONS = 1000
GOAL_THRESHOLD = 0.4
FREQUENCY = 50.0

# Output
RECORD_VIDEO = False
OUTPUT_TAG = "thesis"

# Parallelism — "sequential", "controllers", "axis", or "all"
PARALLEL = "all"
MAX_WORKERS = "12"  # int or "auto" (= total jobs in batch)
NUM_GPUS = 1  # number of GPUs for round-robin assignment (1 = single GPU)

# ════════════════════════════════════════════════════════════════════════
# CONTROLLER PARAMETERS — must match examples/ur5e.py
# ════════════════════════════════════════════════════════════════════════

# Shared MPPI
NOISE_LEVEL = 3.0
TEMPERATURE = 0.01
NUM_KNOTS = 8
SEED = 5

# Staged rollout
NUM_KNOTS_PER_STAGE = 2
KDE_BANDWIDTH = 0.30
INVERSE_DENSITY_POWER = 0.5
STATE_DIM = 3
STATE_SOURCE_FIELD = "site_xpos"

# Memory
MEMORY_GRID_MIN = -2.0
MEMORY_GRID_MAX = 2.0
HASHGRID_NUM_LEVELS = 8
HASHGRID_TABLE_SIZE = 262144
HASHGRID_MIN_RESOLUTION = 65.0
HASHGRID_MAX_RESOLUTION = 130.0
ONLINE_LEARNING_RATE = 1e-4
ONLINE_UPDATE_EPOCHS = 2
ONLINE_BATCH_SIZE = 2
ONLINE_ANCHOR_SAMPLES = 0
ONLINE_NEW_STATE_WEIGHT = 10.0
GOAL_VALUE = 0.0
GOAL_WEIGHT = 1000.0

# Pretraining
PRETRAIN_SAMPLE_COUNT = 10000000
PRETRAIN_EPOCHS = 101
PRETRAIN_BATCH_SIZE = 51200
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_TARGET_SCALE = 1.0

# ════════════════════════════════════════════════════════════════════════


def _goal_xyz(task: UR5e) -> np.ndarray:
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id], dtype=np.float32)


def _memory_controller(
    task: UR5e,
    horizon: float,
    *,
    use_staged_rollout: bool,
    num_samples: Optional[int] = None,
) -> MPPIMemoryContinuous:
    goal_xyz = _goal_xyz(task)
    ee_site_id = task.end_effector_pos_id
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)

    controller = MPPIMemoryContinuous(
        task=task,
        num_samples=n,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=horizon,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=1,
        seed=SEED,
        state_dim=STATE_DIM,
        state_source_field=STATE_SOURCE_FIELD,
        state_source_body_id=ee_site_id,
        memory_grid_min=MEMORY_GRID_MIN,
        memory_grid_max=MEMORY_GRID_MAX,
        hashgrid_num_levels=HASHGRID_NUM_LEVELS,
        hashgrid_table_size=HASHGRID_TABLE_SIZE,
        hashgrid_min_resolution=HASHGRID_MIN_RESOLUTION,
        hashgrid_max_resolution=HASHGRID_MAX_RESOLUTION,
        online_learning_rate=ONLINE_LEARNING_RATE,
        online_update_epochs=ONLINE_UPDATE_EPOCHS,
        online_batch_size=ONLINE_BATCH_SIZE,
        online_anchor_samples=ONLINE_ANCHOR_SAMPLES,
        online_new_state_weight=ONLINE_NEW_STATE_WEIGHT,
        goal_state=goal_xyz[None, :],
        goal_value=GOAL_VALUE,
        goal_weight=GOAL_WEIGHT,
        use_staged_rollout=use_staged_rollout,
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        kde_bandwidth=KDE_BANDWIDTH,
        inverse_density_power=INVERSE_DENSITY_POWER,
    )

    def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(MEMORY_GRID_MIN, MEMORY_GRID_MAX, size=(n, STATE_DIM)).astype(np.float32)

    def target_function(states: np.ndarray) -> np.ndarray:
        diff = states - goal_xyz[None, :]
        return PRETRAIN_TARGET_SCALE * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    controller.configure_pretraining(
        state_sampler=state_sampler,
        target_function=target_function,
        config=MemoryPretrainConfig(
            sample_count=PRETRAIN_SAMPLE_COUNT,
            epochs=PRETRAIN_EPOCHS,
            batch_size=PRETRAIN_BATCH_SIZE,
            learning_rate=PRETRAIN_LEARNING_RATE,
            print_every=50,
        ),
    )
    controller.pretrained_weights_key = "ur5e_memory"
    return controller


def _mppi_factory(task, h, num_samples=None):
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)
    return MPPI(
        task=task, num_samples=n, noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE, plan_horizon=h, spline_type="zero",
        num_knots=NUM_KNOTS, iterations=1, seed=SEED,
    )


def _staged_factory(task, h, num_samples=None):
    n = NUM_SAMPLES_FOR_HORIZON_SWEEP if num_samples is None else int(num_samples)
    return MPPIStagedRollout(
        task=task, num_samples=n, noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE, plan_horizon=h, spline_type="zero",
        num_knots=NUM_KNOTS, iterations=1, seed=SEED,
        num_knots_per_stage=NUM_KNOTS_PER_STAGE,
        kde_bandwidth=KDE_BANDWIDTH,
        inverse_density_power=INVERSE_DENSITY_POWER,
        state_dim=STATE_DIM,
        state_source_field=STATE_SOURCE_FIELD,
        state_source_body_id=task.end_effector_pos_id,
    )


def _memory_factory(task, h, num_samples=None):
    return _memory_controller(task, h, use_staged_rollout=False, num_samples=num_samples)


def _memory_staged_factory(task, h, num_samples=None):
    return _memory_controller(task, h, use_staged_rollout=True, num_samples=num_samples)


def build_controller_specs() -> list[ControllerSpec]:
    return [
        ControllerSpec(name="MPPI", factory=_mppi_factory),
        ControllerSpec(name="MPPI Density", factory=_staged_factory),
        ControllerSpec(name="MPPI Memory", factory=_memory_factory),
        ControllerSpec(name="MPPI Density + Memory", factory=_memory_staged_factory),
    ]


def main() -> None:
    suite = SeniorThesisBenchmarkSuite(
        task_name="ur5e",
        task_factory=UR5e,
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
                    "seed": SEED,
                    "num_samples_default": NUM_SAMPLES_FOR_HORIZON_SWEEP,
                },
                "staged": {
                    "num_knots_per_stage": NUM_KNOTS_PER_STAGE,
                    "kde_bandwidth": KDE_BANDWIDTH,
                    "inverse_density_power": INVERSE_DENSITY_POWER,
                    "state_dim": STATE_DIM,
                    "state_source_field": STATE_SOURCE_FIELD,
                },
                "memory": {
                    "memory_grid_min": MEMORY_GRID_MIN,
                    "memory_grid_max": MEMORY_GRID_MAX,
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
