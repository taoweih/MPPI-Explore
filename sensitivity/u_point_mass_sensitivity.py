"""Sensitivity sweeps on Density-Guided + Value-Guided MPPI hyperparameters
for the U-shaped point mass task.

Holds (plan_horizon, num_samples) fixed at the existing benchmark's "good
operating point" and varies ONE algorithm hyperparameter per sweep.

Sweeps run sequentially.  Comment out entries in `build_sweeps()` to skip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algs import (  # noqa: E402
    MPPI,
    DensityGuidedMPPI,
    ValueGuidedMPPI,
    ValuePretrainConfig,
)
from sensitivity.sensitivity_suite import (  # noqa: E402
    ParamSweep,
    SensitivityConfig,
    SensitivitySuite,
)
from tasks import UPointMass  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
# FIXED OPERATING POINT — held constant across all sweeps
# ════════════════════════════════════════════════════════════════════════

PLAN_HORIZON = 1.5      # matches HORIZON_FOR_SAMPLE_SWEEP in the (T,K) benchmark
NUM_SAMPLES = 512       # matches NUM_SAMPLES_FOR_HORIZON_SWEEP

# Trial settings
NUM_TRIALS = 50
MAX_ITERATIONS = 2000
GOAL_THRESHOLD = 0.4
FREQUENCY = 50.0
OUTPUT_TAG = "u_point_mass"

# ════════════════════════════════════════════════════════════════════════
# DEFAULTS for non-swept hyperparameters
# (matches benchmark/senior_thesis_benchmarks/u_point_mass_benchmark_autotune_mppi_only.py)
# ════════════════════════════════════════════════════════════════════════

# Shared MPPI
NOISE_LEVEL = 3.0
TEMPERATURE = 0.001
NUM_KNOTS = 16

# Density-guided
DEFAULT_NUM_KNOTS_PER_STAGE = 4
DEFAULT_KDE_BANDWIDTH = 0.15
DEFAULT_INVERSE_DENSITY_POWER = 0.5

# Value-guided architecture
VALUE_GRID_MIN = -1.0
VALUE_GRID_MAX = 1.0
DEFAULT_HASHGRID_NUM_LEVELS = 4
DEFAULT_HASHGRID_TABLE_SIZE = 4096
DEFAULT_HASHGRID_MIN_RESOLUTION = 50.0
DEFAULT_HASHGRID_MAX_RESOLUTION = 200.0

# Value-guided online learning
DEFAULT_ONLINE_LEARNING_RATE = 5e-3
DEFAULT_ONLINE_UPDATE_EPOCHS = 1
DEFAULT_ONLINE_BATCH_SIZE = 1
DEFAULT_ONLINE_NEW_STATE_WEIGHT = 10.0
GOAL_VALUE = 0.0
GOAL_WEIGHT = 2000.0

# Pretraining (used by value-guided sweeps)
PRETRAIN_SAMPLE_COUNT = 100000
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_TARGET_SCALE = 1.0


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _goal_xy(task: UPointMass) -> np.ndarray:
    """Read the goal body's xy from a freshly-stepped MjData."""
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id, :2], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════
# Controller factories — accept overrides for the swept parameter
# ════════════════════════════════════════════════════════════════════════


def _baseline_mppi_factory(task: UPointMass) -> MPPI:
    """Plain MPPI at the same fixed (T, K) — used as the reference line."""
    return MPPI(
        task=task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=1,
        seed=0,
    )


def _density_factory(task: UPointMass, **overrides) -> DensityGuidedMPPI:
    return DensityGuidedMPPI(
        task=task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=1,
        seed=0,
        num_knots_per_stage=int(overrides.get("num_knots_per_stage", DEFAULT_NUM_KNOTS_PER_STAGE)),
        kde_bandwidth=float(overrides.get("kde_bandwidth", DEFAULT_KDE_BANDWIDTH)),
        inverse_density_power=float(overrides.get("inverse_density_power", DEFAULT_INVERSE_DENSITY_POWER)),
    )


def _value_factory(task: UPointMass, **overrides) -> ValueGuidedMPPI:
    """Build a Value-Guided MPPI and pretrain its value model in-place.

    Pretraining always runs because architecture-changing sweeps
    (hashgrid_table_size / hashgrid_num_levels) require fresh weights.
    For non-architectural sweeps the cost is the same — pretraining for
    100k samples × 100 epochs is fast (a few seconds on GPU).
    """
    goal_xy = _goal_xy(task)

    controller = ValueGuidedMPPI(
        task=task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=1,
        seed=0,
        # Hash-grid architecture
        value_grid_min=VALUE_GRID_MIN,
        value_grid_max=VALUE_GRID_MAX,
        hashgrid_num_levels=int(overrides.get("hashgrid_num_levels", DEFAULT_HASHGRID_NUM_LEVELS)),
        hashgrid_table_size=int(overrides.get("hashgrid_table_size", DEFAULT_HASHGRID_TABLE_SIZE)),
        hashgrid_min_resolution=float(overrides.get("hashgrid_min_resolution", DEFAULT_HASHGRID_MIN_RESOLUTION)),
        hashgrid_max_resolution=float(overrides.get("hashgrid_max_resolution", DEFAULT_HASHGRID_MAX_RESOLUTION)),
        # Online learning
        online_learning_rate=float(overrides.get("online_learning_rate", DEFAULT_ONLINE_LEARNING_RATE)),
        online_update_epochs=int(overrides.get("online_update_epochs", DEFAULT_ONLINE_UPDATE_EPOCHS)),
        online_batch_size=DEFAULT_ONLINE_BATCH_SIZE,
        online_anchor_samples=0,
        online_new_state_weight=DEFAULT_ONLINE_NEW_STATE_WEIGHT,
        goal_state=goal_xy[None, :],
        goal_value=GOAL_VALUE,
        goal_weight=GOAL_WEIGHT,
        # Density-compose disabled for these value-only sweeps
        use_density_guided=False,
        num_knots_per_stage=DEFAULT_NUM_KNOTS_PER_STAGE,
        kde_bandwidth=DEFAULT_KDE_BANDWIDTH,
        inverse_density_power=DEFAULT_INVERSE_DENSITY_POWER,
        disable_replay_anchors=True,
    )

    def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(VALUE_GRID_MIN, VALUE_GRID_MAX,
                           size=(n, task.state_dim)).astype(np.float32)

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
            print_every=PRETRAIN_EPOCHS,  # quiet
        ),
    )
    controller.pretrain_learned_value(verbose=False)
    return controller


# ════════════════════════════════════════════════════════════════════════
# Sweep definitions
# ════════════════════════════════════════════════════════════════════════


def build_sweeps() -> list[ParamSweep]:
    return [
        # ── Density-Guided ──────────────────────────────────────────
        ParamSweep(
            name="kde_bandwidth",
            axis_label="KDE Bandwidth (σ)",
            controller_name="Density-Guided MPPI",
            values=np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40], dtype=np.float32),
            factory=lambda task, v: _density_factory(task, kde_bandwidth=v),
        ),
        ParamSweep(
            name="inverse_density_power",
            axis_label="Inverse Density Power (α)",
            controller_name="Density-Guided MPPI",
            values=np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float32),
            factory=lambda task, v: _density_factory(task, inverse_density_power=v),
        ),
        ParamSweep(
            name="num_knots_per_stage",
            axis_label="Knots per Stage  (num_stages = 16 / this)",
            controller_name="Density-Guided MPPI",
            # num_knots = 16, so [2, 4, 8, 16] gives [8, 4, 2, 1] stages
            values=np.array([2, 4, 8, 16], dtype=np.int32),
            factory=lambda task, v: _density_factory(task, num_knots_per_stage=int(v)),
        ),

        # ── Value-Guided  (re-pretrains for arch-changing sweeps) ───
        ParamSweep(
            name="hashgrid_table_size",
            axis_label="Hashgrid Table Size",
            controller_name="Value-Guided MPPI",
            values=np.array([256, 1024, 4096, 16384, 65536, 262144], dtype=np.int32),
            factory=lambda task, v: _value_factory(task, hashgrid_table_size=int(v)),
            log_x=True,
        ),
        ParamSweep(
            name="hashgrid_num_levels",
            axis_label="Hashgrid Resolution Levels",
            controller_name="Value-Guided MPPI",
            values=np.array([2, 4, 8, 16], dtype=np.int32),
            factory=lambda task, v: _value_factory(task, hashgrid_num_levels=int(v)),
        ),
        ParamSweep(
            name="online_learning_rate",
            axis_label="Online Learning Rate",
            controller_name="Value-Guided MPPI",
            values=np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2], dtype=np.float32),
            factory=lambda task, v: _value_factory(task, online_learning_rate=float(v)),
            log_x=True,
        ),
        ParamSweep(
            name="online_update_epochs",
            axis_label="Online Update Epochs",
            controller_name="Value-Guided MPPI",
            values=np.array([1, 2, 5, 10], dtype=np.int32),
            factory=lambda task, v: _value_factory(task, online_update_epochs=int(v)),
        ),
    ]


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main() -> None:
    suite = SensitivitySuite(
        task_name="u_point_mass",
        task_factory=UPointMass,
        sweeps=build_sweeps(),
        config=SensitivityConfig(
            num_trials=NUM_TRIALS,
            frequency=FREQUENCY,
            goal_threshold=GOAL_THRESHOLD,
            max_iterations=MAX_ITERATIONS,
            output_tag=OUTPUT_TAG,
        ),
        baseline_factory=_baseline_mppi_factory,
        baseline_name="MPPI",
    )
    suite.run()


if __name__ == "__main__":
    main()
