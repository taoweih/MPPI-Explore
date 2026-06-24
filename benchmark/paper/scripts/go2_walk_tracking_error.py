"""Go2 walk sweeps using DIAL-MPC-style tracking error.

This focused test script matches the two-sweep format of the other benchmark
scripts, but replaces success rate and time-to-success with average tracking
error over the full task running period.

Primary metric is the raw walking-tracking error:

    body_xy_velocity_error + body_yaw_rate_error

where both terms are squared body-frame errors relative to the commanded
constant walking velocity/yaw-rate target, with the same 1s target ramp used by
the Go2 task.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

from algs import MPPI, DensityGuidedMPPI, KNNDensityModel
from benchmark.common.cli import add_output_root_arg
from benchmark.common.paths import runs_dir
from tasks.go2_walk import Go2Walk


# DIAL-MPC unitree_go2_trot.yaml task defaults.
PLANNING_DT = 0.02
FREQUENCY = 50.0
MAX_STEPS = 400
NUM_TRIALS = 10

TARGET_VX = 0.8
TARGET_VY = 0.0
TARGET_VYAW = 0.0
GAIT = "trot"

# Horizon sweep.
HORIZONS = np.linspace(0.16, 0.64, 10, dtype=np.float32)
NUM_SAMPLES_FOR_HORIZON_SWEEP = 2048

# Sample-count sweep.
NUM_SAMPLES_LIST = np.linspace(256, 4096, 10, dtype=int).tolist()
HORIZON_FOR_SAMPLE_SWEEP = 16 * PLANNING_DT

# Shared MPPI parameters.
NOISE_LEVEL = 1.0
TEMPERATURE = 0.05
NUM_KNOTS = 8
ITERATIONS = 1
SEED = 0

# Density-guided parameters.
DENSITY_NUM_KNOTS_PER_STAGE = 2
KNN_K = 10
INVERSE_DENSITY_POWER = 1.0
RESAMPLE_COST_WEIGHT = 1.0
RESAMPLE_COST_TEMPERATURE = None  # None uses the controller temperature.

OUTPUT_NAME = "go2_walk_tracking_error"


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    factory: Callable[[Go2Walk, float, int, int], object]


@dataclass
class SweepResult:
    axis_name: str
    axis_label: str
    axis_values: np.ndarray
    tracking_error_mean: np.ndarray
    tracking_error_std: np.ndarray
    frequency_mean: np.ndarray
    frequency_std: np.ndarray
    tracking_error_trials: np.ndarray
    frequency_trials: np.ndarray


def _body_xy_velocity(q: np.ndarray, vel: np.ndarray) -> tuple[float, float]:
    w, x, y, z = q
    vx, vy, vz = vel[:3]
    body_vx = (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + (2.0 * (x * y + w * z)) * vy
        + (2.0 * (x * z - w * y)) * vz
    )
    body_vy = (
        (2.0 * (x * y - w * z)) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + (2.0 * (y * z + w * x)) * vz
    )
    return float(body_vx), float(body_vy)


def _body_z_angular_velocity(q: np.ndarray, ang_vel: np.ndarray) -> float:
    w, x, y, z = q
    wx, wy, wz = ang_vel[:3]
    return float(
        (2.0 * (x * z + w * y)) * wx
        + (2.0 * (y * z - w * x)) * wy
        + (1.0 - 2.0 * (x * x + y * y)) * wz
    )


def dial_tracking_error(task: Go2Walk, mj_data: mujoco.MjData) -> float:
    """Compute raw velocity/yaw-rate tracking error mirrored from DIAL-MPC."""
    q = np.asarray(mj_data.xquat[task.base_body_id], dtype=np.float64)
    qvel = np.asarray(mj_data.qvel, dtype=np.float64)

    time_s = max(float(mj_data.time) - task.sim_dt, 0.0)
    ramp = np.clip(time_s / task.ramp_up_time, 0.0, 1.0)
    target_vx = task.target_vx * ramp
    target_vy = task.target_vy * ramp
    target_vyaw = task.target_vyaw * ramp

    body_vx, body_vy = _body_xy_velocity(q, qvel[:3])
    body_wz = _body_z_angular_velocity(q, qvel[3:6])

    linear_velocity = (body_vx - target_vx) ** 2 + (body_vy - target_vy) ** 2
    yaw_rate = (body_wz - target_vyaw) ** 2
    return float(linear_velocity + yaw_rate)


def _base_mppi_kwargs(
    task: Go2Walk,
    horizon: float,
    num_samples: int,
    seed: int,
) -> dict:
    return dict(
        task=task,
        num_samples=int(num_samples),
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=float(horizon),
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=ITERATIONS,
        seed=seed,
    )


def build_controller_specs() -> list[ControllerSpec]:
    return [
        ControllerSpec(
            "MPPI",
            lambda task, horizon, num_samples, seed: MPPI(
                **_base_mppi_kwargs(task, horizon, num_samples, seed)
            ),
        ),
        ControllerSpec(
            "Density-Guided MPPI (KNN)",
            lambda task, horizon, num_samples, seed: DensityGuidedMPPI(
                **_base_mppi_kwargs(task, horizon, num_samples, seed),
                density_model=KNNDensityModel(k=KNN_K, alpha=INVERSE_DENSITY_POWER),
                num_knots_per_stage=DENSITY_NUM_KNOTS_PER_STAGE,
                resample_cost_weight=RESAMPLE_COST_WEIGHT,
                resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
            ),
        ),
    ]


def make_task() -> Go2Walk:
    return Go2Walk(
        planning_dt=PLANNING_DT,
        sim_dt=PLANNING_DT,
        target_vx=TARGET_VX,
        target_vy=TARGET_VY,
        target_vyaw=TARGET_VYAW,
        gait=GAIT,
    )


def step_current_plan(controller, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    replan_period = 1.0 / FREQUENCY
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    tq = (
        np.arange(sim_steps_per_replan, dtype=np.float32)
        * mj_model.opt.timestep
        + mj_data.time
    )
    actions = controller.interp_func(tq, controller.tk, controller.mean[None, ...])[0]

    for action in actions:
        controller.task.apply_control_cpu(mj_data, action)
        mujoco.mj_step(mj_model, mj_data)


def run_trial(
    spec: ControllerSpec,
    *,
    horizon: float,
    num_samples: int,
    trial_idx: int,
    max_steps: int,
) -> tuple[float, float]:
    """Return mean raw tracking error and control Hz."""
    task = make_task()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    task.reset_to_home(mj_data)

    seed = SEED + trial_idx
    controller = spec.factory(task, float(horizon), int(num_samples), seed)
    if hasattr(controller, "reset"):
        controller.reset(seed=seed)

    controller.optimize(mj_data)
    controller.optimize(mj_data)
    if getattr(controller, "reset_after_warmup", False) and hasattr(controller, "reset"):
        controller.reset(seed=seed)
    task.reset_to_home(mj_data)

    act_before_plan = bool(getattr(controller, "act_before_plan", False))
    tracking_errors = np.zeros(max_steps, dtype=np.float32)

    total_plan_time = 0.0
    for step_idx in range(max_steps):
        if act_before_plan:
            step_current_plan(controller, mj_model, mj_data)
            start = time.perf_counter()
            controller.optimize(mj_data)
            total_plan_time += time.perf_counter() - start
        else:
            start = time.perf_counter()
            controller.optimize(mj_data)
            total_plan_time += time.perf_counter() - start
            step_current_plan(controller, mj_model, mj_data)

        tracking_errors[step_idx] = dial_tracking_error(task, mj_data)

    control_hz = max_steps / max(total_plan_time, 1e-9)
    return float(tracking_errors.mean()), float(control_hz)


def run_sweep(
    *,
    axis_name: str,
    axis_label: str,
    axis_values: np.ndarray,
    num_trials: int,
    max_steps: int,
    specs: list[ControllerSpec],
    fixed_horizon: Optional[float] = None,
    fixed_num_samples: Optional[int] = None,
) -> SweepResult:
    num_ctrl = len(specs)
    num_vals = len(axis_values)
    tracking_trials = np.zeros((num_ctrl, num_vals, num_trials), dtype=np.float32)
    frequency_trials = np.zeros_like(tracking_trials)

    total_jobs = num_ctrl * num_vals * num_trials
    job_idx = 0
    print(f"\n{axis_name} sweep: {num_vals} values x {num_ctrl} controllers x {num_trials} trials")

    for value_idx, axis_value in enumerate(axis_values):
        print(f"\n=== {axis_label} {float(axis_value):.3g} ===")
        for ctrl_idx, spec in enumerate(specs):
            for trial_idx in range(num_trials):
                job_idx += 1
                horizon = float(axis_value) if fixed_horizon is None else float(fixed_horizon)
                num_samples = (
                    int(axis_value)
                    if fixed_num_samples is None
                    else int(fixed_num_samples)
                )
                print(
                    f"  [{job_idx}/{total_jobs}] {spec.name}, "
                    f"trial {trial_idx + 1}/{num_trials}"
                )
                tracking, hz = run_trial(
                    spec,
                    horizon=horizon,
                    num_samples=num_samples,
                    trial_idx=trial_idx,
                    max_steps=max_steps,
                )
                tracking_trials[ctrl_idx, value_idx, trial_idx] = tracking
                frequency_trials[ctrl_idx, value_idx, trial_idx] = hz
                print(f"    tracking={tracking:.5g}, freq={hz:.1f} Hz")

    return SweepResult(
        axis_name=axis_name,
        axis_label=axis_label,
        axis_values=axis_values,
        tracking_error_mean=tracking_trials.mean(axis=2),
        tracking_error_std=tracking_trials.std(axis=2),
        frequency_mean=frequency_trials.mean(axis=2),
        frequency_std=frequency_trials.std(axis=2),
        tracking_error_trials=tracking_trials,
        frequency_trials=frequency_trials,
    )


def _plot_matrix(
    out_path: Path,
    specs: list[ControllerSpec],
    x_values: np.ndarray,
    values: np.ndarray,
    std: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    for idx, spec in enumerate(specs):
        line, = plt.plot(x_values, values[idx], label=spec.name)
        plt.fill_between(
            x_values,
            values[idx] - std[idx],
            values[idx] + std[idx],
            alpha=0.2,
            color=line.get_color(),
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_sweep(out_dir: Path, specs: list[ControllerSpec], result: SweepResult) -> None:
    prefix = result.axis_name
    np.savetxt(
        out_dir / f"{prefix}_tracking_error_mean.csv",
        result.tracking_error_mean,
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"{prefix}_tracking_error_std.csv",
        result.tracking_error_std,
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"{prefix}_frequency_mean.csv",
        result.frequency_mean,
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"{prefix}_frequency_std.csv",
        result.frequency_std,
        delimiter=",",
    )

    _plot_matrix(
        out_dir / f"{prefix}_tracking_error.png",
        specs,
        result.axis_values,
        result.tracking_error_mean,
        result.tracking_error_std,
        xlabel=result.axis_label,
        ylabel="Average Tracking Error",
        title=f"go2_walk: Tracking Error vs {result.axis_label}",
    )
    _plot_matrix(
        out_dir / f"{prefix}_frequency.png",
        specs,
        result.axis_values,
        result.frequency_mean,
        result.frequency_std,
        xlabel=result.axis_label,
        ylabel="Control Frequency (Hz)",
        title=f"go2_walk: Control Frequency vs {result.axis_label}",
    )


def save_outputs(
    out_dir: Path,
    specs: list[ControllerSpec],
    horizon_result: SweepResult,
    sample_result: SweepResult,
    *,
    max_steps: int,
    num_trials: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    save_sweep(out_dir, specs, horizon_result)
    save_sweep(out_dir, specs, sample_result)

    controller_names = np.asarray([spec.name for spec in specs], dtype=object)
    np.savez(
        out_dir / "summary.npz",
        controller_names=controller_names,
        horizon_values=horizon_result.axis_values,
        horizon_tracking_error_mean=horizon_result.tracking_error_mean,
        horizon_tracking_error_std=horizon_result.tracking_error_std,
        horizon_frequency_mean=horizon_result.frequency_mean,
        horizon_frequency_std=horizon_result.frequency_std,
        horizon_tracking_error_trials=horizon_result.tracking_error_trials,
        num_samples_values=sample_result.axis_values,
        num_samples_tracking_error_mean=sample_result.tracking_error_mean,
        num_samples_tracking_error_std=sample_result.tracking_error_std,
        num_samples_frequency_mean=sample_result.frequency_mean,
        num_samples_frequency_std=sample_result.frequency_std,
        num_samples_tracking_error_trials=sample_result.tracking_error_trials,
    )

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write(
            "sweep,axis_value,controller,tracking_error_mean,tracking_error_std,"
            "frequency_mean,frequency_std\n"
        )
        for result in (horizon_result, sample_result):
            for value_idx, axis_value in enumerate(result.axis_values):
                for ctrl_idx, spec in enumerate(specs):
                    f.write(
                        f"{result.axis_name},{float(axis_value):.8g},{spec.name},"
                        f"{result.tracking_error_mean[ctrl_idx, value_idx]:.8g},"
                        f"{result.tracking_error_std[ctrl_idx, value_idx]:.8g},"
                        f"{result.frequency_mean[ctrl_idx, value_idx]:.8g},"
                        f"{result.frequency_std[ctrl_idx, value_idx]:.8g}\n"
                    )

    metadata = {
        "task_name": "go2_walk",
        "output_name": OUTPUT_NAME,
        "controller_names": [spec.name for spec in specs],
        "num_trials": num_trials,
        "max_steps": max_steps,
        "planning_dt": PLANNING_DT,
        "frequency": FREQUENCY,
        "target_vx": TARGET_VX,
        "target_vy": TARGET_VY,
        "target_vyaw": TARGET_VYAW,
        "gait": GAIT,
        "horizons": horizon_result.axis_values.tolist(),
        "num_samples_list": sample_result.axis_values.tolist(),
        "sweep_horizon_for_samples": HORIZON_FOR_SAMPLE_SWEEP,
        "num_samples_for_horizon_sweep": NUM_SAMPLES_FOR_HORIZON_SWEEP,
        "controller_params": {
            "shared": {
                "noise_level": NOISE_LEVEL,
                "temperature": TEMPERATURE,
                "num_knots": NUM_KNOTS,
                "iterations": ITERATIONS,
                "seed": SEED,
            },
            "density": {
                "num_knots_per_stage": DENSITY_NUM_KNOTS_PER_STAGE,
                "knn_k": KNN_K,
                "inverse_density_power": INVERSE_DENSITY_POWER,
                "resample_cost_weight": RESAMPLE_COST_WEIGHT,
                "resample_cost_temperature": RESAMPLE_COST_TEMPERATURE,
            },
        },
        "metric": {
            "primary": "tracking_error_mean",
            "aggregation": "mean over every control step in the task running period",
            "definition": (
                "linear_velocity_tracking_error + yaw_rate_tracking_error, "
                "where linear_velocity_tracking_error is squared body-frame "
                "XY velocity error and yaw_rate_tracking_error is squared "
                "body-frame yaw-rate error."
            ),
            "source": (
                "DIAL-MPC walking-tracking task: tracking desired linear "
                "velocity and yaw rate."
            ),
        },
        "dial_mpc_reference": {
            "paper": "https://arxiv.org/abs/2409.15610",
            "code": "https://github.com/LeCAR-Lab/dial-mpc/blob/main/dial_mpc/envs/unitree_go2_env.py",
            "config": "https://github.com/LeCAR-Lab/dial-mpc/blob/main/dial_mpc/examples/unitree_go2_trot.yaml",
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_output_root_arg(parser)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    args = parser.parse_args()

    horizons = np.asarray(HORIZONS, dtype=np.float32)
    num_samples_list = np.asarray(NUM_SAMPLES_LIST, dtype=np.int32)
    max_steps = int(args.max_steps)
    num_trials = int(args.num_trials)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = runs_dir("paper", output_root=args.output_root) / (
        f"{OUTPUT_NAME}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_controller_specs()

    print(f"\n{'=' * 60}")
    print("Go2 walk tracking-error benchmark")
    print(f"Controllers: {', '.join(spec.name for spec in specs)}")
    print(f"Trials per point: {num_trials}")
    print(f"Task steps per trial: {max_steps}")
    print(f"Horizon sweep: {len(horizons)} values @ {NUM_SAMPLES_FOR_HORIZON_SWEEP} samples")
    print(
        f"Sample sweep: {len(num_samples_list)} values "
        f"@ horizon={HORIZON_FOR_SAMPLE_SWEEP:.3f}s"
    )
    print(f"{'=' * 60}")

    horizon_result = run_sweep(
        axis_name="horizon",
        axis_label="Horizon (s)",
        axis_values=horizons,
        num_trials=num_trials,
        max_steps=max_steps,
        specs=specs,
        fixed_num_samples=NUM_SAMPLES_FOR_HORIZON_SWEEP,
    )
    sample_result = run_sweep(
        axis_name="num_samples",
        axis_label="Number of Samples",
        axis_values=num_samples_list,
        num_trials=num_trials,
        max_steps=max_steps,
        specs=specs,
        fixed_horizon=HORIZON_FOR_SAMPLE_SWEEP,
    )
    save_outputs(
        out_dir,
        specs,
        horizon_result,
        sample_result,
        max_steps=max_steps,
        num_trials=num_trials,
    )

    print(f"\nSaved Go2 tracking-error sweeps to {out_dir}")


if __name__ == "__main__":
    main()
