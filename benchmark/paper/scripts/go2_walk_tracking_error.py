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
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import mujoco
import numpy as np
from tqdm import tqdm

from algs import MPPI, DIALMPC, DensityGuidedMPPI, KNNDensityModel
from benchmark.common.cli import add_output_root_arg
from benchmark.common.paths import runs_dir
from benchmark.senior_thesis.scripts.benchmark_suite import _mps_start, _mps_stop
from simulation.asyncrousnous import _AsyncPlanner, _step_latest_async_action
from tasks.go2_walk import Go2Walk


_WORKER_FLAG = "--_benchmark_worker"

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

# DIAL-MPC parameters. The shared MPPI parameters above intentionally define
# DIAL's sample count, horizon, noise scale, temperature, knots, and diffusion
# updates; these match this benchmark's MPPI convention. The remaining knobs use
# the DIAL-MPC paper-style defaults for Go2 trot.
DIAL_NUM_INITIAL_DIFFUSION_STEPS = 10
DIAL_HORIZON_DIFFUSE_FACTOR = 0.9
DIAL_TRAJ_DIFFUSE_FACTOR = 0.5
DIAL_INCLUDE_MEAN_SAMPLE = True
DIAL_FIX_FIRST_KNOT = True
DIAL_NORMALIZE_COSTS = True
DIAL_COST_NORMALIZATION_EPS = 1.0e-6

# Density-guided parameters.
DENSITY_NUM_KNOTS_PER_STAGE = 2
KNN_K = 10
INVERSE_DENSITY_POWER = 1.0
RESAMPLE_COST_WEIGHT = 1.0
RESAMPLE_COST_TEMPERATURE = None  # None uses unit temperature for normalized cost scores.

OUTPUT_NAME = "go2_walk_tracking_error_mppi_density_dial"

# Parallelism: "sequential", "controllers", "axis", or "all".
PARALLEL = "all"
MAX_WORKERS = "30"  # int or "auto" (= total jobs in batch)
NUM_GPUS = 2
FREQ_CALIBRATION_ITERS = 50


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
        ControllerSpec(
            "DIAL-MPC",
            lambda task, horizon, num_samples, seed: DIALMPC(
                **_base_mppi_kwargs(task, horizon, num_samples, seed),
                num_initial_diffusion_steps=DIAL_NUM_INITIAL_DIFFUSION_STEPS,
                horizon_diffuse_factor=DIAL_HORIZON_DIFFUSE_FACTOR,
                traj_diffuse_factor=DIAL_TRAJ_DIFFUSE_FACTOR,
                include_mean_sample=DIAL_INCLUDE_MEAN_SAMPLE,
                fix_first_knot=DIAL_FIX_FIRST_KNOT,
                normalize_costs=DIAL_NORMALIZE_COSTS,
                cost_normalization_eps=DIAL_COST_NORMALIZATION_EPS,
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
    simulation_mode: Literal["deterministic", "async"] = "deterministic",
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

    if simulation_mode == "async":
        if hasattr(controller, "warm_start"):
            controller.warm_start(float(mj_data.time))

        planner = _AsyncPlanner(controller, mj_model, mj_data)
        tracking_errors = np.zeros(max_steps, dtype=np.float32)

        replan_period = 1.0 / FREQUENCY
        sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
        step_dt = float(sim_steps_per_replan * mj_model.opt.timestep)
        trial_start = time.perf_counter()
        trial_elapsed = 0.0
        trial_completed = 0
        try:
            planner.start(wait_for_first=True)
            for step_idx in range(max_steps):
                loop_start = time.perf_counter()

                planner.poll()
                _step_latest_async_action(
                    planner,
                    mj_model,
                    mj_data,
                    sim_steps_per_replan,
                )
                planner.poll()

                tracking_errors[step_idx] = dial_tracking_error(task, mj_data)

                elapsed = time.perf_counter() - loop_start
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)
            trial_elapsed = time.perf_counter() - trial_start
        finally:
            if trial_elapsed == 0.0:
                trial_elapsed = time.perf_counter() - trial_start
            trial_completed = planner.stats.completed
            planner.drain(commit=False)
            planner.close()

        control_hz = trial_completed / max(trial_elapsed, 1e-9)
        return float(tracking_errors.mean()), float(control_hz)

    if simulation_mode != "deterministic":
        raise ValueError(
            f"Unknown simulation_mode={simulation_mode!r}; "
            "expected 'deterministic' or 'async'."
        )

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


def run_point(
    spec: ControllerSpec,
    *,
    horizon: float,
    num_samples: int,
    num_trials: int,
    max_steps: int,
    simulation_mode: Literal["deterministic", "async"] = "deterministic",
) -> tuple[np.ndarray, np.ndarray]:
    tracking_trials = np.zeros(num_trials, dtype=np.float32)
    frequency_trials = np.zeros(num_trials, dtype=np.float32)
    for trial_idx in range(num_trials):
        tracking, hz = run_trial(
            spec,
            horizon=horizon,
            num_samples=num_samples,
            trial_idx=trial_idx,
            max_steps=max_steps,
            simulation_mode=simulation_mode,
        )
        tracking_trials[trial_idx] = tracking
        frequency_trials[trial_idx] = hz
    return tracking_trials, frequency_trials


def _save_point_result(
    path: str,
    tracking_trials: np.ndarray,
    frequency_trials: np.ndarray,
) -> None:
    np.savez(
        path,
        tracking_trials=tracking_trials,
        frequency_trials=frequency_trials,
    )


def _load_point_result(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["tracking_trials"].copy(), data["frequency_trials"].copy()


def _worker_module_name() -> str:
    if __spec__ is not None and __spec__.name is not None:
        return __spec__.name
    return __name__


def _resolve_max_workers(
    max_workers: Union[int, Literal["auto"]],
    num_jobs: int,
) -> int:
    if max_workers == "auto":
        return num_jobs
    return min(int(max_workers), num_jobs)


def _assign_worker_gpu(env: dict[str, str], gpu_id: int, num_gpus: int) -> None:
    if num_gpus <= 1:
        return
    parent_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_devices is None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return
    devices = [
        device.strip()
        for device in parent_devices.split(",")
        if device.strip()
    ]
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices[gpu_id % len(devices)]


def _launch_point_subprocess(
    *,
    module_name: str,
    ctrl_idx: int,
    horizon: float,
    num_samples: int,
    num_trials: int,
    max_steps: int,
    simulation_mode: Literal["deterministic", "async"],
    output_path: str,
    gpu_id: int,
    num_gpus: int,
    warm_cache_dir: Optional[str],
) -> tuple[np.ndarray, np.ndarray]:
    cmd = [
        sys.executable,
        "-m",
        module_name,
        _WORKER_FLAG,
        "--ctrl-idx",
        str(ctrl_idx),
        "--horizon",
        str(horizon),
        "--num-samples",
        str(num_samples),
        "--num-trials",
        str(num_trials),
        "--max-steps",
        str(max_steps),
        "--simulation",
        simulation_mode,
        "--output",
        output_path,
    ]
    env = os.environ.copy()
    _assign_worker_gpu(env, gpu_id, num_gpus)

    worker_cache = tempfile.mkdtemp(prefix="go2_warp_worker_")
    if warm_cache_dir is not None:
        shutil.copytree(warm_cache_dir, worker_cache, dirs_exist_ok=True)
    env["WARP_CACHE_PATH"] = worker_cache

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
            raise RuntimeError(
                f"worker failed for controller {ctrl_idx}, horizon={horizon}, "
                f"num_samples={num_samples}:\n{stderr_tail}"
            )
        return _load_point_result(output_path)
    finally:
        shutil.rmtree(worker_cache, ignore_errors=True)


def _warmup_warp_cache(
    *,
    module_name: str,
    specs: list[ControllerSpec],
    horizon: float,
    num_samples: int,
    simulation_mode: Literal["deterministic", "async"],
) -> Optional[str]:
    cache_dir = tempfile.mkdtemp(prefix="go2_warp_warm_")
    env = os.environ.copy()
    env["WARP_CACHE_PATH"] = cache_dir
    print("Warming up Warp kernel cache...")

    for ctrl_idx, spec in enumerate(specs):
        output_path = os.path.join(cache_dir, f"warmup_{ctrl_idx}.npz")
        cmd = [
            sys.executable,
            "-m",
            module_name,
            _WORKER_FLAG,
            "--ctrl-idx",
            str(ctrl_idx),
            "--horizon",
            str(horizon),
            "--num-samples",
            str(num_samples),
            "--num-trials",
            "1",
            "--max-steps",
            "5",
            "--simulation",
            simulation_mode,
            "--output",
            output_path,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
            print(f"WARNING: Warp warmup failed for {spec.name}:\n{stderr_tail}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None
        os.remove(output_path)

    print("Warp kernel cache ready.")
    return cache_dir


def _run_as_worker() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(_WORKER_FLAG, action="store_true")
    parser.add_argument("--ctrl-idx", type=int, required=True)
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--simulation",
        "--sim",
        dest="simulation_mode",
        default="deterministic",
        choices=("deterministic", "async"),
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    specs = build_controller_specs()
    tracking_trials, frequency_trials = run_point(
        specs[args.ctrl_idx],
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        simulation_mode=args.simulation_mode,
    )
    _save_point_result(args.output, tracking_trials, frequency_trials)


def _point_parameters(
    axis_value: float,
    *,
    fixed_horizon: Optional[float],
    fixed_num_samples: Optional[int],
) -> tuple[float, int]:
    horizon = float(axis_value) if fixed_horizon is None else float(fixed_horizon)
    num_samples = (
        int(axis_value) if fixed_num_samples is None else int(fixed_num_samples)
    )
    return horizon, num_samples


def run_sweep(
    *,
    axis_name: str,
    axis_label: str,
    axis_values: np.ndarray,
    num_trials: int,
    max_steps: int,
    specs: list[ControllerSpec],
    parallel: Literal["sequential", "controllers", "axis", "all"],
    max_workers: Union[int, Literal["auto"]],
    num_gpus: int,
    simulation_mode: Literal["deterministic", "async"],
    fixed_horizon: Optional[float] = None,
    fixed_num_samples: Optional[int] = None,
) -> SweepResult:
    num_ctrl = len(specs)
    num_vals = len(axis_values)
    tracking_trials = np.zeros((num_ctrl, num_vals, num_trials), dtype=np.float32)
    frequency_trials = np.zeros_like(tracking_trials)
    all_jobs = [
        (ctrl_idx, value_idx, spec, float(axis_value))
        for value_idx, axis_value in enumerate(axis_values)
        for ctrl_idx, spec in enumerate(specs)
    ]

    if parallel == "sequential":
        print(
            f"\nSequential {axis_name} sweep: {num_vals} values x "
            f"{num_ctrl} controllers = {len(all_jobs)} jobs"
        )
        for ctrl_idx, value_idx, spec, axis_value in tqdm(
            all_jobs,
            desc=f"{axis_name} sweep",
            unit="job",
        ):
            horizon, num_samples = _point_parameters(
                axis_value,
                fixed_horizon=fixed_horizon,
                fixed_num_samples=fixed_num_samples,
            )
            tracking, frequency = run_point(
                spec,
                horizon=horizon,
                num_samples=num_samples,
                num_trials=num_trials,
                max_steps=max_steps,
                simulation_mode=simulation_mode,
            )
            tracking_trials[ctrl_idx, value_idx] = tracking
            frequency_trials[ctrl_idx, value_idx] = frequency
    else:
        if parallel == "controllers":
            batches = [
                [job for job in all_jobs if job[1] == value_idx]
                for value_idx in range(num_vals)
            ]
        elif parallel == "axis":
            batches = [
                [job for job in all_jobs if job[0] == ctrl_idx]
                for ctrl_idx in range(num_ctrl)
            ]
        else:
            batches = [all_jobs]

        module_name = _worker_module_name()
        first_horizon, first_num_samples = _point_parameters(
            float(axis_values[0]),
            fixed_horizon=fixed_horizon,
            fixed_num_samples=fixed_num_samples,
        )
        warm_cache_dir = _warmup_warp_cache(
            module_name=module_name,
            specs=specs,
            horizon=first_horizon,
            num_samples=first_num_samples,
            simulation_mode=simulation_mode,
        )
        gpu_info = f", {num_gpus} GPU(s)" if num_gpus > 1 else ""
        print(
            f"\nParallel {axis_name} sweep ({parallel}): {len(all_jobs)} jobs "
            f"across {len(batches)} batch(es){gpu_info}"
        )

        progress = tqdm(total=len(all_jobs), desc=f"{axis_name} sweep", unit="job")
        try:
            for batch_idx, batch in enumerate(batches):
                workers = _resolve_max_workers(max_workers, len(batch))
                if len(batches) > 1:
                    tqdm.write(
                        f"\n--- Batch {batch_idx + 1}/{len(batches)} "
                        f"({len(batch)} jobs, {workers} workers) ---"
                    )
                else:
                    tqdm.write(f"Running {len(batch)} jobs with {workers} workers")

                result_dir = tempfile.mkdtemp(prefix="go2_bench_")
                failures: list[str] = []
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {}
                        for job_idx, (
                            ctrl_idx,
                            value_idx,
                            spec,
                            axis_value,
                        ) in enumerate(batch):
                            horizon, num_samples = _point_parameters(
                                axis_value,
                                fixed_horizon=fixed_horizon,
                                fixed_num_samples=fixed_num_samples,
                            )
                            output_path = os.path.join(
                                result_dir,
                                f"result_{ctrl_idx}_{value_idx}.npz",
                            )
                            future = pool.submit(
                                _launch_point_subprocess,
                                module_name=module_name,
                                ctrl_idx=ctrl_idx,
                                horizon=horizon,
                                num_samples=num_samples,
                                num_trials=num_trials,
                                max_steps=max_steps,
                                simulation_mode=simulation_mode,
                                output_path=output_path,
                                gpu_id=job_idx % max(num_gpus, 1),
                                num_gpus=num_gpus,
                                warm_cache_dir=warm_cache_dir,
                            )
                            futures[future] = (
                                ctrl_idx,
                                value_idx,
                                spec.name,
                                axis_value,
                            )

                        for future in as_completed(futures):
                            ctrl_idx, value_idx, name, axis_value = futures[future]
                            progress.update(1)
                            try:
                                tracking, frequency = future.result()
                            except Exception as exc:
                                failures.append(
                                    f"{name} @ {axis_name}={axis_value}: {exc}"
                                )
                                continue
                            tracking_trials[ctrl_idx, value_idx] = tracking
                            frequency_trials[ctrl_idx, value_idx] = frequency
                            tqdm.write(
                                f"  done: {name} @ {axis_name}={axis_value} "
                                f"(tracking={tracking.mean():.5g})"
                            )
                finally:
                    shutil.rmtree(result_dir, ignore_errors=True)

                if failures:
                    raise RuntimeError(
                        "One or more benchmark workers failed:\n"
                        + "\n".join(failures)
                    )
        finally:
            progress.close()
            if warm_cache_dir is not None:
                shutil.rmtree(warm_cache_dir, ignore_errors=True)

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


def calibrate_frequency(
    result: SweepResult,
    *,
    specs: list[ControllerSpec],
    num_trials: int,
    calibration_steps: int,
    simulation_mode: Literal["deterministic", "async"],
    fixed_horizon: Optional[float] = None,
    fixed_num_samples: Optional[int] = None,
) -> None:
    total_jobs = len(specs) * len(result.axis_values)
    print(f"\n{'-' * 60}")
    print(
        f"Frequency calibration: {num_trials} trials x {calibration_steps} steps, "
        "sequential (exclusive GPU)"
    )
    print(f"{total_jobs} jobs total")
    print(f"{'-' * 60}")

    jobs = [
        (ctrl_idx, value_idx, spec, float(axis_value))
        for value_idx, axis_value in enumerate(result.axis_values)
        for ctrl_idx, spec in enumerate(specs)
    ]
    for ctrl_idx, value_idx, spec, axis_value in tqdm(
        jobs,
        desc=f"{result.axis_name} frequency calibration",
        unit="job",
    ):
        horizon, num_samples = _point_parameters(
            axis_value,
            fixed_horizon=fixed_horizon,
            fixed_num_samples=fixed_num_samples,
        )
        _, frequencies = run_point(
            spec,
            horizon=horizon,
            num_samples=num_samples,
            num_trials=num_trials,
            max_steps=calibration_steps,
            simulation_mode=simulation_mode,
        )
        result.frequency_trials[ctrl_idx, value_idx] = frequencies
        result.frequency_mean[ctrl_idx, value_idx] = float(frequencies.mean())
        result.frequency_std[ctrl_idx, value_idx] = float(frequencies.std())


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
    parallel: str,
    max_workers: Union[int, Literal["auto"]],
    num_gpus: int,
    simulation_mode: Literal["deterministic", "async"],
    freq_calibration_iters: int,
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
        horizon_frequency_trials=horizon_result.frequency_trials,
        num_samples_values=sample_result.axis_values,
        num_samples_tracking_error_mean=sample_result.tracking_error_mean,
        num_samples_tracking_error_std=sample_result.tracking_error_std,
        num_samples_frequency_mean=sample_result.frequency_mean,
        num_samples_frequency_std=sample_result.frequency_std,
        num_samples_tracking_error_trials=sample_result.tracking_error_trials,
        num_samples_frequency_trials=sample_result.frequency_trials,
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
        "parallel": parallel,
        "max_workers": max_workers,
        "num_gpus": num_gpus,
        "simulation_mode": simulation_mode,
        "freq_calibration_iters": freq_calibration_iters,
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
                "resample_score_normalization": "zscore(-log_density) + zscore(-cost)",
                "resample_cost_temperature_default": 1.0,
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
    if _WORKER_FLAG in sys.argv:
        _run_as_worker()
        return

    parser = argparse.ArgumentParser()
    add_output_root_arg(parser)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument(
        "--parallel",
        choices=("sequential", "controllers", "axis", "all"),
        default=PARALLEL,
    )
    parser.add_argument("--max-workers", default=MAX_WORKERS)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument(
        "--simulation",
        "--sim",
        dest="simulation_mode",
        default="deterministic",
        choices=("deterministic", "async"),
        help="CPU simulation loop to use.",
    )
    parser.add_argument(
        "--freq-calibration-iters",
        type=int,
        default=FREQ_CALIBRATION_ITERS,
    )
    args = parser.parse_args()

    horizons = np.asarray(HORIZONS, dtype=np.float32)
    num_samples_list = np.asarray(NUM_SAMPLES_LIST, dtype=np.int32)
    max_steps = int(args.max_steps)
    num_trials = int(args.num_trials)
    num_gpus = int(args.num_gpus)
    simulation_mode = args.simulation_mode
    freq_calibration_iters = int(args.freq_calibration_iters)
    max_workers: Union[int, Literal["auto"]]
    if args.max_workers == "auto":
        max_workers = "auto"
    else:
        max_workers = int(args.max_workers)

    if max_steps <= 0:
        parser.error("--max-steps must be positive")
    if num_trials <= 0:
        parser.error("--num-trials must be positive")
    if num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if max_workers != "auto" and max_workers <= 0:
        parser.error("--max-workers must be positive or 'auto'")
    if freq_calibration_iters < 0:
        parser.error("--freq-calibration-iters must be nonnegative")

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
    print(f"Simulation: {simulation_mode}")
    print(f"Horizon sweep: {len(horizons)} values @ {NUM_SAMPLES_FOR_HORIZON_SWEEP} samples")
    print(
        f"Sample sweep: {len(num_samples_list)} values "
        f"@ horizon={HORIZON_FOR_SAMPLE_SWEEP:.3f}s"
    )
    if args.parallel == "sequential":
        print("Mode: sequential (in-process)")
    else:
        print(
            f"Mode: parallel ({args.parallel}), workers={max_workers}, "
            f"{num_gpus} GPU(s)"
        )
        if freq_calibration_iters > 0:
            print(
                f"Frequency calibration: {num_trials} trials x "
                f"{freq_calibration_iters} steps (sequential)"
            )
    print(f"{'=' * 60}")

    mps_started = False
    if args.parallel != "sequential":
        mps_started = _mps_start()

    try:
        horizon_result = run_sweep(
            axis_name="horizon",
            axis_label="Horizon (s)",
            axis_values=horizons,
            num_trials=num_trials,
            max_steps=max_steps,
            specs=specs,
            parallel=args.parallel,
            max_workers=max_workers,
            num_gpus=num_gpus,
            simulation_mode=simulation_mode,
            fixed_num_samples=NUM_SAMPLES_FOR_HORIZON_SWEEP,
        )
        sample_result = run_sweep(
            axis_name="num_samples",
            axis_label="Number of Samples",
            axis_values=num_samples_list,
            num_trials=num_trials,
            max_steps=max_steps,
            specs=specs,
            parallel=args.parallel,
            max_workers=max_workers,
            num_gpus=num_gpus,
            simulation_mode=simulation_mode,
            fixed_horizon=HORIZON_FOR_SAMPLE_SWEEP,
        )
    finally:
        if mps_started:
            _mps_stop()

    if args.parallel != "sequential" and freq_calibration_iters > 0:
        calibrate_frequency(
            horizon_result,
            specs=specs,
            num_trials=num_trials,
            calibration_steps=freq_calibration_iters,
            simulation_mode=simulation_mode,
            fixed_num_samples=NUM_SAMPLES_FOR_HORIZON_SWEEP,
        )
        calibrate_frequency(
            sample_result,
            specs=specs,
            num_trials=num_trials,
            calibration_steps=freq_calibration_iters,
            simulation_mode=simulation_mode,
            fixed_horizon=HORIZON_FOR_SAMPLE_SWEEP,
        )

    save_outputs(
        out_dir,
        specs,
        horizon_result,
        sample_result,
        max_steps=max_steps,
        num_trials=num_trials,
        parallel=args.parallel,
        max_workers=max_workers,
        num_gpus=num_gpus,
        simulation_mode=simulation_mode,
        freq_calibration_iters=freq_calibration_iters,
    )

    print(f"\nSaved Go2 tracking-error sweeps to {out_dir}")


if __name__ == "__main__":
    main()
