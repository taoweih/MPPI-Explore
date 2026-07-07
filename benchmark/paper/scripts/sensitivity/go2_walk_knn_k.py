"""Go2 walk KNN-k sensitivity benchmark for paper experiments."""

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

from algs import DIALMPC, DensityGuidedMPPI, KNNDensityModel, MPPI
from benchmark.common.cli import add_output_root_arg
from benchmark.common.paths import runs_dir
from benchmark.senior_thesis.scripts.benchmark_suite import _mps_start, _mps_stop
from simulation.asyncrousnous import _AsyncPlanner, _step_latest_async_action
from tasks.go2_walk import Go2Walk


_WORKER_FLAG = "--_go2_knn_k_worker"

PLANNING_DT = 0.02
FREQUENCY = 50.0
MAX_STEPS = 400
NUM_TRIALS = 10

TARGET_VX = 0.8
TARGET_VY = 0.0
TARGET_VYAW = 0.0
GAIT = "trot"

K_VALUES = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 50], dtype=np.int32)

FIXED_HORIZON = 16 * PLANNING_DT
NUM_SAMPLES = 2048

NOISE_LEVEL = 1.0
TEMPERATURE = 0.05
NUM_KNOTS = 8
ITERATIONS = 1
SEED = 0

DIAL_NUM_INITIAL_DIFFUSION_STEPS = 10
DIAL_HORIZON_DIFFUSE_FACTOR = 0.9
DIAL_TRAJ_DIFFUSE_FACTOR = 0.5
DIAL_INCLUDE_MEAN_SAMPLE = True
DIAL_FIX_FIRST_KNOT = True
DIAL_NORMALIZE_COSTS = True
DIAL_COST_NORMALIZATION_EPS = 1.0e-6

DENSITY_NUM_KNOTS_PER_STAGE = 2
INVERSE_DENSITY_POWER = 1.0
RESAMPLE_COST_WEIGHT = 1.0
RESAMPLE_COST_TEMPERATURE = None

OUTPUT_NAME = "go2_walk_knn_k_sensitivity"

PARALLEL = "all"
MAX_WORKERS = "30"
NUM_GPUS = 2
FREQ_CALIBRATION_ITERS = 50


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    factory: Callable[[Go2Walk, int], object]
    sweeps_k: bool = True


@dataclass
class SweepResult:
    axis_values: np.ndarray
    tracking_error_mean: np.ndarray
    tracking_error_std: np.ndarray
    frequency_mean: np.ndarray
    frequency_std: np.ndarray
    tracking_error_trials: np.ndarray
    frequency_trials: np.ndarray


def make_task() -> Go2Walk:
    return Go2Walk(
        planning_dt=PLANNING_DT,
        sim_dt=PLANNING_DT,
        target_vx=TARGET_VX,
        target_vy=TARGET_VY,
        target_vyaw=TARGET_VYAW,
        gait=GAIT,
    )


def _base_mppi_kwargs(task: Go2Walk, seed: int) -> dict:
    return dict(
        task=task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=FIXED_HORIZON,
        spline_type="zero",
        num_knots=NUM_KNOTS,
        iterations=ITERATIONS,
        seed=seed,
    )


def build_controller_specs(k: int) -> list[ControllerSpec]:
    return [
        ControllerSpec(
            "MPPI",
            lambda task, seed: MPPI(**_base_mppi_kwargs(task, seed)),
            sweeps_k=False,
        ),
        ControllerSpec(
            "DIAL-MPC",
            lambda task, seed: DIALMPC(
                **_base_mppi_kwargs(task, seed),
                num_initial_diffusion_steps=DIAL_NUM_INITIAL_DIFFUSION_STEPS,
                horizon_diffuse_factor=DIAL_HORIZON_DIFFUSE_FACTOR,
                traj_diffuse_factor=DIAL_TRAJ_DIFFUSE_FACTOR,
                include_mean_sample=DIAL_INCLUDE_MEAN_SAMPLE,
                fix_first_knot=DIAL_FIX_FIRST_KNOT,
                normalize_costs=DIAL_NORMALIZE_COSTS,
                cost_normalization_eps=DIAL_COST_NORMALIZATION_EPS,
            ),
            sweeps_k=False,
        ),
        ControllerSpec(
            "Density-Guided MPPI (KNN)",
            lambda task, seed: DensityGuidedMPPI(
                **_base_mppi_kwargs(task, seed),
                density_model=KNNDensityModel(
                    k=k,
                    alpha=INVERSE_DENSITY_POWER,
                ),
                num_knots_per_stage=DENSITY_NUM_KNOTS_PER_STAGE,
                resample_cost_weight=RESAMPLE_COST_WEIGHT,
                resample_cost_temperature=RESAMPLE_COST_TEMPERATURE,
            ),
        ),
    ]


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


def tracking_error(task: Go2Walk, mj_data: mujoco.MjData) -> float:
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


def step_current_plan(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
) -> None:
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
    k: int,
    trial_idx: int,
    max_steps: int,
    simulation_mode: Literal["deterministic", "async"],
) -> tuple[float, float]:
    task = make_task()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    task.reset_to_home(mj_data)

    seed = SEED + trial_idx
    controller = spec.factory(task, seed)
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
                tracking_errors[step_idx] = tracking_error(task, mj_data)

                elapsed = time.perf_counter() - loop_start
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)
        finally:
            trial_elapsed = time.perf_counter() - trial_start
            planner.drain(commit=False)
            planner.close()

        control_hz = planner.stats.completed / max(trial_elapsed, 1e-9)
        return float(tracking_errors.mean()), float(control_hz)

    if simulation_mode != "deterministic":
        raise ValueError(
            f"Unknown simulation_mode={simulation_mode!r}; "
            "expected 'deterministic' or 'async'."
        )

    tracking_errors = np.zeros(max_steps, dtype=np.float32)
    total_plan_time = 0.0
    act_before_plan = bool(getattr(controller, "act_before_plan", False))
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

        tracking_errors[step_idx] = tracking_error(task, mj_data)

    control_hz = max_steps / max(total_plan_time, 1e-9)
    return float(tracking_errors.mean()), float(control_hz)


def run_point(
    spec: ControllerSpec,
    *,
    k: int,
    num_trials: int,
    max_steps: int,
    simulation_mode: Literal["deterministic", "async"],
) -> tuple[np.ndarray, np.ndarray]:
    tracking_trials = np.zeros(num_trials, dtype=np.float32)
    frequency_trials = np.zeros(num_trials, dtype=np.float32)
    for trial_idx in range(num_trials):
        tracking, hz = run_trial(
            spec,
            k=k,
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


def _run_as_worker() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(_WORKER_FLAG, action="store_true")
    parser.add_argument("--ctrl-idx", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
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

    specs = build_controller_specs(args.k)
    tracking_trials, frequency_trials = run_point(
        specs[args.ctrl_idx],
        k=args.k,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        simulation_mode=args.simulation_mode,
    )
    _save_point_result(args.output, tracking_trials, frequency_trials)


def run_sweep(
    *,
    k_values: np.ndarray,
    specs: list[ControllerSpec],
    num_trials: int,
    max_steps: int,
    parallel: Literal["sequential", "controllers", "axis", "all"],
    max_workers: Union[int, Literal["auto"]],
    num_gpus: int,
    simulation_mode: Literal["deterministic", "async"],
) -> SweepResult:
    num_ctrl = len(specs)
    num_vals = len(k_values)
    tracking_trials = np.zeros((num_ctrl, num_vals, num_trials), dtype=np.float32)
    frequency_trials = np.zeros_like(tracking_trials)

    all_jobs = _make_jobs(specs, k_values)

    if parallel == "sequential":
        print(
            f"\nSequential K sweep: {num_vals} values x {num_ctrl} controllers "
            f"= {len(all_jobs)} jobs"
        )
        for ctrl_idx, value_idx, k_value in tqdm(
            all_jobs,
            desc="knn_k sweep",
            unit="job",
        ):
            specs_for_k = build_controller_specs(k_value)
            tracking, frequency = run_point(
                specs_for_k[ctrl_idx],
                k=k_value,
                num_trials=num_trials,
                max_steps=max_steps,
                simulation_mode=simulation_mode,
            )
            if specs_for_k[ctrl_idx].sweeps_k:
                tracking_trials[ctrl_idx, value_idx] = tracking
                frequency_trials[ctrl_idx, value_idx] = frequency
            else:
                tracking_trials[ctrl_idx, :] = tracking
                frequency_trials[ctrl_idx, :] = frequency
    else:
        module_name = _module_name()
        warm_cache_dir = _warmup_warp_cache(
            module_name=module_name,
            num_ctrl=num_ctrl,
            first_k=int(k_values[0]),
            simulation_mode=simulation_mode,
        )
        batches = _make_batches(all_jobs, parallel, num_ctrl, num_vals)
        progress = tqdm(total=len(all_jobs), desc="knn_k sweep", unit="job")
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

                result_dir = tempfile.mkdtemp(prefix="go2_knn_k_")
                failures: list[str] = []
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {}
                        for job_idx, (ctrl_idx, value_idx, k_value) in enumerate(batch):
                            output_path = os.path.join(
                                result_dir,
                                f"result_{ctrl_idx}_{value_idx}.npz",
                            )
                            future = pool.submit(
                                _launch_point_subprocess,
                                module_name=module_name,
                                ctrl_idx=ctrl_idx,
                                k_value=k_value,
                                num_trials=num_trials,
                                max_steps=max_steps,
                                simulation_mode=simulation_mode,
                                output_path=output_path,
                                gpu_id=job_idx % max(num_gpus, 1),
                                num_gpus=num_gpus,
                                warm_cache_dir=warm_cache_dir,
                            )
                            futures[future] = (ctrl_idx, value_idx, k_value)

                        for future in as_completed(futures):
                            ctrl_idx, value_idx, k_value = futures[future]
                            progress.update(1)
                            spec_name = build_controller_specs(k_value)[ctrl_idx].name
                            try:
                                tracking, frequency = future.result()
                            except Exception as exc:
                                failures.append(f"{spec_name} @ k={k_value}: {exc}")
                                continue
                            spec = build_controller_specs(k_value)[ctrl_idx]
                            if spec.sweeps_k:
                                tracking_trials[ctrl_idx, value_idx] = tracking
                                frequency_trials[ctrl_idx, value_idx] = frequency
                            else:
                                tracking_trials[ctrl_idx, :] = tracking
                                frequency_trials[ctrl_idx, :] = frequency
                            tqdm.write(
                                f"  done: {spec_name} @ k={k_value} "
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
        axis_values=k_values.astype(np.float32),
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
    num_trials: int,
    calibration_steps: int,
    simulation_mode: Literal["deterministic", "async"],
) -> None:
    total_jobs = len(result.axis_values) * result.frequency_mean.shape[0]
    print(f"\n{'-' * 60}")
    print(
        f"Frequency calibration: {num_trials} trials x "
        f"{calibration_steps} steps, sequential"
    )
    print(f"{total_jobs} jobs total")
    print(f"{'-' * 60}")

    first_specs = build_controller_specs(int(result.axis_values[0]))
    jobs = _make_jobs(first_specs, result.axis_values.astype(np.int32))
    for ctrl_idx, value_idx, k_value in tqdm(
        jobs,
        desc="frequency calibration",
        unit="job",
    ):
        specs = build_controller_specs(k_value)
        spec = specs[ctrl_idx]
        _, frequencies = run_point(
            spec,
            k=k_value,
            num_trials=num_trials,
            max_steps=calibration_steps,
            simulation_mode=simulation_mode,
        )
        if spec.sweeps_k:
            result.frequency_trials[ctrl_idx, value_idx] = frequencies
            result.frequency_mean[ctrl_idx, value_idx] = float(frequencies.mean())
            result.frequency_std[ctrl_idx, value_idx] = float(frequencies.std())
        else:
            result.frequency_trials[ctrl_idx, :] = frequencies
            result.frequency_mean[ctrl_idx, :] = float(frequencies.mean())
            result.frequency_std[ctrl_idx, :] = float(frequencies.std())


def _launch_point_subprocess(
    *,
    module_name: str,
    ctrl_idx: int,
    k_value: int,
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
        "--k",
        str(k_value),
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

    worker_cache = tempfile.mkdtemp(prefix="go2_knn_k_warp_worker_")
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
            raise RuntimeError(stderr_tail)
        return _load_point_result(output_path)
    finally:
        shutil.rmtree(worker_cache, ignore_errors=True)


def _warmup_warp_cache(
    *,
    module_name: str,
    num_ctrl: int,
    first_k: int,
    simulation_mode: Literal["deterministic", "async"],
) -> Optional[str]:
    cache_dir = tempfile.mkdtemp(prefix="go2_knn_k_warp_warm_")
    env = os.environ.copy()
    env["WARP_CACHE_PATH"] = cache_dir
    print("Warming up Warp kernel cache...")

    for ctrl_idx in range(num_ctrl):
        output_path = os.path.join(cache_dir, f"warmup_{ctrl_idx}.npz")
        cmd = [
            sys.executable,
            "-m",
            module_name,
            _WORKER_FLAG,
            "--ctrl-idx",
            str(ctrl_idx),
            "--k",
            str(first_k),
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
            print(f"WARNING: Warp warmup failed:\n{stderr_tail}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None
        if os.path.exists(output_path):
            os.remove(output_path)

    print("Warp kernel cache ready.")
    return cache_dir


def save_outputs(
    out_dir: Path,
    specs: list[ControllerSpec],
    result: SweepResult,
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
    np.savetxt(
        out_dir / "knn_k_tracking_error_mean.csv",
        result.tracking_error_mean,
        delimiter=",",
    )
    np.savetxt(
        out_dir / "knn_k_tracking_error_std.csv",
        result.tracking_error_std,
        delimiter=",",
    )
    np.savetxt(
        out_dir / "knn_k_frequency_mean.csv",
        result.frequency_mean,
        delimiter=",",
    )
    np.savetxt(
        out_dir / "knn_k_frequency_std.csv",
        result.frequency_std,
        delimiter=",",
    )

    _plot_matrix(
        out_dir / "knn_k_tracking_error.png",
        specs,
        result.axis_values,
        result.tracking_error_mean,
        result.tracking_error_std,
        xlabel="KNN k",
        ylabel="Average Tracking Error",
        title="go2_walk: Tracking Error vs KNN k",
    )
    _plot_matrix(
        out_dir / "knn_k_frequency.png",
        specs,
        result.axis_values,
        result.frequency_mean,
        result.frequency_std,
        xlabel="KNN k",
        ylabel="Control Frequency (Hz)",
        title="go2_walk: Control Frequency vs KNN k",
    )

    controller_names = np.asarray([spec.name for spec in specs], dtype=object)
    np.savez(
        out_dir / "summary.npz",
        controller_names=controller_names,
        controller_sweeps_k=np.array([spec.sweeps_k for spec in specs], dtype=bool),
        knn_k_values=result.axis_values.astype(np.int32),
        knn_k_tracking_error_mean=result.tracking_error_mean,
        knn_k_tracking_error_std=result.tracking_error_std,
        knn_k_frequency_mean=result.frequency_mean,
        knn_k_frequency_std=result.frequency_std,
        knn_k_tracking_error_trials=result.tracking_error_trials,
        knn_k_frequency_trials=result.frequency_trials,
    )

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write(
            "axis,k,controller,tracking_error_mean,tracking_error_std,"
            "frequency_mean,frequency_std\n"
        )
        k_values = result.axis_values.astype(np.int32)
        for ctrl_idx, spec in enumerate(specs):
            value_indices = range(len(k_values)) if spec.sweeps_k else range(1)
            for value_idx in value_indices:
                k_text = str(int(k_values[value_idx])) if spec.sweeps_k else "baseline"
                f.write(
                    f"knn_k,{k_text},{spec.name},"
                    f"{result.tracking_error_mean[ctrl_idx, value_idx]:.8g},"
                    f"{result.tracking_error_std[ctrl_idx, value_idx]:.8g},"
                    f"{result.frequency_mean[ctrl_idx, value_idx]:.8g},"
                    f"{result.frequency_std[ctrl_idx, value_idx]:.8g}\n"
                )

    metadata = {
        "task_name": "go2_walk",
        "output_name": OUTPUT_NAME,
        "controller_names": [spec.name for spec in specs],
        "controller_sweeps_k": [spec.sweeps_k for spec in specs],
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
        "knn_k_values": result.axis_values.astype(np.int32).tolist(),
        "fixed_horizon": FIXED_HORIZON,
        "fixed_num_samples": NUM_SAMPLES,
        "baseline_note": (
            "Controllers with controller_sweeps_k=false are benchmarked once "
            "and broadcast across K values only for reference plots/matrices."
        ),
        "controller_params": {
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
                "num_knots_per_stage": DENSITY_NUM_KNOTS_PER_STAGE,
                "knn_k_values": result.axis_values.astype(np.int32).tolist(),
                "inverse_density_power": INVERSE_DENSITY_POWER,
                "resample_cost_weight": RESAMPLE_COST_WEIGHT,
                "resample_cost_temperature": RESAMPLE_COST_TEMPERATURE,
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
        "metric": {
            "primary": "tracking_error_mean",
            "aggregation": "mean over every control step",
            "definition": (
                "squared body-frame XY velocity error plus squared body-frame "
                "yaw-rate error"
            ),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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


def _parse_k_values(value: str) -> np.ndarray:
    return np.array(
        [int(part.strip()) for part in value.split(",") if part.strip()],
        dtype=np.int32,
    )


def _validate_k_values(k_values: np.ndarray) -> None:
    if k_values.size == 0:
        raise ValueError("At least one K value is required.")
    if np.any(k_values < 1):
        raise ValueError("All K values must be >= 1.")
    if np.any(k_values > KNNDensityModel.K_MAX):
        raise ValueError(
            f"All K values must be <= {KNNDensityModel.K_MAX}; got "
            f"{k_values.tolist()}."
        )


def _make_batches(
    all_jobs: list[tuple[int, int, int]],
    parallel: str,
    num_ctrl: int,
    num_vals: int,
) -> list[list[tuple[int, int, int]]]:
    if parallel == "controllers":
        return [
            [job for job in all_jobs if job[1] == value_idx]
            for value_idx in range(num_vals)
        ]
    if parallel == "axis":
        return [
            [job for job in all_jobs if job[0] == ctrl_idx]
            for ctrl_idx in range(num_ctrl)
        ]
    return [all_jobs]


def _make_jobs(
    specs: list[ControllerSpec],
    k_values: np.ndarray,
) -> list[tuple[int, int, int]]:
    jobs: list[tuple[int, int, int]] = []
    for ctrl_idx, spec in enumerate(specs):
        value_indices = range(len(k_values)) if spec.sweeps_k else range(1)
        for value_idx in value_indices:
            jobs.append((ctrl_idx, value_idx, int(k_values[value_idx])))
    return jobs


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


def _module_name() -> str:
    if __spec__ is not None and __spec__.name is not None:
        return __spec__.name
    return __name__


def main() -> None:
    if _WORKER_FLAG in sys.argv:
        _run_as_worker()
        return

    parser = argparse.ArgumentParser()
    add_output_root_arg(parser)
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated K values for the KNN density estimator.",
    )
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

    k_values = K_VALUES if args.k_values is None else _parse_k_values(args.k_values)
    _validate_k_values(k_values)
    max_workers: Union[int, Literal["auto"]]
    max_workers = "auto" if args.max_workers == "auto" else int(args.max_workers)

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.num_trials <= 0:
        parser.error("--num-trials must be positive")
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if max_workers != "auto" and max_workers <= 0:
        parser.error("--max-workers must be positive or 'auto'")
    if args.freq_calibration_iters < 0:
        parser.error("--freq-calibration-iters must be nonnegative")

    specs = build_controller_specs(int(k_values[0]))

    print(f"\n{'=' * 60}")
    print("Go2 walk KNN-k sensitivity benchmark")
    print(f"Controllers: {', '.join(spec.name for spec in specs)}")
    print(f"K values: {k_values.tolist()}")
    print(f"Fixed horizon: {FIXED_HORIZON:.3f}s")
    print(f"Fixed samples: {NUM_SAMPLES}")
    print(f"Trials per point: {args.num_trials}")
    print(f"Task steps per trial: {args.max_steps}")
    print(f"Simulation: {args.simulation_mode}")
    if args.parallel == "sequential":
        print("Mode: sequential (in-process)")
    else:
        print(
            f"Mode: parallel ({args.parallel}), workers={max_workers}, "
            f"{args.num_gpus} GPU(s)"
        )
        if args.freq_calibration_iters > 0:
            print(
                f"Frequency calibration: {args.num_trials} trials x "
                f"{args.freq_calibration_iters} steps"
            )
    print(f"{'=' * 60}")

    mps_started = False
    if args.parallel != "sequential":
        mps_started = _mps_start()

    try:
        result = run_sweep(
            k_values=k_values,
            specs=specs,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            parallel=args.parallel,
            max_workers=max_workers,
            num_gpus=args.num_gpus,
            simulation_mode=args.simulation_mode,
        )
    finally:
        if mps_started:
            _mps_stop()

    if args.parallel != "sequential" and args.freq_calibration_iters > 0:
        calibrate_frequency(
            result,
            num_trials=args.num_trials,
            calibration_steps=args.freq_calibration_iters,
            simulation_mode=args.simulation_mode,
        )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = runs_dir("paper", output_root=args.output_root) / (
        f"{OUTPUT_NAME}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(
        out_dir,
        specs,
        result,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
        parallel=args.parallel,
        max_workers=max_workers,
        num_gpus=args.num_gpus,
        simulation_mode=args.simulation_mode,
        freq_calibration_iters=args.freq_calibration_iters,
    )

    print(f"\nSaved Go2 KNN-k sensitivity outputs to {out_dir}")


if __name__ == "__main__":
    main()
