"""Shared KNN-k sensitivity runner for success-threshold tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import mujoco
import numpy as np
from tqdm import tqdm

from algs import KNNDensityModel
from benchmark.common.cli import add_output_root_arg
from benchmark.common.paths import runs_dir
from benchmark.senior_thesis.scripts.benchmark_suite import (
    SweepResult,
    _load_benchmark_result,
    _mps_start,
    _mps_stop,
    _save_benchmark_result,
)
from simulation import run_benchmark, run_benchmark_async
from simulation.deterministic import BenchmarkResult


_WORKER_FLAG = "--_knn_k_worker"
_PARALLEL_CHOICES = ("sequential", "controllers", "axis", "all")


@dataclass(frozen=True)
class ControllerSpec:
    """One controller variant for a fixed K value."""

    name: str
    factory: Callable[[object], object]
    sweeps_k: bool = True


@dataclass
class SuccessKConfig:
    """Configuration for one fixed-horizon, fixed-sample K sweep."""

    task_name: str
    output_name: str
    k_values: Sequence[int]
    horizon: float
    num_samples: int
    num_trials: int
    frequency: float
    goal_threshold: float
    max_iterations: int
    record_video: bool = False
    video_trial_index: int = 0
    parallel: Literal["sequential", "controllers", "axis", "all"] = "sequential"
    max_workers: Union[int, Literal["auto"]] = "auto"
    num_gpus: int = 1
    simulation_mode: Literal["deterministic", "async"] = "deterministic"
    freq_calibration_iters: int = 50
    output_root: Optional[str] = None
    controller_params: Optional[dict[str, Any]] = None


def add_success_k_sensitivity_args(parser: argparse.ArgumentParser) -> None:
    add_output_root_arg(parser)
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated K values for the KNN density estimator.",
    )
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--parallel",
        choices=_PARALLEL_CHOICES,
        default=None,
    )
    parser.add_argument("--max-workers", default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument(
        "--simulation",
        "--sim",
        dest="simulation_mode",
        default=None,
        choices=("deterministic", "async"),
        help="CPU simulation loop to use.",
    )
    parser.add_argument("--freq-calibration-iters", type=int, default=None)
    parser.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
    )


def apply_success_k_overrides(
    config: SuccessKConfig,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "k_values", None):
        config.k_values = _parse_k_values(args.k_values)

    for attr in (
        "num_trials",
        "max_iterations",
        "parallel",
        "num_gpus",
        "simulation_mode",
        "freq_calibration_iters",
        "record_video",
    ):
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                setattr(config, attr, value)

    if getattr(args, "max_workers", None) is not None:
        config.max_workers = (
            "auto" if args.max_workers == "auto" else int(args.max_workers)
        )
    if getattr(args, "output_root", None) is not None:
        config.output_root = str(args.output_root)


def run_success_k_sensitivity(
    *,
    task_factory: Callable[[], object],
    build_controller_specs: Callable[[int], list[ControllerSpec]],
    config: SuccessKConfig,
    module_name: str,
) -> Optional[Path]:
    if _WORKER_FLAG in sys.argv:
        _run_as_worker(
            task_factory=task_factory,
            build_controller_specs=build_controller_specs,
            config=config,
        )
        return None

    _validate_config(config)
    specs_for_names = build_controller_specs(int(np.asarray(config.k_values)[0]))
    k_values = np.asarray(config.k_values, dtype=np.int32)
    use_parallel = config.parallel != "sequential"

    print(f"\n{'=' * 60}")
    print(f"KNN-k sensitivity: {config.task_name}")
    print(f"Controllers: {', '.join(spec.name for spec in specs_for_names)}")
    print(f"K values: {k_values.tolist()}")
    print(f"Fixed horizon: {config.horizon:.3f}s")
    print(f"Fixed samples: {config.num_samples}")
    print(f"Trials per point: {config.num_trials}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Simulation: {config.simulation_mode}")
    if use_parallel:
        workers = "auto" if config.max_workers == "auto" else str(config.max_workers)
        print(
            f"Mode: parallel ({config.parallel}), workers={workers}, "
            f"{config.num_gpus} GPU(s)"
        )
        if config.freq_calibration_iters > 0:
            print(
                f"Frequency calibration: {config.num_trials} trials x "
                f"{config.freq_calibration_iters} iters"
            )
    else:
        print("Mode: sequential (in-process)")
    print(f"{'=' * 60}\n")

    mps_started = False
    if use_parallel:
        mps_started = _mps_start()

    try:
        result = _run_k_sweep(
            task_factory=task_factory,
            build_controller_specs=build_controller_specs,
            config=config,
            module_name=_module_name(module_name),
            k_values=k_values,
        )
    finally:
        if mps_started:
            _mps_stop()

    if use_parallel and config.freq_calibration_iters > 0:
        _calibrate_frequency(
            result,
            task_factory=task_factory,
            build_controller_specs=build_controller_specs,
            config=config,
        )

    out_dir = _save_outputs(config, specs_for_names, result)
    print(f"Saved KNN-k sensitivity outputs to {out_dir}")
    return out_dir


def _run_as_worker(
    *,
    task_factory: Callable[[], object],
    build_controller_specs: Callable[[int], list[ControllerSpec]],
    config: SuccessKConfig,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(_WORKER_FLAG, action="store_true")
    parser.add_argument("--ctrl-idx", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--simulation",
        "--sim",
        dest="simulation_mode",
        default=None,
        choices=("deterministic", "async"),
    )
    args, _ = parser.parse_known_args()

    if args.num_trials is not None:
        config.num_trials = args.num_trials
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.simulation_mode is not None:
        config.simulation_mode = args.simulation_mode

    specs = build_controller_specs(int(args.k))
    result = _run_single_point(
        task_factory=task_factory,
        spec=specs[args.ctrl_idx],
        config=config,
        goal_threshold=config.goal_threshold,
        max_iterations=config.max_iterations,
        record_video=False,
    )
    _save_benchmark_result(result, args.output)


def _run_k_sweep(
    *,
    task_factory: Callable[[], object],
    build_controller_specs: Callable[[int], list[ControllerSpec]],
    config: SuccessKConfig,
    module_name: str,
    k_values: np.ndarray,
) -> SweepResult:
    first_specs = build_controller_specs(int(k_values[0]))
    num_ctrl = len(first_specs)
    num_vals = len(k_values)

    success = np.zeros((num_ctrl, num_vals), dtype=np.float32)
    success_time = np.zeros((num_ctrl, num_vals), dtype=np.float32)
    success_time_std = np.zeros((num_ctrl, num_vals), dtype=np.float32)
    frequency_mean = np.zeros((num_ctrl, num_vals), dtype=np.float32)
    frequency_std = np.zeros((num_ctrl, num_vals), dtype=np.float32)

    state_store = [[None for _ in k_values] for _ in range(num_ctrl)]
    control_store = [[None for _ in k_values] for _ in range(num_ctrl)]
    trace_store = [[None for _ in k_values] for _ in range(num_ctrl)]

    tmp_task = task_factory()
    mj_timestep = float(tmp_task.mj_model.opt.timestep)
    del tmp_task

    store_args = dict(
        config=config,
        mj_timestep=mj_timestep,
        success=success,
        success_time=success_time,
        success_time_std=success_time_std,
        frequency_mean=frequency_mean,
        frequency_std=frequency_std,
        state_store=state_store,
        control_store=control_store,
        trace_store=trace_store,
    )

    all_jobs = _make_jobs(first_specs, k_values)

    if config.parallel == "sequential":
        print(
            f"Sequential K sweep: {num_vals} values x {num_ctrl} controllers "
            f"= {len(all_jobs)} jobs"
        )
        for ctrl_idx, value_idx, k_value in tqdm(
            all_jobs,
            desc="knn_k sweep",
            unit="job",
        ):
            specs = build_controller_specs(k_value)
            result = _run_single_point(
                task_factory=task_factory,
                spec=specs[ctrl_idx],
                config=config,
                goal_threshold=config.goal_threshold,
                max_iterations=config.max_iterations,
                record_video=config.record_video,
            )
            _store_or_broadcast_result(
                result,
                specs[ctrl_idx],
                ctrl_idx,
                value_idx,
                num_vals,
                **store_args,
            )
            print(
                f"  {specs[ctrl_idx].name} @ k={k_value}: "
                f"{result.num_success}/{config.num_trials} succeeded"
            )
    else:
        batches = _make_batches(all_jobs, config.parallel, num_ctrl, num_vals)
        warm_cache_dir = _warmup_warp_cache(
            module_name=module_name,
            config=config,
            num_ctrl=num_ctrl,
            first_k=int(k_values[0]),
        )

        progress = tqdm(total=len(all_jobs), desc="knn_k sweep", unit="job")
        try:
            for batch_idx, batch in enumerate(batches):
                workers = _resolve_max_workers(config.max_workers, len(batch))
                if len(batches) > 1:
                    tqdm.write(
                        f"\n--- Batch {batch_idx + 1}/{len(batches)} "
                        f"({len(batch)} jobs, {workers} workers) ---"
                    )
                else:
                    tqdm.write(f"Running {len(batch)} jobs with {workers} workers")

                tmp_dir = tempfile.mkdtemp(prefix="knn_k_bench_")
                failures: list[str] = []
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {}
                        for job_idx, (ctrl_idx, value_idx, k_value) in enumerate(batch):
                            out_path = os.path.join(
                                tmp_dir,
                                f"result_{ctrl_idx}_{value_idx}.npz",
                            )
                            future = pool.submit(
                                _launch_subprocess,
                                module_name=module_name,
                                ctrl_idx=ctrl_idx,
                                k_value=k_value,
                                output_path=out_path,
                                config=config,
                                gpu_id=job_idx % max(config.num_gpus, 1),
                                warm_cache_dir=warm_cache_dir,
                            )
                            futures[future] = (ctrl_idx, value_idx, k_value)

                        for future in as_completed(futures):
                            ctrl_idx, value_idx, k_value = futures[future]
                            progress.update(1)
                            specs = build_controller_specs(k_value)
                            try:
                                result = future.result()
                            except Exception as exc:
                                failures.append(
                                    f"{specs[ctrl_idx].name} @ k={k_value}: {exc}"
                                )
                                continue
                            _store_or_broadcast_result(
                                result,
                                specs[ctrl_idx],
                                ctrl_idx,
                                value_idx,
                                num_vals,
                                **store_args,
                            )
                            tqdm.write(
                                f"  done: {specs[ctrl_idx].name} @ k={k_value} "
                                f"({result.num_success}/{config.num_trials})"
                            )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

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
        axis_name="knn_k",
        axis_label="KNN k",
        axis_values=k_values.astype(np.float32),
        success=success,
        success_time=success_time,
        success_time_std=success_time_std,
        frequency_mean=frequency_mean,
        frequency_std=frequency_std,
        state_store=state_store,
        control_store=control_store,
        trace_store=trace_store,
    )


def _run_single_point(
    *,
    task_factory: Callable[[], object],
    spec: ControllerSpec,
    config: SuccessKConfig,
    goal_threshold: float,
    max_iterations: int,
    record_video: bool,
) -> BenchmarkResult:
    task = task_factory()
    controller = spec.factory(task)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    runner = _benchmark_runner(config.simulation_mode)
    return runner(
        controller=controller,
        mj_model=mj_model,
        mj_data=mj_data,
        frequency=config.frequency,
        goal_threshold=goal_threshold,
        num_trials=config.num_trials,
        max_iterations=max_iterations,
        record_video=record_video,
        video_trial_index=config.video_trial_index,
    )


def _benchmark_runner(simulation_mode: str):
    if simulation_mode == "deterministic":
        return run_benchmark
    if simulation_mode == "async":
        return run_benchmark_async
    raise ValueError(
        f"Unknown simulation_mode={simulation_mode!r}; "
        "expected 'deterministic' or 'async'."
    )


def _store_result(
    result: BenchmarkResult,
    ctrl_idx: int,
    value_idx: int,
    *,
    config: SuccessKConfig,
    mj_timestep: float,
    success: np.ndarray,
    success_time: np.ndarray,
    success_time_std: np.ndarray,
    frequency_mean: np.ndarray,
    frequency_std: np.ndarray,
    state_store: list,
    control_store: list,
    trace_store: list,
) -> None:
    success[ctrl_idx, value_idx] = 100.0 * result.num_success / config.num_trials

    replan_period = 1.0 / config.frequency
    sim_steps_per_replan = max(int(replan_period / mj_timestep), 1)
    step_dt = float(sim_steps_per_replan * mj_timestep)
    trial_times = result.success_iterations.astype(np.float32) * step_dt
    if result.num_success > 0:
        succ_times = trial_times[result.success_mask]
        success_time[ctrl_idx, value_idx] = float(succ_times.mean())
        success_time_std[ctrl_idx, value_idx] = float(succ_times.std())
    else:
        success_time[ctrl_idx, value_idx] = 0.0
        success_time_std[ctrl_idx, value_idx] = 0.0

    frequency_mean[ctrl_idx, value_idx] = float(result.trial_frequencies.mean())
    frequency_std[ctrl_idx, value_idx] = float(result.trial_frequencies.std())

    state_store[ctrl_idx][value_idx] = result.state_trajectories
    control_store[ctrl_idx][value_idx] = result.control_trajectories
    trace_store[ctrl_idx][value_idx] = result.trace_trajectories


def _store_or_broadcast_result(
    result: BenchmarkResult,
    spec: ControllerSpec,
    ctrl_idx: int,
    value_idx: int,
    num_vals: int,
    **store_args,
) -> None:
    if spec.sweeps_k:
        _store_result(result, ctrl_idx, value_idx, **store_args)
        return

    for out_value_idx in range(num_vals):
        _store_result(result, ctrl_idx, out_value_idx, **store_args)


def _calibrate_frequency(
    result: SweepResult,
    *,
    task_factory: Callable[[], object],
    build_controller_specs: Callable[[int], list[ControllerSpec]],
    config: SuccessKConfig,
) -> None:
    print(f"\n{'-' * 60}")
    print(
        f"Frequency calibration: {config.num_trials} trials x "
        f"{config.freq_calibration_iters} iters, sequential"
    )
    print(f"{'-' * 60}")

    first_k = int(result.axis_values[0])
    first_specs = build_controller_specs(first_k)
    jobs = _make_jobs(first_specs, result.axis_values.astype(np.int32))
    for ctrl_idx, value_idx, k_value in tqdm(
        jobs,
        desc="frequency calibration",
        unit="job",
    ):
        specs = build_controller_specs(k_value)
        spec = specs[ctrl_idx]
        if spec.sweeps_k:
            bench_result = _run_single_point(
                task_factory=task_factory,
                spec=spec,
                config=config,
                goal_threshold=1e9,
                max_iterations=config.freq_calibration_iters,
                record_video=False,
            )
            result.frequency_mean[ctrl_idx, value_idx] = float(
                bench_result.trial_frequencies.mean()
            )
            result.frequency_std[ctrl_idx, value_idx] = float(
                bench_result.trial_frequencies.std()
            )
        else:
            bench_result = _run_single_point(
                task_factory=task_factory,
                spec=spec,
                config=config,
                goal_threshold=1e9,
                max_iterations=config.freq_calibration_iters,
                record_video=False,
            )
            mean = float(bench_result.trial_frequencies.mean())
            std = float(bench_result.trial_frequencies.std())
            result.frequency_mean[ctrl_idx, :] = mean
            result.frequency_std[ctrl_idx, :] = std


def _launch_subprocess(
    *,
    module_name: str,
    ctrl_idx: int,
    k_value: int,
    output_path: str,
    config: SuccessKConfig,
    gpu_id: int,
    warm_cache_dir: Optional[str],
) -> BenchmarkResult:
    cmd = [
        sys.executable,
        "-m",
        module_name,
        _WORKER_FLAG,
        "--ctrl-idx",
        str(ctrl_idx),
        "--k",
        str(k_value),
        "--output",
        output_path,
        "--num-trials",
        str(config.num_trials),
        "--max-iterations",
        str(config.max_iterations),
        "--simulation",
        config.simulation_mode,
    ]
    if config.output_root is not None:
        cmd.extend(["--output-root", config.output_root])

    env = os.environ.copy()
    _assign_worker_gpu(env, gpu_id, config.num_gpus)

    worker_cache = tempfile.mkdtemp(prefix="knn_k_warp_worker_")
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
        return _load_benchmark_result(output_path)
    finally:
        shutil.rmtree(worker_cache, ignore_errors=True)


def _warmup_warp_cache(
    *,
    module_name: str,
    config: SuccessKConfig,
    num_ctrl: int,
    first_k: int,
) -> Optional[str]:
    cache_dir = tempfile.mkdtemp(prefix="knn_k_warp_warm_")
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
            "--output",
            output_path,
            "--num-trials",
            "1",
            "--max-iterations",
            "5",
            "--simulation",
            config.simulation_mode,
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
            print(f"WARNING: Warp cache warm-up failed:\n{stderr_tail}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None
        if os.path.exists(output_path):
            os.remove(output_path)

    print("Warp kernel cache ready.")
    return cache_dir


def _save_outputs(
    config: SuccessKConfig,
    specs: list[ControllerSpec],
    result: SweepResult,
) -> Path:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mppi_matplotlib"),
    )
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = runs_dir("paper", output_root=config.output_root) / (
        f"{config.output_name}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = result.axis_name
    np.savetxt(out_dir / f"{prefix}_success_rate.csv", result.success, delimiter=",")
    np.savetxt(
        out_dir / f"{prefix}_success_time_mean.csv",
        result.success_time,
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"{prefix}_success_time_std.csv",
        result.success_time_std,
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
        out_dir / f"{prefix}_success_rate.png",
        specs,
        result.axis_values,
        result.success,
        None,
        xlabel=result.axis_label,
        ylabel="Success Rate (%)",
        title=f"{config.task_name}: Success vs {result.axis_label}",
    )
    _plot_matrix(
        out_dir / f"{prefix}_success_time.png",
        specs,
        result.axis_values,
        result.success_time,
        result.success_time_std,
        xlabel=result.axis_label,
        ylabel="Average Time-To-Success (s)",
        title=f"{config.task_name}: Time-To-Success vs {result.axis_label}",
        skip_zeros=True,
    )
    _plot_matrix(
        out_dir / f"{prefix}_frequency.png",
        specs,
        result.axis_values,
        result.frequency_mean,
        result.frequency_std,
        xlabel=result.axis_label,
        ylabel="Control Frequency (Hz)",
        title=f"{config.task_name}: Control Frequency vs {result.axis_label}",
    )

    controller_names = np.asarray([spec.name for spec in specs], dtype=object)
    np.savez(
        out_dir / "summary.npz",
        controller_names=controller_names,
        controller_sweeps_k=np.array([spec.sweeps_k for spec in specs], dtype=bool),
        knn_k_values=result.axis_values.astype(np.int32),
        knn_k_success=result.success,
        knn_k_success_time=result.success_time,
        knn_k_success_time_std=result.success_time_std,
        knn_k_frequency_mean=result.frequency_mean,
        knn_k_frequency_std=result.frequency_std,
    )

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write(
            "axis,k,controller,success_rate,success_time_mean,"
            "success_time_std,frequency_mean,frequency_std\n"
        )
        k_values = result.axis_values.astype(np.int32)
        for ctrl_idx, spec in enumerate(specs):
            value_indices = range(len(k_values)) if spec.sweeps_k else range(1)
            for value_idx in value_indices:
                k_text = str(int(k_values[value_idx])) if spec.sweeps_k else "baseline"
                f.write(
                    f"{prefix},{k_text},{spec.name},"
                    f"{result.success[ctrl_idx, value_idx]:.8g},"
                    f"{result.success_time[ctrl_idx, value_idx]:.8g},"
                    f"{result.success_time_std[ctrl_idx, value_idx]:.8g},"
                    f"{result.frequency_mean[ctrl_idx, value_idx]:.8g},"
                    f"{result.frequency_std[ctrl_idx, value_idx]:.8g}\n"
                )

    metadata = {
        "task_name": config.task_name,
        "output_name": config.output_name,
        "controller_names": [spec.name for spec in specs],
        "controller_sweeps_k": [spec.sweeps_k for spec in specs],
        "config": _jsonify(asdict(config)),
        "knn_k_values": result.axis_values.astype(np.int32).tolist(),
        "fixed_horizon": config.horizon,
        "fixed_num_samples": config.num_samples,
        "baseline_note": (
            "Controllers with controller_sweeps_k=false are benchmarked once "
            "and broadcast across K values only for reference plots/matrices."
        ),
        "axis": {
            "name": result.axis_name,
            "label": result.axis_label,
        },
    }
    if config.controller_params is not None:
        metadata["controller_params"] = _jsonify(config.controller_params)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def _plot_matrix(
    out_path: Path,
    specs: list[ControllerSpec],
    x_values: np.ndarray,
    values: np.ndarray,
    std: Optional[np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    skip_zeros: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    for idx, spec in enumerate(specs):
        x = x_values
        y = values[idx]
        y_std = None if std is None else std[idx]
        if skip_zeros:
            mask = y != 0
            if not np.any(mask):
                continue
            x = x[mask]
            y = y[mask]
            y_std = None if y_std is None else y_std[mask]
        line, = plt.plot(x, y, label=spec.name)
        if y_std is not None:
            plt.fill_between(
                x,
                y - y_std,
                y + y_std,
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


def _parse_k_values(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _module_name(module_name: str) -> str:
    if module_name == "__main__":
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if main_spec is not None and main_spec.name is not None:
            return main_spec.name
    return module_name


def _validate_config(config: SuccessKConfig) -> None:
    k_values = np.asarray(config.k_values, dtype=np.int32)
    if k_values.size == 0:
        raise ValueError("At least one K value is required.")
    if np.any(k_values < 1):
        raise ValueError("All K values must be >= 1.")
    if np.any(k_values > KNNDensityModel.K_MAX):
        raise ValueError(
            f"All K values must be <= {KNNDensityModel.K_MAX}; got "
            f"{k_values.tolist()}."
        )
    if config.num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if config.horizon <= 0.0:
        raise ValueError("horizon must be positive.")
    if config.frequency <= 0.0:
        raise ValueError("frequency must be positive.")
    if config.num_gpus <= 0:
        raise ValueError("num_gpus must be positive.")
    if config.max_workers != "auto" and int(config.max_workers) <= 0:
        raise ValueError("max_workers must be positive or 'auto'.")
    if config.freq_calibration_iters < 0:
        raise ValueError("freq_calibration_iters must be nonnegative.")


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
