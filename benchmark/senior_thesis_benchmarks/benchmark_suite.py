"""Shared OOP benchmark suite for senior-thesis controller comparisons."""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
from tqdm import tqdm
import mujoco
import numpy as np

from simulation.deterministic import run_benchmark, BenchmarkResult
from tasks.task_base import ROOT

_WORKER_FLAG = "--_benchmark_worker"


def _mps_is_running() -> bool:
    """Check if the NVIDIA MPS daemon is currently running."""
    try:
        proc = subprocess.Popen(
            ["nvidia-cuda-mps-control"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        _, _ = proc.communicate(input="get_server_list\n", timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _mps_start() -> bool:
    """Start the NVIDIA MPS daemon. Returns True if we started it."""
    if _mps_is_running():
        print("MPS daemon already running — reusing existing instance.")
        return False
    print("Starting NVIDIA MPS daemon for parallel subprocess execution...")
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        print(
            "WARNING: nvidia-cuda-mps-control not found. "
            "Running parallel subprocesses without MPS (expect GPU context-switch overhead)."
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to start MPS daemon: {e.stderr.strip()}")
        return False
    print("MPS daemon started.")
    return True


def _mps_stop() -> None:
    """Stop the NVIDIA MPS daemon."""
    try:
        proc = subprocess.Popen(
            ["nvidia-cuda-mps-control"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        proc.communicate(input="quit\n", timeout=10)
        print("MPS daemon stopped.")
    except Exception as e:
        print(f"WARNING: Failed to stop MPS daemon: {e}")


@dataclass
class ControllerSpec:
    """One controller variant in the benchmark sweep."""

    name: str
    factory: Callable[..., object]


@dataclass
class SweepConfig:
    """Sweep-level benchmark options."""

    horizons: Sequence[float]
    num_samples_list: Optional[Sequence[int]] = None
    sweep_horizon_for_samples: float = 0.2
    num_trials: int = 20
    frequency: float = 50.0
    goal_threshold: float = 0.5
    max_iterations: int = 1000
    record_video: bool = False
    video_trial_index: int = 0
    output_tag: str = "thesis"
    pretrain_weights_dir: Optional[str] = None
    controller_params: Optional[dict[str, Any]] = None
    parallel: Literal["sequential", "controllers", "axis", "all"] = "sequential"
    max_workers: Union[int, Literal["auto"]] = "auto"
    num_gpus: int = 1


@dataclass
class SweepResult:
    """Benchmark outputs for one sweep axis."""

    axis_name: str
    axis_label: str
    axis_values: np.ndarray
    success: np.ndarray
    success_time: np.ndarray
    success_time_std: np.ndarray
    frequency_mean: np.ndarray
    frequency_std: np.ndarray
    state_store: list
    control_store: list
    trace_store: list


def _horizon_controller_factory(spec: ControllerSpec, task: object, value: float) -> object:
    """Picklable factory: build controller for a horizon sweep point."""
    return spec.factory(task, float(value), num_samples=None)


def _sample_controller_factory(
    spec: ControllerSpec, task: object, value: float, *, horizon: float,
) -> object:
    """Picklable factory: build controller for a sample-count sweep point."""
    return spec.factory(task, horizon, num_samples=int(value))


def _horizon_header(value: float) -> str:
    return f"\n=== Horizon {float(value):.3f}s ==="


def _sample_header(value: float, *, horizon: float) -> str:
    return f"\n=== Num Samples {int(value)} @ Horizon {horizon:.3f}s ==="


def _save_benchmark_result(result: BenchmarkResult, path: str) -> None:
    """Serialize a BenchmarkResult to an npz file."""
    np.savez(
        path,
        num_success=np.array(result.num_success),
        control_frequency_hz=np.array(result.control_frequency_hz),
        avg_success_iteration=np.array(result.avg_success_iteration),
        success_mask=result.success_mask,
        success_iterations=result.success_iterations,
        trial_frequencies=result.trial_frequencies,
        state_trajectories=result.state_trajectories,
        control_trajectories=result.control_trajectories,
        trace_trajectories=result.trace_trajectories,
    )


def _load_benchmark_result(path: str) -> BenchmarkResult:
    """Deserialize a BenchmarkResult from an npz file."""
    d = np.load(path, allow_pickle=False)
    return BenchmarkResult(
        num_success=int(d["num_success"]),
        control_frequency_hz=float(d["control_frequency_hz"]),
        avg_success_iteration=float(d["avg_success_iteration"]),
        success_mask=d["success_mask"],
        success_iterations=d["success_iterations"],
        trial_frequencies=d["trial_frequencies"],
        state_trajectories=d["state_trajectories"],
        control_trajectories=d["control_trajectories"],
        trace_trajectories=d["trace_trajectories"],
    )


class SeniorThesisBenchmarkSuite:
    """Runs the same sweep across multiple controller variants."""

    def __init__(
        self,
        task_name: str,
        task_factory: Callable[[], object],
        controller_specs: Sequence[ControllerSpec],
        config: SweepConfig,
    ) -> None:
        self.task_name = task_name
        self.task_factory = task_factory
        self.controller_specs = list(controller_specs)
        self.config = config

    def run(self) -> Path:
        # Subprocess worker mode — run a single benchmark point and exit.
        if _WORKER_FLAG in sys.argv:
            self._run_as_worker()
            sys.exit(0)

        use_parallel = self.config.parallel != "sequential"
        mps_started = False
        if use_parallel:
            mps_started = _mps_start()

        try:
            horizon_values = np.asarray(self.config.horizons, dtype=np.float32)
            horizon_result = self._run_sweep(
                axis_name="horizon",
                axis_label="Horizon (s)",
                axis_values=horizon_values,
                header_fn=_horizon_header,
                controller_factory=_horizon_controller_factory,
            )

            sample_result = None
            if self.config.num_samples_list is not None:
                sample_values = np.asarray(self.config.num_samples_list, dtype=np.int32)
                if sample_values.size > 0:
                    fixed_horizon = float(self.config.sweep_horizon_for_samples)
                    sample_result = self._run_sweep(
                        axis_name="num_samples",
                        axis_label="Number of Samples",
                        axis_values=sample_values,
                        header_fn=functools.partial(
                            _sample_header, horizon=fixed_horizon,
                        ),
                        controller_factory=functools.partial(
                            _sample_controller_factory, horizon=fixed_horizon,
                        ),
                    )
        finally:
            if mps_started:
                _mps_stop()

        out_dir = self._save_results(
            horizon_result=horizon_result,
            sample_result=sample_result,
        )
        print(f"Saved benchmark outputs to {out_dir}")
        return out_dir

    def _run_as_worker(self) -> None:
        """Execute a single (controller, axis_value) benchmark in subprocess."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(_WORKER_FLAG, action="store_true")
        parser.add_argument("--ctrl-idx", type=int, required=True)
        parser.add_argument("--axis-value", type=float, required=True)
        parser.add_argument("--axis-name", type=str, required=True)
        parser.add_argument("--sweep-horizon", type=float, default=0.0)
        parser.add_argument("--output", type=str, required=True)
        args, _ = parser.parse_known_args()

        spec = self.controller_specs[args.ctrl_idx]

        if args.axis_name == "horizon":
            controller_factory = _horizon_controller_factory
        else:
            controller_factory = functools.partial(
                _sample_controller_factory, horizon=args.sweep_horizon,
            )

        result = self._run_single_point(spec, args.axis_value, controller_factory)
        _save_benchmark_result(result, args.output)

    def _launch_subprocess(
        self,
        script_path: str,
        ctrl_idx: int,
        axis_value: float,
        axis_name: str,
        output_path: str,
        sweep_horizon: float,
        gpu_id: int = 0,
    ) -> BenchmarkResult:
        """Launch a subprocess to run one benchmark point."""
        import time

        spec_name = self.controller_specs[ctrl_idx].name
        cmd = [
            sys.executable, script_path,
            _WORKER_FLAG,
            "--ctrl-idx", str(ctrl_idx),
            "--axis-value", str(axis_value),
            "--axis-name", axis_name,
            "--sweep-horizon", str(sweep_horizon),
            "--output", output_path,
        ]
        env = os.environ.copy()
        # Give each worker its own Warp kernel cache to avoid compilation races.
        cache_key = os.path.splitext(os.path.basename(output_path))[0]
        env["WARP_CACHE_PATH"] = os.path.join(
            tempfile.gettempdir(), f"warp_cache_{cache_key}",
        )
        if self.config.num_gpus > 1:
            # Respect parent's CUDA_VISIBLE_DEVICES if set.
            parent_devs = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if parent_devs is not None:
                dev_list = [d.strip() for d in parent_devs.split(",")]
                env["CUDA_VISIBLE_DEVICES"] = dev_list[gpu_id % len(dev_list)]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
            raise RuntimeError(
                f"Worker [{spec_name} @ {axis_name}={axis_value}] "
                f"failed after {elapsed:.1f}s:\n{stderr_tail}"
            )
        return _load_benchmark_result(output_path)

    def _resolve_max_workers(self, num_jobs: int) -> int:
        """Return the effective worker count for the thread pool."""
        if self.config.max_workers == "auto":
            return num_jobs
        return min(int(self.config.max_workers), num_jobs)

    def _pretrain_all_memory(
        self,
        controller_factory: Callable[[ControllerSpec, object, float], object],
    ) -> None:
        """Pre-train all memory controllers sequentially before parallel runs."""
        for spec in self.controller_specs:
            if not any(kw in spec.name.lower() for kw in ("memory",)):
                continue
            task = self.task_factory()
            controller = controller_factory(spec, task, 1.0)
            weights_path = self._memory_weights_path(controller, spec)
            if weights_path is not None and weights_path.exists():
                continue
            print(f"Pre-training {spec.name} before parallel runs...")
            self._prepare_controller(controller, spec)
            del controller, task

    def _run_single_point(
        self,
        spec: ControllerSpec,
        axis_value: float,
        controller_factory: Callable[[ControllerSpec, object, float], object],
    ) -> "BenchmarkResult":
        """Run benchmark for a single (controller, axis_value) pair."""
        task = self.task_factory()
        controller = controller_factory(spec, task, float(axis_value))
        self._prepare_controller(controller, spec)

        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        return run_benchmark(
            controller=controller,
            mj_model=mj_model,
            mj_data=mj_data,
            frequency=self.config.frequency,
            goal_threshold=self.config.goal_threshold,
            num_trials=self.config.num_trials,
            max_iterations=self.config.max_iterations,
            record_video=self.config.record_video,
            video_trial_index=self.config.video_trial_index,
        )

    def _store_result(
        self,
        result: "BenchmarkResult",
        ctrl_idx: int,
        value_idx: int,
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
        """Unpack a BenchmarkResult into the output arrays."""
        success[ctrl_idx, value_idx] = (
            100.0 * result.num_success / self.config.num_trials
        )

        replan_period = 1.0 / self.config.frequency
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

        frequency_mean[ctrl_idx, value_idx] = float(
            result.trial_frequencies.mean()
        )
        frequency_std[ctrl_idx, value_idx] = float(
            result.trial_frequencies.std()
        )

        state_store[ctrl_idx][value_idx] = result.state_trajectories
        control_store[ctrl_idx][value_idx] = result.control_trajectories
        trace_store[ctrl_idx][value_idx] = result.trace_trajectories

    def _run_sweep(
        self,
        *,
        axis_name: str,
        axis_label: str,
        axis_values: np.ndarray,
        header_fn: Callable[[float], str],
        controller_factory: Callable[[ControllerSpec, object, float], object],
    ) -> SweepResult:
        num_ctrl = len(self.controller_specs)
        num_vals = len(axis_values)

        success = np.zeros((num_ctrl, num_vals), dtype=np.float32)
        success_time = np.zeros((num_ctrl, num_vals), dtype=np.float32)
        success_time_std = np.zeros((num_ctrl, num_vals), dtype=np.float32)
        frequency_mean = np.zeros((num_ctrl, num_vals), dtype=np.float32)
        frequency_std = np.zeros((num_ctrl, num_vals), dtype=np.float32)

        state_store = [[None for _ in axis_values] for _ in range(num_ctrl)]
        control_store = [[None for _ in axis_values] for _ in range(num_ctrl)]
        trace_store = [[None for _ in axis_values] for _ in range(num_ctrl)]

        # Get mj_timestep for result unpacking.
        tmp_task = self.task_factory()
        mj_timestep = float(tmp_task.mj_model.opt.timestep)
        del tmp_task

        store_args = dict(
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

        parallel = self.config.parallel

        if parallel == "sequential":
            for value_idx, axis_value in enumerate(
                tqdm(axis_values, desc=f"{axis_name} sweep")
            ):
                print(header_fn(float(axis_value)))
                for ctrl_idx, spec in enumerate(self.controller_specs):
                    print(
                        f"[{value_idx + 1}/{num_vals}] Running {spec.name}..."
                    )
                    result = self._run_single_point(
                        spec, float(axis_value), controller_factory,
                    )
                    self._store_result(result, ctrl_idx, value_idx, **store_args)

        else:
            # Pre-train memory controllers sequentially to avoid races.
            self._pretrain_all_memory(controller_factory)

            # Build jobs: list of (ctrl_idx, value_idx, spec, axis_value).
            all_jobs = [
                (ctrl_idx, value_idx, spec, float(axis_value))
                for value_idx, axis_value in enumerate(axis_values)
                for ctrl_idx, spec in enumerate(self.controller_specs)
            ]

            if parallel == "controllers":
                # For each axis value, run controllers in parallel.
                batches = [
                    [j for j in all_jobs if j[1] == vi]
                    for vi in range(num_vals)
                ]
            elif parallel == "axis":
                # For each controller, run axis values in parallel.
                batches = [
                    [j for j in all_jobs if j[0] == ci]
                    for ci in range(num_ctrl)
                ]
            else:  # "all"
                batches = [all_jobs]

            script_path = os.path.abspath(sys.argv[0])
            sweep_horizon = float(self.config.sweep_horizon_for_samples)

            gpu_info = f" | {self.config.num_gpus} GPU(s)" if self.config.num_gpus > 1 else ""
            print(
                f"\nParallel mode: {parallel} | "
                f"{len(all_jobs)} total jobs across {len(batches)} batch(es){gpu_info}"
            )

            pbar = tqdm(total=len(all_jobs), desc=f"{axis_name} sweep", unit="job")

            for batch_idx, batch in enumerate(batches):
                workers = self._resolve_max_workers(len(batch))
                if len(batches) > 1:
                    tqdm.write(
                        f"\n--- Batch {batch_idx + 1}/{len(batches)} "
                        f"({len(batch)} jobs, {workers} workers) ---"
                    )
                else:
                    tqdm.write(f"Running {len(batch)} jobs with {workers} workers")

                num_gpus = max(1, self.config.num_gpus)
                tmp_dir = tempfile.mkdtemp(prefix="bench_")
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {}
                        for job_idx, (ctrl_idx, value_idx, spec, axis_value) in enumerate(batch):
                            out_path = os.path.join(
                                tmp_dir,
                                f"result_{ctrl_idx}_{value_idx}.npz",
                            )
                            gpu_id = job_idx % num_gpus
                            fut = pool.submit(
                                self._launch_subprocess,
                                script_path,
                                ctrl_idx,
                                axis_value,
                                axis_name,
                                out_path,
                                sweep_horizon,
                                gpu_id,
                            )
                            futures[fut] = (
                                ctrl_idx, value_idx, spec.name, axis_value,
                            )
                        import time as _time
                        batch_t0 = _time.perf_counter()
                        for fut in as_completed(futures):
                            ctrl_idx, value_idx, name, axis_value = futures[fut]
                            pbar.update(1)
                            wall = _time.perf_counter() - batch_t0
                            try:
                                result = fut.result()
                            except Exception as exc:
                                tqdm.write(
                                    f"  FAIL [{wall:6.1f}s]: {name} @ "
                                    f"{axis_name}={axis_value}: {exc}"
                                )
                                continue
                            self._store_result(
                                result, ctrl_idx, value_idx, **store_args,
                            )
                            tqdm.write(
                                f"  done [{wall:6.1f}s]: {name} @ "
                                f"{axis_name}={axis_value}"
                                f" — {result.num_success}/"
                                f"{self.config.num_trials} succeeded"
                            )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            pbar.close()

        return SweepResult(
            axis_name=axis_name,
            axis_label=axis_label,
            axis_values=axis_values,
            success=success,
            success_time=success_time,
            success_time_std=success_time_std,
            frequency_mean=frequency_mean,
            frequency_std=frequency_std,
            state_store=state_store,
            control_store=control_store,
            trace_store=trace_store,
        )

    def _prepare_controller(self, controller: object, spec: ControllerSpec) -> None:
        if not hasattr(controller, "pretrain_memory"):
            return

        weights_path = self._memory_weights_path(controller, spec)
        if weights_path is not None and weights_path.exists():
            print(f"Loading pretrained memory weights from {weights_path}")
            controller.load_pretrained_weights(weights_path)
            return

        trained = controller.pretrain_memory(verbose=True)
        if trained and weights_path is not None:
            controller.save_pretrained_weights(weights_path)
            print(f"Saved pretrained memory weights to {weights_path}")

    def _memory_weights_path(
        self,
        controller: object,
        spec: ControllerSpec,
    ) -> Optional[Path]:
        if self.config.pretrain_weights_dir is None:
            return None

        weights_dir = Path(self.config.pretrain_weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

        key = getattr(controller, "pretrained_weights_key", None)
        if key is None:
            key = f"{self.task_name}_{spec.name}"
        return weights_dir / f"{self._slugify(str(key))}.pt"

    def _save_results(
        self,
        *,
        horizon_result: SweepResult,
        sample_result: Optional[SweepResult],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            Path(ROOT)
            / "benchmark"
            / "benchmark_data"
            / f"{self.task_name}_benchmark_{self.config.output_tag}_{timestamp}"
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

    def _save_sweep(self, out_dir: Path, result: SweepResult) -> None:
        prefix = result.axis_name
        np.savetxt(out_dir / f"{prefix}_success_rate.csv", result.success, delimiter=",")
        np.savetxt(out_dir / f"{prefix}_success_time_mean.csv", result.success_time, delimiter=",")
        np.savetxt(out_dir / f"{prefix}_success_time_std.csv", result.success_time_std, delimiter=",")
        np.savetxt(out_dir / f"{prefix}_frequency_mean.csv", result.frequency_mean, delimiter=",")
        np.savetxt(out_dir / f"{prefix}_frequency_std.csv", result.frequency_std, delimiter=",")

        self._plot_matrix(
            result.axis_values,
            result.success,
            xlabel=result.axis_label,
            ylabel="Success Rate (%)",
            title=f"{self.task_name}: Success vs {result.axis_label}",
            out_path=out_dir / f"{prefix}_success_rate.png",
        )
        self._plot_matrix(
            result.axis_values,
            result.success_time,
            std=result.success_time_std,
            xlabel=result.axis_label,
            ylabel="Average Time-To-Success (s)",
            title=f"{self.task_name}: Time-To-Success vs {result.axis_label}",
            out_path=out_dir / f"{prefix}_success_time.png",
            skip_zeros=True,
        )
        self._plot_matrix(
            result.axis_values,
            result.frequency_mean,
            std=result.frequency_std,
            xlabel=result.axis_label,
            ylabel="Control Frequency (Hz)",
            title=f"{self.task_name}: Control Frequency vs {result.axis_label}",
            out_path=out_dir / f"{prefix}_frequency.png",
        )

    def _plot_matrix(
        self,
        x_values: np.ndarray,
        values: np.ndarray,
        *,
        std: Optional[np.ndarray] = None,
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: Path,
        skip_zeros: bool = False,
    ) -> None:
        plt.figure()
        for idx, spec in enumerate(self.controller_specs):
            if skip_zeros:
                mask = values[idx] != 0
                if not np.any(mask):
                    continue
                line, = plt.plot(x_values[mask], values[idx][mask], label=spec.name)
                if std is not None:
                    plt.fill_between(
                        x_values[mask],
                        values[idx][mask] - std[idx][mask],
                        values[idx][mask] + std[idx][mask],
                        alpha=0.2,
                        color=line.get_color(),
                    )
            else:
                line, = plt.plot(x_values, values[idx], label=spec.name)
                if std is not None:
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

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(
            char.lower() if char.isalnum() else "_"
            for char in value
        ).strip("_")

    @staticmethod
    def _jsonify(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: SeniorThesisBenchmarkSuite._jsonify(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SeniorThesisBenchmarkSuite._jsonify(val) for val in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
