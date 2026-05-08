"""Sensitivity sweep suite — vary one algorithm hyperparameter at a time.

Unlike the (T, K) benchmark which sweeps planning horizon and number of samples
across all four controllers, sensitivity sweeps hold (T, K) fixed and vary one
algorithm-internal knob (e.g. KDE bandwidth, hashgrid table size) for ONE
controller.  Each ``ParamSweep`` produces success-rate / time-to-success /
control-frequency curves vs the swept parameter.

Sequential execution — run overnight if needed.  Each task subdir gets a
single timestamped output directory containing one CSV+PNG triplet per sweep
plus a metadata.json describing the experiment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from simulation.deterministic import run_benchmark
from tasks.task_base import ROOT


# ──────────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ParamSweep:
    """One hyperparameter sweep: hold (T, K) fixed, vary one parameter."""

    name: str                             # e.g. "kde_bandwidth"
    axis_label: str                       # plot axis label
    controller_name: str                  # display name (for plot legend / metadata)
    values: np.ndarray                    # sweep values (1-D)
    factory: Callable[..., object]        # (task, value) → controller (must be picklable-free; uses the task)
    log_x: bool = False                   # plot with log-scale x-axis


@dataclass
class SensitivityConfig:
    """Trial-level config (matches SweepConfig fields used by run_benchmark)."""

    num_trials: int = 20
    frequency: float = 50.0
    goal_threshold: float = 0.5
    max_iterations: int = 1000
    output_tag: str = "sensitivity"


# ──────────────────────────────────────────────────────────────────────
# Per-sweep result (saved to disk)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    name: str
    axis_label: str
    controller_name: str
    values: np.ndarray
    success: np.ndarray              # (n,) success rate %
    success_time_mean: np.ndarray    # (n,) seconds
    success_time_std: np.ndarray     # (n,) seconds
    frequency_mean: np.ndarray       # (n,) Hz
    frequency_std: np.ndarray        # (n,) Hz
    log_x: bool = False


@dataclass
class BaselineResult:
    """Single-point reference (e.g. plain MPPI) drawn as a horizontal line on every sweep plot."""
    name: str
    success: float                   # success rate %
    success_time_mean: float         # seconds
    success_time_std: float          # seconds
    frequency_mean: float            # Hz
    frequency_std: float             # Hz


# ──────────────────────────────────────────────────────────────────────
# Suite
# ──────────────────────────────────────────────────────────────────────


class SensitivitySuite:
    """Sequentially run a list of ParamSweeps on one task; save CSVs + plots."""

    def __init__(
        self,
        task_name: str,
        task_factory: Callable[[], object],
        sweeps: Sequence[ParamSweep],
        config: SensitivityConfig,
        baseline_factory: Optional[Callable[..., object]] = None,
        baseline_name: str = "MPPI baseline",
    ) -> None:
        self.task_name = task_name
        self.task_factory = task_factory
        self.sweeps = list(sweeps)
        self.config = config
        self.baseline_factory = baseline_factory
        self.baseline_name = baseline_name

    # ── Top-level entry point ────────────────────────────────────────

    def run(self) -> Path:
        out_dir = self._make_output_dir()
        total_jobs = sum(len(sw.values) for sw in self.sweeps)

        print(f"\n{'='*60}")
        print(f"Sensitivity suite: {self.task_name}")
        print(f"Sweeps: {len(self.sweeps)},  total points: {total_jobs}")
        print(f"Trials per point: {self.config.num_trials},  "
              f"max_iter: {self.config.max_iterations}")
        print(f"Output dir: {out_dir}")
        print(f"{'='*60}")

        # Baseline run (one point, drawn as horizontal reference on every sweep plot).
        baseline: Optional[BaselineResult] = None
        if self.baseline_factory is not None:
            print(f"\n── Baseline: {self.baseline_name} ──")
            baseline = self._run_baseline()
            self._save_baseline(out_dir, baseline)

        all_results: list[SweepResult] = []
        for i, sweep in enumerate(self.sweeps):
            print(f"\n── Sweep {i+1}/{len(self.sweeps)}: {sweep.name} "
                  f"({sweep.controller_name}) ──")
            print(f"   values: {list(sweep.values)}")
            result = self._run_sweep(sweep)
            all_results.append(result)
            self._save_sweep(out_dir, result, baseline)

        self._save_metadata(out_dir, all_results, baseline)
        print(f"\nAll sweeps complete.  Output: {out_dir}")
        return out_dir

    # ── Baseline ─────────────────────────────────────────────────────

    def _run_baseline(self) -> BaselineResult:
        """One-point benchmark used as a horizontal reference on every plot."""
        task = self.task_factory()
        controller = self.baseline_factory(task)

        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        result = run_benchmark(
            controller=controller,
            mj_model=mj_model,
            mj_data=mj_data,
            frequency=self.config.frequency,
            goal_threshold=self.config.goal_threshold,
            num_trials=self.config.num_trials,
            max_iterations=self.config.max_iterations,
        )

        success_rate = 100.0 * result.num_success / self.config.num_trials
        replan_period = 1.0 / self.config.frequency
        sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
        step_dt = float(sim_steps_per_replan * mj_model.opt.timestep)
        trial_times = result.success_iterations.astype(np.float32) * step_dt
        if result.num_success > 0:
            succ = trial_times[result.success_mask]
            t_mean = float(succ.mean())
            t_std = float(succ.std())
        else:
            t_mean = 0.0
            t_std = 0.0

        baseline = BaselineResult(
            name=self.baseline_name,
            success=success_rate,
            success_time_mean=t_mean,
            success_time_std=t_std,
            frequency_mean=float(result.trial_frequencies.mean()),
            frequency_std=float(result.trial_frequencies.std()),
        )
        print(f"   {result.num_success}/{self.config.num_trials} succ "
              f"| time_to_succ {t_mean:.2f}s | freq {baseline.frequency_mean:.1f} Hz")
        return baseline

    def _save_baseline(self, out_dir: Path, baseline: BaselineResult) -> None:
        np.savetxt(
            out_dir / "baseline.csv",
            np.array([[baseline.success, baseline.success_time_mean,
                       baseline.success_time_std, baseline.frequency_mean,
                       baseline.frequency_std]]),
            delimiter=",",
            header="success_rate,success_time_mean,success_time_std,"
                   "frequency_mean,frequency_std",
            comments="",
        )

    # ── One sweep (one parameter, all values, all trials) ───────────

    def _run_sweep(self, sweep: ParamSweep) -> SweepResult:
        n = len(sweep.values)
        success = np.zeros(n, dtype=np.float32)
        success_time_mean = np.zeros(n, dtype=np.float32)
        success_time_std = np.zeros(n, dtype=np.float32)
        frequency_mean = np.zeros(n, dtype=np.float32)
        frequency_std = np.zeros(n, dtype=np.float32)

        for vi, value in enumerate(sweep.values):
            print(f"   [{vi+1}/{n}] {sweep.name} = {value!r}")
            task = self.task_factory()
            controller = sweep.factory(task, value)

            mj_model = task.mj_model
            mj_data = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, mj_data)

            result = run_benchmark(
                controller=controller,
                mj_model=mj_model,
                mj_data=mj_data,
                frequency=self.config.frequency,
                goal_threshold=self.config.goal_threshold,
                num_trials=self.config.num_trials,
                max_iterations=self.config.max_iterations,
            )

            success[vi] = 100.0 * result.num_success / self.config.num_trials

            # Time-to-success in seconds (success_iterations × physics step_dt).
            replan_period = 1.0 / self.config.frequency
            sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
            step_dt = float(sim_steps_per_replan * mj_model.opt.timestep)
            trial_times = result.success_iterations.astype(np.float32) * step_dt
            if result.num_success > 0:
                succ = trial_times[result.success_mask]
                success_time_mean[vi] = float(succ.mean())
                success_time_std[vi] = float(succ.std())

            frequency_mean[vi] = float(result.trial_frequencies.mean())
            frequency_std[vi] = float(result.trial_frequencies.std())

            print(f"       {result.num_success}/{self.config.num_trials} succ "
                  f"| time_to_succ {success_time_mean[vi]:.2f}s "
                  f"| freq {frequency_mean[vi]:.1f} Hz")

        return SweepResult(
            name=sweep.name,
            axis_label=sweep.axis_label,
            controller_name=sweep.controller_name,
            values=np.asarray(sweep.values),
            success=success,
            success_time_mean=success_time_mean,
            success_time_std=success_time_std,
            frequency_mean=frequency_mean,
            frequency_std=frequency_std,
            log_x=sweep.log_x,
        )

    # ── Output ───────────────────────────────────────────────────────

    def _make_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            Path(ROOT)
            / "sensitivity"
            / "sensitivity_data"
            / f"{self.task_name}_sensitivity_{self.config.output_tag}_{timestamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_sweep(
        self,
        out_dir: Path,
        r: SweepResult,
        baseline: Optional[BaselineResult] = None,
    ) -> None:
        np.savetxt(out_dir / f"{r.name}__values.csv", r.values, delimiter=",")
        np.savetxt(out_dir / f"{r.name}__success_rate.csv", r.success, delimiter=",")
        np.savetxt(out_dir / f"{r.name}__success_time_mean.csv", r.success_time_mean, delimiter=",")
        np.savetxt(out_dir / f"{r.name}__success_time_std.csv", r.success_time_std, delimiter=",")
        np.savetxt(out_dir / f"{r.name}__frequency_mean.csv", r.frequency_mean, delimiter=",")
        np.savetxt(out_dir / f"{r.name}__frequency_std.csv", r.frequency_std, delimiter=",")

        self._plot(out_dir, r, r.success, ylabel="Success Rate (%)",
                   filename=f"{r.name}__success_rate.png",
                   baseline_value=baseline.success if baseline else None,
                   baseline_name=baseline.name if baseline else None)
        self._plot(out_dir, r, r.success_time_mean, ylabel="Time to Success (s)",
                   filename=f"{r.name}__success_time.png",
                   std=r.success_time_std, skip_zeros=True,
                   baseline_value=(baseline.success_time_mean
                                   if (baseline and baseline.success_time_mean > 0) else None),
                   baseline_name=baseline.name if baseline else None)
        self._plot(out_dir, r, r.frequency_mean, ylabel="Control Frequency (Hz)",
                   filename=f"{r.name}__frequency.png", std=r.frequency_std,
                   baseline_value=baseline.frequency_mean if baseline else None,
                   baseline_name=baseline.name if baseline else None)

    def _plot(
        self,
        out_dir: Path,
        r: SweepResult,
        y: np.ndarray,
        *,
        ylabel: str,
        filename: str,
        std: Optional[np.ndarray] = None,
        skip_zeros: bool = False,
        baseline_value: Optional[float] = None,
        baseline_name: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = r.values
        if skip_zeros:
            mask = y != 0
            if not np.any(mask):
                plt.close(fig)
                return
            ax.plot(x[mask], y[mask], marker="o", color="#1F77B4", label=r.controller_name)
            if std is not None:
                ax.fill_between(x[mask], y[mask] - std[mask], y[mask] + std[mask],
                                alpha=0.2, color="#1F77B4", linewidth=0)
        else:
            ax.plot(x, y, marker="o", color="#1F77B4", label=r.controller_name)
            if std is not None:
                ax.fill_between(x, y - std, y + std, alpha=0.2, color="#1F77B4", linewidth=0)

        if baseline_value is not None:
            label = baseline_name or "Baseline"
            ax.axhline(baseline_value, linestyle="--", color="#808080",
                       linewidth=1.5, label=f"{label} ({baseline_value:.2f})")

        if r.log_x:
            ax.set_xscale("log")
        ax.set_xlabel(r.axis_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{r.controller_name}: {ylabel} vs {r.axis_label}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    def _save_metadata(
        self,
        out_dir: Path,
        results: list[SweepResult],
        baseline: Optional[BaselineResult] = None,
    ) -> None:
        meta: dict[str, Any] = {
            "task_name": self.task_name,
            "config": asdict(self.config),
            "sweeps": [
                {
                    "name": r.name,
                    "axis_label": r.axis_label,
                    "controller_name": r.controller_name,
                    "values": r.values.tolist(),
                    "log_x": r.log_x,
                }
                for r in results
            ],
        }
        if baseline is not None:
            meta["baseline"] = asdict(baseline)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
