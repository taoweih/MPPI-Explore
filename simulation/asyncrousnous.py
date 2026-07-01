"""Asynchronous CPU-world simulation utilities.

The deterministic loops in :mod:`simulation.deterministic` are synchronous:
``controller.optimize(mj_data)`` blocks, and the live CPU MuJoCo world advances
only after a new plan is available.

This module models a more robot-like timing contract.  The CPU MuJoCo world
keeps stepping at the requested control period using the most recently
committed plan.  A single background worker optimizes from snapshots of the
live state; when a worker finishes, its plan is atomically committed for later
CPU steps.  If planning is late, the world does not wait.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np

from simulation.deterministic import (
    BenchmarkResult,
    _copy_mj_state,
    _save_screenshot,
    _single_world_data,
    _success_metric,
)
from tasks.task_base import ROOT
from utils.video import VideoRecorder

__all__ = ["AsyncPlanStats", "run_benchmark_async", "run_interactive_async"]


@dataclass
class AsyncPlanStats:
    """Bookkeeping for one asynchronous run."""

    submitted: int = 0
    completed: int = 0
    skipped_requests: int = 0
    plan_times: list[float] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)


class _AsyncPlanner:
    """Single-worker asynchronous wrapper around one controller instance."""

    def __init__(self, controller, mj_model: mujoco.MjModel) -> None:
        self.controller = controller
        self.mj_model = mj_model
        self.stats = AsyncPlanStats()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._plan_lock = threading.Lock()
        self._active_tk = np.asarray(controller.tk, dtype=np.float32).copy()
        self._active_mean = np.asarray(controller.mean, dtype=np.float32).copy()
        self._active_snapshot_time = float(self._active_tk[0])

    def reset_active_plan(self) -> None:
        """Refresh the committed plan from the controller's current mean."""
        with self._plan_lock:
            self._active_tk = np.asarray(self.controller.tk, dtype=np.float32).copy()
            self._active_mean = np.asarray(
                self.controller.mean, dtype=np.float32,
            ).copy()
            self._active_snapshot_time = float(self._active_tk[0])

    def is_busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def poll(self) -> bool:
        """Commit a completed background plan without blocking."""
        if self._future is None or not self._future.done():
            return False
        result = self._future.result()
        self._future = None
        self._commit(result)
        return True

    def submit_snapshot(self, mj_data: mujoco.MjData) -> bool:
        """Submit a plan request from a snapshot of the live CPU state."""
        self.poll()
        if self.is_busy():
            self.stats.skipped_requests += 1
            return False

        snapshot = mujoco.MjData(self.mj_model)
        _copy_mj_state(self.mj_model, snapshot, mj_data)
        submitted_at = time.perf_counter()
        self._future = self._executor.submit(
            self._optimize_snapshot,
            snapshot,
            submitted_at,
        )
        self.stats.submitted += 1
        return True

    def get_actions(self, query_times: np.ndarray) -> np.ndarray:
        """Return controls from the latest committed plan for ``query_times``."""
        with self._plan_lock:
            tk = self._active_tk.copy()
            mean = self._active_mean.copy()
        safe_times = np.clip(
            np.asarray(query_times, dtype=np.float32),
            float(tk[0]),
            float(tk[-1]),
        )
        return self.controller.interp_func(safe_times, tk, mean[None, ...])[0]

    def get_plan_copy(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Copy the committed plan for trace prediction or diagnostics."""
        with self._plan_lock:
            return (
                self._active_tk.copy(),
                self._active_mean.copy(),
                float(self._active_snapshot_time),
            )

    def drain(self, *, commit: bool = True) -> None:
        """Wait for an in-flight plan at shutdown or trial boundaries."""
        if self._future is None:
            return
        result = self._future.result()
        self._future = None
        if commit:
            self._commit(result)

    def close(self) -> None:
        self.drain(commit=False)
        self._executor.shutdown(wait=True)

    def _optimize_snapshot(
        self,
        snapshot: mujoco.MjData,
        submitted_at: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        self.controller.optimize(snapshot)
        plan_time = time.perf_counter() - submitted_at
        return (
            np.asarray(self.controller.tk, dtype=np.float32).copy(),
            np.asarray(self.controller.mean, dtype=np.float32).copy(),
            float(snapshot.time),
            float(plan_time),
        )

    def _commit(self, result: tuple[np.ndarray, np.ndarray, float, float]) -> None:
        tk, mean, snapshot_time, plan_time = result
        with self._plan_lock:
            self._active_tk = tk
            self._active_mean = mean
            self._active_snapshot_time = snapshot_time
        self.stats.completed += 1
        self.stats.plan_times.append(plan_time)
        self.stats.snapshot_times.append(snapshot_time)


def _warmup_controller(controller, mj_data: mujoco.MjData) -> None:
    print("Warming up controller...")
    st = time.time()
    controller.optimize(mj_data)
    controller.optimize(mj_data)
    if getattr(controller, "reset_after_warmup", False) and hasattr(controller, "reset"):
        controller.reset(seed=getattr(controller, "seed", 0))
    print(f"Warm-up took {time.time() - st:.3f}s")


def _step_current_async_plan(
    planner: _AsyncPlanner,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    sim_steps_per_replan: int,
) -> None:
    sim_dt = float(mj_model.opt.timestep)
    t_curr = float(mj_data.time)
    tq = np.arange(sim_steps_per_replan, dtype=np.float32) * sim_dt + t_curr
    actions = planner.get_actions(tq.astype(np.float32))
    for action in actions:
        planner.controller.task.apply_control_cpu(mj_data, action)
        mujoco.mj_step(mj_model, mj_data)


def _predict_nominal_traces_from_plan(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    tk: np.ndarray,
    mean: np.ndarray,
    num_points: int,
    temp_data: Optional[mujoco.MjData] = None,
) -> Optional[np.ndarray]:
    ntrace = len(controller.task.trace_site_ids) + len(controller.task.trace_body_ids)
    if ntrace == 0:
        return None

    num_points = max(min(num_points, controller.ctrl_steps), 2)
    if temp_data is None:
        temp_data = mujoco.MjData(mj_model)
    _copy_mj_state(mj_model, temp_data, mj_data)

    tq = np.linspace(
        float(mj_data.time),
        float(mj_data.time) + float(controller.plan_horizon),
        num_points,
        dtype=np.float32,
    )
    tq = np.clip(tq, float(tk[0]), float(tk[-1])).astype(np.float32)
    controls = controller.interp_func(tq, tk, mean[None, ...])[0]

    trace_frames = []
    trace_data = controller.task.get_trace_positions(
        _single_world_data(temp_data, fields=("xpos", "site_xpos"))
    )[0]
    trace_frames.append(trace_data)

    for action in controls[1:]:
        controller.task.apply_control_cpu(temp_data, action)
        mujoco.mj_step(mj_model, temp_data)
        trace_data = controller.task.get_trace_positions(
            _single_world_data(temp_data, fields=("xpos", "site_xpos"))
        )[0]
        trace_frames.append(trace_data)

    return np.stack(trace_frames, axis=1)


def run_interactive_async(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    show_traces: bool = False,
    max_steps: int = 500,
    trace_width: float = 3.0,
    trace_color: Sequence[float] = (1.0, 1.0, 1.0, 0.2),
    max_trace_points: int = 64,
    record_video: bool = False,
    video_output_dir: Optional[str] = None,
    video_width: int = 720,
    video_height: int = 480,
    video_crf: int = 2,
    video_preset: str = "slow",
    fixed_camera_id: Optional[int] = None,
    camera_distance: Optional[float] = None,
    camera_azimuth: Optional[float] = None,
    camera_elevation: Optional[float] = None,
    camera_lookat: Optional[Sequence[float]] = None,
    take_screenshot: bool = False,
    screenshot_path: Optional[str] = None,
    screenshot_dpi: int = 600,
    screenshot_width: int = 2400,
    screenshot_height: int = 2400,
    screenshot_step: int = 0,
    screenshot_every: int = 0,
    visualize_fn: Optional[Callable] = None,
    visualize_every: int = 50,
    warmup: bool = True,
) -> list[float]:
    """Run an interactive asynchronous MPC simulation.

    The live CPU world advances every control period with the last committed
    plan.  At the start of each period, a background plan request is submitted
    if the previous request has finished; otherwise the request is skipped and
    the CPU world keeps moving.
    """
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon}s horizon "
        f"with {controller.num_knots} knots."
    )

    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Requesting plans at {actual_frequency:.1f} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep:.1f} Hz"
    )

    if hasattr(controller, "reset"):
        controller.reset(seed=getattr(controller, "seed", 0))
    if warmup:
        _warmup_controller(controller, mj_data)
        if hasattr(controller, "reset"):
            controller.reset(seed=getattr(controller, "seed", 0))
    if hasattr(controller, "warm_start"):
        controller.warm_start(float(mj_data.time))

    planner = _AsyncPlanner(controller, mj_model)
    planner.reset_active_plan()

    screenshot_pending = bool(take_screenshot and screenshot_path is not None)
    if screenshot_pending and screenshot_step <= 0:
        _save_screenshot(
            mj_model=mj_model,
            mj_data=mj_data,
            screenshot_path=screenshot_path,
            dpi=screenshot_dpi,
            width=screenshot_width,
            height=screenshot_height,
            fixed_camera_id=fixed_camera_id,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_lookat=camera_lookat,
        )
        screenshot_pending = False

    recorder = None
    renderer = None
    if record_video:
        output_dir = video_output_dir or os.path.join(ROOT, "recordings")
        recorder = VideoRecorder(
            output_dir=output_dir,
            width=video_width,
            height=video_height,
            fps=actual_frequency,
            crf=video_crf,
            preset=video_preset,
        )
        if recorder.start():
            mj_model.vis.global_.offwidth = video_width
            mj_model.vis.global_.offheight = video_height
            renderer = mujoco.Renderer(mj_model, height=video_height, width=video_width)
        else:
            recorder = None

    trace_geom_count = 0
    trace_steps = max(min(max_trace_points, controller.ctrl_steps), 2)
    trace_temp_data = mujoco.MjData(mj_model) if show_traces else None
    cost_history: list[float] = []

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            if fixed_camera_id is not None:
                viewer.cam.fixedcamid = fixed_camera_id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            else:
                if camera_distance is not None:
                    viewer.cam.distance = float(camera_distance)
                if camera_azimuth is not None:
                    viewer.cam.azimuth = float(camera_azimuth)
                if camera_elevation is not None:
                    viewer.cam.elevation = float(camera_elevation)
                if camera_lookat is not None:
                    viewer.cam.lookat[:] = np.asarray(camera_lookat, dtype=np.float64)

            if show_traces:
                num_trace_entities = (
                    len(controller.task.trace_site_ids)
                    + len(controller.task.trace_body_ids)
                )
                trace_geom_count = num_trace_entities * (trace_steps - 1)
                for i in range(trace_geom_count):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        size=np.zeros(3),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(trace_color, dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1

            for step_idx in range(max_steps):
                start_time = time.time()

                committed = planner.poll()
                if (
                    visualize_fn is not None
                    and committed
                    and step_idx % visualize_every == 0
                    and not planner.is_busy()
                ):
                    visualize_fn(controller, step_idx)

                planner.submit_snapshot(mj_data)
                _step_current_async_plan(
                    planner,
                    mj_model,
                    mj_data,
                    sim_steps_per_replan,
                )
                planner.poll()

                if show_traces and trace_geom_count > 0:
                    tk, mean, _ = planner.get_plan_copy()
                    trace_paths = _predict_nominal_traces_from_plan(
                        controller,
                        mj_model,
                        mj_data,
                        tk,
                        mean,
                        num_points=trace_steps,
                        temp_data=trace_temp_data,
                    )
                    if trace_paths is not None:
                        ii = 0
                        for trace_id in range(trace_paths.shape[0]):
                            for j in range(trace_paths.shape[1] - 1):
                                mujoco.mjv_connector(
                                    viewer.user_scn.geoms[ii],
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    trace_width,
                                    trace_paths[trace_id, j],
                                    trace_paths[trace_id, j + 1],
                                )
                                ii += 1

                viewer.sync()

                if screenshot_pending and step_idx + 1 >= screenshot_step:
                    _save_screenshot(
                        mj_model=mj_model,
                        mj_data=mj_data,
                        screenshot_path=screenshot_path,
                        dpi=screenshot_dpi,
                        width=screenshot_width,
                        height=screenshot_height,
                        fixed_camera_id=fixed_camera_id,
                        camera_distance=camera_distance,
                        camera_azimuth=camera_azimuth,
                        camera_elevation=camera_elevation,
                        camera_lookat=camera_lookat,
                    )
                    screenshot_pending = False

                if (
                    take_screenshot
                    and screenshot_every > 0
                    and screenshot_path is not None
                    and (step_idx + 1) % screenshot_every == 0
                ):
                    base, ext = os.path.splitext(screenshot_path)
                    if not ext:
                        ext = ".png"
                    periodic_path = f"{base}_step_{step_idx + 1}{ext}"
                    _save_screenshot(
                        mj_model=mj_model,
                        mj_data=mj_data,
                        screenshot_path=periodic_path,
                        dpi=screenshot_dpi,
                        width=screenshot_width,
                        height=screenshot_height,
                        fixed_camera_id=fixed_camera_id,
                        camera_distance=camera_distance,
                        camera_azimuth=camera_azimuth,
                        camera_elevation=camera_elevation,
                        camera_lookat=camera_lookat,
                    )

                if (
                    recorder is not None
                    and recorder.is_recording
                    and renderer is not None
                ):
                    if fixed_camera_id is None:
                        renderer.update_scene(mj_data, viewer.cam)
                    else:
                        renderer.update_scene(mj_data, camera=fixed_camera_id)
                    recorder.add_frame(renderer.render().tobytes())

                cost_history.append(_success_metric(controller, mj_data))

                elapsed = time.time() - start_time
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

                completed = planner.stats.completed
                submitted = planner.stats.submitted
                skipped = planner.stats.skipped_requests
                print(
                    f"Step {step_idx}: plans={completed}/{submitted}, "
                    f"skipped={skipped}",
                    end="\r",
                )

                if not viewer.is_running():
                    break

            if fixed_camera_id is None:
                lx, ly, lz = viewer.cam.lookat
                print(
                    f"\nFinal free-camera state (paste into example):\n"
                    f"    camera_lookat = ({lx:.3f}, {ly:.3f}, {lz:.3f})\n"
                    f"    camera_distance = {viewer.cam.distance:.3f}\n"
                    f"    camera_azimuth = {viewer.cam.azimuth:.1f}\n"
                    f"    camera_elevation = {viewer.cam.elevation:.1f}"
                )
    finally:
        planner.close()
        if recorder is not None:
            recorder.stop()

    print()
    if planner.stats.plan_times:
        avg_plan = float(np.mean(planner.stats.plan_times))
        print(
            f"Async plans completed: {planner.stats.completed}/"
            f"{planner.stats.submitted}, avg plan time={avg_plan:.4f}s, "
            f"skipped requests={planner.stats.skipped_requests}"
        )
    return cost_history


def run_benchmark_async(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    goal_threshold: float = 1.0,
    num_trials: int = 100,
    max_iterations: int = 1000,
    trial_seed_base: int = 5,
    initial_knots: Optional[np.ndarray] = None,
    record_video: bool = False,
    video_trial_index: int = 0,
    video_output_dir: Optional[str] = None,
    video_width: int = 720,
    video_height: int = 480,
    real_time: bool = True,
) -> BenchmarkResult:
    """Benchmark the asynchronous loop over deterministic initial states.

    ``real_time=True`` keeps wall-clock and CPU simulation time aligned.  This
    is the closest model of a real plant that continues moving while planning.
    Set it to ``False`` only for fast functional smoke tests.
    """
    print(
        f"Using async controller {type(controller).__name__}\n"
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Requesting plans at {actual_frequency:.1f} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep:.1f} Hz"
    )

    if hasattr(controller, "reset"):
        controller.reset(seed=trial_seed_base, initial_knots=initial_knots)
    _warmup_controller(controller, mj_data)
    if hasattr(controller, "reset"):
        controller.reset(seed=trial_seed_base, initial_knots=initial_knots)

    mj_data_reset = mujoco.MjData(mj_model)
    _copy_mj_state(mj_model, mj_data_reset, mj_data)

    state_trajectories = np.zeros(
        (num_trials, max_iterations, mj_data.qpos.shape[0]),
        dtype=np.float32,
    )
    control_trajectories = np.zeros(
        (num_trials, max_iterations, mj_data.ctrl.shape[0]),
        dtype=np.float32,
    )

    num_traces = len(controller.task.trace_site_ids) + len(
        controller.task.trace_body_ids
    )
    trace_trajectories = np.zeros(
        (num_trials, max_iterations, num_traces, 3),
        dtype=np.float32,
    )

    success_mask = np.zeros((num_trials,), dtype=bool)
    success_iterations = np.full((num_trials,), -1, dtype=np.int32)
    trial_frequencies = np.zeros((num_trials,), dtype=np.float32)

    total_completed = 0
    total_elapsed = 0.0

    for trial_idx in range(num_trials):
        _copy_mj_state(mj_model, mj_data, mj_data_reset)
        if hasattr(controller, "reset"):
            controller.reset(
                seed=trial_seed_base + trial_idx,
                initial_knots=initial_knots,
            )
        if hasattr(controller, "warm_start"):
            controller.warm_start(float(mj_data.time))

        planner = _AsyncPlanner(controller, mj_model)
        planner.reset_active_plan()

        recorder = None
        renderer = None
        if record_video and trial_idx == video_trial_index:
            output_dir = video_output_dir or os.path.join(ROOT, "recordings")
            recorder = VideoRecorder(
                output_dir=output_dir,
                width=video_width,
                height=video_height,
                fps=actual_frequency,
            )
            if recorder.start():
                mj_model.vis.global_.offwidth = video_width
                mj_model.vis.global_.offheight = video_height
                renderer = mujoco.Renderer(
                    mj_model,
                    height=video_height,
                    width=video_width,
                )
            else:
                recorder = None

        reached_goal = False
        trial_start = time.perf_counter()
        trial_elapsed = 0.0

        try:
            for iter_idx in range(max_iterations):
                loop_start = time.perf_counter()

                planner.poll()
                planner.submit_snapshot(mj_data)
                _step_current_async_plan(
                    planner,
                    mj_model,
                    mj_data,
                    sim_steps_per_replan,
                )
                planner.poll()

                if (
                    recorder is not None
                    and recorder.is_recording
                    and renderer is not None
                ):
                    renderer.update_scene(mj_data)
                    recorder.add_frame(renderer.render().tobytes())

                state_trajectories[trial_idx, iter_idx] = np.asarray(
                    mj_data.qpos,
                    dtype=np.float32,
                )
                control_trajectories[trial_idx, iter_idx] = np.asarray(
                    mj_data.ctrl,
                    dtype=np.float32,
                )

                if num_traces > 0:
                    trace_points = controller.task.get_trace_positions(
                        _single_world_data(mj_data, fields=("xpos", "site_xpos"))
                    )[0]
                    trace_trajectories[trial_idx, iter_idx] = trace_points

                if real_time:
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < step_dt:
                        time.sleep(step_dt - elapsed)

                print(
                    f"  Trial {trial_idx+1}/{num_trials} Step {iter_idx}: "
                    f"plans={planner.stats.completed}/{planner.stats.submitted}, "
                    f"skipped={planner.stats.skipped_requests}",
                    end="\r",
                )

                if _success_metric(controller, mj_data) < goal_threshold:
                    reached_goal = True
                    success_mask[trial_idx] = True
                    success_iterations[trial_idx] = iter_idx
                    break
        finally:
            trial_elapsed = time.perf_counter() - trial_start
            planner.drain(commit=False)
            if recorder is not None:
                recorder.stop()
            planner.close()

        total_elapsed += trial_elapsed
        total_completed += planner.stats.completed
        trial_frequencies[trial_idx] = planner.stats.completed / max(
            trial_elapsed,
            1e-9,
        )

        if not reached_goal:
            success_iterations[trial_idx] = max_iterations - 1

    print()
    num_success = int(success_mask.sum())
    if num_success > 0:
        avg_success_iteration = float(success_iterations[success_mask].mean())
    else:
        avg_success_iteration = 0.0

    control_frequency_hz = total_completed / max(total_elapsed, 1e-9)
    print(
        f"  {num_success}/{num_trials} succeeded, "
        f"avg iter={avg_success_iteration:.1f}, "
        f"async completed-plan freq={control_frequency_hz:.1f} Hz"
    )

    return BenchmarkResult(
        num_success=num_success,
        control_frequency_hz=float(control_frequency_hz),
        avg_success_iteration=avg_success_iteration,
        success_mask=success_mask,
        success_iterations=success_iterations,
        trial_frequencies=trial_frequencies,
        state_trajectories=state_trajectories,
        control_trajectories=control_trajectories,
        trace_trajectories=trace_trajectories,
    )
