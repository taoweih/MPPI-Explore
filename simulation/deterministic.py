"""Deterministic simulation and benchmarking utilities for mujoco_warp MPPI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np

from tasks.task_base import ALL_FIELDS, ROOT
from utils.video import VideoRecorder


@dataclass
class BenchmarkResult:
    """Structured benchmark output for one controller/task sweep."""

    num_success: int
    control_frequency_hz: float
    avg_success_iteration: float
    success_mask: np.ndarray
    success_iterations: np.ndarray
    trial_frequencies: np.ndarray
    state_trajectories: np.ndarray
    control_trajectories: np.ndarray
    trace_trajectories: np.ndarray


def _copy_mj_state(
    mj_model: mujoco.MjModel, dst: mujoco.MjData, src: mujoco.MjData
) -> None:
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.ctrl[:] = src.ctrl
    dst.time = src.time

    if dst.act.shape[0] > 0 and src.act.shape[0] > 0:
        dst.act[:] = src.act

    if src.mocap_pos.shape[0] > 0:
        dst.mocap_pos[:] = src.mocap_pos
        dst.mocap_quat[:] = src.mocap_quat

    mujoco.mj_forward(mj_model, dst)


def _single_world_data(
    mj_data: mujoco.MjData, fields: Sequence[str] = ALL_FIELDS
) -> dict:
    data_np = {}
    for field in fields:
        data_np[field] = np.asarray(getattr(mj_data, field), dtype=np.float32)[None, ...]
    return data_np


def _success_metric(controller, mj_data: mujoco.MjData) -> float:
    data_np = _single_world_data(mj_data)
    ctrl = np.asarray(mj_data.ctrl, dtype=np.float32)[None, ...]
    metric = controller.task.success_function(data_np, ctrl)
    return float(np.asarray(metric).reshape(-1)[0])


def _predict_nominal_traces(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    num_points: int,
    temp_data: Optional[mujoco.MjData] = None,
) -> Optional[np.ndarray]:
    """Predict nominal trace points for current control mean.

    Returns:
        Array of shape (ntrace, num_points, 3) or None if trace entities are
        not configured for the task.
    """
    ntrace = len(controller.task.trace_site_ids) + len(controller.task.trace_body_ids)
    if ntrace == 0:
        return None

    num_points = max(min(num_points, controller.ctrl_steps), 2)

    if temp_data is None:
        temp_data = mujoco.MjData(mj_model)
    _copy_mj_state(mj_model, temp_data, mj_data)

    tq = np.linspace(
        mj_data.time,
        mj_data.time + controller.plan_horizon,
        num_points,
        dtype=np.float32,
    )
    controls = controller.interp_func(tq, controller.tk, controller.mean[None, ...])[0]

    trace_frames = []
    trace_data = controller.task.get_trace_positions(
        _single_world_data(temp_data, fields=("xpos", "site_xpos"))
    )[0]
    trace_frames.append(trace_data)

    for u in controls[1:]:
        temp_data.ctrl[:] = u
        mujoco.mj_step(mj_model, temp_data)
        trace_data = controller.task.get_trace_positions(
            _single_world_data(temp_data, fields=("xpos", "site_xpos"))
        )[0]
        trace_frames.append(trace_data)

    return np.stack(trace_frames, axis=1)


def _save_screenshot(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    screenshot_path: str,
    dpi: int,
    width: int,
    height: int,
    fixed_camera_id: Optional[int],
    camera_distance: Optional[float],
    camera_azimuth: Optional[float],
    camera_elevation: Optional[float],
    camera_lookat: Optional[Sequence[float]],
) -> None:
    """Render one frame offscreen at high resolution and save as a PNG with DPI metadata."""
    from PIL import Image

    shot_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, shot_cam)
    if fixed_camera_id is not None:
        shot_cam.fixedcamid = fixed_camera_id
        shot_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    else:
        if camera_distance is not None:
            shot_cam.distance = float(camera_distance)
        if camera_azimuth is not None:
            shot_cam.azimuth = float(camera_azimuth)
        if camera_elevation is not None:
            shot_cam.elevation = float(camera_elevation)
        if camera_lookat is not None:
            shot_cam.lookat[:] = np.asarray(camera_lookat, dtype=np.float64)

    prev_offwidth = mj_model.vis.global_.offwidth
    prev_offheight = mj_model.vis.global_.offheight
    mj_model.vis.global_.offwidth = max(int(width), int(prev_offwidth))
    mj_model.vis.global_.offheight = max(int(height), int(prev_offheight))
    try:
        renderer = mujoco.Renderer(mj_model, height=int(height), width=int(width))
        try:
            renderer.update_scene(mj_data, shot_cam)
            pixels = renderer.render()
        finally:
            renderer.close()
    finally:
        mj_model.vis.global_.offwidth = prev_offwidth
        mj_model.vis.global_.offheight = prev_offheight

    out_dir = os.path.dirname(os.path.abspath(screenshot_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(pixels).save(screenshot_path, dpi=(int(dpi), int(dpi)))
    print(f"Saved screenshot to {screenshot_path}")


def run_interactive(
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
) -> list[float]:
    """Run deterministic simulation with optional traces and video recording."""
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
        f"Planning at {actual_frequency:.1f} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep:.1f} Hz"
    )

    if hasattr(controller, "reset"):
        controller.reset(seed=getattr(controller, "seed", 0))

    print("Warming up controller...")
    st = time.time()
    controller.optimize(mj_data)
    controller.optimize(mj_data)
    print(f"Warm-up took {time.time() - st:.3f}s")

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
                len(controller.task.trace_site_ids) + len(controller.task.trace_body_ids)
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

            plan_start = time.time()
            controller.optimize(mj_data)
            plan_time = time.time() - plan_start

            if visualize_fn is not None and step_idx % visualize_every == 0:
                visualize_fn(controller, step_idx)

            if show_traces and trace_geom_count > 0:
                trace_paths = _predict_nominal_traces(
                    controller, mj_model, mj_data, num_points=trace_steps,
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

            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time
            tq = np.arange(sim_steps_per_replan, dtype=np.float32) * sim_dt + t_curr
            us = controller.interp_func(tq, controller.tk, controller.mean[None, ...])[0]

            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = us[i]
                mujoco.mj_step(mj_model, mj_data)

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
                screenshot_every > 0
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

            if recorder is not None and recorder.is_recording and renderer is not None:
                if fixed_camera_id is None:
                    renderer.update_scene(mj_data, viewer.cam)
                else:
                    renderer.update_scene(mj_data, camera=fixed_camera_id)
                recorder.add_frame(renderer.render().tobytes())

            cost_history.append(_success_metric(controller, mj_data))

            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            rtr = step_dt / max(time.time() - start_time, 1e-9)
            print(
                f"Step {step_idx}: RTR={rtr:.2f}, plan={plan_time:.4f}s",
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

    if recorder is not None:
        recorder.stop()

    print()
    return cost_history


def run_benchmark(
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
) -> BenchmarkResult:
    """Benchmark controller over repeated deterministic trials."""
    print(
        f"Using controller {type(controller).__name__}\n"
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency:.1f} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep:.1f} Hz"
    )

    if hasattr(controller, "reset"):
        controller.reset(seed=trial_seed_base, initial_knots=initial_knots)

    print("Warming up controller...")
    st = time.time()
    controller.optimize(mj_data)
    controller.optimize(mj_data)
    print(f"Warm-up took {time.time() - st:.3f}s")

    mj_data_reset = mujoco.MjData(mj_model)
    _copy_mj_state(mj_model, mj_data_reset, mj_data)

    state_trajectories = np.zeros(
        (num_trials, max_iterations, mj_data.qpos.shape[0]), dtype=np.float32
    )
    control_trajectories = np.zeros(
        (num_trials, max_iterations, mj_data.ctrl.shape[0]), dtype=np.float32
    )

    num_traces = len(controller.task.trace_site_ids) + len(controller.task.trace_body_ids)
    trace_trajectories = np.zeros(
        (num_trials, max_iterations, num_traces, 3), dtype=np.float32
    )

    success_mask = np.zeros((num_trials,), dtype=bool)
    success_iterations = np.full((num_trials,), -1, dtype=np.int32)
    trial_frequencies = np.zeros((num_trials,), dtype=np.float32)

    total_plan_time = 0.0
    total_plan_steps = 0

    for trial_idx in range(num_trials):
        _copy_mj_state(mj_model, mj_data, mj_data_reset)
        trial_plan_time = 0.0
        trial_plan_steps = 0

        if hasattr(controller, "reset"):
            controller.reset(
                seed=trial_seed_base + trial_idx,
                initial_knots=initial_knots,
            )

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
                    mj_model, height=video_height, width=video_width
                )
            else:
                recorder = None

        reached_goal = False

        for iter_idx in range(max_iterations):
            plan_start = time.time()
            controller.optimize(mj_data)
            plan_time = time.time() - plan_start
            total_plan_time += plan_time
            total_plan_steps += 1
            trial_plan_time += plan_time
            trial_plan_steps += 1

            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time
            tq = np.arange(sim_steps_per_replan, dtype=np.float32) * sim_dt + t_curr
            us = controller.interp_func(tq, controller.tk, controller.mean[None, ...])[0]

            for k in range(sim_steps_per_replan):
                mj_data.ctrl[:] = us[k]
                mujoco.mj_step(mj_model, mj_data)
                if recorder is not None and recorder.is_recording and renderer is not None:
                    renderer.update_scene(mj_data)
                    recorder.add_frame(renderer.render().tobytes())

            state_trajectories[trial_idx, iter_idx] = np.asarray(
                mj_data.qpos, dtype=np.float32
            )
            control_trajectories[trial_idx, iter_idx] = np.asarray(
                mj_data.ctrl, dtype=np.float32
            )

            if num_traces > 0:
                trace_points = controller.task.get_trace_positions(
                    _single_world_data(mj_data, fields=("xpos", "site_xpos"))
                )[0]
                trace_trajectories[trial_idx, iter_idx] = trace_points

            rtr = step_dt / max(plan_time, 1e-9)
            print(
                f"  Trial {trial_idx+1}/{num_trials} "
                f"Step {iter_idx}: RTR={rtr:.2f}, plan={plan_time:.4f}s",
                end="\r",
            )

            if _success_metric(controller, mj_data) < goal_threshold:
                reached_goal = True
                success_mask[trial_idx] = True
                success_iterations[trial_idx] = iter_idx
                break

        if recorder is not None:
            recorder.stop()

        if not reached_goal:
            success_iterations[trial_idx] = max_iterations - 1

        trial_frequencies[trial_idx] = trial_plan_steps / max(trial_plan_time, 1e-9)

    print()
    num_success = int(success_mask.sum())
    if num_success > 0:
        avg_success_iteration = float(success_iterations[success_mask].mean())
    else:
        avg_success_iteration = 0.0

    control_frequency_hz = total_plan_steps / max(total_plan_time, 1e-9)

    print(
        f"  {num_success}/{num_trials} succeeded, "
        f"avg iter={avg_success_iteration:.1f}, "
        f"freq={control_frequency_hz:.1f} Hz"
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
