"""Density-Guided MPPI: stage-resampled rollouts that favour low-density regions.

Algorithm (one optimize step):

    1. sample_knots()
           K_i = clip(μ + σ · 𝒩(0, I),  u_min, u_max)
           u_i(t) = interp(K_i)
    2. rollout(controls, knots)        — staged
           For each stage boundary n = 0 .. S-2:
               (a) physics for stage's timesteps          → accumulate running cost
               (b) extract density state s_i                 via task/model extractor
               (c) density_model.launch_compute(s)          ρ(s_i)
               (d) density_model.launch_resample(idx, n)    idx ∝ resample(1/ρ^α)
               (e) reshuffle (controls, knots, qpos, qvel, time) by idx
               (f) regenerate trailing knots
                       K[k_start:, :] ← clip(μ + σ · 𝒩)
                   and re-interpolate controls (zero-order hold)
           Final stage: physics only, then task terminal cost.
    3. update_weights(costs, knots)
           w_i = exp(-J_i / λ) / Z;   μ ← Σ w_i · K_i

Density estimation + resampling is delegated to a `DensityModel` instance
(default `KDEDensityModel`).  Swap in any other estimator implementing
the `DensityModel` interface without touching this file.

The full staged rollout is captured as a single CUDA graph.  Only host
syncs per optimize: data uploads (controls, knots, mean, density-model
offsets, per-stage noise) and final cost / knot readback.

Requires `spline_type="zero"` because the in-graph re-interpolation step
is zero-order-hold.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from algs._graph import capture_graph
from algs.base import BaseMPPI
from algs.density_models import DensityModel, KDEDensityModel
from utils.warp_kernels import (
    gather_1d_float,
    gather_2d_float,
    gather_3d_float,
)


# ──────────────────────────────────────────────────────────────────────
# Staging kernels (knot regeneration + ZOH re-interpolation)
# ──────────────────────────────────────────────────────────────────────


@wp.kernel
def regenerate_knots(
    knots:        wp.array3d(dtype=wp.float32),  # (N, K, nu)
    mean:         wp.array2d(dtype=wp.float32),  # (K, nu)
    noise:        wp.array3d(dtype=wp.float32),  # (N, remaining_K, nu)
    u_min:        wp.array1d(dtype=wp.float32),
    u_max:        wp.array1d(dtype=wp.float32),
    noise_level:  float,
    k_start:      int,
    remaining_K:  int,
    nu:           int,
):
    """knots[:, k_start:, :] ← clip(μ + σ · 𝒩(0, I))."""
    i = wp.tid()
    for k in range(remaining_K):
        for d in range(nu):
            val = mean[k_start + k, d] + noise_level * noise[i, k, d]
            knots[i, k_start + k, d] = wp.clamp(val, u_min[d], u_max[d])


@wp.kernel
def zero_order_interp(
    knots:        wp.array3d(dtype=wp.float32),  # (N, K, nu)
    controls:     wp.array3d(dtype=wp.float32),  # (N, T, nu) output
    knot_indices: wp.array1d(dtype=wp.int32),    # (T,) zero-order-hold mapping
    T:            int,
    nu:           int,
):
    """controls[i, t, :] = knots[i, knot_indices[t], :]."""
    i = wp.tid()
    for t in range(T):
        k = knot_indices[t]
        for d in range(nu):
            controls[i, t, d] = knots[i, k, d]


# ──────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────


class DensityGuidedMPPI(BaseMPPI):
    """MPPI with KDE-based stage resampling for state-space exploration.

    Density estimation and resampling are delegated to `density_model`
    (an instance of `algs.density_models.DensityModel`; default
    `KDEDensityModel`).  This controller owns only the staging
    machinery — physics stepping, state extraction, reshuffle, and
    trailing-knot regeneration.
    """

    def __init__(
        self,
        task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
        # ── density-guided specific ──
        density_model: Optional[DensityModel] = None,
        num_knots_per_stage: int = 4,
        # ── optional: override the state extraction used for density estimation ──
        state_extract_fn: Optional[Callable] = None,
        state_dim: Optional[int] = None,
    ) -> None:
        """
        Optional overrides:
            state_extract_fn(task_state, out_wp, weight_wp) -> None
                Custom kernel-launching callable that fills `out_wp` (shape
                (num_samples, state_dim)) with the state for density estimation.
                Defaults to a density-model extractor when available (KNN uses
                full qpos+qvel by default), otherwise to `task.extract_state`.
            state_dim:
                Width of the state vector produced by `state_extract_fn`.
                Required when `state_extract_fn` is given; otherwise inferred
                from `task.state_dim`.
        """
        if spline_type != "zero":
            raise NotImplementedError(
                "CUDA graph capture for staged rollout requires spline_type='zero'. "
                f"Got '{spline_type}'."
            )
        super().__init__(
            task=task, num_samples=num_samples, noise_level=noise_level,
            temperature=temperature, plan_horizon=plan_horizon,
            spline_type=spline_type, num_knots=num_knots,
            iterations=iterations, seed=seed,
        )

        self.num_knots_per_stage = num_knots_per_stage
        self.density_model = density_model or KDEDensityModel(bandwidth=1.0)
        configure_density_model_from_task = getattr(
            self.density_model, "configure_from_task", None,
        )
        configure_density_model = getattr(
            self.density_model, "configure_from_model", None,
        )
        if configure_density_model_from_task is not None:
            configure_density_model_from_task(task)
        elif configure_density_model is not None:
            configure_density_model(task.mj_model)

        # Resolve state extraction (default = task's, optional = user-provided).
        if state_extract_fn is not None:
            if state_dim is None:
                raise ValueError(
                    "state_dim must be given alongside state_extract_fn."
            )
            self._state_extract_fn = state_extract_fn
            self._state_dim = int(state_dim)
            self._density_model_extracts_state = False
        elif hasattr(self.density_model, "launch_state_extract"):
            self._state_extract_fn = None
            self._state_dim = int(self.density_model.state_dim)
            self._density_model_extracts_state = True
            self._density_model_task_state_dim = int(
                getattr(self.density_model, "task_state_dim", 0)
            )
        else:
            self._state_extract_fn = task.extract_state
            self._state_dim = task.state_dim
            self._density_model_extracts_state = False
            self._density_model_task_state_dim = 0

        if state_extract_fn is not None:
            self._density_model_task_state_dim = 0

        self._init_stage_geometry()
        self._alloc_staging_buffers()
        self.density_model.alloc(
            num_samples=self.num_samples,
            state_dim=self._state_dim,
            num_resample_stages=max(self._num_stages - 1, 1),
            device=self._device,
        )

        # CUDA graphs (built lazily after kernels are compiled).
        self._density_graph: Optional[wp.Graph] = None
        self._simple_rollout_graph: Optional[wp.Graph] = None

    # ══════════════════════════════════════════════════════════════════
    # Algorithm
    # ══════════════════════════════════════════════════════════════════

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """One density-guided MPPI optimization step.  Returns updated mean knots."""
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        init_state = (
            self._make_initial_state(mj_data) if self.iterations > 1 else None
        )
        for it in range(self.iterations):
            if it > 0:
                self._restore_state(init_state)
            knots, controls = self.sample_knots()
            costs, final_knots = self.rollout(controls, knots)
            self.update_weights(costs, final_knots)

        return self.mean

    def sample_knots(self):
        """Sample noisy control knots and interpolate them to controls.

            K_i = clip(μ + σ · 𝒩(0, I),  u_min, u_max),     i = 1..N
            u_i(t) = interp(K_i)

        Returns:
            knots:    (num_samples, num_knots, nu)
            controls: (num_samples, ctrl_steps, nu)
        """
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu),
        ).astype(np.float32)
        knots = self.mean + self.noise_level * noise
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    def rollout(self, controls: np.ndarray, knots: np.ndarray):
        """Staged rollout via captured CUDA graph.  Returns (costs, final_knots).

        For each stage boundary n = 0 .. S-2:
            (a) physics for stage's timesteps         → accumulate running cost
            (b) state s_i = density/task extract_state(...)
            (c) density_model.launch_compute(s)         ρ(s_i)
            (d) density_model.launch_resample(idx, n)   idx ∝ resample(1/ρ^α)
            (e) reshuffle (controls, knots, qpos, qvel, time) by idx
            (f) regenerate trailing knots; re-interp controls (ZOH)
        Final stage: physics only, then task terminal cost.
        """
        self._controls_wp.assign(controls)

        if self._num_stages <= 1:
            if self._simple_rollout_graph is None:
                with capture_graph() as cap:
                    self._running_costs_wp.zero_()
                    self._terminal_costs_wp.zero_()
                    self._step_physics_range(0, self.ctrl_steps)
                    self._emit_terminal_cost()
                self._simple_rollout_graph = cap.graph
            wp.capture_launch(self._simple_rollout_graph)
            costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
            return costs, knots

        self._knots_wp.assign(knots)
        self._mean_wp.assign(self.mean)

        # Stochastic inputs read inside the captured graph.
        self.density_model.randomize_offsets(self.rng)
        for n in range(self._num_stages - 1):
            buf = self._stage_noise[n]
            if buf is not None:
                k_start = (n + 1) * self.num_knots_per_stage
                remaining = self.num_knots - k_start
                noise = self.rng.standard_normal(
                    (self.num_samples, remaining, self.task.nu),
                ).astype(np.float32)
                buf.assign(noise)

        def emit_density_stages() -> None:
            self._running_costs_wp.zero_()

            for n in range(self._num_stages - 1):
                self._step_physics_range(
                    n * self._timesteps_per_stage,
                    (n + 1) * self._timesteps_per_stage,
                )
                self._extract_density_state()
                self.density_model.launch_compute(self._states_wp)
                self.density_model.launch_resample(self._indices_wp, stage_idx=n)

                self._reshuffle_by_indices()
                self._regenerate_trailing_knots(stage=n)

            # Final stage: physics only, then terminal cost.
            self._step_physics_range(
                (self._num_stages - 1) * self._timesteps_per_stage,
                self.ctrl_steps,
            )

            self._terminal_costs_wp.zero_()
            self._emit_terminal_cost()

        if self._density_graph is None:
            with capture_graph() as cap:
                emit_density_stages()
            self._density_graph = cap.graph
        wp.capture_launch(self._density_graph)

        costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
        final_knots = self._knots_wp.numpy()
        return costs, final_knots

    def update_weights(
        self, total_costs: np.ndarray, knots: np.ndarray,
    ) -> np.ndarray:
        """Softmax mean update.

            w_i = exp(-J_i / λ) / Σ_j exp(-J_j / λ)
            μ ← Σ_i w_i · K_i
        """
        shifted = -total_costs / self.temperature
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0)
        return total_costs

    # ══════════════════════════════════════════════════════════════════
    # Stage operations — one wp.launch per algorithm operation
    # ══════════════════════════════════════════════════════════════════

    def _emit_terminal_cost(self) -> None:
        """Evaluate the task terminal cost into the terminal-cost buffer."""
        self.task.launch_terminal_cost(
            self._task_state, self._terminal_costs_wp,
        )

    def _extract_density_state(self) -> None:
        """Fill the density-model state buffer for the current rollout worlds."""
        if self._density_model_extracts_state:
            if self._density_model_task_state_dim > 0:
                self.task.extract_state(
                    self._task_state,
                    self._density_task_states_wp,
                    self._density_task_state_weight_wp,
                )
                self.density_model.launch_state_extract(
                    self.warp_data.qpos,
                    self.warp_data.qvel,
                    self._states_wp,
                    self._density_task_states_wp,
                )
                return
            self.density_model.launch_state_extract(
                self.warp_data.qpos, self.warp_data.qvel, self._states_wp,
            )
        else:
            self._state_extract_fn(
                self._task_state, self._states_wp, self._state_weight_wp,
            )

    def _step_physics_range(self, t_start: int, t_end: int) -> None:
        """For each t in [t_start, t_end): copy ctrl, mjwarp.step, accumulate running cost."""
        for t in range(t_start, t_end):
            self.task.launch_step_control(
                self._task_state,
                self._controls_wp[:, t, :],
                self.warp_data.ctrl,
            )
            mjwarp.step(self.task.model, self.warp_data)
            self.task.launch_running_cost(
                self._task_state, self.warp_data.ctrl,
                self._running_costs_wp, self.dt,
            )

    def _reshuffle_by_indices(self) -> None:
        """Gather (controls, knots, running_cost, qpos, qvel, time) by self._indices_wp."""
        N, nu = self.num_samples, self.task.nu

        wp.launch(gather_3d_float, dim=N, inputs=[
            self._controls_wp, self._tmp_controls, self._indices_wp,
            self.ctrl_steps, nu])
        wp.copy(self._controls_wp, self._tmp_controls)

        wp.launch(gather_3d_float, dim=N, inputs=[
            self._knots_wp, self._tmp_knots, self._indices_wp,
            self.num_knots, nu])
        wp.copy(self._knots_wp, self._tmp_knots)

        wp.launch(gather_1d_float, dim=N, inputs=[
            self._running_costs_wp, self._tmp_costs, self._indices_wp])
        wp.copy(self._running_costs_wp, self._tmp_costs)

        wp.launch(gather_2d_float, dim=N, inputs=[
            self.warp_data.qpos, self._tmp_qpos, self._indices_wp, self.task.nq])
        wp.copy(self.warp_data.qpos, self._tmp_qpos)

        wp.launch(gather_2d_float, dim=N, inputs=[
            self.warp_data.qvel, self._tmp_qvel, self._indices_wp, self.task.nv])
        wp.copy(self.warp_data.qvel, self._tmp_qvel)

        wp.launch(gather_1d_float, dim=N, inputs=[
            self.warp_data.time, self._tmp_time, self._indices_wp])
        wp.copy(self.warp_data.time, self._tmp_time)

    def _regenerate_trailing_knots(self, stage: int) -> None:
        """knots[:, k_start:, :] ← clip(μ + σ · 𝒩); re-interpolate controls (ZOH)."""
        k_start = (stage + 1) * self.num_knots_per_stage
        remaining_K = self.num_knots - k_start
        if remaining_K <= 0 or self._stage_noise[stage] is None:
            return
        wp.launch(regenerate_knots, dim=self.num_samples, inputs=[
            self._knots_wp, self._mean_wp, self._stage_noise[stage],
            self._u_min_wp, self._u_max_wp, self.noise_level,
            k_start, remaining_K, self.task.nu])
        wp.launch(zero_order_interp, dim=self.num_samples, inputs=[
            self._knots_wp, self._controls_wp, self._knot_indices_wp,
            self.ctrl_steps, self.task.nu])

    # ══════════════════════════════════════════════════════════════════
    # Setup helpers
    # ══════════════════════════════════════════════════════════════════

    def _init_stage_geometry(self) -> None:
        """Pre-compute stage boundaries and the zero-order-hold knot index map."""
        self._num_stages = int(math.floor(self.num_knots / self.num_knots_per_stage))
        self._timesteps_per_stage = (
            int(math.floor(self.ctrl_steps / self.num_knots)) * self.num_knots_per_stage
        )

        tk_rel = np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
        tq_rel = np.linspace(0.0, self.plan_horizon, self.ctrl_steps, dtype=np.float32)
        ki = np.searchsorted(tk_rel, tq_rel, side="right") - 1
        self._knot_indices_np = np.clip(ki, 0, self.num_knots - 1).astype(np.int32)

    def _alloc_staging_buffers(self) -> None:
        """Pre-allocate KDE / reshuffle / per-stage noise buffers."""
        N, nu = self.num_samples, self.task.nu
        D = self._state_dim

        # State extraction + indices.
        self._states_wp = wp.zeros((N, D), dtype=wp.float32, device=self._device)
        self._indices_wp = wp.zeros(N, dtype=wp.int32, device=self._device)
        self._density_task_states_wp = None
        self._density_task_state_weight_wp = None
        if self._density_model_task_state_dim > 0:
            task_D = self._density_model_task_state_dim
            self._density_task_states_wp = wp.zeros(
                (N, task_D), dtype=wp.float32, device=self._device,
            )
            self._density_task_state_weight_wp = wp.zeros(
                task_D, dtype=wp.float32, device=self._device,
            )
            self._density_task_state_weight_wp.assign(
                np.ones(task_D, dtype=np.float32)
            )

        # Per-component state weighting (consumed by extract_state). Currently
        # always 1.0 — per-dimension scaling is the density model's responsibility.
        self._state_weight_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._state_weight_wp.assign(np.ones(D, dtype=np.float32))

        # Knot regen + control re-interp buffers.
        self._knots_wp = wp.zeros(
            (N, self.num_knots, nu), dtype=wp.float32, device=self._device,
        )
        self._mean_wp = wp.zeros(
            (self.num_knots, nu), dtype=wp.float32, device=self._device,
        )
        self._knot_indices_wp = wp.zeros(
            self.ctrl_steps, dtype=wp.int32, device=self._device,
        )
        self._knot_indices_wp.assign(self._knot_indices_np)

        self._u_min_wp = wp.zeros(nu, dtype=wp.float32, device=self._device)
        self._u_min_wp.assign(self.task.u_min.astype(np.float32))
        self._u_max_wp = wp.zeros(nu, dtype=wp.float32, device=self._device)
        self._u_max_wp.assign(self.task.u_max.astype(np.float32))

        # Reshuffle scratch (double-buffered for in-place gather+copy).
        self._tmp_controls = wp.zeros_like(self._controls_wp)
        self._tmp_knots = wp.zeros_like(self._knots_wp)
        self._tmp_costs = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._tmp_qpos = wp.zeros(
            (N, self.task.nq), dtype=wp.float32, device=self._device,
        )
        self._tmp_qvel = wp.zeros(
            (N, self.task.nv), dtype=wp.float32, device=self._device,
        )
        self._tmp_time = wp.zeros(N, dtype=wp.float32, device=self._device)

        # Per-stage trailing-knot noise buffers.  Stage n owns knots
        # [(n+1)*num_knots_per_stage : num_knots] — those NOT yet committed.
        self._stage_noise: list[Optional[wp.array]] = []
        for n in range(max(self._num_stages - 1, 0)):
            k_start = (n + 1) * self.num_knots_per_stage
            remaining = self.num_knots - k_start
            if remaining > 0:
                self._stage_noise.append(wp.zeros(
                    (N, remaining, nu), dtype=wp.float32, device=self._device,
                ))
            else:
                self._stage_noise.append(None)
