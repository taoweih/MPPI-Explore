"""Density-Guided MPPI: KDE-based stage resampling for state-space exploration.

Algorithm: split the planning horizon into S stages (each `num_knots_per_stage`
knots wide).  After every stage boundary, resample trajectories with weights
inversely proportional to the local state density, encouraging exploration
of low-density regions.

For each boundary stage n = 0 .. S-2:

    1. step physics for the stage's timesteps (accumulate running cost)
    2. extract low-D state s_i for every world (via task.extract_state)
    3. KDE density:
           ρ(s_i) = 1/N · Σ_j  K_h(s_i − s_j),     K_h Gaussian
    4. systematic resample with weights w_i ∝ 1 / (ρ(s_i) + ε)^α
       — α=1.0 full inverse-density,  α=0 uniform,  0<α<1 softer
    5. reshuffle (controls, knots, costs, qpos, qvel, time) by indices
    6. regenerate trailing knots
           K[k_start:, :] = clip(μ + σ · 𝒩(0, I))
       and re-interpolate controls (zero-order hold).

Final stage: physics only, then terminal cost.

The whole staged rollout is captured as a single CUDA graph.  Only host
syncs per optimize: the stochastic uploads (resample offsets, per-stage
noise, controls/knots/mean) and the cost/knots readback.

Requires `spline_type="zero"` because the in-graph re-interpolation step
is zero-order-hold.
"""

import math

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from typing import Literal, Optional

from algs._graph import capture_graph
from tasks.task_base import Task
from utils.spline import get_interp_func
from utils.warp_kernels import (
    gather_1d_float,
    gather_2d_float,
    gather_3d_float,
)


# ──────────────────────────────────────────────────────────────────────
# Density-guided algorithm kernels
# ──────────────────────────────────────────────────────────────────────


@wp.kernel
def kde_density(
    states:  wp.array2d(dtype=wp.float32),  # (N, D)
    density: wp.array1d(dtype=wp.float32),  # (N,)
    bw:      wp.array1d(dtype=wp.float32),  # (D,)
    N:       int,
    D:       int,
):
    """ρ(s_i) = 1/N · Σ_j exp(−½ · ‖(s_i − s_j) / bw‖²).  O(N) per thread."""
    i = wp.tid()
    total = float(0.0)
    for j in range(N):
        dist_sq = float(0.0)
        for d in range(D):
            diff = (states[i, d] - states[j, d]) / bw[d]
            dist_sq = dist_sq + diff * diff
        total = total + wp.exp(-0.5 * dist_sq)
    density[i] = total / float(N)


@wp.kernel
def resample_from_density(
    density:   wp.array1d(dtype=wp.float32),  # (N,)
    indices:   wp.array1d(dtype=wp.int32),    # (N,) output
    offsets:   wp.array1d(dtype=wp.float32),  # (n_boundaries,) U[0, 1/N) per stage
    stage_idx: int,
    N:         int,
    alpha:     float,
):
    """Systematic resample with weights w_i ∝ 1 / (ρ_i + ε)^α.

    α = 1.0 : full inverse-density;  α = 0 : uniform;  0 < α < 1 softer.
    Single-threaded (dim=1) — sequential O(N) which is negligible for N=256–4096.
    """
    tid = wp.tid()
    if tid > 0:
        return

    eps = float(1.0e-6)

    # Sum of inverse densities (normalisation Z).
    inv_total = float(0.0)
    for j in range(N):
        inv_total = inv_total + wp.pow(1.0 / (density[j] + eps), alpha)

    # Two-pointer systematic resample.
    u = offsets[stage_idx]
    step = 1.0 / float(N)
    cumulative = float(0.0)
    j = int(0)

    for i in range(N):
        threshold = u + float(i) * step
        can_advance = int(1)
        while can_advance == 1:
            if j >= N - 1:
                can_advance = 0
            else:
                w_j = wp.pow(1.0 / (density[j] + eps), alpha) / inv_total
                if cumulative + w_j < threshold:
                    cumulative = cumulative + w_j
                    j = j + 1
                else:
                    can_advance = 0
        indices[i] = j


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


class DensityGuidedMPPI:
    """MPPI with KDE-based stage resampling for exploration.

    The entire staged rollout — physics + KDE + resample + reshuffle +
    knot regeneration + re-interpolation — is captured as a single
    CUDA graph.  Only host-device syncs per optimize: data uploads
    (controls/knots/mean/offsets/noise) and cost/knot readback.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: Optional[np.ndarray] = None,
        inverse_density_power: float = 1.0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ) -> None:
        if spline_type != "zero":
            raise NotImplementedError(
                "CUDA graph capture for staged rollout requires spline_type='zero'. "
                f"Got '{spline_type}'."
            )

        self.task = task
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.temperature = temperature
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.interp_func = get_interp_func(spline_type)
        self.dt = task.dt
        self.ctrl_steps = int(round(plan_horizon / self.dt))

        self.tk = np.linspace(0.0, plan_horizon, num_knots, dtype=np.float32)
        self._tq_relative = np.linspace(
            0.0, plan_horizon, self.ctrl_steps, dtype=np.float32
        )
        self.mean = np.zeros((num_knots, task.nu), dtype=np.float32)

        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth
        self.inverse_density_power = inverse_density_power

        self._init_stage_geometry()
        self._alloc_rollout_buffers()
        self._alloc_density_buffers(state_weight)

        # CUDA graphs (built lazily after kernels are compiled).
        self._rollout_graph: Optional[wp.Graph] = None
        self._density_graph: Optional[wp.Graph] = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

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

    def _alloc_rollout_buffers(self) -> None:
        """Pre-allocate buffers used by the (non-staged) rollout graph."""
        self.warp_data = mjwarp.make_data(self.task.mj_model, nworld=self.num_samples)
        self._device = self.warp_data.ctrl.device
        N, nu = self.num_samples, self.task.nu

        # Build the task's State struct once; field references stay valid for
        # warp_data's lifetime, so the captured graph keeps seeing fresh values.
        self._task_state = self.task.make_state(self.warp_data)

        self._controls_wp = wp.zeros(
            (N, self.ctrl_steps, nu), dtype=wp.float32, device=self._device,
        )
        self._knots_wp = wp.zeros(
            (N, self.num_knots, nu), dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
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

    def _alloc_density_buffers(self, state_weight: Optional[np.ndarray]) -> None:
        """Pre-allocate KDE / resample / reshuffle / per-stage noise buffers."""
        N, nu = self.num_samples, self.task.nu
        D = self.task.state_dim

        # State extraction + KDE.
        self._states_wp = wp.zeros((N, D), dtype=wp.float32, device=self._device)
        self._density_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._indices_wp = wp.zeros(N, dtype=wp.int32, device=self._device)

        # KDE bandwidth (broadcast scalar to per-dim).
        bw = np.full(D, self.kde_bandwidth, dtype=np.float32)
        self._bandwidth_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._bandwidth_wp.assign(bw)

        # Per-component state weighting (used by task.extract_state).
        if state_weight is not None:
            sw = np.asarray(state_weight, dtype=np.float32).ravel()
        else:
            sw = np.ones(D, dtype=np.float32)
        if sw.shape[0] == 1 and D > 1:
            sw = np.full(D, sw[0], dtype=np.float32)
        self._state_weight_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._state_weight_wp.assign(sw)

        # Reshuffle scratch buffers (double-buffered for in-place gather+copy).
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

        # Per-stage-boundary trailing-knot noise buffers.
        # Stage n owns knots [(n+1)*num_knots_per_stage : num_knots], i.e. those
        # NOT yet committed at the boundary.  None entries mean nothing remains.
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

        n_boundaries = max(self._num_stages - 1, 1)
        self._resample_offsets_wp = wp.zeros(
            n_boundaries, dtype=wp.float32, device=self._device,
        )

    # ------------------------------------------------------------------
    # CUDA graph builders — these read as the algorithm
    # ------------------------------------------------------------------

    def _build_rollout_graph(self) -> None:
        """Non-density fallback (used when num_stages ≤ 1)."""
        with capture_graph() as cap:
            self._running_costs_wp.zero_()
            self._terminal_costs_wp.zero_()
            self._step_physics_range(0, self.ctrl_steps)
            self.task.launch_terminal_cost(self._task_state, self._terminal_costs_wp)
        self._rollout_graph = cap.graph

    def _build_density_graph(self) -> None:
        """Capture the full density-guided staged rollout."""
        with capture_graph() as cap:
            self._running_costs_wp.zero_()

            for n in range(self._num_stages - 1):
                self._step_physics_range(
                    n * self._timesteps_per_stage,
                    (n + 1) * self._timesteps_per_stage,
                )
                self._compute_kde_density()
                self._resample_inverse_density(stage_idx=n)
                self._reshuffle_by_indices()
                self._regenerate_trailing_knots(stage=n)

            # Final stage: physics only (no resampling at the very end).
            self._step_physics_range(
                (self._num_stages - 1) * self._timesteps_per_stage,
                self.ctrl_steps,
            )

            self._terminal_costs_wp.zero_()
            self.task.launch_terminal_cost(self._task_state, self._terminal_costs_wp)
        self._density_graph = cap.graph

    # ------------------------------------------------------------------
    # Launch helpers — one wp.launch per stage operation
    # ------------------------------------------------------------------

    def _step_physics_range(self, t_start: int, t_end: int) -> None:
        """For each t in [t_start, t_end): copy ctrl, mjwarp.step, accumulate running cost."""
        for t in range(t_start, t_end):
            wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
            mjwarp.step(self.task.model, self.warp_data)
            self.task.launch_running_cost(
                self._task_state, self.warp_data.ctrl,
                self._running_costs_wp, self.dt,
            )

    def _compute_kde_density(self) -> None:
        """Extract weighted state and compute Gaussian KDE density per sample."""
        self.task.extract_state(self._task_state, self._states_wp, self._state_weight_wp)
        wp.launch(
            kde_density, dim=self.num_samples,
            inputs=[self._states_wp, self._density_wp,
                    self._bandwidth_wp, self.num_samples, self.task.state_dim],
        )

    def _resample_inverse_density(self, stage_idx: int) -> None:
        """Systematic resample → fills self._indices_wp.  Single-threaded kernel."""
        wp.launch(
            resample_from_density, dim=1,
            inputs=[self._density_wp, self._indices_wp,
                    self._resample_offsets_wp, stage_idx,
                    self.num_samples, self.inverse_density_power],
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
        """knots[:, k_start:, :] ← clip(μ + σ · 𝒩) and re-interpolate controls."""
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

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
    ) -> None:
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.tk = np.linspace(
            0.0, self.plan_horizon, self.num_knots, dtype=np.float32
        )
        if initial_knots is None:
            self.mean = np.zeros((self.num_knots, self.task.nu), dtype=np.float32)
        else:
            knots = np.asarray(initial_knots, dtype=np.float32)
            expected = (self.num_knots, self.task.nu)
            if knots.shape != expected:
                raise ValueError(
                    f"initial_knots shape {knots.shape} != expected {expected}"
                )
            self.mean = knots.copy()

    def set_state_from_mj_data(self, mj_data: mujoco.MjData) -> None:
        nw = self.num_samples
        self.warp_data.qpos.assign(
            np.tile(mj_data.qpos.astype(np.float32), (nw, 1))
        )
        self.warp_data.qvel.assign(
            np.tile(mj_data.qvel.astype(np.float32), (nw, 1))
        )
        self.warp_data.time.assign(
            np.full(nw, mj_data.time, dtype=np.float32)
        )
        if mj_data.mocap_pos.shape[0] > 0:
            self.warp_data.mocap_pos.assign(
                np.tile(mj_data.mocap_pos.astype(np.float32), (nw, 1, 1))
                .reshape(nw, -1, 3)
            )
            self.warp_data.mocap_quat.assign(
                np.tile(mj_data.mocap_quat.astype(np.float32), (nw, 1, 1))
                .reshape(nw, -1, 4)
            )

    def _make_initial_state(self, mj_data: mujoco.MjData) -> dict:
        nw = self.num_samples
        return {
            "qpos": np.tile(mj_data.qpos.astype(np.float32), (nw, 1)),
            "qvel": np.tile(mj_data.qvel.astype(np.float32), (nw, 1)),
            "time": np.full(nw, mj_data.time, dtype=np.float32),
        }

    def _restore_state(self, state: dict) -> None:
        self.warp_data.qpos.assign(state["qpos"])
        self.warp_data.qvel.assign(state["qvel"])
        self.warp_data.time.assign(state["time"])

    # ------------------------------------------------------------------
    # Warm start & action query
    # ------------------------------------------------------------------

    def warm_start(self, current_time: float) -> None:
        new_tk = (
            np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
            + current_time
        )
        new_mean = self.interp_func(new_tk, self.tk, self.mean[None, ...])[0]
        self.tk = new_tk
        self.mean = new_mean

    def get_action(self, t: float) -> np.ndarray:
        return self.interp_func(
            np.array([t], dtype=np.float32),
            self.tk,
            self.mean[None, ...],
        )[0, 0, :]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_knots(self):
        """K_i = clip(μ + σ · 𝒩(0, I)),  controls = ZOH-interpolate(K_i)."""
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        ).astype(np.float32)
        knots = self.mean + self.noise_level * noise
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def _rollout(self, controls: np.ndarray) -> np.ndarray:
        """Non-density GPU rollout via CUDA graph."""
        self._controls_wp.assign(controls)
        if self._rollout_graph is None:
            self._build_rollout_graph()
        wp.capture_launch(self._rollout_graph)
        return self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()

    def _density_rollout(
        self, controls: np.ndarray, knots: np.ndarray,
    ):
        """Density-guided GPU rollout via CUDA graph.  Returns (costs, final_knots)."""
        if self._num_stages <= 1:
            return self._rollout(controls), knots

        # Upload inputs to pre-allocated GPU buffers (same addresses as captured).
        self._controls_wp.assign(controls)
        self._knots_wp.assign(knots)
        self._mean_wp.assign(self.mean)

        # Stochastic inputs read inside the captured graph.
        n_boundaries = self._num_stages - 1
        offsets = self.rng.uniform(
            0.0, 1.0 / self.num_samples, size=n_boundaries,
        ).astype(np.float32)
        self._resample_offsets_wp.assign(offsets)

        for n in range(n_boundaries):
            buf = self._stage_noise[n]
            if buf is not None:
                k_start = (n + 1) * self.num_knots_per_stage
                remaining = self.num_knots - k_start
                noise = self.rng.standard_normal(
                    (self.num_samples, remaining, self.task.nu),
                ).astype(np.float32)
                buf.assign(noise)

        if self._density_graph is None:
            self._build_density_graph()
        wp.capture_launch(self._density_graph)

        costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
        final_knots = self._knots_wp.numpy()
        return costs, final_knots

    # ------------------------------------------------------------------
    # Weight update & optimize
    # ------------------------------------------------------------------

    def _update_weights(
        self, total_costs: np.ndarray, knots: np.ndarray,
    ) -> np.ndarray:
        """Softmax mean update:  w_i = exp(-J_i / λ) / Z;  μ ← Σ_i w_i · K_i."""
        shifted = -total_costs / self.temperature
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0)
        return total_costs

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        if self.iterations == 1:
            knots, controls = self._sample_knots()
            costs, final_knots = self._density_rollout(controls, knots)
            self._update_weights(costs, final_knots)
        else:
            init_state = self._make_initial_state(mj_data)
            for _ in range(self.iterations):
                self._restore_state(init_state)
                knots, controls = self._sample_knots()
                costs, final_knots = self._density_rollout(controls, knots)
                self._update_weights(costs, final_knots)

        return self.mean
