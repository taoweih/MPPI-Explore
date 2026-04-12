"""MPPI with staged rollout and KDE-based resampling (fully GPU graph-captured)."""

import math

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from typing import Literal, Optional

from tasks.task_base import Task
from utils.spline import get_interp_func
from utils.warp_kernels import (
    extract_float_slice,
    extract_vec3_row,
    kde_density,
    resample_from_density,
    gather_1d_float,
    gather_2d_float,
    gather_3d_float,
    regenerate_knots,
    zero_order_interp,
)


class MPPIStagedRollout:
    """MPPI with staged rollout and KDE resampling.

    Splits the planning horizon into stages. After each stage, computes a KDE
    over reached states and resamples trajectories inversely proportional to
    density to encourage state-space exploration.

    The entire staged rollout — including KDE density computation,
    systematic resampling, state reshuffling, knot regeneration, and
    zero-order interpolation — is captured as a single CUDA graph.
    The only GPU-CPU syncs are the initial data upload (before graph
    launch) and the final cost readback.
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
        state_dim: int = 2,
        state_source_field: str = "qpos",
        state_source_start: int = 0,
        state_source_body_id: Optional[int] = None,
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
        self.state_dim = state_dim
        self._state_source_field = state_source_field
        self._state_source_start = state_source_start
        self._state_source_body_id = state_source_body_id

        # Stage geometry (fixed).
        self._num_stages = int(math.floor(num_knots / num_knots_per_stage))
        self._timesteps_per_stage = (
            int(math.floor(self.ctrl_steps / num_knots)) * num_knots_per_stage
        )

        # Pre-compute zero-order-hold knot index mapping (fixed for all time).
        tk_rel = np.linspace(0.0, plan_horizon, num_knots, dtype=np.float32)
        tq_rel = np.linspace(0.0, plan_horizon, self.ctrl_steps, dtype=np.float32)
        ki = np.searchsorted(tk_rel, tq_rel, side="right") - 1
        self._knot_indices_np = np.clip(ki, 0, num_knots - 1).astype(np.int32)

        # ── GPU arrays ────────────────────────────────────────────────
        self.warp_data = mjwarp.make_data(task.mj_model, nworld=num_samples)
        self._device = self.warp_data.ctrl.device

        self._controls_wp = wp.zeros(
            (num_samples, self.ctrl_steps, task.nu),
            dtype=wp.float32, device=self._device,
        )
        self._knots_wp = wp.zeros(
            (num_samples, num_knots, task.nu),
            dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._terminal_costs_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._mean_wp = wp.zeros(
            (num_knots, task.nu), dtype=wp.float32, device=self._device,
        )
        self._knot_indices_wp = wp.zeros(
            self.ctrl_steps, dtype=wp.int32, device=self._device,
        )
        self._knot_indices_wp.assign(self._knot_indices_np)

        # State extraction / KDE buffers.
        self._states_wp = wp.zeros(
            (num_samples, state_dim), dtype=wp.float32, device=self._device,
        )
        self._density_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._indices_wp = wp.zeros(
            num_samples, dtype=wp.int32, device=self._device,
        )

        # Expand state_weight to (state_dim,).
        if state_weight is not None:
            sw = np.asarray(state_weight, dtype=np.float32).ravel()
        else:
            sw = np.ones(state_dim, dtype=np.float32)
        if sw.shape[0] == 1 and state_dim > 1:
            sw = np.full(state_dim, sw[0], dtype=np.float32)
        self._state_weight_wp = wp.zeros(
            state_dim, dtype=wp.float32, device=self._device,
        )
        self._state_weight_wp.assign(sw)

        bw = np.full(state_dim, kde_bandwidth, dtype=np.float32)
        self._bandwidth_wp = wp.zeros(
            state_dim, dtype=wp.float32, device=self._device,
        )
        self._bandwidth_wp.assign(bw)

        self._u_min_wp = wp.zeros(task.nu, dtype=wp.float32, device=self._device)
        self._u_min_wp.assign(task.u_min.astype(np.float32))
        self._u_max_wp = wp.zeros(task.nu, dtype=wp.float32, device=self._device)
        self._u_max_wp.assign(task.u_max.astype(np.float32))

        # Gather temp buffers (double-buffering for in-place reshuffle).
        self._tmp_controls = wp.zeros_like(self._controls_wp)
        self._tmp_knots = wp.zeros_like(self._knots_wp)
        self._tmp_costs = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._tmp_qpos = wp.zeros(
            (num_samples, task.nq), dtype=wp.float32, device=self._device,
        )
        self._tmp_qvel = wp.zeros(
            (num_samples, task.nv), dtype=wp.float32, device=self._device,
        )
        self._tmp_time = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )

        # Per-stage-boundary noise buffers and resample offsets.
        self._stage_noise: list[Optional[wp.array]] = []
        for n in range(max(self._num_stages - 1, 0)):
            k_start = (n + 1) * num_knots_per_stage
            remaining = num_knots - k_start
            if remaining > 0:
                self._stage_noise.append(
                    wp.zeros(
                        (num_samples, remaining, task.nu),
                        dtype=wp.float32, device=self._device,
                    )
                )
            else:
                self._stage_noise.append(None)

        n_boundaries = max(self._num_stages - 1, 1)
        self._resample_offsets_wp = wp.zeros(
            n_boundaries, dtype=wp.float32, device=self._device,
        )

        # CUDA graphs (built lazily).
        self._rollout_graph: Optional[wp.Graph] = None
        self._staged_graph: Optional[wp.Graph] = None

    # ------------------------------------------------------------------
    # CUDA graph builders
    # ------------------------------------------------------------------

    def _build_rollout_graph(self) -> None:
        """Non-staged fallback: step + cost in one graph."""
        wp.capture_begin(force_module_load=False)
        try:
            self._running_costs_wp.zero_()
            self._terminal_costs_wp.zero_()
            for t in range(self.ctrl_steps):
                wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                mjwarp.step(self.task.model, self.warp_data)
                self.task.launch_running_cost(
                    self.warp_data, self._running_costs_wp, self.dt,
                )
            self.task.launch_terminal_cost(self.warp_data, self._terminal_costs_wp)
            self._rollout_graph = wp.capture_end()
        except Exception:
            wp.capture_end()
            raise

    def _launch_extract_state(self) -> None:
        """Emit GPU kernel(s) to extract weighted state into _states_wp."""
        src = getattr(self.warp_data, self._state_source_field)
        if self._state_source_body_id is not None:
            wp.launch(
                extract_vec3_row, dim=self.num_samples,
                inputs=[src, self._states_wp, self._state_weight_wp,
                        self._state_source_body_id],
            )
        else:
            wp.launch(
                extract_float_slice, dim=self.num_samples,
                inputs=[src, self._states_wp, self._state_weight_wp,
                        self._state_source_start, self.state_dim],
            )

    def _build_staged_graph(self) -> None:
        """Capture the full staged rollout as a single CUDA graph."""
        N = self.num_samples
        nu = self.task.nu

        wp.capture_begin(force_module_load=False)
        try:
            self._running_costs_wp.zero_()

            for n in range(self._num_stages - 1):
                t_start = n * self._timesteps_per_stage
                t_end = (n + 1) * self._timesteps_per_stage

                # ── Physics stage ──
                for t in range(t_start, t_end):
                    wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                    mjwarp.step(self.task.model, self.warp_data)
                    self.task.launch_running_cost(
                        self.warp_data, self._running_costs_wp, self.dt,
                    )

                # ── KDE density ──
                self._launch_extract_state()
                wp.launch(
                    kde_density, dim=N,
                    inputs=[self._states_wp, self._density_wp,
                            self._bandwidth_wp, N, self.state_dim],
                )

                # ── Systematic resampling ──
                wp.launch(
                    resample_from_density, dim=1,
                    inputs=[self._density_wp, self._indices_wp,
                            self._resample_offsets_wp, n, N,
                            self.inverse_density_power],
                )

                # ── Gather (reshuffle by resampled indices) ──
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
                    self.warp_data.qpos, self._tmp_qpos, self._indices_wp,
                    self.task.nq])
                wp.copy(self.warp_data.qpos, self._tmp_qpos)

                wp.launch(gather_2d_float, dim=N, inputs=[
                    self.warp_data.qvel, self._tmp_qvel, self._indices_wp,
                    self.task.nv])
                wp.copy(self.warp_data.qvel, self._tmp_qvel)

                wp.launch(gather_1d_float, dim=N, inputs=[
                    self.warp_data.time, self._tmp_time, self._indices_wp])
                wp.copy(self.warp_data.time, self._tmp_time)

                # ── Regenerate remaining knots & re-interpolate ──
                k_start = (n + 1) * self.num_knots_per_stage
                remaining_K = self.num_knots - k_start
                if remaining_K > 0 and self._stage_noise[n] is not None:
                    wp.launch(
                        regenerate_knots, dim=N,
                        inputs=[self._knots_wp, self._mean_wp,
                                self._stage_noise[n], self._u_min_wp,
                                self._u_max_wp, self.noise_level,
                                k_start, remaining_K, nu],
                    )
                    wp.launch(
                        zero_order_interp, dim=N,
                        inputs=[self._knots_wp, self._controls_wp,
                                self._knot_indices_wp, self.ctrl_steps, nu],
                    )

            # ── Final stage ──
            t_start = (self._num_stages - 1) * self._timesteps_per_stage
            for t in range(t_start, self.ctrl_steps):
                wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                mjwarp.step(self.task.model, self.warp_data)
                self.task.launch_running_cost(
                    self.warp_data, self._running_costs_wp, self.dt,
                )

            # ── Terminal cost ──
            self._terminal_costs_wp.zero_()
            self.task.launch_terminal_cost(self.warp_data, self._terminal_costs_wp)

            self._staged_graph = wp.capture_end()
        except Exception:
            wp.capture_end()
            raise

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
        """Non-staged GPU rollout via CUDA graph."""
        self._controls_wp.assign(controls)
        if self._rollout_graph is None:
            self._build_rollout_graph()
        wp.capture_launch(self._rollout_graph)
        return self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()

    def _staged_rollout(
        self, controls: np.ndarray, knots: np.ndarray,
    ) -> np.ndarray:
        """Staged rollout via CUDA graph with GPU-native KDE resampling."""
        if self._num_stages <= 1:
            return self._rollout(controls)

        # Upload data to pre-allocated GPU buffers.
        self._controls_wp.assign(controls)
        self._knots_wp.assign(knots)
        self._mean_wp.assign(self.mean)

        # Pre-generate stochastic inputs and upload.
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

        # Build graph lazily, then replay.
        if self._staged_graph is None:
            self._build_staged_graph()
        wp.capture_launch(self._staged_graph)

        costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
        final_knots = self._knots_wp.numpy()
        return costs, final_knots

    # ------------------------------------------------------------------
    # Weight update & optimize
    # ------------------------------------------------------------------

    def _update_weights(
        self, total_costs: np.ndarray, knots: np.ndarray,
    ) -> np.ndarray:
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
            costs, final_knots = self._staged_rollout(controls, knots)
            self._update_weights(costs, final_knots)
        else:
            init_state = self._make_initial_state(mj_data)
            for _ in range(self.iterations):
                self._restore_state(init_state)
                knots, controls = self._sample_knots()
                costs, final_knots = self._staged_rollout(controls, knots)
                self._update_weights(costs, final_knots)

        return self.mean
