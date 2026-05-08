"""Model-Predictive Path Integral (MPPI) control via mujoco_warp.

Algorithm (one optimize step):

    1. warm-start the knot trajectory by shifting along time.
    2. sample noisy knots
           K_i = clip(μ + σ · 𝒩(0, I),  u_min, u_max),     i = 1..N
       and zero/linear/cubic-interpolate to a control sequence u_i(t).
    3. roll out all samples in parallel through mujoco_warp:
           x^i_{t+1} = f(x^i_t, u_i(t))
           J_i = Σ_t ℓ(x^i_t, u_i(t)) · dt + φ(x^i_T)
    4. softmax weight update
           w_i = exp(-J_i / λ) / Σ_j exp(-J_j / λ)
           μ ← Σ_i w_i · K_i

The rollout (physics step + cost accumulation) is captured once as a
CUDA graph and replayed each call.  The only per-optimize host-device
syncs are: one CPU→GPU upload of sampled controls, one GPU→CPU
readback of total costs (length N).
"""

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from typing import Literal, Optional

from algs._graph import capture_graph
from tasks.task_base import Task
from utils.spline import get_interp_func


class MPPI:
    """Base MPPI controller — sampling + softmax weighting + CUDA-graph rollout."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ) -> None:
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

        self._alloc_buffers()

        # CUDA graph is built lazily on first rollout call (kernels must be
        # compiled before capture).  run_interactive / run_benchmark warm
        # this up by calling optimize() twice before timing.
        self._rollout_graph: Optional[wp.Graph] = None

    # ------------------------------------------------------------------
    # GPU buffer allocation
    # ------------------------------------------------------------------

    def _alloc_buffers(self) -> None:
        """Pre-allocate every GPU buffer the captured rollout will read/write.

        Buffer addresses are fixed for the lifetime of the controller — the
        captured CUDA graph references these exact addresses, so we never
        reassign them (use `.assign()` for re-uploads).
        """
        self.warp_data = mjwarp.make_data(self.task.mj_model, nworld=self.num_samples)
        self._device = self.warp_data.ctrl.device

        # Build the task's State struct once; field references stay valid for
        # warp_data's lifetime, so the captured graph keeps seeing fresh values.
        self._task_state = self.task.make_state(self.warp_data)

        self._controls_wp = wp.zeros(
            (self.num_samples, self.ctrl_steps, self.task.nu),
            dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(
            self.num_samples, dtype=wp.float32, device=self._device,
        )
        self._terminal_costs_wp = wp.zeros(
            self.num_samples, dtype=wp.float32, device=self._device,
        )

    # ------------------------------------------------------------------
    # CUDA graph
    # ------------------------------------------------------------------

    def _build_rollout_graph(self) -> None:
        """Capture the full rollout (physics + per-step running cost + terminal) as a CUDA graph."""
        with capture_graph() as cap:
            self._running_costs_wp.zero_()
            self._terminal_costs_wp.zero_()
            for t in range(self.ctrl_steps):
                self._step_physics(t)
                self.task.launch_running_cost(
                    self._task_state, self.warp_data.ctrl,
                    self._running_costs_wp, self.dt,
                )
            self._launch_terminal_cost(self._terminal_costs_wp)
        self._rollout_graph = cap.graph

    def _step_physics(self, t: int) -> None:
        """Apply the t-th sampled control and step mujoco_warp once for every world."""
        wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
        mjwarp.step(self.task.model, self.warp_data)

    def _launch_terminal_cost(self, out_wp: wp.array) -> None:
        """Write terminal cost for all worlds into out_wp.

        Override in subclasses to swap in a learned heuristic.
        """
        self.task.launch_terminal_cost(self._task_state, out_wp)

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
                np.tile(mj_data.mocap_pos.astype(np.float32), (nw, 1, 1)).reshape(nw, -1, 3)
            )
            self.warp_data.mocap_quat.assign(
                np.tile(mj_data.mocap_quat.astype(np.float32), (nw, 1, 1)).reshape(nw, -1, 4)
            )

    def _make_initial_state(self, mj_data: mujoco.MjData) -> dict:
        """Build tiled initial state arrays from mj_data (CPU only, no GPU sync)."""
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
        """Sample noisy control knots and interpolate to a full sequence.

        Returns:
            knots:    (num_samples, num_knots, nu)  numpy
            controls: (num_samples, ctrl_steps, nu) numpy
        """
        # K_i = clip(μ + σ · 𝒩(0, I), u_min, u_max)
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        ).astype(np.float32)
        knots = self.mean + self.noise_level * noise
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, controls: np.ndarray) -> np.ndarray:
        """GPU-native rollout via CUDA graph.  Returns total costs (num_samples,)."""
        # Upload controls (CPU→GPU); graph reads from this buffer.
        self._controls_wp.assign(controls)

        # Build graph lazily after kernels are compiled (first optimize warm-up).
        if self._rollout_graph is None:
            self._build_rollout_graph()

        # Single GPU dispatch for the entire rollout.
        wp.capture_launch(self._rollout_graph)

        # Single GPU→CPU sync for the entire rollout.
        return self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def _update_weights(self, total_costs: np.ndarray, knots: np.ndarray) -> np.ndarray:
        """Softmax mean update:  w_i = exp(-J_i / λ) / Z;  μ ← Σ_i w_i · K_i."""
        shifted = -total_costs / self.temperature
        shifted -= shifted.max()  # for numerical stability of exp
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0)
        return total_costs

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Run one MPPI optimization step.  Returns updated mean knots."""
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        if self.iterations == 1:
            knots, controls = self._sample_knots()
            self._update_weights(self._rollout(controls), knots)
        else:
            # Save from mj_data directly — no GPU→CPU sync.
            init_state = self._make_initial_state(mj_data)
            for _ in range(self.iterations):
                self._restore_state(init_state)
                knots, controls = self._sample_knots()
                self._update_weights(self._rollout(controls), knots)

        return self.mean
