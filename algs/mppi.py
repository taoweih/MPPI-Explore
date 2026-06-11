"""Model-Predictive Path Integral (MPPI) control via mujoco_warp.

Algorithm (one optimize step):

    1. sample_knots()
           K_i = clip(μ + σ · 𝒩(0, I),  u_min, u_max),     i = 1..N
       and zero/linear/cubic-interpolate to a control sequence u_i(t).
    2. rollout(controls)
           x^i_{t+1} = f(x^i_t, u_i(t))                     # mujoco_warp step
           J_i = Σ_t ℓ(x^i_t, u_i(t)) · dt + φ(x^i_T)       # task running + terminal
    3. update_weights(costs, knots)
           w_i = exp(-J_i / λ) / Σ_j exp(-J_j / λ)
           μ ← Σ_i w_i · K_i

The rollout (physics step + cost accumulation) is captured once as a
CUDA graph and replayed each call.  The only per-optimize host-device
syncs are: one CPU→GPU upload of sampled controls, one GPU→CPU readback
of total costs (length N).
"""

from typing import Optional

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjwarp

from algs._graph import capture_graph
from algs.base import BaseMPPI


class MPPI(BaseMPPI):
    """Standard MPPI: sample → parallel rollout → softmax weight update."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # CUDA graph is built lazily on first rollout (kernels must be
        # compiled before capture).  run_interactive / run_benchmark warm
        # this up by calling optimize() twice before timing.
        self._rollout_graph: Optional[wp.Graph] = None

    # ══════════════════════════════════════════════════════════════════
    # Algorithm
    # ══════════════════════════════════════════════════════════════════

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """One MPPI optimization step.  Returns updated mean knots."""
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        init_state = (
            self._make_initial_state(mj_data) if self.iterations > 1 else None
        )
        for it in range(self.iterations):
            if it > 0:
                self._restore_state(init_state)
            knots, controls = self.sample_knots()
            costs = self.rollout(controls)
            self.update_weights(costs, knots)

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

    def rollout(self, controls: np.ndarray) -> np.ndarray:
        """Replay the captured rollout graph; return total costs (num_samples,).

            J_i = Σ_t ℓ(x^i_t, u_i(t)) · dt  +  φ(x^i_T)
        """
        # Upload controls (CPU→GPU); graph reads from this fixed-address buffer.
        self._controls_wp.assign(controls)

        # Build graph lazily after kernels are compiled (first optimize warm-up).
        if self._rollout_graph is None:
            with capture_graph() as cap:
                self._running_costs_wp.zero_()
                self._terminal_costs_wp.zero_()
                for t in range(self.ctrl_steps):
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
                self.task.launch_terminal_cost(
                    self._task_state, self._terminal_costs_wp,
                )
            self._rollout_graph = cap.graph

        # Single GPU dispatch for the entire rollout.
        wp.capture_launch(self._rollout_graph)

        # Single GPU→CPU sync for the entire rollout.
        return self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()

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
