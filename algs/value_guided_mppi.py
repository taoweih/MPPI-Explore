"""Value-Guided MPPI: terminal cost replaced by a learned heuristic V_θ(s).

Algorithm:

    pretrain(state_sampler, target_function)            # offline, called once
        Fit V_θ on (state, target) pairs from the user-supplied callbacks
        (typically distance-to-goal as the target heuristic).

    optimize(mj_data)                                   # one MPC step
        1. sample_knots()
               K_i = clip(μ + σ · 𝒩(0, I),  u_min, u_max)
               u_i(t) = interp(K_i)
        2. rollout(controls, knots)
               x^i_{t+1} = f(x^i_t, u_i(t))                            mujoco_warp step
               φ(x^i_T)  = V_θ(s_T)  via value_model.launch_inference  (learned terminal)
               J_i       = Σ_t ℓ(x^i_t, u_i(t))·dt + φ(x^i_T)
        3. update_weights(costs, knots)
               w_i = exp(-J_i / λ) / Z;   μ ← Σ w_i · K_i
        4. update_learned_value(mj_data, best_cost)
               One online training step on (current_state, best_rollout_cost).
               One-sided MSE; hashgrid embeddings only (MLP frozen for HashGridValueModel).

Both inference and training are pluggable behind the abstract
`ValueModel` interface (`algs/value_models/base.py`):
    - `HashGridValueModel`        — multi-resolution hash grid + MLP
    - `RandomFourierValueModel`   — linear in random sin/cos features
See `value_density_guided_mppi.py` for the staged density + value composition.

CUDA-graph contract: the captured rollout graph reads weights from the
value_model's pre-allocated Warp arrays.  Training writes new weights into
those same arrays in-place, so the captured graph picks up updated values
on its next replay without rebuild.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from algs._graph import capture_graph
from algs.base import BaseMPPI
from algs.value_models import ValueModel


class ValueGuidedMPPI(BaseMPPI):
    """MPPI whose terminal cost is a learned V_θ(s)."""

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
        *,
        # Required: the learned value function.
        value_model: ValueModel,
        # Online learning hyperparameters (one captured-CUDA-graph step after each optimize).
        online_learning_rate: float = 1e-3,
        online_update_epochs: int = 1,
        online_batch_size: int = 1,
    ) -> None:
        if spline_type != "zero":
            raise NotImplementedError(
                "CUDA graph capture requires spline_type='zero'. "
                f"Got '{spline_type}'."
            )

        super().__init__(
            task=task, num_samples=num_samples, noise_level=noise_level,
            temperature=temperature, plan_horizon=plan_horizon,
            spline_type=spline_type, num_knots=num_knots,
            iterations=iterations, seed=seed,
        )

        self.value_model = value_model

        self.online_learning_rate = float(online_learning_rate)
        self.online_update_epochs = max(int(online_update_epochs), 1)
        self.online_batch_size = max(int(online_batch_size), 1)

        # Always-needed buffers for terminal-state extraction (V_θ inference).
        self._alloc_value_state_extract_buffer()
        self.value_model.alloc(self.num_samples, self._device)

        # CUDA graph (built lazily after kernels are compiled).
        self._rollout_graph: Optional[wp.Graph] = None

        # Snapshot of the post-pretrain weights (for reset_value_to_pretrained=True).
        self._pretrained_snapshot = None

    # ══════════════════════════════════════════════════════════════════
    # Algorithm
    # ══════════════════════════════════════════════════════════════════

    def pretrain(
        self,
        state_sampler: Callable[[np.random.Generator, int], np.ndarray],
        target_function: Callable[[np.ndarray], np.ndarray],
        *,
        sample_count: int = 100_000,
        epochs: int = 300,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        verbose: bool = False,
        print_every: int = 50,
    ) -> float:
        """Fit V_θ on (state, target) pairs from user-supplied callbacks.

        `state_sampler(rng, n) -> states (n, state_dim)`
        `target_function(states) -> targets (n,)`

        After fitting, snapshot the weights so `reset(reset_value_to_pretrained=True)`
        can restore them at the start of each benchmark trial.
        """
        states = np.asarray(
            state_sampler(self.rng, sample_count), dtype=np.float32,
        )
        targets = np.asarray(
            target_function(states), dtype=np.float32,
        ).reshape(-1)
        if verbose:
            print(
                f"Pretraining V_θ with {states.shape[0]} samples, "
                f"{epochs} epochs, batch_size={batch_size}."
            )
        last_loss = self.value_model.fit_pretrain(
            states, targets,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            verbose=verbose, print_every=print_every,
        )
        self._pretrained_snapshot = self.value_model.copy_weights()
        return last_loss

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """One value-guided MPPI optimization step.  Returns updated mean knots."""
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        init_state = (
            self._make_initial_state(mj_data) if self.iterations > 1 else None
        )
        best_cost = np.inf
        for it in range(self.iterations):
            if it > 0:
                self._restore_state(init_state)
            knots, controls = self.sample_knots()
            costs, final_knots = self.rollout(controls, knots)
            self.update_weights(costs, final_knots)
            best_cost = min(best_cost, float(np.min(costs)))

        self.update_learned_value(mj_data, best_cost)
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
        """V-guided rollout via captured CUDA graph.  Returns (costs, final_knots).

        Physics loop → V_θ(s_T) terminal.  final_knots = knots (unchanged).
        """
        self._controls_wp.assign(controls)

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
                self._emit_terminal_cost()
            self._rollout_graph = cap.graph

        wp.capture_launch(self._rollout_graph)

        costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
        return costs, knots

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

    def update_learned_value(self, mj_data: mujoco.MjData, best_cost: float) -> None:
        """Online update V_θ on (current_state, best_cost).  One-sided MSE.

        The captured rollout graph reads from the value_model's Warp arrays;
        `value_model.fit_online` updates those arrays in-place via the
        captured PyTorch CUDA graph + `_sync_to_warp`.  No graph rebuild needed.
        """
        state = self.task.extract_state_cpu(mj_data)
        self.value_model.fit_online(
            state[None, :], np.array([best_cost], dtype=np.float32),
            epochs=self.online_update_epochs,
            batch_size=self.online_batch_size,
            learning_rate=self.online_learning_rate,
        )

    # ══════════════════════════════════════════════════════════════════
    # State management (override to also reset the value model)
    # ══════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
        reset_value_to_pretrained: bool = True,
    ) -> None:
        super().reset(seed=seed, initial_knots=initial_knots)
        if reset_value_to_pretrained and self._pretrained_snapshot is not None:
            # Restore in-place; captured rollout graph reads from the same
            # Warp arrays so it picks up the reloaded weights at next replay.
            self.value_model.restore_weights(self._pretrained_snapshot)

    def load_pretrained_weights(self, path) -> None:
        """Load pretrained value weights from a file; snapshot for reset."""
        self.value_model.load_weights_from_file(path)
        self._pretrained_snapshot = self.value_model.copy_weights()

    def save_pretrained_weights(self, path) -> None:
        """Save current value weights to a file; snapshot for reset."""
        self.value_model.save_weights_to_file(path)
        self._pretrained_snapshot = self.value_model.copy_weights()

    # ══════════════════════════════════════════════════════════════════
    # Setup helpers
    # ══════════════════════════════════════════════════════════════════

    def _emit_terminal_cost(self) -> None:
        """Evaluate V_θ(s_T) into the terminal-cost buffer."""
        self.task.extract_state(
            self._task_state, self._value_states_wp, self._value_state_weight_wp,
        )
        self.value_model.launch_inference(
            self._value_states_wp, self._terminal_costs_wp,
        )

    def _alloc_value_state_extract_buffer(self) -> None:
        """Always-needed buffer for terminal-state extraction (V_θ inference)."""
        D = self.task.state_dim
        N = self.num_samples
        self._value_states_wp = wp.zeros((N, D), dtype=wp.float32, device=self._device)
        # Per-component state weighting (consumed by task.extract_state). Always
        # 1.0 — per-dim scaling is the model's responsibility (KDE bandwidth /
        # value-model normalisation).
        self._value_state_weight_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._value_state_weight_wp.assign(np.ones(D, dtype=np.float32))
