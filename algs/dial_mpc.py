"""DIAL-MPC: diffusion-inspired annealing for sampling-based MPC.

This implementation adapts the DIAL-MPC update used in
`tmp/dial-mpc/dial_mpc/core/dial_core.py` to this repository's
cost-minimizing MuJoCo/Warp controller API:

    1. warm-start the current mean knot trajectory
    2. run one or more diffusion updates
           K_i = clip(mu + eps * sigma_h * sigma_traj, u_min, u_max)
           optionally K_i[0] = mu[0] and include the current mean as a sample
    3. rollout sampled trajectories with the MPPI physics graph
    4. update the mean with a softmax over normalized negative costs

DIAL's reference code maximizes normalized rewards.  Here tasks expose costs,
so the same update is written as `softmax((J_ref - J_i) / (std(J) * temp))`.
The reference cost is a constant inside the softmax and is used only to match
the reward-form numerics.
"""

from __future__ import annotations

from typing import Literal, Optional

import mujoco
import numpy as np

from algs.mppi import MPPI


class DIALMPC(MPPI):
    """Diffusion-inspired annealed MPPI.

    DIAL-specific parameter mapping:
        - `sigma_scale` / `noise_level`: base control-node noise scale
        - `temp_sample` / `temperature`: normalized-cost softmax temperature
        - `num_diffusion_steps` / `iterations`: reverse diffusion updates
        - `num_initial_diffusion_steps`: first optimize call after reset
        - `horizon_diffuse_factor`: smaller noise near the start of the horizon
        - `traj_diffuse_factor`: anneal noise across diffusion updates

    The constructor accepts the existing MPPI names (`noise_level`,
    `temperature`, `iterations`) for compatibility, plus DIAL's names as
    keyword-only aliases.  If both names for the same value are supplied they
    must agree.
    """

    def __init__(
        self,
        task,
        num_samples: int,
        noise_level: Optional[float] = None,
        temperature: Optional[float] = None,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "quadratic", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: Optional[int] = None,
        seed: int = 0,
        *,
        sigma_scale: float = 1.0,
        temp_sample: float = 0.06,
        num_diffusion_steps: int = 2,
        num_initial_diffusion_steps: Optional[int] = 10,
        horizon_diffuse_factor: float = 0.9,
        traj_diffuse_factor: float = 0.5,
        include_mean_sample: bool = True,
        fix_first_knot: bool = True,
        normalize_costs: bool = True,
        cost_normalization_eps: float = 1e-6,
    ) -> None:
        sigma_scale = self._resolve_float_alias(
            "noise_level", noise_level, "sigma_scale", sigma_scale, default=1.0,
        )
        temp_sample = self._resolve_float_alias(
            "temperature", temperature, "temp_sample", temp_sample, default=0.06,
        )
        num_diffusion_steps = self._resolve_int_alias(
            "iterations", iterations,
            "num_diffusion_steps", num_diffusion_steps,
            default=2,
        )

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive; got {num_samples}.")
        if sigma_scale < 0.0:
            raise ValueError(f"sigma_scale must be non-negative; got {sigma_scale}.")
        if temp_sample <= 0.0:
            raise ValueError(f"temp_sample must be positive; got {temp_sample}.")
        if num_diffusion_steps < 1:
            raise ValueError(
                "num_diffusion_steps must be at least 1; "
                f"got {num_diffusion_steps}."
            )
        if num_initial_diffusion_steps is not None and num_initial_diffusion_steps < 1:
            raise ValueError(
                "num_initial_diffusion_steps must be at least 1 or None; "
                f"got {num_initial_diffusion_steps}."
            )
        if horizon_diffuse_factor <= 0.0:
            raise ValueError(
                "horizon_diffuse_factor must be positive; "
                f"got {horizon_diffuse_factor}."
            )
        if traj_diffuse_factor <= 0.0:
            raise ValueError(
                "traj_diffuse_factor must be positive; "
                f"got {traj_diffuse_factor}."
            )
        if cost_normalization_eps <= 0.0:
            raise ValueError(
                "cost_normalization_eps must be positive; "
                f"got {cost_normalization_eps}."
            )

        self.random_sample_count = int(num_samples)
        rollout_sample_count = (
            self.random_sample_count + 1
            if include_mean_sample
            else self.random_sample_count
        )

        super().__init__(
            task=task,
            num_samples=rollout_sample_count,
            noise_level=float(sigma_scale),
            temperature=float(temp_sample),
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=int(num_diffusion_steps),
            seed=seed,
        )

        # DIAL rolls out actions on linspace(0, Hsample * dt, Hsample + 1).
        self.ctrl_steps = int(round(self.plan_horizon / self.dt)) + 1
        self._tq_relative = np.linspace(
            0.0, self.plan_horizon, self.ctrl_steps, dtype=np.float32,
        )
        self._alloc_rollout_buffers()
        self._rollout_graph = None

        self.num_diffusion_steps = int(num_diffusion_steps)
        self.num_initial_diffusion_steps = (
            None
            if num_initial_diffusion_steps is None
            else int(num_initial_diffusion_steps)
        )
        self.horizon_diffuse_factor = float(horizon_diffuse_factor)
        self.traj_diffuse_factor = float(traj_diffuse_factor)
        self.include_mean_sample = bool(include_mean_sample)
        self.fix_first_knot = bool(fix_first_knot)
        self.normalize_costs = bool(normalize_costs)
        self.cost_normalization_eps = float(cost_normalization_eps)
        self.reset_after_warmup = True
        self.act_before_plan = True
        self._last_time: Optional[float] = None

        # DIAL samples later horizon nodes more aggressively than near-term
        # nodes.  The first knot receives the smallest scale and the final knot
        # receives sigma_scale.
        horizon_powers = np.arange(self.num_knots, dtype=np.float32)[::-1]
        self._horizon_noise_scale = (
            self.noise_level
            * (self.horizon_diffuse_factor ** horizon_powers)
        ).astype(np.float32)

        self._optimize_calls = 0

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Run DIAL-MPC reverse diffusion updates; returns updated mean knots."""
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)
        init_state = self._make_initial_state(mj_data)

        num_steps = self.num_diffusion_steps
        if self._optimize_calls == 0 and self.num_initial_diffusion_steps is not None:
            num_steps = self.num_initial_diffusion_steps

        for step_idx in range(num_steps):
            if step_idx > 0:
                self._restore_state(init_state)
            traj_scale = self.traj_diffuse_factor ** step_idx
            noise_scale = self._horizon_noise_scale * np.float32(traj_scale)
            knots, controls = self.sample_knots(noise_scale=noise_scale)
            costs = self.rollout(controls)
            self.update_weights(costs, knots)

        self._optimize_calls += 1
        return self.mean

    def warm_start(self, current_time: float) -> None:
        """Shift the control sequence the same way DIAL shifts its action buffer."""
        node_times = np.linspace(
            0.0, self.plan_horizon, self.num_knots, dtype=np.float32,
        )
        self.tk = node_times + np.float32(current_time)

        if self._last_time is None:
            self._last_time = float(current_time)
            return

        elapsed = max(float(current_time) - self._last_time, 0.0)
        shift_steps = int(round(elapsed / self.dt))
        self._last_time = float(current_time)
        if shift_steps <= 0:
            return

        action_times = self._tq_relative
        actions = self.interp_func(action_times, node_times, self.mean[None, ...])[0]
        if shift_steps >= self.ctrl_steps:
            actions[:] = 0.0
        else:
            actions = np.roll(actions, -shift_steps, axis=0)
            actions[-shift_steps:, :] = 0.0

        self.mean = self.interp_func(node_times, action_times, actions[None, ...])[0]
        self.mean = np.clip(self.mean, self.task.u_min, self.task.u_max).astype(
            np.float32
        )

    def sample_knots(
        self,
        noise_scale: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample annealed control knots and interpolate them to controls.

        Args:
            noise_scale:
                Per-knot scale with shape `(num_knots,)`.  When omitted, the
                base DIAL horizon scale is used.

        Returns:
            knots:    `(num_samples, num_knots, nu)`
            controls: `(num_samples, ctrl_steps, nu)`
        """
        if noise_scale is None:
            noise_scale = self._horizon_noise_scale
        else:
            noise_scale = np.asarray(noise_scale, dtype=np.float32)
            expected = (self.num_knots,)
            if noise_scale.shape != expected:
                raise ValueError(
                    f"noise_scale shape {noise_scale.shape} != expected {expected}"
                )

        noise = self.rng.standard_normal(
            (self.random_sample_count, self.num_knots, self.task.nu),
        ).astype(np.float32)
        sampled_knots = self.mean + noise * noise_scale[None, :, None]

        if self.fix_first_knot:
            sampled_knots[:, 0, :] = self.mean[0, :]

        if self.include_mean_sample:
            knots = np.concatenate(
                [sampled_knots, self.mean[None, :, :]], axis=0,
            )
        else:
            knots = sampled_knots

        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    def update_weights(
        self,
        total_costs: np.ndarray,
        knots: np.ndarray,
    ) -> np.ndarray:
        """DIAL softmax update over normalized negative rollout costs."""
        costs = np.asarray(total_costs, dtype=np.float32)

        if self.normalize_costs:
            cost_std = float(np.std(costs))
            norm = max(cost_std, self.cost_normalization_eps)
        else:
            norm = 1.0

        if self.include_mean_sample:
            reference_cost = float(costs[-1])
        else:
            reference_cost = float(np.mean(costs))

        shifted = (reference_cost - costs) / (norm * self.temperature)
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0).astype(np.float32)
        return total_costs

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
    ) -> None:
        super().reset(seed=seed, initial_knots=initial_knots)
        self._optimize_calls = 0
        self._last_time = None

    # ------------------------------------------------------------------
    # Alias validation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_float_alias(
        legacy_name: str,
        legacy_value: Optional[float],
        dial_name: str,
        dial_value: float,
        *,
        default: float,
    ) -> float:
        if legacy_value is None:
            return float(dial_value)
        legacy_value = float(legacy_value)
        dial_value = float(dial_value)
        if not np.isclose(dial_value, default) and not np.isclose(
            legacy_value, dial_value,
        ):
            raise ValueError(
                f"Conflicting {legacy_name}={legacy_value} and "
                f"{dial_name}={dial_value}; pass only one value."
            )
        return legacy_value

    @staticmethod
    def _resolve_int_alias(
        legacy_name: str,
        legacy_value: Optional[int],
        dial_name: str,
        dial_value: int,
        *,
        default: int,
    ) -> int:
        if legacy_value is None:
            return int(dial_value)
        legacy_value = int(legacy_value)
        dial_value = int(dial_value)
        if dial_value != default and legacy_value != dial_value:
            raise ValueError(
                f"Conflicting {legacy_name}={legacy_value} and "
                f"{dial_name}={dial_value}; pass only one value."
            )
        return legacy_value
