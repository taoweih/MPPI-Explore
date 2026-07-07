"""DIAL-MPC: diffusion-inspired annealing for sampling-based MPC.

This implementation follows Hydrax's DIAL controller while preserving this
repository's stateful MuJoCo/Warp controller API and existing constructor names:

    1. warm-start the current mean knot trajectory
    2. run one or more diffusion updates
           K_i = clip(mu + eps * sigma(iteration, horizon), u_min, u_max)
    3. rollout sampled trajectories with the MPPI physics graph
    4. update the mean with a softmax over negative costs

Hydrax names the two annealing constants `beta_opt_iter` and `beta_horizon`.
For compatibility, MPPI-Explore keeps the existing keyword names
`traj_diffuse_factor` and `horizon_diffuse_factor` for those two values.
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
        - `temp_sample` / `temperature`: softmax temperature
        - `num_diffusion_steps` / `iterations`: reverse diffusion updates
        - `num_initial_diffusion_steps`: accepted for compatibility; Hydrax DIAL
          uses the regular iteration count for every optimize call
        - `horizon_diffuse_factor`: Hydrax beta for horizon-level annealing
        - `traj_diffuse_factor`: Hydrax beta for optimization-iteration annealing

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

        super().__init__(
            task=task,
            num_samples=int(num_samples),
            noise_level=float(sigma_scale),
            temperature=float(temp_sample),
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=int(num_diffusion_steps),
            seed=seed,
        )

        self.num_diffusion_steps = int(num_diffusion_steps)
        self.num_initial_diffusion_steps = (
            None
            if num_initial_diffusion_steps is None
            else int(num_initial_diffusion_steps)
        )
        self.horizon_diffuse_factor = float(horizon_diffuse_factor)
        self.traj_diffuse_factor = float(traj_diffuse_factor)
        # Accepted as no-op compatibility flags. Hydrax DIAL does not include
        # a mean sample, fix the first knot, or normalize costs before softmax.
        self.include_mean_sample = bool(include_mean_sample)
        self.fix_first_knot = bool(fix_first_knot)
        self.normalize_costs = bool(normalize_costs)
        self.cost_normalization_eps = float(cost_normalization_eps)
        self.reset_after_warmup = True
        self.act_before_plan = False
        self._opt_iteration = 0

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Run Hydrax-style DIAL updates; returns updated mean knots."""
        return super().optimize(mj_data)

    def warm_start(self, current_time: float) -> None:
        """Use the shared Hydrax-style clipped spline warm start."""
        super().warm_start(current_time)

    def sample_knots(
        self,
        noise_scale: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample Hydrax DIAL knots and interpolate them to controls."""
        if noise_scale is not None:
            noise_scale = np.asarray(noise_scale, dtype=np.float32)
            expected = (self.num_knots,)
            if noise_scale.shape != expected:
                raise ValueError(
                    f"noise_scale shape {noise_scale.shape} != expected {expected}"
                )
        else:
            horizon_idx = np.arange(self.num_knots, dtype=np.float32)
            opt_iter = np.float32(self._opt_iteration)
            noise_scale = self.noise_level * np.exp(
                -opt_iter / (self.traj_diffuse_factor * self.iterations)
                - (self.num_knots - 1 - horizon_idx)
                / (self.horizon_diffuse_factor * self.num_knots)
            )
            noise_scale = noise_scale.astype(np.float32)

        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu),
        ).astype(np.float32)
        knots = self.mean + noise * noise_scale[None, :, None]
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        self._opt_iteration = (self._opt_iteration + 1) % self.iterations
        return knots, controls

    def update_weights(
        self,
        total_costs: np.ndarray,
        knots: np.ndarray,
    ) -> np.ndarray:
        """Hydrax DIAL uses the same softmax mean update as MPPI."""
        costs = np.asarray(total_costs, dtype=np.float32)
        shifted = -costs / self.temperature
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
        self._opt_iteration = 0

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
