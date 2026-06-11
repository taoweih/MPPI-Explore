"""Value + Density-Guided MPPI composition.

This controller combines density-guided staged resampling with a learned
terminal value function without duplicating either implementation:

    - density staging comes from `DensityGuidedMPPI.rollout`
    - value pretraining, online updates, reset/load/save, and terminal value
      inference come from `ValueGuidedMPPI`

The inheritance order intentionally puts `ValueGuidedMPPI` first so the
density rollout's `_emit_terminal_cost()` hook resolves to learned value
inference instead of the task's hand-written terminal cost.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from algs.density_guided_mppi import DensityGuidedMPPI
from algs.density_models import DensityModel
from algs.value_guided_mppi import ValueGuidedMPPI
from algs.value_models import ValueModel


class ValueDensityGuidedMPPI(ValueGuidedMPPI, DensityGuidedMPPI):
    """Density-guided rollout with learned value terminal cost and update."""

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
        value_model: ValueModel,
        density_model: Optional[DensityModel] = None,
        num_knots_per_stage: int = 4,
        state_extract_fn: Optional[Callable] = None,
        state_dim: Optional[int] = None,
        online_learning_rate: float = 1e-3,
        online_update_epochs: int = 1,
        online_batch_size: int = 1,
    ) -> None:
        DensityGuidedMPPI.__init__(
            self,
            task=task,
            num_samples=num_samples,
            noise_level=noise_level,
            temperature=temperature,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
            seed=seed,
            density_model=density_model,
            num_knots_per_stage=num_knots_per_stage,
            state_extract_fn=state_extract_fn,
            state_dim=state_dim,
        )

        self.value_model = value_model
        self.online_learning_rate = float(online_learning_rate)
        self.online_update_epochs = max(int(online_update_epochs), 1)
        self.online_batch_size = max(int(online_batch_size), 1)

        self._alloc_value_state_extract_buffer()
        self.value_model.alloc(self.num_samples, self._device)
        self._pretrained_snapshot = None

    def rollout(self, controls: np.ndarray, knots: np.ndarray):
        """Use density staging; terminal cost resolves to learned V_θ(s_T)."""
        return DensityGuidedMPPI.rollout(self, controls, knots)
