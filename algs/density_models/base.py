"""Base interface for density models used by density-guided MPPI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import warp as wp


class DensityModel(ABC):
    """Abstract sample-density estimator used by staged density-guided rollouts.

    Owns its density buffer + any stochastic inputs (e.g. resample offsets).
    The controller owns `states_wp` (input) and `indices_wp` (resample output).
    """

    @abstractmethod
    def alloc(
        self,
        num_samples: int,
        state_dim: int,
        num_resample_stages: int,
        device,
    ) -> None:
        """One-time allocation of internal buffers (density, offsets, ...)."""

    @abstractmethod
    def launch_compute(self, states_wp: wp.array) -> None:
        """Compute density into the internal density buffer.
        Reads `states_wp` shape (N, state_dim)."""

    @abstractmethod
    def launch_resample(self, indices_wp: wp.array, stage_idx: int) -> None:
        """Systematic resample.  Writes into `indices_wp` shape (N,);
        reads internal density buffer and the `stage_idx`-th stochastic offset."""

    def launch_resample_with_cost(
        self,
        costs_wp: wp.array,
        indices_wp: wp.array,
        stage_idx: int,
        cost_temperature: float,
        cost_weight: float,
    ) -> None:
        """Systematic resample using inverse density and low rollout cost.

        Implementations should use weights proportional to their existing
        inverse-density weight multiplied by a low-cost preference, typically
        ``exp(-cost_weight * (J_i - min_j J_j) / cost_temperature)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support cost-aware resampling."
        )

    @abstractmethod
    def randomize_offsets(self, rng: np.random.Generator) -> None:
        """Pre-call: refill stochastic inputs for the next captured-graph replay.
        Uploads to the same device buffer the captured graph reads from, so no
        graph rebuild is needed."""
