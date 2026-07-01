"""Gaussian KDE density model."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import warp as wp

from algs.density_models.base import DensityModel


class KDEDensityModel(DensityModel):
    """Gaussian KDE density estimator with inverse-density systematic resampling.

    ρ(s_i) = 1/N · Σ_j exp(−½ ‖(s_i − s_j) / bw‖²)        # Gaussian KDE
    w_i    = (1 / (ρ_i + ε))^α  /  Σ_k (1 / (ρ_k + ε))^α  # inverse-density weights
    indices ← systematic_resample(w)                       # decorrelated draws

    `bandwidth` may be a scalar (broadcast to all state dimensions) or an
    array of length state_dim.  `alpha` controls how strongly samples in
    low-density regions are favoured (α=1 full inverse-density, α=0 uniform).
    DensityGuidedMPPI can additionally combine normalized low-density and
    low-cost scores at each stage boundary.
    """

    def __init__(
        self,
        bandwidth: Union[float, np.ndarray],
        alpha: float = 1.0,
    ) -> None:
        self.bandwidth = bandwidth
        self.alpha = float(alpha)

        # Lazily set in alloc().
        self._num_samples = 0
        self._state_dim = 0
        self._num_resample_stages = 0
        self._density_wp: Optional[wp.array] = None
        self._bandwidth_wp: Optional[wp.array] = None
        self._offsets_wp: Optional[wp.array] = None

    # ── Math ──────────────────────────────────────────────────────────

    def launch_compute(self, states_wp: wp.array) -> None:
        """ρ(s_i) = 1/N · Σ_j exp(−½ ‖(s_i − s_j) / bw‖²)."""
        kernel = getattr(type(self), "_kde_density_kernel", None)
        if kernel is None:
            @wp.kernel
            def kde_density(
                states:  wp.array2d(dtype=wp.float32),  # (N, D)
                density: wp.array1d(dtype=wp.float32),  # (N,) output
                bw:      wp.array1d(dtype=wp.float32),  # (D,)
                N:       int,
                D:       int,
            ):
                i = wp.tid()
                total = float(0.0)
                for j in range(N):
                    dist_sq = float(0.0)
                    for d in range(D):
                        diff = (states[i, d] - states[j, d]) / bw[d]
                        dist_sq = dist_sq + diff * diff
                    total = total + wp.exp(-0.5 * dist_sq)
                density[i] = total / float(N)

            type(self)._kde_density_kernel = kde_density
            kernel = kde_density

        wp.launch(kernel, dim=self._num_samples, inputs=[
            states_wp, self._density_wp, self._bandwidth_wp,
            self._num_samples, self._state_dim,
        ])

    def launch_resample(self, indices_wp: wp.array, stage_idx: int) -> None:
        """Systematic resample with weights ∝ 1 / (ρ + ε)^α."""
        kernel = getattr(type(self), "_resample_from_density_kernel", None)
        if kernel is None:
            @wp.kernel
            def resample_from_density(
                density:   wp.array1d(dtype=wp.float32),  # (N,)
                indices:   wp.array1d(dtype=wp.int32),    # (N,) output
                offsets:   wp.array1d(dtype=wp.float32),  # (n_boundaries,) U[0, 1/N)
                stage_idx: int,
                N:         int,
                alpha:     float,
            ):
                tid = wp.tid()
                if tid > 0:
                    return

                eps = float(1.0e-6)
                inv_total = float(0.0)
                for j in range(N):
                    inv_total = inv_total + wp.pow(1.0 / (density[j] + eps), alpha)

                # Systematic resampling: one random phase, then N evenly spaced thresholds.
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

            type(self)._resample_from_density_kernel = resample_from_density
            kernel = resample_from_density

        wp.launch(kernel, dim=1, inputs=[
            self._density_wp, indices_wp, self._offsets_wp,
            int(stage_idx), self._num_samples, self.alpha,
        ])

    def launch_resample_with_cost(
        self,
        costs_wp: wp.array,
        indices_wp: wp.array,
        stage_idx: int,
        cost_temperature: float,
        cost_weight: float,
    ) -> None:
        """Systematic resample with softmax over normalized density/cost scores."""
        kernel = getattr(type(self), "_resample_from_density_and_cost_kernel", None)
        if kernel is None:
            @wp.kernel
            def resample_from_density_and_cost(
                density:          wp.array1d(dtype=wp.float32),  # (N,)
                costs:            wp.array1d(dtype=wp.float32),  # (N,)
                indices:          wp.array1d(dtype=wp.int32),    # (N,) output
                offsets:          wp.array1d(dtype=wp.float32),  # (n_boundaries,)
                stage_idx:        int,
                N:                int,
                alpha:            float,
                cost_temperature: float,
                cost_weight:      float,
            ):
                tid = wp.tid()
                if tid > 0:
                    return

                eps = float(1.0e-6)
                score_eps = float(1.0e-6)

                density_mean = float(0.0)
                cost_mean = float(0.0)
                for j in range(N):
                    density_score = -wp.log(density[j] + eps)
                    cost_score = -costs[j]
                    density_mean = density_mean + density_score
                    cost_mean = cost_mean + cost_score
                inv_N = 1.0 / float(N)
                density_mean = density_mean * inv_N
                cost_mean = cost_mean * inv_N

                density_var = float(0.0)
                cost_var = float(0.0)
                for j in range(N):
                    density_delta = -wp.log(density[j] + eps) - density_mean
                    cost_delta = -costs[j] - cost_mean
                    density_var = density_var + density_delta * density_delta
                    cost_var = cost_var + cost_delta * cost_delta

                density_std = wp.sqrt(density_var * inv_N)
                cost_std = wp.sqrt(cost_var * inv_N)
                if density_std < score_eps:
                    density_std = score_eps
                if cost_std < score_eps:
                    cost_std = score_eps

                max_log_w = (
                    alpha * ((-wp.log(density[0] + eps) - density_mean) / density_std)
                    + (cost_weight / cost_temperature)
                    * ((-costs[0] - cost_mean) / cost_std)
                )
                for j_max in range(1, N):
                    density_z = (-wp.log(density[j_max] + eps) - density_mean) / density_std
                    cost_z = (-costs[j_max] - cost_mean) / cost_std
                    log_w = alpha * density_z + (cost_weight / cost_temperature) * cost_z
                    if log_w > max_log_w:
                        max_log_w = log_w

                total = float(0.0)
                for j in range(N):
                    density_z = (-wp.log(density[j] + eps) - density_mean) / density_std
                    cost_z = (-costs[j] - cost_mean) / cost_std
                    log_w = alpha * density_z + (cost_weight / cost_temperature) * cost_z
                    total = total + wp.exp(log_w - max_log_w)

                # Systematic resampling: one random phase, then N evenly spaced thresholds.
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
                            density_z = (
                                -wp.log(density[j] + eps) - density_mean
                            ) / density_std
                            cost_z = (-costs[j] - cost_mean) / cost_std
                            log_w = (
                                alpha * density_z
                                + (cost_weight / cost_temperature) * cost_z
                            )
                            w_j = wp.exp(log_w - max_log_w) / total
                            if cumulative + w_j < threshold:
                                cumulative = cumulative + w_j
                                j = j + 1
                            else:
                                can_advance = 0
                    indices[i] = j

            type(self)._resample_from_density_and_cost_kernel = (
                resample_from_density_and_cost
            )
            kernel = resample_from_density_and_cost

        wp.launch(kernel, dim=1, inputs=[
            self._density_wp, costs_wp, indices_wp, self._offsets_wp,
            int(stage_idx), self._num_samples, self.alpha,
            float(cost_temperature), float(cost_weight),
        ])

    # ── Helpers ───────────────────────────────────────────────────────

    def alloc(
        self,
        num_samples: int,
        state_dim: int,
        num_resample_stages: int,
        device,
    ) -> None:
        self._num_samples = num_samples
        self._state_dim = state_dim
        self._num_resample_stages = max(num_resample_stages, 1)

        bw = np.asarray(self.bandwidth, dtype=np.float32).ravel()
        if bw.size == 1 and state_dim > 1:
            bw = np.full(state_dim, float(bw[0]), dtype=np.float32)
        if bw.size != state_dim:
            raise ValueError(
                f"bandwidth must be scalar or length {state_dim}; got shape {bw.shape}"
            )

        self._bandwidth_wp = wp.zeros(state_dim, dtype=wp.float32, device=device)
        self._bandwidth_wp.assign(bw)

        self._density_wp = wp.zeros(num_samples, dtype=wp.float32, device=device)
        self._offsets_wp = wp.zeros(
            self._num_resample_stages, dtype=wp.float32, device=device,
        )

    def randomize_offsets(self, rng: np.random.Generator) -> None:
        """Refill the per-stage U[0, 1/N) offsets for the systematic resample."""
        offsets = rng.uniform(
            0.0, 1.0 / float(self._num_samples),
            size=self._num_resample_stages,
        ).astype(np.float32)
        self._offsets_wp.assign(offsets)
