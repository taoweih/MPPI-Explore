"""Random-Fourier-feature value model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import warp as wp

from algs.value_models.base import ValueModel


@wp.kernel
def _random_fourier_inference(
    states:          wp.array2d(dtype=wp.float32),  # (N, in_dim)
    state_min:       wp.array1d(dtype=wp.float32),  # (in_dim,)
    state_range_inv: wp.array1d(dtype=wp.float32),  # (in_dim,)
    W:               wp.array2d(dtype=wp.float32),  # (num_features, in_dim)
    b:               wp.array1d(dtype=wp.float32),  # (num_features,)
    theta:           wp.array1d(dtype=wp.float32),  # (2*num_features + 1,)
    out:             wp.array1d(dtype=wp.float32),  # (N,) output
    in_dim:          int,
    num_features:    int,
):
    """V(s) = θ_sin · sin(W·s̃ + b) + θ_cos · cos(W·s̃ + b) + θ_bias

    where s̃ = 2·clip((s − s_min) / (s_max − s_min), 0, 1) − 1   ∈ [−1, 1]^d.
    """
    i = wp.tid()
    val = theta[2 * num_features]  # bias term

    for k in range(num_features):
        phase = b[k]
        for d in range(in_dim):
            x_norm = (states[i, d] - state_min[d]) * state_range_inv[d]
            x_norm = 2.0 * wp.clamp(x_norm, 0.0, 1.0) - 1.0
            phase = phase + W[k, d] * x_norm
        val = val + theta[k] * wp.sin(phase) + theta[num_features + k] * wp.cos(phase)

    out[i] = val


class RandomFourierValueModel(ValueModel):
    """V_θ(s) = θ · [sin(W·s̃ + b), cos(W·s̃ + b), 1]   (linear in random features).

    Trained in numpy via mini-batch SGD on the linear head θ; W and b are
    random and frozen.  Inferenced as a single fused Warp kernel reading
    from pre-allocated Warp arrays kept in sync with the numpy θ.
    """

    def __init__(
        self,
        state_dim: int,
        state_min: np.ndarray,
        state_max: np.ndarray,
        num_features: int = 256,
        seed: int = 0,
        device: str = "cuda:0",
    ) -> None:
        self.state_dim = state_dim
        self.num_features = num_features
        self.rng = np.random.default_rng(seed)
        self._device = device

        self.state_min = np.asarray(state_min, dtype=np.float32).reshape(state_dim)
        self.state_max = np.asarray(state_max, dtype=np.float32).reshape(state_dim)

        # Random feature parameters (fixed); only the linear head θ trains.
        self._W = self.rng.normal(
            0.0, 1.0, size=(num_features, state_dim),
        ).astype(np.float32)
        self._b = self.rng.uniform(
            0.0, 2.0 * np.pi, size=(num_features,),
        ).astype(np.float32)
        self._theta = np.zeros((2 * num_features + 1,), dtype=np.float32)

        # Warp arrays for graph-captured inference.
        self._state_min_wp = wp.zeros(state_dim, dtype=wp.float32, device=device)
        self._state_min_wp.assign(self.state_min)
        denom = np.maximum(self.state_max - self.state_min, 1e-6)
        self._state_range_inv_wp = wp.zeros(state_dim, dtype=wp.float32, device=device)
        self._state_range_inv_wp.assign((1.0 / denom).astype(np.float32))
        self._W_wp = wp.zeros((num_features, state_dim), dtype=wp.float32, device=device)
        self._W_wp.assign(self._W)
        self._b_wp = wp.zeros(num_features, dtype=wp.float32, device=device)
        self._b_wp.assign(self._b)
        self._theta_wp = wp.zeros(2 * num_features + 1, dtype=wp.float32, device=device)
        self._sync_to_warp()

        self._num_samples = 0

    # ── Allocation (no per-sample scratch needed for fused kernel) ────

    def alloc(self, num_samples: int, device) -> None:
        self._num_samples = num_samples

    # ── Inference launch (in captured graph) ──────────────────────────

    def launch_inference(self, states_wp: wp.array, out_wp: wp.array) -> None:
        wp.launch(_random_fourier_inference, dim=self._num_samples, inputs=[
            states_wp, self._state_min_wp, self._state_range_inv_wp,
            self._W_wp, self._b_wp, self._theta_wp, out_wp,
            self.state_dim, self.num_features,
        ])

    # ── Sync numpy θ → Warp θ ─────────────────────────────────────────

    def _sync_to_warp(self) -> None:
        self._theta_wp.assign(self._theta)

    # ── Predict (numpy-side) ──────────────────────────────────────────

    def _normalize(self, states: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.state_max - self.state_min, 1e-6)
        x01 = (states - self.state_min[None, :]) / denom[None, :]
        return 2.0 * np.clip(x01, 0.0, 1.0) - 1.0

    def _features(self, states: np.ndarray) -> np.ndarray:
        x = self._normalize(states)
        phase = x @ self._W.T + self._b[None, :]
        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)
        ones = np.ones((states.shape[0], 1), dtype=np.float32)
        return np.concatenate([sin_phi, cos_phi, ones], axis=1).astype(np.float32)

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states[None, :]
        return (self._features(states) @ self._theta).astype(np.float32)

    # ── Pretrain / online (mini-batch SGD on θ) ──────────────────────

    def fit_pretrain(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool = False,
        print_every: int = 50,
    ) -> float:
        return self._fit(
            states, targets, epochs=epochs, batch_size=batch_size, lr=learning_rate,
            sample_weights=None, one_sided=False, l2=0.0,
            verbose=verbose, print_every=print_every,
        )

    def fit_online(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        self._fit(
            states, targets, epochs=epochs, batch_size=batch_size, lr=learning_rate,
            sample_weights=sample_weights, one_sided=True, l2=0.0,
            verbose=False, print_every=50,
        )

    def _fit(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        sample_weights: Optional[np.ndarray],
        one_sided: bool,
        l2: float,
        verbose: bool,
        print_every: int,
    ) -> float:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]
        n = states.shape[0]

        if sample_weights is None:
            sample_weights = np.ones((n,), dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        bs = min(batch_size, n)
        num_batches = max(1, n // bs)
        used = num_batches * bs

        last_loss = 0.0
        for epoch in range(max(epochs, 1)):
            perm = self.rng.permutation(n)[:used]
            batch_idx = perm.reshape(num_batches, bs)
            epoch_loss = 0.0
            for b in range(num_batches):
                idx = batch_idx[b]
                xb, yb, wb = states[idx], targets[idx], sample_weights[idx]
                phi = self._features(xb)
                pred = phi @ self._theta

                if one_sided:
                    residual = np.maximum(yb - pred, 0.0)
                    grad = -(2.0 / bs) * (phi.T @ (wb * residual))
                    batch_loss = float(np.mean(wb * residual * residual))
                else:
                    residual = pred - yb
                    grad = (2.0 / bs) * (phi.T @ (wb * residual))
                    batch_loss = float(np.mean(wb * residual * residual))

                if l2 > 0.0:
                    grad += 2.0 * l2 * self._theta

                self._theta -= lr * grad.astype(np.float32)
                epoch_loss += batch_loss

            last_loss = epoch_loss / num_batches
            if verbose and epoch % max(print_every, 1) == 0:
                print(f"  epoch {epoch:4d}/{epochs} | loss={last_loss:.6f}")

        self._sync_to_warp()
        return last_loss

    # ── Snapshot / save / load ────────────────────────────────────────

    def copy_weights(self):
        return self._theta.copy()

    def restore_weights(self, snapshot) -> None:
        self._theta = np.asarray(snapshot, dtype=np.float32).copy()
        self._sync_to_warp()

    def save_weights_to_file(self, path) -> None:
        np.save(Path(path), self._theta)

    def load_weights_from_file(self, path) -> None:
        self._theta = np.load(Path(path)).astype(np.float32)
        self._sync_to_warp()
