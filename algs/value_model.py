"""Random Fourier feature value model for memory-augmented MPPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MemoryPretrainConfig:
    """Configuration for memory pretraining."""

    sample_count: int = 100000
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    print_every: int = 50


class RandomFeatureValueModel:
    """Continuous value model based on random Fourier features.

    Linear in random sinusoidal features — online updates are lightweight
    while still providing a continuous function approximation.
    """

    def __init__(
        self,
        input_dim: int,
        state_min: np.ndarray,
        state_max: np.ndarray,
        num_features: int = 256,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.num_features = num_features
        self.rng = np.random.default_rng(seed)

        self.state_min = np.asarray(state_min, dtype=np.float32).reshape(input_dim)
        self.state_max = np.asarray(state_max, dtype=np.float32).reshape(input_dim)

        # Random feature parameters (fixed); only the linear head is trained.
        self.W = self.rng.normal(0.0, 1.0, size=(num_features, input_dim)).astype(
            np.float32
        )
        self.b = self.rng.uniform(0.0, 2.0 * np.pi, size=(num_features,)).astype(
            np.float32
        )
        self.theta = np.zeros((2 * num_features + 1,), dtype=np.float32)

    def copy_weights(self) -> np.ndarray:
        return self.theta.copy()

    def load_weights(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=np.float32).copy()

    def _normalize(self, states: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.state_max - self.state_min, 1e-6)
        x01 = (states - self.state_min[None, :]) / denom[None, :]
        return 2.0 * np.clip(x01, 0.0, 1.0) - 1.0

    def _features(self, states: np.ndarray) -> np.ndarray:
        x = self._normalize(states)
        phase = x @ self.W.T + self.b[None, :]
        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)
        ones = np.ones((states.shape[0], 1), dtype=np.float32)
        return np.concatenate([sin_phi, cos_phi, ones], axis=1).astype(np.float32)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict value for batched states, shape (B, input_dim) -> (B,)."""
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states[None, :]
        phi = self._features(states)
        return (phi @ self.theta).astype(np.float32)

    def fit(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        sample_weights: Optional[np.ndarray] = None,
        one_sided: bool = False,
        l2: float = 0.0,
        verbose: bool = False,
        print_every: int = 50,
    ) -> float:
        """Train the linear head with epoch-based mini-batch SGD.

        Each epoch shuffles the data and iterates through all mini-batches.
        If *one_sided* is True the loss is ``max(target - pred, 0)^2``,
        penalising under-estimates only.
        """
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]
        n = states.shape[0]

        if sample_weights is None:
            sample_weights = np.ones((n,), dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        batch_size = min(batch_size, n)
        num_batches = max(1, n // batch_size)
        used_samples = num_batches * batch_size

        last_loss = 0.0
        for epoch in range(max(epochs, 1)):
            perm = self.rng.permutation(n)[:used_samples]
            batch_indices = perm.reshape(num_batches, batch_size)

            epoch_loss = 0.0
            for b in range(num_batches):
                idx = batch_indices[b]
                xb, yb, wb = states[idx], targets[idx], sample_weights[idx]
                phi = self._features(xb)
                pred = phi @ self.theta

                if one_sided:
                    residual = np.maximum(yb - pred, 0.0)
                    grad = -(2.0 / batch_size) * (phi.T @ (wb * residual))
                    batch_loss = float(np.mean(wb * residual * residual))
                else:
                    residual = pred - yb
                    grad = (2.0 / batch_size) * (phi.T @ (wb * residual))
                    batch_loss = float(np.mean(wb * residual * residual))

                if l2 > 0.0:
                    grad += 2.0 * l2 * self.theta

                self.theta -= learning_rate * grad.astype(np.float32)
                epoch_loss += batch_loss

            last_loss = epoch_loss / num_batches
            if verbose and epoch % max(print_every, 1) == 0:
                print(f"  epoch {epoch:4d}/{epochs} | loss={last_loss:.6f}")

        return last_loss
