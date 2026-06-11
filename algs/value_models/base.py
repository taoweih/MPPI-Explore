"""Base interface for terminal-value models used by value-guided MPPI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import warp as wp


class ValueModel(ABC):
    """Abstract terminal-cost value function V_θ(s) for value-guided MPPI.

    The model owns its trainable parameters and (for graph-compatible
    implementations) the pre-allocated Warp arrays the captured rollout
    graph reads from.  Training methods write updated weights into those
    same Warp arrays in-place, so the captured graph picks up new values
    at next replay without rebuild.

    Lifecycle:
        m = HashGridValueModel(state_dim=2, ...)         # construct
        m.alloc(num_samples=N, device=dev)               # one-time scratch buffers
        m.fit_pretrain(states, targets, ...)             # offline, all params
        # ── inside controller's captured rollout graph: ──
        m.launch_inference(states_wp, terminal_costs_wp)
        m.fit_online(states, targets, ...)               # online, subset of params
    """

    state_dim: int

    @abstractmethod
    def alloc(self, num_samples: int, device) -> None:
        """One-time allocation of per-sample inference scratch buffers.
        Tied to the controller's `num_samples` (sample batch size)."""

    @abstractmethod
    def launch_inference(self, states_wp: wp.array, out_wp: wp.array) -> None:
        """Launch V_θ(s) inference kernels into the open CUDA-graph capture.
        Reads `states_wp` shape (N, state_dim).  Overwrites `out_wp` shape (N,)."""

    @abstractmethod
    def predict(self, states: np.ndarray) -> np.ndarray:
        """Numpy in/out V_θ(s) for CPU-side use (visualization, anchors)."""

    @abstractmethod
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
        """Train ALL parameters on (state, target) pairs.  Returns final epoch loss."""

    @abstractmethod
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
        """One online training pass; subclasses define what's frozen / loss type."""

    # Snapshot / restore — used by `reset(reset_value_to_pretrained=True)`.

    @abstractmethod
    def copy_weights(self):
        """Deep snapshot of trainable state (for restore_weights)."""

    @abstractmethod
    def restore_weights(self, snapshot) -> None:
        """Reload weights from a snapshot in place; sync to Warp arrays."""

    @abstractmethod
    def save_weights_to_file(self, path) -> None: ...

    @abstractmethod
    def load_weights_from_file(self, path) -> None: ...
