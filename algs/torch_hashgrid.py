"""PyTorch multi-resolution hash grid + MLP value model.

Architecture based on the hydrax NeuralNet / instant-NGP design:
  - Multi-resolution hash grid, table_size=4096, 2 features/level
  - MLP: (num_levels*2) → 64 (swish) → 64 (swish) → 1

Pretraining updates all parameters (hash grid + MLP).
Online updates freeze the MLP and only train hash grid embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashGridMLP(nn.Module):
    """Multi-resolution hash grid encoder followed by a small MLP."""

    FEATURES_PER_LEVEL = 2
    HIDDEN_DIM = 64

    def __init__(
        self,
        din: int,
        grid_min: float,
        grid_max: float,
        num_levels: int = 16,
        table_size: int = 4096,
        min_resolution: float = 16.0,
        max_resolution: float = 2048.0,
    ) -> None:
        super().__init__()
        if din not in (2, 3):
            raise ValueError(f"din={din} not supported; use 2 or 3.")
        self.din = din
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.num_levels = num_levels
        self.table_size = table_size
        self.input_dim = num_levels * self.FEATURES_PER_LEVEL

        # Exponentially spaced resolutions.
        self.register_buffer(
            "resolutions",
            torch.exp(torch.linspace(
                math.log(min_resolution), math.log(max_resolution), num_levels,
            )),
        )

        # Hash grid embeddings (trainable).
        self.embeddings = nn.Parameter(
            torch.empty(
                num_levels, table_size, self.FEATURES_PER_LEVEL,
            ).uniform_(-1e-4, 1e-4)
        )

        # Hash primes (matching hydrax).
        if din == 2:
            primes = [1, 2654435761]
        else:
            primes = [1, 2654435761, 805459861]
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long))

        # Corner offsets for multi-linear interpolation.
        offsets = [
            [int(b) for b in format(i, f"0{din}b")] for i in range(2**din)
        ]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # MLP: input_dim → 64 → 64 → 1.
        self.linear1 = nn.Linear(self.input_dim, self.HIDDEN_DIM)
        self.linear2 = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.linear_out = nn.Linear(self.HIDDEN_DIM, 1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (B, din) → (B,)."""
        x_norm = (x - self.grid_min) / (self.grid_max - self.grid_min)

        all_features: list[torch.Tensor] = []
        for level in range(self.num_levels):
            res = self.resolutions[level]
            x_grid = x_norm * res
            x0 = torch.floor(x_grid).long()
            w = x_grid - x0.float()

            # All 2^din corner coordinates: (B, 2^din, din).
            grid_coords = x0.unsqueeze(1) + self.offsets.unsqueeze(0)

            # Spatial hash → table index: (B, 2^din).
            hashed = (grid_coords * self.primes).sum(-1) % self.table_size

            # Look up embeddings: (B, 2^din, features_per_level).
            corners = self.embeddings[level][hashed]

            # Multi-linear interpolation weights: (B, 2^din).
            off_f = self.offsets.unsqueeze(0).float()
            per_dim = 1.0 - off_f + w.unsqueeze(1) * (2.0 * off_f - 1.0)
            corner_weights = per_dim.prod(-1)

            value = (corner_weights.unsqueeze(-1) * corners).sum(1)
            all_features.append(value)

        encoded = torch.cat(all_features, dim=-1)  # (B, 32)

        h = F.silu(self.linear1(encoded))
        h = F.silu(self.linear2(h))
        return self.linear_out(h).squeeze(-1)

    # ------------------------------------------------------------------

    def hashgrid_params(self) -> list[nn.Parameter]:
        """Return only hash grid parameters (for online-only optimisation)."""
        return [self.embeddings]

    def mlp_params(self) -> list[nn.Parameter]:
        """Return all MLP parameters."""
        params: list[nn.Parameter] = []
        for layer in (self.linear1, self.linear2, self.linear_out):
            params.extend(layer.parameters())
        return params
