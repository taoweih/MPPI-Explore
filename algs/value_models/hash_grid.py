"""Hash-grid MLP value model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

from algs.value_models.base import ValueModel


# Architectural constants (matching hydrax / instant-NGP).
_HG_FEATURES_PER_LEVEL = 2
_HG_HIDDEN_DIM = 64


class _HashGridMLP(nn.Module):
    """Multi-resolution hash grid encoder followed by a 2-layer MLP.

    Architecture:
        - Hash grid: `num_levels` resolutions exponentially spaced in
          [min_resolution, max_resolution]; `table_size` per level;
          `_HG_FEATURES_PER_LEVEL=2` features per level.
        - MLP: input_dim → 64 → 64 → 1, swish (silu) activations.

    Pretraining updates ALL parameters.
    Online updates freeze the MLP and train only the hashgrid embeddings.
    """

    FEATURES_PER_LEVEL = _HG_FEATURES_PER_LEVEL
    HIDDEN_DIM = _HG_HIDDEN_DIM

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
            torch.empty(num_levels, table_size, self.FEATURES_PER_LEVEL)
            .uniform_(-1e-4, 1e-4)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, din) → V (B,)."""
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

        encoded = torch.cat(all_features, dim=-1)  # (B, num_levels * 2)

        h = F.silu(self.linear1(encoded))
        h = F.silu(self.linear2(h))
        return self.linear_out(h).squeeze(-1)

    def hashgrid_params(self) -> list[nn.Parameter]:
        """Return only hash-grid parameters (used for online-only optimisation)."""
        return [self.embeddings]

    def mlp_params(self) -> list[nn.Parameter]:
        """Return all MLP parameters (frozen during online updates)."""
        params: list[nn.Parameter] = []
        for layer in (self.linear1, self.linear2, self.linear_out):
            params.extend(layer.parameters())
        return params


@wp.kernel
def _hashgrid_encode_2d(
    states:         wp.array2d(dtype=wp.float32),  # (N, 2)
    embeddings:     wp.array1d(dtype=wp.float32),  # (num_levels * table_size * 2,)
    resolutions:    wp.array1d(dtype=wp.float32),  # (num_levels,)
    features:       wp.array2d(dtype=wp.float32),  # (N, num_levels * 2) output
    grid_min:       float,
    grid_range_inv: float,
    num_levels:     int,
    table_size:     int,
):
    """Multi-resolution hash-grid encode of a 2-D state."""
    i = wp.tid()
    x0 = (states[i, 0] - grid_min) * grid_range_inv
    x1 = (states[i, 1] - grid_min) * grid_range_inv

    P0 = wp.uint32(1)
    P1 = wp.uint32(2654435761)
    ts = wp.uint32(table_size)

    for lev in range(num_levels):
        res = resolutions[lev]
        gx0 = x0 * res
        gx1 = x1 * res
        fx0 = wp.floor(gx0)
        fx1 = wp.floor(gx1)
        ix0 = int(fx0)
        ix1 = int(fx1)
        wx0 = gx0 - fx0
        wx1 = gx1 - fx1
        base = lev * table_size * 2

        f0 = float(0.0)
        f1 = float(0.0)
        for c in range(4):
            c0 = c & 1
            c1 = (c >> 1) & 1
            w = (1.0 - wx0 + float(c0) * (2.0 * wx0 - 1.0)) * (
                1.0 - wx1 + float(c1) * (2.0 * wx1 - 1.0)
            )
            h = int(
                (wp.uint32(ix0 + c0) * P0 + wp.uint32(ix1 + c1) * P1) % ts
            )
            f0 = f0 + w * embeddings[base + h * 2]
            f1 = f1 + w * embeddings[base + h * 2 + 1]

        features[i, lev * 2] = f0
        features[i, lev * 2 + 1] = f1


@wp.kernel
def _hashgrid_encode_3d(
    states:         wp.array2d(dtype=wp.float32),
    embeddings:     wp.array1d(dtype=wp.float32),
    resolutions:    wp.array1d(dtype=wp.float32),
    features:       wp.array2d(dtype=wp.float32),
    grid_min:       float,
    grid_range_inv: float,
    num_levels:     int,
    table_size:     int,
):
    """Multi-resolution hash-grid encode of a 3-D state."""
    i = wp.tid()
    x0 = (states[i, 0] - grid_min) * grid_range_inv
    x1 = (states[i, 1] - grid_min) * grid_range_inv
    x2 = (states[i, 2] - grid_min) * grid_range_inv

    P0 = wp.uint32(1)
    P1 = wp.uint32(2654435761)
    P2 = wp.uint32(805459861)
    ts = wp.uint32(table_size)

    for lev in range(num_levels):
        res = resolutions[lev]
        gx0 = x0 * res
        gx1 = x1 * res
        gx2 = x2 * res
        fx0 = wp.floor(gx0)
        fx1 = wp.floor(gx1)
        fx2 = wp.floor(gx2)
        ix0 = int(fx0)
        ix1 = int(fx1)
        ix2 = int(fx2)
        wx0 = gx0 - fx0
        wx1 = gx1 - fx1
        wx2 = gx2 - fx2
        base = lev * table_size * 2

        f0 = float(0.0)
        f1 = float(0.0)
        for c in range(8):
            c0 = c & 1
            c1 = (c >> 1) & 1
            c2 = (c >> 2) & 1
            w = (
                (1.0 - wx0 + float(c0) * (2.0 * wx0 - 1.0))
                * (1.0 - wx1 + float(c1) * (2.0 * wx1 - 1.0))
                * (1.0 - wx2 + float(c2) * (2.0 * wx2 - 1.0))
            )
            h = int(
                (
                    wp.uint32(ix0 + c0) * P0
                    + wp.uint32(ix1 + c1) * P1
                    + wp.uint32(ix2 + c2) * P2
                ) % ts
            )
            f0 = f0 + w * embeddings[base + h * 2]
            f1 = f1 + w * embeddings[base + h * 2 + 1]

        features[i, lev * 2] = f0
        features[i, lev * 2 + 1] = f1


@wp.kernel
def _dense_swish(
    inp:    wp.array2d(dtype=wp.float32),  # (N, in_dim)
    weight: wp.array2d(dtype=wp.float32),  # (in_dim, out_dim)
    bias:   wp.array1d(dtype=wp.float32),  # (out_dim,)
    out:    wp.array2d(dtype=wp.float32),  # (N, out_dim) output
    in_dim:  int,
    out_dim: int,
):
    """Fused linear + swish (silu).  Launch with dim = N * out_dim."""
    tid = wp.tid()
    i = tid / out_dim
    j = tid - i * out_dim
    val = bias[j]
    for k in range(in_dim):
        val = val + inp[i, k] * weight[k, j]
    out[i, j] = val / (1.0 + wp.exp(-val))


@wp.kernel
def _dense_linear_1d(
    inp:    wp.array2d(dtype=wp.float32),  # (N, in_dim)
    weight: wp.array1d(dtype=wp.float32),  # (in_dim,)
    bias:   wp.array1d(dtype=wp.float32),  # (1,)
    out:    wp.array1d(dtype=wp.float32),  # (N,) output
    in_dim:  int,
):
    """Linear layer with scalar output.  Launch with dim = N."""
    i = wp.tid()
    val = bias[0]
    for k in range(in_dim):
        val = val + inp[i, k] * weight[k]
    out[i] = val


class HashGridValueModel(ValueModel):
    """Hash-grid + MLP value model.  See module docstring for the math.

    Parameters live in PyTorch tensors (CUDA, used for training) and in
    pre-allocated Warp arrays (same CUDA device, read by the captured
    rollout graph).  `_sync_to_warp()` is a GPU-to-GPU copy from the
    PyTorch tensors into the Warp arrays — no host round trip.

    Online training is captured as a `torch.cuda.CUDAGraph` (forward +
    backward + Adam step in one replay).  The MLP is frozen for online
    updates; only hashgrid embeddings are trained.
    """

    def __init__(
        self,
        state_dim: int,
        grid_min: float,
        grid_max: float,
        num_levels: int = 16,
        table_size: int = 4096,
        min_resolution: float = 16.0,
        max_resolution: float = 2048.0,
        seed: int = 0,
        device: str = "cuda:0",
    ) -> None:
        self.state_dim = state_dim
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.num_levels = num_levels
        self.table_size = table_size
        self._torch_device = torch.device(device)
        self._device = device

        input_dim = num_levels * _HG_FEATURES_PER_LEVEL
        self._hg_input_dim = input_dim

        # PyTorch model (CUDA, used for training).
        torch.manual_seed(seed)
        self.model = _HashGridMLP(
            din=state_dim, grid_min=grid_min, grid_max=grid_max,
            num_levels=num_levels, table_size=table_size,
            min_resolution=min_resolution, max_resolution=max_resolution,
        )
        self.model.to(self._torch_device)
        self.model.eval()

        # Warp arrays (GPU, read by captured inference graph).
        total_emb = num_levels * table_size * _HG_FEATURES_PER_LEVEL
        self._embeddings_wp = wp.zeros(total_emb, dtype=wp.float32, device=device)

        self._resolutions_wp = wp.zeros(num_levels, dtype=wp.float32, device=device)
        self._resolutions_wp.assign(
            self.model.resolutions.cpu().numpy().astype(np.float32),
        )

        # MLP weights — stored as (in_dim, out_dim) for the warp kernel.
        self._W1_wp = wp.zeros((input_dim, _HG_HIDDEN_DIM), dtype=wp.float32, device=device)
        self._b1_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self._W2_wp = wp.zeros((_HG_HIDDEN_DIM, _HG_HIDDEN_DIM), dtype=wp.float32, device=device)
        self._b2_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self._W_out_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self._b_out_wp = wp.zeros(1, dtype=wp.float32, device=device)

        self._grid_range_inv = 1.0 / max(self.grid_max - self.grid_min, 1e-6)

        # Per-sample inference scratch — sized in alloc().
        self._features_wp: Optional[wp.array] = None
        self._hidden1_wp: Optional[wp.array] = None
        self._hidden2_wp: Optional[wp.array] = None
        self._num_samples = 0

        # Online-training CUDA graph (lazily built by _ensure_online_graph).
        self._online_graph = None
        self._online_opt = None
        self._online_bs: int = 0
        self._g_states = None
        self._g_targets = None
        self._g_weights = None

        self._sync_to_warp()

    # ── Allocation ────────────────────────────────────────────────────

    def alloc(self, num_samples: int, device) -> None:
        """Pre-allocate per-sample inference scratch (features, hidden1, hidden2)."""
        self._num_samples = num_samples
        self._features_wp = wp.zeros(
            (num_samples, self._hg_input_dim), dtype=wp.float32, device=device,
        )
        self._hidden1_wp = wp.zeros(
            (num_samples, _HG_HIDDEN_DIM), dtype=wp.float32, device=device,
        )
        self._hidden2_wp = wp.zeros(
            (num_samples, _HG_HIDDEN_DIM), dtype=wp.float32, device=device,
        )

    # ── Inference launch (in captured graph) ──────────────────────────

    def launch_inference(self, states_wp: wp.array, out_wp: wp.array) -> None:
        """ψ → swish(ψ·W₁+b₁) → swish(·W₂+b₂) → ·W_out+b_out → V."""
        N = self._num_samples
        encode_kernel = _hashgrid_encode_2d if self.state_dim == 2 else _hashgrid_encode_3d

        wp.launch(encode_kernel, dim=N, inputs=[
            states_wp, self._embeddings_wp, self._resolutions_wp,
            self._features_wp,
            self.grid_min, self._grid_range_inv,
            self.num_levels, self.table_size,
        ])
        wp.launch(_dense_swish, dim=N * _HG_HIDDEN_DIM, inputs=[
            self._features_wp, self._W1_wp, self._b1_wp, self._hidden1_wp,
            self._hg_input_dim, _HG_HIDDEN_DIM,
        ])
        wp.launch(_dense_swish, dim=N * _HG_HIDDEN_DIM, inputs=[
            self._hidden1_wp, self._W2_wp, self._b2_wp, self._hidden2_wp,
            _HG_HIDDEN_DIM, _HG_HIDDEN_DIM,
        ])
        wp.launch(_dense_linear_1d, dim=N, inputs=[
            self._hidden2_wp, self._W_out_wp, self._b_out_wp, out_wp,
            _HG_HIDDEN_DIM,
        ])

    # ── Predict (CPU-facing) ──────────────────────────────────────────

    def predict(self, states: np.ndarray) -> np.ndarray:
        """V_θ(s) for batched states; numpy in/out."""
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(
                self._torch_device,
            )
            return self.model(t).cpu().numpy()

    # ── Sync PyTorch → Warp arrays (GPU-to-GPU) ──────────────────────

    @staticmethod
    def _wp_copy_from_torch(dst: wp.array, src_tensor: torch.Tensor) -> None:
        """GPU-to-GPU copy from a contiguous PyTorch CUDA tensor into a warp array."""
        src_flat = src_tensor.detach().contiguous().view(-1)
        src_wp = wp.from_torch(src_flat, dtype=wp.float32)
        wp.copy(dst, src_wp)

    def _sync_to_warp(self) -> None:
        """Copy all PyTorch weights into the pre-allocated Warp arrays."""
        with torch.no_grad():
            self._wp_copy_from_torch(self._embeddings_wp, self.model.embeddings.data)
            # PyTorch Linear stores (out, in); warp kernel expects (in, out).
            self._wp_copy_from_torch(
                self._W1_wp, self.model.linear1.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self._b1_wp, self.model.linear1.bias.data)
            self._wp_copy_from_torch(
                self._W2_wp, self.model.linear2.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self._b2_wp, self.model.linear2.bias.data)
            self._wp_copy_from_torch(self._W_out_wp, self.model.linear_out.weight.data)
            self._wp_copy_from_torch(self._b_out_wp, self.model.linear_out.bias.data)

    # ── Pretraining (all parameters, plain MSE) ──────────────────────

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
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        states_t = torch.from_numpy(
            np.asarray(states, dtype=np.float32),
        ).to(self._torch_device)
        targets_t = torch.from_numpy(
            np.asarray(targets, dtype=np.float32).ravel(),
        ).to(self._torch_device)

        n = states_t.shape[0]
        bs = min(batch_size, n)
        num_batches = max(1, n // bs)
        used = num_batches * bs

        last_loss = 0.0
        for epoch in range(max(epochs, 1)):
            perm = torch.randperm(n, device=self._torch_device)[:used]
            batch_idx = perm.reshape(num_batches, bs)

            epoch_loss = 0.0
            for b in range(num_batches):
                idx = batch_idx[b]
                pred = self.model(states_t[idx])
                loss = ((pred - targets_t[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            last_loss = epoch_loss / num_batches
            if verbose and epoch % max(print_every, 1) == 0:
                print(f"  epoch {epoch:4d}/{epochs} | loss={last_loss:.6f}")

        self.model.eval()
        self._sync_to_warp()
        return last_loss

    # ── Online update (hash grid only, captured CUDA graph) ──────────

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
        """Train hashgrid embeddings (MLP frozen) via captured CUDA-graph step.

        Loss: one-sided MSE
            L = mean( w_i · max(target_i − V_θ(s_i), 0)² )
        """
        dev = self._torch_device
        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=dev)
        targets_t = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).ravel(), device=dev,
        )
        if sample_weights is not None:
            weights_t = torch.as_tensor(
                np.asarray(sample_weights, dtype=np.float32).ravel(), device=dev,
            )
        else:
            weights_t = torch.ones(states_t.shape[0], device=dev)

        n = states_t.shape[0]
        bs = min(batch_size, n)
        num_batches = max(1, n // bs)
        used = num_batches * bs

        self._ensure_online_graph(bs, learning_rate)
        self._reset_online_optimizer_state()

        for _ in range(max(epochs, 1)):
            perm = torch.randperm(n, device=dev)[:used]
            batch_idx = perm.reshape(num_batches, bs)
            for b in range(num_batches):
                idx = batch_idx[b]
                self._g_states.copy_(states_t[idx])
                self._g_targets.copy_(targets_t[idx])
                self._g_weights.copy_(weights_t[idx])
                self._online_graph.replay()

        self._sync_to_warp()

    def _ensure_online_graph(self, batch_size: int, learning_rate: float) -> None:
        """Lazily build a torch CUDA graph for one online training step."""
        if self._online_graph is not None and self._online_bs == batch_size:
            return

        dev = self._torch_device

        # Freeze MLP before capture (only hashgrid embeddings train online).
        for p in self.model.mlp_params():
            p.requires_grad_(False)
        self.model.train()

        self._online_opt = torch.optim.Adam(
            self.model.hashgrid_params(), lr=learning_rate,
            capturable=True, foreach=False,
        )
        self._online_bs = batch_size

        # Static I/O buffers — addresses fixed for the captured graph.
        self._g_states = torch.zeros(batch_size, self.state_dim, device=dev)
        self._g_targets = torch.zeros(batch_size, device=dev)
        self._g_weights = torch.zeros(batch_size, device=dev)

        # Warm-up on a side stream to prime the allocator.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                pred = self.model(self._g_states)
                diff = self._g_targets - pred
                residual = torch.clamp(diff, min=0.0)
                loss = (self._g_weights * residual**2).mean()
                self._online_opt.zero_grad()
                loss.backward()
                self._online_opt.step()
        torch.cuda.current_stream().wait_stream(s)

        # Capture one training step (one-sided MSE, hashgrid only).
        self._online_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._online_graph):
            pred = self.model(self._g_states)
            diff = self._g_targets - pred
            residual = torch.clamp(diff, min=0.0)
            loss = (self._g_weights * residual**2).mean()
            self._online_opt.zero_grad()
            loss.backward()
            self._online_opt.step()

    def _reset_online_optimizer_state(self) -> None:
        """Zero Adam momentum so each fit_online call starts fresh."""
        if self._online_opt is None:
            return
        for state in self._online_opt.state.values():
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if isinstance(state["step"], torch.Tensor):
                state["step"].zero_()

    def _invalidate_online_graph(self) -> None:
        self._online_graph = None
        self._online_opt = None

    # ── Snapshot / save / load ────────────────────────────────────────

    def copy_weights(self):
        """Deep snapshot of the PyTorch state dict."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def restore_weights(self, snapshot) -> None:
        """Reload weights from a snapshot in place; sync to Warp arrays.
        Online graph is invalidated (param storage may have changed)."""
        self.model.load_state_dict(snapshot)
        self._invalidate_online_graph()
        self._sync_to_warp()

    def save_weights_to_file(self, path) -> None:
        torch.save(self.model.state_dict(), Path(path))

    def load_weights_from_file(self, path) -> None:
        self.model.load_state_dict(
            torch.load(Path(path), map_location=self._torch_device, weights_only=True),
        )
        self._invalidate_online_graph()
        self._sync_to_warp()
