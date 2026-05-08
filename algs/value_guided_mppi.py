"""Value-Guided MPPI: terminal cost replaced by a learned hash-grid + MLP heuristic.

Algorithm: same MPPI loop, but the rollout's terminal cost φ(x_T) is replaced
by a learned value V_θ(s_T) where s_T is a low-D state extracted by the task.

    V_θ(s) = MLP(hashgrid(s))
        ψ  = hashgrid_encode(s)              # multi-resolution hash grid
        h₁ = swish(ψ · W₁ + b₁)              # 64-d
        h₂ = swish(h₁ · W₂ + b₂)             # 64-d
        V  = h₂ · W_out + b_out              # scalar

Pretraining (`fit_pretrain`): updates ALL parameters (hashgrid + MLP) on a
dataset of (state, target) pairs supplied by the user.  Targets are typically
distance-to-goal — a hand-crafted heuristic.

Online updates (`fit_online`, called after every optimize): freeze the MLP,
train ONLY the hashgrid embeddings on a one-sided MSE loss
        loss = mean(weight_i · max(target_i − V_θ(s_i), 0)²)
where target = best rollout cost from this optimize step.  Online updates are
captured as a `torch.cuda.CUDAGraph` (forward + backward + Adam in one replay).

Composition: `use_density_guided=True` runs the staged density-guided rollout
from `density_guided_mppi.py` (KDE / inverse-density resample / reshuffle /
knot regeneration) and uses the learned V as the final terminal cost.

CUDA-graph contract: weights are stored in pre-allocated wp arrays at fixed
addresses.  `sync_to_warp()` GPU-to-GPU copies the latest PyTorch weights
into those arrays, so the captured rollout graph picks up new values on its
next replay without needing to be rebuilt.  Graphs are invalidated only when
the *code path* changes (learned_value_ready toggles, weights loaded from
file).  In particular, `reset(reset_value_to_pretrained=True)` reloads
weights into the same tensors and does NOT invalidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from algs._graph import capture_graph
from algs.density_guided_mppi import (
    kde_density,
    resample_from_density,
    regenerate_knots,
    zero_order_interp,
)
from tasks.task_base import Task
from utils.spline import get_interp_func
from utils.warp_kernels import (
    gather_1d_float,
    gather_2d_float,
    gather_3d_float,
)


# ──────────────────────────────────────────────────────────────────────
# Pretraining config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ValuePretrainConfig:
    sample_count: int = 100000
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    print_every: int = 50


# ──────────────────────────────────────────────────────────────────────
# Hash-grid + MLP architectural constants (matching hydrax / instant-NGP)
# ──────────────────────────────────────────────────────────────────────


_HG_FEATURES_PER_LEVEL = 2
_HG_HIDDEN_DIM = 64


# ──────────────────────────────────────────────────────────────────────
# Learned-value kernels (the value-net inference math)
# ──────────────────────────────────────────────────────────────────────


@wp.kernel
def hashgrid_encode_2d(
    states:         wp.array2d(dtype=wp.float32),  # (N, 2)
    embeddings:     wp.array1d(dtype=wp.float32),  # (num_levels * table_size * 2,)
    resolutions:    wp.array1d(dtype=wp.float32),  # (num_levels,)
    features:       wp.array2d(dtype=wp.float32),  # (N, num_levels * 2) output
    grid_min:       float,
    grid_range_inv: float,
    num_levels:     int,
    table_size:     int,
):
    """Multi-resolution hash-grid encode of a 2-D state (matches hydrax/instant-NGP)."""
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
def hashgrid_encode_3d(
    states:         wp.array2d(dtype=wp.float32),  # (N, 3)
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
                )
                % ts
            )
            f0 = f0 + w * embeddings[base + h * 2]
            f1 = f1 + w * embeddings[base + h * 2 + 1]

        features[i, lev * 2] = f0
        features[i, lev * 2 + 1] = f1


@wp.kernel
def dense_swish(
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
def dense_linear_1d(
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


# ──────────────────────────────────────────────────────────────────────
# Learned value model (PyTorch for training, Warp arrays for graph inference)
# ──────────────────────────────────────────────────────────────────────


class _LearnedValueModel:
    """Hash-grid + MLP value model.

    Parameters live in PyTorch tensors (CUDA) for training and in
    pre-allocated Warp arrays (same CUDA device) for graph-captured
    inference.  `sync_to_warp()` is a GPU-to-GPU copy from the PyTorch
    tensors into the Warp arrays — no host round trip.

    The captured rollout graph reads weights from the Warp arrays' fixed
    addresses, so any sync that targets those same addresses is picked
    up at the next replay without a graph rebuild.
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
        import torch
        from algs.torch_hashgrid import HashGridMLP

        self.state_dim = state_dim
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.num_levels = num_levels
        self.table_size = table_size
        self._torch_device = torch.device(device)

        input_dim = num_levels * _HG_FEATURES_PER_LEVEL

        # PyTorch model (CUDA, used for training).
        torch.manual_seed(seed)
        self.model = HashGridMLP(
            din=state_dim, grid_min=grid_min, grid_max=grid_max,
            num_levels=num_levels, table_size=table_size,
            min_resolution=min_resolution, max_resolution=max_resolution,
        )
        self.model.to(self._torch_device)
        self.model.eval()

        # Warp arrays (GPU, read by captured inference graph).
        total_emb = num_levels * table_size * _HG_FEATURES_PER_LEVEL
        self.embeddings_wp = wp.zeros(total_emb, dtype=wp.float32, device=device)

        self.resolutions_wp = wp.zeros(num_levels, dtype=wp.float32, device=device)
        self.resolutions_wp.assign(
            self.model.resolutions.cpu().numpy().astype(np.float32),
        )

        # MLP weights — stored as (in_dim, out_dim) for the warp kernel.
        self.W1_wp = wp.zeros(
            (input_dim, _HG_HIDDEN_DIM), dtype=wp.float32, device=device,
        )
        self.b1_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self.W2_wp = wp.zeros(
            (_HG_HIDDEN_DIM, _HG_HIDDEN_DIM), dtype=wp.float32, device=device,
        )
        self.b2_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self.W_out_wp = wp.zeros(_HG_HIDDEN_DIM, dtype=wp.float32, device=device)
        self.b_out_wp = wp.zeros(1, dtype=wp.float32, device=device)

        # Online-training CUDA graph (lazily built).
        self._online_graph = None
        self._online_opt = None
        self._online_bs: int = 0
        self._g_states = None
        self._g_targets = None
        self._g_weights = None

        self.sync_to_warp()

    # ── Online-training CUDA graph ────────────────────────────────────

    def _invalidate_online_graph(self) -> None:
        self._online_graph = None
        self._online_opt = None

    def _ensure_online_graph(self, batch_size: int, learning_rate: float) -> None:
        """Lazily build a torch CUDA graph for one online training step."""
        import torch

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

        # Capture one training step:
        #   loss = mean(weight · max(target − V_θ(s), 0)²)   [one-sided MSE]
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
        import torch
        if self._online_opt is None:
            return
        for state in self._online_opt.state.values():
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if isinstance(state["step"], torch.Tensor):
                state["step"].zero_()

    # ── GPU-to-GPU PyTorch → Warp sync ────────────────────────────────

    @staticmethod
    def _wp_copy_from_torch(dst, src_tensor) -> None:
        """GPU-to-GPU copy from a contiguous PyTorch CUDA tensor into a warp array."""
        src_flat = src_tensor.detach().contiguous().view(-1)
        src_wp = wp.from_torch(src_flat, dtype=wp.float32)
        wp.copy(dst, src_wp)

    def sync_to_warp(self) -> None:
        """Copy all PyTorch weights into the pre-allocated Warp arrays."""
        import torch
        with torch.no_grad():
            self._wp_copy_from_torch(self.embeddings_wp, self.model.embeddings.data)
            # PyTorch Linear stores (out, in); warp kernel expects (in, out).
            self._wp_copy_from_torch(
                self.W1_wp, self.model.linear1.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self.b1_wp, self.model.linear1.bias.data)
            self._wp_copy_from_torch(
                self.W2_wp, self.model.linear2.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self.b2_wp, self.model.linear2.bias.data)
            self._wp_copy_from_torch(self.W_out_wp, self.model.linear_out.weight.data)
            self._wp_copy_from_torch(self.b_out_wp, self.model.linear_out.bias.data)

    # ── Prediction (CUDA) ─────────────────────────────────────────────

    def predict(self, states: np.ndarray) -> np.ndarray:
        """V_θ(s) for batched states.  numpy in/out."""
        import torch
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(
                self._torch_device,
            )
            return self.model(t).cpu().numpy()

    # ── Pretraining (all parameters, plain MSE) ───────────────────────

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
        import torch

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
        self.sync_to_warp()
        return last_loss

    # ── Online update (hash grid only, one-sided MSE, captured graph) ──

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
        """Train hash-grid embeddings via the captured CUDA-graph training step.

        The forward + backward + Adam step is captured as a torch.cuda.CUDAGraph
        on first call.  Subsequent calls replay it once per batch with zero
        Python-side dispatch overhead.
        """
        import torch

        dev = self._torch_device
        states_t = torch.as_tensor(
            np.asarray(states, dtype=np.float32), device=dev,
        )
        targets_t = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).ravel(), device=dev,
        )
        if sample_weights is not None:
            weights_t = torch.as_tensor(
                np.asarray(sample_weights, dtype=np.float32).ravel(),
                device=dev,
            )
        else:
            weights_t = torch.ones(states_t.shape[0], device=dev)

        n = states_t.shape[0]
        bs = min(batch_size, n)
        num_batches = max(1, n // bs)
        used = num_batches * bs

        # Build / reuse the captured training step.
        self._ensure_online_graph(bs, learning_rate)
        self._reset_online_optimizer_state()

        for _epoch in range(max(epochs, 1)):
            perm = torch.randperm(n, device=dev)[:used]
            batch_idx = perm.reshape(num_batches, bs)

            for b in range(num_batches):
                idx = batch_idx[b]
                self._g_states.copy_(states_t[idx])
                self._g_targets.copy_(targets_t[idx])
                self._g_weights.copy_(weights_t[idx])
                self._online_graph.replay()

        self.sync_to_warp()

    # ── Save / load ───────────────────────────────────────────────────

    def save_weights_to_file(self, path) -> None:
        import torch
        torch.save(self.model.state_dict(), Path(path))

    def load_weights_from_file(self, path) -> None:
        import torch
        self.model.load_state_dict(
            torch.load(Path(path), map_location=self._torch_device, weights_only=True),
        )
        self._invalidate_online_graph()
        self.sync_to_warp()

    def copy_weights(self):
        """Deep copy the state dict (used for reset-to-pretrained)."""
        import torch
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def load_weights(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)
        self._invalidate_online_graph()
        self.sync_to_warp()


# ──────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────


class ValueGuidedMPPI:
    """MPPI with a learned hash-grid + MLP value as the terminal cost.

    Both the non-staged and staged rollout paths are CUDA-graph captured.
    Per optimize: data uploads (controls, knots, mean, offsets, noise) + final
    cost/knot readback.  Online training runs after the rollout (PyTorch CUDA
    graph) and syncs updated hashgrid weights into the Warp arrays the
    captured rollout graph reads from.

    Setting `use_density_guided=True` composes this with the staged
    density-guided rollout from `density_guided_mppi.py` — physics → KDE →
    inverse-density resample → reshuffle → knot regen, then learned terminal V.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        *,
        # Density-guided composition (off by default)
        use_density_guided: bool = False,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: Optional[np.ndarray] = None,
        inverse_density_power: float = 1.0,
        # MPPI core
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
        # Hash-grid learned value
        value_grid_min: float = -10.0,
        value_grid_max: float = 10.0,
        hashgrid_num_levels: int = 16,
        hashgrid_table_size: int = 4096,
        hashgrid_min_resolution: float = 16.0,
        hashgrid_max_resolution: float = 2048.0,
        # Online learning hyperparams
        online_learning_rate: float = 1e-3,
        online_update_epochs: int = 3,
        online_batch_size: int = 512,
        online_anchor_samples: int = 10000,
        online_new_state_weight: float = 1000.0,
        online_anchor_weight: float = 1.0,
        # Goal-anchor hyperparams (replay anchors path; off by default)
        goal_state: Optional[np.ndarray] = None,
        goal_value: float = 0.0,
        goal_weight: float = 200000.0,
        auto_pretrain_value: bool = False,
        disable_replay_anchors: bool = True,
    ) -> None:
        if spline_type != "zero":
            raise NotImplementedError(
                "CUDA graph capture requires spline_type='zero'. "
                f"Got '{spline_type}'."
            )

        self.task = task
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.temperature = temperature
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.interp_func = get_interp_func(spline_type)
        self.dt = task.dt
        self.ctrl_steps = int(round(plan_horizon / self.dt))

        self.tk = np.linspace(0.0, plan_horizon, num_knots, dtype=np.float32)
        self._tq_relative = np.linspace(
            0.0, plan_horizon, self.ctrl_steps, dtype=np.float32,
        )
        self.mean = np.zeros((num_knots, task.nu), dtype=np.float32)

        self.use_density_guided = use_density_guided
        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth
        self.inverse_density_power = inverse_density_power
        self.pretrained_value_key: Optional[str] = None

        self._init_stage_geometry()
        self._alloc_rollout_buffers()
        self._alloc_density_buffers(state_weight)
        self._alloc_value_buffers(hashgrid_num_levels)

        # Hash-grid + MLP learned value model.
        self.learned_value = _LearnedValueModel(
            state_dim=task.state_dim,
            grid_min=float(value_grid_min),
            grid_max=float(value_grid_max),
            num_levels=hashgrid_num_levels,
            table_size=hashgrid_table_size,
            min_resolution=hashgrid_min_resolution,
            max_resolution=hashgrid_max_resolution,
            seed=seed,
            device=str(self._device),
        )

        # Online-learning hyperparams.
        self.online_learning_rate = online_learning_rate
        self.online_update_epochs = max(int(online_update_epochs), 1)
        self.online_batch_size = max(int(online_batch_size), 1)
        self.online_anchor_samples = max(int(online_anchor_samples), 0)
        self.online_new_state_weight = float(online_new_state_weight)
        self.online_anchor_weight = float(online_anchor_weight)

        self.goal_state = (
            None if goal_state is None
            else np.asarray(goal_state, dtype=np.float32).reshape(1, -1)
        )
        self.goal_value = float(goal_value)
        self.goal_weight = float(goal_weight)
        self.disable_replay_anchors = bool(disable_replay_anchors)

        self.auto_pretrain_value = auto_pretrain_value
        self.learned_value_ready = False
        self._pretrained_value_weights = None
        self._state_sampler: Optional[Callable] = None
        self._target_function: Optional[Callable] = None
        self._pretrain_config = ValuePretrainConfig()

        # Pre-compute grid-range inverse for the encode kernels.
        self._grid_range_inv = 1.0 / max(
            self.learned_value.grid_max - self.learned_value.grid_min, 1e-6,
        )

        # CUDA graphs (built lazily; invalidated when learned_value_ready toggles).
        self._rollout_graph: Optional[wp.Graph] = None
        self._density_graph: Optional[wp.Graph] = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_stage_geometry(self) -> None:
        """Pre-compute stage boundaries and the zero-order-hold knot index map."""
        self._num_stages = int(math.floor(self.num_knots / self.num_knots_per_stage))
        self._timesteps_per_stage = (
            int(math.floor(self.ctrl_steps / self.num_knots)) * self.num_knots_per_stage
        )

        tk_rel = np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
        tq_rel = np.linspace(0.0, self.plan_horizon, self.ctrl_steps, dtype=np.float32)
        ki = np.searchsorted(tk_rel, tq_rel, side="right") - 1
        self._knot_indices_np = np.clip(ki, 0, self.num_knots - 1).astype(np.int32)

    def _alloc_rollout_buffers(self) -> None:
        """Buffers for the (non-staged) rollout graph."""
        self.warp_data = mjwarp.make_data(self.task.mj_model, nworld=self.num_samples)
        self._device = self.warp_data.ctrl.device
        N, nu = self.num_samples, self.task.nu

        # Build the task's State struct once; field references stay valid for
        # warp_data's lifetime, so the captured graph keeps seeing fresh values.
        self._task_state = self.task.make_state(self.warp_data)

        self._controls_wp = wp.zeros(
            (N, self.ctrl_steps, nu), dtype=wp.float32, device=self._device,
        )
        self._knots_wp = wp.zeros(
            (N, self.num_knots, nu), dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._mean_wp = wp.zeros(
            (self.num_knots, nu), dtype=wp.float32, device=self._device,
        )

        self._knot_indices_wp = wp.zeros(
            self.ctrl_steps, dtype=wp.int32, device=self._device,
        )
        self._knot_indices_wp.assign(self._knot_indices_np)

        self._u_min_wp = wp.zeros(nu, dtype=wp.float32, device=self._device)
        self._u_min_wp.assign(self.task.u_min.astype(np.float32))
        self._u_max_wp = wp.zeros(nu, dtype=wp.float32, device=self._device)
        self._u_max_wp.assign(self.task.u_max.astype(np.float32))

    def _alloc_density_buffers(self, state_weight: Optional[np.ndarray]) -> None:
        """Buffers for KDE / resample / reshuffle / per-stage noise (used by the staged path)."""
        N, nu = self.num_samples, self.task.nu
        D = self.task.state_dim

        self._states_wp = wp.zeros((N, D), dtype=wp.float32, device=self._device)
        self._density_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._indices_wp = wp.zeros(N, dtype=wp.int32, device=self._device)

        bw = np.full(D, self.kde_bandwidth, dtype=np.float32)
        self._bandwidth_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._bandwidth_wp.assign(bw)

        if state_weight is not None:
            sw = np.asarray(state_weight, dtype=np.float32).ravel()
        else:
            sw = np.ones(D, dtype=np.float32)
        if sw.shape[0] == 1 and D > 1:
            sw = np.full(D, sw[0], dtype=np.float32)
        self._state_weight_wp = wp.zeros(D, dtype=wp.float32, device=self._device)
        self._state_weight_wp.assign(sw)

        # Reshuffle scratch buffers (double-buffered for in-place gather+copy).
        self._tmp_controls = wp.zeros_like(self._controls_wp)
        self._tmp_knots = wp.zeros_like(self._knots_wp)
        self._tmp_costs = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._tmp_qpos = wp.zeros(
            (N, self.task.nq), dtype=wp.float32, device=self._device,
        )
        self._tmp_qvel = wp.zeros(
            (N, self.task.nv), dtype=wp.float32, device=self._device,
        )
        self._tmp_time = wp.zeros(N, dtype=wp.float32, device=self._device)

        # Per-stage trailing-knot noise.
        self._stage_noise: list[Optional[wp.array]] = []
        for n in range(max(self._num_stages - 1, 0)):
            k_start = (n + 1) * self.num_knots_per_stage
            remaining = self.num_knots - k_start
            if remaining > 0:
                self._stage_noise.append(wp.zeros(
                    (N, remaining, nu), dtype=wp.float32, device=self._device,
                ))
            else:
                self._stage_noise.append(None)

        n_boundaries = max(self._num_stages - 1, 1)
        self._resample_offsets_wp = wp.zeros(
            n_boundaries, dtype=wp.float32, device=self._device,
        )

    def _alloc_value_buffers(self, hashgrid_num_levels: int) -> None:
        """Intermediate buffers for hash-grid encode + 2-layer MLP inference."""
        N = self.num_samples
        hg_input_dim = hashgrid_num_levels * _HG_FEATURES_PER_LEVEL
        self._hg_num_levels = hashgrid_num_levels
        self._hg_input_dim = hg_input_dim

        self._features_wp = wp.zeros(
            (N, hg_input_dim), dtype=wp.float32, device=self._device,
        )
        self._hidden1_wp = wp.zeros(
            (N, _HG_HIDDEN_DIM), dtype=wp.float32, device=self._device,
        )
        self._hidden2_wp = wp.zeros(
            (N, _HG_HIDDEN_DIM), dtype=wp.float32, device=self._device,
        )

    # ------------------------------------------------------------------
    # CUDA graph builders — read as the algorithm
    # ------------------------------------------------------------------

    def _invalidate_graphs(self) -> None:
        """Force graph rebuild on next rollout (used when the *code path* changes)."""
        self._rollout_graph = None
        self._density_graph = None

    def _build_rollout_graph(self) -> None:
        """Capture (physics + per-step running cost + learned/task terminal) as a CUDA graph."""
        with capture_graph() as cap:
            self._running_costs_wp.zero_()
            self._terminal_costs_wp.zero_()
            for t in range(self.ctrl_steps):
                self._step_physics(t)
                self.task.launch_running_cost(
                    self._task_state, self.warp_data.ctrl,
                    self._running_costs_wp, self.dt,
                )
            self._emit_terminal_cost()
        self._rollout_graph = cap.graph

    def _build_density_graph(self) -> None:
        """Capture density-guided staged rollout + learned/task terminal."""
        with capture_graph() as cap:
            self._running_costs_wp.zero_()

            for n in range(self._num_stages - 1):
                self._step_physics_range(
                    n * self._timesteps_per_stage,
                    (n + 1) * self._timesteps_per_stage,
                )
                self._compute_kde_density()
                self._resample_inverse_density(stage_idx=n)
                self._reshuffle_by_indices()
                self._regenerate_trailing_knots(stage=n)

            self._step_physics_range(
                (self._num_stages - 1) * self._timesteps_per_stage,
                self.ctrl_steps,
            )

            self._terminal_costs_wp.zero_()
            self._emit_terminal_cost()
        self._density_graph = cap.graph

    # ------------------------------------------------------------------
    # Launch helpers — one wp.launch per algorithm operation
    # ------------------------------------------------------------------

    def _step_physics(self, t: int) -> None:
        wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
        mjwarp.step(self.task.model, self.warp_data)

    def _step_physics_range(self, t_start: int, t_end: int) -> None:
        for t in range(t_start, t_end):
            self._step_physics(t)
            self.task.launch_running_cost(
                self._task_state, self.warp_data.ctrl,
                self._running_costs_wp, self.dt,
            )

    def _compute_kde_density(self) -> None:
        self.task.extract_state(self._task_state, self._states_wp, self._state_weight_wp)
        wp.launch(
            kde_density, dim=self.num_samples,
            inputs=[self._states_wp, self._density_wp,
                    self._bandwidth_wp, self.num_samples, self.task.state_dim],
        )

    def _resample_inverse_density(self, stage_idx: int) -> None:
        wp.launch(
            resample_from_density, dim=1,
            inputs=[self._density_wp, self._indices_wp,
                    self._resample_offsets_wp, stage_idx,
                    self.num_samples, self.inverse_density_power],
        )

    def _reshuffle_by_indices(self) -> None:
        N, nu = self.num_samples, self.task.nu

        wp.launch(gather_3d_float, dim=N, inputs=[
            self._controls_wp, self._tmp_controls, self._indices_wp,
            self.ctrl_steps, nu])
        wp.copy(self._controls_wp, self._tmp_controls)

        wp.launch(gather_3d_float, dim=N, inputs=[
            self._knots_wp, self._tmp_knots, self._indices_wp,
            self.num_knots, nu])
        wp.copy(self._knots_wp, self._tmp_knots)

        wp.launch(gather_1d_float, dim=N, inputs=[
            self._running_costs_wp, self._tmp_costs, self._indices_wp])
        wp.copy(self._running_costs_wp, self._tmp_costs)

        wp.launch(gather_2d_float, dim=N, inputs=[
            self.warp_data.qpos, self._tmp_qpos, self._indices_wp, self.task.nq])
        wp.copy(self.warp_data.qpos, self._tmp_qpos)

        wp.launch(gather_2d_float, dim=N, inputs=[
            self.warp_data.qvel, self._tmp_qvel, self._indices_wp, self.task.nv])
        wp.copy(self.warp_data.qvel, self._tmp_qvel)

        wp.launch(gather_1d_float, dim=N, inputs=[
            self.warp_data.time, self._tmp_time, self._indices_wp])
        wp.copy(self.warp_data.time, self._tmp_time)

    def _regenerate_trailing_knots(self, stage: int) -> None:
        k_start = (stage + 1) * self.num_knots_per_stage
        remaining_K = self.num_knots - k_start
        if remaining_K <= 0 or self._stage_noise[stage] is None:
            return
        wp.launch(regenerate_knots, dim=self.num_samples, inputs=[
            self._knots_wp, self._mean_wp, self._stage_noise[stage],
            self._u_min_wp, self._u_max_wp, self.noise_level,
            k_start, remaining_K, self.task.nu])
        wp.launch(zero_order_interp, dim=self.num_samples, inputs=[
            self._knots_wp, self._controls_wp, self._knot_indices_wp,
            self.ctrl_steps, self.task.nu])

    def _emit_terminal_cost(self) -> None:
        """Write the terminal cost into self._terminal_costs_wp.

        learned_value_ready is checked at GRAPH BUILD TIME — once captured,
        the branch is fixed in the graph.  _invalidate_graphs() must be
        called whenever this flag changes.

        Learned heuristic V_θ(s_T) = MLP(hashgrid(s_T)):
            ψ  = hashgrid_encode(s_T)
            h₁ = swish(ψ · W₁ + b₁)
            h₂ = swish(h₁ · W₂ + b₂)
            V  = h₂ · W_out + b_out
        """
        if not self.learned_value_ready:
            self.task.launch_terminal_cost(self._task_state, self._terminal_costs_wp)
            return

        self.task.extract_state(self._task_state, self._states_wp, self._state_weight_wp)
        self._encode_hashgrid()
        self._mlp_swish_layer(
            self._features_wp, self._hidden1_wp,
            self.learned_value.W1_wp, self.learned_value.b1_wp,
            in_dim=self._hg_input_dim,
        )
        self._mlp_swish_layer(
            self._hidden1_wp, self._hidden2_wp,
            self.learned_value.W2_wp, self.learned_value.b2_wp,
            in_dim=_HG_HIDDEN_DIM,
        )
        self._mlp_linear_out(self._hidden2_wp, self._terminal_costs_wp)

    def _encode_hashgrid(self) -> None:
        """ψ = hashgrid_encode(s)  →  self._features_wp."""
        encode_kernel = (
            hashgrid_encode_2d if self.task.state_dim == 2 else hashgrid_encode_3d
        )
        wp.launch(
            encode_kernel, dim=self.num_samples,
            inputs=[
                self._states_wp,
                self.learned_value.embeddings_wp,
                self.learned_value.resolutions_wp,
                self._features_wp,
                self.learned_value.grid_min,
                self._grid_range_inv,
                self._hg_num_levels,
                self.learned_value.table_size,
            ],
        )

    def _mlp_swish_layer(
        self,
        inp: wp.array,
        out: wp.array,
        W: wp.array,
        b: wp.array,
        *,
        in_dim: int,
    ) -> None:
        """out = swish(inp · W + b).  Launch dim = N * out_dim."""
        wp.launch(
            dense_swish, dim=self.num_samples * _HG_HIDDEN_DIM,
            inputs=[inp, W, b, out, in_dim, _HG_HIDDEN_DIM],
        )

    def _mlp_linear_out(self, inp: wp.array, out: wp.array) -> None:
        """V = inp · W_out + b_out  (scalar per sample)."""
        wp.launch(
            dense_linear_1d, dim=self.num_samples,
            inputs=[inp, self.learned_value.W_out_wp, self.learned_value.b_out_wp,
                    out, _HG_HIDDEN_DIM],
        )

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def configure_value_pretraining(
        self, state_sampler, target_function, config=None,
    ):
        self._state_sampler = state_sampler
        self._target_function = target_function
        if config is not None:
            self._pretrain_config = config

    def pretrain_learned_value(self, force=False, verbose=True):
        if self.learned_value_ready and not force:
            return True
        if self._state_sampler is None or self._target_function is None:
            return False
        cfg = self._pretrain_config
        states = np.asarray(
            self._state_sampler(self.rng, cfg.sample_count), dtype=np.float32,
        )
        targets = np.asarray(
            self._target_function(states), dtype=np.float32,
        ).reshape(-1)
        if verbose:
            print(
                f"Pretraining learned value with {states.shape[0]} samples, "
                f"{cfg.epochs} epochs, batch_size={cfg.batch_size}."
            )
        self.learned_value.fit_pretrain(
            states, targets, epochs=cfg.epochs,
            batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
            verbose=verbose, print_every=cfg.print_every,
        )
        self._pretrained_value_weights = self.learned_value.copy_weights()
        self.learned_value_ready = True
        self._invalidate_graphs()
        return True

    def restore_pretrained_value(self):
        if self._pretrained_value_weights is None:
            return False
        self.learned_value.load_weights(self._pretrained_value_weights)
        self.learned_value_ready = True
        self._invalidate_graphs()
        return True

    def save_pretrained_value_weights(self, path):
        self.learned_value.save_weights_to_file(path)
        self._pretrained_value_weights = self.learned_value.copy_weights()
        self.learned_value_ready = True
        self._invalidate_graphs()

    def load_pretrained_value_weights(self, path):
        self.learned_value.load_weights_from_file(path)
        self._pretrained_value_weights = self.learned_value.copy_weights()
        self.learned_value_ready = True
        self._invalidate_graphs()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, seed=None, initial_knots=None, reset_value_to_pretrained=True):
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.tk = np.linspace(
            0.0, self.plan_horizon, self.num_knots, dtype=np.float32,
        )
        if initial_knots is None:
            self.mean = np.zeros(
                (self.num_knots, self.task.nu), dtype=np.float32,
            )
        else:
            knots = np.asarray(initial_knots, dtype=np.float32)
            expected = (self.num_knots, self.task.nu)
            if knots.shape != expected:
                raise ValueError(
                    f"initial_knots shape {knots.shape} != expected {expected}"
                )
            self.mean = knots.copy()
        if reset_value_to_pretrained and self._pretrained_value_weights is not None:
            # Reload pretrained weights without invalidating CUDA graphs.
            # load_state_dict copies values into the EXISTING parameter
            # tensors (same memory), and sync_to_warp copies into the
            # EXISTING warp arrays.  The captured graph reads from those
            # addresses, so replaying it picks up the new contents.
            self.learned_value.model.load_state_dict(self._pretrained_value_weights)
            self.learned_value.sync_to_warp()
            self.learned_value_ready = True

    def set_state_from_mj_data(self, mj_data: mujoco.MjData) -> None:
        nw = self.num_samples
        self.warp_data.qpos.assign(
            np.tile(mj_data.qpos.astype(np.float32), (nw, 1)),
        )
        self.warp_data.qvel.assign(
            np.tile(mj_data.qvel.astype(np.float32), (nw, 1)),
        )
        self.warp_data.time.assign(
            np.full(nw, mj_data.time, dtype=np.float32),
        )
        if mj_data.mocap_pos.shape[0] > 0:
            self.warp_data.mocap_pos.assign(
                np.tile(mj_data.mocap_pos.astype(np.float32), (nw, 1, 1))
                .reshape(nw, -1, 3)
            )
            self.warp_data.mocap_quat.assign(
                np.tile(mj_data.mocap_quat.astype(np.float32), (nw, 1, 1))
                .reshape(nw, -1, 4)
            )

    def _make_initial_state(self, mj_data: mujoco.MjData) -> dict:
        nw = self.num_samples
        return {
            "qpos": np.tile(mj_data.qpos.astype(np.float32), (nw, 1)),
            "qvel": np.tile(mj_data.qvel.astype(np.float32), (nw, 1)),
            "time": np.full(nw, mj_data.time, dtype=np.float32),
        }

    def _restore_state(self, state: dict) -> None:
        self.warp_data.qpos.assign(state["qpos"])
        self.warp_data.qvel.assign(state["qvel"])
        self.warp_data.time.assign(state["time"])

    def warm_start(self, current_time: float) -> None:
        new_tk = (
            np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
            + current_time
        )
        new_mean = self.interp_func(new_tk, self.tk, self.mean[None, ...])[0]
        self.tk = new_tk
        self.mean = new_mean

    def get_action(self, t: float) -> np.ndarray:
        return self.interp_func(
            np.array([t], dtype=np.float32),
            self.tk,
            self.mean[None, ...],
        )[0, 0, :]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_knots(self):
        """K_i = clip(μ + σ · 𝒩(0, I)),  controls = ZOH-interpolate(K_i)."""
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu),
        ).astype(np.float32)
        knots = self.mean + self.noise_level * noise
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def _rollout(self, controls: np.ndarray) -> np.ndarray:
        self._controls_wp.assign(controls)
        if self._rollout_graph is None:
            self._build_rollout_graph()
        wp.capture_launch(self._rollout_graph)
        return self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()

    def _density_rollout(
        self, controls: np.ndarray, knots: np.ndarray,
    ):
        if self._num_stages <= 1:
            return self._rollout(controls), knots

        self._controls_wp.assign(controls)
        self._knots_wp.assign(knots)
        self._mean_wp.assign(self.mean)

        n_boundaries = self._num_stages - 1
        offsets = self.rng.uniform(
            0.0, 1.0 / self.num_samples, size=n_boundaries,
        ).astype(np.float32)
        self._resample_offsets_wp.assign(offsets)

        for n in range(n_boundaries):
            buf = self._stage_noise[n]
            if buf is not None:
                k_start = (n + 1) * self.num_knots_per_stage
                remaining = self.num_knots - k_start
                noise = self.rng.standard_normal(
                    (self.num_samples, remaining, self.task.nu),
                ).astype(np.float32)
                buf.assign(noise)

        if self._density_graph is None:
            self._build_density_graph()
        wp.capture_launch(self._density_graph)

        costs = self._running_costs_wp.numpy() + self._terminal_costs_wp.numpy()
        final_knots = self._knots_wp.numpy()
        return costs, final_knots

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def _update_weights(
        self, total_costs: np.ndarray, knots: np.ndarray,
    ) -> np.ndarray:
        """Softmax mean update:  w_i = exp(-J_i / λ) / Z;  μ ← Σ_i w_i · K_i."""
        shifted = -total_costs / self.temperature
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0)
        return total_costs

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        if self.auto_pretrain_value and not self.learned_value_ready:
            self.pretrain_learned_value(force=False, verbose=False)

        current_state_vec = self._current_state_vector(mj_data)

        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)

        best_cost = np.inf
        if self.iterations == 1:
            knots, controls = self._sample_knots()
            if self.use_density_guided:
                total_costs, final_knots = self._density_rollout(controls, knots)
                self._update_weights(total_costs, final_knots)
            else:
                total_costs = self._rollout(controls)
                self._update_weights(total_costs, knots)
            best_cost = float(np.min(total_costs))
        else:
            init_state = self._make_initial_state(mj_data)
            for _ in range(self.iterations):
                self._restore_state(init_state)
                knots, controls = self._sample_knots()
                if self.use_density_guided:
                    total_costs, final_knots = self._density_rollout(controls, knots)
                    self._update_weights(total_costs, final_knots)
                else:
                    total_costs = self._rollout(controls)
                    self._update_weights(total_costs, knots)
                best_cost = min(best_cost, float(np.min(total_costs)))

        self._update_learned_value_online(
            states=current_state_vec[None, :],
            values=np.array([best_cost], dtype=np.float32),
        )
        return self.mean

    # ------------------------------------------------------------------
    # Learned-value online update (CPU side; GPU work is graph-captured)
    # ------------------------------------------------------------------

    def _current_state_vector(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Extract the same low-D state that task.extract_state extracts on GPU,
        but from a single MjData on CPU (used as the online-learning target site)."""
        if isinstance(self.task.state_dim, int) and self.task.state_dim == 0:
            raise RuntimeError("task does not define state_dim; cannot use learned value")
        # Mirror the per-task extraction in numpy. The GPU kernel reads from
        # warp_data fields; the Python side reads from mj_data fields with the
        # same semantics (qpos[:state_dim] for u_point/ant; site_xpos[ee_id]
        # for ur5e).
        D = self.task.state_dim
        # u_point_mass / ant: state = qpos[:D]
        if hasattr(self.task, "ee_vel_sensor_adr"):
            # ur5e branch — state = site_xpos[end_effector_pos_id]
            return mj_data.site_xpos[self.task.end_effector_pos_id, :D].astype(np.float32)
        return mj_data.qpos[:D].astype(np.float32)

    def _update_learned_value_online(
        self, states: np.ndarray, values: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]

        # Build the training batch once so anchor targets are fixed to
        # pre-update values.  Recomputing anchors each epoch would let
        # them drift along with the model rather than resist change.
        all_states = [states]
        all_targets = [values]
        all_weights = [
            np.full(states.shape[0], self.online_new_state_weight, dtype=np.float32),
        ]

        if not self.disable_replay_anchors:
            if self.online_anchor_samples > 0:
                anchors = self.rng.uniform(
                    self.learned_value.grid_min,
                    self.learned_value.grid_max,
                    size=(self.online_anchor_samples, self.task.state_dim),
                ).astype(np.float32)
                anchor_targets = self.learned_value.predict(anchors)
                all_states.append(anchors)
                all_targets.append(anchor_targets)
                all_weights.append(
                    np.full(anchors.shape[0], self.online_anchor_weight, dtype=np.float32),
                )
            if self.goal_state is not None:
                all_states.append(self.goal_state)
                all_targets.append(np.array([self.goal_value], dtype=np.float32))
                all_weights.append(np.array([self.goal_weight], dtype=np.float32))

        self.learned_value.fit_online(
            np.concatenate(all_states, axis=0),
            np.concatenate(all_targets, axis=0),
            epochs=self.online_update_epochs,
            batch_size=min(
                self.online_batch_size,
                sum(s.shape[0] for s in all_states),
            ),
            learning_rate=self.online_learning_rate,
            sample_weights=np.concatenate(all_weights, axis=0),
        )
        self.learned_value_ready = True
