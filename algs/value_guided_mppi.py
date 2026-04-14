"""Value-guided MPPI with hash-grid + MLP learned value function.

The learned value heuristic uses a multi-resolution hash grid encoder
followed by a small MLP (matching the hydrax NeuralNet architecture).

Inference runs as a sequence of warp kernels, keeping the full rollout
(including terminal cost blend) capturable as a CUDA graph.  Training
uses PyTorch on CPU and syncs weights to warp arrays between optimize
calls.

Pretraining: updates **all** parameters (hash grid + MLP).
Online:      updates **hash grid only** (MLP frozen).
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

from tasks.task_base import Task
from utils.spline import get_interp_func
from utils.warp_kernels import (
    extract_float_slice,
    extract_vec3_row,
    kde_density,
    resample_from_density,
    gather_1d_float,
    gather_2d_float,
    gather_3d_float,
    regenerate_knots,
    zero_order_interp,
    hashgrid_encode_2d,
    hashgrid_encode_3d,
    dense_swish,
    dense_linear_1d,
)


# ---------------------------------------------------------------------------
# Pretraining config
# ---------------------------------------------------------------------------

@dataclass
class ValuePretrainConfig:
    sample_count: int = 100000
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    print_every: int = 50


# ---------------------------------------------------------------------------
# Hash-grid defaults (matching hydrax / instant-NGP)
# ---------------------------------------------------------------------------

_HG_FEATURES_PER_LEVEL = 2
_HG_HIDDEN_DIM = 64


# ---------------------------------------------------------------------------
# Warp-backed hash-grid + MLP learned value model
# ---------------------------------------------------------------------------

class _LearnedValueModel:
    """Hash grid + MLP value model: PyTorch on CUDA for training, warp for graph inference.

    The model lives on the same CUDA device as the warp simulation.
    After each training call the weights are copied GPU-to-GPU into
    pre-allocated warp arrays so the captured CUDA graph sees the
    latest parameters without any CPU round-trip.
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

        # ── PyTorch model (CUDA, for training) ────────────────────────
        torch.manual_seed(seed)
        self.model = HashGridMLP(
            din=state_dim, grid_min=grid_min, grid_max=grid_max,
            num_levels=num_levels, table_size=table_size,
            min_resolution=min_resolution, max_resolution=max_resolution,
        )
        self.model.to(self._torch_device)
        self.model.eval()

        # ── Warp arrays (GPU, for inference inside CUDA graph) ────────
        total_emb = num_levels * table_size * _HG_FEATURES_PER_LEVEL
        self.embeddings_wp = wp.zeros(total_emb, dtype=wp.float32, device=device)

        self.resolutions_wp = wp.zeros(
            num_levels, dtype=wp.float32, device=device,
        )
        self.resolutions_wp.assign(
            self.model.resolutions.cpu().numpy().astype(np.float32),
        )

        # MLP weights (stored as in_dim × out_dim for the warp kernel).
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

        # ── Online training CUDA graph (lazily built) ─────────────────
        self._online_graph = None
        self._online_opt = None
        self._online_bs: int = 0
        self._g_states = None
        self._g_targets = None
        self._g_weights = None

        self.sync_to_warp()

    # ── Online training CUDA graph ───────────────────────────────────

    def _invalidate_online_graph(self) -> None:
        self._online_graph = None
        self._online_opt = None

    def _ensure_online_graph(self, batch_size: int, learning_rate: float) -> None:
        """Lazily build the torch CUDA graph for one online training step."""
        import torch

        if self._online_graph is not None and self._online_bs == batch_size:
            return

        dev = self._torch_device

        # Freeze MLP before capture.
        for p in self.model.mlp_params():
            p.requires_grad_(False)
        self.model.train()

        self._online_opt = torch.optim.Adam(
            self.model.hashgrid_params(), lr=learning_rate,
            capturable=True, foreach=False,
        )
        self._online_bs = batch_size

        # Static I/O buffers (addresses fixed for the graph).
        self._g_states = torch.zeros(batch_size, self.state_dim, device=dev)
        self._g_targets = torch.zeros(batch_size, device=dev)
        self._g_weights = torch.zeros(batch_size, device=dev)

        # Warm-up on a side stream so the allocator is primed.
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

        # Capture one training step.
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
        """Zero Adam momentum so each online call starts fresh."""
        import torch
        if self._online_opt is None:
            return
        for state in self._online_opt.state.values():
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if isinstance(state["step"], torch.Tensor):
                state["step"].zero_()

    # ── Sync (GPU-to-GPU via warp/torch interop) ─────────────────────

    @staticmethod
    def _wp_copy_from_torch(dst, src_tensor) -> None:
        """GPU-to-GPU copy from a contiguous PyTorch CUDA tensor into a warp array."""
        src_flat = src_tensor.detach().contiguous().view(-1)
        src_wp = wp.from_torch(src_flat, dtype=wp.float32)
        wp.copy(dst, src_wp)

    def sync_to_warp(self) -> None:
        """Copy all PyTorch weights to warp arrays (GPU-to-GPU)."""
        import torch
        with torch.no_grad():
            self._wp_copy_from_torch(
                self.embeddings_wp, self.model.embeddings.data,
            )
            # PyTorch Linear stores (out, in); warp kernel expects (in, out).
            self._wp_copy_from_torch(
                self.W1_wp, self.model.linear1.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self.b1_wp, self.model.linear1.bias.data)
            self._wp_copy_from_torch(
                self.W2_wp, self.model.linear2.weight.data.T.contiguous(),
            )
            self._wp_copy_from_torch(self.b2_wp, self.model.linear2.bias.data)
            self._wp_copy_from_torch(
                self.W_out_wp, self.model.linear_out.weight.data,
            )
            self._wp_copy_from_torch(
                self.b_out_wp, self.model.linear_out.bias.data,
            )

    # ── Prediction (CUDA) ─────────────────────────────────────────────

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict values.  states: (B, din) → (B,).  numpy in/out."""
        import torch
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(
                self._torch_device,
            )
            return self.model(t).cpu().numpy()

    # ── Pretraining (all parameters) ──────────────────────────────────

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

    # ── Online update (hash grid only, one-sided loss, CUDA-graph) ──

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
        """Train hash-grid embeddings with CUDA-graph-captured steps.

        The forward + backward + Adam step is captured as a
        ``torch.cuda.CUDAGraph`` on first call.  Subsequent calls
        replay the graph with zero Python dispatch overhead per batch.
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
        """Return a deep copy of the state dict (for reset-to-pretrained)."""
        import torch
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def load_weights(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)
        self._invalidate_online_graph()
        self.sync_to_warp()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ValueGuidedMPPI:
    """MPPI augmented with a learned hash-grid + MLP value heuristic.

    Both the non-staged and staged rollout paths are fully CUDA-graph
    captured.  The only host-device syncs per optimize call are:

      * Initial data upload (controls, knots, noise, mean, weights)
      * Final cost readback (one float per sample)

    Online learned value training runs on the CPU after the rollout and syncs
    the updated hash-grid weights to the GPU for the next call.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        *,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: Optional[np.ndarray] = None,
        inverse_density_power: float = 1.0,
        use_density_guided: bool = False,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
        # GPU state extraction
        state_dim: int = 2,
        state_source_field: str = "qpos",
        state_source_start: int = 0,
        state_source_body_id: Optional[int] = None,
        # Hash-grid learned value model
        value_grid_min: float = -10.0,
        value_grid_max: float = 10.0,
        hashgrid_num_levels: int = 16,
        hashgrid_table_size: int = 4096,
        hashgrid_min_resolution: float = 16.0,
        hashgrid_max_resolution: float = 2048.0,
        # Online learning
        online_learning_rate: float = 1e-3,
        online_update_epochs: int = 3,
        online_batch_size: int = 512,
        online_anchor_samples: int = 10000,
        online_new_state_weight: float = 1000.0,
        online_anchor_weight: float = 1.0,
        # Goal
        goal_state: Optional[np.ndarray] = None,
        goal_value: float = 0.0,
        goal_weight: float = 200000.0,
        auto_pretrain_value: bool = False,
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
        self.state_dim = state_dim
        self._state_source_field = state_source_field
        self._state_source_start = state_source_start
        self._state_source_body_id = state_source_body_id
        self.pretrained_value_key: Optional[str] = None

        # Stage geometry.
        self._num_stages = int(math.floor(num_knots / num_knots_per_stage))
        self._timesteps_per_stage = (
            int(math.floor(self.ctrl_steps / num_knots)) * num_knots_per_stage
        )

        # Pre-compute zero-order-hold knot indices.
        tk_rel = np.linspace(0.0, plan_horizon, num_knots, dtype=np.float32)
        tq_rel = np.linspace(0.0, plan_horizon, self.ctrl_steps, dtype=np.float32)
        ki = np.searchsorted(tk_rel, tq_rel, side="right") - 1
        self._knot_indices_np = np.clip(ki, 0, num_knots - 1).astype(np.int32)

        # ── GPU arrays ────────────────────────────────────────────────
        self.warp_data = mjwarp.make_data(task.mj_model, nworld=num_samples)
        self._device = self.warp_data.ctrl.device

        self._controls_wp = wp.zeros(
            (num_samples, self.ctrl_steps, task.nu),
            dtype=wp.float32, device=self._device,
        )
        self._knots_wp = wp.zeros(
            (num_samples, num_knots, task.nu),
            dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._terminal_costs_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._mean_wp = wp.zeros(
            (num_knots, task.nu), dtype=wp.float32, device=self._device,
        )
        self._knot_indices_wp = wp.zeros(
            self.ctrl_steps, dtype=wp.int32, device=self._device,
        )
        self._knot_indices_wp.assign(self._knot_indices_np)

        # State extraction / KDE buffers.
        self._states_wp = wp.zeros(
            (num_samples, state_dim), dtype=wp.float32, device=self._device,
        )
        self._density_wp = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._indices_wp = wp.zeros(
            num_samples, dtype=wp.int32, device=self._device,
        )

        # Hash-grid + MLP intermediate buffers.
        hg_input_dim = hashgrid_num_levels * _HG_FEATURES_PER_LEVEL
        self._hg_num_levels = hashgrid_num_levels
        self._hg_input_dim = hg_input_dim
        self._features_wp = wp.zeros(
            (num_samples, hg_input_dim), dtype=wp.float32, device=self._device,
        )
        self._hidden1_wp = wp.zeros(
            (num_samples, _HG_HIDDEN_DIM), dtype=wp.float32, device=self._device,
        )
        self._hidden2_wp = wp.zeros(
            (num_samples, _HG_HIDDEN_DIM), dtype=wp.float32, device=self._device,
        )


        # Expand state_weight.
        if state_weight is not None:
            sw = np.asarray(state_weight, dtype=np.float32).ravel()
        else:
            sw = np.ones(state_dim, dtype=np.float32)
        if sw.shape[0] == 1 and state_dim > 1:
            sw = np.full(state_dim, sw[0], dtype=np.float32)
        self._state_weight_wp = wp.zeros(
            state_dim, dtype=wp.float32, device=self._device,
        )
        self._state_weight_wp.assign(sw)

        bw = np.full(state_dim, kde_bandwidth, dtype=np.float32)
        self._bandwidth_wp = wp.zeros(
            state_dim, dtype=wp.float32, device=self._device,
        )
        self._bandwidth_wp.assign(bw)

        self._u_min_wp = wp.zeros(task.nu, dtype=wp.float32, device=self._device)
        self._u_min_wp.assign(task.u_min.astype(np.float32))
        self._u_max_wp = wp.zeros(task.nu, dtype=wp.float32, device=self._device)
        self._u_max_wp.assign(task.u_max.astype(np.float32))

        # Gather temp buffers (staged path).
        self._tmp_controls = wp.zeros_like(self._controls_wp)
        self._tmp_knots = wp.zeros_like(self._knots_wp)
        self._tmp_costs = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )
        self._tmp_qpos = wp.zeros(
            (num_samples, task.nq), dtype=wp.float32, device=self._device,
        )
        self._tmp_qvel = wp.zeros(
            (num_samples, task.nv), dtype=wp.float32, device=self._device,
        )
        self._tmp_time = wp.zeros(
            num_samples, dtype=wp.float32, device=self._device,
        )

        # Per-stage-boundary noise + resample offsets.
        self._stage_noise: list[Optional[wp.array]] = []
        for n in range(max(self._num_stages - 1, 0)):
            k_start = (n + 1) * num_knots_per_stage
            remaining = num_knots - k_start
            if remaining > 0:
                self._stage_noise.append(
                    wp.zeros(
                        (num_samples, remaining, task.nu),
                        dtype=wp.float32, device=self._device,
                    )
                )
            else:
                self._stage_noise.append(None)

        n_boundaries = max(self._num_stages - 1, 1)
        self._resample_offsets_wp = wp.zeros(
            n_boundaries, dtype=wp.float32, device=self._device,
        )

        # ── Hash-grid + MLP learned value model ──────────────────────
        self.learned_value = _LearnedValueModel(
            state_dim=state_dim,
            grid_min=float(value_grid_min),
            grid_max=float(value_grid_max),
            num_levels=hashgrid_num_levels,
            table_size=hashgrid_table_size,
            min_resolution=hashgrid_min_resolution,
            max_resolution=hashgrid_max_resolution,
            seed=seed,
            device=str(self._device),
        )

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

        self.auto_pretrain_value = auto_pretrain_value
        self.learned_value_ready = False
        self._pretrained_value_weights = None
        self._state_sampler: Optional[Callable] = None
        self._target_function: Optional[Callable] = None
        self._pretrain_config = ValuePretrainConfig()

        # Pre-compute grid range inverse for the warp encode kernel.
        self._grid_range_inv = 1.0 / max(
            self.learned_value.grid_max - self.learned_value.grid_min, 1e-6,
        )

        # CUDA graphs (built lazily, invalidated when learned_value_ready changes).
        self._rollout_graph: Optional[wp.Graph] = None
        self._density_graph: Optional[wp.Graph] = None

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _invalidate_graphs(self) -> None:
        self._rollout_graph = None
        self._density_graph = None

    def _launch_extract_state(self) -> None:
        src = getattr(self.warp_data, self._state_source_field)
        if self._state_source_body_id is not None:
            wp.launch(
                extract_vec3_row, dim=self.num_samples,
                inputs=[src, self._states_wp, self._state_weight_wp,
                        self._state_source_body_id],
            )
        else:
            wp.launch(
                extract_float_slice, dim=self.num_samples,
                inputs=[src, self._states_wp, self._state_weight_wp,
                        self._state_source_start, self.state_dim],
            )

    def _emit_terminal_cost(self) -> None:
        """Emit GPU kernels for terminal cost.

        When learned_value_ready the learned heuristic fully replaces the
        task's terminal cost.  Otherwise falls back to the task default.

        Called inside graph capture — adds kernel launches to the graph.
        """
        N = self.num_samples
        if self.learned_value_ready:
            self._launch_extract_state()

            # Hash-grid encode.
            encode_kernel = (
                hashgrid_encode_2d if self.state_dim == 2
                else hashgrid_encode_3d
            )
            wp.launch(
                encode_kernel, dim=N,
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

            # MLP layer 1: (N, 32) → (N, 64) with swish.
            wp.launch(
                dense_swish, dim=N * _HG_HIDDEN_DIM,
                inputs=[
                    self._features_wp,
                    self.learned_value.W1_wp,
                    self.learned_value.b1_wp,
                    self._hidden1_wp,
                    self._hg_input_dim,
                    _HG_HIDDEN_DIM,
                ],
            )

            # MLP layer 2: (N, 64) → (N, 64) with swish.
            wp.launch(
                dense_swish, dim=N * _HG_HIDDEN_DIM,
                inputs=[
                    self._hidden1_wp,
                    self.learned_value.W2_wp,
                    self.learned_value.b2_wp,
                    self._hidden2_wp,
                    _HG_HIDDEN_DIM,
                    _HG_HIDDEN_DIM,
                ],
            )

            # MLP output: (N, 64) → (N,) = learned heuristic as terminal cost.
            wp.launch(
                dense_linear_1d, dim=N,
                inputs=[
                    self._hidden2_wp,
                    self.learned_value.W_out_wp,
                    self.learned_value.b_out_wp,
                    self._terminal_costs_wp,
                    _HG_HIDDEN_DIM,
                ],
            )
        else:
            self.task.launch_terminal_cost(
                self.warp_data, self._terminal_costs_wp,
            )

    # ------------------------------------------------------------------
    # CUDA graph builders
    # ------------------------------------------------------------------

    def _build_rollout_graph(self) -> None:
        wp.capture_begin(force_module_load=False)
        try:
            self._running_costs_wp.zero_()
            self._terminal_costs_wp.zero_()
            for t in range(self.ctrl_steps):
                wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                mjwarp.step(self.task.model, self.warp_data)
                self.task.launch_running_cost(
                    self.warp_data, self._running_costs_wp, self.dt,
                )
            self._emit_terminal_cost()
            self._rollout_graph = wp.capture_end()
        except Exception:
            wp.capture_end()
            raise

    def _build_density_graph(self) -> None:
        N = self.num_samples
        nu = self.task.nu

        wp.capture_begin(force_module_load=False)
        try:
            self._running_costs_wp.zero_()

            for n in range(self._num_stages - 1):
                t_start = n * self._timesteps_per_stage
                t_end = (n + 1) * self._timesteps_per_stage

                for t in range(t_start, t_end):
                    wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                    mjwarp.step(self.task.model, self.warp_data)
                    self.task.launch_running_cost(
                        self.warp_data, self._running_costs_wp, self.dt,
                    )

                # KDE + resample.
                self._launch_extract_state()
                wp.launch(kde_density, dim=N, inputs=[
                    self._states_wp, self._density_wp,
                    self._bandwidth_wp, N, self.state_dim])
                wp.launch(resample_from_density, dim=1, inputs=[
                    self._density_wp, self._indices_wp,
                    self._resample_offsets_wp, n, N,
                    self.inverse_density_power])

                # Gather.
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
                    self.warp_data.qpos, self._tmp_qpos, self._indices_wp,
                    self.task.nq])
                wp.copy(self.warp_data.qpos, self._tmp_qpos)
                wp.launch(gather_2d_float, dim=N, inputs=[
                    self.warp_data.qvel, self._tmp_qvel, self._indices_wp,
                    self.task.nv])
                wp.copy(self.warp_data.qvel, self._tmp_qvel)
                wp.launch(gather_1d_float, dim=N, inputs=[
                    self.warp_data.time, self._tmp_time, self._indices_wp])
                wp.copy(self.warp_data.time, self._tmp_time)

                # Regenerate + re-interpolate.
                k_start = (n + 1) * self.num_knots_per_stage
                remaining_K = self.num_knots - k_start
                if remaining_K > 0 and self._stage_noise[n] is not None:
                    wp.launch(regenerate_knots, dim=N, inputs=[
                        self._knots_wp, self._mean_wp,
                        self._stage_noise[n], self._u_min_wp,
                        self._u_max_wp, self.noise_level,
                        k_start, remaining_K, nu])
                    wp.launch(zero_order_interp, dim=N, inputs=[
                        self._knots_wp, self._controls_wp,
                        self._knot_indices_wp, self.ctrl_steps, nu])

            # Final stage.
            t_start = (self._num_stages - 1) * self._timesteps_per_stage
            for t in range(t_start, self.ctrl_steps):
                wp.copy(self.warp_data.ctrl, self._controls_wp[:, t, :])
                mjwarp.step(self.task.model, self.warp_data)
                self.task.launch_running_cost(
                    self.warp_data, self._running_costs_wp, self.dt,
                )

            # Terminal cost (with or without value blend).
            self._terminal_costs_wp.zero_()
            self._emit_terminal_cost()

            self._density_graph = wp.capture_end()
        except Exception:
            wp.capture_end()
            raise

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
            # load_state_dict copies values into existing parameter tensors
            # (same memory addresses), and sync_to_warp copies into existing
            # Warp arrays.  The captured graphs read from these addresses, so
            # replaying them with updated contents is safe.
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
    ) -> np.ndarray:
        if self._num_stages <= 1:
            return self._rollout(controls)

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
    # Learned value helpers
    # ------------------------------------------------------------------

    def _current_state_vector(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Extract state vector from a single MjData (CPU-side)."""
        field = getattr(mj_data, self._state_source_field)
        if self._state_source_body_id is not None:
            return field[self._state_source_body_id].astype(np.float32)
        start = self._state_source_start
        return field[start : start + self.state_dim].astype(np.float32)

    def _update_learned_value_online(
        self, states: np.ndarray, values: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]

        # Build the training batch once so anchor targets are fixed to
        # pre-update values.  This prevents drift: if anchors were
        # recomputed each epoch they would track the changing model
        # instead of resisting unwanted changes.
        all_states = [states]
        all_targets = [values]
        all_weights = [
            np.full(states.shape[0], self.online_new_state_weight, dtype=np.float32),
        ]

        if self.online_anchor_samples > 0:
            anchors = self.rng.uniform(
                self.learned_value.grid_min,
                self.learned_value.grid_max,
                size=(self.online_anchor_samples, self.state_dim),
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
