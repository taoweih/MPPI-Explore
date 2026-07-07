"""Shared MPPI plumbing.

`BaseMPPI` holds everything that all MPPI variants do identically:

  - GPU buffer allocation (mujoco_warp Data, controls, running/terminal costs)
  - State synchronisation between the CPU `MjData` and the GPU rollout worlds
  - Warm-starting the knot trajectory between replans

The algorithm-specific pieces — sampling, rollout, weighting, and how
`optimize()` strings them together — live in the subclass files
(`mppi.py`, `density_guided_mppi.py`, `value_guided_mppi.py`).  See the
subclass docstring for the math.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from tasks.task_base import Task
from utils.spline import get_interp_func


class BaseMPPI(ABC):
    """Shared MPPI plumbing.  Subclasses implement `optimize` and `rollout`."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "quadratic", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ) -> None:
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
        self.mean = self._make_default_mean()

        self._alloc_rollout_buffers()

    # ══════════════════════════════════════════════════════════════════
    # Subclass contract
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Run one optimization step; returns updated mean knots."""

    # `rollout(...)` is also subclass-defined but its signature varies
    # across variants (full-horizon vs staged), so it isn't declared here.

    # ══════════════════════════════════════════════════════════════════
    # GPU buffer allocation
    # ══════════════════════════════════════════════════════════════════

    def _alloc_rollout_buffers(self) -> None:
        """Pre-allocate GPU buffers shared by every variant.

        Buffer addresses are fixed for the lifetime of the controller — the
        captured CUDA graph references these exact addresses, so we never
        reassign them (use `.assign()` for re-uploads).
        """
        self.warp_data = mjwarp.make_data(self.task.mj_model, nworld=self.num_samples)
        self._device = self.warp_data.ctrl.device

        # Build the task's State struct once; field references stay valid for
        # warp_data's lifetime, so the captured graph keeps seeing fresh values.
        self._task_state = self.task.make_state(self.warp_data)

        N, nu = self.num_samples, self.task.nu
        self._controls_wp = wp.zeros(
            (N, self.ctrl_steps, nu), dtype=wp.float32, device=self._device,
        )
        self._running_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)
        self._terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device=self._device)

    # ══════════════════════════════════════════════════════════════════
    # State management
    # ══════════════════════════════════════════════════════════════════

    def _make_default_mean(self) -> np.ndarray:
        default_ctrl = getattr(self.task, "default_ctrl", None)
        if default_ctrl is None:
            return np.zeros((self.num_knots, self.task.nu), dtype=np.float32)

        ctrl = np.asarray(default_ctrl, dtype=np.float32)
        expected = (self.task.nu,)
        if ctrl.shape != expected:
            raise ValueError(
                f"{type(self.task).__name__}.default_ctrl shape {ctrl.shape} "
                f"!= expected {expected}"
            )
        return np.tile(ctrl, (self.num_knots, 1))

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
    ) -> None:
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.tk = np.linspace(
            0.0, self.plan_horizon, self.num_knots, dtype=np.float32,
        )
        if initial_knots is None:
            self.mean = self._make_default_mean()
        else:
            knots = np.asarray(initial_knots, dtype=np.float32)
            expected = (self.num_knots, self.task.nu)
            if knots.shape != expected:
                raise ValueError(
                    f"initial_knots shape {knots.shape} != expected {expected}"
                )
            self.mean = knots.copy()

    def set_state_from_mj_data(self, mj_data: mujoco.MjData) -> None:
        """Tile the CPU MjData state across all sample worlds on GPU."""
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
        """Tiled initial state arrays (CPU-only; used by multi-iteration optimize)."""
        nw = self.num_samples
        return {
            "qpos": np.tile(mj_data.qpos.astype(np.float32), (nw, 1)),
            "qvel": np.tile(mj_data.qvel.astype(np.float32), (nw, 1)),
            "time": np.full(nw, mj_data.time, dtype=np.float32),
        }

    def _restore_state(self, state: dict) -> None:
        """Reload tiled initial state into the GPU rollout worlds."""
        self.warp_data.qpos.assign(state["qpos"])
        self.warp_data.qvel.assign(state["qvel"])
        self.warp_data.time.assign(state["time"])

    # ══════════════════════════════════════════════════════════════════
    # Warm start & action query
    # ══════════════════════════════════════════════════════════════════

    def warm_start(self, current_time: float) -> None:
        """Shift the knot trajectory to start at `current_time`; re-interpolate."""
        new_tk = (
            np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
            + current_time
        )
        query_tk = np.clip(new_tk, float(self.tk[0]), float(self.tk[-1]))
        new_mean = self.interp_func(query_tk, self.tk, self.mean[None, ...])[0]
        self.tk = new_tk
        self.mean = new_mean

    def get_action(self, t: float) -> np.ndarray:
        """Interpolated control at time `t` from the current mean knots."""
        return self.interp_func(
            np.array([t], dtype=np.float32),
            self.tk,
            self.mean[None, ...],
        )[0, 0, :]
