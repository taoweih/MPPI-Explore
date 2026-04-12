"""Base task interface for mujoco_warp-based MPPI."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import mujoco
import mujoco_warp as mjwarp
import warp as wp

# Package root for model paths
ROOT = str(Path(__file__).parent.parent.absolute())

# Fields extractable from mujoco_warp Data (used for success metrics and traces)
ALL_FIELDS = ("xpos", "qpos", "qvel", "xquat", "sensordata", "site_xpos", "qfrc_constraint")


class Task(ABC):
    """Abstract task defining dynamics and GPU cost kernels for mujoco_warp MPPI.

    Subclasses implement launch_running_cost and launch_terminal_cost as thin
    wrappers around module-level @wp.kernel functions, writing cost values
    directly into warp GPU arrays with no CPU involvement during rollouts.

    success_function remains numpy-based for use in benchmark metrics only.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        sim_dt: Optional[float] = None,
        trace_sites: Optional[Sequence[str]] = None,
        trace_bodies: Optional[Sequence[str]] = None,
    ) -> None:
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjwarp.put_model(mj_model)

        self.u_min = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -np.inf,
        ).astype(np.float32)
        self.u_max = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            np.inf,
        ).astype(np.float32)

        # Planning timestep (captured by put_model for GPU rollouts)
        self.dt = mj_model.opt.timestep
        # Simulation timestep for CPU-side mj_step (finer for collision accuracy)
        self.sim_dt = sim_dt if sim_dt is not None else self.dt
        # Restore mj_model timestep to sim_dt so CPU mj_step uses it
        mj_model.opt.timestep = self.sim_dt
        self.nu = mj_model.nu
        self.nq = mj_model.nq
        self.nv = mj_model.nv

        self.trace_site_ids: list[int] = []
        self.trace_body_ids: list[int] = []

        for site_name in trace_sites or []:
            site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise ValueError(f"Unknown trace site: {site_name}")
            self.trace_site_ids.append(site_id)

        for body_name in trace_bodies or []:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"Unknown trace body: {body_name}")
            self.trace_body_ids.append(body_id)

    @abstractmethod
    def launch_running_cost(
        self,
        warp_data: mjwarp.Data,
        out_wp: wp.array,
        dt: float,
    ) -> None:
        """Launch warp kernel to accumulate dt * running_cost into out_wp.

        out_wp shape: (nworld,). Values are accumulated (+=), so the caller
        must zero-initialise before the rollout loop.
        """

    @abstractmethod
    def launch_terminal_cost(
        self,
        warp_data: mjwarp.Data,
        out_wp: wp.array,
    ) -> None:
        """Launch warp kernel to write terminal_cost into out_wp.

        out_wp shape: (nworld,). Values are overwritten (=).
        """

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        """CPU-side success metric used only in benchmarks and interactive viz.

        data_np: dict of numpy arrays extracted from a single MjData instance.
        Returns shape (1,) or scalar.
        """
        raise NotImplementedError

    def get_trace_positions(self, data_np: dict) -> np.ndarray:
        """Return trace points for each world, shape (nworld, ntrace, 3)."""
        nworld = None
        if data_np:
            first_val = next(iter(data_np.values()))
            nworld = int(first_val.shape[0])

        chunks = []

        if self.trace_site_ids:
            if "site_xpos" not in data_np:
                raise KeyError("site_xpos required for trace site visualization")
            chunks.append(data_np["site_xpos"][:, self.trace_site_ids, :])

        if self.trace_body_ids:
            if "xpos" not in data_np:
                raise KeyError("xpos required for trace body visualization")
            chunks.append(data_np["xpos"][:, self.trace_body_ids, :])

        if not chunks:
            if nworld is None:
                nworld = 1
            return np.zeros((nworld, 0, 3), dtype=np.float32)

        return np.concatenate(chunks, axis=1).astype(np.float32)
