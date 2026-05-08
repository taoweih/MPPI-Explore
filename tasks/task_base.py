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

    Each subclass defines a module-level ``@wp.struct`` named ``State`` that
    bundles the warp_data fields its cost kernels read.  Cost signatures
    follow the standard MPC convention:

        running_cost(x, u, ..., dt)
        terminal_cost(x, ...)

    where ``x`` is the State struct and ``u`` is the per-world control array.
    The subclass implements:

        make_state(warp_data) -> State    # populate the struct from warp_data
        launch_running_cost(state, ctrl_arr, out_wp, dt)
        launch_terminal_cost(state, out_wp)

    The kernel definitions sit inside the class next to their launchers so
    the cost math is in one place.  All cost values are written into warp GPU
    arrays with no CPU involvement during rollouts.

    Variants that need a low-D state representation for KDE / learned-value
    lookup also override ``state_dim`` and ``extract_state``.

    success_function remains numpy-based for use in benchmark metrics only.
    """

    # Subclasses override if used with density-guided / value-guided variants.
    state_dim: int = 0

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
    def make_state(self, warp_data: mjwarp.Data):
        """Return a Warp State struct populated with references to warp_data fields.

        Built once by the controller after `mjwarp.make_data` and reused on
        every cost kernel launch.  Because struct fields are array references
        (pointers), the struct sees current warp_data contents on every replay.
        """

    @abstractmethod
    def launch_running_cost(
        self,
        state,
        ctrl_arr: wp.array,
        out_wp: wp.array,
        dt: float,
    ) -> None:
        """Accumulate dt * running_cost(x, u) into out_wp (shape (nworld,), `+=`)."""

    @abstractmethod
    def launch_terminal_cost(
        self,
        state,
        out_wp: wp.array,
    ) -> None:
        """Overwrite terminal_cost(x) into out_wp (shape (nworld,), `=`)."""

    def extract_state(
        self,
        state,
        out_wp: wp.array,
        weight_wp: wp.array,
    ) -> None:
        """Write each world's low-D state vector into `out_wp` (shape (nworld, state_dim)).

        Reads from the State struct produced by `make_state`.  ``weight_wp`` is
        a length-`state_dim` per-component scaling applied during extraction
        (used for KDE-bandwidth weighting).

        Default implementation raises — only needed by density-guided and
        value-guided variants.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_state; "
            "this task cannot be used with density-guided or value-guided MPPI."
        )

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
