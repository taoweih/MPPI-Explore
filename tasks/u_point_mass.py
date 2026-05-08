"""U-shaped point mass task for mujoco_warp."""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT

# mujoco_warp field dtypes (verified at runtime):
#   xpos, site_xpos, mocap_pos  -> vec3f  shape (nworld, nbody/nsite/nmocap)
#   xquat, mocap_quat           -> quatf  shape (nworld, nbody/nmocap)
#   ctrl, qpos, qvel, sensordata -> float32 shape (nworld, dim)


@wp.struct
class State:
    """Warp State struct bundling the warp_data fields read by this task's cost kernels.

    Field references stay valid for the lifetime of warp_data, so a single
    State built by `UPointMass.make_state` is reused for every kernel launch.
    """

    qpos: wp.array2d(dtype=wp.float32)
    xpos: wp.array2d(dtype=wp.vec3f)


class UPointMass(Task):
    """Point mass navigation task."""

    state_dim = 2  # KDE / learned-value state = (x, y) from qpos[:2]

    def __init__(self, planning_dt: float = 0.02, sim_dt: float = 0.01) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/u_point_mass/scene.xml"
        )
        mj_model.opt.timestep = planning_dt
        super().__init__(mj_model, sim_dt=sim_dt, trace_bodies=("point_mass",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "point_mass"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

    @wp.kernel
    def running_cost(
        x:   State,                              # (unused — pure control penalty)
        u:   wp.array2d(dtype=wp.float32),       # (nworld, nu)
        out: wp.array1d(dtype=wp.float32),       # (nworld,) accumulated
        dt:  float,
    ) -> None:
        i = wp.tid()
        c = wp.vec2f(u[i, 0], u[i, 1])
        out[i] += dt * wp.sqrt(wp.dot(c, c))

    @wp.kernel
    def terminal_cost(
        x:       State,
        out:     wp.array1d(dtype=wp.float32),
        ee_id:   int,
        goal_id: int,
    ) -> None:
        i = wp.tid()
        diff = x.xpos[i, ee_id] - x.xpos[i, goal_id]
        out[i] = wp.sqrt(wp.dot(diff, diff))

    @wp.kernel
    def state_extract(
        x:      State,
        weight: wp.array1d(dtype=wp.float32),    # (state_dim,)
        out:    wp.array2d(dtype=wp.float32),    # (nworld, state_dim)
    ) -> None:
        i = wp.tid()
        out[i, 0] = x.qpos[i, 0] * weight[0]
        out[i, 1] = x.qpos[i, 1] * weight[1]

    def make_state(self, warp_data) -> State:
        s = State()
        s.qpos = warp_data.qpos
        s.xpos = warp_data.xpos
        return s

    def launch_running_cost(self, state, ctrl_arr, out_wp, dt):
        wp.launch(self.running_cost, dim=out_wp.shape[0],
                  inputs=[state, ctrl_arr, out_wp, dt])

    def launch_terminal_cost(self, state, out_wp):
        wp.launch(self.terminal_cost, dim=out_wp.shape[0],
                  inputs=[state, out_wp,
                          self.end_effector_pos_id, self.goal_pos_id])

    def extract_state(self, state, out_wp, weight_wp):
        wp.launch(self.state_extract, dim=out_wp.shape[0],
                  inputs=[state, weight_wp, out_wp])

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        xpos = data_np["xpos"]
        ee_pos = xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
