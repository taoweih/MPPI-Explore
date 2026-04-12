"""U-shaped point mass task for mujoco_warp."""

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT

# mujoco_warp field dtypes (verified at runtime):
#   xpos, site_xpos, mocap_pos  -> vec3f  shape (nworld, nbody/nsite/nmocap)
#   xquat, mocap_quat           -> quatf  shape (nworld, nbody/nmocap)
#   ctrl, qpos, qvel, sensordata -> float32 shape (nworld, dim)


@wp.kernel
def _running_cost(
    ctrl: wp.array2d(dtype=wp.float32),  # (nworld, nu)
    out:  wp.array1d(dtype=wp.float32),  # (nworld,) accumulated
    dt:   float,
) -> None:
    i = wp.tid()
    c0 = ctrl[i, 0]
    c1 = ctrl[i, 1]
    out[i] += dt * 1.0 * wp.sqrt(c0 * c0 + c1 * c1)


@wp.kernel
def _terminal_cost(
    xpos:    wp.array2d(dtype=wp.vec3f),  # (nworld, nbody)
    out:     wp.array1d(dtype=wp.float32),
    ee_id:   int,
    goal_id: int,
) -> None:
    i = wp.tid()
    ee   = xpos[i, ee_id]
    goal = xpos[i, goal_id]
    dx = ee[0] - goal[0]
    dy = ee[1] - goal[1]
    dz = ee[2] - goal[2]
    out[i] = 1.0 * wp.sqrt(dx * dx + dy * dy + dz * dz)


class UPointMass(Task):
    """Point mass navigation task."""

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

    def launch_running_cost(self, warp_data, out_wp: wp.array, dt: float) -> None:
        wp.launch(
            _running_cost,
            dim=out_wp.shape[0],
            inputs=[warp_data.ctrl, out_wp, dt],
        )

    def launch_terminal_cost(self, warp_data, out_wp: wp.array) -> None:
        wp.launch(
            _terminal_cost,
            dim=out_wp.shape[0],
            inputs=[warp_data.xpos, out_wp, self.end_effector_pos_id, self.goal_pos_id],
        )

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        xpos = data_np["xpos"]
        ee_pos = xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
