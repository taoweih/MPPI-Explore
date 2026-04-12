"""Ant locomotion task for mujoco_warp."""

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT

# mujoco_warp stores xquat as wp.quatf with component order matching MuJoCo's
# raw memory layout (w, x, y, z) → quatf.x=w, quatf.y=x, quatf.z=y, quatf.w=z.


@wp.kernel
def _running_cost(
    qvel:  wp.array2d(dtype=wp.float32),  # (nworld, nv)
    xquat: wp.array2d(dtype=wp.quatf),    # (nworld, nbody)
    out:   wp.array1d(dtype=wp.float32),  # (nworld,) accumulated
    ee_id: int,
    dt:    float,
) -> None:
    i = wp.tid()
    # Speed: xy velocity magnitude
    vx = qvel[i, 0]
    vy = qvel[i, 1]
    speed = wp.sqrt(vx * vx + vy * vy)

    # Orientation: torso upright.
    # MuJoCo (w,x,y,z) maps to quatf as (.x=w, .y=mjx, .z=mjy, .w=mjz).
    q = xquat[i, ee_id]
    mj_x = q[1]   # MuJoCo's x component
    mj_y = q[2]   # MuJoCo's y component
    upright = 1.0 - 2.0 * (mj_x * mj_x + mj_y * mj_y)
    orientation_cost = (1.0 - upright) * (1.0 - upright)

    out[i] += dt * (1.0 * speed + 1.0 * orientation_cost)


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


class Ant(Task):
    """Ant locomotion task."""

    def __init__(self, planning_dt: float = 0.02, sim_dt: float = 0.01) -> None:
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ant/scene.xml")
        mj_model.opt.timestep = planning_dt
        super().__init__(mj_model, sim_dt=sim_dt, trace_sites=("torso_site",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

    def launch_running_cost(self, warp_data, out_wp: wp.array, dt: float) -> None:
        wp.launch(
            _running_cost,
            dim=out_wp.shape[0],
            inputs=[warp_data.qvel, warp_data.xquat, out_wp, self.end_effector_pos_id, dt],
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
