"""Ant locomotion task for mujoco_warp."""

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT

# mujoco_warp stores xquat as wp.quatf with component order matching MuJoCo's
# raw memory layout (w, x, y, z) → quatf.x=w, quatf.y=x, quatf.z=y, quatf.w=z.


@wp.struct
class State:
    """Warp State struct bundling the warp_data fields read by Ant's cost kernels."""

    qpos:  wp.array2d(dtype=wp.float32)
    qvel:  wp.array2d(dtype=wp.float32)
    xpos:  wp.array2d(dtype=wp.vec3f)
    xquat: wp.array2d(dtype=wp.quatf)


class Ant(Task):
    """Ant locomotion task."""

    state_dim = 2  # KDE / learned-value state = (x, y) from qpos[:2]

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

    @wp.kernel
    def running_cost(
        x:     State,
        u:     wp.array2d(dtype=wp.float32),     # (nworld, nu) — unused; signature for MPC standard form
        out:   wp.array1d(dtype=wp.float32),     # (nworld,) accumulated
        ee_id: int,
        dt:    float,
    ) -> None:
        i = wp.tid()
        # Speed: xy velocity magnitude
        v_xy = wp.vec2f(x.qvel[i, 0], x.qvel[i, 1])
        speed = wp.sqrt(wp.dot(v_xy, v_xy))

        # Orientation: torso upright.
        # MuJoCo (w,x,y,z) maps to quatf as (.x=w, .y=mjx, .z=mjy, .w=mjz).
        q = x.xquat[i, ee_id]
        q_xy = wp.vec2f(q[1], q[2])
        upright = 1.0 - 2.0 * wp.dot(q_xy, q_xy)
        orientation_cost = (1.0 - upright) * (1.0 - upright)

        out[i] += dt * (speed + orientation_cost)

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
        s.qvel = warp_data.qvel
        s.xpos = warp_data.xpos
        s.xquat = warp_data.xquat
        return s

    def launch_running_cost(self, state, ctrl_arr, out_wp, dt):
        wp.launch(self.running_cost, dim=out_wp.shape[0],
                  inputs=[state, ctrl_arr, out_wp, self.end_effector_pos_id, dt])

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
