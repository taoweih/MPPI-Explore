"""UR5e reach task for mujoco_warp."""

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT


@wp.struct
class State:
    """Warp State struct bundling the warp_data fields read by UR5e's cost kernels."""

    sensordata: wp.array2d(dtype=wp.float32)
    site_xpos:  wp.array2d(dtype=wp.vec3f)
    xpos:       wp.array2d(dtype=wp.vec3f)


class UR5e(Task):
    """Reach task for the UR5e robot arm."""

    state_dim = 3  # KDE / learned-value state = end-effector (x, y, z) from site_xpos

    def __init__(self, planning_dt: float = 0.02, sim_dt: float = 0.01) -> None:
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ur5e/scene.xml")
        mj_model.opt.timestep = planning_dt
        super().__init__(mj_model, sim_dt=sim_dt, trace_sites=("attachment_site",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

        ee_vel_sensor_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_linvel"
        )
        self.ee_vel_sensor_adr = int(self.mj_model.sensor_adr[ee_vel_sensor_id])

    @wp.kernel
    def running_cost(
        x:       State,
        u:       wp.array2d(dtype=wp.float32),   # (nworld, nu) — unused; signature for MPC standard form
        out:     wp.array1d(dtype=wp.float32),   # (nworld,) accumulated
        vel_adr: int,
        dt:      float,
    ) -> None:
        i = wp.tid()
        v = wp.vec3f(
            x.sensordata[i, vel_adr + 0],
            x.sensordata[i, vel_adr + 1],
            x.sensordata[i, vel_adr + 2],
        )
        out[i] += dt * wp.sqrt(wp.dot(v, v))

    @wp.kernel
    def terminal_cost(
        x:       State,
        out:     wp.array1d(dtype=wp.float32),
        ee_id:   int,
        goal_id: int,
    ) -> None:
        i = wp.tid()
        diff = x.site_xpos[i, ee_id] - x.xpos[i, goal_id]
        out[i] = wp.sqrt(wp.dot(diff, diff))

    @wp.kernel
    def state_extract(
        x:      State,
        weight: wp.array1d(dtype=wp.float32),    # (state_dim,)
        out:    wp.array2d(dtype=wp.float32),    # (nworld, state_dim)
        ee_id:  int,
    ) -> None:
        i = wp.tid()
        v = x.site_xpos[i, ee_id]
        out[i, 0] = v[0] * weight[0]
        out[i, 1] = v[1] * weight[1]
        out[i, 2] = v[2] * weight[2]

    def make_state(self, warp_data) -> State:
        s = State()
        s.sensordata = warp_data.sensordata
        s.site_xpos = warp_data.site_xpos
        s.xpos = warp_data.xpos
        return s

    def launch_running_cost(self, state, ctrl_arr, out_wp, dt):
        wp.launch(self.running_cost, dim=out_wp.shape[0],
                  inputs=[state, ctrl_arr, out_wp, self.ee_vel_sensor_adr, dt])

    def launch_terminal_cost(self, state, out_wp):
        wp.launch(self.terminal_cost, dim=out_wp.shape[0],
                  inputs=[state, out_wp,
                          self.end_effector_pos_id, self.goal_pos_id])

    def extract_state(self, state, out_wp, weight_wp):
        wp.launch(self.state_extract, dim=out_wp.shape[0],
                  inputs=[state, weight_wp, out_wp, self.end_effector_pos_id])

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        site_xpos = data_np["site_xpos"]
        xpos = data_np["xpos"]
        ee_pos = site_xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
