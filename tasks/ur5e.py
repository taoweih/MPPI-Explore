"""UR5e reach task for mujoco_warp."""

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT


@wp.kernel
def _running_cost(
    sensordata: wp.array2d(dtype=wp.float32),  # (nworld, nsensordata)
    out:        wp.array1d(dtype=wp.float32),  # (nworld,) accumulated
    vel_adr:    int,
    dt:         float,
) -> None:
    i = wp.tid()
    vx = sensordata[i, vel_adr + 0]
    vy = sensordata[i, vel_adr + 1]
    vz = sensordata[i, vel_adr + 2]
    out[i] += dt * 1.0 * wp.sqrt(vx * vx + vy * vy + vz * vz)


@wp.kernel
def _terminal_cost(
    site_xpos: wp.array2d(dtype=wp.vec3f),  # (nworld, nsite)
    xpos:      wp.array2d(dtype=wp.vec3f),  # (nworld, nbody)
    out:       wp.array1d(dtype=wp.float32),
    ee_id:     int,
    goal_id:   int,
) -> None:
    i = wp.tid()
    ee   = site_xpos[i, ee_id]
    goal = xpos[i, goal_id]
    dx = ee[0] - goal[0]
    dy = ee[1] - goal[1]
    dz = ee[2] - goal[2]
    out[i] = 1.0 * wp.sqrt(dx * dx + dy * dy + dz * dz)


class UR5e(Task):
    """Reach task for the UR5e robot arm."""

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

    def launch_running_cost(self, warp_data, out_wp: wp.array, dt: float) -> None:
        wp.launch(
            _running_cost,
            dim=out_wp.shape[0],
            inputs=[warp_data.sensordata, out_wp, self.ee_vel_sensor_adr, dt],
        )

    def launch_terminal_cost(self, warp_data, out_wp: wp.array) -> None:
        wp.launch(
            _terminal_cost,
            dim=out_wp.shape[0],
            inputs=[
                warp_data.site_xpos, warp_data.xpos, out_wp,
                self.end_effector_pos_id, self.goal_pos_id,
            ],
        )

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        site_xpos = data_np["site_xpos"]
        xpos = data_np["xpos"]
        ee_pos = site_xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
