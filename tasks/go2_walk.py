"""Unitree Go2 walk task for mujoco_warp."""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from tasks.task_base import Task, ROOT


PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966


@wp.func
def _wrap_to_pi(angle: float) -> float:
    return angle - TWO_PI * wp.floor((angle + PI) / TWO_PI)


@wp.func
def _foot_step_height(
    time: float,
    duty_ratio: float,
    cadence: float,
    amplitude: float,
    phase: float,
) -> float:
    angle = _wrap_to_pi(time * TWO_PI * cadence + PI - TWO_PI * phase)
    if duty_ratio < 1.0:
        angle = angle * 0.5 / (1.0 - duty_ratio)
    else:
        angle = 0.0
    return amplitude * wp.cos(wp.clamp(angle, -HALF_PI, HALF_PI))


@wp.func
def _body_xy_velocity(q: wp.quatf, vx: float, vy: float, vz: float) -> wp.vec2f:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # R(q)^T * world_velocity, using MuJoCo's raw wxyz quaternion layout.
    body_vx = (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + (2.0 * (x * y + w * z)) * vy
        + (2.0 * (x * z - w * y)) * vz
    )
    body_vy = (
        (2.0 * (x * y - w * z)) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + (2.0 * (y * z + w * x)) * vz
    )
    return wp.vec2f(body_vx, body_vy)


@wp.func
def _body_z_velocity(q: wp.quatf, vx: float, vy: float, vz: float) -> float:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    return (
        (2.0 * (x * z + w * y)) * vx
        + (2.0 * (y * z - w * x)) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz
    )


@wp.func
def _yaw_from_quat(q: wp.quatf) -> float:
    return wp.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


@wp.func
def _ramp_target(target: float, ramp: float) -> float:
    value = target * ramp
    if value < target:
        return value
    return target


@wp.func
def _joint_target_low(j: int) -> float:
    if j % 3 == 0:
        return -0.5
    if j % 3 == 1:
        return 0.4
    return -2.3


@wp.func
def _joint_target_high(j: int) -> float:
    if j % 3 == 0:
        return 0.5
    if j % 3 == 1:
        return 1.4
    if j < 6:
        return -0.85
    return -1.3


@wp.func
def _torque_limit(j: int) -> float:
    if j % 3 == 2:
        return 45.43
    return 24.0


@wp.struct
class State:
    """Warp State struct bundling fields read by Go2's cost kernels."""

    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    xpos: wp.array2d(dtype=wp.vec3f)
    xquat: wp.array2d(dtype=wp.quatf)
    site_xpos: wp.array2d(dtype=wp.vec3f)
    time: wp.array1d(dtype=wp.float32)


class Go2Walk(Task):
    """Unitree Go2 walk environment ported from DIAL-MPC's unitree_go2_walk."""

    dial_env_name = "unitree_go2_walk"
    state_dim = 3  # KDE / learned-value state = base (x, y, z)

    def __init__(
        self,
        planning_dt: float = 0.02,
        sim_dt: float = 0.02,
        target_vx: float = 0.8,
        target_vy: float = 0.0,
        target_vyaw: float = 0.0,
        gait: str = "trot",
    ) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/go2/mjx_scene_force.xml"
        )
        mj_model.opt.timestep = planning_dt
        super().__init__(
            mj_model,
            sim_dt=sim_dt,
            trace_sites=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            trace_bodies=("base",),
        )

        self.base_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.foot_site_ids = tuple(
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        )
        if self.base_body_id < 0 or any(site_id < 0 for site_id in self.foot_site_ids):
            raise ValueError("Go2 model is missing required base body or foot sites.")

        self.target_vx = float(target_vx)
        self.target_vy = float(target_vy)
        self.target_vyaw = float(target_vyaw)
        self.target_height = 0.30
        self.foot_radius = 0.0175
        self.kp = 30.0
        self.kd = 0.0
        self.action_scale = 1.0
        self.ramp_up_time = 1.0

        gait_phases = {
            "stand": (0.0, 0.0, 0.0, 0.0),
            "walk": (0.0, 0.5, 0.75, 0.25),
            "trot": (0.0, 0.5, 0.5, 0.0),
            "canter": (0.0, 0.33, 0.33, 0.66),
            "gallop": (0.0, 0.05, 0.4, 0.35),
        }
        gait_params = {
            "stand": (1.0, 1.0, 0.0),
            "walk": (0.75, 1.0, 0.08),
            "trot": (0.45, 2.0, 0.08),
            "canter": (0.4, 4.0, 0.06),
            "gallop": (0.3, 3.5, 0.10),
        }
        if gait not in gait_phases:
            raise ValueError(f"Unknown Go2 gait {gait!r}.")
        self.gait_phase = gait_phases[gait]
        self.gait_duty_ratio, self.gait_cadence, self.gait_amplitude = gait_params[gait]

        self.home_qpos = np.asarray(self.mj_model.key_qpos[0], dtype=np.float32).copy()
        self.default_ctrl = np.zeros(self.nu, dtype=np.float32)
        self.joint_target_low = np.asarray(
            [
                -0.5, 0.4, -2.3,
                -0.5, 0.4, -2.3,
                -0.5, 0.4, -2.3,
                -0.5, 0.4, -2.3,
            ],
            dtype=np.float32,
        )
        self.joint_target_high = np.asarray(
            [
                0.5, 1.4, -0.85,
                0.5, 1.4, -0.85,
                0.5, 1.4, -1.3,
                0.5, 1.4, -1.3,
            ],
            dtype=np.float32,
        )
        self.torque_limit = np.asarray(
            [
                24.0, 24.0, 45.43,
                24.0, 24.0, 45.43,
                24.0, 24.0, 45.43,
                24.0, 24.0, 45.43,
            ],
            dtype=np.float32,
        )

        # Match DIAL-MPC's action space: normalized leg actions in [-1, 1].
        # The Warp control hook maps these actions to PD torques before stepping.
        self.u_min = -np.ones(self.nu, dtype=np.float32)
        self.u_max = np.ones(self.nu, dtype=np.float32)

    @wp.kernel
    def action_to_torque(
        x: State,
        action: wp.array2d(dtype=wp.float32),
        ctrl: wp.array2d(dtype=wp.float32),
        kp: float,
        kd: float,
        action_scale: float,
    ) -> None:
        i = wp.tid()
        for j in range(12):
            lo = _joint_target_low(j)
            hi = _joint_target_high(j)
            act_normalized = (action[i, j] * action_scale + 1.0) * 0.5
            joint_target = lo + act_normalized * (hi - lo)
            joint_target = wp.clamp(joint_target, lo, hi)

            q = x.qpos[i, 7 + j]
            qd = x.qvel[i, 6 + j]
            tau = kp * (joint_target - q) - kd * qd
            limit = _torque_limit(j)
            ctrl[i, j] = wp.clamp(tau, -limit, limit)

    @wp.kernel
    def running_cost(
        x: State,
        u: wp.array2d(dtype=wp.float32),
        out: wp.array1d(dtype=wp.float32),
        base_id: int,
        fl_site_id: int,
        fr_site_id: int,
        rl_site_id: int,
        rr_site_id: int,
        target_vx: float,
        target_vy: float,
        target_vyaw: float,
        target_height: float,
        ramp_up_time: float,
        duty_ratio: float,
        cadence: float,
        amplitude: float,
        fl_phase: float,
        fr_phase: float,
        rl_phase: float,
        rr_phase: float,
        dt: float,
    ) -> None:
        i = wp.tid()

        q = x.xquat[i, base_id]
        base_pos = x.xpos[i, base_id]

        time = x.time[i] - dt
        if time < 0.0:
            time = 0.0
        ramp = wp.clamp(time / ramp_up_time, 0.0, 1.0)
        vel_tar_x = _ramp_target(target_vx, ramp)
        vel_tar_y = _ramp_target(target_vy, ramp)
        ang_vel_tar_z = _ramp_target(target_vyaw, ramp)

        body_v = _body_xy_velocity(q, x.qvel[i, 0], x.qvel[i, 1], x.qvel[i, 2])
        vel_cost = (
            (body_v[0] - vel_tar_x) * (body_v[0] - vel_tar_x)
            + (body_v[1] - vel_tar_y) * (body_v[1] - vel_tar_y)
        )

        body_wz = _body_z_velocity(q, x.qvel[i, 3], x.qvel[i, 4], x.qvel[i, 5])
        yaw_rate_cost = (body_wz - ang_vel_tar_z) * (body_wz - ang_vel_tar_z)

        qx = q[1]
        qy = q[2]
        qz = q[3]
        up_x = 2.0 * (qx * qz + q[0] * qy)
        up_y = 2.0 * (qy * qz - q[0] * qx)
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright_cost = up_x * up_x + up_y * up_y + (up_z - 1.0) * (up_z - 1.0)

        yaw_tar = target_vyaw * time
        d_yaw = _wrap_to_pi(_yaw_from_quat(q) - yaw_tar)
        yaw_cost = d_yaw * d_yaw

        height_error = base_pos[2] - target_height
        height_cost = height_error * height_error

        fl_target = _foot_step_height(time, duty_ratio, cadence, amplitude, fl_phase)
        fr_target = _foot_step_height(time, duty_ratio, cadence, amplitude, fr_phase)
        rl_target = _foot_step_height(time, duty_ratio, cadence, amplitude, rl_phase)
        rr_target = _foot_step_height(time, duty_ratio, cadence, amplitude, rr_phase)

        fl_error = (fl_target - x.site_xpos[i, fl_site_id][2]) / 0.05
        fr_error = (fr_target - x.site_xpos[i, fr_site_id][2]) / 0.05
        rl_error = (rl_target - x.site_xpos[i, rl_site_id][2]) / 0.05
        rr_error = (rr_target - x.site_xpos[i, rr_site_id][2]) / 0.05
        gait_cost = (
            fl_error * fl_error
            + fr_error * fr_error
            + rl_error * rl_error
            + rr_error * rr_error
        )

        energy_cost = float(0.0)
        for j in range(12):
            positive_power = u[i, j] * x.qvel[i, 6 + j] / 160.0
            if positive_power > 0.0:
                energy_cost += positive_power * positive_power

        out[i] += dt * (
            0.1 * gait_cost
            + 0.5 * upright_cost
            + 0.3 * yaw_cost
            + 1.0 * vel_cost
            + 1.0 * yaw_rate_cost
            + 1.0 * height_cost
            + 0.0 * energy_cost
        )

    @wp.kernel
    def terminal_cost(
        x: State,
        out: wp.array1d(dtype=wp.float32),
        base_id: int,
        target_vx: float,
        target_vy: float,
        target_height: float,
    ) -> None:
        i = wp.tid()
        q = x.xquat[i, base_id]
        out[i] = 0.0

    @wp.kernel
    def state_extract(
        x: State,
        weight: wp.array1d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
        base_id: int,
    ) -> None:
        i = wp.tid()
        base_pos = x.xpos[i, base_id]
        out[i, 0] = base_pos[0] * weight[0]
        out[i, 1] = base_pos[1] * weight[1]
        out[i, 2] = base_pos[2] * weight[2]

    def make_state(self, warp_data) -> State:
        s = State()
        s.qpos = warp_data.qpos
        s.qvel = warp_data.qvel
        s.xpos = warp_data.xpos
        s.xquat = warp_data.xquat
        s.site_xpos = warp_data.site_xpos
        s.time = warp_data.time
        return s

    def launch_step_control(self, state, action_arr, ctrl_arr):
        wp.launch(
            self.action_to_torque,
            dim=ctrl_arr.shape[0],
            inputs=[
                state,
                action_arr,
                ctrl_arr,
                self.kp,
                self.kd,
                self.action_scale,
            ],
        )

    def apply_control_cpu(self, mj_data: mujoco.MjData, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.u_min, self.u_max)
        action_normalized = (action * self.action_scale + 1.0) * 0.5
        joint_target = self.joint_target_low + action_normalized * (
            self.joint_target_high - self.joint_target_low
        )
        joint_target = np.clip(
            joint_target,
            self.joint_target_low,
            self.joint_target_high,
        )
        q = np.asarray(mj_data.qpos[7:19], dtype=np.float32)
        qd = np.asarray(mj_data.qvel[6:18], dtype=np.float32)
        tau = self.kp * (joint_target - q) - self.kd * qd
        mj_data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)

    def launch_running_cost(self, state, ctrl_arr, out_wp, dt):
        wp.launch(
            self.running_cost,
            dim=out_wp.shape[0],
            inputs=[
                state,
                ctrl_arr,
                out_wp,
                self.base_body_id,
                self.foot_site_ids[0],
                self.foot_site_ids[1],
                self.foot_site_ids[2],
                self.foot_site_ids[3],
                self.target_vx,
                self.target_vy,
                self.target_vyaw,
                self.target_height,
                self.ramp_up_time,
                self.gait_duty_ratio,
                self.gait_cadence,
                self.gait_amplitude,
                self.gait_phase[0],
                self.gait_phase[1],
                self.gait_phase[2],
                self.gait_phase[3],
                dt,
            ],
        )

    def launch_terminal_cost(self, state, out_wp):
        wp.launch(
            self.terminal_cost,
            dim=out_wp.shape[0],
            inputs=[
                state,
                out_wp,
                self.base_body_id,
                self.target_vx,
                self.target_vy,
                self.target_height,
            ],
        )

    def extract_state(self, state, out_wp, weight_wp):
        wp.launch(
            self.state_extract,
            dim=out_wp.shape[0],
            inputs=[state, weight_wp, out_wp, self.base_body_id],
        )

    def extract_state_cpu(self, mj_data) -> np.ndarray:
        return np.asarray(mj_data.xpos[self.base_body_id, :3], dtype=np.float32)

    def reset_to_home(self, mj_data: mujoco.MjData) -> None:
        mj_data.qpos[:] = self.home_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = self.default_ctrl
        mj_data.time = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        qvel = data_np["qvel"]
        xquat = data_np["xquat"][:, self.base_body_id, :]
        xpos = data_np["xpos"][:, self.base_body_id, :]

        w = xquat[:, 0]
        qx = xquat[:, 1]
        qy = xquat[:, 2]
        qz = xquat[:, 3]

        vx = qvel[:, 0]
        vy = qvel[:, 1]
        vz = qvel[:, 2]
        body_vx = (
            (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
            + (2.0 * (qx * qy + w * qz)) * vy
            + (2.0 * (qx * qz - w * qy)) * vz
        )
        body_vy = (
            (2.0 * (qx * qy - w * qz)) * vx
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
            + (2.0 * (qy * qz + w * qx)) * vz
        )

        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        height_error = xpos[:, 2] - self.target_height
        return (
            (body_vx - self.target_vx) ** 2
            + (body_vy - self.target_vy) ** 2
            + height_error**2
            + 0.5 * (1.0 - upright) ** 2
        )
