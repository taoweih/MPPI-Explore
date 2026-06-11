"""Unitree H1 crate-pushing task for mujoco_warp."""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from tasks.go2_walk import (
    PI,
    _body_xy_velocity,
    _body_z_velocity,
    _foot_step_height,
    _ramp_target,
    _wrap_to_pi,
    _yaw_from_quat,
)
from tasks.task_base import ROOT, Task


@wp.func
def _up_vector_cost(q: wp.quatf) -> float:
    up_x = 2.0 * (q[1] * q[3] + q[0] * q[2])
    up_y = 2.0 * (q[2] * q[3] - q[0] * q[1])
    up_z = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return up_x * up_x + up_y * up_y + (up_z - 1.0) * (up_z - 1.0)


@wp.struct
class H1PushCrateState:
    """Warp State struct bundling fields read by H1 crate-pushing kernels."""

    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    xpos: wp.array2d(dtype=wp.vec3f)
    xquat: wp.array2d(dtype=wp.quatf)
    site_xpos: wp.array2d(dtype=wp.vec3f)
    contact_pos: wp.array1d(dtype=wp.vec3f)
    contact_dist: wp.array1d(dtype=wp.float32)
    contact_geom: wp.array1d(dtype=wp.vec2i)
    contact_worldid: wp.array1d(dtype=wp.int32)
    cvel: wp.array2d(dtype=wp.spatial_vectorf)
    time: wp.array1d(dtype=wp.float32)


class H1PushCrate(Task):
    """Unitree H1 humanoid crate-pushing environment ported from DIAL-MPC."""

    dial_env_name = "unitree_h1_push_crate"
    state_dim = 4  # torso xyz plus crate x-position

    def __init__(
        self,
        planning_dt: float = 0.02,
        sim_dt: float = 0.02,
        target_vx: float = 0.8,
        target_vy: float = 0.0,
        target_vyaw: float = 0.0,
        gait: str = "slow_walk",
    ) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/h1/mjx_scene_h1_push_crate.xml"
        )
        mj_model.opt.timestep = planning_dt
        super().__init__(
            mj_model,
            sim_dt=sim_dt,
            trace_sites=("left_foot", "right_foot"),
            trace_bodies=("torso_link", "box_body"),
        )

        self.pelvis_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.torso_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box_body"
        )
        self.foot_site_ids = tuple(
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("left_foot", "right_foot")
        )
        if (
            self.pelvis_body_id < 0
            or self.torso_body_id < 0
            or self.box_body_id < 0
            or any(site_id < 0 for site_id in self.foot_site_ids)
        ):
            raise ValueError("H1 push-crate model is missing required bodies or sites.")

        self.box_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "static_box"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.left_hand_geom_id = self._single_collision_geom_id("left_elbow_link")
        self.right_hand_geom_id = self._single_collision_geom_id("right_elbow_link")
        self.left_knee_geom_id = self._single_collision_geom_id("left_knee_link")
        self.left_foot_geom_id = self._single_collision_geom_id("left_ankle_link")
        self.right_knee_geom_id = self._single_collision_geom_id("right_knee_link")
        self.right_foot_geom_id = self._single_collision_geom_id("right_ankle_link")
        self.torso_geom_id = self._single_collision_geom_id("torso_link")
        if self.box_geom_id < 0 or self.floor_geom_id < 0:
            raise ValueError("H1 push-crate model is missing static_box or floor geom.")

        self.target_vx = float(target_vx)
        self.target_vy = float(target_vy)
        self.target_vyaw = float(target_vyaw)
        self.target_height = 1.2
        self.ramp_up_time = 2.0
        self.action_scale = 1.0
        self.hand_contact_max_z = 1.1
        self.box_target_x = 2.5
        self.contact_count = 0
        self._contact_reward_wp = None
        self._left_foot_dist_wp = None
        self._right_foot_dist_wp = None
        self._joint_low_wp = None
        self._joint_high_wp = None
        self._physical_joint_low_wp = None
        self._physical_joint_high_wp = None
        self._kp_wp = None
        self._kd_wp = None
        self._torque_limit_wp = None

        gait_phases = {
            "stand": (0.0, 0.0),
            "slow_walk": (0.0, 0.5),
            "walk": (0.0, 0.5),
            "jog": (0.0, 0.5),
        }
        gait_params = {
            "stand": (1.0, 1.0, 0.0),
            "slow_walk": (0.6, 0.8, 0.15),
            "walk": (0.5, 1.0, 0.15),
            "jog": (0.3, 2.0, 0.2),
        }
        if gait not in gait_phases:
            raise ValueError(f"Unknown H1 gait {gait!r}.")
        self.gait_phase = gait_phases[gait]
        self.gait_duty_ratio, self.gait_cadence, self.gait_amplitude = gait_params[
            gait
        ]

        self.home_qpos = np.asarray(self.mj_model.key_qpos[0], dtype=np.float32).copy()
        self.default_ctrl = np.zeros(self.nu, dtype=np.float32)
        self.joint_target_low = np.asarray(
            [
                -0.3, -0.3, -1.0, 0.0, -0.6,
                -0.3, -0.3, -1.0, 0.0, -0.6,
                -0.5,
                -0.78, -0.3, -0.3, -0.3,
                -0.78, -0.3, -0.3, -0.3,
            ],
            dtype=np.float32,
        )
        self.joint_target_high = np.asarray(
            [
                0.3, 0.3, 1.0, 1.74, 0.4,
                0.3, 0.3, 1.0, 1.74, 0.4,
                0.5,
                0.78, 0.3, 0.3, 0.3,
                0.78, 0.3, 0.3, 0.3,
            ],
            dtype=np.float32,
        )
        self.kp = np.asarray(
            [
                200.0, 200.0, 200.0, 200.0, 60.0,
                200.0, 200.0, 200.0, 200.0, 60.0,
                200.0,
                60.0, 60.0, 60.0, 60.0,
                60.0, 60.0, 60.0, 60.0,
            ],
            dtype=np.float32,
        )
        self.kd = np.asarray(
            [
                5.0, 5.0, 5.0, 5.0, 1.5,
                5.0, 5.0, 5.0, 5.0, 1.5,
                5.0,
                1.5, 1.5, 1.5, 1.5,
                1.5, 1.5, 1.5, 1.5,
            ],
            dtype=np.float32,
        )
        self.torque_limit = np.asarray(
            self.mj_model.actuator_ctrlrange[:, 1], dtype=np.float32
        )
        actuator_joint_ids = np.asarray(
            self.mj_model.actuator_trnid[: self.nu, 0], dtype=np.int32
        )
        if np.any(actuator_joint_ids < 0):
            raise ValueError("H1 push-crate actuators must be joint motors.")
        self.physical_joint_low = np.asarray(
            self.mj_model.jnt_range[actuator_joint_ids, 0], dtype=np.float32
        )
        self.physical_joint_high = np.asarray(
            self.mj_model.jnt_range[actuator_joint_ids, 1], dtype=np.float32
        )

        if self.nu != 19:
            raise ValueError(f"Expected 19 H1 actuators, got {self.nu}.")
        for name, arr in (
            ("joint_target_low", self.joint_target_low),
            ("joint_target_high", self.joint_target_high),
            ("physical_joint_low", self.physical_joint_low),
            ("physical_joint_high", self.physical_joint_high),
            ("kp", self.kp),
            ("kd", self.kd),
            ("torque_limit", self.torque_limit),
        ):
            if arr.shape != (self.nu,):
                raise ValueError(f"{name} shape {arr.shape} != ({self.nu},).")

        self.u_min = -np.ones(self.nu, dtype=np.float32)
        self.u_max = np.ones(self.nu, dtype=np.float32)

    def _single_collision_geom_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise ValueError(f"H1 push-crate model is missing body {body_name!r}.")

        geom_ids = [
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if self.mj_model.geom_bodyid[geom_id] == body_id
            and self.mj_model.geom_contype[geom_id] != 0
        ]
        if len(geom_ids) != 1:
            raise ValueError(
                f"Expected exactly one collision geom on {body_name!r}; "
                f"found {geom_ids}."
            )
        return int(geom_ids[0])

    @wp.kernel
    def action_to_torque(
        x: H1PushCrateState,
        action: wp.array2d(dtype=wp.float32),
        ctrl: wp.array2d(dtype=wp.float32),
        joint_low: wp.array1d(dtype=wp.float32),
        joint_high: wp.array1d(dtype=wp.float32),
        physical_low: wp.array1d(dtype=wp.float32),
        physical_high: wp.array1d(dtype=wp.float32),
        kp: wp.array1d(dtype=wp.float32),
        kd: wp.array1d(dtype=wp.float32),
        torque_limit: wp.array1d(dtype=wp.float32),
        action_scale: float,
    ) -> None:
        i = wp.tid()
        for j in range(19):
            lo = joint_low[j]
            hi = joint_high[j]
            act_normalized = (action[i, j] * action_scale + 1.0) * 0.5
            joint_target = lo + act_normalized * (hi - lo)
            joint_target = wp.clamp(joint_target, physical_low[j], physical_high[j])

            q = x.qpos[i, 7 + j]
            qd = x.qvel[i, 6 + j]
            tau = kp[j] * (joint_target - q) - kd[j] * qd
            limit = torque_limit[j]
            ctrl[i, j] = wp.clamp(tau, -limit, limit)

    @wp.kernel
    def reset_contact_features(
        contact_reward: wp.array1d(dtype=wp.float32),
        left_foot_dist: wp.array1d(dtype=wp.float32),
        right_foot_dist: wp.array1d(dtype=wp.float32),
    ) -> None:
        i = wp.tid()
        contact_reward[i] = 0.0
        left_foot_dist[i] = 1.0e6
        right_foot_dist[i] = 1.0e6

    @wp.kernel
    def collect_contact_features(
        x: H1PushCrateState,
        contact_reward: wp.array1d(dtype=wp.float32),
        left_foot_dist: wp.array1d(dtype=wp.float32),
        right_foot_dist: wp.array1d(dtype=wp.float32),
        floor_geom_id: int,
        box_geom_id: int,
        left_hand_geom_id: int,
        right_hand_geom_id: int,
        left_knee_geom_id: int,
        left_foot_geom_id: int,
        right_knee_geom_id: int,
        right_foot_geom_id: int,
        torso_geom_id: int,
        hand_contact_max_z: float,
    ) -> None:
        contact_idx = wp.tid()
        world_id = x.contact_worldid[contact_idx]
        if world_id < 0:
            return
        if world_id >= contact_reward.shape[0]:
            return

        geoms = x.contact_geom[contact_idx]

        if geoms[0] == floor_geom_id:
            if geoms[1] == left_foot_geom_id:
                wp.atomic_min(left_foot_dist, world_id, x.contact_dist[contact_idx])
            if geoms[1] == right_foot_geom_id:
                wp.atomic_min(right_foot_dist, world_id, x.contact_dist[contact_idx])
        if geoms[1] == floor_geom_id:
            if geoms[0] == left_foot_geom_id:
                wp.atomic_min(left_foot_dist, world_id, x.contact_dist[contact_idx])
            if geoms[0] == right_foot_geom_id:
                wp.atomic_min(right_foot_dist, world_id, x.contact_dist[contact_idx])

        other_geom = int(-1)
        if geoms[0] == box_geom_id:
            other_geom = geoms[1]
        if geoms[1] == box_geom_id:
            other_geom = geoms[0]
        if other_geom < 0:
            return

        contact_dist = x.contact_dist[contact_idx]
        if contact_dist >= 1.0e-3:
            return

        contact_z = x.contact_pos[contact_idx][2]
        if other_geom == left_hand_geom_id or other_geom == right_hand_geom_id:
            if contact_z < hand_contact_max_z:
                wp.atomic_add(contact_reward, world_id, 1.0)
            return

        if (
            other_geom == left_knee_geom_id
            or other_geom == left_foot_geom_id
            or other_geom == right_knee_geom_id
            or other_geom == right_foot_geom_id
            or other_geom == torso_geom_id
        ):
            wp.atomic_add(contact_reward, world_id, -1.0)

    @wp.kernel
    def running_cost(
        x: H1PushCrateState,
        u: wp.array2d(dtype=wp.float32),
        out: wp.array1d(dtype=wp.float32),
        contact_reward_arr: wp.array1d(dtype=wp.float32),
        left_foot_dist_arr: wp.array1d(dtype=wp.float32),
        right_foot_dist_arr: wp.array1d(dtype=wp.float32),
        torque_limit: wp.array1d(dtype=wp.float32),
        pelvis_id: int,
        torso_id: int,
        left_foot_site_id: int,
        right_foot_site_id: int,
        target_vx: float,
        target_vy: float,
        target_vyaw: float,
        target_height: float,
        ramp_up_time: float,
        duty_ratio: float,
        cadence: float,
        amplitude: float,
        left_phase: float,
        right_phase: float,
        dt: float,
    ) -> None:
        i = wp.tid()

        time = x.time[i] - dt
        if time < 0.0:
            time = 0.0
        ramp = wp.clamp(time / ramp_up_time, 0.0, 1.0)
        vel_tar_x = _ramp_target(target_vx, ramp)
        vel_tar_y = _ramp_target(target_vy, ramp)
        ang_vel_tar_z = _ramp_target(target_vyaw, ramp)

        torso_q = x.xquat[i, torso_id]
        torso_pos = x.xpos[i, torso_id]

        left_target = _foot_step_height(
            time, duty_ratio, cadence, amplitude, left_phase
        )
        right_target = _foot_step_height(
            time, duty_ratio, cadence, amplitude, right_phase
        )
        left_foot_z = x.site_xpos[i, left_foot_site_id][2]
        right_foot_z = x.site_xpos[i, right_foot_site_id][2]
        left_measure = left_foot_dist_arr[i]
        right_measure = right_foot_dist_arr[i]
        if left_measure > 1.0e5:
            left_measure = left_foot_z
        if right_measure > 1.0e5:
            right_measure = right_foot_z
        left_error = left_target - left_measure
        right_error = right_target - right_measure
        gait_cost = left_error * left_error + right_error * right_error

        upright_cost = _up_vector_cost(x.xquat[i, pelvis_id])

        yaw_tar = target_vyaw * time
        d_yaw = _wrap_to_pi(_yaw_from_quat(torso_q) - yaw_tar)
        yaw_cost = d_yaw * d_yaw

        torso_cvel = x.cvel[i, torso_id]
        torso_lin_vel = wp.spatial_bottom(torso_cvel)
        torso_ang_vel = wp.spatial_top(torso_cvel)

        body_v = _body_xy_velocity(
            torso_q, torso_lin_vel[0], torso_lin_vel[1], torso_lin_vel[2]
        )
        vel_cost = (
            (body_v[0] - vel_tar_x) * (body_v[0] - vel_tar_x)
            + (body_v[1] - vel_tar_y) * (body_v[1] - vel_tar_y)
        )

        body_wz = _body_z_velocity(
            torso_q,
            torso_ang_vel[0] * PI / 180.0,
            torso_ang_vel[1] * PI / 180.0,
            torso_ang_vel[2] * PI / 180.0,
        )
        yaw_rate_cost = (body_wz - ang_vel_tar_z) * (
            body_wz - ang_vel_tar_z
        )

        height_error = torso_pos[2] - target_height
        height_cost = height_error * height_error

        energy_cost = float(0.0)
        for j in range(19):
            normalized_tau = u[i, j] / torque_limit[j]
            energy_cost += normalized_tau * normalized_tau

        out[i] += dt * (
            5.0 * gait_cost
            + 0.01 * upright_cost
            + 0.1 * yaw_cost
            + vel_cost
            + yaw_rate_cost
            + 0.5 * height_cost
            + 0.01 * energy_cost
            - 0.05 * contact_reward_arr[i]
        )

    @wp.kernel
    def terminal_cost(
        x: H1PushCrateState,
        out: wp.array1d(dtype=wp.float32),
    ) -> None:
        i = wp.tid()
        out[i] = 0.0

    @wp.kernel
    def state_extract(
        x: H1PushCrateState,
        weight: wp.array1d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
        torso_id: int,
        box_id: int,
    ) -> None:
        i = wp.tid()
        torso_pos = x.xpos[i, torso_id]
        box_pos = x.xpos[i, box_id]
        out[i, 0] = torso_pos[0] * weight[0]
        out[i, 1] = torso_pos[1] * weight[1]
        out[i, 2] = torso_pos[2] * weight[2]
        out[i, 3] = box_pos[0] * weight[3]

    def make_state(self, warp_data) -> H1PushCrateState:
        s = H1PushCrateState()
        s.qpos = warp_data.qpos
        s.qvel = warp_data.qvel
        s.xpos = warp_data.xpos
        s.xquat = warp_data.xquat
        s.site_xpos = warp_data.site_xpos
        s.contact_pos = warp_data.contact.pos
        s.contact_dist = warp_data.contact.dist
        s.contact_geom = warp_data.contact.geom
        s.contact_worldid = warp_data.contact.worldid
        s.cvel = warp_data.cvel
        s.time = warp_data.time
        self.contact_count = int(warp_data.contact.pos.shape[0])
        device = warp_data.qpos.device
        self._contact_reward_wp = wp.zeros(
            warp_data.qpos.shape[0],
            dtype=wp.float32,
            device=device,
        )
        self._left_foot_dist_wp = wp.zeros(
            warp_data.qpos.shape[0],
            dtype=wp.float32,
            device=device,
        )
        self._right_foot_dist_wp = wp.zeros(
            warp_data.qpos.shape[0],
            dtype=wp.float32,
            device=device,
        )
        self._joint_low_wp = wp.array(
            self.joint_target_low, dtype=wp.float32, device=device
        )
        self._joint_high_wp = wp.array(
            self.joint_target_high, dtype=wp.float32, device=device
        )
        self._physical_joint_low_wp = wp.array(
            self.physical_joint_low, dtype=wp.float32, device=device
        )
        self._physical_joint_high_wp = wp.array(
            self.physical_joint_high, dtype=wp.float32, device=device
        )
        self._kp_wp = wp.array(self.kp, dtype=wp.float32, device=device)
        self._kd_wp = wp.array(self.kd, dtype=wp.float32, device=device)
        self._torque_limit_wp = wp.array(
            self.torque_limit, dtype=wp.float32, device=device
        )
        return s

    def launch_step_control(self, state, action_arr, ctrl_arr):
        wp.launch(
            self.action_to_torque,
            dim=ctrl_arr.shape[0],
            inputs=[
                state,
                action_arr,
                ctrl_arr,
                self._joint_low_wp,
                self._joint_high_wp,
                self._physical_joint_low_wp,
                self._physical_joint_high_wp,
                self._kp_wp,
                self._kd_wp,
                self._torque_limit_wp,
                self.action_scale,
            ],
        )

    def apply_control_cpu(self, mj_data: mujoco.MjData, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        action_normalized = (action * self.action_scale + 1.0) * 0.5
        joint_target = self.joint_target_low + action_normalized * (
            self.joint_target_high - self.joint_target_low
        )
        joint_target = np.clip(
            joint_target,
            self.physical_joint_low,
            self.physical_joint_high,
        )
        q = np.asarray(mj_data.qpos[7:26], dtype=np.float32)
        qd = np.asarray(mj_data.qvel[6:25], dtype=np.float32)
        tau = self.kp * (joint_target - q) - self.kd * qd
        mj_data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)

    def launch_running_cost(self, state, ctrl_arr, out_wp, dt):
        wp.launch(
            self.reset_contact_features,
            dim=out_wp.shape[0],
            inputs=[
                self._contact_reward_wp,
                self._left_foot_dist_wp,
                self._right_foot_dist_wp,
            ],
        )
        wp.launch(
            self.collect_contact_features,
            dim=self.contact_count,
            inputs=[
                state,
                self._contact_reward_wp,
                self._left_foot_dist_wp,
                self._right_foot_dist_wp,
                self.floor_geom_id,
                self.box_geom_id,
                self.left_hand_geom_id,
                self.right_hand_geom_id,
                self.left_knee_geom_id,
                self.left_foot_geom_id,
                self.right_knee_geom_id,
                self.right_foot_geom_id,
                self.torso_geom_id,
                self.hand_contact_max_z,
            ],
        )
        wp.launch(
            self.running_cost,
            dim=out_wp.shape[0],
            inputs=[
                state,
                ctrl_arr,
                out_wp,
                self._contact_reward_wp,
                self._left_foot_dist_wp,
                self._right_foot_dist_wp,
                self._torque_limit_wp,
                self.pelvis_body_id,
                self.torso_body_id,
                self.foot_site_ids[0],
                self.foot_site_ids[1],
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
                dt,
            ],
        )

    def launch_terminal_cost(self, state, out_wp):
        wp.launch(self.terminal_cost, dim=out_wp.shape[0], inputs=[state, out_wp])

    def extract_state(self, state, out_wp, weight_wp):
        wp.launch(
            self.state_extract,
            dim=out_wp.shape[0],
            inputs=[
                state,
                weight_wp,
                out_wp,
                self.torso_body_id,
                self.box_body_id,
            ],
        )

    def extract_state_cpu(self, mj_data) -> np.ndarray:
        torso_pos = np.asarray(mj_data.xpos[self.torso_body_id, :3], dtype=np.float32)
        box_x = np.float32(mj_data.xpos[self.box_body_id, 0])
        return np.asarray(
            [torso_pos[0], torso_pos[1], torso_pos[2], box_x],
            dtype=np.float32,
        )

    def reset_to_home(self, mj_data: mujoco.MjData) -> None:
        mj_data.qpos[:] = self.home_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mj_data.time = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        box_x = data_np["xpos"][:, self.box_body_id, 0]
        return np.maximum(self.box_target_x - box_x, 0.0)
