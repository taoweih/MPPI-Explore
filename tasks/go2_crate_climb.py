"""Unitree Go2 crate-climb task for mujoco_warp."""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from tasks.go2_walk import (
    _torque_limit,
    _wrap_to_pi,
    _yaw_from_quat,
)
from tasks.task_base import ROOT, Task


@wp.func
def _crate_joint_target_low(j: int) -> float:
    if j % 3 == 0:
        return -0.25
    if j % 3 == 1:
        if j < 6:
            return -1.0
        return 0.0
    return -2.7


@wp.func
def _crate_joint_target_high(j: int) -> float:
    if j % 3 == 0:
        return 0.25
    if j % 3 == 1:
        if j < 6:
            return 1.4
        return 1.8
    return -1.0


@wp.func
def _body_forward_axis(q: wp.quatf) -> wp.vec3f:
    return wp.vec3f(
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
        2.0 * (q[1] * q[2] + q[0] * q[3]),
        2.0 * (q[1] * q[3] - q[0] * q[2]),
    )


@wp.func
def _up_vector_cost(q: wp.quatf) -> float:
    up_x = 2.0 * (q[1] * q[3] + q[0] * q[2])
    up_y = 2.0 * (q[2] * q[3] - q[0] * q[1])
    up_z = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return up_x * up_x + up_y * up_y + (up_z - 1.0) * (up_z - 1.0)


@wp.func
def _contact_on_box_bonus(contact_pos: wp.vec3f) -> float:
    if contact_pos[0] <= 1.0:
        return 0.0
    if contact_pos[0] >= 1.6:
        return 0.0
    if contact_pos[1] <= -0.45:
        return 0.0
    if contact_pos[1] >= 0.45:
        return 0.0
    if contact_pos[2] <= 0.59:
        return 0.0
    if contact_pos[2] >= 0.61:
        return 0.0
    return 1.0


@wp.struct
class CrateState:
    """Warp State struct bundling fields read by Go2 crate-climb kernels."""

    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    xpos: wp.array2d(dtype=wp.vec3f)
    xquat: wp.array2d(dtype=wp.quatf)
    site_xpos: wp.array2d(dtype=wp.vec3f)
    contact_pos: wp.array1d(dtype=wp.vec3f)
    contact_geom: wp.array1d(dtype=wp.vec2i)
    contact_worldid: wp.array1d(dtype=wp.int32)
    time: wp.array1d(dtype=wp.float32)


class Go2CrateClimb(Task):
    """Unitree Go2 crate-climb environment ported from DIAL-MPC."""

    dial_env_name = "unitree_go2_crate_climb"
    state_dim = 3

    def __init__(
        self,
        planning_dt: float = 0.02,
        sim_dt: float = 0.02,
    ) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/go2/mjx_scene_force_crate.xml"
        )
        # mujoco_warp rejects nonzero box/mesh margins under CCD. DIAL's model
        # uses 0.001 foot margins, so keep them and disable unsupported CCD paths.
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD"):
            mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_NATIVECCD"):
            mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
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
            raise ValueError("Go2 crate model is missing required body or foot sites.")

        self.target_pos = np.array([1.45, 0.0, 0.87], dtype=np.float32)
        self.target_yaw = 0.0
        self.head_offset_x = 0.285
        self.kp = 30.0
        self.kd = 0.0
        self.action_scale = 1.0
        self.contact_count = 0
        self.box_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "static_box"
        )
        self.foot_geom_ids = tuple(
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("FL", "FR", "RL", "RR")
        )
        if self.box_geom_id < 0 or any(geom_id < 0 for geom_id in self.foot_geom_ids):
            raise ValueError("Go2 crate model is missing required box or foot geoms.")
        self._contact_bonus_wp = None

        self.home_qpos = np.asarray(self.mj_model.key_qpos[0], dtype=np.float32).copy()
        self.default_ctrl = np.zeros(self.nu, dtype=np.float32)
        self.joint_target_low = np.asarray(
            [
                -0.25, -1.0, -2.7,
                -0.25, -1.0, -2.7,
                -0.25, 0.0, -2.7,
                -0.25, 0.0, -2.7,
            ],
            dtype=np.float32,
        )
        self.joint_target_high = np.asarray(
            [
                0.25, 1.4, -1.0,
                0.25, 1.4, -1.0,
                0.25, 1.8, -1.0,
                0.25, 1.8, -1.0,
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

        self.u_min = -np.ones(self.nu, dtype=np.float32)
        self.u_max = np.ones(self.nu, dtype=np.float32)

    @wp.kernel
    def action_to_torque(
        x: CrateState,
        action: wp.array2d(dtype=wp.float32),
        ctrl: wp.array2d(dtype=wp.float32),
        kp: float,
        kd: float,
        action_scale: float,
    ) -> None:
        i = wp.tid()
        for j in range(12):
            lo = _crate_joint_target_low(j)
            hi = _crate_joint_target_high(j)
            act_normalized = (action[i, j] * action_scale + 1.0) * 0.5
            joint_target = lo + act_normalized * (hi - lo)
            joint_target = wp.clamp(joint_target, lo, hi)

            q = x.qpos[i, 7 + j]
            qd = x.qvel[i, 6 + j]
            tau = kp * (joint_target - q) - kd * qd
            limit = _torque_limit(j)
            ctrl[i, j] = wp.clamp(tau, -limit, limit)

    @wp.kernel
    def contact_box_reward(
        x: CrateState,
        out: wp.array1d(dtype=wp.float32),
        box_geom_id: int,
        fl_geom_id: int,
        fr_geom_id: int,
        rl_geom_id: int,
        rr_geom_id: int,
    ) -> None:
        contact_idx = wp.tid()
        world_id = x.contact_worldid[contact_idx]
        if world_id < 0:
            return
        if world_id >= out.shape[0]:
            return

        geoms = x.contact_geom[contact_idx]
        bonus = float(0.0)
        if geoms[0] == box_geom_id:
            if geoms[1] == fl_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[1] == fr_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[1] == rl_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[1] == rr_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
        if geoms[1] == box_geom_id:
            if geoms[0] == fl_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[0] == fr_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[0] == rl_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])
            if geoms[0] == rr_geom_id:
                bonus = _contact_on_box_bonus(x.contact_pos[contact_idx])

        if bonus > 0.0:
            wp.atomic_add(out, world_id, bonus)

    @wp.kernel
    def running_cost(
        x: CrateState,
        u: wp.array2d(dtype=wp.float32),
        out: wp.array1d(dtype=wp.float32),
        contact_bonus_arr: wp.array1d(dtype=wp.float32),
        base_id: int,
        target_x: float,
        target_y: float,
        target_z: float,
        target_yaw: float,
        head_offset_x: float,
        dt: float,
    ) -> None:
        i = wp.tid()

        q = x.xquat[i, base_id]
        head_pos = x.xpos[i, base_id] + _body_forward_axis(q) * head_offset_x
        target = wp.vec3f(target_x, target_y, target_z)
        pos_diff = head_pos - target
        pos_cost = wp.dot(pos_diff, pos_diff)

        upright_cost = _up_vector_cost(q)
        yaw_error = _wrap_to_pi(_yaw_from_quat(q) - target_yaw)
        yaw_cost = yaw_error * yaw_error

        contact_bonus = contact_bonus_arr[i]
        if contact_bonus > 4.0:
            contact_bonus = 4.0

        # DIAL reward: pos*1.0 + upright*0.01 + yaw*0.3 + contact*0.02.
        # Convert reward maximization to MPPI cost minimization.
        out[i] += dt * (
            pos_cost
            + 0.01 * upright_cost
            + 0.3 * yaw_cost
            - 0.02 * contact_bonus
        )

    @wp.kernel
    def terminal_cost(
        x: CrateState,
        out: wp.array1d(dtype=wp.float32),
    ) -> None:
        i = wp.tid()
        out[i] = 0.0

    @wp.kernel
    def state_extract(
        x: CrateState,
        weight: wp.array1d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
        base_id: int,
    ) -> None:
        i = wp.tid()
        base_pos = x.xpos[i, base_id]
        out[i, 0] = base_pos[0] * weight[0]
        out[i, 1] = base_pos[1] * weight[1]
        out[i, 2] = base_pos[2] * weight[2]

    def make_state(self, warp_data) -> CrateState:
        s = CrateState()
        s.qpos = warp_data.qpos
        s.qvel = warp_data.qvel
        s.xpos = warp_data.xpos
        s.xquat = warp_data.xquat
        s.site_xpos = warp_data.site_xpos
        s.contact_pos = warp_data.contact.pos
        s.contact_geom = warp_data.contact.geom
        s.contact_worldid = warp_data.contact.worldid
        s.time = warp_data.time
        self.contact_count = int(warp_data.contact.pos.shape[0])
        self._contact_bonus_wp = wp.zeros(
            warp_data.qpos.shape[0],
            dtype=wp.float32,
            device=warp_data.qpos.device,
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
        self._contact_bonus_wp.zero_()
        wp.launch(
            self.contact_box_reward,
            dim=self.contact_count,
            inputs=[
                state,
                self._contact_bonus_wp,
                self.box_geom_id,
                self.foot_geom_ids[0],
                self.foot_geom_ids[1],
                self.foot_geom_ids[2],
                self.foot_geom_ids[3],
            ],
        )
        wp.launch(
            self.running_cost,
            dim=out_wp.shape[0],
            inputs=[
                state,
                ctrl_arr,
                out_wp,
                self._contact_bonus_wp,
                self.base_body_id,
                self.target_pos[0],
                self.target_pos[1],
                self.target_pos[2],
                self.target_yaw,
                self.head_offset_x,
                dt,
            ],
        )

    def launch_terminal_cost(self, state, out_wp):
        wp.launch(self.terminal_cost, dim=out_wp.shape[0], inputs=[state, out_wp])

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
        mj_data.ctrl[:] = 0.0
        mj_data.time = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        xpos = data_np["xpos"][:, self.base_body_id, :]
        xquat = data_np["xquat"][:, self.base_body_id, :]
        head_offset = np.zeros_like(xpos)
        q = xquat
        head_offset[:, 0] = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        head_offset[:, 1] = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        head_offset[:, 2] = 2.0 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        head_pos = xpos + self.head_offset_x * head_offset
        return np.sqrt(np.sum((head_pos - self.target_pos[None, :]) ** 2, axis=1))
