"""MuJoCo-joint-aware k-nearest-neighbor density model."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np
import warp as wp

from algs.density_models.base import DensityModel


class KNNDensityModel(DensityModel):
    """kNN density over full MuJoCo state `(qpos, qvel)`.

    The distance metric respects MuJoCo joint semantics:

      - free joint: Euclidean root position, quaternion geodesic orientation,
        linear velocity, angular velocity
      - ball joint: quaternion geodesic orientation, angular velocity
      - slide joint: scalar position distance, linear velocity
      - hinge joint: wrapped angular distance, angular velocity

    Distances are normalized by the current rollout batch before the kNN
    radius is measured.  For each semantic component `c` of each joint
    (position, orientation angle, linear velocity, angular velocity, and
    optional task-state dimensions), compute the pairwise RMS spread

        sigma_c = sqrt( sum_{a != b} delta_c(a, b)^2 / (2 N (N - 1)) )

    and use

        d(i, j)^2 = sum_c w_c^2 * delta_c(i, j)^2 / max(sigma_c, eps)^2.

    This keeps prismatic, angular, velocity, and task-state coordinates
    comparable within the current resampling batch while preserving the
    joint-aware angle/quaternion distance definitions below.

    Density is the standard kNN estimate up to constants:

        rho_i ∝ 1 / r_k(s_i)^D

    where `D = 2 * nv` is the effective manifold dimension for qpos and qvel.
    Set `include_task_state=True` to append the task-defined `extract_state`
    output to the density state, i.e. `[qpos, qvel, task_state]`.
    The density-guided controller then resamples with weights
    `(1 / (rho_i + eps)) ** alpha`, so lower-density states are copied forward
    more often. DensityGuidedMPPI can additionally combine normalized
    low-density and low-cost scores at each stage boundary.
    """

    K_MAX = 64

    def __init__(
        self,
        k: int = 5,
        alpha: float = 1.0,
        position_weight: float = 1.0,
        angle_weight: float = 1.0,
        linear_velocity_weight: float = 1.0,
        angular_velocity_weight: float = 1.0,
        include_task_state: bool = False,
        task_state_weight: float = 1.0,
        min_scale: float = 1.0e-6,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if k > self.K_MAX:
            raise ValueError(f"k={k} exceeds K_MAX={self.K_MAX}")
        if min_scale <= 0.0:
            raise ValueError(f"min_scale must be > 0; got {min_scale}")

        self.k = int(k)
        self.alpha = float(alpha)
        self.position_weight = float(position_weight)
        self.angle_weight = float(angle_weight)
        self.linear_velocity_weight = float(linear_velocity_weight)
        self.angular_velocity_weight = float(angular_velocity_weight)
        self.include_task_state = bool(include_task_state)
        self.task_state_weight = float(task_state_weight)
        self.min_scale = float(min_scale)

        # Filled by configure_from_task()/configure_from_model(), which
        # DensityGuidedMPPI calls.
        self.state_dim = 0
        self._nq = 0
        self._nv = 0
        self._num_joints = 0
        self._metric_dim = 0
        self._task_state_dim = 0
        self._joint_type_np: Optional[np.ndarray] = None
        self._qposadr_np: Optional[np.ndarray] = None
        self._dofadr_np: Optional[np.ndarray] = None

        # Filled by alloc().
        self._num_samples = 0
        self._num_resample_stages = 0
        self._density_wp: Optional[wp.array] = None
        self._offsets_wp: Optional[wp.array] = None
        self._dists_scratch_wp: Optional[wp.array] = None
        self._joint_type_wp: Optional[wp.array] = None
        self._qposadr_wp: Optional[wp.array] = None
        self._dofadr_wp: Optional[wp.array] = None
        self._position_scale_wp: Optional[wp.array] = None
        self._angle_scale_wp: Optional[wp.array] = None
        self._linear_velocity_scale_wp: Optional[wp.array] = None
        self._angular_velocity_scale_wp: Optional[wp.array] = None
        self._task_state_scale_wp: Optional[wp.array] = None

    # ── Math ──────────────────────────────────────────────────────────

    def launch_state_extract(
        self,
        qpos_wp: wp.array,
        qvel_wp: wp.array,
        out_wp: wp.array,
        task_state_wp: Optional[wp.array] = None,
    ) -> None:
        """Write `[qpos, qvel]`, optionally followed by task-extracted state."""
        if self.include_task_state:
            if task_state_wp is None:
                raise ValueError(
                    "task_state_wp is required when include_task_state=True."
                )
            kernel = getattr(type(self), "_extract_qpos_qvel_task_state_kernel", None)
            if kernel is None:
                @wp.kernel
                def extract_qpos_qvel_task_state(
                    qpos:       wp.array2d(dtype=wp.float32),
                    qvel:       wp.array2d(dtype=wp.float32),
                    task_state: wp.array2d(dtype=wp.float32),
                    out:        wp.array2d(dtype=wp.float32),
                    nq:         int,
                    nv:         int,
                    task_dim:   int,
                ):
                    i = wp.tid()
                    for q in range(nq):
                        out[i, q] = qpos[i, q]
                    for v in range(nv):
                        out[i, nq + v] = qvel[i, v]
                    offset = nq + nv
                    for d in range(task_dim):
                        out[i, offset + d] = task_state[i, d]

                type(self)._extract_qpos_qvel_task_state_kernel = (
                    extract_qpos_qvel_task_state
                )
                kernel = extract_qpos_qvel_task_state

            wp.launch(kernel, dim=self._num_samples, inputs=[
                qpos_wp, qvel_wp, task_state_wp, out_wp,
                self._nq, self._nv, self._task_state_dim,
            ])
            return

        kernel = getattr(type(self), "_extract_qpos_qvel_kernel", None)
        if kernel is None:
            @wp.kernel
            def extract_qpos_qvel(
                qpos: wp.array2d(dtype=wp.float32),
                qvel: wp.array2d(dtype=wp.float32),
                out:  wp.array2d(dtype=wp.float32),
                nq:   int,
                nv:   int,
            ):
                i = wp.tid()
                for q in range(nq):
                    out[i, q] = qpos[i, q]
                for v in range(nv):
                    out[i, nq + v] = qvel[i, v]

            type(self)._extract_qpos_qvel_kernel = extract_qpos_qvel
            kernel = extract_qpos_qvel

        wp.launch(kernel, dim=self._num_samples, inputs=[
            qpos_wp, qvel_wp, out_wp, self._nq, self._nv,
        ])

    def launch_compute(self, states_wp: wp.array) -> None:
        """Compute joint-aware kNN density for each configured state sample."""
        zero_scale_kernel = getattr(type(self), "_zero_semantic_scale_kernel", None)
        if zero_scale_kernel is None:
            @wp.kernel
            def zero_semantic_scale(
                position_scale: wp.array1d(dtype=wp.float32),
                angle_scale:    wp.array1d(dtype=wp.float32),
                linvel_scale:   wp.array1d(dtype=wp.float32),
                angvel_scale:   wp.array1d(dtype=wp.float32),
                task_scale:     wp.array1d(dtype=wp.float32),
                num_joints:     int,
                task_state_dim: int,
            ):
                tid = wp.tid()

                if tid < num_joints:
                    position_scale[tid] = float(0.0)
                    angle_scale[tid] = float(0.0)
                    linvel_scale[tid] = float(0.0)
                    angvel_scale[tid] = float(0.0)
                    return

                task_d = tid - num_joints
                if task_d < task_state_dim:
                    task_scale[task_d] = float(0.0)

            type(self)._zero_semantic_scale_kernel = zero_semantic_scale
            zero_scale_kernel = zero_semantic_scale

        accumulate_scale_kernel = getattr(
            type(self), "_accumulate_semantic_scale_kernel", None,
        )
        if accumulate_scale_kernel is None:
            @wp.kernel
            def accumulate_semantic_scale(
                states:         wp.array2d(dtype=wp.float32),
                position_sum:   wp.array1d(dtype=wp.float32),
                angle_sum:      wp.array1d(dtype=wp.float32),
                linvel_sum:     wp.array1d(dtype=wp.float32),
                angvel_sum:     wp.array1d(dtype=wp.float32),
                task_sum:       wp.array1d(dtype=wp.float32),
                joint_type:     wp.array1d(dtype=wp.int32),
                qposadr:        wp.array1d(dtype=wp.int32),
                dofadr:         wp.array1d(dtype=wp.int32),
                N:              int,
                nq:             int,
                nv:             int,
                num_joints:     int,
                task_state_dim: int,
            ):
                tid = wp.tid()
                component = tid // N
                a = tid - component * N
                PI = float(3.141592653589793)
                TWO_PI = float(6.283185307179586)

                if component < num_joints:
                    jt = joint_type[component]
                    qp = qposadr[component]
                    dv = dofadr[component]

                    pos_local = float(0.0)
                    angle_local = float(0.0)
                    linvel_local = float(0.0)
                    angvel_local = float(0.0)

                    for b in range(N):
                        if a == b:
                            continue

                        if jt == 0:  # free
                            pos_sq = float(0.0)
                            for d in range(3):
                                diff = states[a, qp + d] - states[b, qp + d]
                                pos_sq = pos_sq + diff * diff
                            pos_local = pos_local + pos_sq

                            dot = (
                                states[a, qp + 3] * states[b, qp + 3]
                                + states[a, qp + 4] * states[b, qp + 4]
                                + states[a, qp + 5] * states[b, qp + 5]
                                + states[a, qp + 6] * states[b, qp + 6]
                            )
                            if dot < 0.0:
                                dot = -dot
                            dot = wp.clamp(dot, 0.0, 1.0)
                            angle = 2.0 * wp.acos(dot)
                            angle_local = angle_local + angle * angle

                            linvel_sq = float(0.0)
                            for d in range(3):
                                diff = states[a, nq + dv + d] - states[b, nq + dv + d]
                                linvel_sq = linvel_sq + diff * diff
                            linvel_local = linvel_local + linvel_sq

                            angvel_sq = float(0.0)
                            for d in range(3, 6):
                                diff = states[a, nq + dv + d] - states[b, nq + dv + d]
                                angvel_sq = angvel_sq + diff * diff
                            angvel_local = angvel_local + angvel_sq

                        elif jt == 1:  # ball
                            dot = (
                                states[a, qp] * states[b, qp]
                                + states[a, qp + 1] * states[b, qp + 1]
                                + states[a, qp + 2] * states[b, qp + 2]
                                + states[a, qp + 3] * states[b, qp + 3]
                            )
                            if dot < 0.0:
                                dot = -dot
                            dot = wp.clamp(dot, 0.0, 1.0)
                            angle = 2.0 * wp.acos(dot)
                            angle_local = angle_local + angle * angle

                            angvel_sq = float(0.0)
                            for d in range(3):
                                diff = states[a, nq + dv + d] - states[b, nq + dv + d]
                                angvel_sq = angvel_sq + diff * diff
                            angvel_local = angvel_local + angvel_sq

                        elif jt == 2:  # slide
                            diff = states[a, qp] - states[b, qp]
                            pos_local = pos_local + diff * diff

                            vdiff = states[a, nq + dv] - states[b, nq + dv]
                            linvel_local = linvel_local + vdiff * vdiff

                        else:  # hinge
                            diff = states[a, qp] - states[b, qp]
                            diff = diff - TWO_PI * wp.floor((diff + PI) / TWO_PI)
                            angle_local = angle_local + diff * diff

                            vdiff = states[a, nq + dv] - states[b, nq + dv]
                            angvel_local = angvel_local + vdiff * vdiff

                    wp.atomic_add(position_sum, component, pos_local)
                    wp.atomic_add(angle_sum, component, angle_local)
                    wp.atomic_add(linvel_sum, component, linvel_local)
                    wp.atomic_add(angvel_sum, component, angvel_local)
                    return

                task_d = component - num_joints
                if task_d < task_state_dim:
                    offset = nq + nv
                    task_local = float(0.0)
                    for b in range(N):
                        if a == b:
                            continue
                        diff = states[a, offset + task_d] - states[b, offset + task_d]
                        task_local = task_local + diff * diff
                    wp.atomic_add(task_sum, task_d, task_local)

            type(self)._accumulate_semantic_scale_kernel = accumulate_semantic_scale
            accumulate_scale_kernel = accumulate_semantic_scale

        finalize_scale_kernel = getattr(type(self), "_finalize_semantic_scale_kernel", None)
        if finalize_scale_kernel is None:
            @wp.kernel
            def finalize_semantic_scale(
                position_scale: wp.array1d(dtype=wp.float32),
                angle_scale:    wp.array1d(dtype=wp.float32),
                linvel_scale:   wp.array1d(dtype=wp.float32),
                angvel_scale:   wp.array1d(dtype=wp.float32),
                task_scale:     wp.array1d(dtype=wp.float32),
                N:              int,
                num_joints:     int,
                task_state_dim: int,
                min_scale:      float,
            ):
                tid = wp.tid()

                if N <= 1:
                    if tid < num_joints:
                        position_scale[tid] = float(1.0)
                        angle_scale[tid] = float(1.0)
                        linvel_scale[tid] = float(1.0)
                        angvel_scale[tid] = float(1.0)
                    else:
                        task_d = tid - num_joints
                        if task_d < task_state_dim:
                            task_scale[task_d] = float(1.0)
                    return

                denom = 2.0 * float(N) * float(N - 1)

                if tid < num_joints:
                    pos_scale = wp.sqrt(position_scale[tid] / denom)
                    angle_scale_v = wp.sqrt(angle_scale[tid] / denom)
                    linvel_scale_v = wp.sqrt(linvel_scale[tid] / denom)
                    angvel_scale_v = wp.sqrt(angvel_scale[tid] / denom)

                    if pos_scale < min_scale:
                        pos_scale = min_scale
                    if angle_scale_v < min_scale:
                        angle_scale_v = min_scale
                    if linvel_scale_v < min_scale:
                        linvel_scale_v = min_scale
                    if angvel_scale_v < min_scale:
                        angvel_scale_v = min_scale

                    position_scale[tid] = pos_scale
                    angle_scale[tid] = angle_scale_v
                    linvel_scale[tid] = linvel_scale_v
                    angvel_scale[tid] = angvel_scale_v
                    return

                task_d = tid - num_joints
                if task_d < task_state_dim:
                    scale = wp.sqrt(task_scale[task_d] / denom)
                    if scale < min_scale:
                        scale = min_scale
                    task_scale[task_d] = scale

            type(self)._finalize_semantic_scale_kernel = finalize_semantic_scale
            finalize_scale_kernel = finalize_semantic_scale

        scale_dim = self._num_joints + self._task_state_dim
        wp.launch(zero_scale_kernel, dim=scale_dim, inputs=[
            self._position_scale_wp,
            self._angle_scale_wp,
            self._linear_velocity_scale_wp,
            self._angular_velocity_scale_wp,
            self._task_state_scale_wp,
            self._num_joints,
            self._task_state_dim,
        ])
        wp.launch(accumulate_scale_kernel, dim=scale_dim * self._num_samples, inputs=[
            states_wp,
            self._position_scale_wp,
            self._angle_scale_wp,
            self._linear_velocity_scale_wp,
            self._angular_velocity_scale_wp,
            self._task_state_scale_wp,
            self._joint_type_wp,
            self._qposadr_wp,
            self._dofadr_wp,
            self._num_samples,
            self._nq,
            self._nv,
            self._num_joints,
            self._task_state_dim,
        ])
        wp.launch(finalize_scale_kernel, dim=scale_dim, inputs=[
            self._position_scale_wp,
            self._angle_scale_wp,
            self._linear_velocity_scale_wp,
            self._angular_velocity_scale_wp,
            self._task_state_scale_wp,
            self._num_samples,
            self._num_joints,
            self._task_state_dim,
            self.min_scale,
        ])

        kernel = getattr(type(self), "_joint_knn_density_kernel", None)
        if kernel is None:
            @wp.kernel
            def joint_knn_density(
                states:        wp.array2d(dtype=wp.float32),  # (N, state_dim)
                density:       wp.array1d(dtype=wp.float32),  # (N,) output
                dists_scratch: wp.array2d(dtype=wp.float32),  # (N, K_MAX)
                position_scale: wp.array1d(dtype=wp.float32),
                angle_scale:   wp.array1d(dtype=wp.float32),
                linvel_scale:  wp.array1d(dtype=wp.float32),
                angvel_scale:  wp.array1d(dtype=wp.float32),
                task_scale:    wp.array1d(dtype=wp.float32),
                joint_type:    wp.array1d(dtype=wp.int32),
                qposadr:       wp.array1d(dtype=wp.int32),
                dofadr:        wp.array1d(dtype=wp.int32),
                N:             int,
                nq:            int,
                nv:            int,
                num_joints:    int,
                k:             int,
                metric_dim:    int,
                task_state_dim: int,
                position_w:    float,
                angle_w:       float,
                linvel_w:      float,
                angvel_w:      float,
                task_state_w:  float,
            ):
                i = wp.tid()
                INF = float(1.0e30)
                PI = float(3.141592653589793)
                TWO_PI = float(6.283185307179586)

                for s in range(k):
                    dists_scratch[i, s] = INF

                for j in range(N):
                    if j == i:
                        continue

                    dist_sq = float(0.0)

                    for joint in range(num_joints):
                        jt = joint_type[joint]
                        qp = qposadr[joint]
                        dv = dofadr[joint]

                        if jt == 0:  # free: xyz + quaternion, linear + angular velocity
                            pos_inv_scale = position_w / position_scale[joint]
                            for d in range(3):
                                diff = states[i, qp + d] - states[j, qp + d]
                                dist_sq = dist_sq + pos_inv_scale * pos_inv_scale * diff * diff

                            dot = (
                                states[i, qp + 3] * states[j, qp + 3]
                                + states[i, qp + 4] * states[j, qp + 4]
                                + states[i, qp + 5] * states[j, qp + 5]
                                + states[i, qp + 6] * states[j, qp + 6]
                            )
                            if dot < 0.0:
                                dot = -dot
                            dot = wp.clamp(dot, 0.0, 1.0)
                            angle = 2.0 * wp.acos(dot)
                            angle_inv_scale = angle_w / angle_scale[joint]
                            dist_sq = dist_sq + angle_inv_scale * angle_inv_scale * angle * angle

                            linvel_inv_scale = linvel_w / linvel_scale[joint]
                            for d in range(3):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = (
                                    dist_sq
                                    + linvel_inv_scale * linvel_inv_scale * diff * diff
                                )
                            angvel_inv_scale = angvel_w / angvel_scale[joint]
                            for d in range(3, 6):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = (
                                    dist_sq
                                    + angvel_inv_scale * angvel_inv_scale * diff * diff
                                )

                        elif jt == 1:  # ball: quaternion, angular velocity
                            dot = (
                                states[i, qp] * states[j, qp]
                                + states[i, qp + 1] * states[j, qp + 1]
                                + states[i, qp + 2] * states[j, qp + 2]
                                + states[i, qp + 3] * states[j, qp + 3]
                            )
                            if dot < 0.0:
                                dot = -dot
                            dot = wp.clamp(dot, 0.0, 1.0)
                            angle = 2.0 * wp.acos(dot)
                            angle_inv_scale = angle_w / angle_scale[joint]
                            dist_sq = dist_sq + angle_inv_scale * angle_inv_scale * angle * angle

                            angvel_inv_scale = angvel_w / angvel_scale[joint]
                            for d in range(3):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = (
                                    dist_sq
                                    + angvel_inv_scale * angvel_inv_scale * diff * diff
                                )

                        elif jt == 2:  # slide: scalar position, linear velocity
                            diff = states[i, qp] - states[j, qp]
                            pos_inv_scale = position_w / position_scale[joint]
                            dist_sq = (
                                dist_sq
                                + pos_inv_scale * pos_inv_scale * diff * diff
                            )
                            vdiff = states[i, nq + dv] - states[j, nq + dv]
                            linvel_inv_scale = linvel_w / linvel_scale[joint]
                            dist_sq = dist_sq + linvel_inv_scale * linvel_inv_scale * vdiff * vdiff

                        else:  # hinge: wrapped scalar angle, angular velocity
                            diff = states[i, qp] - states[j, qp]
                            diff = diff - TWO_PI * wp.floor((diff + PI) / TWO_PI)
                            angle_inv_scale = angle_w / angle_scale[joint]
                            dist_sq = dist_sq + angle_inv_scale * angle_inv_scale * diff * diff
                            vdiff = states[i, nq + dv] - states[j, nq + dv]
                            angvel_inv_scale = angvel_w / angvel_scale[joint]
                            dist_sq = dist_sq + angvel_inv_scale * angvel_inv_scale * vdiff * vdiff

                    task_offset = nq + nv
                    for d in range(task_state_dim):
                        diff = states[i, task_offset + d] - states[j, task_offset + d]
                        task_inv_scale = task_state_w / task_scale[d]
                        dist_sq = (
                            dist_sq
                            + task_inv_scale * task_inv_scale * diff * diff
                        )

                    max_idx = int(0)
                    max_val = dists_scratch[i, 0]
                    for s in range(1, k):
                        v = dists_scratch[i, s]
                        if v > max_val:
                            max_val = v
                            max_idx = s

                    if dist_sq < max_val:
                        dists_scratch[i, max_idx] = dist_sq

                r_k_sq = float(0.0)
                for s in range(k):
                    v = dists_scratch[i, s]
                    if v > r_k_sq:
                        r_k_sq = v

                r_k = wp.sqrt(r_k_sq)
                r_k_pow_dim = float(1.0)
                for _ in range(metric_dim):
                    r_k_pow_dim = r_k_pow_dim * r_k

                eps = float(1.0e-12)
                density[i] = 1.0 / (r_k_pow_dim + eps)

            type(self)._joint_knn_density_kernel = joint_knn_density
            kernel = joint_knn_density

        wp.launch(kernel, dim=self._num_samples, inputs=[
            states_wp, self._density_wp, self._dists_scratch_wp,
            self._position_scale_wp, self._angle_scale_wp,
            self._linear_velocity_scale_wp, self._angular_velocity_scale_wp,
            self._task_state_scale_wp,
            self._joint_type_wp, self._qposadr_wp, self._dofadr_wp,
            self._num_samples, self._nq, self._nv, self._num_joints, self.k,
            self._metric_dim, self._task_state_dim,
            self.position_weight, self.angle_weight,
            self.linear_velocity_weight, self.angular_velocity_weight,
            self.task_state_weight,
        ])

    def launch_resample(self, indices_wp: wp.array, stage_idx: int) -> None:
        """Systematic resample with weights ∝ 1 / (rho + eps)^alpha."""
        kernel = getattr(type(self), "_resample_from_density_kernel", None)
        if kernel is None:
            @wp.kernel
            def resample_from_density(
                density:   wp.array1d(dtype=wp.float32),  # (N,)
                indices:   wp.array1d(dtype=wp.int32),    # (N,) output
                offsets:   wp.array1d(dtype=wp.float32),  # (n_boundaries,) U[0, 1/N)
                stage_idx: int,
                N:         int,
                alpha:     float,
            ):
                tid = wp.tid()
                if tid > 0:
                    return

                eps = float(1.0e-6)
                inv_total = float(0.0)
                for j in range(N):
                    inv_total = inv_total + wp.pow(1.0 / (density[j] + eps), alpha)

                # Systematic resampling: one random phase, then N evenly spaced thresholds.
                u = offsets[stage_idx]
                step = 1.0 / float(N)
                cumulative = float(0.0)
                j = int(0)

                for i in range(N):
                    threshold = u + float(i) * step
                    can_advance = int(1)
                    while can_advance == 1:
                        if j >= N - 1:
                            can_advance = 0
                        else:
                            w_j = wp.pow(1.0 / (density[j] + eps), alpha) / inv_total
                            if cumulative + w_j < threshold:
                                cumulative = cumulative + w_j
                                j = j + 1
                            else:
                                can_advance = 0
                    indices[i] = j

            type(self)._resample_from_density_kernel = resample_from_density
            kernel = resample_from_density

        wp.launch(kernel, dim=1, inputs=[
            self._density_wp, indices_wp, self._offsets_wp,
            int(stage_idx), self._num_samples, self.alpha,
        ])

    def launch_resample_with_cost(
        self,
        costs_wp: wp.array,
        indices_wp: wp.array,
        stage_idx: int,
        cost_temperature: float,
        cost_weight: float,
    ) -> None:
        """Systematic resample with softmax over normalized density/cost scores."""
        kernel = getattr(type(self), "_resample_from_density_and_cost_kernel", None)
        if kernel is None:
            @wp.kernel
            def resample_from_density_and_cost(
                density:          wp.array1d(dtype=wp.float32),  # (N,)
                costs:            wp.array1d(dtype=wp.float32),  # (N,)
                indices:          wp.array1d(dtype=wp.int32),    # (N,) output
                offsets:          wp.array1d(dtype=wp.float32),  # (n_boundaries,)
                stage_idx:        int,
                N:                int,
                alpha:            float,
                cost_temperature: float,
                cost_weight:      float,
            ):
                tid = wp.tid()
                if tid > 0:
                    return

                eps = float(1.0e-6)
                score_eps = float(1.0e-6)

                density_mean = float(0.0)
                cost_mean = float(0.0)
                for j in range(N):
                    density_score = -wp.log(density[j] + eps)
                    cost_score = -costs[j]
                    density_mean = density_mean + density_score
                    cost_mean = cost_mean + cost_score
                inv_N = 1.0 / float(N)
                density_mean = density_mean * inv_N
                cost_mean = cost_mean * inv_N

                density_var = float(0.0)
                cost_var = float(0.0)
                for j in range(N):
                    density_delta = -wp.log(density[j] + eps) - density_mean
                    cost_delta = -costs[j] - cost_mean
                    density_var = density_var + density_delta * density_delta
                    cost_var = cost_var + cost_delta * cost_delta

                density_std = wp.sqrt(density_var * inv_N)
                cost_std = wp.sqrt(cost_var * inv_N)
                if density_std < score_eps:
                    density_std = score_eps
                if cost_std < score_eps:
                    cost_std = score_eps

                max_log_w = (
                    alpha * ((-wp.log(density[0] + eps) - density_mean) / density_std)
                    + (cost_weight / cost_temperature)
                    * ((-costs[0] - cost_mean) / cost_std)
                )
                for j_max in range(1, N):
                    density_z = (-wp.log(density[j_max] + eps) - density_mean) / density_std
                    cost_z = (-costs[j_max] - cost_mean) / cost_std
                    log_w = alpha * density_z + (cost_weight / cost_temperature) * cost_z
                    if log_w > max_log_w:
                        max_log_w = log_w

                total = float(0.0)
                for j in range(N):
                    density_z = (-wp.log(density[j] + eps) - density_mean) / density_std
                    cost_z = (-costs[j] - cost_mean) / cost_std
                    log_w = alpha * density_z + (cost_weight / cost_temperature) * cost_z
                    total = total + wp.exp(log_w - max_log_w)

                # Systematic resampling: one random phase, then N evenly spaced thresholds.
                u = offsets[stage_idx]
                step = 1.0 / float(N)
                cumulative = float(0.0)
                j = int(0)

                for i in range(N):
                    threshold = u + float(i) * step
                    can_advance = int(1)
                    while can_advance == 1:
                        if j >= N - 1:
                            can_advance = 0
                        else:
                            density_z = (
                                -wp.log(density[j] + eps) - density_mean
                            ) / density_std
                            cost_z = (-costs[j] - cost_mean) / cost_std
                            log_w = (
                                alpha * density_z
                                + (cost_weight / cost_temperature) * cost_z
                            )
                            w_j = wp.exp(log_w - max_log_w) / total
                            if cumulative + w_j < threshold:
                                cumulative = cumulative + w_j
                                j = j + 1
                            else:
                                can_advance = 0
                    indices[i] = j

            type(self)._resample_from_density_and_cost_kernel = (
                resample_from_density_and_cost
            )
            kernel = resample_from_density_and_cost

        wp.launch(kernel, dim=1, inputs=[
            self._density_wp, costs_wp, indices_wp, self._offsets_wp,
            int(stage_idx), self._num_samples, self.alpha,
            float(cost_temperature), float(cost_weight),
        ])

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def task_state_dim(self) -> int:
        return self._task_state_dim

    def configure_from_task(self, task) -> "KNNDensityModel":
        """Read model layout and, optionally, append the task's extracted state."""
        self.configure_from_model(task.mj_model)
        if not self.include_task_state:
            return self

        self._task_state_dim = int(getattr(task, "state_dim", 0))
        if self._task_state_dim <= 0:
            raise ValueError(
                "include_task_state=True requires task.state_dim > 0 and a "
                "task.extract_state implementation."
            )
        self.state_dim = self._nq + self._nv + self._task_state_dim
        self._metric_dim = 2 * self._nv + self._task_state_dim
        return self

    def configure_from_model(self, mj_model: mujoco.MjModel) -> "KNNDensityModel":
        """Read MuJoCo joint layout used by the full-state distance metric."""
        self._nq = int(mj_model.nq)
        self._nv = int(mj_model.nv)
        self._num_joints = int(mj_model.njnt)
        self._task_state_dim = 0
        self.state_dim = self._nq + self._nv
        self._metric_dim = 2 * self._nv
        self._joint_type_np = np.asarray(mj_model.jnt_type, dtype=np.int32).copy()
        self._qposadr_np = np.asarray(mj_model.jnt_qposadr, dtype=np.int32).copy()
        self._dofadr_np = np.asarray(mj_model.jnt_dofadr, dtype=np.int32).copy()
        return self

    def alloc(
        self,
        num_samples: int,
        state_dim: int,
        num_resample_stages: int,
        device,
    ) -> None:
        if self.state_dim == 0:
            raise RuntimeError(
                "KNNDensityModel must be configured with a MuJoCo model before alloc(). "
                "DensityGuidedMPPI does this automatically."
            )
        if int(state_dim) != self.state_dim:
            expected = "nq+nv"
            if self.include_task_state:
                expected = "nq+nv+task.state_dim"
            raise ValueError(
                f"KNNDensityModel expects state_dim={expected}={self.state_dim}; "
                f"got {state_dim}."
            )

        self._num_samples = int(num_samples)
        self._num_resample_stages = max(int(num_resample_stages), 1)

        self._density_wp = wp.zeros(num_samples, dtype=wp.float32, device=device)
        self._offsets_wp = wp.zeros(
            self._num_resample_stages, dtype=wp.float32, device=device,
        )
        self._dists_scratch_wp = wp.zeros(
            (num_samples, self.K_MAX), dtype=wp.float32, device=device,
        )

        self._joint_type_wp = wp.zeros(self._num_joints, dtype=wp.int32, device=device)
        self._joint_type_wp.assign(self._joint_type_np)
        self._qposadr_wp = wp.zeros(self._num_joints, dtype=wp.int32, device=device)
        self._qposadr_wp.assign(self._qposadr_np)
        self._dofadr_wp = wp.zeros(self._num_joints, dtype=wp.int32, device=device)
        self._dofadr_wp.assign(self._dofadr_np)

        self._position_scale_wp = wp.zeros(
            self._num_joints, dtype=wp.float32, device=device,
        )
        self._angle_scale_wp = wp.zeros(
            self._num_joints, dtype=wp.float32, device=device,
        )
        self._linear_velocity_scale_wp = wp.zeros(
            self._num_joints, dtype=wp.float32, device=device,
        )
        self._angular_velocity_scale_wp = wp.zeros(
            self._num_joints, dtype=wp.float32, device=device,
        )
        self._task_state_scale_wp = wp.zeros(
            max(self._task_state_dim, 1), dtype=wp.float32, device=device,
        )

    def randomize_offsets(self, rng: np.random.Generator) -> None:
        """Refill the per-stage U[0, 1/N) offsets for systematic resampling."""
        offsets = rng.uniform(
            0.0, 1.0 / float(self._num_samples),
            size=self._num_resample_stages,
        ).astype(np.float32)
        self._offsets_wp.assign(offsets)
