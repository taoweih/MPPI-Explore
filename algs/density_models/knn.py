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

    Density is the standard kNN estimate up to constants:

        rho_i ∝ 1 / r_k(s_i)^D

    where `D = 2 * nv` is the effective manifold dimension for qpos and qvel.
    Set `include_task_state=True` to append the task-defined `extract_state`
    output to the density state, i.e. `[qpos, qvel, task_state]`.
    The density-guided controller then resamples with weights
    `(1 / (rho_i + eps)) ** alpha`, so lower-density states are copied forward
    more often.
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
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if k > self.K_MAX:
            raise ValueError(f"k={k} exceeds K_MAX={self.K_MAX}")

        self.k = int(k)
        self.alpha = float(alpha)
        self.position_weight = float(position_weight)
        self.angle_weight = float(angle_weight)
        self.linear_velocity_weight = float(linear_velocity_weight)
        self.angular_velocity_weight = float(angular_velocity_weight)
        self.include_task_state = bool(include_task_state)
        self.task_state_weight = float(task_state_weight)

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
        kernel = getattr(type(self), "_joint_knn_density_kernel", None)
        if kernel is None:
            @wp.kernel
            def joint_knn_density(
                states:        wp.array2d(dtype=wp.float32),  # (N, state_dim)
                density:       wp.array1d(dtype=wp.float32),  # (N,) output
                dists_scratch: wp.array2d(dtype=wp.float32),  # (N, K_MAX)
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
                            for d in range(3):
                                diff = states[i, qp + d] - states[j, qp + d]
                                dist_sq = dist_sq + position_w * position_w * diff * diff

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
                            dist_sq = dist_sq + angle_w * angle_w * angle * angle

                            for d in range(3):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = dist_sq + linvel_w * linvel_w * diff * diff
                            for d in range(3, 6):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = dist_sq + angvel_w * angvel_w * diff * diff

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
                            dist_sq = dist_sq + angle_w * angle_w * angle * angle

                            for d in range(3):
                                diff = states[i, nq + dv + d] - states[j, nq + dv + d]
                                dist_sq = dist_sq + angvel_w * angvel_w * diff * diff

                        elif jt == 2:  # slide: scalar position, linear velocity
                            diff = states[i, qp] - states[j, qp]
                            dist_sq = dist_sq + position_w * position_w * diff * diff
                            vdiff = states[i, nq + dv] - states[j, nq + dv]
                            dist_sq = dist_sq + linvel_w * linvel_w * vdiff * vdiff

                        else:  # hinge: wrapped scalar angle, angular velocity
                            diff = states[i, qp] - states[j, qp]
                            diff = diff - TWO_PI * wp.floor((diff + PI) / TWO_PI)
                            dist_sq = dist_sq + angle_w * angle_w * diff * diff
                            vdiff = states[i, nq + dv] - states[j, nq + dv]
                            dist_sq = dist_sq + angvel_w * angvel_w * vdiff * vdiff

                    task_offset = nq + nv
                    for d in range(task_state_dim):
                        diff = states[i, task_offset + d] - states[j, task_offset + d]
                        dist_sq = (
                            dist_sq
                            + task_state_w * task_state_w * diff * diff
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

    def randomize_offsets(self, rng: np.random.Generator) -> None:
        """Refill the per-stage U[0, 1/N) offsets for systematic resampling."""
        offsets = rng.uniform(
            0.0, 1.0 / float(self._num_samples),
            size=self._num_resample_stages,
        ).astype(np.float32)
        self._offsets_wp.assign(offsets)
