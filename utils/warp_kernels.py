"""Warp GPU kernels for density-guided and value-guided MPPI.

All kernels here are compatible with CUDA graph capture — they use only
pre-allocated warp arrays and avoid any host/device synchronisation.
"""

import warp as wp


# ── State extraction ─────────────────────────────────────────────────────

@wp.kernel
def extract_float_slice(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    col_start: int,
    state_dim: int,
):
    """Extract src[i, col_start:col_start+state_dim] * weight into dst."""
    i = wp.tid()
    for d in range(state_dim):
        dst[i, d] = src[i, col_start + d] * weight[d]


@wp.kernel
def extract_vec3_row(
    src: wp.array2d(dtype=wp.vec3f),
    dst: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    row_id: int,
):
    """Extract src[i, row_id] (vec3) * weight into dst[i, 0:3]."""
    i = wp.tid()
    v = src[i, row_id]
    dst[i, 0] = v[0] * weight[0]
    dst[i, 1] = v[1] * weight[1]
    dst[i, 2] = v[2] * weight[2]


# ── KDE density ──────────────────────────────────────────────────────────

@wp.kernel
def kde_density(
    states: wp.array2d(dtype=wp.float32),
    density: wp.array1d(dtype=wp.float32),
    bw: wp.array1d(dtype=wp.float32),
    N: int,
    D: int,
):
    """Gaussian KDE density at each sample.  O(N) per thread, N threads."""
    i = wp.tid()
    total = float(0.0)
    for j in range(N):
        dist_sq = float(0.0)
        for d in range(D):
            diff = (states[i, d] - states[j, d]) / bw[d]
            dist_sq = dist_sq + diff * diff
        total = total + wp.exp(-0.5 * dist_sq)
    density[i] = total / float(N)


# ── Resampling ───────────────────────────────────────────────────────────

@wp.kernel
def resample_from_density(
    density: wp.array1d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
    offsets: wp.array1d(dtype=wp.float32),
    stage_idx: int,
    N: int,
    alpha: float,
):
    """Systematic resampling inversely proportional to KDE density.

    The ``alpha`` parameter controls how aggressively low-density samples
    are promoted:
      * alpha = 1.0  – full inverse-density resampling (original behaviour).
      * alpha = 0.0  – uniform resampling (density ignored).
      * 0 < alpha < 1 – softer inverse-density resampling.

    Launch with dim=1 (single-threaded).  Sequential cost is O(N) which
    is negligible for typical N (256-4096).
    """
    tid = wp.tid()
    if tid > 0:
        return

    eps = float(1.0e-6)

    # Sum of inverse densities (normalisation constant).
    inv_total = float(0.0)
    for j in range(N):
        inv_total = inv_total + wp.pow(1.0 / (density[j] + eps), alpha)

    # Two-pointer systematic resample.
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


# ── Gather (index-based copy) ────────────────────────────────────────────

@wp.kernel
def gather_1d_float(
    src: wp.array1d(dtype=wp.float32),
    dst: wp.array1d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
):
    """dst[i] = src[indices[i]]."""
    i = wp.tid()
    dst[i] = src[indices[i]]


@wp.kernel
def gather_2d_float(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
    D: int,
):
    """dst[i, :] = src[indices[i], :]."""
    i = wp.tid()
    idx = indices[i]
    for d in range(D):
        dst[i, d] = src[idx, d]


@wp.kernel
def gather_3d_float(
    src: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
    T: int,
    D: int,
):
    """dst[i, :, :] = src[indices[i], :, :]."""
    i = wp.tid()
    idx = indices[i]
    for t in range(T):
        for d in range(D):
            dst[i, t, d] = src[idx, t, d]


# ── Knot regeneration & zero-order interpolation ─────────────────────────

@wp.kernel
def regenerate_knots(
    knots: wp.array3d(dtype=wp.float32),
    mean: wp.array2d(dtype=wp.float32),
    noise: wp.array3d(dtype=wp.float32),
    u_min: wp.array1d(dtype=wp.float32),
    u_max: wp.array1d(dtype=wp.float32),
    noise_level: float,
    k_start: int,
    remaining_K: int,
    nu: int,
):
    """Overwrite knots[:, k_start:, :] = clamp(mean + noise_level * noise)."""
    i = wp.tid()
    for k in range(remaining_K):
        for d in range(nu):
            val = mean[k_start + k, d] + noise_level * noise[i, k, d]
            knots[i, k_start + k, d] = wp.clamp(val, u_min[d], u_max[d])


@wp.kernel
def zero_order_interp(
    knots: wp.array3d(dtype=wp.float32),
    controls: wp.array3d(dtype=wp.float32),
    knot_indices: wp.array1d(dtype=wp.int32),
    T: int,
    nu: int,
):
    """controls[i, t, :] = knots[i, knot_indices[t], :]."""
    i = wp.tid()
    for t in range(T):
        k = knot_indices[t]
        for d in range(nu):
            controls[i, t, d] = knots[i, k, d]


# ── Random Fourier Feature value prediction ──────────────────────────────

@wp.kernel
def rff_predict(
    states: wp.array2d(dtype=wp.float32),
    W: wp.array2d(dtype=wp.float32),
    b: wp.array1d(dtype=wp.float32),
    theta: wp.array1d(dtype=wp.float32),
    state_min: wp.array1d(dtype=wp.float32),
    state_max: wp.array1d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
    num_features: int,
    state_dim: int,
):
    """Compute RFF features and linear prediction in one fused pass."""
    i = wp.tid()
    pred = float(0.0)
    for f in range(num_features):
        phase = b[f]
        for d in range(state_dim):
            denom = wp.max(state_max[d] - state_min[d], float(1.0e-6))
            x_n = 2.0 * wp.clamp((states[i, d] - state_min[d]) / denom, 0.0, 1.0) - 1.0
            phase = phase + x_n * W[f, d]
        pred = pred + wp.sin(phase) * theta[f] + wp.cos(phase) * theta[num_features + f]
    pred = pred + theta[2 * num_features]
    out[i] = pred


@wp.kernel
def blend_costs(
    base: wp.array1d(dtype=wp.float32),
    learned_value: wp.array1d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
    mix: float,
):
    """out = (1 - mix) * base + mix * learned_value."""
    i = wp.tid()
    out[i] = (1.0 - mix) * base[i] + mix * learned_value[i]


# ── Hash-grid encoding (2-D and 3-D) ───────────────────────────────────

@wp.kernel
def hashgrid_encode_2d(
    states: wp.array2d(dtype=wp.float32),
    embeddings: wp.array1d(dtype=wp.float32),
    resolutions: wp.array1d(dtype=wp.float32),
    features: wp.array2d(dtype=wp.float32),
    grid_min: float,
    grid_range_inv: float,
    num_levels: int,
    table_size: int,
):
    """Encode 2-D states through a multi-resolution hash grid.

    *embeddings* is flattened (num_levels * table_size * 2,).
    *features* is (N, num_levels * 2).
    """
    i = wp.tid()
    x0 = (states[i, 0] - grid_min) * grid_range_inv
    x1 = (states[i, 1] - grid_min) * grid_range_inv

    P0 = wp.uint32(1)
    P1 = wp.uint32(2654435761)
    ts = wp.uint32(table_size)

    for lev in range(num_levels):
        res = resolutions[lev]
        gx0 = x0 * res
        gx1 = x1 * res
        fx0 = wp.floor(gx0)
        fx1 = wp.floor(gx1)
        ix0 = int(fx0)
        ix1 = int(fx1)
        wx0 = gx0 - fx0
        wx1 = gx1 - fx1
        base = lev * table_size * 2

        f0 = float(0.0)
        f1 = float(0.0)
        for c in range(4):
            c0 = c & 1
            c1 = (c >> 1) & 1
            w = (1.0 - wx0 + float(c0) * (2.0 * wx0 - 1.0)) * (
                1.0 - wx1 + float(c1) * (2.0 * wx1 - 1.0)
            )
            h = int(
                (wp.uint32(ix0 + c0) * P0 + wp.uint32(ix1 + c1) * P1) % ts
            )
            f0 = f0 + w * embeddings[base + h * 2]
            f1 = f1 + w * embeddings[base + h * 2 + 1]

        features[i, lev * 2] = f0
        features[i, lev * 2 + 1] = f1


@wp.kernel
def hashgrid_encode_3d(
    states: wp.array2d(dtype=wp.float32),
    embeddings: wp.array1d(dtype=wp.float32),
    resolutions: wp.array1d(dtype=wp.float32),
    features: wp.array2d(dtype=wp.float32),
    grid_min: float,
    grid_range_inv: float,
    num_levels: int,
    table_size: int,
):
    """Encode 3-D states through a multi-resolution hash grid."""
    i = wp.tid()
    x0 = (states[i, 0] - grid_min) * grid_range_inv
    x1 = (states[i, 1] - grid_min) * grid_range_inv
    x2 = (states[i, 2] - grid_min) * grid_range_inv

    P0 = wp.uint32(1)
    P1 = wp.uint32(2654435761)
    P2 = wp.uint32(805459861)
    ts = wp.uint32(table_size)

    for lev in range(num_levels):
        res = resolutions[lev]
        gx0 = x0 * res
        gx1 = x1 * res
        gx2 = x2 * res
        fx0 = wp.floor(gx0)
        fx1 = wp.floor(gx1)
        fx2 = wp.floor(gx2)
        ix0 = int(fx0)
        ix1 = int(fx1)
        ix2 = int(fx2)
        wx0 = gx0 - fx0
        wx1 = gx1 - fx1
        wx2 = gx2 - fx2
        base = lev * table_size * 2

        f0 = float(0.0)
        f1 = float(0.0)
        for c in range(8):
            c0 = c & 1
            c1 = (c >> 1) & 1
            c2 = (c >> 2) & 1
            w = (
                (1.0 - wx0 + float(c0) * (2.0 * wx0 - 1.0))
                * (1.0 - wx1 + float(c1) * (2.0 * wx1 - 1.0))
                * (1.0 - wx2 + float(c2) * (2.0 * wx2 - 1.0))
            )
            h = int(
                (
                    wp.uint32(ix0 + c0) * P0
                    + wp.uint32(ix1 + c1) * P1
                    + wp.uint32(ix2 + c2) * P2
                )
                % ts
            )
            f0 = f0 + w * embeddings[base + h * 2]
            f1 = f1 + w * embeddings[base + h * 2 + 1]

        features[i, lev * 2] = f0
        features[i, lev * 2 + 1] = f1


# ── Dense MLP layers ───────────────────────────────────────────────────

@wp.kernel
def dense_swish(
    inp: wp.array2d(dtype=wp.float32),
    weight: wp.array2d(dtype=wp.float32),
    bias: wp.array1d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    in_dim: int,
    out_dim: int,
):
    """Fused linear + swish.  Launch with dim = N * out_dim."""
    tid = wp.tid()
    i = tid / out_dim
    j = tid - i * out_dim
    val = bias[j]
    for k in range(in_dim):
        val = val + inp[i, k] * weight[k, j]
    out[i, j] = val / (1.0 + wp.exp(-val))


@wp.kernel
def dense_linear_1d(
    inp: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    bias: wp.array1d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
    in_dim: int,
):
    """Linear layer with scalar output.  Launch with dim = N."""
    i = wp.tid()
    val = bias[0]
    for k in range(in_dim):
        val = val + inp[i, k] * weight[k]
    out[i] = val
