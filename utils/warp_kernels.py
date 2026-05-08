"""Generic Warp gather kernels (algorithm-agnostic).

These are used by density-guided MPPI to reshuffle per-sample state by an
index array.  Algorithm-specific kernels (KDE, resampling, hash-grid, MLP)
live next to the variant that owns them in `algs/density_guided_mppi.py`
and `algs/value_guided_mppi.py`.

All kernels here are CUDA-graph-capture compatible.
"""

import warp as wp


@wp.kernel
def gather_1d_float(
    src:     wp.array1d(dtype=wp.float32),
    dst:     wp.array1d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
):
    """dst[i] = src[indices[i]]."""
    i = wp.tid()
    dst[i] = src[indices[i]]


@wp.kernel
def gather_2d_float(
    src:     wp.array2d(dtype=wp.float32),
    dst:     wp.array2d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
    D:       int,
):
    """dst[i, :] = src[indices[i], :]."""
    i = wp.tid()
    idx = indices[i]
    for d in range(D):
        dst[i, d] = src[idx, d]


@wp.kernel
def gather_3d_float(
    src:     wp.array3d(dtype=wp.float32),
    dst:     wp.array3d(dtype=wp.float32),
    indices: wp.array1d(dtype=wp.int32),
    T:       int,
    D:       int,
):
    """dst[i, :, :] = src[indices[i], :, :]."""
    i = wp.tid()
    idx = indices[i]
    for t in range(T):
        for d in range(D):
            dst[i, t, d] = src[idx, t, d]
