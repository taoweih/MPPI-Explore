"""Spline interpolation utilities using numpy."""

import numpy as np
from typing import Callable, Literal

InterpMethodType = Literal["zero", "linear", "cubic"]
InterpFuncType = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def interp_zero(tq: np.ndarray, tk: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Zero-order hold spline interpolation.

    Args:
        tq: Query times, shape (H,).
        tk: Knot times, shape (num_knots,).
        knots: Control spline knots, shape (num_rollouts, num_knots, nu).

    Returns:
        Interpolated controls, shape (num_rollouts, H, nu).
    """
    inds = np.searchsorted(tk, tq, side="right") - 1
    inds = np.clip(inds, 0, len(tk) - 1)
    return knots[:, inds, :]  # (num_rollouts, H, nu)


def interp_linear(tq: np.ndarray, tk: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear spline interpolation.

    Args:
        tq: Query times, shape (H,).
        tk: Knot times, shape (num_knots,).
        knots: Control spline knots, shape (num_rollouts, num_knots, nu).

    Returns:
        Interpolated controls, shape (num_rollouts, H, nu).
    """
    num_rollouts, num_knots, nu = knots.shape
    # Fully vectorized linear interpolation
    idx = np.searchsorted(tk, tq, side='right') - 1
    idx = np.clip(idx, 0, num_knots - 2)
    t0 = tk[idx]
    t1 = tk[idx + 1]
    dt = t1 - t0
    alpha = np.where(dt > 0, (tq - t0) / dt, 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)  # (H,)
    v0 = knots[:, idx, :]       # (num_rollouts, H, nu)
    v1 = knots[:, idx + 1, :]   # (num_rollouts, H, nu)
    return v0 + alpha[None, :, None] * (v1 - v0)


def interp_cubic(tq: np.ndarray, tk: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Cubic spline interpolation.

    Args:
        tq: Query times, shape (H,).
        tk: Knot times, shape (num_knots,).
        knots: Control spline knots, shape (num_rollouts, num_knots, nu).

    Returns:
        Interpolated controls, shape (num_rollouts, H, nu).
    """
    from scipy.interpolate import CubicSpline

    num_rollouts, num_knots, nu = knots.shape
    result = np.zeros((num_rollouts, len(tq), nu), dtype=knots.dtype)
    for i in range(num_rollouts):
        cs = CubicSpline(tk, knots[i], axis=0, bc_type="not-a-knot")
        result[i] = cs(tq)
    return result


def get_interp_func(method: InterpMethodType) -> InterpFuncType:
    """Get the interpolation function for the specified method.

    Args:
        method: "zero", "linear", or "cubic".

    Returns:
        Interpolation function with signature (tq, tk, knots) -> controls.
    """
    if method == "zero":
        return interp_zero
    elif method == "linear":
        return interp_linear
    elif method == "cubic":
        return interp_cubic
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic']."
        )
