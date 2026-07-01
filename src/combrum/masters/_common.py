"""Shared helpers for concrete master backends."""

from __future__ import annotations

import numpy as np


def validated_construction(
    K: int,
    theta_bounds: tuple[np.ndarray, np.ndarray],
    c_theta: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Validate backend-independent master constructor inputs."""
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(f"K must be an integer >= 1; got {K!r}")
    K = int(K)
    try:
        lower_in, upper_in = theta_bounds
    except (TypeError, ValueError):
        raise ValueError(
            "theta_bounds must be a (lower, upper) pair of (K,) arrays"
        ) from None
    lower = np.array(lower_in, dtype=np.float64)
    upper = np.array(upper_in, dtype=np.float64)
    c = np.array(c_theta, dtype=np.float64)
    for name, arr in (("lower", lower), ("upper", upper), ("c_theta", c)):
        if arr.shape != (K,):
            raise ValueError(f"{name} must have shape ({K},); got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite everywhere")
    if (lower > upper).any():
        bad = np.flatnonzero(lower > upper).tolist()
        raise ValueError(f"lower must be <= upper; violated at {bad}")
    for arr in (lower, upper, c):
        arr.setflags(write=False)
    return K, lower, upper, c
