"""Shared helpers for concrete master backends."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def validated_u_coefs(
    u_coef: object,
) -> tuple[Callable[[int], float] | None, np.ndarray | None]:
    if callable(u_coef):
        return u_coef, None
    coefs = np.asarray(u_coef)
    if coefs.ndim != 1 or not np.issubdtype(coefs.dtype, np.number):
        raise ValueError(
            "u_coef must be a callable or a 1-D numeric array of per-agent"
            f" coefficients; got {type(u_coef).__name__}"
        )
    coefs = np.ascontiguousarray(coefs, dtype=np.float64)
    if coefs.size and not np.isfinite(coefs).all():
        bad = int(np.flatnonzero(~np.isfinite(coefs))[0])
        raise ValueError(f"u_coef[{bad}] must be finite; got {coefs[bad]!r}")
    coefs.setflags(write=False)
    return None, coefs


def validated_u_coef_cover(u_coefs: np.ndarray | None, n_agents: int | None) -> None:
    if u_coefs is not None and n_agents is not None and u_coefs.size < n_agents:
        raise ValueError(
            f"u_coef array must cover all {n_agents} agents,"
            f" but has only {u_coefs.size} coefficients"
        )


def validated_construction(
    K: int,
    theta_bounds: tuple[np.ndarray, np.ndarray],
    c_theta: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
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
