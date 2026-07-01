"""Shared validation helpers for transport implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def scatter_arrays_validated(
    arrays: object,
) -> tuple[dict[str, np.ndarray], int | None]:
    if not isinstance(arrays, dict):
        raise ValueError(
            "scatter_by_agent: root must pass a dict of arrays;"
            f" got {type(arrays).__name__}"
        )
    normalized: dict[str, np.ndarray] = {}
    n_global: int | None = None
    for key, value in arrays.items():
        if not isinstance(key, str):
            raise ValueError(f"scatter_by_agent: array keys must be str; got {key!r}")
        full = np.asarray(value)
        if full.ndim < 1:
            raise ValueError(
                f"scatter_by_agent: array {key!r} must have at least one"
                " dimension (axis 0 is the global agent axis)"
            )
        if full.dtype == object:
            raise ValueError(
                f"scatter_by_agent: array {key!r} must be numeric; got dtype object"
            )
        if n_global is None:
            n_global = full.shape[0]
        elif full.shape[0] != n_global:
            raise ValueError(
                "scatter_by_agent: all arrays must share the axis-0 length"
                f" (the global agent axis); {key!r} has {full.shape[0]},"
                f" expected {n_global}"
            )
        normalized[key] = full
    return normalized, n_global


def ids_validated(local_ids: object, n_global: int | None, what: str) -> np.ndarray:
    ids = np.asarray(local_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"{what}: local_ids must be a 1-D integer array of global agent"
            f" ids; got shape {ids.shape}, dtype {ids.dtype}"
        )
    if ids.size and n_global is not None:
        if int(ids.min()) < 0 or int(ids.max()) >= n_global:
            raise ValueError(
                f"{what}: local_ids must lie in [0, {n_global});"
                f" got range [{int(ids.min())}, {int(ids.max())}]"
            )
    return ids


def observation_owner_rank(obs_id: int, n_observations: int, size: int) -> int:
    """Rank owning an observation under contiguous observation sharding."""
    if n_observations <= 0:
        raise ValueError(f"n_observations must be > 0; got {n_observations}")
    if size <= 0:
        raise ValueError(f"size must be > 0; got {size}")
    obs = int(obs_id)
    if obs < 0 or obs >= n_observations:
        raise ValueError(f"obs_id must lie in [0, {n_observations}); got {obs_id}")
    base, extra = divmod(int(n_observations), int(size))
    front = (base + 1) * extra
    if obs < front:
        return obs // (base + 1)
    return extra + (obs - front) // base


def agent_owner_rank(agent_id: int, n_observations: int, size: int) -> int:
    """Rank owning an agent id when ``agent_id % N`` is its observation."""
    return observation_owner_rank(
        int(agent_id) % int(n_observations), int(n_observations), int(size)
    )


def route_geometry_validated(
    n_observations: Any,
    n_simulations: Any,
    *,
    size: int,
    source: Any,
    what: str,
) -> tuple[int, int, int, int]:
    if (
        isinstance(n_observations, (bool, np.bool_))
        or not isinstance(n_observations, (int, np.integer))
        or int(n_observations) <= 0
    ):
        raise ValueError(
            f"{what}: n_observations must be an integer > 0; got {n_observations!r}"
        )
    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, (int, np.integer))
        or int(n_simulations) <= 0
    ):
        raise ValueError(
            f"{what}: n_simulations must be an integer > 0; got {n_simulations!r}"
        )
    if (
        isinstance(source, (bool, np.bool_))
        or not isinstance(source, (int, np.integer))
        or not 0 <= int(source) < int(size)
    ):
        raise ValueError(f"{what}: source must lie in [0, {size}); got {source!r}")
    n_obs = int(n_observations)
    n_sims = int(n_simulations)
    return n_obs, n_sims, n_obs * n_sims, int(source)


def route_local_ids_shape_validated(
    local_ids: object,
    *,
    what: str,
) -> np.ndarray:
    if not isinstance(local_ids, np.ndarray):
        raise ValueError(
            f"{what}: local_ids must be a numpy ndarray; got {type(local_ids).__name__}"
        )
    ids = local_ids
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"{what}: local_ids must be a 1-D integer array of global agent"
            f" ids; got shape {ids.shape}, dtype {ids.dtype}"
        )
    return ids


def route_values_validated(
    values: object,
    *,
    n_agents: int,
    rank: int,
    source: int,
    what: str,
) -> dict[int, float]:
    if rank != source:
        if values is not None:
            raise ValueError(
                f"{what}: only source rank {source} may pass values;"
                f" rank {rank} passed a payload"
            )
        return {}
    if values is None:
        raise ValueError(
            f"{what}: source rank {source} must pass a mapping of values; got None"
        )
    if not isinstance(values, Mapping):
        raise ValueError(
            f"{what}: values must be a mapping from agent id to float;"
            f" got {type(values).__name__}"
        )
    normalized: dict[int, float] = {}
    for key, value in values.items():
        if (
            isinstance(key, (bool, np.bool_))
            or not isinstance(key, (int, np.integer))
            or int(key) < 0
            or int(key) >= int(n_agents)
        ):
            raise ValueError(
                f"{what}: value keys must lie in [0, {n_agents}); got {key!r}"
            )
        gid = int(key)
        if gid in normalized:
            raise ValueError(f"{what}: duplicate value key {gid}")
        scalar = np.asarray(value)
        if scalar.shape != ():
            raise ValueError(
                f"{what}: value for agent {gid} must be scalar;"
                f" got shape {scalar.shape}"
            )
        try:
            normalized[gid] = float(np.float64(scalar))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{what}: value for agent {gid} must be convertible to"
                f" float64; got {value!r}"
            ) from exc
    return normalized


def route_bucket_for_rank(
    values: Mapping[int, float],
    *,
    n_observations: int,
    size: int,
    rank: int,
) -> dict[int, float]:
    return {
        int(gid): float(value)
        for gid, value in sorted(values.items())
        if agent_owner_rank(int(gid), n_observations, size) == int(rank)
    }
