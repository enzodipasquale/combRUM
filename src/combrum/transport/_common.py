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
            f"{what}: agent_ids must be a 1-D integer array of global agent"
            f" ids; got shape {ids.shape}, dtype {ids.dtype}"
        )
    if ids.size and n_global is not None:
        if int(ids.min()) < 0 or int(ids.max()) >= n_global:
            raise ValueError(
                f"{what}: agent_ids must lie in [0, {n_global});"
                f" got range [{int(ids.min())}, {int(ids.max())}]"
            )
    return ids


def agent_owner_ranks(agent_ids: np.ndarray, n_agents: int, size: int) -> np.ndarray:
    base, extra = divmod(int(n_agents), int(size))
    ranks = np.arange(size, dtype=np.int64)
    starts = ranks * base + np.minimum(ranks, extra)
    return np.searchsorted(starts, agent_ids, side="right") - 1


def owned_agent_ids(n_agents: int, rank: int, size: int) -> np.ndarray:
    start, stop = owned_agent_bounds(n_agents, rank, size)
    return np.arange(start, stop, dtype=np.int64)


def owned_agent_bounds(n_agents: int, rank: int, size: int) -> tuple[int, int]:
    """Half-open global-agent shard bounds owned by ``rank``."""

    if n_agents < 0:
        raise ValueError(f"n_agents must be >= 0; got {n_agents}")
    if size <= 0:
        raise ValueError(f"size must be > 0; got {size}")
    if rank < 0 or rank >= size:
        raise ValueError(f"rank must lie in [0, {size}); got {rank}")
    base, extra = divmod(int(n_agents), int(size))
    start = rank * base + min(rank, extra)
    stop = start + base + (1 if rank < extra else 0)
    return int(start), int(stop)


def route_agent_axis_validated(
    n_agents: Any,
    *,
    size: int,
    source: Any,
    what: str,
) -> tuple[int, int]:
    if (
        isinstance(n_agents, (bool, np.bool_))
        or not isinstance(n_agents, (int, np.integer))
        or int(n_agents) <= 0
    ):
        raise ValueError(
            f"{what}: n_agents must be an integer > 0; got {n_agents!r}"
        )
    if (
        isinstance(source, (bool, np.bool_))
        or not isinstance(source, (int, np.integer))
        or not 0 <= int(source) < int(size)
    ):
        raise ValueError(f"{what}: source must lie in [0, {size}); got {source!r}")
    return int(n_agents), int(source)


def route_local_ids_owned_validated(
    local_ids: object,
    *,
    n_agents: int,
    rank: int,
    size: int,
    what: str,
) -> np.ndarray:
    if not isinstance(local_ids, np.ndarray):
        raise ValueError(
            f"{what}: agent_ids must be a numpy ndarray; got {type(local_ids).__name__}"
        )
    ids = local_ids
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"{what}: agent_ids must be a 1-D integer array of global agent"
            f" ids; got shape {ids.shape}, dtype {ids.dtype}"
        )
    if ids.size:
        if int(ids.min()) < 0 or int(ids.max()) >= int(n_agents):
            raise ValueError(
                f"{what}: agent_ids must lie in [0, {n_agents});"
                f" got range [{int(ids.min())}, {int(ids.max())}]"
            )
        if ids.size > 1 and not bool(np.all(ids[1:] > ids[:-1])):
            raise ValueError(f"{what}: agent_ids must be sorted and unique")
    start, stop = owned_agent_bounds(n_agents, rank, size)
    if ids.size and (int(ids.min()) < start or int(ids.max()) >= stop):
        if start < stop:
            expected_desc = f"[{start}, {stop - 1}]"
        else:
            expected_desc = "empty"
        raise ValueError(
            f"{what}: agent_ids must belong to rank {rank}'s contiguous"
            f" global-agent shard ({expected_desc}); got out-of-shard ids"
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
            f"{what}: source rank {source} must pass a mapping of values"
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
        if isinstance(value, float):
            normalized[gid] = float(value)
            continue
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
    n_agents: int,
    size: int,
    rank: int,
) -> dict[int, float]:
    if not values:
        return {}
    gids = np.fromiter(values.keys(), dtype=np.int64, count=len(values))
    vals = np.fromiter(values.values(), dtype=np.float64, count=len(values))
    order = np.argsort(gids)
    gids = gids[order]
    vals = vals[order]
    mine = agent_owner_ranks(gids, n_agents, size) == int(rank)
    return dict(zip(gids[mine].tolist(), vals[mine].tolist()))
