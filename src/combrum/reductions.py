"""Deterministic reduction kernel, free of any transport dependency.

Float addition is not associative, so an arrival-order sum depends on rank
count and message timing. :func:`canonical_sum` sorts contributions by global
agent id before reducing, making each sum a pure function of the (id, value)
pairs regardless of sharding or interleaving.
"""

from __future__ import annotations

import numpy as np

_CANONICAL_SUM_WINDOW_BYTES = 64 * 1024 * 1024


def canonical_sum_window_rows(width: int) -> int:
    """Maximum global-id window width for memory-bounded row-keyed sums."""
    w = max(1, int(width))
    return max(1, _CANONICAL_SUM_WINDOW_BYTES // (8 * (w + 1)))


def canonical_sum(values: np.ndarray, global_ids: np.ndarray) -> np.ndarray | float:
    """Sum contributions in canonical (ascending global-id) order.

    Parameters
    ----------
    values:
        Contributions, shape ``(n,)`` or ``(n, M)``; converted to float64.
        ``(n, M)`` input is summed over axis 0 after sorting, one value per
        column.
    global_ids:
        Shape ``(n,)`` integer global agent ids, one per contribution row.
        Ids must be unique within a call (pre-combine per-agent rows locally);
        repeated ids would reintroduce the input-order dependence this kernel
        removes.

    Returns
    -------
    ``float`` for ``(n,)`` input, a fresh ``(M,)`` float64 array for
    ``(n, M)`` input. Result is bitwise identical under any permutation of the
    input rows.
    """
    ids = np.asarray(global_ids)
    if ids.ndim != 1:
        raise ValueError(f"expected one-dimensional global_ids, got shape {ids.shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"expected an integer dtype for global_ids, got {ids.dtype}"
        )
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim not in (1, 2):
        raise ValueError(
            f"values must have shape (n,) or (n, M); got shape {vals.shape}"
        )
    if vals.shape[0] != ids.shape[0]:
        raise ValueError(
            f"values has {vals.shape[0]} rows but global_ids has"
            f" {ids.shape[0]} entries; one id per contribution row required"
        )
    sorted_unique = ids.size < 2 or bool(np.all(ids[1:] > ids[:-1]))
    if sorted_unique:
        canonical_ids = ids
        canonical = vals
    else:
        unique, counts = np.unique(ids, return_counts=True)
        if unique.size != ids.size:
            raise ValueError(
                "global_ids must be unique within one canonical_sum call"
                f" (pre-combine per-agent contributions rank-locally);"
                f" duplicated ids: {unique[counts > 1].tolist()}"
            )
        order = np.argsort(ids, kind="stable")
        # Fancy indexing materializes a fresh contiguous array so the reduce
        # sees identical bytes/layout for any input ordering.
        canonical_ids = ids[order]
        canonical = vals[order]
    width = int(canonical.shape[1]) if vals.ndim == 2 else 1
    window_rows = canonical_sum_window_rows(width)
    span = int(canonical_ids[-1]) - int(canonical_ids[0]) + 1 if ids.size else 0
    if span <= window_rows:
        total = np.add.reduce(canonical, axis=0)
    else:
        total = np.zeros(width, dtype=np.float64)
        left = 0
        lo = int(canonical_ids[0])
        stop = int(canonical_ids[-1]) + 1
        while lo < stop:
            hi = lo + window_rows
            right = int(np.searchsorted(canonical_ids, hi, side="left"))
            if right > left:
                total += np.atleast_1d(np.add.reduce(canonical[left:right], axis=0))
            left = right
            lo = hi
    if vals.ndim == 1:
        return float(np.asarray(total, dtype=np.float64).reshape(-1)[0])
    return total
