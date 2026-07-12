"""Priced-subproblem outcome types. Outcomes are validated at construction."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np


def _strictly_increasing(values: np.ndarray) -> bool:
    return values.size < 2 or bool(np.all(values[1:] > values[:-1]))


def _coerce_ids(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"expected 1-D ids, got shape {raw.shape}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("ids must be integer ids, not bool")
    if not isinstance(values, np.ndarray):
        obj = np.asarray(values, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in obj.flat):
            raise ValueError("ids must be integer ids, not bool")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size:
            if int(raw.max()) > np.iinfo(np.int64).max:
                raise ValueError("ids exceed int64 range")
        return np.asarray(raw, dtype=np.int64)
    if np.issubdtype(raw.dtype, np.floating):
        if np.any(~np.isfinite(raw)):
            raise ValueError("ids must be finite integer ids")
        if np.any(raw != np.trunc(raw)):
            raise ValueError("ids must be integer ids")
        if raw.size and (
            float(raw.min()) < np.iinfo(np.int64).min or float(raw.max()) >= 2.0**63
        ):
            raise ValueError("ids exceed int64 range")
        return raw.astype(np.int64)
    raise ValueError(f"ids must be integer ids; got dtype {raw.dtype}")


def _coerce_id_key(agent_id: int) -> int:
    if isinstance(agent_id, (bool, np.bool_)):
        raise KeyError(agent_id)
    if isinstance(agent_id, (int, np.integer)):
        return int(agent_id)
    if isinstance(agent_id, (float, np.floating)):
        value = float(agent_id)
        if math.isfinite(value) and value.is_integer():
            if np.iinfo(np.int64).min <= value < 2.0**63:
                return int(value)
    raise KeyError(agent_id)


@dataclass(frozen=True)
class Demand:
    """One agent's priced outcome: chosen bundle, payoff, certified gap.

    ``bundle`` is the chosen bundle (shape/dtype are model-owned, stored
    read-only). ``payoff`` is its achieved subproblem value at the queried
    theta. ``gap`` is the certified optimality gap of this pricing call,
    with ``0.0`` meaning proven exact and ``math.inf`` meaning an incumbent
    was found but the oracle has no usable finite certificate.
    """

    bundle: np.ndarray
    payoff: float
    gap: float = 0.0

    def __post_init__(self) -> None:
        bundle = np.asarray(self.bundle)
        if bundle.dtype == object:
            # Read-only flags can't protect object-array contents; frozen would break.
            raise ValueError("bundle must be a non-object ndarray payload")
        bundle.setflags(write=False)
        object.__setattr__(self, "bundle", bundle)
        payoff = float(self.payoff)
        if not math.isfinite(payoff):
            raise ValueError(f"payoff must be finite; got {payoff}")
        object.__setattr__(self, "payoff", payoff)
        gap = float(self.gap)
        # "not >=" also rejects NaN (compares False both ways).
        if not gap >= 0.0:
            raise ValueError(f"gap must be >= 0 (0.0 = proven exact); got {gap}")
        object.__setattr__(self, "gap", gap)

    @classmethod
    def exact(cls, bundle: np.ndarray, payoff: float) -> Demand:
        """Outcome of a pricing call proven optimal (``gap == 0``)."""
        return cls(bundle=bundle, payoff=payoff, gap=0.0)

    @classmethod
    def inexact(cls, bundle: np.ndarray, payoff: float, gap: float) -> Demand:
        """Outcome with a finite certified positive optimality gap."""
        gap_value = float(gap)
        if not math.isfinite(gap_value) or gap_value <= 0.0:
            raise ValueError(f"inexact requires finite gap > 0; got {gap}")
        return cls(bundle=bundle, payoff=payoff, gap=gap_value)

    @classmethod
    def uncertified(
        cls, bundle: np.ndarray, payoff: float, *, gap: float | None = None
    ) -> Demand:
        """Feasible incumbent without proof of exactness.

        ``gap`` is an optional bound: a finite positive value is kept as a
        certificate, while missing, zero, negative, NaN, and infinite gaps
        become ``math.inf``.
        """
        if gap is None:
            gap_value = math.inf
        else:
            raw = float(gap)
            gap_value = raw if math.isfinite(raw) and raw > 0.0 else math.inf
        return cls(bundle=bundle, payoff=payoff, gap=gap_value)


@dataclass(frozen=True)
class DemandBatch(Mapping[int, Demand]):
    """Array-backed batch of priced outcomes.

    Vectorized ``Mapping[int, Demand]`` for ``Oracle.price_batch``: keeps
    ``ids``, ``bundles``, ``payoffs``, and ``gaps`` as arrays rather than
    materializing one :class:`Demand` per agent.
    """

    ids: np.ndarray
    bundles: np.ndarray
    payoffs: np.ndarray
    gaps: np.ndarray
    _index: dict[int, int] | None = field(default=None, init=False, repr=False)
    _ids_strictly_increasing: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        ids = _coerce_ids(self.ids)
        ids_strictly_increasing = _strictly_increasing(ids)
        if not ids_strictly_increasing:
            unique, counts = np.unique(ids, return_counts=True)
            if unique.size != ids.size:
                raise ValueError(
                    f"ids contain duplicates: {unique[counts > 1].tolist()}"
                )

        bundles = np.asarray(self.bundles)
        if bundles.ndim < 1 or bundles.shape[0] != ids.size:
            raise ValueError(
                "expected bundles with leading dimension len(ids) ="
                f" {ids.size}, got shape {bundles.shape}"
            )

        payoffs = np.asarray(self.payoffs, dtype=np.float64)
        gaps = np.asarray(self.gaps, dtype=np.float64)
        if payoffs.shape != (ids.size,):
            raise ValueError(
                f"expected payoffs of shape ({ids.size},), got {payoffs.shape}"
            )
        if gaps.shape != (ids.size,):
            raise ValueError(f"expected gaps of shape ({ids.size},), got {gaps.shape}")
        if np.any(~np.isfinite(payoffs)):
            raise ValueError("payoffs must be finite")
        if np.any(~np.isfinite(gaps)) or np.any(gaps < 0.0):
            raise ValueError("gaps must be finite values >= 0")

        ids.setflags(write=False)
        bundles.setflags(write=False)
        payoffs.setflags(write=False)
        gaps.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "bundles", bundles)
        object.__setattr__(self, "payoffs", payoffs)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "_ids_strictly_increasing", ids_strictly_increasing)

    @classmethod
    def exact(
        cls, ids: np.ndarray, bundles: np.ndarray, payoffs: np.ndarray
    ) -> DemandBatch:
        """Batch whose bundles are all exact optima (every gap zero)."""
        ids_arr = _coerce_ids(ids)
        return cls(
            ids=ids_arr,
            bundles=bundles,
            payoffs=payoffs,
            gaps=np.zeros(ids_arr.shape, dtype=np.float64),
        )

    def __len__(self) -> int:
        return int(self.ids.size)

    def __iter__(self) -> Iterator[int]:
        return (int(agent_id) for agent_id in self.ids)

    def __getitem__(self, agent_id: int) -> Demand:
        agent = _coerce_id_key(agent_id)
        if self._ids_strictly_increasing:
            row = int(np.searchsorted(self.ids, agent))
            if row >= self.ids.size or int(self.ids[row]) != agent:
                raise KeyError(agent_id)
            index = row
        else:
            index = self._lookup().get(agent)
            if index is None:
                raise KeyError(agent_id)
        return Demand(
            self.bundles[index],
            float(self.payoffs[index]),
            float(self.gaps[index]),
        )

    def _lookup(self) -> dict[int, int]:
        index = self._index
        if index is None:
            index = {int(agent_id): row for row, agent_id in enumerate(self.ids)}
            object.__setattr__(self, "_index", index)
        return index
