"""Self-contained dual payload of one replication's solve."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CutDualRow:
    agent_id: int
    observation_id: int
    simulation_id: int
    generated_bundle: np.ndarray
    pi: float


@dataclass(frozen=True)
class DualSolution:
    """One replication's dual measure, stored as parallel arrays.

    ``agent_ids``, ``bundle_row_ids`` and ``pis`` are parallel ``(n,)``
    arrays: row ``r`` says agent ``agent_ids[r]`` puts dual mass
    ``pis[r]`` on bundle ``bundle_table[bundle_row_ids[r]]``. The table
    has shape ``(n_bundles, M)``. ``bound_duals`` maps theta coordinate
    index -> bound reduced cost for exactly the coordinates at a box
    bound (empty when none is tight).
    """

    rep_id: int
    agent_ids: np.ndarray
    bundle_row_ids: np.ndarray
    pis: np.ndarray
    bundle_table: np.ndarray
    bound_duals: Mapping[int, float]

    def __post_init__(self) -> None:
        rep_id = operator.index(self.rep_id)
        if rep_id < 0:
            raise ValueError(f"rep_id must be >= 0; got {rep_id}")
        object.__setattr__(self, "rep_id", rep_id)

        agent_ids = np.array(self.agent_ids)
        bundle_row_ids = np.array(self.bundle_row_ids)
        for name, arr in (
            ("agent_ids", agent_ids),
            ("bundle_row_ids", bundle_row_ids),
        ):
            if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"expected a 1-D integer array for {name},"
                    f" got shape {arr.shape}, dtype {arr.dtype}"
                )
        if agent_ids.size and int(agent_ids.min()) < 0:
            raise ValueError(
                f"agent_ids must be >= 0, got {agent_ids[agent_ids < 0].tolist()}"
            )

        pis = np.array(self.pis, dtype=np.float64)
        if pis.ndim != 1:
            raise ValueError(f"expected a 1-D pis array, got shape {pis.shape}")
        if not (agent_ids.shape == bundle_row_ids.shape == pis.shape):
            raise ValueError(
                "agent_ids, bundle_row_ids, pis must be parallel arrays"
                " with one entry per dual row,"
                f" got shapes {agent_ids.shape}, {bundle_row_ids.shape},"
                f" {pis.shape}"
            )
        if not np.isfinite(pis).all():
            raise ValueError(
                f"pis must be finite, got {pis[~np.isfinite(pis)].tolist()}"
            )

        bundle_table = np.array(self.bundle_table)
        if bundle_table.ndim != 2:
            raise ValueError(
                "expected a 2-D (n_bundles, M) bundle_table,"
                f" got shape {bundle_table.shape}"
            )
        n_bundles = bundle_table.shape[0]
        out_of_range = (bundle_row_ids < 0) | (bundle_row_ids >= n_bundles)
        if out_of_range.any():
            raise ValueError(
                f"bundle_row_ids must index bundle_table rows in"
                f" [0, {n_bundles}), got {bundle_row_ids[out_of_range].tolist()}"
            )

        bound_duals: dict[int, float] = {}
        for key, value in self.bound_duals.items():
            coord = operator.index(key)
            if coord < 0:
                raise ValueError(
                    "bound_duals keys are theta coordinate indices and"
                    f" must be >= 0, got {key}"
                )
            bound = float(value)
            if not math.isfinite(bound):
                raise ValueError(f"bound_duals[{coord}] must be finite, got {value!r}")
            bound_duals[coord] = bound

        object.__setattr__(self, "agent_ids", _frozen(agent_ids))
        object.__setattr__(self, "bundle_row_ids", _frozen(bundle_row_ids))
        object.__setattr__(self, "pis", _frozen(pis))
        object.__setattr__(self, "bundle_table", _frozen(bundle_table))
        object.__setattr__(self, "bound_duals", MappingProxyType(bound_duals))

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["bound_duals"] = dict(self.bound_duals)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        object.__setattr__(self, "bound_duals", MappingProxyType(self.bound_duals))
        for name in ("agent_ids", "bundle_row_ids", "pis", "bundle_table"):
            _frozen(self.__dict__[name])

    def with_rep_id(self, rep_id: int) -> "DualSolution":
        """Return this validated payload re-keyed to ``rep_id`` without copies."""
        rep = operator.index(rep_id)
        if rep < 0:
            raise ValueError(f"rep_id must be >= 0; got {rep}")
        clone = object.__new__(type(self))
        object.__setattr__(clone, "rep_id", rep)
        object.__setattr__(clone, "agent_ids", self.agent_ids)
        object.__setattr__(clone, "bundle_row_ids", self.bundle_row_ids)
        object.__setattr__(clone, "pis", self.pis)
        object.__setattr__(clone, "bundle_table", self.bundle_table)
        object.__setattr__(clone, "bound_duals", self.bound_duals)
        return clone

    def moment(self) -> np.ndarray:
        """Dual-weighted bundle aggregate, shape ``(M,)`` float64.

        ``sum_r pis[r] * bundle_table[bundle_row_ids[r]]``. An empty
        payload yields ``zeros(M)``, with M read off the stored table.
        """
        rows = self.bundle_table[self.bundle_row_ids].astype(np.float64, copy=False)
        return self.pis @ rows

    def rows(self, *, n_obs: int) -> Iterator[CutDualRow]:
        """Lazily decoded diagnostic rows for a point-estimate dual payload.

        ``n_obs`` is the number of observations in the fit. The global agent id
        uses the package convention
        ``a = simulation_id * n_obs + observation_id``.
        """
        try:
            n = operator.index(n_obs)
        except TypeError as exc:
            raise ValueError(
                f"n_obs must be a positive integer, got {n_obs!r}"
            ) from exc
        if n < 1:
            raise ValueError(f"n_obs must be positive, got {n}")

        def _iter_rows() -> Iterator[CutDualRow]:
            for agent_id, bundle_row_id, pi in zip(
                self.agent_ids,
                self.bundle_row_ids,
                self.pis,
            ):
                agent = int(agent_id)
                bundle_row = int(bundle_row_id)
                yield CutDualRow(
                    agent_id=agent,
                    observation_id=agent % n,
                    simulation_id=agent // n,
                    generated_bundle=self.bundle_table[bundle_row],
                    pi=float(pi),
                )

        return _iter_rows()
