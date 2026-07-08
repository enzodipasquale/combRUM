"""Self-contained dual payload of one replication's solve.

:class:`DualSolution` holds the active-cut duals, the bundle rows they
price, and the multipliers of the theta-box bounds tight at the solution.
Every accessor computes from stored fields alone, so the payload can be
re-checked or aggregated after the producing master problem is gone.

The parallel-array layout stores one bundle snapshot per payload
(``bundle_table``) with each dual row holding an index into it, keeping
the payload O(rows + table) rather than O(rows x M).

``bound_duals`` lists exactly the coordinates at a box bound (an empty mapping
means no theta coordinate is tight); it stores the box-bound reduced costs the
master reported, for callers that re-check or aggregate them.
"""

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
    """One lazily decoded cut-dual diagnostic row."""

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
    index -> bound reduced cost for the coordinates at a box bound.

    All arrays are copied at construction and frozen read-only so the
    payload cannot alias caller memory. Point-estimate fits use ``rep_id == 0``;
    bootstrap payloads use it as replication provenance.
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

        # np.array (not asarray) copies unconditionally. Integer dtype is
        # required, never coerced: a float->int truncation would corrupt ids.
        agent_ids = np.array(self.agent_ids)
        bundle_row_ids = np.array(self.bundle_row_ids)
        for name, arr in (
            ("agent_ids", agent_ids),
            ("bundle_row_ids", bundle_row_ids),
        ):
            if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"{name} must be a 1-D integer array;"
                    f" got shape {arr.shape}, dtype {arr.dtype}"
                )
        if agent_ids.size and int(agent_ids.min()) < 0:
            raise ValueError(
                f"agent_ids must be >= 0; got {agent_ids[agent_ids < 0].tolist()}"
            )

        pis = np.array(self.pis, dtype=np.float64)
        if pis.ndim != 1:
            raise ValueError(f"pis must be a 1-D array; got shape {pis.shape}")
        if not (agent_ids.shape == bundle_row_ids.shape == pis.shape):
            raise ValueError(
                "agent_ids, bundle_row_ids, pis must be parallel arrays"
                " with one entry per dual row;"
                f" got shapes {agent_ids.shape}, {bundle_row_ids.shape},"
                f" {pis.shape}"
            )
        if not np.isfinite(pis).all():
            raise ValueError(
                f"pis must be finite; got {pis[~np.isfinite(pis)].tolist()}"
            )

        # dtype preserved (bool/int8/... tables are legal); re-encoding
        # would corrupt the snapshot.
        bundle_table = np.array(self.bundle_table)
        if bundle_table.ndim != 2:
            raise ValueError(
                "bundle_table must be 2-D (n_bundles, M);"
                f" got shape {bundle_table.shape}"
            )
        n_bundles = bundle_table.shape[0]
        out_of_range = (bundle_row_ids < 0) | (bundle_row_ids >= n_bundles)
        if out_of_range.any():
            raise ValueError(
                f"bundle_row_ids must index bundle_table rows in"
                f" [0, {n_bundles}); got {bundle_row_ids[out_of_range].tolist()}"
            )

        # Normalized copy (keys to int, values to float) so the caller's
        # dict cannot reach the payload.
        bound_duals: dict[int, float] = {}
        for key, value in self.bound_duals.items():
            coord = operator.index(key)
            if coord < 0:
                raise ValueError(
                    "bound_duals keys are theta coordinate indices and"
                    f" must be >= 0; got {key!r}"
                )
            bound = float(value)
            if not math.isfinite(bound):
                raise ValueError(f"bound_duals[{coord}] must be finite; got {value!r}")
            bound_duals[coord] = bound

        object.__setattr__(self, "agent_ids", _frozen(agent_ids))
        object.__setattr__(self, "bundle_row_ids", _frozen(bundle_row_ids))
        object.__setattr__(self, "pis", _frozen(pis))
        object.__setattr__(self, "bundle_table", _frozen(bundle_table))
        # A plain dict on a frozen dataclass is still mutable in place;
        # the proxy makes it read-only.
        object.__setattr__(self, "bound_duals", MappingProxyType(bound_duals))

    def __getstate__(self) -> dict[str, object]:
        # mappingproxy has no pickle/deepcopy support; ship a plain dict.
        state = dict(self.__dict__)
        state["bound_duals"] = dict(self.bound_duals)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        # Restore what the round trip drops: the read-only proxy and
        # numpy's WRITEABLE=False flags.
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
        # Fancy indexing copies, so the float64 view never aliases the
        # frozen snapshot; n == 0 falls out as zeros(M).
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
                f"n_obs must be an integer greater than zero; got {n_obs!r}"
            ) from exc
        if n < 1:
            raise ValueError(f"n_obs must be greater than zero; got {n}")

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
