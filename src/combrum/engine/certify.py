"""Aggregate per-call pricing gaps into a frozen :class:`Certification`."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from combrum.certification import Certification, certification_metadata
from combrum.demand import Demand
from combrum.transport.base import Transport

__all__ = ["GapTally", "certification_metadata"]


class GapTally:
    def __init__(self) -> None:
        self._n_priced: int = 0
        self._n_inexact: int = 0
        self._worst_local: float = 0.0

    def observe(
        self,
        demands: Mapping[int, Demand],
        *,
        n_priced: int | None = None,
        n_inexact: int | None = None,
        worst_gap: float | None = None,
    ) -> None:
        if n_priced is not None:
            if n_inexact is None or worst_gap is None:
                raise ValueError(
                    "n_inexact and worst_gap must be supplied with n_priced"
                )
            self.observe_counts(n_priced, n_inexact, worst_gap)
            return

        priced = len(demands)
        inexact = 0
        worst = 0.0
        for demand in demands.values():
            gap = float(demand.gap)
            if gap > 0.0:
                inexact += 1
                if gap > worst:
                    worst = gap
        self.observe_counts(priced, inexact, worst)

    def observe_counts(self, n_priced: int, n_inexact: int, worst_gap: float) -> None:
        self._n_priced += int(n_priced)
        self._n_inexact += int(n_inexact)
        worst = float(worst_gap)
        if worst > self._worst_local:
            self._worst_local = worst

    def certify(self, transport: Transport) -> Certification:
        ids = np.array([transport.rank], dtype=np.int64)
        rows = np.array([[self._n_priced, self._n_inexact]], dtype=np.float64)
        totals = np.asarray(
            transport.sum_reproducible(rows, ids), dtype=np.float64
        ).reshape(2)
        n_priced = int(round(float(totals[0])))
        n_inexact = int(round(float(totals[1])))
        worst_gap = float(transport.allreduce_max(self._worst_local))
        return Certification(
            n_priced=n_priced, n_inexact=n_inexact, worst_gap=worst_gap
        )
