"""Aggregate per-call pricing gaps into a frozen :class:`Certification`.

Each priced :class:`~combrum.demand.Demand` carries the certified
optimality ``gap`` of that pricing call. A :class:`GapTally` accumulates
rank-local counts of those gaps; :meth:`GapTally.certify` reduces across
ranks (rank-keyed SUM of counts, MAX of worst gap) into the report.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from combrum.certification import Certification
from combrum.demand import Demand
from combrum.transport.base import Transport


class GapTally:
    """Rank-local accumulator of per-call pricing gaps, reduced on demand.

    Each :meth:`observe` folds one iteration's priced demands into scalar
    rank-local counts ``(priced, inexact)`` plus a rank-local worst gap;
    :meth:`certify` reduces those across ranks. ``n_priced`` counts total
    pricing calls (an agent priced once per scheduled iteration), not
    distinct agents.
    """

    def __init__(self) -> None:
        self._n_priced: int = 0
        self._n_inexact: int = 0
        # 0.0 on an empty shard so an idle rank contributes 0.0 to global MAX.
        self._worst_local: float = 0.0

    def observe(
        self,
        demands: Mapping[int, Demand],
        *,
        n_priced: int | None = None,
        n_inexact: int | None = None,
        worst_gap: float | None = None,
    ) -> None:
        """Fold one price phase's ``{global_id: Demand}`` into the tally.

        Every call is counted; a ``gap > 0`` call is also counted inexact
        and raises the rank-local worst gap. Pre-counted totals may be
        passed via ``n_priced``/``n_inexact``/``worst_gap`` instead.
        """
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

    def observe_counts(
        self, n_priced: int, n_inexact: int, worst_gap: float
    ) -> None:
        """Fold pre-counted price-phase gap totals into the tally."""
        self._n_priced += int(n_priced)
        self._n_inexact += int(n_inexact)
        worst = float(worst_gap)
        if worst > self._worst_local:
            self._worst_local = worst

    def certify(self, transport: Transport) -> Certification:
        """Reduce the rank-local tally across ranks into a Certification.

        Counts reduce with the transport's deterministic rank-keyed SUM;
        the worst gap with an order-independent MAX.
        """
        ids = np.array([transport.rank], dtype=np.int64)
        rows = np.array([[self._n_priced, self._n_inexact]], dtype=np.float64)
        # Counts are exact integers carried as float64 only to ride the
        # deterministic SUM kernel; rounding back is lossless.
        totals = np.asarray(
            transport.sum_reproducible(rows, ids), dtype=np.float64
        ).reshape(2)
        n_priced = int(round(float(totals[0])))
        n_inexact = int(round(float(totals[1])))
        worst_gap = float(transport.allreduce_max(self._worst_local))
        # Certification re-asserts the cross-field honesty invariant, so a
        # lossy reduce fails here rather than publishing a contradictory triple.
        return Certification(
            n_priced=n_priced, n_inexact=n_inexact, worst_gap=worst_gap
        )


def certification_metadata(certification: Certification) -> dict[str, object]:
    """JSON-plain dict of a Certification for ``FitResult.metadata``.

    Returns native ints/float, never the typed object: ``to_dict()`` passes
    ``metadata`` through unchanged, so a typed object there would break
    ``json.dumps(fit.to_dict())``. Unknown bound gaps are encoded without
    non-finite floats so callers can use strict JSON.
    """
    worst_gap = float(certification.worst_gap)
    worst_gap_unknown = not np.isfinite(worst_gap)
    return {
        "n_priced": int(certification.n_priced),
        "n_inexact": int(certification.n_inexact),
        "worst_gap": None if worst_gap_unknown else worst_gap,
        "worst_gap_unknown": bool(worst_gap_unknown),
    }
