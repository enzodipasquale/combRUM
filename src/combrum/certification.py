"""Frozen contract for the exactness report of a pricing sweep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Certification:
    """Aggregate exactness of the pricing calls behind one result.

    ``math.inf`` means at least one inexact call had an
    unknown finite bound gap.
    """

    n_priced: int
    n_inexact: int
    worst_gap: float

    def __post_init__(self) -> None:
        if not isinstance(self.n_priced, (int, np.integer)) or self.n_priced < 0:
            raise ValueError(f"n_priced must be an integer >= 0; got {self.n_priced!r}")
        object.__setattr__(self, "n_priced", int(self.n_priced))
        if not isinstance(self.n_inexact, (int, np.integer)):
            raise ValueError(f"n_inexact must be an integer; got {self.n_inexact!r}")
        if not 0 <= self.n_inexact <= self.n_priced:
            raise ValueError(
                "n_inexact must lie in [0, n_priced] ="
                f" [0, {self.n_priced}]; got {self.n_inexact}"
            )
        object.__setattr__(self, "n_inexact", int(self.n_inexact))
        worst_gap = float(self.worst_gap)
        if not worst_gap >= 0.0:
            raise ValueError(f"worst_gap must be >= 0; got {worst_gap}")
        if self.n_inexact == 0 and worst_gap != 0.0:
            raise ValueError(
                "worst_gap must be 0 when every call was exact"
                f" (n_inexact = 0); got {worst_gap}"
            )
        if self.n_inexact > 0 and not worst_gap > 0.0:
            raise ValueError(
                "worst_gap must be > 0 when some call was inexact"
                f" (n_inexact = {self.n_inexact}); got {worst_gap}"
            )
        object.__setattr__(self, "worst_gap", worst_gap)


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
