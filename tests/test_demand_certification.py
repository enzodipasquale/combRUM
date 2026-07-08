from __future__ import annotations

import json
import math

import numpy as np
import pytest

from combrum.certification import Certification
from combrum.demand import Demand, DemandBatch
from combrum.engine.certify import GapTally, certification_metadata
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.result import FitResult
from combrum.transport import SerialTransport, Transport

# --- Demand ----------------------------------------------------------------


def test_exact_demand() -> None:
    d = Demand.exact([1, 0, 1], payoff=2.5)
    assert d.gap == 0.0
    assert type(d.payoff) is float and d.payoff == 2.5
    np.testing.assert_array_equal(d.bundle, np.array([1, 0, 1]))


def test_plain_construction_defaults_to_exact() -> None:
    d = Demand(bundle=np.array([True, False]), payoff=np.float64(1.0))
    assert d.gap == 0.0
    assert type(d.payoff) is float
    assert d.bundle.dtype == bool  # bundle dtype is model-owned, not coerced


def test_inexact_requires_positive_gap() -> None:
    d = Demand.inexact([1], payoff=1.0, gap=1e-4)
    assert d.gap == 1e-4
    # gap == 0 is really an exact call; NaN, inf, and negative gaps
    # cannot certify a finite bound.
    for bad in (0.0, -1e-9, float("nan"), math.inf):
        with pytest.raises(ValueError, match="gap > 0"):
            Demand.inexact([1], payoff=1.0, gap=bad)


def test_uncertified_marks_unknown_or_preserves_finite_gap() -> None:
    d = Demand.uncertified([1], payoff=1.0, gap=0.25)
    assert d.gap == 0.25
    for raw in (None, 0.0, -1e-12, float("nan"), math.inf):
        unknown = Demand.uncertified([1], payoff=1.0, gap=raw)
        assert unknown.gap == math.inf


def test_constructor_rejects_negative_or_nan_gap() -> None:
    for bad in (-0.5, float("nan")):
        with pytest.raises(ValueError, match="gap must be >= 0"):
            Demand(bundle=np.array([1.0]), payoff=0.0, gap=bad)


def test_demand_rejects_nonfinite_payoff() -> None:
    # __post_init__ guards payoff via math.isfinite, so a finite gap
    # cannot smuggle in an inf or NaN payoff.
    for bad in (math.inf, -math.inf, float("nan")):
        with pytest.raises(ValueError, match="payoff must be finite"):
            Demand(bundle=np.array([1.0]), payoff=bad)
        with pytest.raises(ValueError, match="payoff must be finite"):
            Demand.exact([1, 0, 1], payoff=bad)
        with pytest.raises(ValueError, match="payoff must be finite"):
            Demand.inexact([1], payoff=bad, gap=1e-4)
        with pytest.raises(ValueError, match="payoff must be finite"):
            Demand.uncertified([1], payoff=bad)


def test_bundle_stored_read_only() -> None:
    d = Demand.exact(np.array([1.0, 0.0]), payoff=0.0)
    assert not d.bundle.flags.writeable
    with pytest.raises(ValueError):
        d.bundle[0] = 9.0


def test_bundle_rejects_object_dtype() -> None:
    # Read-only flags cannot protect object-array contents, so an object
    # bundle would break the frozen promise.
    with pytest.raises(ValueError, match="non-object"):
        Demand.exact(np.array([{"not": "numeric"}], dtype=object), payoff=0.0)


def test_demand_batch_is_array_backed_mapping() -> None:
    ids = np.array([3, 7], dtype=np.int64)
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    payoffs = np.array([1.25, 2.5])

    batch = DemandBatch.exact(ids, bundles, payoffs)

    assert list(batch) == [3, 7]
    assert len(batch) == 2
    demand = batch[7]
    np.testing.assert_array_equal(demand.bundle, np.array([0.0, 1.0]))
    assert demand.payoff == 2.5
    assert demand.gap == 0.0
    assert not batch.ids.flags.writeable
    assert not batch.bundles.flags.writeable
    assert not batch.payoffs.flags.writeable
    assert not batch.gaps.flags.writeable
    assert batch._index is None


def test_demand_batch_unsorted_unique_ids_lookup_correctly() -> None:
    batch = DemandBatch.exact(
        np.array([7, 3], dtype=np.int64),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([2.5, 1.25]),
    )

    assert batch._index is None
    demand = batch[3]
    assert batch._index == {7: 0, 3: 1}
    np.testing.assert_array_equal(demand.bundle, np.array([1.0, 0.0]))
    assert demand.payoff == 1.25

    # a miss on the dict path must raise, not fall back to a row
    with pytest.raises(KeyError):
        batch[999]


def test_demand_batch_sorted_missing_id_raises_without_lookup_dict() -> None:
    batch = DemandBatch.exact(
        np.array([3, 7], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.25, 2.5]),
    )

    # id 5 falls strictly between the ids (searchsorted -> in-bounds row 1),
    # so this hits the `ids[row] != agent` half of the guard.
    with pytest.raises(KeyError):
        batch[5]
    # id 9 is past max(ids) (searchsorted -> row 2, past the end), so only
    # the `row >= self.ids.size` bounds half can turn this into a KeyError.
    with pytest.raises(KeyError):
        batch[9]
    # id 1 is below min(ids) (searchsorted -> row 0), the other end of the range.
    with pytest.raises(KeyError):
        batch[1]
    assert batch._index is None


def test_demand_batch_rejects_nonfinite_gaps() -> None:
    ids = np.array([3, 7], dtype=np.int64)
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    payoffs = np.array([1.25, 2.5])
    with pytest.raises(ValueError, match="finite"):
        DemandBatch(ids, bundles, payoffs, np.array([0.0, math.inf]))
    # a negative gap is finite, so only the `gaps < 0.0` half can reject it;
    # the guard shares one message for both halves
    with pytest.raises(ValueError, match=">= 0"):
        DemandBatch(ids, bundles, payoffs, np.array([0.0, -1.0]))
    # NaN is non-finite too: the check is np.isfinite, not an inf-only test
    with pytest.raises(ValueError, match="finite"):
        DemandBatch(ids, bundles, payoffs, np.array([0.0, math.nan]))
    # accepting side: a zero gap (exact) and a strictly positive gap are
    # both admitted and stored verbatim
    ok = DemandBatch(ids, bundles, payoffs, np.array([0.0, 0.5]))
    np.testing.assert_array_equal(ok.gaps, np.array([0.0, 0.5]))
    assert ok[3].gap == 0.0 and ok[7].gap == 0.5


def test_demand_batch_rejects_bool_ids() -> None:
    # _coerce_ids rejects bool dtype before integer/float handling;
    # otherwise True/False would silently coerce to ids 1/0.
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    payoffs = np.array([1.25, 2.5])
    gaps = np.zeros(2)
    with pytest.raises(ValueError, match="must be integer ids, not bool"):
        DemandBatch(np.array([True, False]), bundles, payoffs, gaps)
    with pytest.raises(ValueError, match="must be integer ids, not bool"):
        DemandBatch.exact(np.array([True, False]), bundles, payoffs)


def test_demand_batch_rejects_noninteger_float_ids() -> None:
    # Finite non-whole floats fail the trunc check in _coerce_ids
    # (isfinite passes first, so this exercises the trunc branch).
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    payoffs = np.array([1.25, 2.5])
    gaps = np.zeros(2)
    with pytest.raises(ValueError, match=r"must be integer ids$"):
        DemandBatch(np.array([0.5, 1.5]), bundles, payoffs, gaps)
    with pytest.raises(ValueError, match=r"must be integer ids$"):
        DemandBatch.exact(np.array([0.5, 1.5]), bundles, payoffs)


def test_demand_batch_rejects_nonfinite_float_ids() -> None:
    # Non-finite float ids fail the isfinite check in _coerce_ids
    # before the trunc/integer checks are reached.
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    payoffs = np.array([1.25, 2.5])
    gaps = np.zeros(2)
    with pytest.raises(ValueError, match="must be finite integer ids"):
        DemandBatch(np.array([0.0, math.inf]), bundles, payoffs, gaps)
    with pytest.raises(ValueError, match="must be finite integer ids"):
        DemandBatch.exact(np.array([0.0, math.nan]), bundles, payoffs)


# --- Oracle ------------------------------------------------------------------


def test_oracle_setup_defaults_to_no_op_but_pricing_is_required() -> None:
    oracle = Oracle()
    oracle.setup(SerialTransport(), np.arange(0, dtype=np.int64))

    with pytest.raises(NotImplementedError, match="Oracle.price"):
        oracle.price(np.zeros(1), agent_id=0)
    with pytest.raises(NotImplementedError, match="Oracle.price_batch"):
        oracle.price_batch(np.zeros(1), np.array([0], dtype=np.int64))


class _TableOracle(Oracle):
    """Minimal Oracle that prices from a node-shared payoff table."""

    def setup(self, transport: Transport, local_ids: np.ndarray) -> None:
        self._shared = transport.node_shared(
            {"weights": np.array([1.0, 2.0, 3.0])}
        )
        self._local_ids = np.array(local_ids, copy=True)

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        weights = self._shared["weights"]
        payoff = float(weights[agent_id % weights.shape[0]] * theta.sum())
        return Demand.exact(np.array([1.0]), payoff)


def test_oracle_setup_hook_usable_as_documented() -> None:
    oracle = _TableOracle()
    oracle.setup(SerialTransport(), np.arange(3, dtype=np.int64))
    first = oracle.price(np.array([2.0]), agent_id=1)
    again = oracle.price(np.array([2.0]), agent_id=1)
    assert first.payoff == again.payoff == 4.0  # deterministic in (theta, id)
    oracle.teardown()  # default no-op must be callable without override


# --- Certification -----------------------------------------------------------


def test_certification_all_exact() -> None:
    report = Certification(n_priced=5, n_inexact=0, worst_gap=0.0)
    assert (report.n_priced, report.n_inexact, report.worst_gap) == (5, 0, 0.0)


def test_certification_some_inexact() -> None:
    report = Certification(n_priced=5, n_inexact=2, worst_gap=1e-3)
    assert report.worst_gap == 1e-3
    # the finite branch must carry the numeric worst_gap through to metadata
    assert certification_metadata(report) == {
        "n_priced": 5,
        "n_inexact": 2,
        "worst_gap": 1e-3,
        "worst_gap_unknown": False,
    }


def test_gap_tally_counts_only_positive_gaps_with_finite_worst() -> None:
    # the max gap (0.9) sits in the middle, so first/last/min selection would
    # each report a different worst_gap; exact demands do not count as inexact
    tally = GapTally()
    tally.observe(
        {
            0: Demand.exact([1], 1.0),
            1: Demand.inexact([2], 2.0, gap=0.5),
            2: Demand.inexact([3], 3.0, gap=0.9),
            3: Demand.inexact([4], 4.0, gap=0.3),
            4: Demand.exact([5], 5.0),
        }
    )
    report = tally.certify(SerialTransport())
    assert report.n_priced == 5
    assert report.n_inexact == 3
    assert report.worst_gap == 0.9


def test_unknown_gap_certification_metadata_is_strict_json_safe() -> None:
    tally = GapTally()
    tally.observe({0: Demand.uncertified([1], payoff=1.0)})
    report = tally.certify(SerialTransport())
    assert report.n_inexact == 1
    assert report.worst_gap == math.inf

    metadata = certification_metadata(report)
    assert metadata == {
        "n_priced": 1,
        "n_inexact": 1,
        "worst_gap": None,
        "worst_gap_unknown": True,
    }
    fit = FitResult(
        theta_hat=np.array([0.0]),
        objective=0.0,
        empirical_moment=np.array([0.0]),
        runtime_seconds=0.0,
        n_active_cuts=0,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        metadata={"certification": metadata},
    )
    json.dumps(fit.to_dict(), allow_nan=False)


def test_certification_zero_calls_valid() -> None:
    # validation runs against local variables, so check all three stored
    # fields, not just n_priced
    report = Certification(n_priced=0, n_inexact=0, worst_gap=0.0)
    assert (report.n_priced, report.n_inexact, report.worst_gap) == (0, 0, 0.0)


def test_certification_rejects_invalid_field_combinations() -> None:
    # each case is paired with the message its guard must raise; the -1.0/nan
    # cases must hit the >= 0 guard, not the downstream inexact-gap one
    bad = [
        (dict(n_priced=-1, n_inexact=0, worst_gap=0.0), "n_priced must be an integer >= 0"),
        (dict(n_priced=2.0, n_inexact=0, worst_gap=0.0), "n_priced must be an integer >= 0"),
        (dict(n_priced=2, n_inexact=-1, worst_gap=0.0), r"n_inexact must lie in \[0, n_priced\]"),
        (dict(n_priced=2, n_inexact=3, worst_gap=1.0), r"n_inexact must lie in \[0, n_priced\]"),
        (dict(n_priced=2, n_inexact=0, worst_gap=1e-9), "worst_gap must be 0 when every call was exact"),
        (dict(n_priced=2, n_inexact=1, worst_gap=0.0), "worst_gap must be > 0 when some call was inexact"),
        (dict(n_priced=2, n_inexact=1, worst_gap=-1.0), "worst_gap must be >= 0"),
        (dict(n_priced=2, n_inexact=1, worst_gap=float("nan")), "worst_gap must be >= 0"),
    ]
    for kwargs, message in bad:
        with pytest.raises(ValueError, match=message):
            Certification(**kwargs)  # type: ignore[arg-type]
