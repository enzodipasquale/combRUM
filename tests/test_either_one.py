"""The either-one resolution guard + the price/features either-one surfaces.

Three layers, all on the in-process transport doubles:

* the generic guard (:mod:`combrum.interface_resolution`) on a tiny synthetic ABC —
  at-least-one rejection, whole-MRO override detection, class-pure dispatch
  (instance monkeypatch ignored), the both-supplied conformance gate, and
  rank-agreement raising when ranks resolve differently;
* the price either-one on the family oracles — price-only / price_batch-only
  / both-supplied (conformance + divergent fail), and the
  price_batch(theta, ids) == [price(theta, i)] conformance on toy and QKP;
* the features either-one on both formulations end-to-end — bare-callable
  byte-for-byte vs a FeatureMap providing features_batch, on toy and QKP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _family_oracles import (
    QKPOracle,
    ToyOracle,
    divergent_feature_map,
    qkp_feature_map,
    qkp_feature_map_batch_only,
    qkp_problem,
    toy_feature_map,
    toy_feature_map_batch_only,
    toy_problem,
)
from _walk import run_walk
from combrum.demand import Demand
from combrum.engine.fitstep import fit_step
from combrum.interface_resolution import (
    CONTINUOUS_TOL,
    FeatureMap,
    Mode,
    Resolution,
    assert_conforms,
    dispatch,
    feature_batch_aggregate,
    feature_rows,
    price_demands,
    resolve,
    resolve_features,
    resolve_local,
    supports_feature_batch_aggregate,
)
from combrum.formulations import NSlack, OneSlack
from combrum.rowgen import MaxContribution
from _support.families import load_family
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.oracle import Oracle
from combrum.transport import LocalCluster, SerialTransport, TransportError
from combrum.transport.base import Transport

FAMILY_DIR = Path(__file__).resolve().parent / "fixtures" / "families"

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)
REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param("highs", marks=needs_highs),
)

# Apple Accelerate raises spurious FP-status warnings on provably finite
# matmuls at these sizes; the feature maps are guarded regardless.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::RuntimeWarning:.*matmul.*"
    ),
]


def load_toy() -> dict[str, np.ndarray]:
    return load_family("toy", FAMILY_DIR)


def load_qkp() -> dict[str, np.ndarray]:
    return load_family("qkp", FAMILY_DIR)


# --- the generic guard on a synthetic either-one ABC -------------------------


class _Surface:
    """A minimal either-one ABC: ``plain`` (default) | ``fast`` (optimized).

    Both members raise by default; resolution must pick whichever is
    overridden. A surface with neither overridden is the rejection probe.
    """

    def plain(self, x: int) -> int:
        raise NotImplementedError

    def fast(self, x: int) -> int:
        raise NotImplementedError


class _PlainOnly(_Surface):
    def plain(self, x: int) -> int:
        return x + 1


class _FastOnly(_Surface):
    def fast(self, x: int) -> int:
        return x * 10


class _Both(_Surface):
    def plain(self, x: int) -> int:
        return x + 1

    def fast(self, x: int) -> int:
        return x + 1  # agrees with plain by construction


class _NeitherOverridden(_Surface):
    """Inherits only the two raising defaults — the at-least-one reject."""


class _PlainViaIntermediate(_PlainOnly):
    """Override lives on an intermediate base: whole-MRO detection must see
    it, where a ``cls.__dict__`` check would miss it."""


def _resolve_surface(instance: _Surface, transport: Transport) -> Resolution:
    return resolve(
        instance,
        surface="probe",
        default_name="plain",
        optimized_name="fast",
        default_func=_Surface.plain,
        optimized_func=_Surface.fast,
        transport=transport,
    )


def test_at_least_one_rejects_base_only() -> None:
    with pytest.raises(TypeError, match="must override"):
        _resolve_surface(_NeitherOverridden(), SerialTransport())


def test_plain_only_resolves_default() -> None:
    res = _resolve_surface(_PlainOnly(), SerialTransport())
    assert res.mode is Mode.DEFAULT
    assert not res.runs_optimized
    assert res.reference is None
    assert res.active(5) == 6


def test_fast_only_resolves_optimized() -> None:
    res = _resolve_surface(_FastOnly(), SerialTransport())
    assert res.mode is Mode.OPTIMIZED
    assert res.runs_optimized
    assert res.reference is None
    assert res.active(5) == 50


def test_both_resolves_to_optimized_with_reference() -> None:
    inst = _Both()
    res = _resolve_surface(inst, SerialTransport())
    assert res.mode is Mode.BOTH
    assert res.runs_optimized
    # Both members compute x+1, so value checks alone cannot tell which physical
    # member got bound where. Pin the binding by identity: active must be the
    # OPTIMIZED (fast) member bound to this instance, reference the per-agent
    # (plain) member. A swap of either binding is caught here.
    assert res.active.__func__ is _Both.fast
    assert res.active.__self__ is inst
    assert res.reference is not None
    assert res.reference.__func__ is _Both.plain
    assert res.reference.__self__ is inst
    assert res.active(5) == 6
    assert res.reference(5) == 6


def test_whole_mro_override_detected_through_intermediate_base() -> None:
    # The override lives on _PlainOnly, reached via the MRO from a subclass
    # that adds nothing — a cls.__dict__ check would miss it.
    res = _resolve_surface(_PlainViaIntermediate(), SerialTransport())
    assert res.mode is Mode.DEFAULT
    assert res.active(7) == 8


def test_instance_monkeypatch_is_ignored() -> None:
    # Class-pure dispatch: an instance attribute shadowing the member must
    # not change the resolved callable (resolution binds the class function).
    inst = _PlainOnly()
    inst.plain = lambda x: 999  # type: ignore[method-assign]
    res = _resolve_surface(inst, SerialTransport())
    assert res.mode is Mode.DEFAULT
    assert res.active(5) == 6


def test_resolution_token_is_surface_qualname_mode() -> None:
    res = resolve_local(
        _Both(),
        surface="probe",
        default_name="plain",
        optimized_name="fast",
        default_func=_Surface.plain,
        optimized_func=_Surface.fast,
    )
    assert res.token == ("probe", "test_either_one", "_Both", "both")


# --- rank-agreement: a divergent build is the agreed transport verdict -------


def test_rank_agreement_raises_when_ranks_resolve_differently() -> None:
    # Rank 0 wires the both-member class, rank 1 the plain-only class: the
    # resolved modes differ, so the agreement round raises the same verdict
    # on every rank (inside the collective).
    def per_rank(transport: Transport) -> str:
        instance = _Both() if transport.rank == 0 else _PlainOnly()
        try:
            _resolve_surface(instance, transport)
        except TransportError as exc:
            return exc.message
        return "no-error"

    outcomes = LocalCluster(2).run(per_rank)
    assert all("disagrees across ranks" in o for o in outcomes), outcomes


def test_rank_agreement_passes_when_ranks_agree() -> None:
    # Same class on every rank (an empty-shard rank included): the agreement
    # round is transparent and every rank resolves identically.
    def per_rank(transport: Transport) -> tuple[str, str, str]:
        return _resolve_surface(_FastOnly(), transport).token

    tokens = LocalCluster(4).run(per_rank)
    assert len(set(tokens)) == 1
    assert tokens[0] == ("probe", "test_either_one", "_FastOnly", "optimized")


# --- the exhaustive typed dispatch -------------------------------------------


def test_dispatch_routes_each_mode() -> None:
    plain = _resolve_surface(_PlainOnly(), SerialTransport())
    fast = _resolve_surface(_FastOnly(), SerialTransport())
    both = _resolve_surface(_Both(), SerialTransport())
    tag = lambda name: lambda *_: name  # noqa: E731 — terse test stub
    assert dispatch(plain, on_default=tag("d"), on_optimized=tag("o"),
                    on_both=tag("b")) == "d"
    assert dispatch(fast, on_default=tag("d"), on_optimized=tag("o"),
                    on_both=tag("b")) == "o"
    assert dispatch(both, on_default=tag("d"), on_optimized=tag("o"),
                    on_both=tag("b")) == "b"


# --- the conformance gate primitive ------------------------------------------


def _conformance_grid() -> tuple[list, list, tuple, tuple]:
    """A 2-item grid, each item carrying two discrete and two continuous fields.

    Field layout per item tuple: (d0, c0, d1, c1) so discrete=(0, 2),
    continuous=(1, 3) interleave — a loop that only visited leading indices or
    only the first item would miss a perturbed cell. Each continuous field sits
    at a DISTINCT just-under-bar drift; discrete fields are byte-identical.
    Returned as (optimized, reference, discrete, continuous).
    """
    d00 = np.array([1.0, 2.0, 0.0])
    c00 = np.array([-3.0, 4.5])
    d01 = np.array([7.0, -8.0, 9.0])
    c01 = np.array([0.0, 11.0, -0.25])
    d10 = np.array([2.0, 2.0])
    c10 = np.array([5.5])
    d11 = np.array([-1.0, 0.0, 4.0])
    c11 = np.array([12.0, -13.5])
    # per-field just-under-bar drifts (all <= 1e-13); distinct so each field's
    # own comparison is exercised, not a shared one.
    opt = [
        (d00.copy(), c00 + 0.5e-13, d01.copy(), c01 + 0.9e-13),
        (d10.copy(), c10 + 0.3e-13, d11.copy(), c11 + 0.7e-13),
    ]
    ref = [
        (d00.copy(), c00.copy(), d01.copy(), c01.copy()),
        (d10.copy(), c10.copy(), d11.copy(), c11.copy()),
    ]
    return opt, ref, (0, 2), (1, 3)


def test_assert_conforms_passes_within_tol_and_byte_exact() -> None:
    # Distinct from the tolerance-bracket test (single item, single field of
    # each kind): this pins that assert_conforms visits EVERY item and EVERY
    # discrete/continuous field. The whole 2x(2+2) grid sits just under the
    # bar and must pass; then each cell is perturbed one at a time and the raise
    # must name exactly that field/item coordinate — proving the item loop and
    # both field loops are complete (no truncation, no break-for-continue).
    opt, ref, discrete, continuous = _conformance_grid()
    assert_conforms(
        "probe", optimized=opt, reference=ref, discrete=discrete, continuous=continuous
    )

    # Perturbation one continuous cell per (item, field): an above-tolerance drift injected
    # into item `pos`, continuous field `idx`, must raise naming that cell.
    for pos in range(2):
        for idx in continuous:
            perturbed = [tuple(arr.copy() for arr in item) for item in opt]
            fields = list(perturbed[pos])
            fields[idx] = fields[idx] + 1e-6
            perturbed[pos] = tuple(fields)
            with pytest.raises(
                AssertionError,
                match=rf"continuous field {idx} of item {pos}",
            ):
                assert_conforms(
                    "probe",
                    optimized=perturbed,
                    reference=ref,
                    discrete=discrete,
                    continuous=continuous,
                )

    # Perturbation one discrete cell per (item, field): a one-bit flip in item `pos`,
    # discrete field `idx`, must raise naming that cell.
    for pos in range(2):
        for idx in discrete:
            perturbed = [tuple(arr.copy() for arr in item) for item in opt]
            fields = list(perturbed[pos])
            flipped = fields[idx].copy()
            flipped[0] += 1.0
            fields[idx] = flipped
            perturbed[pos] = tuple(fields)
            with pytest.raises(
                AssertionError,
                match=rf"discrete field {idx} of item {pos}",
            ):
                assert_conforms(
                    "probe",
                    optimized=perturbed,
                    reference=ref,
                    discrete=discrete,
                    continuous=continuous,
                )


def test_conformance_bar_is_1e_13_and_brackets_tightly() -> None:
    # The two other tolerance tests leave a ~6-order gap (pass drives 0.5e-13,
    # fail drives 1e-6): a bar widened anywhere in (~1e-13, 1e-6) is invisible to
    # them. Bracket the bar tightly so a widening past ~1e-13 (or a tightening
    # below it) is caught, and pin the constant itself.
    assert CONTINUOUS_TOL == 1e-13
    a = np.array([1.0, 2.0, 0.0])
    # Just under the bar passes (guards against a tightening below 1e-13).
    assert_conforms(
        "probe",
        optimized=[(a + 0.5e-13, a)],
        reference=[(a, a)],
        discrete=(1,),
        continuous=(0,),
    )
    # Just over the bar fails (guards against a widening above 1e-13). The
    # probe sits ~5% over the bar (real drift ~1.05e-13), so any effective
    # widening into (1e-13, ~1.05e-13] -- e.g. tol*1.5 = 1.5e-13 -- is caught;
    # a ~2e-13 probe would leave that whole window invisible.
    over = a + 1.05e-13
    assert 1e-13 < float(np.max(np.abs(over - a))) < 1.1e-13
    with pytest.raises(AssertionError, match="continuous field 0"):
        assert_conforms(
            "probe",
            optimized=[(over, a)],
            reference=[(a, a)],
            discrete=(1,),
            continuous=(0,),
        )


def test_assert_conforms_accepts_matching_infinite_continuous_fields() -> None:
    bundle = np.array([1.0, 0.0])
    gap = np.array([np.inf])
    assert_conforms(
        "price",
        optimized=[(bundle, np.array([2.0]), gap)],
        reference=[(bundle, np.array([2.0]), gap.copy())],
        discrete=(0,),
        continuous=(1, 2),
    )


def test_assert_conforms_rejects_divergence_on_opposite_sign_infinite_field() -> None:
    # The same_inf mask only excuses SAME-sign infinities. Opposite-sign inf
    # (feasible payoff +inf vs infeasible -inf, e.g. a capacity-infeasible
    # demand) is a real divergence and must fail: +inf - -inf = +inf,
    # which is > any tol. Dropping the signbit half of the mask would swallow
    # this feasible/infeasible mismatch.
    bundle = np.array([1.0, 0.0])
    with pytest.raises(AssertionError, match="continuous field"):
        assert_conforms(
            "price",
            optimized=[(bundle, np.array([np.inf]))],
            reference=[(bundle, np.array([-np.inf]))],
            discrete=(0,),
            continuous=(1,),
        )


def test_assert_conforms_rejects_divergence_on_supra_bar_continuous() -> None:
    a = np.array([1.0, 2.0, 0.0])
    with pytest.raises(AssertionError, match="continuous field 0"):
        assert_conforms(
            "probe",
            optimized=[(a + 1e-6, a)],
            reference=[(a, a)],
            discrete=(1,),
            continuous=(0,),
        )


def test_assert_conforms_rejects_divergence_on_discrete_flip() -> None:
    a = np.array([1.0, 2.0, 0.0])
    b = np.array([1.0, 2.0, 1.0])  # a one-bit-different discrete field
    with pytest.raises(AssertionError, match="discrete field 0"):
        assert_conforms(
            "probe",
            optimized=[(b,)],
            reference=[(a,)],
            discrete=(0,),
            continuous=(),
        )


# --- the PRICE either-one on the family oracles ------------------------------


class _PriceOnlyToy(ToyOracle):
    """Overrides only price (the frozen per-agent path)."""

    price_batch = Oracle.price_batch  # keep the raising default


class _BatchOnlyToy(ToyOracle):
    """Overrides only price_batch (the optimized path)."""

    price = Oracle.price


class _NeitherToy(Oracle):
    """Overrides neither price nor price_batch — rejected at resolve."""


class _DivergentBatchToy(ToyOracle):
    """Both supplied, but price_batch perturbs one payoff past the bar."""

    def price_batch(self, theta, local_ids):
        out = dict(super().price_batch(theta, local_ids))
        first = next(iter(out))
        d = out[first]
        out[first] = Demand.exact(bundle=d.bundle, payoff=d.payoff + 1e-6)
        return out


class _ExtraIdBatchToy(ToyOracle):
    """Both supplied; price_batch leaks one id outside the requested shard.

    The requested subset itself conforms to the per-agent path — the perturbation
    is purely the extra id (a stale/foreign shard leak), which the shard-exact
    domain check must catch even though every requested id agrees.
    """

    def price_batch(self, theta, local_ids):
        out = dict(super().price_batch(theta, local_ids))
        requested = {int(i) for i in np.asarray(local_ids)}
        extra = next(i for i in range(self._r.shape[0]) if i not in requested)
        out[extra] = self.price(theta, extra)
        return out


class _MissingIdBatchToy(ToyOracle):
    """Optimized-only; price_batch drops one requested id from its shard.

    The other half of the shard-exact domain check: a batch that returns fewer
    ids than requested (a shard-local truncation) must fail. Batch-only, so
    the raise comes from ``_batched_demands``' domain check itself, not the
    both-supplied conformance gate.
    """

    price = Oracle.price  # keep the raising default -> optimized-only

    def price_batch(self, theta, local_ids):
        out = dict(super().price_batch(theta, local_ids))
        out.pop(next(iter(out)))
        return out


class _RankPartialDivergentOracle(Oracle):
    def price(self, theta, agent_id):  # type: ignore[no-untyped-def]
        bundle = np.array([int(agent_id)], dtype=np.float64)
        return Demand.exact(bundle=bundle, payoff=float(agent_id))

    def price_batch(self, theta, local_ids):  # type: ignore[no-untyped-def]
        out = {
            int(agent_id): self.price(theta, int(agent_id))
            for agent_id in np.asarray(local_ids, dtype=np.int64)
        }
        if 0 in out:
            demand = out[0]
            out[0] = Demand.exact(
                bundle=demand.bundle,
                payoff=demand.payoff + 1e-6,
            )
        return out


class _StubMaxFormulation:
    def contribute(self, demands):  # type: ignore[no-untyped-def]
        return MaxContribution(worst=0.0, local_rows=())


def _price_resolution(oracle: Oracle, transport: Transport) -> Resolution:
    return resolve(
        oracle,
        surface="price",
        default_name="price",
        optimized_name="price_batch",
        default_func=Oracle.price,
        optimized_func=Oracle.price_batch,
        transport=transport,
    )


def _ids(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.arange(arrays["observed"].shape[0], dtype=np.int64)


def test_price_only_oracle_resolves_default() -> None:
    res = _price_resolution(_PriceOnlyToy(load_toy()), SerialTransport())
    assert res.mode is Mode.DEFAULT


def test_price_batch_only_oracle_resolves_optimized() -> None:
    res = _price_resolution(_BatchOnlyToy(load_toy()), SerialTransport())
    assert res.mode is Mode.OPTIMIZED
    toy = load_toy()
    theta = toy["theta_true"]
    out = res.active(theta, _ids(toy))
    assert set(out) == set(int(i) for i in _ids(toy))


def test_neither_price_member_is_rejected() -> None:
    with pytest.raises(TypeError, match="must override"):
        _price_resolution(_NeitherToy(), SerialTransport())


def _demand_fingerprint(d) -> tuple[bytes, str, float, float]:
    bundle = np.asarray(d.bundle)
    return (bundle.tobytes(), str(bundle.dtype), d.payoff, d.gap)


def _qkp_price_from_arrays(
    arrays: dict[str, np.ndarray], theta: np.ndarray, agent_id: int
) -> tuple[np.ndarray, float]:
    """The QKP demand recomputed by brute-force enumeration over the raw arrays.

    Independent of ``QKPOracle`` (does not share its precomputed ``_quad`` /
    ``_bundles`` accessors): enumerate every subset, mask by capacity, take the
    argmax of ``b.(alpha*x - delta + nu) + lambda*0.5*b'Qb``. Used as the
    conformance oracle so the check below is a genuine differential against the
    combrum gate, not a comparison of the code-under-test against itself.
    """
    x = np.asarray(arrays["x"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    cap = np.asarray(arrays["capacities"], dtype=np.float64)
    q = np.asarray(arrays["Q"], dtype=np.float64)
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    m = weights.shape[0]
    alpha = float(theta[0])
    delta = np.asarray(theta[1:-1], dtype=np.float64)
    lam = float(theta[-1])
    best_val = -np.inf
    best_bundle = np.zeros(m)
    for count in range(2**m):
        b = np.array([(count >> j) & 1 for j in range(m)], dtype=np.float64)
        if float(b @ weights) > cap[agent_id]:
            continue
        val = float(
            b @ (alpha * x[agent_id] - delta + nu[agent_id])
            + lam * 0.5 * (b @ (q @ b))
        )
        if val > best_val:
            best_val = val
            best_bundle = b
    return best_bundle > 0.5, best_val


def _assert_price_batch_conforms(
    oracle: Oracle,
    arrays: dict[str, np.ndarray],
    reference_rule,
) -> None:
    """Drive combrum's BOTH-mode price gate and check its output against an
    oracle-independent rule.

    ``oracle`` supplies both ``price`` and ``price_batch``, so it resolves to
    :attr:`Mode.BOTH`: :func:`price_demands` runs the real combrum path
    (``_conform_demands`` -> ``_batched_demands`` -> ``conform_demands``) rather
    than the fixture doubles, and the returned demands are matched byte-for-byte
    (bundle) / within the bar (payoff, gap) against ``reference_rule`` — a rule
    recomputed straight from the raw arrays, never ``oracle.price``. If any of
    those combrum symbols stops participating, this test calls it directly; a
    wrong batch return is caught by the independent oracle. Two thetas.
    """
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.BOTH
    ids = _ids(arrays)
    thetas = [arrays["theta_true"], np.zeros_like(arrays["theta_true"])]
    for theta in thetas:
        demands = price_demands(res, theta, ids)
        assert set(demands) == {int(i) for i in ids}
        for i in ids:
            exp_bundle, exp_payoff = reference_rule(arrays, theta, int(i))
            d = demands[int(i)]
            assert np.array_equal(np.asarray(d.bundle), exp_bundle)
            assert abs(d.payoff - exp_payoff) <= CONTINUOUS_TOL
            assert d.gap == 0.0


def test_price_batch_conforms_on_toy() -> None:
    _assert_price_batch_conforms(
        ToyOracle(load_toy()), load_toy(), _toy_price_from_arrays
    )


@needs_gurobi
def test_price_batch_conforms_on_qkp() -> None:
    qkp = load_qkp()
    _assert_price_batch_conforms(QKPOracle(qkp), qkp, _qkp_price_from_arrays)


def test_both_supplied_price_conformance_passes() -> None:
    # A both-supplied oracle resolves to the batch path; price_demands runs
    # the engine's price-phase gate and, since the batch twin is bitwise the
    # per-agent path, returns the batched demands unflagged.
    toy = load_toy()
    oracle = ToyOracle(toy)  # overrides both price and price_batch
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.BOTH
    ids = _ids(toy)
    theta = toy["theta_true"]
    demands = price_demands(res, theta, ids)
    for i in ids:
        ref = oracle.price(theta, int(i))
        assert _demand_fingerprint(demands[int(i)]) == _demand_fingerprint(ref)


def test_both_supplied_divergent_price_rejects_divergence() -> None:
    # The divergent both-supplied oracle resolves to BOTH; price_demands runs
    # the conformance gate at the call site, which fails on the above-tolerance
    # payoff perturbation (coverage for the price A(i) condition).
    toy = load_toy()
    oracle = _DivergentBatchToy(toy)
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.BOTH
    ids = _ids(toy)
    theta = toy["theta_true"]
    with pytest.raises(AssertionError, match="conformance for 'price'"):
        price_demands(res, theta, ids)


def _toy_price_from_arrays(
    arrays: dict[str, np.ndarray], theta: np.ndarray, agent_id: int
) -> tuple[np.ndarray, float]:
    """The toy price rule recomputed straight from the raw fixture arrays.

    Independent of ``ToyOracle.price``: ``scores = r[i]*theta + nu[i]``, the
    chosen bundle is ``scores > 0``, the payoff sums the positive scores. Used
    so the fallback assertion below is not tautological against the very member
    it is checking (``res.reference`` *is* the bound per-agent ``price``).
    """
    r = np.asarray(arrays["observables"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    scores = r[agent_id] * theta + nu[agent_id]
    bundle = scores > 0.0
    return bundle, float(np.where(bundle, scores, 0.0).sum())


def test_price_both_divergence_falls_back_to_per_agent() -> None:
    # On a conformance fail the per-agent member is the documented
    # recovery path: pricing the shard through Resolution.reference recovers
    # the correct (non-perturbed) demands the batch gate refused. Because
    # res.reference IS the bound per-agent price, the recovered demands are
    # checked against a value recomputed independently from the raw arrays,
    # not against oracle.price (which would be tautological).
    toy = load_toy()
    oracle = _DivergentBatchToy(toy)
    res = _price_resolution(oracle, SerialTransport())
    ids = _ids(toy)
    theta = toy["theta_true"]

    with pytest.raises(AssertionError, match="conformance for 'price'"):
        price_demands(res, theta, ids)

    # The batch perturbation is real: the first id's batched payoff differs from the
    # independent rule by more than tolerance, so the gate refused an actual divergence.
    batched = oracle.price_batch(theta, ids)
    perturbed = int(min(batched))
    _, expected_payoff = _toy_price_from_arrays(toy, theta, perturbed)
    assert abs(batched[perturbed].payoff - expected_payoff) > 1e-9

    # The reference recovers the correct demand for every id, matched against
    # the independently recomputed toy rule (not against oracle.price).
    fallback = {int(i): res.reference(theta, int(i)) for i in ids}
    for i in ids:
        exp_bundle, exp_payoff = _toy_price_from_arrays(toy, theta, int(i))
        recovered = fallback[int(i)]
        assert np.array_equal(np.asarray(recovered.bundle), exp_bundle)
        assert abs(recovered.payoff - exp_payoff) <= CONTINUOUS_TOL
        assert recovered.gap == 0.0


def test_price_batch_extra_id_outside_request_rejects_divergence() -> None:
    # price_batch must key exactly local_ids. Returning the conforming
    # requested subset plus one extra id (a stale/foreign shard) breaks the
    # shard-local O(len(ids)) contract and must fail, even though every
    # requested id conforms — extra demands cannot escape to the engine.
    toy = load_toy()
    oracle = _ExtraIdBatchToy(toy)
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.BOTH
    all_ids = _ids(toy)
    subset = all_ids[: max(1, all_ids.size - 1)]  # leave room for the extra id
    theta = toy["theta_true"]
    with pytest.raises(ValueError, match="outside its requested domain"):
        price_demands(res, theta, subset)


def test_price_batch_missing_id_inside_request_rejects_divergence() -> None:
    # The other half of the shard-exact domain check: a price_batch that DROPS a
    # requested id (returns fewer ids than asked) must fail too. Optimized-
    # only, so the ValueError comes from the domain check, not the conformance
    # gate. The oracle drops the first inserted id, so want-got == {ids[0]} and
    # got-want is empty; pin the whole parenthetical against those independently
    # known sets so a swapped/garbled payload (extra and missing reported under
    # each other's labels, or the wrong id) is caught, not just the raise.
    toy = load_toy()
    oracle = _MissingIdBatchToy(toy)
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.OPTIMIZED
    ids = _ids(toy)
    theta = toy["theta_true"]
    dropped_id = int(ids[0])  # _MissingIdBatchToy pops next(iter(out)) == ids[0]
    with pytest.raises(
        ValueError,
        match=rf"extra ids \[\], missing ids \[{dropped_id}\]",
    ) as excinfo:
        price_demands(res, theta, ids)
    # Also assert the exact reported sets straight from the message payload, so
    # a regression that reports the right label with wrong contents is caught.
    text = str(excinfo.value)
    assert f"extra ids {sorted(set())}" in text
    assert f"missing ids {sorted({dropped_id})}" in text


def test_fit_step_price_conformance_failure_is_rank_agreed() -> None:
    def per_rank(transport: Transport) -> tuple[int, str]:
        oracle = _RankPartialDivergentOracle()
        resolution = _price_resolution(oracle, transport)
        try:
            fit_step(
                _StubMaxFormulation(),  # type: ignore[arg-type]
                transport=transport,
                price_resolution=resolution,
                theta=np.zeros(1, dtype=np.float64),
                scheduled_local_ids=[transport.rank],
            )
        except TransportError as exc:
            return exc.rank, exc.message
        return -1, "no-error"

    outcomes = LocalCluster(2).run(per_rank)
    assert outcomes[0][0] == outcomes[1][0] == 0
    assert all("conformance for 'price'" in message for _, message in outcomes)


def test_feature_rows_accepts_empty_ndarray_ids() -> None:
    # The empty-shard guard exists to protect the BATCHED path: without it an
    # empty shard falls into _batched_rows, whose np.stack over zero rows
    # raises. Resolve an OPTIMIZED (batch-only) map so the guarded branch is
    # the one under test — a DEFAULT resolution would return [] with or
    # without the guard (the per-agent zip never runs), so it cannot catch a
    # guard deletion. The member also asserts it is never handed an empty
    # shard, so the guard is what keeps that promise.
    called_with: list[int] = []

    class _RecordingBatchOnly(FeatureMap):
        features = FeatureMap.features  # raising default -> optimized-only

        def features_batch(self, ids, bundles):  # type: ignore[no-untyped-def]
            id_arr = np.asarray(ids, dtype=np.int64)
            called_with.append(int(id_arr.size))
            assert id_arr.size > 0, "empty shard reached features_batch"
            phi = np.asarray(bundles, dtype=np.float64).reshape(id_arr.size, -1)
            return phi, np.zeros(id_arr.size)

    res = resolve_features(_RecordingBatchOnly())
    assert res.mode is Mode.OPTIMIZED

    assert feature_rows(res, np.array([], dtype=np.int64), np.empty((0, 2))) == []
    # The batched member was never invoked for the empty shard.
    assert called_with == []
    # And a real shard does route through the batched member (the branch is
    # live, so the empty case genuinely skips it).
    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.array([[1.0, 0.0], [0.0, 1.0]])
    rows = feature_rows(res, ids, bundles)
    assert called_with == [2] and len(rows) == 2


def test_price_instance_monkeypatch_is_ignored() -> None:
    # Class-pure: shadowing price_batch on the instance must not change the
    # resolved mode (it stays the class's resolution).
    oracle = _PriceOnlyToy(load_toy())
    oracle.price_batch = lambda theta, ids: {}  # type: ignore[method-assign]
    res = _price_resolution(oracle, SerialTransport())
    assert res.mode is Mode.DEFAULT


def test_price_rank_agreement_raises_on_divergent_build() -> None:
    toy = load_toy()

    def per_rank(transport: Transport) -> str:
        oracle = (
            ToyOracle(toy) if transport.rank == 0 else _PriceOnlyToy(toy)
        )
        try:
            _price_resolution(oracle, transport)
        except TransportError as exc:
            return exc.message
        return "no-error"

    outcomes = LocalCluster(2).run(per_rank)
    assert all("disagrees across ranks" in o for o in outcomes), outcomes


# --- the FEATURES either-one: resolution + the bare-callable fallback --------


def test_bare_callable_features_resolves_default() -> None:
    res = resolve_features(toy_problem(load_toy()).features)
    assert res.mode is Mode.DEFAULT
    assert not res.runs_optimized


def test_feature_map_both_resolves_to_optimized() -> None:
    res = resolve_features(toy_feature_map(load_toy()))
    assert res.mode is Mode.BOTH


def test_feature_map_batch_only_resolves_optimized() -> None:
    res = resolve_features(toy_feature_map_batch_only(load_toy()))
    assert res.mode is Mode.OPTIMIZED


def test_feature_map_neither_overridden_is_rejected() -> None:
    class _BareFeatureMap(FeatureMap):
        pass

    with pytest.raises(TypeError, match="must override"):
        resolve_features(_BareFeatureMap())


def test_non_callable_non_featuremap_features_rejected() -> None:
    with pytest.raises(TypeError, match="must be a callable"):
        resolve_features(object())


def test_feature_batch_aggregate_requires_explicit_keywords() -> None:
    def row_batch(ids, bundles, **kwargs):
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    def aggregate_batch(ids, bundles, *, weights=None, aggregate=False):
        Phi = np.asarray(bundles, dtype=np.float64)
        if aggregate:
            return np.asarray(weights, dtype=np.float64) @ Phi, 0.0
        return Phi, np.zeros(len(ids))

    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.eye(2, dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)

    # A **kwargs member advertises no explicit weights/aggregate params, so the
    # aggregate mode is not offered.
    assert not supports_feature_batch_aggregate(row_batch)
    assert feature_batch_aggregate(row_batch, ids, bundles, weights, K=2) is None
    assert supports_feature_batch_aggregate(aggregate_batch)
    phi, eps = feature_batch_aggregate(aggregate_batch, ids, bundles, weights, K=2)
    np.testing.assert_allclose(phi, np.ones(2))
    assert eps == 0.0


def test_feature_batch_aggregate_validates_weights_shape() -> None:
    # The weights-shape guard is dead in the happy-path test above (weights are
    # always (ids.size,)). Feed a mismatched length: the member would return a
    # correctly-shaped phi regardless, so only the guard can catch it.
    def aggregate_batch(ids, bundles, *, weights=None, aggregate=False):
        if aggregate:
            return np.ones(np.asarray(bundles).shape[1], dtype=np.float64), 0.0
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        feature_batch_aggregate(
            aggregate_batch, ids, bundles, np.ones(3, dtype=np.float64), K=2
        )


def test_feature_batch_aggregate_validates_returned_phi_shape() -> None:
    # The returned-phi-shape guard is dead in the happy path too (phi is always
    # (K,)). A member returning a wrong-length aggregate must be rejected.
    def bad_phi(ids, bundles, *, weights=None, aggregate=False):
        if aggregate:
            return np.ones(3, dtype=np.float64), 0.0  # (3,) != (K=2,)
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.eye(2, dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        feature_batch_aggregate(bad_phi, ids, bundles, weights, K=2)


def test_feature_batch_aggregate_forwards_K_to_capable_member() -> None:
    # A member with a keyword-capable K param is supported, and feature_batch_
    # aggregate forwards the K value. Deleting the K-forwarding branch leaves K
    # unset, which this catches.
    seen: dict[str, object] = {}

    def aggregate_with_K(ids, bundles, *, weights=None, aggregate=False, K=None):
        seen["K"] = K
        if aggregate:
            return np.asarray(weights, dtype=np.float64) @ np.asarray(
                bundles, dtype=np.float64
            ), 0.0
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.eye(2, dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    assert supports_feature_batch_aggregate(aggregate_with_K)
    phi, eps = feature_batch_aggregate(aggregate_with_K, ids, bundles, weights, K=2)
    assert seen["K"] == 2
    np.testing.assert_allclose(phi, np.ones(2))


def test_feature_batch_aggregate_rejects_positional_only_params() -> None:
    # supports_* requires weights/aggregate (and any K) to be keyword-capable.
    # Positional-only forms must be rejected so the caller never passes them by
    # keyword into a slot the signature forbids.
    def posonly_weights(ids, bundles, weights, /, *, aggregate=False):
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    def posonly_K(ids, bundles, K, /, *, weights=None, aggregate=False):
        return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    assert not supports_feature_batch_aggregate(posonly_weights)
    assert not supports_feature_batch_aggregate(posonly_K)
    ids = np.array([0, 1], dtype=np.int64)
    bundles = np.eye(2, dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    # Rejected members fall through to the no-aggregate return.
    assert (
        feature_batch_aggregate(posonly_weights, ids, bundles, weights, K=2) is None
    )
    assert feature_batch_aggregate(posonly_K, ids, bundles, weights, K=2) is None


# --- the FEATURES either-one end-to-end on both formulations -----------------
#
# The load-bearing invariant: a FeatureMap providing features_batch yields
# identical published answers to the bare-callable per-agent path, byte-for-
# byte, on toy and QKP for both formulations.


def _swap_features(problem, features):
    # Swap the features member; oracle + geometry unchanged.
    return type(problem)(
        oracle=problem.oracle,
        features=features,
        observed_features=problem.observed_features,
        K=problem.K,
        theta_bounds=problem.theta_bounds,
    )


def _toy_walk_bare(transport, formulation_cls, backend):
    toy = load_toy()
    return run_walk(toy, toy_problem(toy), formulation_cls, transport,
                    backend=backend)


def _toy_walk_featuremap(transport, formulation_cls, backend, make_map):
    toy = load_toy()
    problem = _swap_features(toy_problem(toy), make_map(toy))
    return run_walk(toy, problem, formulation_cls, transport, backend=backend)


def _qkp_walk_bare(transport, formulation_cls, backend):
    qkp = load_qkp()
    return run_walk(qkp, qkp_problem(qkp), formulation_cls, transport,
                    backend=backend)


def _qkp_walk_featuremap(transport, formulation_cls, backend, make_map):
    qkp = load_qkp()
    problem = _swap_features(qkp_problem(qkp), make_map(qkp))
    return run_walk(qkp, problem, formulation_cls, transport, backend=backend)


def _assert_nslack_identical(a, b) -> None:
    assert a.result.theta_hat.tobytes() == b.result.theta_hat.tobytes()
    assert a.result.objective == b.result.objective
    assert a.iterations == b.iterations
    assert a.cuts_admitted == b.cuts_admitted
    assert a.result.n_active_cuts == b.result.n_active_cuts
    assert a.result.slack.tobytes() == b.result.slack.tobytes()
    assert [
        (r.agent_id, r.bundle_key, r.phi.tobytes(), r.epsilon)
        for r in a.result.active_set
    ] == [
        (r.agent_id, r.bundle_key, r.phi.tobytes(), r.epsilon)
        for r in b.result.active_set
    ]


def _assert_oneslack_identical(a, b) -> None:
    assert a.result.theta_hat.tobytes() == b.result.theta_hat.tobytes()
    assert a.result.objective == b.result.objective
    assert a.iterations == b.iterations
    assert a.result.n_active_cuts == b.result.n_active_cuts


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("make_map", [toy_feature_map, toy_feature_map_batch_only])
def test_nslack_featuremap_matches_bare_callable_toy(backend, make_map) -> None:
    bare = _toy_walk_bare(SerialTransport(), NSlack, backend)
    mapped = _toy_walk_featuremap(SerialTransport(), NSlack, backend, make_map)
    assert bare.converged and mapped.converged
    _assert_nslack_identical(bare, mapped)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("make_map", [toy_feature_map, toy_feature_map_batch_only])
def test_oneslack_featuremap_matches_bare_callable_toy(backend, make_map) -> None:
    bare = _toy_walk_bare(SerialTransport(), OneSlack, backend)
    mapped = _toy_walk_featuremap(SerialTransport(), OneSlack, backend, make_map)
    assert bare.converged and mapped.converged
    _assert_oneslack_identical(bare, mapped)


@needs_gurobi
@pytest.mark.parametrize("make_map", [qkp_feature_map, qkp_feature_map_batch_only])
def test_nslack_featuremap_matches_bare_callable_qkp(make_map) -> None:
    bare = _qkp_walk_bare(SerialTransport(), NSlack, "gurobi")
    mapped = _qkp_walk_featuremap(SerialTransport(), NSlack, "gurobi", make_map)
    assert bare.converged and mapped.converged
    _assert_nslack_identical(bare, mapped)


@needs_gurobi
@pytest.mark.parametrize("make_map", [qkp_feature_map, qkp_feature_map_batch_only])
def test_oneslack_featuremap_matches_bare_callable_qkp(make_map) -> None:
    bare = _qkp_walk_bare(SerialTransport(), OneSlack, "gurobi")
    mapped = _qkp_walk_featuremap(SerialTransport(), OneSlack, "gurobi", make_map)
    assert bare.converged and mapped.converged
    _assert_oneslack_identical(bare, mapped)


@needs_highs
def test_features_both_supplied_conformance_runs_through_walk(monkeypatch) -> None:
    # Distinct from the byte-identity parametrized cases: those pass even if the
    # BOTH gate never fired (batch == per-agent by construction). Here we spy on
    # assert_conforms to prove the gate is actually on the happy path — it fires
    # at least once per walk — and that the batched-feature answer still matches
    # the bare callable byte-for-byte.
    import combrum.interface_resolution as ir

    gate_calls = {"n": 0}
    real_assert_conforms = ir.assert_conforms

    def counting_assert_conforms(*args, **kwargs):
        gate_calls["n"] += 1
        return real_assert_conforms(*args, **kwargs)

    monkeypatch.setattr(ir, "assert_conforms", counting_assert_conforms)

    bare = _toy_walk_bare(SerialTransport(), NSlack, "highs")
    mapped = _toy_walk_featuremap(
        SerialTransport(), NSlack, "highs", toy_feature_map
    )
    assert mapped.converged
    # The BOTH-mode conformance gate ran inside the featuremap walk. The bare
    # walk resolves to DEFAULT (no gate), so every count here is the mapped
    # walk's — a gate that was silently skipped would leave this at zero.
    assert gate_calls["n"] >= 1
    _assert_nslack_identical(bare, mapped)


@needs_highs
def test_features_divergent_batch_rejects_divergence_in_walk() -> None:
    # A both-supplied FeatureMap whose batch diverges past the bar must make
    # the walk fail at the conformance gate inside contribute — wrapped
    # by the transport collective into the agreed verdict.
    toy = load_toy()
    problem = _swap_features(toy_problem(toy), divergent_feature_map(toy))
    # match= pins the failure to the features conformance gate (mirroring the
    # price sibling test_both_supplied_divergent_price_rejects_divergence): the walk
    # has other bare asserts, so a bare pytest.raises would pass even if the
    # divergence slipped past the gate and tripped an unrelated assert. The
    # substring survives the TransportError wrapper ("[rank N] ..." prefix).
    with pytest.raises((AssertionError, TransportError),
                       match="conformance for 'features'"):
        run_walk(toy, problem, NSlack, SerialTransport(), backend="highs")


@pytest.mark.parametrize("size", [2, 4])
@needs_highs
def test_nslack_featuremap_rank_invariant_matches_bare(size) -> None:
    # The FeatureMap path holds bitwise rank-invariance too: interleaved
    # shards re-route every cut, and the batched-features answer must match
    # the serial bare-callable answer on every rank.
    serial = _toy_walk_bare(SerialTransport(), NSlack, "highs")
    results = LocalCluster(size).run(
        lambda transport: _toy_walk_featuremap(
            transport, NSlack, "highs", toy_feature_map
        )
    )
    for outcome in results:
        assert outcome.converged
        _assert_nslack_identical(serial, outcome)
