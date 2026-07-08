"""Conformance for the row-generation formulations on the toy family.

Everything runs through the test-local walk driver: convergence and the
published-result contract on both real master backends, bitwise
rank-invariance between the serial transport and interleaved cluster
shards, the cut-policy admission/retirement path, and comm discipline
measured through the counting transport wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from _family_oracles import FamilyProblem, toy_problem
from _walk import WalkOutcome, run_walk
from combrum._bundle_key import pack_bundle
from combrum.context import (
    FitContext,
    ResultPublication,
    _coerce_result_publication,
)
from combrum.demand import DemandBatch
from combrum.dual import DualSolution
from combrum.formulations import NSlack, OneSlack
from combrum.formulations.oneslack import AGGREGATE_AGENT_ID
import combrum.formulations.nslack as nslack_mod
from combrum.interface_resolution import Mode, Resolution
from _support.commprobe import _ROW_HEADER_BYTES, CountingTransport
from _support.families import DEFAULT_SEED, load_family, toy_family
from combrum.master import CutReadings, MasterBackend
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.policies import CutPolicy, CutPolicyProfile
from combrum.transport import (
    CutRow,
    LocalCluster,
    SerialTransport,
    TransportError,
)

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
TOY_RAW_OBJECTIVE = 15.80392824850934


def load_toy() -> dict[str, np.ndarray]:
    return load_family("toy", FAMILY_DIR)


def _oneslack_criterion_at(
    toy: dict[str, np.ndarray], theta: np.ndarray
) -> float:
    # Independent recomputation of the OneSlack row-generation criterion at a
    # given theta: c_theta . theta + sum_a payoff_a(theta), with the walk's
    # c_theta = -sum_a phi_a(observed_a) (theta_coef == agent_weight == 1) and
    # the aggregate epigraph value equal to the priced total at the iterate.
    # Pricing the toy oracle at theta never touches the master's objective()
    # accessor, so pinning res.objective to this value ties the published theta
    # into the published objective -- a corrupted objective OR a theta whose
    # priced criterion moves fails against it.
    problem = toy_problem(toy)
    observed = np.asarray(toy["observed"])
    n_obs = observed.shape[0]
    theta = np.asarray(theta, dtype=np.float64)
    total = 0.0
    for agent in range(n_obs):
        total -= float(
            np.asarray(problem.observed_features(agent, observed[agent]), dtype=np.float64)
            @ theta
        )
        total += float(problem.oracle.price(theta, agent).payoff)
    return total


def toy_walk(
    transport: object,
    formulation_cls: type,
    backend: str,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    cut_policy: CutPolicy | None = None,
    capture_installed: bool = False,
) -> WalkOutcome:
    family = arrays if arrays is not None else load_toy()
    return run_walk(
        family,
        toy_problem(family),
        formulation_cls,
        transport,
        backend=backend,
        cut_policy=cut_policy,
        capture_installed=capture_installed,
    )


def _nslack_for_contribute(tolerance: float) -> NSlack:
    def features(agent_id: int, bundle: np.ndarray):
        return np.asarray(bundle, dtype=np.float64), float(agent_id)

    formulation = NSlack(features)
    formulation._ctx = type(
        "Ctx", (), {"K": 2, "tolerance": tolerance}
    )()
    formulation._u = {3: 0.5}
    formulation._features_res = Resolution(
        surface="features",
        mode=Mode.DEFAULT,
        active=features,
        reference=None,
        _module="test_formulations",
        _qualname="features",
    )
    formulation._trace_sink = None
    formulation._iteration = 0
    return formulation


def test_nslack_batch_contribute_avoids_demandbatch_getitem(
    monkeypatch,
) -> None:
    batch = DemandBatch.exact(
        np.array([1, 3], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.25, 2.0]),
    )
    formulation = _nslack_for_contribute(tolerance=1.0)

    def fail_getitem(self, agent_id):  # type: ignore[no-untyped-def]
        raise AssertionError("DemandBatch.__getitem__ is not on the hot path")

    monkeypatch.setattr(DemandBatch, "__getitem__", fail_getitem)

    contribution = formulation.contribute(batch)

    assert contribution.worst == 1.5
    assert len(contribution.local_rows) == 1
    row = contribution.local_rows[0]
    assert row.agent_id == 3
    np.testing.assert_array_equal(row.phi, np.array([0.0, 1.0]))

    # All-negative-rc batch: every payoff sits below its agent's _u, so no cut
    # is violated and the worst must floor at 0.0. A residual that leaks the raw
    # negative maximum (dropping the 0.0 floor) would reach the stop rule here.
    # ids [1, 3] map to u = [0.0, 0.5]; payoffs [-1.0, 0.25] give rc [-1.0, -0.25].
    negative_batch = DemandBatch.exact(
        np.array([1, 3], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([-1.0, 0.25]),
    )
    negative_contribution = formulation.contribute(negative_batch)
    assert negative_contribution.worst == 0.0
    assert negative_contribution.local_rows == ()

    # Ship filter is strict `rc > tolerance`, not `>=`: an on-boundary reduced
    # cost (rc == tolerance) must NOT ship. Build a batch straddling the
    # boundary -- strictly below, exactly at, strictly above, exactly at -- and
    # pin the WHOLE shipped set against an independent strict-`>` recomputation,
    # so both `>=` (an extra on-boundary row ships) and any looser/tighter
    # threshold that changes the survivor set fail at once.
    boundary = _nslack_for_contribute(tolerance=1.0)
    boundary._u = {3: 0.5, 5: 2.0, 7: -1.0}
    boundary_ids = np.array([1, 3, 5, 7], dtype=np.int64)
    boundary_bundles = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    )
    # u = [0.0, 0.5, 2.0, -1.0]; payoffs chosen so rc = payoff - u is
    # [0.5, 1.0, 1.5, 1.0] -- below, ON, above, ON the tolerance of 1.0.
    boundary_payoffs = np.array([0.5, 1.5, 3.5, 0.0])
    boundary_batch = DemandBatch.exact(
        boundary_ids, boundary_bundles, boundary_payoffs
    )

    boundary_u = np.array([0.0, 0.5, 2.0, -1.0])
    boundary_rc = boundary_payoffs - boundary_u
    strict_keep = boundary_rc > 1.0  # the contract; `>=` would also keep the ONs
    # Independent full expected set: features(agent_id, bundle) = (bundle, agent_id),
    # so a shipped row carries phi == bundle and epsilon == agent_id.
    expected_rows = [
        (int(agent_id), bundle, float(agent_id))
        for agent_id, bundle, keep in zip(
            boundary_ids, boundary_bundles, strict_keep
        )
        if keep
    ]
    # worst is the max positive rc regardless of tolerance (uses rc > 0.0).
    expected_worst = float(boundary_rc[boundary_rc > 0.0].max())

    boundary_contribution = boundary.contribute(boundary_batch)
    assert boundary_contribution.worst == expected_worst
    assert len(boundary_contribution.local_rows) == len(expected_rows)
    got_rows = [
        (row.agent_id, row.phi, row.epsilon)
        for row in boundary_contribution.local_rows
    ]
    assert [aid for aid, _, _ in got_rows] == [aid for aid, _, _ in expected_rows]
    for (got_id, got_phi, got_eps), (exp_id, exp_bundle, exp_eps) in zip(
        got_rows, expected_rows
    ):
        assert got_id == exp_id
        assert got_eps == exp_eps
        np.testing.assert_array_equal(got_phi, exp_bundle)
    # Explicit boundary statement: the two rc == tolerance agents (3 and 7) are
    # absent, the sole strictly-above agent (5) is present.
    assert {row.agent_id for row in boundary_contribution.local_rows} == {5}


def test_nslack_batch_contribute_chunks_violated_features(monkeypatch) -> None:
    monkeypatch.setattr(nslack_mod, "_CONTRIBUTE_FEATURE_BLOCK_ELEMENTS", 2)
    calls: list[int] = []
    real_feature_rows = nslack_mod.feature_rows

    def recording_feature_rows(resolution, ids, bundles):  # type: ignore[no-untyped-def]
        calls.append(len(ids))
        return real_feature_rows(resolution, ids, bundles)

    monkeypatch.setattr(nslack_mod, "feature_rows", recording_feature_rows)
    batch = DemandBatch.exact(
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        ),
        np.array([2.0, 2.0, 2.0, 2.0]),
    )
    formulation = _nslack_for_contribute(tolerance=0.0)

    contribution = formulation.contribute(batch)

    assert calls == [1, 1, 1, 1]
    assert [row.agent_id for row in contribution.local_rows] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "bad_payoff",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "inf", "neg_inf"],
)
def test_demand_batch_rejects_nonfinite_payoffs(bad_payoff: float) -> None:
    # Non-finite payoffs are rejected at construction, so a bad oracle cannot
    # slip a bad reduced cost past the stop rule as a silent zero floor. The
    # guard is isfinite, not merely isnan: +/-inf must be rejected too, or an
    # infinite reduced cost would flow into worst/allreduce_max unchecked.
    with pytest.raises(ValueError, match="payoffs must be finite"):
        DemandBatch(
            ids=np.array([1, 3], dtype=np.int64),
            bundles=np.array([[1.0, 0.0], [0.0, 1.0]]),
            payoffs=np.array([bad_payoff, -2.0]),
            gaps=np.zeros(2, dtype=np.float64),
        )


@pytest.mark.parametrize(
    "bad_payoff",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "neg_inf"],
)
def test_nslack_mapping_contribute_rejects_nonfinite_payoff(
    bad_payoff: float,
) -> None:
    # The per-agent mapping path (a plain dict of demands, not a DemandBatch)
    # guards non-finite payoffs itself, the sibling of the construction-time
    # DemandBatch guard above. The guard is isfinite, not isnan: an inf payoff
    # yields rc = inf - u that would otherwise reach worst, so pin +/-inf too.
    class _Demand:
        payoff = bad_payoff
        bundle = np.array([1.0, 0.0])
        gap = 0.0

    formulation = _nslack_for_contribute(tolerance=1.0)

    with pytest.raises(ValueError, match="demand payoffs must be finite"):
        formulation.contribute({3: _Demand()})


def test_nslack_received_violations_bound_dense_transient(monkeypatch) -> None:
    class _Master:
        def theta(self) -> np.ndarray:
            raise AssertionError("_received_violations must use cached theta")

    formulation = _nslack_for_contribute(tolerance=0.0)
    formulation._master = _Master()
    formulation._theta = np.array([2.0, 3.0], dtype=np.float64)
    formulation._u = {0: 0.5, 1: 1.5, 2: 2.5}
    rows = (
        CutRow(0, 0, np.array([1.0, 0.0]), 0.25, b"a"),
        CutRow(0, 1, np.array([0.0, 1.0]), 0.5, b"b"),
        CutRow(0, 2, np.array([1.0, 1.0]), 0.75, b"c"),
    )
    calls: list[int] = []
    original_vstack = np.vstack

    def counting_vstack(values):  # type: ignore[no-untyped-def]
        materialized = list(values)
        calls.append(len(materialized))
        return original_vstack(materialized)

    monkeypatch.setattr(nslack_mod, "_RECEIVED_VIOLATION_BLOCK_ELEMENTS", 4)
    monkeypatch.setattr(nslack_mod.np, "vstack", counting_vstack)

    got = formulation._received_violations(rows)

    expected = np.array([1.75, 2.0, 3.25], dtype=np.float64)
    np.testing.assert_allclose(got, expected)
    assert calls == [2, 1]


def test_nslack_deduplicates_received_rows_before_policy_admission() -> None:
    first = CutRow(0, 3, np.array([1.0, 0.0]), 0.25, b"a")
    other = CutRow(0, 4, np.array([0.0, 1.0]), 0.5, b"b")
    duplicate = CutRow(0, 3, np.array([9.0, 9.0]), 9.0, b"a")

    got = nslack_mod._deduplicate_cut_rows((first, other, duplicate))

    assert got == (first, other)


def test_nslack_dedup_canonicalizes_prior_explicit_bundle_keys() -> None:
    bundle = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    prior_key = bundle.dtype.str.encode("ascii") + b":" + bundle.tobytes()
    modern_key = pack_bundle(bundle)
    prior = CutRow(0, 3, np.array([1.0]), 0.25, prior_key)
    modern = CutRow(0, 3, np.array([9.0]), 9.0, modern_key)

    got = nslack_mod._deduplicate_cut_rows((prior, modern))

    assert len(got) == 1
    assert got[0].bundle_key == modern_key
    np.testing.assert_array_equal(got[0].bundle, bundle)


def test_nslack_rejects_malformed_modern_bundle_keys() -> None:
    bad = b"CB1-truncated"
    row = CutRow(0, 3, np.array([1.0]), 0.25, bad)

    with pytest.raises(ValueError, match="bundle_key"):
        nslack_mod._deduplicate_cut_rows((row,))


# --- convergence + the published-result contract ------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_converges_and_publishes_full_contract(backend: str) -> None:
    toy = load_toy()
    n_obs, n_items = toy["observed"].shape
    outcome = toy_walk(SerialTransport(), NSlack, backend)
    assert outcome.converged, "NSlack must reach violation <= tolerance"
    res = outcome.result

    assert res.theta_hat.shape == (n_items,)
    assert res.theta_hat.dtype == np.float64
    assert not res.theta_hat.flags.writeable
    assert isinstance(res.objective, float)

    # The installed-cut surface: count, canonical order, key uniqueness.
    assert isinstance(res.active_set, tuple)
    assert res.n_active_cuts == len(res.active_set) > 0
    keys = [(row.agent_id, row.bundle_key) for row in res.active_set]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert all(isinstance(row, CutRow) for row in res.active_set)

    # Per-agent epigraph values: full length, nonnegative (u >= 0 in the
    # hosted relaxation), zero for agents holding no cuts.
    assert res.slack is not None and res.slack.shape == (n_obs,)
    assert np.all(res.slack >= 0.0)
    cutless = set(range(n_obs)) - {row.agent_id for row in res.active_set}
    for agent in cutless:
        assert res.slack[agent] == 0.0

    # Value oracle for the slack (u) publication. Each installed row imposes
    # the epigraph bound u_a >= phi.theta + eps; at the optimum u_a saturates
    # the tightest one it holds, floored at the u >= 0 box. Recomputing that
    # per-agent max from the published rows + theta_hat is a KKT identity
    # independent of the master's u_values() accessor that fills res.slack, so
    # an all-zero (or otherwise corrupt) slack fill fails against it. On this
    # family every agent holds a binding cut, so the epigraph is strictly
    # positive everywhere -- pin that too, or a zeroed slack slips the loop.
    expected_slack = np.zeros(n_obs, dtype=np.float64)
    for row in res.active_set:
        bound = float(row.phi @ res.theta_hat + row.epsilon)
        expected_slack[row.agent_id] = max(expected_slack[row.agent_id], bound)
    np.testing.assert_allclose(res.slack, expected_slack, rtol=0, atol=1e-7)
    assert np.count_nonzero(res.slack) == n_obs

    # The dual payload: one row per installed cut, working accessors.
    dual = res.dual
    assert isinstance(dual, DualSolution)
    assert dual.rep_id == 0
    assert dual.pis.shape == (res.n_active_cuts,)
    assert dual.agent_ids.shape == (res.n_active_cuts,)
    moment = dual.moment()
    assert moment.shape == (n_items,)
    assert dual.bundle_table.shape[1] == n_items
    # Value oracle for moment(). It aggregates the dual mass over generating
    # bundles: sum_r pi_r * b_r. Recompute it from the published active-set rows
    # (each cut carries its generating bundle in bundle_key) paired with the
    # parallel pis, a path that never touches the dual's own bundle_table /
    # bundle_row_ids indexing -- so a rescale of moment() fails against it. On
    # this family the aggregate is nonzero everywhere, so a zeroed moment slips
    # nothing.
    expected_moment = np.zeros(n_items, dtype=np.float64)
    for row, pi in zip(res.active_set, dual.pis.tolist()):
        expected_moment += pi * np.asarray(row.bundle, dtype=np.float64)
    np.testing.assert_allclose(moment, expected_moment, rtol=0, atol=1e-9)
    assert np.count_nonzero(expected_moment) == n_items
    for coordinate, value in dual.bound_duals.items():
        assert 0 <= coordinate < n_items
        assert np.isfinite(value)

    # Value oracle for the dual pis. The master minimises
    #   c.theta + sum_a w_a u_a   s.t.  u_a - phi.theta >= eps   (pi >= 0)
    # with the walk's per-agent u weight w_a = agent_weights[a] = 1. Every
    # agent's u_a is strictly positive here (checked above: slack has no zeros),
    # so its u-column sits interior and its reduced cost vanishes: stationarity
    # in u_a forces the pis on that agent's installed rows to sum to exactly its
    # weight, sum_{rows of a} pi = w_a = 1. This is a KKT identity derived from
    # the fixture's known objective weights, independent of the master's
    # dual_values() accessor that fills pis -- a sign-flip or rescale of the
    # published multipliers fails it. Dual feasibility pins the sign too.
    assert np.all(dual.pis >= -1e-9)
    pi_by_agent: dict[int, float] = {}
    for agent_id, pi in zip(dual.agent_ids.tolist(), dual.pis.tolist()):
        pi_by_agent[agent_id] = pi_by_agent.get(agent_id, 0.0) + pi
    # Every observed agent holds at least one binding cut on this family.
    assert set(pi_by_agent) == set(range(n_obs))
    for agent_id, total in pi_by_agent.items():
        assert total == pytest.approx(1.0, abs=1e-7)


# theta coordinate 2's unconstrained toy optimum is +0.95; capping its upper
# box below that pins theta_2 to the bound at the solution, so the master
# reports a nonzero box-bound reduced cost there. The rest of the box stays
# slack (its optimum is interior on the untouched coordinates).
_BOUND_ACTIVE_COORD = 2
_BOUND_ACTIVE_CAP = 0.5


def _bound_active_toy_problem(toy: dict[str, np.ndarray]) -> FamilyProblem:
    base = toy_problem(toy)
    n_items = base.K
    lower = np.full(n_items, -10.0, dtype=np.float64)
    upper = np.full(n_items, 10.0, dtype=np.float64)
    upper[_BOUND_ACTIVE_COORD] = _BOUND_ACTIVE_CAP
    return FamilyProblem(
        oracle=base.oracle,
        features=base.features,
        observed_features=base.observed_features,
        K=base.K,
        theta_bounds=(lower, upper),
    )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_full_contract_publishes_bound_duals(backend: str) -> None:
    # The interior toy optimum leaves the box slack, so the full-contract test
    # above only ever sees bound_duals == {} and its bound_duals loop never
    # runs. Cap one coordinate below its unconstrained optimum so the bound
    # binds, then pin the box-bound reduced cost that must propagate through
    # NSlack.result().
    toy = load_toy()
    n_obs, n_items = toy["observed"].shape
    problem = _bound_active_toy_problem(toy)
    outcome = run_walk(
        toy, problem, NSlack, SerialTransport(), backend=backend
    )
    assert outcome.converged
    res = outcome.result

    lower, upper = problem.theta_bounds
    assert res.theta_hat[_BOUND_ACTIVE_COORD] == pytest.approx(
        _BOUND_ACTIVE_CAP, abs=1e-7
    )

    dual = res.dual
    bound_duals = dict(dual.bound_duals)
    # The capped coordinate must appear; the loop in the full-contract test is
    # now exercised on this case.
    assert _BOUND_ACTIVE_COORD in bound_duals
    assert len(bound_duals) >= 1

    # Value oracle for the box-bound reduced costs. The master minimises
    #   c_theta . theta + sum_a u_a   s.t.  u_a - phi.theta >= eps,  box on theta
    # so the reduced cost of the theta_k column is
    #   z_k = c_theta[k] + sum_r pi_r * phi_r[k]
    # (the cut row's theta_k coefficient is -phi_r[k]). Rebuilding c_theta from
    # the fixture's observed features and pairing it with the published pis/phi
    # reproduces the master's reported bound dual without ever calling
    # master.bound_duals() -- so a formulation that drops the propagated mapping
    # (publishing {}) fails the membership assert, and a corrupted value fails
    # this one.
    observed = np.asarray(toy["observed"])
    c_theta = np.zeros(n_items, dtype=np.float64)
    for agent in range(n_obs):
        c_theta -= np.asarray(
            problem.observed_features(agent, observed[agent]),
            dtype=np.float64,
        )
    reduced_cost = c_theta.copy()
    for row, pi in zip(res.active_set, dual.pis.tolist()):
        reduced_cost += pi * np.asarray(row.phi, dtype=np.float64)

    for coordinate, value in bound_duals.items():
        assert 0 <= coordinate < n_items
        # Only coordinates sitting on a box bound may carry a reduced cost.
        on_bound = res.theta_hat[coordinate] == pytest.approx(
            upper[coordinate], abs=1e-7
        ) or res.theta_hat[coordinate] == pytest.approx(
            lower[coordinate], abs=1e-7
        )
        assert on_bound
        assert value == pytest.approx(reduced_cost[coordinate], abs=1e-7)
    # The capped coordinate's reduced cost is fixed by the fixture geometry.
    assert bound_duals[_BOUND_ACTIVE_COORD] == pytest.approx(-2.0, abs=1e-7)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_oneslack_converges_with_optionals_none(backend: str) -> None:
    toy = load_toy()
    outcome = toy_walk(SerialTransport(), OneSlack, backend)
    assert outcome.converged, "OneSlack must reach violation <= tolerance"
    res = outcome.result
    assert res.theta_hat.shape == toy["theta_true"].shape
    # Value oracle for the published objective. OneSlack and NSlack solve the
    # same relaxation on this family and reach the same optimum -- a degenerate
    # face where OneSlack's theta[1]/theta[3] sit on the box, so theta itself is
    # backend-dependent but the objective is not. TOY_RAW_OBJECTIVE is the
    # fixture's known row-generation optimum (the NSlack oracle at line 775);
    # pinning res.objective to it here gives the OneSlack path a value oracle it
    # otherwise lacks, so a corrupted objective publication in result() fails.
    assert res.objective == pytest.approx(TOY_RAW_OBJECTIVE, abs=1e-9)
    # Second oracle, tying the published theta into the objective. Re-pricing
    # the toy oracle at res.theta_hat and adding c_theta . theta_hat rebuilds
    # the criterion the master reports, without touching master.objective().
    # The constant pin above alone leaves theta unchecked: on this degenerate
    # face a permuted/reversed theta keeps the objective, so it slips a
    # constant. Pinning the objective as a function of the published theta
    # catches a theta corruption wherever the priced criterion actually moves.
    assert res.objective == pytest.approx(
        _oneslack_criterion_at(toy, res.theta_hat), abs=1e-7
    )
    # OneSlack never retires an aggregate row, so every admitted cut is still
    # installed at the end: the master's own n_active_cuts must equal the
    # driver-side admitted total (summed in _walk, independent of the master
    # accessor). The converging iteration ships no cut on this monotone family,
    # so the admitted count is exactly iterations - 1. Backend-independent:
    # gurobi and highs disagree on the count but agree on both relations.
    assert res.n_active_cuts == outcome.cuts_admitted
    assert outcome.cuts_admitted == outcome.iterations - 1
    assert res.n_active_cuts >= 1
    # The optional fields exist for exactly this method: no per-agent
    # slack, cut set, or dual exists to publish.
    assert res.slack is None
    assert res.active_set is None
    assert res.dual is None
    assert AGGREGATE_AGENT_ID == 0


class _OneSlackUMaster(MasterBackend):
    """Master double that exposes u only through the solver-state accessor."""

    def __init__(self) -> None:
        self.u_reads = 0

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        return len(rows)

    def solve(self) -> None:
        return None

    def theta(self) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=np.float64)

    def objective(self) -> float:
        return 12.5

    @property
    def n_active_cuts(self) -> int:
        return 1

    def u_values(self) -> dict[int, float]:
        self.u_reads += 1
        return {AGGREGATE_AGENT_ID: 3.25}

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        return {}

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        return None

    def extract_cuts(self) -> tuple[CutRow, ...]:
        raise AssertionError("OneSlack must read u from master.u_values()")

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        return None

    def bound_duals(self) -> dict[int, float]:
        return {}


def test_oneslack_state_reads_master_epigraph_variable() -> None:
    master = _OneSlackUMaster()
    formulation = OneSlack(_publication_features)
    formulation._master = master

    state = formulation._state(progressed=7)

    assert state.u == 3.25
    assert state.objective == 12.5
    assert state.n_installed == 1
    assert state.progressed == 7
    assert master.u_reads == 1


class _PenaltyOrderMaster(MasterBackend):
    def __init__(self, K: int) -> None:
        self.K = int(K)
        self.log: list[str] = []
        self._installed: dict[tuple[int, bytes], CutRow] = {}

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        fresh = 0
        for row in rows:
            key = (row.agent_id, row.bundle_key)
            if key not in self._installed:
                self._installed[key] = row
                fresh += 1
        self.log.append(f"add_cuts:{fresh}")
        return fresh

    def solve(self) -> None:
        self.log.append("solve")

    def theta(self) -> np.ndarray:
        return np.zeros(self.K, dtype=np.float64)

    def objective(self) -> float:
        return 0.0

    def u_values(self) -> dict[int, float]:
        return {int(agent_id): 0.0 for agent_id, _key in self._installed}

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        return {}

    def cut_readings(self, *, dual: bool = False, slack: bool = False) -> CutReadings:
        self.log.append(f"cut_readings:{int(dual)}:{int(slack)}")
        return CutReadings(
            keys=tuple(sorted(self._installed)),
            dual=np.zeros(len(self._installed), dtype=np.float64) if dual else None,
            slack=np.zeros(len(self._installed), dtype=np.float64) if slack else None,
        )

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        self.log.append(f"set_penalty:{float(weight):.1f}")

    def extract_cuts(self) -> tuple[CutRow, ...]:
        return tuple(self._installed[key] for key in sorted(self._installed))

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        self._installed = {(row.agent_id, row.bundle_key): row for row in rows}

    def bound_duals(self) -> dict[int, float]:
        return {}


def _penalty_order_ctx(master: MasterBackend, K: int = 1) -> FitContext:
    return FitContext(
        K=K,
        N=2,
        S=1,
        theta_bounds=(
            np.full(K, -1.0, dtype=np.float64),
            np.full(K, 1.0, dtype=np.float64),
        ),
        theta_coef=np.ones(2, dtype=np.float64),
        agent_weights=np.ones(2, dtype=np.float64),
        local_ids=np.arange(2, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-8,
        master_backend=master,
    )


def test_nslack_penalty_shares_the_post_install_solve() -> None:
    master = _PenaltyOrderMaster(K=1)
    formulation = NSlack(_publication_features)
    formulation.setup(_penalty_order_ctx(master))
    master.log.clear()

    row = CutRow(
        rep_id=0,
        agent_id=0,
        phi=np.array([1.0], dtype=np.float64),
        epsilon=1.0,
        bundle_key=pack_bundle(np.array([True])),
    )
    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 1.0)
    progressed = formulation.apply_step((row,))

    assert progressed == 1
    assert master.log == ["add_cuts:1", "set_penalty:1.0", "solve"]


def test_nslack_penalty_revert_solves_without_new_cuts() -> None:
    master = _PenaltyOrderMaster(K=1)
    formulation = NSlack(_publication_features)
    formulation.setup(_penalty_order_ctx(master))
    master.log.clear()

    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 1.0)
    formulation.apply_step(())
    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 0.0)
    formulation.apply_step(())

    assert master.log == [
        "add_cuts:0",
        "set_penalty:1.0",
        "solve",
        "add_cuts:0",
        "set_penalty:0.0",
        "solve",
    ]


class _RecordingDualPurge(CutPolicy):
    profile = CutPolicyProfile(
        needs_admit_violations=False,
        retires_cuts=True,
        needs_purge_duals=True,
        needs_purge_slacks=False,
    )

    def __init__(self) -> None:
        self.dual_args: list[Mapping[tuple[int, bytes], float] | None] = []

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return tuple(candidates)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        self.dual_args.append(dual)
        return ()


def test_nslack_dual_purge_skips_qp_duals() -> None:
    master = _PenaltyOrderMaster(K=1)
    policy = _RecordingDualPurge()
    ctx = _penalty_order_ctx(master)
    object.__setattr__(ctx, "cut_policy", policy)
    formulation = NSlack(_publication_features)
    formulation.setup(ctx)

    row = CutRow(
        rep_id=0,
        agent_id=0,
        phi=np.array([1.0], dtype=np.float64),
        epsilon=1.0,
        bundle_key=pack_bundle(np.array([True])),
    )
    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 1.0)
    formulation.apply_step((row,))
    master.log.clear()

    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 0.0)
    formulation.apply_step(())

    assert policy.dual_args[-1] is None
    assert "cut_readings:1:0" not in master.log


def test_oneslack_penalty_shares_the_post_install_solve() -> None:
    master = _PenaltyOrderMaster(K=1)
    formulation = OneSlack(_publication_features)
    formulation.setup(_penalty_order_ctx(master))
    master.log.clear()

    formulation.prepare_penalty_solve(np.array([0.25], dtype=np.float64), 1.0)
    progressed = formulation.apply_step((np.array([0.0], dtype=np.float64), 1.0))

    assert progressed == 1
    assert master.log == ["add_cuts:1", "set_penalty:1.0", "solve"]


class _ResultPublicationMaster(MasterBackend):
    """Minimal master whose heavy accessors fail the summary-result test."""

    def __init__(self) -> None:
        self._theta = np.array([0.25, -0.5], dtype=np.float64)
        self._objective = 1.25
        self._n_active = 187_826
        self._u = {0: 0.0, 2: 1.5}
        self.calls = {
            "extract_cuts": 0,
            "dual_values": 0,
            "bound_duals": 0,
            "u_values": 0,
        }

    def reset_calls(self) -> None:
        for key in self.calls:
            self.calls[key] = 0

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        return len(rows)

    def solve(self) -> None:
        return None

    def theta(self) -> np.ndarray:
        return self._theta.copy()

    def objective(self) -> float:
        return self._objective

    @property
    def n_active_cuts(self) -> int:
        return self._n_active

    def u_values(self) -> dict[int, float]:
        self.calls["u_values"] += 1
        return dict(self._u)

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        self.calls["dual_values"] += 1
        return {}

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        return None

    def extract_cuts(self) -> tuple[CutRow, ...]:
        self.calls["extract_cuts"] += 1
        return ()

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        return None

    def bound_duals(self) -> dict[int, float]:
        self.calls["bound_duals"] += 1
        return {}


def _publication_features(agent_id: int, bundle: np.ndarray) -> tuple[np.ndarray, float]:
    return np.zeros(2, dtype=np.float64), 0.0


def _publication_ctx(master: MasterBackend, publication) -> FitContext:
    return FitContext(
        K=2,
        N=4,
        S=1,
        theta_bounds=(
            np.full(2, -1.0, dtype=np.float64),
            np.full(2, 1.0, dtype=np.float64),
        ),
        theta_coef=np.ones(4, dtype=np.float64),
        agent_weights=np.ones(4, dtype=np.float64),
        local_ids=np.arange(4, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-8,
        master_backend=master,
        result_publication=publication,
    )


def test_nslack_summary_result_publishes_no_large_artifacts() -> None:
    master = _ResultPublicationMaster()
    ctx = _publication_ctx(master, ResultPublication.SUMMARY)
    formulation = NSlack(_publication_features)
    formulation.setup(ctx)

    master.reset_calls()
    result = formulation.result()

    assert result.theta_hat.tobytes() == master.theta().tobytes()
    assert result.objective == master.objective()
    assert result.n_active_cuts == master.n_active_cuts
    assert result.slack is None
    assert result.active_set is None
    assert result.dual is None
    assert master.calls == {
        "extract_cuts": 0,
        "dual_values": 0,
        "bound_duals": 0,
        "u_values": 0,
    }


def test_publication_full_is_broadcast_mode() -> None:
    ordinary = _coerce_result_publication(("slack", "active_set", "dual"))

    assert _coerce_result_publication("full") == ResultPublication.FULL
    assert ResultPublication.FULL & ResultPublication.BROADCAST
    assert ordinary == (
        ResultPublication.SLACK
        | ResultPublication.ACTIVE_SET
        | ResultPublication.DUAL
    )
    assert not ordinary & ResultPublication.BROADCAST


def test_nslack_dual_only_result_publishes_no_active_set() -> None:
    master = _ResultPublicationMaster()
    ctx = _publication_ctx(master, ResultPublication.DUAL)
    formulation = NSlack(_publication_features)
    formulation.setup(ctx)

    master.reset_calls()
    result = formulation.result()

    assert result.active_set is None
    assert isinstance(result.dual, DualSolution)
    assert result.dual.pis.shape == (0,)
    assert result.dual.bundle_table.shape == (0, 2)
    np.testing.assert_array_equal(result.dual.moment(), np.zeros(2))
    assert master.calls == {
        "extract_cuts": 1,
        "dual_values": 1,
        "bound_duals": 1,
        "u_values": 0,
    }


def test_nslack_active_set_and_dual_share_one_extract_pass() -> None:
    master = _ResultPublicationMaster()
    ctx = _publication_ctx(
        master, ResultPublication.ACTIVE_SET | ResultPublication.DUAL
    )
    formulation = NSlack(_publication_features)
    formulation.setup(ctx)

    master.reset_calls()
    result = formulation.result()

    assert result.active_set == ()
    assert isinstance(result.dual, DualSolution)
    assert master.calls["extract_cuts"] == 1
    assert master.calls["dual_values"] == 1
    assert master.calls["bound_duals"] == 1


# --- root-only master is mandatory --------------------------------------------


@pytest.mark.parametrize("formulation_cls", [NSlack, OneSlack])
def test_setup_without_root_master_raises_transport_error(
    formulation_cls: type,
) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    n_obs = toy["observed"].shape[0]

    ctx = FitContext(
        K=problem.K,
        N=n_obs,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=np.ones(n_obs),
        agent_weights=np.ones(n_obs),
        local_ids=np.arange(n_obs, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-8,
        master_backend=None,
    )
    formulation = formulation_cls(problem.features)
    # The check sits inside the collective guard, so the failure arrives
    # as the agreed transport verdict rather than stranding peer ranks.
    with pytest.raises(TransportError, match="master_backend"):
        formulation.setup(ctx)


# --- bitwise rank invariance ---------------------------------------------------


def _assert_identical_nslack(outcome: WalkOutcome, anchor: WalkOutcome) -> None:
    res, ref = outcome.result, anchor.result
    assert res.theta_hat.tobytes() == ref.theta_hat.tobytes()
    assert res.objective == ref.objective
    assert outcome.objective == anchor.objective
    assert outcome.iterations == anchor.iterations
    assert outcome.cuts_admitted == anchor.cuts_admitted
    assert res.n_active_cuts == ref.n_active_cuts
    assert res.slack.tobytes() == ref.slack.tobytes()
    assert [
        (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
        for row in res.active_set
    ] == [
        (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
        for row in ref.active_set
    ]
    for field in ("agent_ids", "bundle_row_ids", "pis", "bundle_table"):
        assert (
            getattr(res.dual, field).tobytes()
            == getattr(ref.dual, field).tobytes()
        ), field
    assert dict(res.dual.bound_duals) == dict(ref.dual.bound_duals)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_rank_invariance_bitwise(backend: str, size: int) -> None:
    # Interleaved shards (a % size == rank) re-route every cut and every
    # reduction contribution; the result must match the serial answer bitwise.
    toy = load_toy()
    serial = toy_walk(SerialTransport(), NSlack, backend)
    results = LocalCluster(size).run(
        lambda transport: toy_walk(transport, NSlack, backend, arrays=toy)
    )
    assert len(results) == size
    for outcome in results:
        _assert_identical_nslack(outcome, serial)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_oneslack_rank_invariance_bitwise(backend: str, size: int) -> None:
    toy = load_toy()
    serial = toy_walk(SerialTransport(), OneSlack, backend)
    results = LocalCluster(size).run(
        lambda transport: toy_walk(transport, OneSlack, backend, arrays=toy)
    )
    assert len(results) == size
    for outcome in results:
        res = outcome.result
        assert res.theta_hat.tobytes() == serial.result.theta_hat.tobytes()
        assert res.objective == serial.result.objective
        assert outcome.objective == serial.objective
        assert outcome.iterations == serial.iterations
        assert res.n_active_cuts == serial.result.n_active_cuts


# --- cut policy: admission + retirement through the live walk -----------------


class _RetireOnePolicy(CutPolicy):
    """Recording double: admits everything, retires one row at iteration 2.

    The single retirement forces the extract -> filter -> reinstall path to
    run; recording per-cut signals checks the caller keys dual/slack
    mappings by installed cut.
    """

    def __init__(self) -> None:
        self.admit_iterations: list[int] = []
        self.signals: list[tuple[set, set, set, list[float]]] = []
        self.retired: list[CutRow] = []
        self.admit_violations: list[np.ndarray] = []
        self.admit_counts: list[int] = []
        # Optional master probe (a lazy theta/u accessor) attached by the test
        # after setup, so purge can snapshot the solver state each call for the
        # per-row slack-value oracle. None -> no snapshot.
        self.master_probe: object | None = None
        self.slack_probe: list[
            tuple[tuple[CutRow, ...], dict[tuple[int, bytes], float], np.ndarray, dict[int, float]]
        ] = []
        # Per admit call: the priced candidates paired with the recorded
        # violation array and a snapshot of the pre-solve master state (admit
        # runs before this step's solve), so the test can recompute each row's
        # violation phi.theta + eps - u independently of the recorded values.
        self.admit_probe: list[
            tuple[tuple[CutRow, ...], np.ndarray, np.ndarray, dict[int, float]]
        ] = []

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        self.admit_iterations.append(iteration)
        # One violation per candidate, parallel to the rows; recorded so the
        # test can check the admit contract.
        self.admit_violations.append(np.asarray(violations))
        self.admit_counts.append(len(candidates))
        if self.master_probe is not None:
            # Snapshot the pre-solve theta/u behind this admit so the test can
            # recompute each candidate's violation independently of the array.
            self.admit_probe.append(
                (
                    tuple(candidates),
                    np.asarray(violations, dtype=np.float64),
                    np.asarray(self.master_probe.theta(), dtype=np.float64),
                    dict(self.master_probe.u_values()),
                )
            )
        return tuple(candidates)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        installed_keys = {(row.agent_id, row.bundle_key) for row in installed}
        self.signals.append(
            (
                installed_keys,
                set(dual or {}),
                set(slack or {}),
                list((slack or {}).values()),
            )
        )
        if self.master_probe is not None:
            # Snapshot theta and the epigraph values behind this solve so the
            # test can recompute each row's slack independently of the map.
            self.slack_probe.append(
                (
                    tuple(installed),
                    dict(slack or {}),
                    np.asarray(self.master_probe.theta(), dtype=np.float64),
                    dict(self.master_probe.u_values()),
                )
            )
        if iteration == 2 and not self.retired and installed:
            self.retired.append(installed[-1])
            return (installed[-1],)
        return ()


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_cut_policy_admission_and_retirement(
    backend: str, monkeypatch
) -> None:
    policy = _RetireOnePolicy()

    # Capture the owner-rank master so the policy can snapshot theta/u each
    # purge (the master lives inside the driver and is not otherwise exposed).
    # A lazy accessor defers the read to purge time, when the state is set.
    captured: dict[str, MasterBackend] = {}
    original_setup = nslack_mod.NSlack.setup

    def capturing_setup(self, ctx):  # type: ignore[no-untyped-def]
        original_setup(self, ctx)
        if getattr(self, "_is_owner", False):
            captured["master"] = self._master

    monkeypatch.setattr(nslack_mod.NSlack, "setup", capturing_setup)

    class _LazyMaster:
        def theta(self) -> np.ndarray:
            return captured["master"].theta()

        def u_values(self) -> dict[int, float]:
            return captured["master"].u_values()

    policy.master_probe = _LazyMaster()

    outcome = toy_walk(SerialTransport(), NSlack, backend, cut_policy=policy)
    assert outcome.converged
    assert len(policy.retired) == 1
    # Every install is counted once, so survivors == admitted minus the one
    # retirement (a re-entered bundle counts on both sides).
    assert outcome.result.n_active_cuts == outcome.cuts_admitted - 1
    assert policy.admit_iterations == list(range(outcome.iterations))
    # The admit contract: each call's violations array is parallel to the
    # candidates it priced and nonnegative (a candidate is a row whose
    # violation cleared tolerance), so the magnitude signal is real.
    for (_, _, _, _), violations, admitted in zip(
        policy.signals, policy.admit_violations, policy.admit_counts
    ):
        assert violations.shape == (admitted,)
        assert np.all(violations >= -1e-9)
    # Value oracle for the admit-side violation array. The driver hands the
    # policy, for each received row, phi.theta + eps - u_a at the pre-solve
    # master solution (the violation the row still carries before this step
    # resolves). Recomputing that per row from the admit-time theta()/u_values()
    # snapshot -- accessors distinct from the _received_violations path that
    # fills the array off the formulation's cached self._u -- reproduces the
    # whole array without touching it, so a rescale or +eps-drop of the admit
    # violations (which the shape/sign check above lets through) fails here.
    assert policy.admit_probe, "master probe must have recorded every admit"
    assert len(policy.admit_probe) == len(policy.admit_violations)
    for candidates, violations, theta, u_map in policy.admit_probe:
        assert violations.shape == (len(candidates),)
        expected = np.array(
            [
                float(np.asarray(row.phi, dtype=np.float64) @ theta)
                + row.epsilon
                - u_map.get(row.agent_id, 0.0)
                for row in candidates
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(violations, expected, rtol=0, atol=1e-7)
    for installed_keys, dual_keys, slack_keys, slack_values in policy.signals:
        # The master keys cut_readings over its sorted_row_keys -- exactly the
        # last-solved installed rows -- so the dual/slack maps handed to the
        # policy must cover the installed set completely, not merely a subset.
        # Equality catches a map that drops (or carries an extra) installed
        # key; a subset check would let a short map through undetected.
        assert dual_keys == installed_keys
        assert slack_keys == installed_keys
        # Installed rows satisfied their epigraph at the last solve, so
        # supplied slacks are nonnegative up to solver tolerance.
        assert all(value >= -1e-9 for value in slack_values)
    # Value oracle for the slack map the policy reads. The master keys the row
    # slack of installed row (a, bundle) at the last solve as u_a - phi.theta -
    # eps: the amount by which its epigraph bound is loose. Recomputing that per
    # row from theta()/u_values() (accessors distinct from cut_readings(), which
    # sources slack from the solver's row activity) reproduces the whole map
    # without touching it, so a rescaled, permuted, or otherwise value-corrupted
    # slack map -- not just a key-set change -- fails here. The nonnegativity
    # check above passes any sign-preserving corruption; this pins the value.
    assert policy.slack_probe, "master probe must have recorded every purge"
    for installed, slack_map, theta, u_map in policy.slack_probe:
        assert {(row.agent_id, row.bundle_key) for row in installed} == set(
            slack_map
        )
        for row in installed:
            key = (row.agent_id, row.bundle_key)
            expected = (
                u_map.get(row.agent_id, 0.0)
                - float(np.asarray(row.phi, dtype=np.float64) @ theta)
                - row.epsilon
            )
            assert slack_map[key] == pytest.approx(expected, abs=1e-7)
    # The published value is the raw row-generation master objective
    # sum_a u_a - theta dot observed_features.
    assert outcome.objective == pytest.approx(TOY_RAW_OBJECTIVE, abs=1e-9)


# --- comm discipline ------------------------------------------------------------


@needs_highs
def test_nslack_comm_rounds_constant_and_exchange_bytes_scale() -> None:
    # The exact round formula holding at both family sizes is the whole
    # discipline claim: one round of each kind per iteration plus the
    # fixed setup/result rounds, independent of N.
    for arrays in (load_toy(), toy_family(24, 5, DEFAULT_SEED)):
        n_obs, n_items = arrays["observed"].shape
        probe = CountingTransport(SerialTransport())
        outcome = toy_walk(
            probe, NSlack, "highs", arrays=arrays, capture_installed=True
        )
        assert outcome.converged
        T = outcome.iterations
        assert T >= 2
        assert probe.counts() == {
            "allreduce_max": T,
            "exchange_cuts": T,
            "collective_guard": T + 2,
            "bcast": T + 2,
            "scatter_by_agent": T + 1,
        }
        assert probe.bytes_moved()["scatter_by_agent"] == (T + 1) * n_obs * 8
        # Exchange payload is exactly the shipped rows: with no policy
        # every shipped row is new to the master (an installed bundle
        # can never price violated again), so the admitted total IS the
        # shipped total.
        installed = outcome.installed_snapshots[-1]
        assert installed
        key_nbytes = len(installed[0].bundle_key)
        row_bytes = (
            n_items * 8
            + key_nbytes
            + _ROW_HEADER_BYTES
        )
        assert (
            probe.bytes_moved()["exchange_cuts"]
            == outcome.cuts_admitted * row_bytes
        )
        assert 0 < outcome.cuts_admitted < n_obs * T


@needs_highs
def test_oneslack_comm_rounds_constant_and_reduction_bytes_o_shard() -> None:
    arrays = load_toy()
    n_obs, n_items = arrays["observed"].shape
    probe = CountingTransport(SerialTransport())
    outcome = toy_walk(probe, OneSlack, "highs", arrays=arrays)
    assert outcome.converged
    T = outcome.iterations
    assert T >= 2
    assert probe.counts() == {
        "sum_reproducible": T,
        "collective_guard": T + 1,
        "bcast": T + 1,
    }
    # The reduction ships one (K+1)-row plus one id per local agent per
    # iteration: payload O(shard), never O(cuts) or O(N * iterations).
    contributed = n_obs * (n_items + 1) * 8 + n_obs * 8
    assert probe.bytes_moved()["sum_reproducible"] == T * contributed
