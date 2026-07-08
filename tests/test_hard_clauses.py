"""The two hard clauses: per-agent vs batched features must produce the
same discrete row identities end-to-end.

Each clause runs two walks differing only in the features provider — the
per-agent ``problem.features`` callable vs a ``*_batch_only`` FeatureMap —
over the same family, backend, transport, tolerance, schedule and penalty,
so any difference in the captured ``installed_snapshots`` is down to the
features path alone.

* **Aggregate identity.** The OneSlack aggregate row is keyed by a SHA-256
  over the float64 bytes of ``[phi_agg, eps_agg]``
  (``oneslack.py:_aggregate_key``), so even a ``<=1e-13`` drift in the
  summed aggregate changes the row key — a discrete identity flip, not a
  continuous nudge. The batched path must produce a byte-identical
  aggregate; the clause compares the full installed row tuple bytewise.

* **Support mask.** ``HighsMaster._install_batch`` sparsifies every row
  before handing the column set to ``addRows``,
  so an exact ``0.0`` turning into a tiny nonzero (or back) changes
  installed structure, not just a coefficient — discrete zero-mask
  equality, never ``<=1e-13``. The clause compares the per-row nonzero
  pattern across the two feature paths for both formulations on HiGHS.
  Only NSlack rows carry structural zeros (a single agent's ``b * r_a``);
  the OneSlack aggregate ``phi_agg = sum_a w_a * phi_a`` is dense on these
  families, so its mask check only confirms the aggregate stays full. The
  ``require_zeros`` flag asserts whichever assumption applies. The
  cross-path comparison cannot see a change that hits both paths
  identically, so ``test_support_mask_install_sparsifies_row`` also checks
  the column set ``_install_batch`` actually submits to ``addRows``
  against a hand-picked support.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from _family_oracles import (
    FamilyProblem,
    qkp_feature_map_batch_only,
    qkp_problem,
    toy_feature_map_batch_only,
    toy_problem,
)
from _walk import WalkOutcome, run_walk
from combrum.context import FitContext
from combrum.formulations import NSlack, OneSlack
import combrum.formulations.oneslack as oneslack_mod
from combrum.interface_resolution import FeatureMap
from _support.constants import TOLERANCE
from _support.families import load_qkp, load_toy
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

# Apple Accelerate raises spurious FP-status warnings on provably finite
# matmuls at these sizes; the feature maps are guarded regardless.
pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning:.*matmul.*"),
]


# --- the walk pair: per-agent features vs batched features -------------------
#
# _with_features rebuilds a FamilyProblem swapping only the feature map;
# _walk_pair runs the per-agent and batch-only walks over identical
# arrays/formulation/backend so any row difference is the features path alone.


def _with_features(problem: FamilyProblem, features: object) -> FamilyProblem:
    return type(problem)(
        oracle=problem.oracle,
        features=features,
        observed_features=problem.observed_features,
        K=problem.K,
        theta_bounds=problem.theta_bounds,
    )


def _walk_pair(
    arrays: dict[str, np.ndarray],
    base_problem: FamilyProblem,
    batch_only_map: object,
    formulation_cls: type,
    backend: str,
) -> tuple[WalkOutcome, WalkOutcome]:
    """(per-agent walk, batched walk) — identical but for the features path."""
    per_agent = run_walk(
        arrays,
        base_problem,
        formulation_cls,
        SerialTransport(),
        backend=backend,
        capture_installed=True,
    )
    batched = run_walk(
        arrays,
        _with_features(base_problem, batch_only_map),
        formulation_cls,
        SerialTransport(),
        backend=backend,
        capture_installed=True,
    )
    return per_agent, batched


def _assert_same_run(a: WalkOutcome, b: WalkOutcome) -> None:
    """Both walks must take the same shape of run, else the snapshot
    comparison below would compare different runs."""
    assert a.converged and b.converged
    assert a.iterations == b.iterations
    assert a.result.theta_hat.tobytes() == b.result.theta_hat.tobytes()
    assert a.result.objective == b.result.objective
    assert len(a.installed_snapshots) == len(b.installed_snapshots)


def _aligned_rows(
    a: WalkOutcome, b: WalkOutcome
) -> list[tuple[CutRow, CutRow]]:
    """Installed rows paired across the two paths, snapshot by snapshot.

    extract_cuts() returns rows in canonical-key order, so when the runs
    match, the i-th row of snapshot t on one path corresponds to the i-th
    row on the other.
    """
    pairs: list[tuple[CutRow, CutRow]] = []
    for snap_a, snap_b in zip(a.installed_snapshots, b.installed_snapshots):
        assert len(snap_a) == len(snap_b)
        pairs.extend(zip(snap_a, snap_b))
    return pairs


# --- OneSlack aggregate identity --------------------------------------------


def _row_bytes(row: CutRow) -> tuple[int, bytes, bytes, str, float]:
    # (agent_id, bundle_key, phi bytes, phi dtype, epsilon) — the discrete
    # identity (agent_id + key + dtype) plus the exact float64 payload.
    return (
        row.agent_id,
        row.bundle_key,
        row.phi.tobytes(),
        str(row.phi.dtype),
        float(row.epsilon),
    )


def _priced_aggregate(
    base_problem: FamilyProblem, theta: np.ndarray, n_agents: int
) -> tuple[np.ndarray, float]:
    # Aggregate at one pricing theta, rebuilt from the family oracle: price
    # every agent, featurise, and reduce (phi_a|eps_a) over ascending agent
    # id — the same ascending-id np.add.reduce as canonical_sum, so the
    # result is bit-comparable. agent_weights default to 1.0 in the walk
    # (_walk.py), matched here.
    K = base_problem.K
    rows = np.empty((n_agents, K + 1), dtype=np.float64)
    for a in range(n_agents):
        demand = base_problem.oracle.price(theta, a)
        phi_a, eps_a = base_problem.features(a, demand.bundle)
        rows[a, :K] = np.asarray(phi_a, dtype=np.float64)
        rows[a, K] = float(eps_a)
    agg = np.add.reduce(rows, axis=0)
    return agg[:K].copy(), float(agg[K])


def _assert_aggregate_value_anchor(
    arrays, base_problem: FamilyProblem, backend: str
) -> None:
    # The byte-equality and key checks both reference the row's own phi/eps;
    # a scaled phi_agg or eps_agg that hits both feature paths alike would
    # pass them. So capture the pricing theta each iteration (OneSlack.solve
    # returns it), rebuild each aggregate from the priced bundles, and
    # require every installed row to match bitwise (the reduce is
    # bit-reproducible on one rank).
    n_agents = int(np.asarray(arrays["observed"]).shape[0])
    captured: list[np.ndarray] = []
    orig_solve = OneSlack.solve

    def spy_solve(self):
        theta = orig_solve(self)
        captured.append(np.asarray(theta, dtype=np.float64).copy())
        return theta

    OneSlack.solve = spy_solve
    try:
        per_agent = run_walk(
            arrays,
            base_problem,
            OneSlack,
            SerialTransport(),
            backend=backend,
            capture_installed=True,
        )
    finally:
        OneSlack.solve = orig_solve

    # One aggregate per captured pricing theta, keyed by its SHA-256 digest.
    # First occurrence wins, matching add_cuts' de-dup.
    expected: dict[bytes, tuple[np.ndarray, float]] = {}
    for theta in captured:
        phi_agg, eps_agg = _priced_aggregate(base_problem, theta, n_agents)
        key = _expected_aggregate_key(phi_agg, eps_agg)
        expected.setdefault(key, (phi_agg, eps_agg))

    installed = per_agent.installed_snapshots[-1] if per_agent.installed_snapshots else ()
    assert installed, "no aggregate row was installed"
    for row in installed:
        assert row.bundle_key in expected, (
            "installed aggregate key matches no priced aggregate"
        )
        exp_phi, exp_eps = expected[row.bundle_key]
        np.testing.assert_array_equal(np.asarray(row.phi), exp_phi)
        assert float(row.epsilon) == exp_eps


def _assert_aggregate_key(arrays, base_problem, batch_only_map, backend) -> None:
    per_agent, batched = _walk_pair(
        arrays, base_problem, batch_only_map, OneSlack, backend
    )
    _assert_same_run(per_agent, batched)
    pairs = _aligned_rows(per_agent, batched)
    assert pairs, "no aggregate row was ever installed"
    for row_pa, row_b in pairs:
        assert _row_bytes(row_pa) == _row_bytes(row_b)
    # Byte equality alone misses a finalise value bug that hits both feature
    # paths alike; anchor the values too.
    _assert_aggregate_value_anchor(arrays, base_problem, backend)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_aggregate_key_oneslack_aggregate_bytes_toy(backend) -> None:
    toy = load_toy()
    _assert_aggregate_key(toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend)


@needs_gurobi
def test_aggregate_key_oneslack_aggregate_bytes_qkp() -> None:
    # QKP enumeration is exact only under the gurobi-solved master the
    # references were captured against, so this gate is gurobi-only.
    qkp = load_qkp()
    _assert_aggregate_key(
        qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi"
    )


def _expected_aggregate_key(phi_agg: np.ndarray, eps_agg: float) -> bytes:
    # SHA-256 over the float64 bytes of [phi_agg, eps_agg], kept independent
    # of oneslack._aggregate_key.
    payload = np.empty(phi_agg.size + 1, dtype=np.float64)
    payload[: phi_agg.size] = np.asarray(phi_agg, dtype=np.float64)
    payload[phi_agg.size] = float(eps_agg)
    return hashlib.sha256(payload.tobytes()).digest()


@needs_highs
def test_aggregate_key_bundle_key_is_independent_sha256() -> None:
    # A key-derivation change that hits both feature paths alike (e.g.
    # dropping eps_agg from the hashed payload) leaves them byte-identical,
    # so check bundle_key against an independently computed digest.
    toy = load_toy()
    per_agent, _ = _walk_pair(
        toy,
        toy_problem(toy),
        toy_feature_map_batch_only(toy),
        OneSlack,
        "highs",
    )
    rows = [row for snap in per_agent.installed_snapshots for row in snap]
    assert rows, "no aggregate row was ever installed"
    for row in rows:
        assert row.bundle_key == _expected_aggregate_key(row.phi, row.epsilon)

    # The key check references the row's own phi/epsilon, so also anchor the
    # installed values to the priced bundles.
    _assert_aggregate_value_anchor(toy, toy_problem(toy), "highs")

    # Two rows sharing phi but differing only in epsilon must get distinct
    # keys; a payload that omitted eps_agg would collide them.
    phi = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    key_a = _expected_aggregate_key(phi, 0.25)
    key_b = _expected_aggregate_key(phi, 0.75)
    assert key_a != key_b
    # The real key function must track epsilon the same way.
    from combrum.formulations.oneslack import _aggregate_key

    assert _aggregate_key(phi, 0.25) == key_a
    assert _aggregate_key(phi, 0.75) == key_b
    assert _aggregate_key(phi, 0.25) != _aggregate_key(phi, 0.75)


# --- OneSlack aggregate weights ---------------------------------------------
#
# The walk driver above uses agent_weights = np.ones(n_agents), where the
# per-agent weighting w_a * (phi_a | eps_a) is the identity, so the checks
# above never exercise the weight factor. These tests drive
# OneSlack.contribute directly under distinct non-unit weights and compare
# the returned matrix to hand-computed weighted rows. The master is only
# needed so setup() succeeds; contribute itself is solver-free, so HiGHS
# covers both families.

_CONTRIB_PROBLEMS = {"toy": (toy_problem, load_toy), "qkp": (qkp_problem, load_qkp)}


def _nonunit_weights(n_agents: int) -> np.ndarray:
    # Off 1.0 so the weighting is not the identity, and distinct per agent so
    # a mis-indexed weight vector shows up.
    return 1.5 + 0.25 * np.arange(n_agents, dtype=np.float64)


def _contribute_fixture(problem, arrays, features, weights, theta, local_ids=None):
    # A set-up OneSlack over `features`, the demands priced at `theta`, and
    # the expected (n_agents, K+1) matrix of w_a * (phi_a | eps_a) computed
    # from the family feature map. Returns (formulation, demands,
    # expected_rows, local_ids, master, oracle); the caller closes the master
    # and tears down the oracle.
    #
    # local_ids defaults to arange(n_agents). Pass a permutation to exercise
    # the id->position mapping: contribute prices the shard in local_ids
    # order but indexes the global agent_weights by id (weights[ids]), so the
    # caller must realign the ascending-id expected rows to demand order
    # before comparing.
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    K = problem.K
    transport = SerialTransport()
    if local_ids is None:
        local_ids = np.arange(n_agents, dtype=np.int64)
    else:
        local_ids = np.asarray(local_ids, dtype=np.int64)
    c_theta = np.zeros(K, dtype=np.float64)
    for a in range(n_agents):
        c_theta -= np.asarray(problem.observed_features(a, observed[a]))
    master = make_master(
        K, problem.theta_bounds, c_theta, (lambda agent_id: 1.0), backend="highs"
    )
    ctx = FitContext(
        K=K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=np.ones(n_agents, dtype=np.float64),
        agent_weights=weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = problem.oracle
    oracle.setup(transport, local_ids)
    demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
    expected = np.empty((n_agents, K + 1), dtype=np.float64)
    for a in range(n_agents):
        phi_a, eps_a = problem.features(a, demands[a].bundle)
        expected[a, :K] = weights[a] * np.asarray(phi_a, dtype=np.float64)
        expected[a, K] = weights[a] * float(eps_a)
    formulation = OneSlack(features)
    formulation.setup(ctx)
    return formulation, demands, expected, local_ids, master, oracle


def _pricing_theta(K: int) -> np.ndarray:
    # A single in-bounds pricing point. QKP needs alpha>=0 and lambda>=0 (the
    # first/last coords sit at box lower bound 0); a flat 0.3 clears the toy box.
    theta = np.full(K, 0.3, dtype=np.float64)
    theta[0] = 0.5
    theta[-1] = 0.1
    return theta


@needs_highs
@pytest.mark.parametrize("family", ["toy", "qkp"])
def test_aggregate_key_contribute_weights_per_agent_rows(family) -> None:
    # Per-agent (non-optimized) features path: contribute must return one
    # w_a * (phi_a | eps_a) row per local agent, keyed by that agent's id.
    problem_fn, load_fn = _CONTRIB_PROBLEMS[family]
    arrays = load_fn()
    problem = problem_fn(arrays)
    n_agents = int(np.asarray(arrays["observed"]).shape[0])
    weights = _nonunit_weights(n_agents)
    theta = _pricing_theta(problem.K)
    form, demands, expected, local_ids, master, oracle = _contribute_fixture(
        problem, arrays, problem.features, weights, theta
    )
    try:
        contribution = form.contribute(demands)
        terms = np.asarray(contribution.terms, dtype=np.float64)
        ids = np.asarray(contribution.ids)
        # ids in demand order == local_ids; the expected rows are in agent-id
        # order, so align both by id before the bytewise compare.
        assert ids.tolist() == local_ids.tolist()
        order = np.argsort(ids)
        np.testing.assert_array_equal(
            terms[order], expected[np.argsort(local_ids)]
        )
        # The weights must actually move the rows, else the compare above
        # says nothing about the weighting.
        unweighted = expected / weights[np.argsort(local_ids)][:, None]
        assert not np.allclose(expected[np.argsort(local_ids)], unweighted)
    finally:
        master.close()
        oracle.teardown()


class _AggregateFeatureMap(FeatureMap):
    """A batch feature map that also offers the aggregate mode.

    Resolution picks the OPTIMIZED path, so OneSlack.contribute takes its
    aggregate fast branch and hands ``weights[ids]`` to this member, which
    applies them verbatim (``w @ phi``).
    """

    def __init__(self, per_agent) -> None:
        self._per_agent = per_agent

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):
        id_arr = np.asarray(ids, dtype=np.int64)
        barr = np.asarray(bundles)
        rows = [self._per_agent(int(a), barr[r]) for r, a in enumerate(id_arr)]
        phi = np.stack(
            [np.asarray(p, dtype=np.float64) for p, _ in rows], axis=0
        )
        eps = np.array([float(e) for _, e in rows], dtype=np.float64)
        if aggregate:
            w = np.asarray(weights, dtype=np.float64)
            return w @ phi, float(w @ eps)
        return phi, eps


@needs_highs
@pytest.mark.parametrize("family", ["toy", "qkp"])
def test_aggregate_key_contribute_weights_optimized_aggregate(family) -> None:
    # OPTIMIZED aggregate fast path: contribute passes weights[ids] into
    # feature_batch_aggregate. The walk fixtures do not carry an aggregate
    # member, so this test drives a map that has one and compares the
    # returned aggregate row to the weighted-sum expectation.
    #
    # The shard runs on a reversed local_ids: with an identity shard,
    # weights[ids] is byte-equal to weights and a weights[ids] -> weights
    # mis-indexing is invisible. expected.sum(axis=0) sums over all agents,
    # so it is invariant to the shard order.
    problem_fn, load_fn = _CONTRIB_PROBLEMS[family]
    arrays = load_fn()
    problem = problem_fn(arrays)
    n_agents = int(np.asarray(arrays["observed"]).shape[0])
    weights = _nonunit_weights(n_agents)
    theta = _pricing_theta(problem.K)
    local_ids = np.arange(n_agents, dtype=np.int64)[::-1].copy()
    # An n_agents==1 (or accidentally identity) fixture would defeat the
    # reversal.
    assert local_ids.tolist() != list(range(n_agents)), (
        "reversed local_ids must be a non-identity permutation"
    )
    amap = _AggregateFeatureMap(problem.features)
    form, demands, expected, resolved_ids, master, oracle = _contribute_fixture(
        problem, arrays, amap, weights, theta, local_ids=local_ids
    )
    try:
        # The permutation must survive into the priced demands; a re-sorted
        # shard would mask the mis-indexing again.
        assert list(demands.keys()) == local_ids.tolist()
        assert resolved_ids.tolist() == local_ids.tolist()
        # weights[ids] must actually differ from weights.
        assert not np.array_equal(weights[local_ids], weights)
        assert form._features_res.mode.value == "optimized", (
            "aggregate map must resolve OPTIMIZED for the fast path to run"
        )
        contribution = form.contribute(demands)
        terms = np.asarray(contribution.terms, dtype=np.float64)
        # The fast path emits exactly one aggregate row under the aggregate id.
        assert terms.shape == (1, problem.K + 1)
        assert contribution.ids.tolist() == [oneslack_mod.AGGREGATE_AGENT_ID]
        expected_aggregate = expected.sum(axis=0)
        # w @ phi accumulates in a different order than the column sum, so
        # the two agree only to fp noise (~1e-15), not bitwise; 1e-9 sits far
        # above the noise and far below any mis-weighting (O(1) here).
        np.testing.assert_allclose(
            terms[0], expected_aggregate, rtol=0.0, atol=1e-9
        )
    finally:
        master.close()
        oracle.teardown()


# --- HiGHS support mask ------------------------------------------------------


def _support(row: CutRow) -> tuple[int, ...]:
    # Nonzero pattern of the returned CutRow's phi; the column set
    # _install_batch actually submits to addRows is checked in
    # test_support_mask_install_sparsifies_row.
    return tuple(int(i) for i in np.flatnonzero(row.phi))


def _assert_support_mask(
    arrays,
    base_problem,
    batch_only_map,
    formulation_cls,
    *,
    require_zeros: bool,
) -> None:
    per_agent, batched = _walk_pair(
        arrays, base_problem, batch_only_map, formulation_cls, "highs"
    )
    _assert_same_run(per_agent, batched)
    pairs = _aligned_rows(per_agent, batched)
    assert pairs, "no row was ever installed"
    zeros_present = any(
        (np.asarray(row_pa.phi) == 0.0).any() for row_pa, _ in pairs
    )
    if require_zeros:
        # NSlack: at least one installed phi must contain an exact zero, or
        # the mask comparison is trivially equal.
        assert zeros_present, (
            "no installed phi contains an exact zero"
        )
    else:
        # OneSlack's installed row is the dense aggregate, which fills every
        # column on these families; check that assumption still holds.
        assert not zeros_present, (
            "OneSlack aggregate unexpectedly sparse"
        )
    for row_pa, row_b in pairs:
        assert _support(row_pa) == _support(row_b)
        if not require_zeros:
            # For OneSlack the cross-path equality is trivially satisfied
            # (both sides are dense), so also require the aggregate to
            # occupy all K columns.
            full_support = tuple(range(base_problem.K))
            assert _support(row_pa) == full_support
            assert _support(row_b) == full_support


@needs_highs
def test_support_mask_nslack_support_mask_toy() -> None:
    toy = load_toy()
    _assert_support_mask(
        toy,
        toy_problem(toy),
        toy_feature_map_batch_only(toy),
        NSlack,
        require_zeros=True,
    )


@needs_highs
def test_support_mask_oneslack_support_mask_toy() -> None:
    toy = load_toy()
    _assert_support_mask(
        toy,
        toy_problem(toy),
        toy_feature_map_batch_only(toy),
        OneSlack,
        require_zeros=False,
    )


# --- HiGHS install sparse column set ----------------------------------------
#
# The support mask tests compare the two feature paths to each other, so a
# change that installs the dense row on both paths goes unseen. This test
# checks the column indices HighsMaster._install_batch actually passes to
# the solver's addRows against a hand-picked support.


@needs_highs
def test_support_mask_install_sparsifies_row() -> None:
    import highspy

    from combrum.masters.highs import HighsMaster

    K = 5
    # phi is nonzero exactly at columns 0, 2, 4, with structural zeros at 1
    # and 3. All nonzero coefficients are positive, so the row's theta
    # entries are strictly negative (-phi) and the one slack entry is +1.0,
    # letting us split the two without reading master internals.
    phi = np.array([2.0, 0.0, 3.0, 0.0, 5.0])
    expected_theta_cols = (0, 2, 4)
    expected_theta_vals = {0: -2.0, 2: -3.0, 4: -5.0}
    epsilon = 1.5

    captured: list[tuple[int, int, np.ndarray, np.ndarray, float, float]] = []
    orig_add_rows = highspy.Highs.addRows

    def spy_add_rows(self, num_rows, lower, upper, num_nz, starts, indices, values):
        captured.append(
            (
                int(num_rows),
                int(num_nz),
                np.asarray(indices).copy(),
                np.asarray(values).copy(),
                float(np.asarray(lower)[0]),
                float(np.asarray(upper)[0]),
            )
        )
        return orig_add_rows(
            self, num_rows, lower, upper, num_nz, starts, indices, values
        )

    highspy.Highs.addRows = spy_add_rows
    try:
        master = HighsMaster(
            K,
            (np.zeros(K), np.full(K, 10.0)),
            np.zeros(K),
            lambda agent_id: 1.0,
            n_agents=1,
        )
        master.add_cuts(
            [CutRow(rep_id=0, agent_id=0, phi=phi, epsilon=epsilon, bundle_key=b"k")]
        )
        master.close()
    finally:
        highspy.Highs.addRows = orig_add_rows

    assert len(captured) == 1, "expected exactly one addRows batch for one cut"
    num_rows, num_nz, indices, values, lower, upper = captured[0]
    assert num_rows == 1

    # The slack column is the sole +1.0 entry; every theta column is -phi<0.
    entries = dict(zip((int(i) for i in indices), (float(v) for v in values)))
    slack_cols = tuple(c for c, v in entries.items() if v == 1.0)
    theta_cols = tuple(sorted(c for c, v in entries.items() if v != 1.0))
    assert len(slack_cols) == 1, "the row must carry exactly one +1 slack column"

    # The submitted columns are the sparse support, not the dense range.
    assert theta_cols == expected_theta_cols
    # num_nz counts the theta support plus the one slack column.
    assert num_nz == len(expected_theta_cols) + 1
    # Coefficients: -phi at each submitted theta column, and the RHS is epsilon.
    assert {c: entries[c] for c in theta_cols} == expected_theta_vals
    assert lower == epsilon
    # The cut is one-sided `>=`: the RHS is the lower bound and the upper bound
    # stays at +infinity. An equality/upper-bounded row (upper == epsilon) would
    # over-constrain phi.theta + u == epsilon instead of the epigraph inequality.
    assert upper == highspy.kHighsInf
