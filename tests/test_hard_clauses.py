"""The two hard clauses: per-agent vs batched features must produce the
same discrete row identities end-to-end.

The batched-features path must not move a discrete identity a downstream
filter keys on. Each clause runs two walks differing only in the features
provider — the bare per-agent ``problem.features`` callable vs a
``*_batch_only`` FeatureMap — over the same family, backend, transport,
tolerance, schedule and penalty, so any difference in the captured
``installed_snapshots`` is attributable to the features path alone.

* **Aggregate identity.** The OneSlack aggregate row's identity is a
  SHA-256 over the float64 bytes of ``[phi_agg, eps_agg]``
  (``oneslack.py:_aggregate_key``), and the install gate reads the raw
  aggregate. A ``<=1e-13`` drift in the summed aggregate would change the
  row key — a discrete identity flip, not a continuous nudge — so the
  batched path must produce a byte-identical aggregate to the per-agent
  path. The clause compares the full installed row tuple bytewise.

* **Support mask.** ``HighsMaster._install``
  sparsifies every row with ``np.flatnonzero(row.phi)`` before handing the
  column set to ``addRow``; the nonzero pattern fixes the installed column
  set. A batched path that turns an exact ``0.0`` into a tiny nonzero (or
  back) changes installed structure, not just a coefficient — so this is
  discrete zero-mask equality, never ``<=1e-13``. The metamorphic clause
  compares the per-row nonzero pattern of ``row.phi`` across the per-agent
  and batched feature paths for both formulations on the HiGHS backend. The
  zero-mask "bites" on NSlack, whose rows are a single agent's ``b * r_a``
  and so carry structural exact zeros at the unselected items; the OneSlack
  installed row is the dense aggregate ``phi_agg = sum_a w_a * phi_a``,
  which fills every column on these families, so its meaningful coverage
  is the aggregate byte equality; the OneSlack support-mask check only pins
  that the batched aggregate stays full (no spurious zero) and the masks still match. The
  split is enforced via the ``require_zeros`` flag, which also asserts the
  dense/sparse assumption so a future regression cannot slip past either
  branch.

  That metamorphic pair is feature-path-differential: it cannot see a
  regression that stops sparsifying on *both* paths identically (e.g.
  ``_install`` installing the dense row). ``test_support_mask_install_sparsifies_row``
  closes that by spying on the solver's ``addRow`` and asserting the actual
  column set ``_install`` submits equals a hand-picked support, so it
  exercises ``_install``'s ``np.flatnonzero`` directly rather than
  re-deriving the mask on the returned ``CutRow``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
from _support.families import load_family
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow

FAMILY_DIR = Path(__file__).resolve().parent / "fixtures" / "families"

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


def load_toy() -> dict[str, np.ndarray]:
    return load_family("toy", FAMILY_DIR)


def load_qkp() -> dict[str, np.ndarray]:
    return load_family("qkp", FAMILY_DIR)


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
    """(per-agent walk, batched walk) — identical but for the features path.

    Both capture per-iteration installed snapshots; everything else
    (transport, backend, tolerance, schedule, penalty) is the default and
    identical across the two, so the snapshots are comparable cut-for-cut.
    """
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
    """Both walks must take the same shape of run before comparing rows.

    If the two paths converged in a different iteration count or to a
    different answer, a per-iteration snapshot comparison would be
    comparing different runs — so pin the run shape first.
    """
    assert a.converged and b.converged
    assert a.iterations == b.iterations
    assert a.result.theta_hat.tobytes() == b.result.theta_hat.tobytes()
    assert a.result.objective == b.result.objective
    # The per-iteration snapshot streams must line up one-to-one.
    assert len(a.installed_snapshots) == len(b.installed_snapshots)


def _aligned_rows(
    a: WalkOutcome, b: WalkOutcome
) -> list[tuple[CutRow, CutRow]]:
    """Every installed row paired across the two paths, snapshot by snapshot.

    extract_cuts() returns rows in canonical-key order, so the i-th row of
    snapshot t on one path is the i-th row of snapshot t on the other when
    the runs match. The row counts must agree at every iteration (a divergent
    install count is itself a structure difference, caught here).
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


def _oracle_aggregate(
    base_problem: FamilyProblem, theta: np.ndarray, n_agents: int
) -> tuple[np.ndarray, float]:
    # Independent aggregate at one pricing theta: price every agent with the
    # family oracle, featurise its bundle, and reduce w_a*(phi_a|eps_a) over
    # ascending agent id. This mirrors the *math* OneSlack must produce
    # (contribute's weighted rows + canonical_sum's ascending-id np.add.reduce)
    # but is computed here from the family oracle, never from OneSlack's
    # aggregate output — so it pins the installed numbers, not just their
    # path-to-path or key-to-key agreement. agent_weights default to 1.0 in the
    # walk (_walk.py), matched here.
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
    # Non-metamorphic value anchor for the aggregate. The byte-equality clause
    # and the SHA-256 anchor both derive their reference from the row's OWN phi/eps, so
    # a value bug in OneSlack.finalise (e.g. phi_agg = agg[:K] * 2.0, or a
    # scaled/dropped eps_agg) rides through untouched — both feature paths stay
    # byte-identical and the key still matches its own mutated payload. Pin the
    # installed aggregate numbers to an oracle instead: run one per-agent walk,
    # capture the pricing theta at every iteration (OneSlack.solve returns it at
    # the top of each loop), rebuild each aggregate from the oracle-priced
    # bundles, and require every installed row to reproduce the oracle's
    # phi/eps bitwise (the reduce is bit-reproducible on one rank).
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

    # One oracle aggregate per captured pricing theta, indexed by the same
    # SHA-256 identity the master keys installed rows on — recomputed here from
    # the oracle phi/eps via _expected_aggregate_key (the test's independent
    # digest), never through the src key function. First occurrence wins, so a
    # theta whose aggregate repeats a prior row's key does not overwrite it,
    # matching add_cuts' de-dup.
    oracle: dict[bytes, tuple[np.ndarray, float]] = {}
    for theta in captured:
        phi_agg, eps_agg = _oracle_aggregate(base_problem, theta, n_agents)
        key = _expected_aggregate_key(phi_agg, eps_agg)
        oracle.setdefault(key, (phi_agg, eps_agg))

    installed = per_agent.installed_snapshots[-1] if per_agent.installed_snapshots else ()
    assert installed, "aggregate identity value anchor unexercised: no aggregate row was installed"
    for row in installed:
        # The installed row's key must name an oracle-priced aggregate; a value
        # bug moves row.phi/epsilon (hence its digest) off every oracle key.
        assert row.bundle_key in oracle, (
            "installed aggregate key matches no oracle-priced aggregate:"
            " OneSlack shipped a row the priced bundles do not produce"
        )
        exp_phi, exp_eps = oracle[row.bundle_key]
        # Bitwise: the summed reduce is bit-reproducible on one rank, and phi
        # is compared in full so any single mutated coefficient is caught, not
        # just a uniform rescale.
        np.testing.assert_array_equal(np.asarray(row.phi), exp_phi)
        assert float(row.epsilon) == exp_eps


def _assert_aggregate_key(arrays, base_problem, batch_only_map, backend) -> None:
    per_agent, batched = _walk_pair(
        arrays, base_problem, batch_only_map, OneSlack, backend
    )
    _assert_same_run(per_agent, batched)
    pairs = _aligned_rows(per_agent, batched)
    # Non-vacuity: the walk actually installed at least one aggregate row.
    assert pairs, "aggregate identity unexercised: no aggregate row was ever installed"
    for row_pa, row_b in pairs:
        assert _row_bytes(row_pa) == _row_bytes(row_b)
    # The metamorphic clause and the key anchor both reference the row's own
    # payload, so a value bug in finalise slips both. Pin the aggregate numbers
    # to an independent oracle.
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
    # Independent restatement of the aggregate identity the module docstring
    # bills: SHA-256 over the float64 bytes of [phi_agg, eps_agg]. Spelled out
    # here from phi/epsilon directly — never through oneslack._aggregate_key —
    # so it is an oracle, not a mirror of the code under test.
    payload = np.empty(phi_agg.size + 1, dtype=np.float64)
    payload[: phi_agg.size] = np.asarray(phi_agg, dtype=np.float64)
    payload[phi_agg.size] = float(eps_agg)
    return hashlib.sha256(payload.tobytes()).digest()


@needs_highs
def test_aggregate_key_bundle_key_is_independent_sha256() -> None:
    # The metamorphic aggregate check only compares the two feature paths
    # against each other, so a shared change to the key derivation (e.g. dropping
    # eps_agg from the hashed payload) leaves both paths byte-identical and
    # slips through. Pin the installed aggregate's bundle_key to a digest
    # recomputed independently in the test from the row's own phi/epsilon.
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

    # The check above ties the key to the row's own phi/epsilon, so it cannot
    # see a value bug that moves both together (finalise scaling phi_agg or
    # eps_agg). Anchor the installed aggregate numbers to the priced-bundle
    # oracle so such a bug cannot ride through key-consistent.
    _assert_aggregate_value_anchor(toy, toy_problem(toy), "highs")

    # eps-membership by construction: two rows sharing phi but differing only in
    # epsilon must receive distinct keys. This holds independently of any walk —
    # a payload that omits eps_agg would collide these, silently pooling two
    # distinct aggregate identities into one installed cut.
    phi = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    key_a = _expected_aggregate_key(phi, 0.25)
    key_b = _expected_aggregate_key(phi, 0.75)
    assert key_a != key_b
    # And the digest a live OneSlack install would key on must track epsilon the
    # same way: build both rows through the real key function and confirm eps
    # moves the key. (Read-only: exercises the src key, does not stand in as the
    # oracle — the oracle is _expected_aggregate_key above.)
    from combrum.formulations.oneslack import _aggregate_key

    assert _aggregate_key(phi, 0.25) == key_a
    assert _aggregate_key(phi, 0.75) == key_b
    assert _aggregate_key(phi, 0.25) != _aggregate_key(phi, 0.75)


# --- OneSlack aggregate weights ---------------------------------------------
#
# Every aggregate anchor above runs through the walk driver, which pins
# agent_weights = np.ones(n_agents). With unit weights the per-agent weighting
# w_a * (phi_a | eps_a) is the identity, so any weight transform that fixes 1.0
# — w_a**2, 2*w_a-1, or dropping the factor outright — reproduces the same
# aggregate and slips every one of those anchors. contribute states the
# aggregate as sum_a w_a * phi_a; the factor is only exercised at w_a != 1.
#
# These tests drive OneSlack.contribute directly under distinct non-unit
# weights (none equal to 1, so w, w**2 and 2w-1 all differ per agent) and pin
# the *whole* returned matrix against a hand-computed weighted-rows oracle. The
# master is only needed so setup() succeeds; the contribute arithmetic under
# test is solver-free, so HiGHS covers both families.

_CONTRIB_PROBLEMS = {"toy": (toy_problem, load_toy), "qkp": (qkp_problem, load_qkp)}


def _nonunit_weights(n_agents: int) -> np.ndarray:
    # Strictly increasing off 1.0 so the three weight-fixing-at-1.0 mutations
    # (w**2, 2w-1, drop) each move at least one row; distinct per agent so a
    # mis-indexed weight vector is caught too.
    return 1.5 + 0.25 * np.arange(n_agents, dtype=np.float64)


def _contribute_fixture(problem, arrays, features, weights, theta, local_ids=None):
    # A set-up OneSlack over `features`, the demands priced at `theta`, and the
    # hand-derived (n_agents, K+1) matrix of w_a * (phi_a | eps_a) computed from
    # the family feature map — never from contribute — as the oracle. Returns
    # (formulation, demands, expected_rows, local_ids, master, oracle); the
    # caller closes the master and tears down the oracle.
    #
    # local_ids defaults to arange(n_agents) — every agent, in id order. Pass a
    # permutation to exercise the id->position mapping: contribute prices the
    # local shard in demand (i.e. local_ids) order but indexes the global
    # agent_weights by those ids (weights[ids]), so a non-identity local_ids
    # makes weights[ids] differ from weights and the ascending-id oracle rows
    # must be realigned to demand order by the caller before a bytewise compare.
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
    # w_a * (phi_a | eps_a) row per local agent, keyed by that agent's id. Pin
    # the full matrix and the id vector against the oracle so a mis-weighted,
    # mis-indexed, or partially-weighted row is caught wholesale — not just the
    # named w_a**2 regression.
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
        # ids in demand order == local_ids; the oracle rows are in agent-id
        # order, so align both by id before the bytewise compare.
        assert ids.tolist() == local_ids.tolist()
        order = np.argsort(ids)
        np.testing.assert_array_equal(
            terms[order], expected[np.argsort(local_ids)]
        )
        # Guard the guard: the weights must actually move the rows, else the
        # oracle would agree with an unweighted contribute and the test is
        # unexercised. Compare against the unweighted rows and require a difference.
        unweighted = expected / weights[np.argsort(local_ids)][:, None]
        assert not np.allclose(expected[np.argsort(local_ids)], unweighted)
    finally:
        master.close()
        oracle.teardown()


class _AggregateFeatureMap(FeatureMap):
    """A batch feature map that also offers the aggregate mode.

    Resolution picks the OPTIMIZED path, so OneSlack.contribute takes its
    aggregate fast branch and hands ``weights[ids]`` to this member. The member
    applies the weights it is given verbatim (``w @ phi``), so a regression that
    corrupts ``weights[ids]`` inside contribute before the call surfaces in the
    aggregate this returns.
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
    # OPTIMIZED aggregate fast-path: contribute passes weights[ids] into
    # feature_batch_aggregate. The walk fixtures do not carry an aggregate member,
    # so this test drives a map that has one and pins the returned aggregate row
    # against the weighted-sum oracle.
    #
    # The shard runs on a REVERSED local_ids (a non-identity permutation of the
    # agent range), so the weight vector contribute selects, weights[ids], is a
    # reordering of the global weights rather than the whole vector unchanged.
    # An identity shard (ids == arange) makes weights[ids] byte-equal to
    # weights, so a weights[ids] -> weights mis-indexing is
    # invisible; under the reversal it pairs every weight with the wrong agent's
    # phi and the aggregate the member returns moves by O(1) — caught here. The
    # aggregate oracle expected.sum(axis=0) is a set sum over all agents, so it
    # is invariant to the shard order and remains the correct target.
    problem_fn, load_fn = _CONTRIB_PROBLEMS[family]
    arrays = load_fn()
    problem = problem_fn(arrays)
    n_agents = int(np.asarray(arrays["observed"]).shape[0])
    weights = _nonunit_weights(n_agents)
    theta = _pricing_theta(problem.K)
    local_ids = np.arange(n_agents, dtype=np.int64)[::-1].copy()
    # Guard the guard: a non-identity shard is what makes weights[ids] != weights
    # and thus makes the mis-indexing observable. An n_agents==1 (or accidentally
    # identity) fixture would silently restore the blind spot.
    assert local_ids.tolist() != list(range(n_agents)), (
        "reversed local_ids must be a non-identity permutation so weights[ids]"
        " reorders the global weights"
    )
    amap = _AggregateFeatureMap(problem.features)
    form, demands, expected, resolved_ids, master, oracle = _contribute_fixture(
        problem, arrays, amap, weights, theta, local_ids=local_ids
    )
    try:
        # Confirm the shard actually reaches contribute in the permuted order:
        # if the fixture silently re-sorted the ids the mis-indexing would again
        # be masked, so the permutation must survive into the priced demands.
        assert list(demands.keys()) == local_ids.tolist()
        assert resolved_ids.tolist() == local_ids.tolist()
        # weights[ids] and weights must actually differ; otherwise the reordering
        # is a no-op and the regression would ride through this test too.
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
        # The member reduces w @ phi in a different accumulation order than the
        # oracle's column sum, so the two agree only to fp noise (~1e-15), not
        # bitwise. atol stays far below any weighting signal (weights[ids]
        # -> weights, w**2, 2w-1 all move the aggregate by O(1) on these
        # reversed-shard fixtures) and far above the noise.
        np.testing.assert_allclose(
            terms[0], expected_aggregate, rtol=0.0, atol=1e-9
        )
    finally:
        master.close()
        oracle.teardown()


# --- HiGHS support mask ------------------------------------------------------


def _support(row: CutRow) -> tuple[int, ...]:
    # The nonzero pattern of the row's phi, re-derived on the returned CutRow.
    # This is the feature-path-differential probe; the column set _install
    # actually submits to addRow is checked directly in
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
    # Non-vacuity #1: the walk installed at least one row to compare.
    assert pairs, "support mask unexercised: no row was ever installed"
    zeros_present = any(
        (np.asarray(row_pa.phi) == 0.0).any() for row_pa, _ in pairs
    )
    if require_zeros:
        # Non-vacuity #2 (NSlack): at least one installed phi contains an
        # exact zero, so the mask comparison bites — an all-nonzero phi would
        # make flatnonzero trivially equal regardless of the features path.
        assert zeros_present, (
            "support mask unexercised: no installed phi contains an exact zero"
        )
    else:
        # OneSlack's installed row is the dense aggregate phi_agg =
        # sum_a w_a * phi_a, which fills every column on these families, so
        # there is no exact zero to perturb. The real coverage for OneSlack
        # is the aggregate byte equality; here we pin that the aggregate stays dense so
        # this branch cannot silently mask a regression to sparse rows.
        assert not zeros_present, (
            "OneSlack aggregate unexpectedly sparse: dense-aggregate"
            " assumption no longer holds — re-scope the zero coverage"
        )
    for row_pa, row_b in pairs:
        # Discrete: the exact nonzero pattern (the installed columns) is
        # identical across the per-agent and batched feature paths. For
        # NSlack the mask carries structural zeros (the probe bites); for
        # OneSlack the mask is full on both paths (no spurious zero crept in).
        assert _support(row_pa) == _support(row_b)
        if not require_zeros:
            # OneSlack: pin the support to the full column set. With no
            # structural zeros to perturb, the cross-path equality above is
            # trivially satisfied (both sides are the dense aggregate), so it
            # cannot catch a densification/column-drop regression that hits
            # both feature paths identically. The dense aggregate must occupy
            # every one of the K columns; anything shorter is a dropped column,
            # anything reordered is a mangled identity.
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


# --- HiGHS _install sparse column set ---------------------------------------
#
# The support mask tests are feature-path-differential, so they cannot
# catch a regression that installs the dense row on both paths identically.
# This test observes the column indices HighsMaster._install actually passes
# to the solver's addRow and pins them to a hand-picked support, exercising
# _install's np.flatnonzero(row.phi) instead of re-deriving it on the CutRow.


@needs_highs
def test_support_mask_install_sparsifies_row() -> None:
    import highspy

    from combrum.masters.highs import HighsMaster

    K = 5
    # Hand-picked support: phi is nonzero exactly at columns 0, 2, 4 and holds
    # structural zeros at 1 and 3. The expected column set is stated here by
    # construction — not via np.flatnonzero — so it is an independent oracle
    # for what _install must submit. All nonzero coefficients are positive, so
    # the row's theta entries carry strictly negative values (-phi) and the one
    # slack entry carries +1.0, letting us split the two without reading any
    # master internals.
    phi = np.array([2.0, 0.0, 3.0, 0.0, 5.0])
    expected_theta_cols = (0, 2, 4)
    expected_theta_vals = {0: -2.0, 2: -3.0, 4: -5.0}
    epsilon = 1.5

    captured: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    orig_add_row = highspy.Highs.addRow

    def spy_add_row(self, lower, upper, num_nz, indices, values):
        captured.append(
            (
                int(num_nz),
                np.asarray(indices).copy(),
                np.asarray(values).copy(),
                float(lower),
                float(upper),
            )
        )
        return orig_add_row(self, lower, upper, num_nz, indices, values)

    highspy.Highs.addRow = spy_add_row
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
        highspy.Highs.addRow = orig_add_row

    assert len(captured) == 1, "expected exactly one addRow for one installed cut"
    num_nz, indices, values, lower, upper = captured[0]

    # The slack column is the sole +1.0 entry; every theta column is -phi<0.
    entries = dict(zip((int(i) for i in indices), (float(v) for v in values)))
    slack_cols = tuple(c for c, v in entries.items() if v == 1.0)
    theta_cols = tuple(sorted(c for c, v in entries.items() if v != 1.0))
    assert len(slack_cols) == 1, "the row must carry exactly one +1 slack column"

    # Discrete: the exact columns _install submitted are the sparse support,
    # never the dense column range — the structural zeros at 1 and 3 are gone.
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
