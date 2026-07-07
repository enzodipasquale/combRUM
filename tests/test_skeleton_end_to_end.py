"""End-to-end conformance for the skeleton fixture.

The fixture pins the public interfaces together: Oracle pricing, the
Formulation walk, CutRow exchange, and canonical reductions. The known-green toy
fit reaches zero regret bitwise identically under serial and multirank
transports, with O(1) collectives per iteration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from combrum.formulation import Evaluation
from _support.commprobe import _ROW_HEADER_BYTES, CountingTransport
from _support.families import (
    DEFAULT_SEED,
    load_family,
    qkp_family,
    toy_family,
)
from _support.skeleton import (
    QkpProblem,
    SkeletonFormulation,
    SkeletonOracle,
    ToyProblem,
    run_skeleton,
)
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import CutRow

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "families"


def load_toy() -> dict[str, np.ndarray]:
    return load_family("toy", FIXTURE_DIR)


def load_qkp() -> dict[str, np.ndarray]:
    return load_family("qkp", FIXTURE_DIR)


def _qkp_argmax_bundle(
    qkp: dict[str, np.ndarray], theta: np.ndarray, agent: int
) -> np.ndarray:
    # Independent constrained argmax over {0,1}^M, structurally distinct from
    # QkpProblem.price: explicit bit loops, no shared enumeration/phi cache.
    x = qkp["x"][agent]
    q = qkp["Q"]
    weights = qkp["weights"]
    capacity = float(qkp["capacities"][agent])
    nu = qkp["shocks"][agent, 0, :]
    n_items = x.shape[0]
    alpha = float(theta[0])
    delta = theta[1 : 1 + n_items]
    lam = float(theta[-1])
    best_bundle: np.ndarray | None = None
    best_util = -np.inf
    for mask in range(1 << n_items):
        bundle = np.array(
            [(mask >> k) & 1 for k in range(n_items)], dtype=np.float64
        )
        if float(bundle @ weights) > capacity:
            continue
        modular = float(np.sum(bundle * (alpha * x - delta + nu)))
        quad = 0.0
        for j in range(n_items):
            for k in range(n_items):
                quad += 0.5 * lam * bundle[j] * q[j, k] * bundle[k]
        util = modular + quad
        if util > best_util:
            best_util = util
            best_bundle = bundle
    assert best_bundle is not None
    return best_bundle > 0.5


def assert_results_bitwise_equal(a, b) -> None:
    assert a.theta_hat.tobytes() == b.theta_hat.tobytes()
    assert a.objective == b.objective
    assert a.n_active_cuts == b.n_active_cuts
    assert a.metadata == b.metadata


# The toy theta_hat the vehicle produced before the family-parameterized
# refactor: the refactor routes toy pricing/payoff/cut-moment through the
# ToyProblem adapter using the identical float expressions, so the walk
# must reproduce these exact bytes. A mismatch means the adapter perturbed
# the toy arithmetic — the one thing the refactor was forbidden to do.
_TOY_THETA_HAT_HEX = (
    "ceccccccccccecbf010000000000f8bf000000000000f03f"
    "333333333333fbbf343333333333e3bf"
)


def test_refactor_preserves_toy_theta_hat_bitwise() -> None:
    res = run_skeleton(load_toy(), SerialTransport())
    assert res.theta_hat.tobytes().hex() == _TOY_THETA_HAT_HEX
    assert res.n_active_cuts == 39
    assert res.metadata["n_iterations"] == 8


# --- end-to-end serial -------------------------------------------------------


def test_serial_run_converges_to_exact_zero_regret() -> None:
    toy = load_toy()
    res = run_skeleton(toy, SerialTransport())
    # The objective is minus the best max regret; the fixture is
    # rationalisable, so it converges to exactly 0.0.
    assert res.objective == 0.0
    assert res.metadata["best_total_regret"] == 0.0
    assert res.theta_hat.shape == toy["theta_true"].shape
    # A real solve ships cuts and moves theta off the zero seed; both fail
    # for a stuck walk (zeroed step) or a dropped cut tally that objective
    # == 0.0 alone would not catch.
    assert res.n_active_cuts > 0
    assert np.any(res.theta_hat != 0.0)


# One item per agent, r = +1, distinct negative shocks so pricing at the zero
# seed takes nothing and every agent is violated. best_total_regret recorded on
# the first step (before the theta move) is then sum |nu| over agents. The
# magnitudes are order-sensitive: 2**53 has ULP 2.0, so the low bit of the sum
# depends on the accumulation order, and a round-robin shard delivers the ids
# unsorted. The convergence tests only ever read best_total_regret at 0.0, where
# every per-agent term is already zero; this fixture reads it at a known nonzero
# value, so a mis-combined sum_reproducible total (or a dropped id-sort) shows.
_MIDWALK_BIG = 2.0**53
_MIDWALK_NU = np.array([-_MIDWALK_BIG, -1.0, -2.0, -4.0])
_MIDWALK_ARRAYS: dict[str, np.ndarray] = {
    "observables": np.ones((4, 1)),
    "shocks": _MIDWALK_NU.reshape(4, 1, 1),
    "observed": np.ones((4, 1), dtype=bool),
    # theta_true large enough that pricing there reproduces every observed
    # bundle; the recorded total is still taken at the zero seed.
    "theta_true": np.array([_MIDWALK_BIG + 10.0]),
}
# Hand-derived canonical (ascending-id) accumulation with IEEE round-half-even
# at ULP 2.0: 2**53 + 1 -> 2**53, + 2 -> 2**53 + 2, + 4 -> 2**53 + 6. A naive
# arrival-order reduce over the shard-delivered [2**53, 2, 1, 4] gives 2**53 + 4,
# one ULP off; the canonical id-sort is what recovers 2**53 + 6.
_MIDWALK_EXPECTED_TOTAL = _MIDWALK_BIG + 6.0


def test_best_total_regret_pins_nonzero_mid_walk_sum() -> None:
    # The end-to-end zero-regret tests can never witness a wrong mid-walk total:
    # they only read best_total_regret at convergence (0.0). Drive one step of a
    # non-converged fixture through the public run/result path and pin the total
    # to the hand-summed oracle, bit-for-bit, on serial and on a two-rank shard.
    serial = run_skeleton(
        _MIDWALK_ARRAYS, SerialTransport(), family="toy", max_iterations=1
    )
    # One iteration is not enough to rationalise this fixture, so the recorded
    # total is genuinely nonzero (the contract only surfaces off the 0.0 case).
    assert serial.objective != 0.0
    assert serial.metadata["best_total_regret"].hex() == (
        _MIDWALK_EXPECTED_TOTAL.hex()
    )
    # The interleaved shard delivers ids [0, 2, 1, 3] unsorted, so the canonical
    # id-sort inside sum_reproducible is load-bearing; every rank must still land
    # on the same hand-derived value, killing an arrival-order (drop-sort) reduce.
    multi = LocalCluster(size=2).run(
        lambda transport: run_skeleton(
            _MIDWALK_ARRAYS, transport, family="toy", max_iterations=1
        ).metadata["best_total_regret"]
    )
    assert all(
        total.hex() == _MIDWALK_EXPECTED_TOTAL.hex() for total in multi
    )


def test_theta_hat_rationalises_the_data() -> None:
    # Recomputed independently of the walk: at theta_hat every observed
    # bundle must be optimal. Toy demand is item-separable, so an item is
    # taken exactly when its score is positive; rationalisation means the
    # sign of every score agrees with observed, strictly (no item sits on
    # the zero-score knife edge on this fixture).
    toy = load_toy()
    res = run_skeleton(toy, SerialTransport())
    scores = toy["observables"] * res.theta_hat[None, :] + toy["shocks"][:, 0, :]
    assert np.array_equal(scores > 0.0, toy["observed"])
    # And the fit is interior, not marginal: every score is bounded away
    # from the knife edge, so a tie-break change could not flip the demand.
    assert np.min(np.abs(scores)) > 0.0
    # The same statement in payoff form: observed achieves the maximum.
    best = np.clip(scores, 0.0, None).sum(axis=1)
    achieved = np.where(toy["observed"], scores, 0.0).sum(axis=1)
    assert np.all(achieved == best)


def test_oracle_at_theta_true_reproduces_observed_bundles() -> None:
    # The identity that makes the fixture rationalisable: the oracle
    # implements the same demand rule the generator used.
    toy = load_toy()
    n_obs = toy["observed"].shape[0]
    theta_true = toy["theta_true"]
    r = toy["observables"]
    nu = toy["shocks"][:, 0, :]
    oracle = SkeletonOracle(ToyProblem(toy))
    oracle.setup(SerialTransport(), np.arange(n_obs, dtype=np.int64))
    for a in range(n_obs):
        demand = oracle.price(theta_true, a)
        assert np.array_equal(np.asarray(demand.bundle), toy["observed"][a]), a
        # The priced payoff must equal the observed bundle's payoff recomputed
        # from the fixture (sum of positive item scores). Demand.exact hardcodes
        # gap == 0.0, so only this independent value discriminates a wrong payoff.
        scores = r[a] * theta_true + nu[a]
        expected_payoff = float(np.where(toy["observed"][a], scores, 0.0).sum())
        assert demand.payoff == expected_payoff, a
    oracle.teardown()


# --- determinism + rank invariance -------------------------------------------


def test_serial_runs_are_bitwise_identical() -> None:
    toy = load_toy()
    first = run_skeleton(toy, SerialTransport())
    second = run_skeleton(toy, SerialTransport())
    assert_results_bitwise_equal(first, second)


@pytest.mark.parametrize("size", [2, 4])
def test_local_cluster_matches_serial_bitwise(size: int) -> None:
    # Interleaved shards (a % size == rank) re-route every cut and every
    # reduction contribution; the result must match the serial answer bitwise.
    toy = load_toy()
    serial = run_skeleton(toy, SerialTransport())
    results = LocalCluster(size).run(lambda transport: run_skeleton(toy, transport))
    assert len(results) == size
    for res in results:
        assert_results_bitwise_equal(res, serial)


# Four cuts owned by rank 0, contributed by non-rank-0 shards so the pooled
# (rank-major) order does NOT match the canonical (agent-id) order, and phi
# values whose float sum is order-dependent. The toy end-to-end fixture cannot
# reach this contract (its {-1,0,1} moments sum order-invariantly), so exercise
# exchange_cuts directly.
#
# The magnitudes are chosen so the reduced float differs for the canonical
# order AND for BOTH rank-major pooled orders the parametrized sizes produce
# ([1, 3, 0, 2] at size=2, [3, 2, 1, 0] at size=4). Ascending agent-id
# (canonical) sums to 3.0; the size=2 pooled order sums to 4.0 and the size=4
# pooled order to 5.0. So the reduced phi witnesses canonical delivery order
# at every size, not just size=2.
_CUT_PHI: dict[int, float] = {0: 1.0, 1: 1e16, 2: -1e16, 3: 3.0}

# Independent oracle: the phi sum in canonical (agent-id ascending) order,
# accumulated by an explicit ascending-id loop that shares no code with
# exchange_cuts or the reference reduction.
_CANONICAL_PHI_SUM: float = float(
    np.add.reduce(
        np.array([[_CUT_PHI[a]] for a in sorted(_CUT_PHI)], dtype=np.float64)
    )[0]
)


def _cut_row(agent_id: int) -> CutRow:
    return CutRow(
        rep_id=0,
        agent_id=agent_id,
        phi=np.array([_CUT_PHI[agent_id]], dtype=np.float64),
        epsilon=1.0,
        bundle_key=bytes([agent_id + 1]),
    )


def _cut_rows_for_rank(rank: int, size: int) -> list[CutRow]:
    # Reverse each rank block's agent order so the rank-major pooled order
    # differs from canonical (agent-id ascending) at every size.
    return [
        _cut_row(agent_id)
        for agent_id in _CUT_PHI
        if (size - 1) - (agent_id % size) == rank
    ]


@pytest.mark.parametrize("size", [2, 4])
def test_exchange_cuts_delivers_canonical_order_across_ranks(size: int) -> None:
    # exchange_cuts must deliver cuts to the owner in canonical (agent-id)
    # order regardless of which rank contributed them, so the reduced phi is
    # bitwise identical to serial even when float addition is non-associative.
    owners = np.zeros(1, dtype=np.int64)
    serial = SerialTransport().exchange_cuts(
        [_cut_row(a) for a in _CUT_PHI], owners
    )
    # Expected order derived independently: canonical key sorts by agent_id.
    expected_order = tuple(sorted(_CUT_PHI))
    assert tuple(row.agent_id for row in serial) == expected_order
    serial_sum = float(np.add.reduce(np.stack([row.phi for row in serial]))[0])
    # Serial delivery is the pool-of-one canonical order, so its reduced phi
    # must already equal the hand-derived ascending-id oracle bit-for-bit.
    assert serial_sum == _CANONICAL_PHI_SUM

    def on_rank(transport):
        received = transport.exchange_cuts(
            _cut_rows_for_rank(transport.rank, transport.size), owners
        )
        reduced = (
            float(np.add.reduce(np.stack([row.phi for row in received]))[0])
            if received
            else 0.0
        )
        return tuple(row.agent_id for row in received), reduced

    results = LocalCluster(size).run(on_rank)
    owner_order, owner_phi_sum = results[0]
    assert owner_order == expected_order
    # Pin the reduced phi against the independent canonical-order oracle, not
    # just against serial: the crafted magnitudes make the size=2 pooled order
    # sum to 4.0 and the size=4 pooled order to 5.0, so any non-canonical
    # delivery order is caught at BOTH sizes, not only size=2.
    assert owner_phi_sum == _CANONICAL_PHI_SUM
    # Every non-owner rank receives nothing (all cuts are owned by rank 0).
    for order, reduced in results[1:]:
        assert order == ()
        assert reduced == 0.0


# --- the evaluated-step channel ----------------------------------------------


def test_update_consumes_the_same_evaluation_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The typed evaluate -> update channel must be real inside the actual
    # walk: spy on both methods and demand object identity, not equality.
    returned: list[Evaluation] = []
    consumed: list[Evaluation] = []
    orig_evaluate = SkeletonFormulation.evaluate
    orig_update = SkeletonFormulation.update

    def spy_evaluate(self: SkeletonFormulation, demands: object) -> Evaluation:
        evaluation = orig_evaluate(self, demands)
        returned.append(evaluation)
        return evaluation

    def spy_update(self: SkeletonFormulation, step: Evaluation) -> int:
        consumed.append(step)
        return orig_update(self, step)

    monkeypatch.setattr(SkeletonFormulation, "evaluate", spy_evaluate)
    monkeypatch.setattr(SkeletonFormulation, "update", spy_update)
    run_skeleton(load_toy(), SerialTransport())
    assert returned and len(returned) == len(consumed)
    assert all(a is b for a, b in zip(returned, consumed))
    # The channel is load-bearing: the first step is violated at
    # theta = 0, so its payload must carry violated-agent data for
    # update() to turn into cuts.
    assert returned[0].violation > 0.0
    assert returned[0].payload[1]


# --- comm discipline -----------------------------------------------------------

# The complete set of collective kinds a real solve fires, one round of each per
# iteration. Pinning the whole set (not just the audited four) rejects a stray
# extra collective per iteration that per-kind counts alone would miss.
_EXPECTED_COLLECTIVE_KINDS: frozenset[str] = frozenset(
    ("sum_reproducible", "allreduce_max", "exchange_cuts", "bcast", "collective_guard")
)


def test_comm_rounds_per_iteration_o1_and_bytes_scale_with_cuts() -> None:
    audited = ("sum_reproducible", "allreduce_max", "exchange_cuts", "bcast")
    rounds_per_iteration: list[dict[str, float]] = []
    iteration_counts: list[int] = []
    for arrays in (load_toy(), toy_family(24, 5, DEFAULT_SEED)):
        probe = CountingTransport(SerialTransport())
        res = run_skeleton(arrays, probe)
        assert res.objective == 0.0  # the audit only counts a real solve
        iterations = res.metadata["n_iterations"]
        assert iterations >= 2
        iteration_counts.append(iterations)
        counts = probe.counts()
        # O(1) collectives per iteration: exactly one round of each kind
        # per iteration and none at setup, whatever N is.
        for kind in audited:
            assert counts[kind] == iterations, kind
        assert counts["collective_guard"] == iterations
        # No other collective kind fired: the exact key set is the expected one,
        # so a spurious per-iteration collective (a stray scatter/gather/node
        # broadcast) is caught, not just an inflated count of an audited kind.
        assert frozenset(counts) == _EXPECTED_COLLECTIVE_KINDS
        rounds_per_iteration.append(
            {kind: counts[kind] / iterations for kind in audited}
        )
        # Exchange payload scales with the cuts shipped (= n_active_cuts
        # under one serial rank), never with N * iterations.
        n_obs, n_items = arrays["observed"].shape
        row_bytes = n_items * 8 + 1 + _ROW_HEADER_BYTES  # phi + key + header
        assert probe.bytes_moved()["exchange_cuts"] == res.n_active_cuts * row_bytes
        assert 0 < res.n_active_cuts < n_obs * iterations
    # The two fixtures take a different number of iterations, so the round-count
    # comparison below is a genuine cross-run check, not two identical runs.
    assert iteration_counts[0] != iteration_counts[1]
    # Every kind runs exactly once per iteration on both fixtures, independent of
    # N: both ratio dicts equal the hand-derived one-round-per-iteration target.
    expected_ratio = {kind: 1.0 for kind in audited}
    assert rounds_per_iteration[0] == expected_ratio
    assert rounds_per_iteration[1] == expected_ratio


# --- QKP family: the same vehicle, a different adapter ------------------------


def test_qkp_serial_converges_to_exact_zero_regret() -> None:
    qkp = load_qkp()
    res = run_skeleton(qkp, SerialTransport(), family="qkp")
    # K = M + 2 for the [alpha, delta, lambda] parameterisation.
    assert res.theta_hat.shape == qkp["theta_true"].shape
    assert res.objective == 0.0
    assert res.metadata["best_total_regret"] == 0.0
    # The old [-10, 10] box check was unexercised: this fixture converges to an
    # interior fixpoint, so the theta_bounds clip never binds (that contract is
    # tested in _support/skeleton.py::test_update_enforces_theta_bounds_clip).
    # Instead pin the meaning of "zero regret" as a full-output oracle: at
    # theta_hat the whole observed demand matrix is reproduced by a test-owned
    # brute-force argmax that shares no code with QkpProblem.price. Comparing
    # the entire (N, M) matrix at once kills any theta_hat corruption that
    # breaks rationalisation, not just one failure mode.
    assert np.all(np.isfinite(res.theta_hat))
    priced = np.array(
        [_qkp_argmax_bundle(qkp, res.theta_hat, a) for a in range(qkp["observed"].shape[0])]
    )
    assert priced.dtype == np.bool_
    np.testing.assert_array_equal(priced, qkp["observed"].astype(bool))


def test_qkp_theta_hat_rationalises_the_data() -> None:
    # Recomputed independently of the walk AND of QkpProblem.price: at
    # theta_hat a test-owned brute-force constrained argmax reproduces every
    # observed bundle. Sharing QkpProblem.price would let an echo-observed
    # pricing bug (theta_hat stuck at 0, argmax replaced by the observed
    # bundle) pass; the independent oracle rejects theta_hat = 0.
    qkp = load_qkp()
    res = run_skeleton(qkp, SerialTransport(), family="qkp")
    n_obs = qkp["observed"].shape[0]
    for a in range(n_obs):
        priced = _qkp_argmax_bundle(qkp, res.theta_hat, a)
        assert np.array_equal(priced, qkp["observed"][a]), a


def test_qkp_oracle_at_theta_true_reproduces_observed_bundles() -> None:
    qkp = load_qkp()
    n_obs = qkp["observed"].shape[0]
    theta_true = qkp["theta_true"]
    x = qkp["x"]
    q = qkp["Q"]
    nu = qkp["shocks"][:, 0, :]
    n_items = qkp["observed"].shape[1]
    alpha = float(theta_true[0])
    delta = theta_true[1 : 1 + n_items]
    lam = float(theta_true[-1])
    oracle = SkeletonOracle(QkpProblem(qkp))
    oracle.setup(SerialTransport(), np.arange(n_obs, dtype=np.int64))
    for a in range(n_obs):
        demand = oracle.price(theta_true, a)
        assert np.array_equal(np.asarray(demand.bundle), qkp["observed"][a]), a
        # Independent payoff of the observed bundle from the fixture utility;
        # Demand.exact hardcodes gap == 0.0, so only this value catches a
        # wrong payoff. Tolerance absorbs the reassociated float sum.
        b = qkp["observed"][a].astype(np.float64)
        expected_payoff = float(
            alpha * (x[a] @ b) - delta @ b + 0.5 * lam * (b @ q @ b) + nu[a] @ b
        )
        assert demand.payoff == pytest.approx(expected_payoff, abs=1e-9), a
    oracle.teardown()


def test_qkp_serial_runs_are_bitwise_identical() -> None:
    qkp = load_qkp()
    first = run_skeleton(qkp, SerialTransport(), family="qkp")
    second = run_skeleton(qkp, SerialTransport(), family="qkp")
    assert_results_bitwise_equal(first, second)


@pytest.mark.parametrize("size", [2, 4])
def test_qkp_local_cluster_matches_serial_bitwise(size: int) -> None:
    qkp = load_qkp()
    serial = run_skeleton(qkp, SerialTransport(), family="qkp")
    results = LocalCluster(size).run(
        lambda transport: run_skeleton(qkp, transport, family="qkp")
    )
    assert len(results) == size
    for res in results:
        assert_results_bitwise_equal(res, serial)


def test_qkp_comm_rounds_per_iteration_o1_and_n_independent() -> None:
    audited = ("sum_reproducible", "allreduce_max", "exchange_cuts", "bcast")
    rounds_per_iteration: list[dict[str, float]] = []
    iteration_counts: list[int] = []
    for arrays in (load_qkp(), qkp_family(20, 6, DEFAULT_SEED)):
        probe = CountingTransport(SerialTransport())
        res = run_skeleton(arrays, probe, family="qkp")
        assert res.objective == 0.0  # real solve
        iterations = res.metadata["n_iterations"]
        assert iterations >= 2
        iteration_counts.append(iterations)
        counts = probe.counts()
        for kind in audited:
            assert counts[kind] == iterations, kind
        assert counts["collective_guard"] == iterations
        # No other collective kind fired: pin the exact key set so a stray
        # per-iteration collective is caught, not just an inflated audited count.
        assert frozenset(counts) == _EXPECTED_COLLECTIVE_KINDS
        rounds_per_iteration.append(
            {kind: counts[kind] / iterations for kind in audited}
        )
        # Exchange payload scales with the cuts shipped, never with N: each
        # CutRow carries a length-K phi (K = M + 2), the packed bundle key
        # (ceil(M/8) bytes), and the fixed header.
        n_items = arrays["observed"].shape[1]
        k = n_items + 2
        key_bytes = (n_items + 7) // 8
        row_bytes = k * 8 + key_bytes + _ROW_HEADER_BYTES
        assert probe.bytes_moved()["exchange_cuts"] == res.n_active_cuts * row_bytes
    # The two fixtures take a different number of iterations, so the round-count
    # comparison below is a genuine cross-run check, not two identical runs.
    assert iteration_counts[0] != iteration_counts[1]
    # Every kind runs exactly once per iteration on both fixtures, independent of
    # N: both ratio dicts equal the hand-derived one-round-per-iteration target.
    expected_ratio = {kind: 1.0 for kind in audited}
    assert rounds_per_iteration[0] == expected_ratio
    assert rounds_per_iteration[1] == expected_ratio
