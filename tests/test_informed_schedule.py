"""DualInformed schedule: unit semantics + the live pricing-budget gate."""

from __future__ import annotations

import numpy as np
import pytest

from _family_oracles import toy_problem
from _walk import run_walk
from _support.families import DEFAULT_SEED, toy_family
from combrum.informed_schedule import (
    _SUPPORT_ATOL,
    DualConcentration,
    DualInformed,
)
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters.gurobi import GurobiMaster
from combrum.formulations import NSlack
from combrum.transport import LocalCluster, SerialTransport

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()
REAL_BACKENDS = (
    pytest.param(
        "gurobi",
        marks=pytest.mark.skipif(not GUROBI_AVAILABLE, reason="no gurobi"),
    ),
    pytest.param(
        "highs",
        marks=pytest.mark.skipif(not HIGHS_AVAILABLE, reason="no highs"),
    ),
)

PARITY_BAND = 1e-9
# A fixture big enough that skipping settled agents actually saves pricing:
# on a tiny fit the forced re-certification sweeps dominate the budget.
N_OBS, N_ITEMS = 40, 6
PERIOD = 5


# --- DualConcentration --------------------------------------------------------


def test_concentration_from_cut_duals_normalizes_per_agent() -> None:
    duals = {
        (3, b"x"): 3.0,
        (3, b"y"): 1.0,  # agent 3 total 4.0, max share 0.75
        (1, b"z"): 2.0,  # agent 1 single cut, share 1.0
        (5, b"w"): 1e-20,  # below support atol: dropped
    }
    conc = DualConcentration.from_cut_duals(duals)
    # Strictly increasing agent ids, the agent with no real support gone.
    assert conc.agent_ids.tolist() == [1, 3]
    np.testing.assert_allclose(conc.max_weights, [1.0, 0.75])


def test_concentration_from_cut_duals_support_tolerance_band() -> None:
    # The two small masses bracket the 1e-10 support tolerance from both sides,
    # so any drift larger than 2x flips membership.
    duals = {
        (1, b"a"): 2.0,  # agent 1 single cut, share 1.0
        (3, b"b"): 2e-10,  # just above 1e-10: KEEP (share 1.0)
        (5, b"c"): 5e-11,  # just below 1e-10 noise: DROP
    }
    conc = DualConcentration.from_cut_duals(duals)
    # both kept agents are single-cut, so share 1.0
    assert conc.agent_ids.tolist() == [1, 3]
    np.testing.assert_allclose(conc.max_weights, [1.0, 1.0])
    # The live test below reads _SUPPORT_ATOL on both sides of its comparison,
    # so pin the literal here.
    assert _SUPPORT_ATOL == 1e-10


def test_concentration_rejects_unnormalized_or_unsorted() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        DualConcentration(
            agent_ids=np.array([2, 1]), max_weights=np.array([0.5, 0.5])
        )
    with pytest.raises(ValueError, match="lie in"):
        DualConcentration(
            agent_ids=np.array([1]), max_weights=np.array([1.5])
        )
    # lower end of the (0, 1] contract: a share is positive mass over a
    # positive total, so non-positive weights are malformed
    with pytest.raises(ValueError, match="lie in"):
        DualConcentration(
            agent_ids=np.array([1]), max_weights=np.array([0.0])
        )
    with pytest.raises(ValueError, match="lie in"):
        DualConcentration(
            agent_ids=np.array([1]), max_weights=np.array([-0.3])
        )


def test_concentration_payload_arrays_are_frozen() -> None:
    # both stored arrays are frozen so the broadcast payload cannot be
    # mutated via an alias
    conc = DualConcentration(
        agent_ids=np.array([1, 4]), max_weights=np.array([0.5, 0.75])
    )
    assert not conc.agent_ids.flags.writeable
    assert not conc.max_weights.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        conc.max_weights[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        conc.agent_ids[0] = 99
    # values unchanged after the attempted writes
    assert conc.agent_ids.tolist() == [1, 4]
    np.testing.assert_allclose(conc.max_weights, [0.5, 0.75])


def test_concentration_rejects_non_1d_or_float_agent_ids() -> None:
    with pytest.raises(ValueError, match="1-D integer array"):
        DualConcentration(
            agent_ids=np.array([[0, 1]]), max_weights=np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="1-D integer array"):
        DualConcentration(
            agent_ids=np.array([0.0, 1.0]), max_weights=np.array([1.0, 1.0])
        )


def test_concentration_rejects_negative_agent_ids() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        DualConcentration(
            agent_ids=np.array([-1, 2]), max_weights=np.array([1.0, 1.0])
        )


def test_concentration_rejects_nonparallel_max_weights() -> None:
    with pytest.raises(ValueError, match="parallel to agent_ids"):
        DualConcentration(
            agent_ids=np.array([0, 1]), max_weights=np.array([1.0])
        )


# --- DualInformed.select ------------------------------------------------------


def test_select_resolves_all_without_signal() -> None:
    sched = DualInformed()
    # Iteration 0, or missing dual / last_resolved: re-price everyone. The walk
    # consumes the mask as a boolean index (`local_ids[mask[local_ids]]`), so
    # this branch must return a (n,) bool array — an int mask would fancy-index.
    m = sched.select(0, 4)
    assert m.dtype == bool and m.shape == (4,) and m.all()
    conc = DualConcentration(np.array([0]), np.array([1.0]))
    m = sched.select(3, 4, dual=conc, last_resolved=None)
    assert m.dtype == bool and m.shape == (4,) and m.all()
    m = sched.select(3, 4, dual=None, last_resolved=np.zeros(4, int))
    assert m.dtype == bool and m.shape == (4,) and m.all()
    # At iteration 0, even a payload that would skip agent 1 at any later
    # iteration must still re-price everyone.
    conc_skip = DualConcentration(np.array([1]), np.array([0.99]))
    m = sched.select(0, 4, dual=conc_skip, last_resolved=np.zeros(4, np.int64))
    assert m.dtype == bool and m.shape == (4,) and m.all()


def test_select_skips_concentrated_but_recent() -> None:
    sched = DualInformed(concentration_threshold=0.9, max_staleness=5)
    conc = DualConcentration(np.array([1, 2]), np.array([0.95, 0.40]))
    last = np.array([3, 3, 3, 3], dtype=np.int64)  # all priced at iter 3
    mask = sched.select(4, 4, dual=conc, last_resolved=last)
    # Agent 1 is concentrated (0.95 >= 0.9) and recent -> skipped; agent 2
    # is below threshold -> re-priced; agents 0,3 absent from payload ->
    # re-priced (no evidence).
    assert mask.tolist() == [True, False, True, True]
    assert mask.dtype == bool and mask.shape == (4,)
    # Boundary: weight exactly at the threshold is skipped (>=, not >). Kept
    # recent (iter 1 - last 0 < period) so only the concentration clause decides.
    exact = DualInformed(concentration_threshold=1.0, max_staleness=5)
    conc_exact = DualConcentration(np.array([1]), np.array([1.0]))
    edge = exact.select(1, 4, dual=conc_exact, last_resolved=np.zeros(4, np.int64))
    assert edge.tolist() == [True, False, True, True]


def test_select_forced_revisit_overrides_concentration() -> None:
    sched = DualInformed(concentration_threshold=0.9, max_staleness=5)
    conc = DualConcentration(np.array([1]), np.array([0.99]))
    last = np.array([0, 0, 0, 0], dtype=np.int64)
    # Agent 1 is concentrated, but last priced 6 iters ago (>= period): the
    # staleness bound forces the revisit anyway.
    assert sched.select(6, 4, dual=conc, last_resolved=last).all()
    # gap == period exactly still forces the revisit (>=, not >)
    assert sched.select(5, 4, dual=conc, last_resolved=last).all()
    # one step below the bound (gap == period-1): concentration alone keeps
    # agent 1 skipped
    edge = sched.select(4, 4, dual=conc, last_resolved=last)
    assert edge.tolist() == [True, False, True, True]


def test_select_validates_construction() -> None:
    with pytest.raises(ValueError, match="concentration_threshold"):
        DualInformed(concentration_threshold=0.0)
    with pytest.raises(ValueError, match="max_staleness"):
        DualInformed(max_staleness=0)
    # max_weights never exceed 1.0, so a threshold above 1.0 would make the
    # skip filter a permanent no-op
    with pytest.raises(ValueError, match="lie in"):
        DualInformed(concentration_threshold=1.5)
    # the period is an iteration count; a float would break the staleness
    # arithmetic
    with pytest.raises(ValueError, match="max_staleness"):
        DualInformed(max_staleness=5.5)


def test_select_rejects_non_concentration_dual() -> None:
    sched = DualInformed()
    # non-zero iteration with dual/last_resolved present reaches the payload
    # type check
    with pytest.raises(ValueError, match="must be a DualConcentration payload"):
        sched.select(3, 4, dual=object(), last_resolved=np.zeros(4, np.int64))


def test_select_rejects_bad_last_resolved_shape_or_dtype() -> None:
    sched = DualInformed()
    conc = DualConcentration(np.array([0]), np.array([1.0]))
    with pytest.raises(ValueError, match=r"\(n_agents,\) integer array"):
        sched.select(3, 4, dual=conc, last_resolved=np.zeros(3, np.int64))
    with pytest.raises(ValueError, match=r"\(n_agents,\) integer array"):
        sched.select(3, 4, dual=conc, last_resolved=np.zeros(4, np.float64))


# --- the live pricing-budget gate ---------------------------------------------


def _arrays():
    return toy_family(N_OBS, N_ITEMS, DEFAULT_SEED)


def _run_dual_informed(arrays, backend: str = "gurobi"):
    return run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend=backend,
        schedule=DualInformed(concentration_threshold=0.9, max_staleness=PERIOD),
    )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_dual_informed_same_objective_fewer_prices(backend: str) -> None:
    arrays = _arrays()
    free = run_walk(arrays, toy_problem(arrays), NSlack, SerialTransport(), backend=backend)
    di = _run_dual_informed(arrays, backend=backend)
    assert di.converged and free.converged
    # Same optimal face: the objective is path-independent and must match.
    assert abs(di.objective - free.objective) < PARITY_BAND
    # The point of the schedule: a smaller pricing budget by skipping
    # dual-settled agents. (theta_hat is set-identified under the changed path,
    # so only the objective is asserted.)
    assert di.pricing_calls < free.pricing_calls


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="no gurobi")
def test_dual_informed_forced_revisit_bound_holds() -> None:
    # Two mechanisms bound the inter-price gap: the schedule's staleness clause
    # and run_walk's force_full re-certification, which alone caps the gap at 9
    # on this fixture. period=3 sits below that floor, so the staleness clause
    # is the binding one and the observed gap belongs to the schedule alone.
    period = 3
    arrays = _arrays()
    run = run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend="gurobi",
        schedule=DualInformed(concentration_threshold=0.9, max_staleness=period),
    )
    masks = np.array(run.schedule_masks)  # (iterations, n_agents)
    gaps = [
        int(np.diff(priced).max())
        for agent in range(masks.shape[1])
        if (priced := np.flatnonzero(masks[:, agent])).size >= 2
    ]
    # A skipped agent's largest gap is exactly `period`: no gap may exceed it,
    # and since concentration does cause skips here, at least one agent hits it.
    assert max(gaps) == period
    assert period in gaps


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="no gurobi")
def test_dual_informed_payload_is_support_sized(monkeypatch) -> None:
    # Check each recorded payload size against a recount of the dual support
    # taken from the raw per-cut duals the master hands out.
    recorded: list[dict] = []
    orig_dual_values = GurobiMaster.dual_values

    def recording_dual_values(self):
        duals = orig_dual_values(self)
        recorded.append(duals)
        return duals

    monkeypatch.setattr(GurobiMaster, "dual_values", recording_dual_values)
    di = _run_dual_informed(_arrays())
    supports = di.payload_supports

    def recount_support(duals: dict) -> int:
        return len(
            {int(agent) for (agent, _key), pi in duals.items() if pi > _SUPPORT_ATOL}
        )

    hand = [recount_support(recorded[i]) for i in range(len(supports))]
    assert list(supports) == hand
    # the first solve installs no cuts, so it carries no dual mass
    assert supports[0] == 0
    # O(support), not O(n_agents): smaller than a full mask on most iterations.
    # Skip the seed solve (supports[0] == 0 by construction) so the minimum is
    # taken over real post-seed payloads.
    assert min(supports[1:]) < N_OBS
    below = [s for s in supports if s < N_OBS]
    assert len(below) > len(supports) // 2


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_dual_informed_rank_invariant(backend: str) -> None:
    arrays = _arrays()
    serial = run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend=backend,
        schedule=DualInformed(max_staleness=PERIOD),
    )
    results = LocalCluster(2).run(
        lambda transport: run_walk(
            arrays,
            toy_problem(arrays),
            NSlack,
            transport,
            backend=backend,
            schedule=DualInformed(max_staleness=PERIOD),
        )
    )
    assert len(results) == 2
    # The mask is computed rank-locally from the bcast payload; an identical
    # bitwise answer on every interleaved shard proves the derivation agrees
    # across ranks without broadcasting the mask.
    for outcome in results:
        assert outcome.result.theta_hat.tobytes() == serial.result.theta_hat.tobytes()
        assert outcome.objective == serial.objective
        assert outcome.pricing_calls == serial.pricing_calls
