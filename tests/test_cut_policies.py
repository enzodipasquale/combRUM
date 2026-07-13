"""Built-in cut policies: unit semantics, keep-set snapshots, live fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from _family_oracles import toy_problem
from _walk import run_walk
from combrum.cut_policies import (
    AddAll,
    Compose,
    MostViolated,
    PurgeInactive,
    SlackStrip,
    ViolationThreshold,
)
from _support.families import (
    DEFAULT_SEED,
    TOY_DEFAULT_N_ITEMS,
    TOY_DEFAULT_N_OBS,
    toy_family,
)
from _support.synthetic import (
    STRIP_HARD_THRESHOLD,
    STRIP_PERCENTILE,
    stripping_snapshot_fixture,
)
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.transport import CutRow, SerialTransport

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


def _row(agent_id: int, key: bytes) -> CutRow:
    return CutRow(
        rep_id=0,
        agent_id=agent_id,
        phi=np.array([1.0]),
        epsilon=0.0,
        bundle_key=key,
    )


# --- MostViolated -------------------------------------------------------------
# Selection basics for AddAll and MostViolated — k-largest, nonpositive
# filtering, tie order — live in test_policy_contract.py; this file covers
# the knobs and edge cases.


def test_most_violated_fraction_rounds_and_keeps_at_least_one() -> None:
    policy = MostViolated(fraction=0.5)
    rows = [_row(i, bytes([i])) for i in range(4)]
    # 0.5 * 4 = 2 kept (the two largest: indices 3 and 1), input order.
    got = policy.admit(rows, np.array([1.0, 3.0, 0.5, 9.0]), 0)
    assert got == (rows[1], rows[3])

    # Single positive candidate: int(0.5*1)==0, so the max(1, ...) clamp must
    # still admit that one row (idx 2) rather than dropping everything.
    single = policy.admit(rows, np.array([-1.0, 0.0, 4.0, -2.0]), 0)
    assert single == (rows[2],)

    # fraction=0.75 also floors to zero with one candidate: clamped back to 1.
    floored = MostViolated(fraction=0.75).admit(
        rows, np.array([0.0, 7.0, -3.0, -1.0]), 0
    )
    assert floored == (rows[1],)

    # 0.5 * 3 positive candidates = 1.5: `int()` floors to 1, so only the
    # largest (idx 0) is admitted; round or ceil would take 2.
    three = [_row(i, bytes([10 + i])) for i in range(3)]
    floor_one = MostViolated(fraction=0.5).admit(
        three, np.array([9.0, 3.0, 7.0]), 0
    )
    assert floor_one == (three[0],)


def test_most_violated_breaks_cutoff_ties_toward_earlier() -> None:
    policy = MostViolated(k=1)
    a, b = _row(1, b"a"), _row(2, b"b")
    assert policy.admit([a, b], np.array([4.0, 4.0]), 0) == (a,)


def test_most_violated_validates_exactly_one_knob() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MostViolated()
    with pytest.raises(ValueError, match="exactly one"):
        MostViolated(k=2, fraction=0.5)
    with pytest.raises(ValueError, match="k must be"):
        MostViolated(k=0)
    with pytest.raises(ValueError, match="fraction must"):
        MostViolated(fraction=0.0)
    with pytest.raises(ValueError, match="fraction must"):
        MostViolated(fraction=1.5)

    # fraction=1.0 is valid: admit every positive candidate, in input order.
    rows = [_row(i, bytes([i])) for i in range(4)]
    admitted = MostViolated(fraction=1.0).admit(
        rows, np.array([1.0, -1.0, 2.0, 3.0]), 0
    )
    assert admitted == (rows[0], rows[2], rows[3])


def test_admission_policies_require_violations_parallel_to_candidates() -> None:
    rows = [_row(1, b"a"), _row(2, b"b"), _row(3, b"c")]

    with pytest.raises(ValueError, match="parallel to candidates"):
        MostViolated(k=1).admit(rows, np.array([1.0]), 0)

    with pytest.raises(ValueError, match="parallel to candidates"):
        ViolationThreshold(epsilon=0.1).admit(rows, np.ones((3, 1)), 0)


def test_most_violated_purge_is_noop() -> None:
    policy = MostViolated(k=2)
    rows = [_row(1, b"a"), _row(2, b"b")]
    # Admission-only policy: purge retires nothing whatever the dual/slack.
    assert policy.purge(rows, dual=None, slack=None, iteration=0) == ()


# --- ViolationThreshold -----------------------------------------------------------


def test_slack_threshold_purge_is_noop() -> None:
    policy = ViolationThreshold(epsilon=1.0)
    rows = [_row(1, b"a"), _row(2, b"b")]
    assert policy.purge(rows, dual=None, slack=None, iteration=0) == ()


def test_slack_threshold_admits_above_floor_only() -> None:
    policy = ViolationThreshold(epsilon=1.0)
    a, b, c = _row(1, b"a"), _row(2, b"b"), _row(3, b"c")
    assert policy.admit([a, b, c], np.array([0.5, 1.0, 2.0]), 0) == (c,)


def test_slack_threshold_validates_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be"):
        ViolationThreshold(epsilon=-1.0)


# --- Compose ------------------------------------------------------------------


def _expected_compose_admit(
    rows: Sequence[CutRow], viol: Sequence[float], *, k: int, epsilon: float
) -> tuple[CutRow, ...]:
    """MostViolated(k) then ViolationThreshold(epsilon), in plain Python."""
    # top k positive rows by violation, ties toward earlier, back in input order
    positive = [i for i, v in enumerate(viol) if v > 0.0]
    top = sorted(positive, key=lambda i: (-viol[i], i))[:k]
    stage1 = sorted(top)
    # the threshold stage filters on each row's original violation
    stage2 = [i for i in stage1 if viol[i] > epsilon]
    return tuple(rows[i] for i in stage2)


def test_compose_admit_chains_and_keeps_violations_parallel() -> None:
    # Top-3 is (1, 2, 3); idx 3 (viol 1.5) fails the 2.0 floor, so both stages
    # do work and the threshold stage must see violations realigned to the
    # top-3 rows.
    k, epsilon = 3, 2.0
    policy = Compose(
        admit_chain=[MostViolated(k=k), ViolationThreshold(epsilon=epsilon)],
        purge_chain=[],
    )
    rows = [_row(i, bytes([i])) for i in range(4)]
    viol = [0.5, 3.0, 9.0, 1.5]

    got = policy.admit(rows, np.array(viol), 0)

    expected = _expected_compose_admit(rows, viol, k=k, epsilon=epsilon)
    assert got == expected
    assert got == (rows[1], rows[2])

    # stage 1 alone keeps one more row, so the floor stage is not a no-op here
    stage1_only = MostViolated(k=k).admit(rows, np.array(viol), 0)
    assert stage1_only == (rows[1], rows[2], rows[3])
    assert len(got) < len(stage1_only)


def _expected_purge_union(
    installed: Sequence[CutRow], stage_votes: Sequence[Sequence[CutRow]]
) -> tuple[CutRow, ...]:
    """Union of per-stage retirement votes, in installed order, deduped."""
    voted: set[tuple[int, bytes]] = set()
    for votes in stage_votes:
        for row in votes:
            voted.add((row.agent_id, row.bundle_key))
    out: list[CutRow] = []
    emitted: set[tuple[int, bytes]] = set()
    for row in installed:
        key = (row.agent_id, row.bundle_key)
        if key in voted and key not in emitted:
            out.append(row)
            emitted.add(key)
    return tuple(out)


def test_compose_purge_unions_votes() -> None:
    # r2 is voted by both stages (loose AND stale) and is SlackStrip's only
    # vote, so the union must come back in installed order (r0, r2) with r2
    # appearing once -- not a per-stage concat (r2, r0, r2) or set order.
    r0 = _row(1, b"a")
    r1 = _row(2, b"b")
    r2 = _row(3, b"c")
    installed = [r0, r1, r2]

    slack_stage = SlackStrip(percentile=50.0, max_live_cuts=10.0)
    purge_stage = PurgeInactive(max_age=1)
    policy = Compose(admit_chain=[], purge_chain=[slack_stage, purge_stage])

    # r0, r2 stale (dual within noise); r1 active. r2 loosest by slack.
    dual = {(1, b"a"): 0.0, (2, b"b"): 0.5, (3, b"c"): 0.0}
    slack = {(1, b"a"): 0.0, (2, b"b"): 0.0, (3, b"c"): 9.0}

    retired = policy.purge(installed, dual=dual, slack=slack, iteration=1)

    # compare against the union of each stage's own votes
    ss_votes = SlackStrip(percentile=50.0, max_live_cuts=10.0).purge(
        installed, dual=None, slack=slack, iteration=1
    )
    pi_votes = PurgeInactive(max_age=1).purge(
        installed, dual=dual, slack=None, iteration=1
    )
    expected = _expected_purge_union(installed, [ss_votes, pi_votes])
    assert retired == expected
    assert retired == (r0, r2)

    assert ss_votes == (r2,)
    assert set(pi_votes) == {r0, r2}


def test_compose_validates_stage_types() -> None:
    with pytest.raises(ValueError, match="admit_chain"):
        Compose(admit_chain=[object()], purge_chain=[])
    with pytest.raises(ValueError, match="purge_chain"):
        Compose(admit_chain=[], purge_chain=[object()])


# --- PurgeInactive ------------------------------------------------------------
# Streak aging, resets, the dual-noise tolerance, and counter pruning for
# cuts that leave the master are covered in test_policy_contract.py.


def test_purge_inactive_degrades_without_dual() -> None:
    # A dual=None call must not advance the streak: it still takes two
    # explicit zero readings to retire at max_age=2.
    policy = PurgeInactive(max_age=2)
    row = _row(1, b"a")
    assert policy.purge([row], dual=None, slack=None, iteration=0) == ()
    # first zero reading: streak 1, no retire
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=1) == ()
    # second zero reading: streak reaches max_age=2
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=2) == (
        row,
    )


def test_purge_inactive_missing_key_in_present_dual_holds_counter() -> None:
    # A cut absent from a *present* (non-None) dual dict has no reading, so
    # its streak holds rather than advancing: missing != zero.
    policy = PurgeInactive(max_age=2)
    held = _row(9, b"held")
    zeroed = _row(9, b"zeroed")
    dual = {(9, b"zeroed"): 0.0}  # held's key omitted; dict is non-empty

    assert policy.purge([held, zeroed], dual, slack=None, iteration=0) == ()
    retired = policy.purge([held, zeroed], dual, slack=None, iteration=1)
    assert retired == (zeroed,)  # held holds its counter; only zeroed ages out


def test_purge_inactive_validates_max_age() -> None:
    with pytest.raises(ValueError, match="max_age"):
        PurgeInactive(max_age=0)


# --- SlackStrip ---------------------------------------------------------------
# Percentile semantics and the no-signal degrade (slack=None / unmatched /
# empty) live in test_policy_contract.py; this file covers the constructor,
# the hard cap, and the keep-set snapshot.


def test_slack_strip_validates_constructor() -> None:
    # percentile must lie in (0, 100]: excluded low end, above the ceiling, and
    # NaN all reject.
    for bad in (0.0, 100.1, float("nan")):
        with pytest.raises(ValueError, match=r"percentile must lie in \(0, 100\]"):
            SlackStrip(percentile=bad)

    # max_live_cuts is a max-live constraint count, so non-integral values
    # reject and so do integral values below 1.
    with pytest.raises(ValueError, match="integer-valued"):
        SlackStrip(max_live_cuts=2.5)
    with pytest.raises(ValueError, match=r">= 1"):
        SlackStrip(max_live_cuts=0)

    # inclusive boundaries construct: percentile=100.0 and max_live_cuts=1;
    # inf disables the cap.
    assert SlackStrip(percentile=100.0) is not None
    assert SlackStrip(max_live_cuts=1) is not None
    assert SlackStrip(max_live_cuts=float("inf")) is not None


def test_slack_strip_default_percentile_is_max_live_cuts_only() -> None:
    rows = [_row(1, bytes([97 + i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {
        keys[0]: 4.0,
        keys[1]: 0.5,
        keys[2]: 7.0,
        keys[3]: 0.25,
        keys[4]: 2.0,
    }
    policy = SlackStrip(max_live_cuts=2.0)

    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)

    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    kept = {
        (row.agent_id, row.bundle_key)
        for row in rows
        if (row.agent_id, row.bundle_key) not in retired_keys
    }
    assert kept == {keys[1], keys[3]}


def test_slack_strip_default_percentile_keeps_all_with_cap_inactive() -> None:
    # The default percentile of 100.0 puts the cutoff at max(slacks), so
    # `slack <= cutoff` keeps every row. Any default below 100 would drop the
    # interpolated cutoff under the loosest slack (100.0) and strip it.
    rows = [_row(i, bytes([i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = dict(zip(keys, [1.0, 2.0, 3.0, 4.0, 100.0]))

    policy = SlackStrip(max_live_cuts=float("inf"))
    assert policy.purge(rows, dual=None, slack=slack, iteration=0) == ()


def test_slack_strip_master_size_guard_reaches_composed_policies() -> None:
    SlackStrip(max_live_cuts=8).validate_master_size(
        n_parameters=3, n_agents=5
    )
    SlackStrip(max_live_cuts=float("inf")).validate_master_size(
        n_parameters=3, n_agents=5
    )

    with pytest.raises(ValueError, match=r"K \+ n_agents"):
        SlackStrip(max_live_cuts=7).validate_master_size(
            n_parameters=3, n_agents=5
        )

    policy = Compose(
        admit_chain=[AddAll()],
        purge_chain=[SlackStrip(max_live_cuts=7)],
    )
    with pytest.raises(ValueError, match="max_live_cuts=7"):
        policy.validate_master_size(n_parameters=3, n_agents=5)


def _run_strip(
    slacks: Sequence[float], *, percentile: float, max_live_cuts: float
) -> tuple[bool, ...]:
    """Run SlackStrip.purge over a slack vector; return the per-row keep mask."""
    rows = [_row(i, bytes([i])) for i in range(len(slacks))]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {keys[i]: float(slacks[i]) for i in range(len(rows))}
    policy = SlackStrip(percentile=percentile, max_live_cuts=max_live_cuts)
    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    return tuple(k not in retired_keys for k in keys)


def test_slack_strip_keep_set_matches_hand_derived_percentile() -> None:
    # linear percentile: cutoff sits at position (n-1)*p/100 in the sorted
    # slacks; keep iff slack <= cutoff.

    # sorted slacks [0,1,2,3,100], n=5, pos=(5-1)*0.95=3.8 ->
    # cutoff = 3 + 0.8*(100-3) = 80.6. Only 100.0 exceeds it.
    keep = _run_strip(
        [0.0, 1.0, 2.0, 3.0, 100.0], percentile=95.0, max_live_cuts=float("inf")
    )
    assert keep == (True, True, True, True, False)

    # boundary tie: sorted [0,1,2,3,3], pos=3.8 -> cutoff = 3 + 0.8*(3-3) = 3.0.
    # Rows sitting exactly at 3.0 are kept because the rule is `slack <= cutoff`.
    tie_keep = _run_strip(
        [0.0, 1.0, 2.0, 3.0, 3.0], percentile=95.0, max_live_cuts=float("inf")
    )
    assert tie_keep == (True, True, True, True, True)


def test_slack_strip_hard_cap_keeps_the_smallest_slacks() -> None:
    # When the percentile keep would retain more than max_live_cuts rows,
    # only the max_live_cuts smallest (most-binding) slacks stay. p=95 keeps
    # all six; cap=3 keeps values 1,2,3 at indices 1,3,5.
    keep = _run_strip(
        [5.0, 1.0, 4.0, 2.0, 6.0, 3.0], percentile=95.0, max_live_cuts=3
    )
    assert keep == (False, True, False, True, False, True)


def test_slack_strip_hard_cap_ties_are_stable_in_installed_order() -> None:
    # Equal-looseness ties under the cap break in installed-row order via the
    # stable sort; small tie groups come back in input order even under numpy's
    # default introselect, so use a 20-wide tie. p95 strips only the tail
    # (rows 21, 22), the cap=3 keeps row 0 plus the two earliest tied rows.
    slacks = [1.0] + [5.0] * 20 + [9.0, 8.0]
    keep = _run_strip(slacks, percentile=95.0, max_live_cuts=3)
    expected = tuple(i in (0, 1, 2) for i in range(len(slacks)))
    assert keep == expected


def test_slack_strip_ignores_cuts_without_a_reading() -> None:
    # A cut absent from `slack` is neither strippable nor part of the
    # percentile population. Slacks [1,2,3,10] at p=70 strip only idx 3; a
    # phantom 0.0 reading would drag the cutoff down and also strip idx 2.
    rows = [_row(i, bytes([i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 3.0, keys[3]: 10.0}
    # keys[4] has no slack reading.
    policy = SlackStrip(percentile=70.0, max_live_cuts=float("inf"))

    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    assert retired_keys == {keys[3]}
    assert keys[4] not in retired_keys

    # same strip set as running without the extra row: the missing reading
    # contributes nothing to the population, so the cutoff is unchanged
    retired_no_extra = policy.purge(rows[:4], dual=None, slack=slack, iteration=0)
    assert {(row.agent_id, row.bundle_key) for row in retired_no_extra} == {
        keys[3]
    }


def test_slack_strip_reproduces_keep_set_on_snapshot() -> None:
    # Frozen 200-cut snapshot -- the only test that drives the policy at a
    # population above the hard cap.
    fixture = stripping_snapshot_fixture(n_cuts=200, seed=7)
    agent_ids = np.asarray(fixture["agent_ids"])
    bundle_keys = np.asarray(fixture["bundle_keys"])
    slacks = np.asarray(fixture["slacks"], dtype=np.float64)
    # the fixture recomputes the keep mask from the stated rule, not by
    # rerunning the policy
    expected_keep = np.asarray(fixture["expected_keep"], dtype=bool)

    rows = [
        _row(int(agent_ids[i]), bytes(bundle_keys[i]))
        for i in range(agent_ids.size)
    ]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    assert len(set(keys)) == len(keys), "snapshot cut identities must be unique"
    slack = {keys[i]: float(slacks[i]) for i in range(len(rows))}

    policy = SlackStrip(
        percentile=STRIP_PERCENTILE, max_live_cuts=STRIP_HARD_THRESHOLD
    )
    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    kept_keys = [k for k in keys if k not in retired_keys]

    keep_mask = np.array([k not in retired_keys for k in keys])
    np.testing.assert_array_equal(keep_mask, expected_keep)

    # the percentile leg keeps more than the cap, so the hard cap decides
    # the split
    assert bool(fixture["hard_cap_active"]) is True
    assert int(keep_mask.sum()) == STRIP_HARD_THRESHOLD == 150
    assert len(retired) == int((~expected_keep).sum()) == 50

    # the rule keeps the smallest slacks, so no kept slack exceeds any
    # stripped one
    assert retired_keys, "snapshot must strip at least one cut"
    assert kept_keys, "snapshot must keep at least one cut"
    max_kept_slack = max(slack[k] for k in kept_keys)
    min_stripped_slack = min(slack[k] for k in retired_keys)
    assert max_kept_slack <= min_stripped_slack


# --- live fits: built-in policies through NSlack ------------------------------


def _toy_arrays() -> Mapping[str, np.ndarray]:
    return toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)


def _fit(backend: str, cut_policy) -> object:
    arrays = _toy_arrays()
    from combrum.formulations import NSlack

    return run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend=backend,
        cut_policy=cut_policy,
    )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_addall_is_transparent_vs_policy_free(backend: str) -> None:
    free = _fit(backend, None)
    addall = _fit(backend, AddAll())
    assert addall.converged and free.converged
    # AddAll admits everything and retires nothing, so the fit matches the
    # policy-free run exactly.
    np.testing.assert_array_equal(addall.result.theta_hat, free.result.theta_hat)
    assert addall.result.n_active_cuts == free.result.n_active_cuts
    assert addall.cuts_admitted == free.cuts_admitted


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_most_violated_throttles_admission_same_objective(backend: str) -> None:
    free = _fit(backend, None)
    # k=1 admits at most one cut per iteration: a different admission path
    # (strictly more iterations) reaching the same optimal face. The objective
    # is path-independent, so it must match; theta_hat is only set-identified,
    # so a degenerate face may publish a different argmin vertex.
    throttled = _fit(backend, MostViolated(k=1))
    assert throttled.converged

    # the free run admits many cuts per iteration on this fixture, so it is a
    # real non-throttled baseline
    assert free.cuts_admitted > free.iterations

    # k=1 admits at most one cut per iteration, so admissions <= iterations
    assert throttled.cuts_admitted <= throttled.iterations

    # fewer cuts per iteration means strictly more iterations to the same face
    assert throttled.iterations > free.iterations

    assert abs(throttled.objective - free.objective) < PARITY_BAND
