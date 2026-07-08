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
    SlackThreshold,
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


# --- AddAll -------------------------------------------------------------------


def test_addall_admits_all_retires_none() -> None:
    policy = AddAll()
    rows = [_row(1, b"a"), _row(2, b"b")]
    assert policy.admit(rows, np.array([3.0, 1.0]), 0) == tuple(rows)
    assert policy.purge(rows, dual=None, slack=None, iteration=0) == ()


# --- MostViolated -------------------------------------------------------------


def test_most_violated_keeps_k_largest_in_input_order() -> None:
    policy = MostViolated(k=2)
    a, b, c = _row(1, b"a"), _row(2, b"b"), _row(3, b"c")
    # b and c are the two most violated; result keeps input order (b, c).
    assert policy.admit([a, b, c], np.array([0.2, 5.0, 1.0]), 0) == (b, c)


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


def test_most_violated_ignores_nonpositive_violations() -> None:
    policy = MostViolated(k=3)
    a, b, c = _row(1, b"a"), _row(2, b"b"), _row(3, b"c")
    # Only one candidate has positive violation, so only it is admitted
    # even though k=3 would allow more.
    assert policy.admit([a, b, c], np.array([0.0, -1.0, 2.0]), 0) == (c,)


def test_most_violated_breaks_cutoff_ties_toward_earlier() -> None:
    policy = MostViolated(k=1)
    a, b = _row(1, b"a"), _row(2, b"b")
    # Equal violations: the earlier candidate wins the single slot.
    assert policy.admit([a, b], np.array([4.0, 4.0]), 0) == (a,)


def test_most_violated_ties_in_large_group_keep_earliest() -> None:
    # Cutoff ties rely on the stable sort. Small tie groups come back in input
    # order even under numpy's default introselect, so use a 20-wide tie group:
    # only kind="stable" is guaranteed to keep the earliest tied indices.
    rows = [_row(i, bytes([i])) for i in range(22)]
    viol = np.array([5.0] * 20 + [1.0, 0.5])
    assert MostViolated(k=2).admit(rows, viol, 0) == (rows[0], rows[1])
    assert MostViolated(k=3).admit(rows, viol, 0) == (rows[0], rows[1], rows[2])


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
        SlackThreshold(epsilon=0.1).admit(rows, np.ones((3, 1)), 0)


def test_most_violated_purge_is_noop() -> None:
    policy = MostViolated(k=2)
    rows = [_row(1, b"a"), _row(2, b"b")]
    # Admission-only policy: purge retires nothing whatever the dual/slack.
    assert policy.purge(rows, dual=None, slack=None, iteration=0) == ()


# --- SlackThreshold -----------------------------------------------------------


def test_slack_threshold_purge_is_noop() -> None:
    policy = SlackThreshold(epsilon=1.0)
    rows = [_row(1, b"a"), _row(2, b"b")]
    assert policy.purge(rows, dual=None, slack=None, iteration=0) == ()


def test_slack_threshold_admits_above_floor_only() -> None:
    policy = SlackThreshold(epsilon=1.0)
    a, b, c = _row(1, b"a"), _row(2, b"b"), _row(3, b"c")
    assert policy.admit([a, b, c], np.array([0.5, 1.0, 2.0]), 0) == (c,)


def test_slack_threshold_validates_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be"):
        SlackThreshold(epsilon=-1.0)


# --- Compose ------------------------------------------------------------------


def _compose_admit_oracle(
    rows: Sequence[CutRow], viol: Sequence[float], *, k: int, epsilon: float
) -> tuple[CutRow, ...]:
    """MostViolated(k) then SlackThreshold(epsilon), in plain Python."""
    # top k positive rows by violation, ties toward earlier, back in input order
    positive = [i for i, v in enumerate(viol) if v > 0.0]
    top = sorted(positive, key=lambda i: (-viol[i], i))[:k]
    stage1 = sorted(top)
    # survivors are filtered on their own original violation
    stage2 = [i for i in stage1 if viol[i] > epsilon]
    return tuple(rows[i] for i in stage2)


def test_compose_admit_chains_and_keeps_violations_parallel() -> None:
    # Top-3 is (1, 2, 3); idx 3 (viol 1.5) fails the 2.0 floor, so both stages
    # do work and the threshold stage must see violations realigned to the
    # top-3 rows.
    k, epsilon = 3, 2.0
    policy = Compose(
        admit_chain=[MostViolated(k=k), SlackThreshold(epsilon=epsilon)],
        purge_chain=[],
    )
    rows = [_row(i, bytes([i])) for i in range(4)]
    viol = [0.5, 3.0, 9.0, 1.5]

    got = policy.admit(rows, np.array(viol), 0)

    # Full-output pin against the structurally-distinct oracle: the composed
    # tuple must equal the independent two-stage computation exactly (identity,
    # order, membership). This kills a whole class of mutations at once --
    # dropping/no-oping either stage, reordering the chain, breaking the
    # violation realignment, or losing input order.
    expected = _compose_admit_oracle(rows, viol, k=k, epsilon=epsilon)
    assert got == expected
    assert got == (rows[1], rows[2])

    # Sanity that this fixture actually exercises multi-stage thinning: stage 1
    # alone keeps one more row than the composed result, so the floor stage is
    # not a no-op here.
    stage1_only = MostViolated(k=k).admit(rows, np.array(viol), 0)
    assert stage1_only == (rows[1], rows[2], rows[3])
    assert len(got) < len(stage1_only)


def _union_votes_oracle(
    installed: Sequence[CutRow], stage_votes: Sequence[Sequence[CutRow]]
) -> tuple[CutRow, ...]:
    """Independent union of per-stage retirement votes, installed order, deduped.

    Structurally distinct from Compose.purge (which accumulates keys into a set
    then filters installed): here we walk installed once and keep a row iff any
    stage voted its identity, tracking seen keys explicitly so a doubly-voted
    row is emitted exactly once. Used to pin the composed tuple wholesale.
    """
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
    # Cross both boundaries the union guarantee protects in ONE call:
    #  - stage-vote order != installed order (SlackStrip votes only the LAST
    #    installed row r2, so a naive per-stage concat would surface r2 first);
    #  - a row voted by BOTH stages (r2 is stripped for looseness AND stale),
    #    so dedup must collapse it to a single appearance.
    # A concat regression `out.extend(stage.purge(...))` yields (r2, r0, r2) --
    # wrong order and a duplicate. A set-order return (dropping the installed
    # walk) yields (r2, r0). The correct union is (r0, r2), each once.
    r0 = _row(1, b"a")
    r1 = _row(2, b"b")
    r2 = _row(3, b"c")
    installed = [r0, r1, r2]

    slack_stage = SlackStrip(percentile=50.0, hard_threshold=10.0)
    purge_stage = PurgeInactive(max_age=1)
    policy = Compose(admit_chain=[], purge_chain=[slack_stage, purge_stage])

    # r0, r2 stale (dual within noise); r1 active. r2 loosest by slack.
    dual = {(1, b"a"): 0.0, (2, b"b"): 0.5, (3, b"c"): 0.0}
    slack = {(1, b"a"): 0.0, (2, b"b"): 0.0, (3, b"c"): 9.0}

    retired = policy.purge(installed, dual=dual, slack=slack, iteration=1)

    # Full-output pin against a structurally-distinct union oracle fed the two
    # stages' own votes: identity, installed order, and single-appearance all at
    # once. This kills the concat regression and its order/dedup siblings together.
    ss_votes = SlackStrip(percentile=50.0, hard_threshold=10.0).purge(
        installed, dual=None, slack=slack, iteration=1
    )
    pi_votes = PurgeInactive(max_age=1).purge(
        installed, dual=dual, slack=None, iteration=1
    )
    expected = _union_votes_oracle(installed, [ss_votes, pi_votes])
    assert retired == expected
    assert retired == (r0, r2)

    # The fixture genuinely exercises both boundaries: SlackStrip's lone vote is
    # the last installed row (so concat-order != installed-order), and r2 is
    # voted by both stages (so dedup is load-bearing).
    assert ss_votes == (r2,)
    assert set(pi_votes) == {r0, r2}


def test_compose_validates_stage_types() -> None:
    with pytest.raises(ValueError, match="admit_chain"):
        Compose(admit_chain=[object()], purge_chain=[])
    with pytest.raises(ValueError, match="purge_chain"):
        Compose(admit_chain=[], purge_chain=[object()])


# --- PurgeInactive ------------------------------------------------------------


def test_purge_inactive_ages_then_retires() -> None:
    policy = PurgeInactive(max_age=2)
    row = _row(1, b"a")
    dual = {(1, b"a"): 0.0}
    assert policy.purge([row], dual, slack=None, iteration=0) == ()  # age 1
    assert policy.purge([row], dual, slack=None, iteration=1) == (row,)  # age 2


def test_purge_inactive_resets_on_nonzero_dual() -> None:
    policy = PurgeInactive(max_age=2)
    row = _row(1, b"a")
    policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=0)  # age 1
    policy.purge([row], {(1, b"a"): 0.9}, slack=None, iteration=1)  # reset
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=2) == ()


def test_purge_inactive_degrades_without_dual() -> None:
    # A dual=None call must move no counter, not merely return (). With
    # max_age=2 the streak position after the None call is observable: if the
    # None call left the streak at 0 (correct), it takes TWO explicit zero
    # readings to retire; if the None call silently advanced the streak, the
    # FIRST zero reading would already retire. Pin the full retire-sequence
    # (nothing, nothing, then the row) so any counter movement on the None call
    # flips the middle assertion.
    policy = PurgeInactive(max_age=2)
    row = _row(1, b"a")
    assert policy.purge([row], dual=None, slack=None, iteration=0) == ()
    # First zero reading: streak advances to 1 (< max_age), still no retire.
    # This is the leg the regression trips -- a streak-advancing None call would
    # make this reading the second in the streak and retire here.
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=1) == ()
    # Second zero reading: streak reaches max_age=2, now the row retires.
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=2) == (
        row,
    )


def test_purge_inactive_missing_key_in_present_dual_holds_counter() -> None:
    # A cut absent from a *present* (non-None) dual dict has no reading, so its
    # streak must hold rather than advance: missing != zero. Treating a missing
    # key as 0.0 would advance the streak and retire it. Here rows share the
    # same call: `held` has no reading (its key omitted) and must never retire
    # even after max_age calls, while `zeroed` has an explicit zero reading and
    # does retire at max_age -- the two legs share the identical call, so only
    # the missing-vs-zero distinction can separate them.
    policy = PurgeInactive(max_age=2)
    held = _row(9, b"held")
    zeroed = _row(9, b"zeroed")
    dual = {(9, b"zeroed"): 0.0}  # held's key omitted; dict is non-empty

    assert policy.purge([held, zeroed], dual, slack=None, iteration=0) == ()
    retired = policy.purge([held, zeroed], dual, slack=None, iteration=1)
    assert retired == (zeroed,)  # held holds its counter; only zeroed ages out


def test_purge_inactive_prunes_vanished_cuts() -> None:
    policy = PurgeInactive(max_age=2)
    row = _row(1, b"a")
    policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=0)  # age 1
    # The cut vanishes for a call, then returns: its streak restarts, so it
    # is not retired on its first re-appearance.
    policy.purge([], dual={}, slack=None, iteration=1)
    assert policy.purge([row], {(1, b"a"): 0.0}, slack=None, iteration=2) == ()


def test_purge_inactive_dual_noise_band_is_atol_wide() -> None:
    # The zero-dual test is `abs(pi) <= _DUAL_ATOL` (1e-10), so a dual reading
    # inside the noise band counts as inactive while one an order of magnitude
    # above it counts as active. Both probe values straddle the hand-derived
    # 1e-10 boundary; neither is exactly 0.0. A `pi == 0.0` test would miss the
    # 1e-11 case, and a wider atol would wrongly retire the 1e-8 case.
    row = _row(1, b"a")
    key = (1, b"a")

    noise = PurgeInactive(max_age=1)
    assert noise.purge([row], {key: 1e-11}, slack=None, iteration=0) == (row,)

    # 1e-8 is two orders above the atol: active, so the streak never advances
    # across repeated calls and the cut is never retired.
    support = PurgeInactive(max_age=1)
    assert support.purge([row], {key: 1e-8}, slack=None, iteration=0) == ()
    assert support.purge([row], {key: 1e-8}, slack=None, iteration=1) == ()

    # Tight bracket on the 1e-10 boundary: probes an order of magnitude away
    # leave atol inflatable ~100x. 0.9e-10 sits just below (inactive, retires)
    # and 1.1e-10 just above (active, never retires), so any shift of the atol
    # off 1e-10 in either direction flips one of these.
    below = PurgeInactive(max_age=1)
    assert below.purge([row], {key: 0.9e-10}, slack=None, iteration=0) == (row,)

    above = PurgeInactive(max_age=1)
    assert above.purge([row], {key: 1.1e-10}, slack=None, iteration=0) == ()
    assert above.purge([row], {key: 1.1e-10}, slack=None, iteration=1) == ()

    # A support reading mid-streak resets the counter, so a following zero
    # cannot retire at max_age=2 (streak is back to 1, not 2).
    reset = PurgeInactive(max_age=2)
    reset.purge([row], {key: 0.0}, slack=None, iteration=0)  # streak 1
    reset.purge([row], {key: 1e-8}, slack=None, iteration=1)  # active -> reset
    assert reset.purge([row], {key: 0.0}, slack=None, iteration=2) == ()


def test_purge_inactive_validates_max_age() -> None:
    with pytest.raises(ValueError, match="max_age"):
        PurgeInactive(max_age=0)


# --- SlackStrip: keep-set snapshot -------------------------------------------


def test_slack_strip_degrades_without_slack() -> None:
    policy = SlackStrip(percentile=95.0, hard_threshold=float("inf"))
    assert policy.purge([_row(1, b"a")], dual=None, slack=None, iteration=0) == ()


def test_slack_strip_validates_constructor() -> None:
    # SlackStrip's own reject boundaries, mirroring the per-policy validation
    # tests for MostViolated/SlackThreshold/PurgeInactive. Nothing else in the
    # suite ever constructs an invalid SlackStrip, so these guards would
    # otherwise be a suite-wide escape.

    # percentile must lie in (0, 100]: excluded low end, above the ceiling, and
    # NaN all reject.
    for bad in (0.0, 100.1, float("nan")):
        with pytest.raises(ValueError, match=r"percentile must lie in \(0, 100\]"):
            SlackStrip(percentile=bad)

    # hard_threshold is a max-live constraint COUNT: a non-integer value rejects
    # as non-integral, and an integral value below 1 rejects on the floor.
    with pytest.raises(ValueError, match="integer-valued"):
        SlackStrip(hard_threshold=2.5)
    with pytest.raises(ValueError, match=r">= 1"):
        SlackStrip(hard_threshold=0)

    # The inclusive-valid boundaries must NOT reject: percentile=100.0 (ceiling)
    # and hard_threshold=1 (floor) construct, and inf disables the cap.
    assert SlackStrip(percentile=100.0) is not None
    assert SlackStrip(hard_threshold=1) is not None
    assert SlackStrip(hard_threshold=float("inf")) is not None


def test_slack_strip_default_percentile_is_hard_threshold_only() -> None:
    rows = [_row(1, bytes([97 + i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {
        keys[0]: 4.0,
        keys[1]: 0.5,
        keys[2]: 7.0,
        keys[3]: 0.25,
        keys[4]: 2.0,
    }
    policy = SlackStrip(hard_threshold=2.0)

    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)

    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    kept = {
        (row.agent_id, row.bundle_key)
        for row in rows
        if (row.agent_id, row.bundle_key) not in retired_keys
    }
    assert kept == {keys[1], keys[3]}


def test_slack_strip_default_percentile_keeps_all_with_cap_inactive() -> None:
    # Isolate the default-percentile leg (hard cap disabled): the constructor
    # default of 100.0 puts the cutoff at max(slacks), so `slack <= cutoff`
    # keeps every row and nothing is stripped. Any default below 100 would push
    # the cutoff under the loosest value (100.0) and strip it, so this pins the
    # default itself, not just the hard-threshold branch. The slacks are chosen
    # so the loosest sits far above the rest: at p=100 the cutoff is 100.0; at
    # any p<100 the linear-interpolation cutoff drops below 100.0.
    rows = [_row(i, bytes([i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = dict(zip(keys, [1.0, 2.0, 3.0, 4.0, 100.0]))

    policy = SlackStrip(hard_threshold=float("inf"))
    assert policy.purge(rows, dual=None, slack=slack, iteration=0) == ()


def test_slack_strip_master_size_guard_reaches_composed_policies() -> None:
    SlackStrip(hard_threshold=8).validate_master_size(
        n_parameters=3, n_agents=5
    )
    SlackStrip(hard_threshold=float("inf")).validate_master_size(
        n_parameters=3, n_agents=5
    )

    with pytest.raises(ValueError, match=r"K \+ n_agents"):
        SlackStrip(hard_threshold=7).validate_master_size(
            n_parameters=3, n_agents=5
        )

    policy = Compose(
        admit_chain=[AddAll()],
        purge_chain=[SlackStrip(hard_threshold=7)],
    )
    with pytest.raises(ValueError, match="hard_threshold=7"):
        policy.validate_master_size(n_parameters=3, n_agents=5)


def _run_strip(
    slacks: Sequence[float], *, percentile: float, hard_threshold: float
) -> tuple[bool, ...]:
    """Drive SlackStrip.purge over a hand-built slack vector; return keep-mask.

    Rows are given distinct identities so the (agent_id, bundle_key) keying
    and slack-dict lookup are exercised on the way through.
    """
    rows = [_row(i, bytes([i])) for i in range(len(slacks))]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {keys[i]: float(slacks[i]) for i in range(len(rows))}
    policy = SlackStrip(percentile=percentile, hard_threshold=hard_threshold)
    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    return tuple(k not in retired_keys for k in keys)


def test_slack_strip_keep_set_matches_hand_derived_percentile() -> None:
    # Independent oracle: numpy's default (linear) percentile puts the p-th
    # cut at position (n-1)*p/100 in the sorted slacks. Keep iff slack <=
    # that cutoff. Expected keep-sets below are hand-computed from that
    # formula, NOT from the policy or the fixture's recomputed mask.

    # Clear strip: sorted slacks [0,1,2,3,100], n=5, pos=(5-1)*0.95=3.8 ->
    # cutoff = 3 + 0.8*(100-3) = 80.6. Only 100.0 exceeds it.
    keep = _run_strip(
        [0.0, 1.0, 2.0, 3.0, 100.0], percentile=95.0, hard_threshold=float("inf")
    )
    assert keep == (True, True, True, True, False)

    # Strict-boundary tie: sorted [0,1,2,3,3], pos=3.8 -> cutoff =
    # 3 + 0.8*(3-3) = 3.0. Both rows sitting exactly at 3.0 must be KEPT,
    # because the rule is `slack <= cutoff` (a `<` would strip them). Nothing
    # here exceeds the cutoff, so the whole set is kept.
    tie_keep = _run_strip(
        [0.0, 1.0, 2.0, 3.0, 3.0], percentile=95.0, hard_threshold=float("inf")
    )
    assert tie_keep == (True, True, True, True, True)


def test_slack_strip_hard_cap_keeps_the_smallest_slacks() -> None:
    # Independent oracle for the hard-cap override: when the percentile keep
    # would retain more than hard_threshold rows, keep only the hard_threshold
    # SMALLEST slacks (most-binding). slacks [5,1,4,2,6,3] with p=95 keep all
    # six; cap=3 forces keeping the three smallest values 1,2,3 at indices
    # 1,3,5 — hand-picked, not policy-derived.
    keep = _run_strip(
        [5.0, 1.0, 4.0, 2.0, 6.0, 3.0], percentile=95.0, hard_threshold=3
    )
    assert keep == (False, True, False, True, False, True)


def test_slack_strip_hard_cap_ties_are_stable_in_installed_order() -> None:
    # The hard-cap tie rule ("stable sort makes equal-looseness ties
    # deterministic in installed-row order") needs a tie group wide enough that
    # numpy's default (introselect) argsort reorders it. A 20-wide tie doesn't
    # separate stable from a 5-element tie, but here it does: row 0 is the
    # unique smallest slack, rows 1..20 all tie at 5.0, rows 21..22 are the
    # loosest tail. The p95 cutoff strips only the tail, leaving 21 rows, so the
    # cap=3 fires and must keep the three most-binding: row 0 plus the two
    # EARLIEST tied rows (1, 2). A non-stable sort surfaces late tied indices
    # (e.g. 19, 20) into the kept prefix instead.
    slacks = [1.0] + [5.0] * 20 + [9.0, 8.0]
    keep = _run_strip(slacks, percentile=95.0, hard_threshold=3)
    expected = tuple(i in (0, 1, 2) for i in range(len(slacks)))
    assert keep == expected


def test_slack_strip_ignores_cuts_without_a_reading() -> None:
    # Documented invariant: a cut absent from `slack` is neither strippable nor
    # part of the percentile population. Treating a missing reading as 0.0 would
    # (a) make the unsignalled row itself a candidate and (b) shift the cutoff,
    # changing which SIGNALLED rows strip. Real slacks [1,2,3,10] at p=70 strip
    # only the loosest (idx 3); injecting a phantom 0.0 for the extra row would
    # drag the cutoff down and also strip idx 2.
    rows = [_row(i, bytes([i])) for i in range(5)]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    slack = {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 3.0, keys[3]: 10.0}
    # keys[4] deliberately absent from the slack dict.
    policy = SlackStrip(percentile=70.0, hard_threshold=float("inf"))

    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    # Only the loosest signalled row strips; the reading-less row never does.
    assert retired_keys == {keys[3]}
    assert keys[4] not in retired_keys

    # The strip decision for the signalled rows is identical to running without
    # the extra row at all: the missing reading contributes nothing to the
    # population, so the cutoff (and thus the strip set) is unchanged.
    retired_no_extra = policy.purge(rows[:4], dual=None, slack=slack, iteration=0)
    assert {(row.agent_id, row.bundle_key) for row in retired_no_extra} == {
        keys[3]
    }


def test_slack_strip_reproduces_keep_set_on_snapshot() -> None:
    # Snapshot plumbing gate: the policy must strip on the frozen 200-cut
    # snapshot with correct row->(agent_id, bundle_key) keying, slack-dict
    # lookup, and kept/stripped split. This is the only test in the suite that
    # drives the policy at a population above the hard cap, so it is the sole
    # gate on the cap-governed regime.
    fixture = stripping_snapshot_fixture(n_cuts=200, seed=7)
    agent_ids = np.asarray(fixture["agent_ids"])
    bundle_keys = np.asarray(fixture["bundle_keys"])
    slacks = np.asarray(fixture["slacks"], dtype=np.float64)
    # The fixture recomputes the keep mask from the stated rule in a separate
    # module; it is an independent oracle for the whole per-row split, not a
    # rerun of the policy.
    expected_keep = np.asarray(fixture["expected_keep"], dtype=bool)

    rows = [
        _row(int(agent_ids[i]), bytes(bundle_keys[i]))
        for i in range(agent_ids.size)
    ]
    keys = [(row.agent_id, row.bundle_key) for row in rows]
    assert len(set(keys)) == len(keys), "snapshot cut identities must be unique"
    slack = {keys[i]: float(slacks[i]) for i in range(len(rows))}

    policy = SlackStrip(
        percentile=STRIP_PERCENTILE, hard_threshold=STRIP_HARD_THRESHOLD
    )
    retired = policy.purge(rows, dual=None, slack=slack, iteration=0)
    retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
    kept_keys = [k for k in keys if k not in retired_keys]

    # Full-output pin: the policy's per-row keep decision must equal the
    # fixture's independently-derived mask row-for-row. This kills a whole
    # class of mutations at once -- a scale-dependent hard cap (off only above
    # some population), a hardcoded percentile, a drifted STRIP_HARD_THRESHOLD,
    # keeping the loosest instead of the tightest under the cap, or a keying
    # bug -- every one of which flips at least one row here.
    keep_mask = np.array([k not in retired_keys for k in keys])
    np.testing.assert_array_equal(keep_mask, expected_keep)

    # This snapshot exercises the cap-governed regime: the percentile leg keeps
    # more than the cap, so the hard cap decides the split. Documenting it here
    # pins the regime the row-for-row mask is asserting.
    assert bool(fixture["hard_cap_active"]) is True
    assert int(keep_mask.sum()) == STRIP_HARD_THRESHOLD == 150
    assert len(retired) == int((~expected_keep).sum()) == 50

    # Structural invariants an independent reader can state without rerunning
    # the rule: (1) every stripped row is looser than every kept row, since the
    # rule keeps the smallest slacks; (2) the split is non-trivial; (3) retired
    # rows are a subset of the installed identities and disjoint from the kept
    # set.
    assert retired_keys, "snapshot must strip at least one cut"
    assert kept_keys, "snapshot must keep at least one cut"
    max_kept_slack = max(slack[k] for k in kept_keys)
    min_stripped_slack = min(slack[k] for k in retired_keys)
    assert max_kept_slack <= min_stripped_slack
    assert retired_keys <= set(keys)
    assert retired_keys.isdisjoint(set(kept_keys))


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

    # The free run admits many cuts per iteration on this fixture, so it is a
    # real (non-throttled) baseline; if it didn't, the strict comparison below
    # would be unexercised.
    assert free.cuts_admitted > free.iterations

    # Throttling oracle 1: k=1 admits <= 1 cut per progressing iteration, so
    # total admissions cannot exceed the iteration count. A policy that ignored
    # k and admitted every positive candidate would blow past this bound.
    assert throttled.cuts_admitted <= throttled.iterations

    # Throttling oracle 2: fewer cuts/iteration forces strictly more iterations
    # than the batch-admitting free run to reach the same face. `>=` would let a
    # policy that admits everything (equal iterations) pass; `>` pins throttling.
    assert throttled.iterations > free.iterations

    assert abs(throttled.objective - free.objective) < PARITY_BAND
