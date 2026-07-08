"""The CutPolicy ABC contract and the built-in policies' core semantics."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from combrum.cut_policies import (
    _DUAL_ATOL,
    AddAll,
    MostViolated,
    PurgeInactive,
    SlackStrip,
)
from combrum.policies import CutPolicy
from combrum.transport import CutRow


def make_row(agent_id: int, key: bytes) -> CutRow:
    return CutRow(
        rep_id=0,
        agent_id=agent_id,
        phi=np.array([1.0, 0.0]),
        epsilon=0.0,
        bundle_key=key,
    )


def expected_slack_strip(
    installed: list[CutRow],
    slack: dict[tuple[int, bytes], float],
    percentile: float,
) -> tuple[CutRow, ...]:
    # SlackStrip's no-cap rule, restated in plain Python: population is the
    # signalled subset only, cutoff is np.percentile, a row is stripped iff
    # its looseness is strictly above the cutoff, retired rows in installed
    # order.
    signalled = [
        (row, slack[(row.agent_id, row.bundle_key)])
        for row in installed
        if (row.agent_id, row.bundle_key) in slack
    ]
    if not signalled:
        return ()
    values = [value for _, value in signalled]
    cutoff = float(np.percentile(np.asarray(values), percentile))
    return tuple(row for row, value in signalled if value > cutoff)


def test_abc_declares_admit_and_purge_with_documented_signatures() -> None:
    # Both methods must be abstract, and the parameter names/order are part
    # of the public contract.
    assert getattr(CutPolicy.admit, "__isabstractmethod__", False)
    assert getattr(CutPolicy.purge, "__isabstractmethod__", False)
    assert CutPolicy.__abstractmethods__ == frozenset({"admit", "purge"})

    admit_params = list(inspect.signature(CutPolicy.admit).parameters)
    assert admit_params == ["self", "candidates", "violations", "iteration"]

    purge_params = list(inspect.signature(CutPolicy.purge).parameters)
    assert purge_params == ["self", "installed", "dual", "slack", "iteration"]


def test_abc_not_instantiable() -> None:
    with pytest.raises(TypeError):
        CutPolicy()  # type: ignore[abstract]


def test_partial_override_remains_abstract() -> None:
    # Overriding only one of the two abstract methods must not yield an
    # instantiable class: the ABC machinery still reports the other missing.
    class AdmitOnly(CutPolicy):
        def admit(self, candidates, violations, iteration):  # type: ignore[override]
            return tuple(candidates)

    assert AdmitOnly.__abstractmethods__ == frozenset({"purge"})
    with pytest.raises(TypeError):
        AdmitOnly()  # type: ignore[abstract]

    class PurgeOnly(CutPolicy):
        def purge(self, installed, dual, slack, iteration):  # type: ignore[override]
            return ()

    assert PurgeOnly.__abstractmethods__ == frozenset({"admit"})
    with pytest.raises(TypeError):
        PurgeOnly()  # type: ignore[abstract]


def test_addall_admits_every_candidate_in_order() -> None:
    policy = AddAll()
    assert isinstance(policy, CutPolicy)
    a, b = make_row(1, b"a"), make_row(2, b"b")
    admitted = policy.admit([a, b], np.array([1.0, 2.0]), iteration=0)
    assert isinstance(admitted, tuple)
    # Identity policy: same rows, same order, retires nothing.
    assert admitted == (a, b)
    assert policy.purge([a, b], dual=None, slack=None, iteration=0) == ()


def test_most_violated_admits_the_k_largest_violations() -> None:
    # b (5.0) and c (1.0) carry the two largest violations; admitted rows
    # come back in input order, so expect (b, c).
    policy = MostViolated(k=2)
    a, b, c = make_row(1, b"a"), make_row(2, b"b"), make_row(3, b"c")
    violations = np.array([0.2, 5.0, 1.0])
    assert policy.admit([a, b, c], violations, iteration=0) == (b, c)

    # Swapping which row carries the large violation moves the selection.
    violations_swapped = np.array([5.0, 0.2, 1.0])
    assert policy.admit([a, b, c], violations_swapped, iteration=0) == (a, c)

    # Only strictly-positive violations are eligible: a is zero and c is
    # negative, so despite k=2 only b is admitted.
    mixed = np.array([0.0, 3.0, -1.0])
    assert policy.admit([a, b, c], mixed, iteration=0) == (b,)

    # Admitted rows come back in input order, not descending-violation
    # order: both are admitted here, and y has the larger violation.
    x, y = make_row(4, b"x"), make_row(5, b"y")
    diverging = np.array([1.0, 5.0])
    assert policy.admit([x, y], diverging, iteration=0) == (x, y)

    # When no candidate is strictly positive nothing is admitted, regardless
    # of k -- for a negative-and-zero mix, all-negative, and all-zero.
    assert policy.admit([a, b], np.array([0.0, -1.0]), iteration=0) == ()
    assert policy.admit([a, b], np.array([-2.0, -1.0]), iteration=0) == ()
    assert policy.admit([a, b], np.array([0.0, 0.0]), iteration=0) == ()


def test_most_violated_ties_break_toward_earlier_candidate() -> None:
    # Docstring: "ties at the cutoff break toward the earlier candidate".
    # k=2 over three identical violations keeps (a, b) and drops c.
    policy = MostViolated(k=2)
    a, b, c = make_row(1, b"a"), make_row(2, b"b"), make_row(3, b"c")
    tied = np.array([3.0, 3.0, 3.0])
    assert policy.admit([a, b, c], tied, iteration=0) == (a, b)

    # Three tied rows can't distinguish a stable sort from numpy's default
    # (quicksort preserves order on tiny arrays). With 20 tied candidates
    # and k=5, only a stable sort keeps the first five indices.
    big_k = 5
    big_rows = [make_row(i, bytes([i])) for i in range(20)]
    big_tied = np.full(len(big_rows), 3.0)
    big_policy = MostViolated(k=big_k)
    kept = big_policy.admit(big_rows, big_tied, iteration=0)
    assert kept == tuple(big_rows[:big_k])


def test_slack_strip_returns_empty_without_slack_signal() -> None:
    # slack=None means the caller could not supply slack this iteration;
    # a slack-driven policy must degrade to retiring nothing.
    policy = SlackStrip(percentile=50.0)
    installed = [make_row(1, b"a"), make_row(2, b"b")]
    assert policy.purge(installed, dual=None, slack=None, iteration=3) == ()

    # Slack present but keying none of the installed rows (e.g. readings for
    # cuts that just left the master): the signalled population is empty, so
    # purge still retires nothing.
    unmatched_slack = {(99, b"z"): 1.0}
    assert policy.purge(installed, dual=None, slack=unmatched_slack, iteration=3) == ()

    # Same boundary via an empty slack map.
    assert policy.purge(installed, dual=None, slack={}, iteration=3) == ()


def test_slack_strip_retires_loosest_by_percentile() -> None:
    # Population {loose: 5.0, tight: 0.0}; np.percentile([5.0, 0.0], 50) =
    # 2.5, strip iff looseness > 2.5 -> only loose. The unkeyed row has no
    # slack reading, so it is neither population nor candidate.
    policy = SlackStrip(percentile=50.0)
    loose = make_row(1, b"a")
    tight = make_row(2, b"b")
    unkeyed = make_row(3, b"c")
    installed = [loose, tight, unkeyed]
    slack = {
        (1, b"a"): 5.0,
        (2, b"b"): 0.0,
    }
    retired = policy.purge(installed, dual=None, slack=slack, iteration=3)
    assert retired == (loose,)

    # Unkeyed rows must be excluded from the percentile population, not
    # merely spared from retirement: keyed population {3.0, 5.0} gives a
    # cutoff of 4.0, retiring only ``far``. Counting the two unkeyed rows at
    # looseness 0.0 would drop the cutoff to 1.5 and retire both keyed rows.
    near = make_row(4, b"d")
    far = make_row(5, b"e")
    unkeyed_1 = make_row(6, b"f")
    unkeyed_2 = make_row(7, b"g")
    pop_installed = [near, far, unkeyed_1, unkeyed_2]
    pop_slack = {
        (4, b"d"): 3.0,
        (5, b"e"): 5.0,
    }
    pop_retired = policy.purge(pop_installed, dual=None, slack=pop_slack, iteration=3)
    assert pop_retired == (far,)


def test_slack_strip_keeps_ties_at_the_cutoff() -> None:
    # "Ties at the cutoff are kept": np.percentile([1.0, 1.0, 3.0], 50) =
    # 1.0 lands exactly on the two tied rows, so only the 3.0 row is
    # stripped. A strict-< keep rule would strip all three.
    policy = SlackStrip(percentile=50.0)
    tied_a = make_row(1, b"a")
    tied_b = make_row(2, b"b")
    loosest = make_row(3, b"c")
    installed = [tied_a, tied_b, loosest]
    slack = {
        (1, b"a"): 1.0,
        (2, b"b"): 1.0,
        (3, b"c"): 3.0,
    }
    retired = policy.purge(installed, dual=None, slack=slack, iteration=3)
    assert retired == (loosest,)


def test_slack_strip_cutoff_tracks_the_configured_percentile() -> None:
    # The cutoff must track self._percentile. Same population {0,1,2,3,4}:
    #   np.percentile(., 25) = 1.0 -> strip {2,3,4};
    #   np.percentile(., 75) = 3.0 -> strip {4}.
    def population():
        rows = [make_row(i, bytes([ord("a") + i])) for i in range(5)]
        slack = {(i, bytes([ord("a") + i])): float(i) for i in range(5)}
        return rows, slack

    rows_25, slack_25 = population()
    lenient = SlackStrip(percentile=25.0)
    retired_25 = lenient.purge(rows_25, dual=None, slack=slack_25, iteration=3)
    assert retired_25 == (rows_25[2], rows_25[3], rows_25[4])
    assert retired_25 == expected_slack_strip(rows_25, slack_25, 25.0)

    rows_75, slack_75 = population()
    strict = SlackStrip(percentile=75.0)
    retired_75 = strict.purge(rows_75, dual=None, slack=slack_75, iteration=3)
    assert retired_75 == (rows_75[4],)
    assert retired_75 == expected_slack_strip(rows_75, slack_75, 75.0)

    # Guard: the two percentiles must keep producing different retired sets.
    assert len(retired_25) != len(retired_75)

    # Retired rows come back in installed order, not sorted by slack. The
    # fixtures above have installed order equal to slack-ascending order, so
    # use one where they diverge: np.percentile([5,1,3], 25) = 2.0 strips
    # {high, mid}, which in installed order is (high, mid), not (mid, high).
    r_high = make_row(1, b"h")
    r_low = make_row(2, b"l")
    r_mid = make_row(3, b"m")
    diverging_installed = [r_high, r_low, r_mid]
    diverging_slack = {(1, b"h"): 5.0, (2, b"l"): 1.0, (3, b"m"): 3.0}
    diverging_policy = SlackStrip(percentile=25.0)
    diverging_retired = diverging_policy.purge(
        diverging_installed, dual=None, slack=diverging_slack, iteration=3
    )
    assert diverging_retired == (r_high, r_mid)
    assert diverging_retired == expected_slack_strip(
        diverging_installed, diverging_slack, 25.0
    )


def test_purge_consumes_dual_only_signal() -> None:
    # Both calls pass slack=None, so the outcome depends on the dual values
    # alone: with max_age=1 a zero dual retires after one signalled call, a
    # nonzero dual retires nothing.
    row = make_row(1, b"a")

    zero_dual = PurgeInactive(max_age=1)
    assert zero_dual.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == (
        row,
    )

    nonzero_dual = PurgeInactive(max_age=1)
    assert (
        nonzero_dual.purge([row], dual={(1, b"a"): 0.5}, slack=None, iteration=0) == ()
    )

    # dual=None must retire nothing and advance no counter. Use max_age=2 so
    # the zero reading that follows is observably the first of a streak; had
    # the dual=None call counted as a zero, it would retire below.
    dual_none = PurgeInactive(max_age=2)
    other = make_row(2, b"b")
    assert dual_none.purge([row, other], dual=None, slack=None, iteration=0) == ()
    # first real zero reading: streak 1 < max_age, no retire
    assert dual_none.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=1) == ()


def test_purge_inactive_requires_consecutive_zero_streak() -> None:
    # max_age=2: one zero reading is streak 1 and retires nothing; the
    # second consecutive zero retires.
    row = make_row(1, b"a")
    zero = {(1, b"a"): 0.0}

    policy = PurgeInactive(max_age=2)
    assert policy.purge([row], dual=zero, slack=None, iteration=0) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=1) == (row,)
    # A cut dead past max_age keeps retiring (streak 3 > 2), not just at
    # streak == max_age exactly.
    assert policy.purge([row], dual=zero, slack=None, iteration=2) == (row,)


def test_purge_inactive_nonzero_reading_resets_streak() -> None:
    # A nonzero dual mid-streak resets the counter. With max_age=2:
    #   zero    -> streak 1, no retire
    #   nonzero -> reset to 0, no retire
    #   zero    -> streak 1 (not 2), no retire
    #   zero    -> streak 2, retire
    # Without the reset the third call would already retire.
    row = make_row(1, b"a")
    zero = {(1, b"a"): 0.0}
    nonzero = {(1, b"a"): 0.5}

    policy = PurgeInactive(max_age=2)
    assert policy.purge([row], dual=zero, slack=None, iteration=0) == ()
    assert policy.purge([row], dual=nonzero, slack=None, iteration=1) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=2) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=3) == (row,)


def test_purge_inactive_prunes_counter_when_cut_leaves_installed() -> None:
    # Docstring: "Counters of cuts absent from installed are pruned each
    # call, so a re-entering cut starts a fresh streak." Seed streak 1 on
    # ``a``, drop it for one call, re-add it with a zero reading: a fresh
    # streak of 1 retires nothing, while a stale counter would land at
    # streak 2 and retire.
    a = make_row(1, b"a")
    b = make_row(2, b"b")

    policy = PurgeInactive(max_age=2)
    # streak(a) -> 1
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == ()
    # a absent: its counter is pruned; b's live reading starts no streak
    assert policy.purge([b], dual={(2, b"b"): 0.5}, slack=None, iteration=1) == ()
    # a re-enters: fresh streak of 1, no retire
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=2) == ()
    # the fresh streak reaches max_age here, so pruning restarts the counter
    # rather than disabling it
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=3) == (a,)

    # "Each call" includes dual=None calls; the sequence above only prunes on
    # a call carrying a real dual. Repeat it with the absence on a dual=None
    # call: the re-entry must still be a fresh streak of 1.
    none_pruned = PurgeInactive(max_age=2)
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == ()
    assert none_pruned.purge([b], dual=None, slack=None, iteration=1) == ()
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=2) == ()
    # the fresh streak reaches max_age on this next zero
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=3) == (
        a,
    )


def test_purge_inactive_treats_near_zero_dual_within_tolerance_as_inactive() -> None:
    # Inactive means within _DUAL_ATOL (1e-10) of zero, not exact zero.
    # max_age=1 keeps streak arithmetic out of the way: 1e-11 is inside the
    # band and retires, 1e-9 is outside and resets.
    row = make_row(1, b"a")

    within_tol = PurgeInactive(max_age=1)
    assert within_tol.purge(
        [row], dual={(1, b"a"): 1e-11}, slack=None, iteration=0
    ) == (row,)

    outside_tol = PurgeInactive(max_age=1)
    assert (
        outside_tol.purge([row], dual={(1, b"a"): 1e-9}, slack=None, iteration=0) == ()
    )

    # A near-zero reading also feeds the streak like an exact zero: with
    # max_age=2 the first 1e-11 call is streak 1, the second retires.
    streaked = PurgeInactive(max_age=2)
    near_zero = {(1, b"a"): 1e-11}
    assert streaked.purge([row], dual=near_zero, slack=None, iteration=0) == ()
    assert streaked.purge([row], dual=near_zero, slack=None, iteration=1) == (row,)

    # Absolute bracket around the band edge: 0.9e-10 is inside and retires,
    # 1.1e-10 is outside and never does.
    below = PurgeInactive(max_age=1)
    assert below.purge(
        [row], dual={(1, b"a"): 0.9e-10}, slack=None, iteration=0
    ) == (row,)
    above = PurgeInactive(max_age=1)
    assert (
        above.purge([row], dual={(1, b"a"): 1.1e-10}, slack=None, iteration=0) == ()
    )
    assert (
        above.purge([row], dual={(1, b"a"): 1.1e-10}, slack=None, iteration=1) == ()
    )

    # A reading just above the band (1e-8) is a full support reading for the
    # streak too: zero, 1e-8, zero cannot retire at max_age=2 because the
    # middle call resets the counter.
    reset = PurgeInactive(max_age=2)
    assert reset.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == ()
    assert reset.purge([row], dual={(1, b"a"): 1e-8}, slack=None, iteration=1) == ()
    assert reset.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=2) == ()

    # The comparison is `abs(pi) <= _DUAL_ATOL`: a reading exactly at the
    # threshold is inactive, the next representable double above it is
    # active.
    for sign in (+1.0, -1.0):
        at_boundary = PurgeInactive(max_age=1)
        assert at_boundary.purge(
            [row], dual={(1, b"a"): sign * _DUAL_ATOL}, slack=None, iteration=0
        ) == (row,)

    just_above = np.nextafter(_DUAL_ATOL, np.inf)
    assert just_above > _DUAL_ATOL  # guard against nextafter rounding back
    for sign in (+1.0, -1.0):
        above_boundary = PurgeInactive(max_age=1)
        assert (
            above_boundary.purge(
                [row], dual={(1, b"a"): sign * just_above}, slack=None, iteration=0
            )
            == ()
        )

    # The exact-threshold reading feeds the streak like any inactive one:
    # streak 1, then retire at streak 2.
    boundary_streak = PurgeInactive(max_age=2)
    at_edge = {(1, b"a"): _DUAL_ATOL}
    assert boundary_streak.purge([row], dual=at_edge, slack=None, iteration=0) == ()
    assert boundary_streak.purge([row], dual=at_edge, slack=None, iteration=1) == (row,)
