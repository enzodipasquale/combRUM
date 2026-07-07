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


def slack_strip_oracle(
    installed: list[CutRow],
    slack: dict[tuple[int, bytes], float],
    percentile: float,
) -> tuple[CutRow, ...]:
    # Independent reference for SlackStrip's no-cap path, written against the
    # documented contract rather than combrum's control flow: population is the
    # signalled subset only, cutoff is np.percentile, a row is stripped iff its
    # looseness is STRICTLY above the cutoff, and survivors come back in
    # installed order. Rebuilding the whole retired tuple this way pins the full
    # output, so cutoff-direction, population, and ordering mutations all fail
    # against it at once.
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
    # The public contract is the abstract interface: both methods must be
    # abstract, and their parameter order/names are what every concrete
    # policy and every caller relies on. Pin them explicitly so a rename or
    # reorder in policies.py trips this test.
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
    # MostViolated ranks candidates by their per-candidate violation, a
    # quantity that lives only in ``violations`` and cannot be recovered
    # from the rows: b (5.0) and c (1.0) are the two worst, a (0.2) is not.
    # Admitted rows come back in input order, so the expected result is
    # (b, c) -- derived by hand from the violation array, not from combrum.
    policy = MostViolated(k=2)
    a, b, c = make_row(1, b"a"), make_row(2, b"b"), make_row(3, b"c")
    violations = np.array([0.2, 5.0, 1.0])
    assert policy.admit([a, b, c], violations, iteration=0) == (b, c)

    # Swapping which row carries the large violation moves the selection,
    # proving the ranking flows through the ``violations`` argument.
    violations_swapped = np.array([5.0, 0.2, 1.0])
    assert policy.admit([a, b, c], violations_swapped, iteration=0) == (a, c)

    # Positivity gate: only strictly-positive violations are eligible, so k is
    # capped by the positive population, not blindly filled. Here only b is
    # positive (a is zero, c is negative), so despite k=2 the admitted set is
    # (b,) alone -- a cut with viol <= 0 must never be installed.
    mixed = np.array([0.0, 3.0, -1.0])
    assert policy.admit([a, b, c], mixed, iteration=0) == (b,)

    # Input-order contract: admitted rows come back in INPUT order, not in
    # descending-violation order. Here both x and y are admitted (k=2) but the
    # smaller violation carries the earlier row, so the two orderings diverge:
    # input order is (x, y) while descending-violation order would be (y, x).
    # Asserting (x, y) pins the docstring's "returned in input order" clause.
    x, y = make_row(4, b"x"), make_row(5, b"y")
    diverging = np.array([1.0, 5.0])
    assert policy.admit([x, y], diverging, iteration=0) == (x, y)

    # Empty-positive boundary: when NO candidate is strictly positive there is
    # nothing eligible to admit, so the result is () regardless of k. The mixed
    # case above proves non-positive rows are individually excluded, but only an
    # all-non-positive population exercises the ``positive.size == 0`` early
    # return. A regression that admitted every row in that case (falling back to
    # the full candidate set when nothing is positive) would install cuts that
    # can never bind, so pin () for a negative-and-zero mix, all-negative, and
    # all-zero -- the three ways the positive population can be empty.
    assert policy.admit([a, b], np.array([0.0, -1.0]), iteration=0) == ()
    assert policy.admit([a, b], np.array([-2.0, -1.0]), iteration=0) == ()
    assert policy.admit([a, b], np.array([0.0, 0.0]), iteration=0) == ()


def test_most_violated_ties_break_toward_earlier_candidate() -> None:
    # The docstring promises "ties at the cutoff break toward the earlier
    # candidate". With k=2 over three rows carrying IDENTICAL violations, the
    # cutoff falls between the second and third row, so exactly one of the
    # three tied rows is dropped. Keeping the two EARLIER rows (a, b) and
    # dropping the last (c) is the documented tie direction; the expected set
    # is fixed by the tie rule, not by combrum. A policy that broke ties toward
    # the later candidate would keep (b, c) and this assertion would fail.
    policy = MostViolated(k=2)
    a, b, c = make_row(1, b"a"), make_row(2, b"b"), make_row(3, b"c")
    tied = np.array([3.0, 3.0, 3.0])
    assert policy.admit([a, b, c], tied, iteration=0) == (a, b)

    # Three tied rows is too small to distinguish a stable sort from numpy's
    # default (quicksort happens to preserve order on tiny arrays), so a
    # `kind="stable"` -> default-sort refactor slips through the case above.
    # Use a tied block large enough that the two sorts diverge at the cutoff:
    # for 20 all-tied candidates with k=5, stable keeps the first five indices
    # while quicksort keeps a later scattered subset. Pin the ENTIRE kept set
    # to the earliest k rows (indices 0..k-1), derived from the "toward the
    # earlier candidate" rule alone. This kills every non-order-preserving
    # sort at the cutoff (default/quicksort/heapsort), not just heapsort.
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

    # Distinct branch: slack is PRESENT but keys NONE of the installed rows
    # (e.g. this iteration's readings are all for cuts that just left the
    # master). The signalled population is empty, so purge must still retire
    # nothing -- it must not fall back to retiring the whole installed set.
    # This crosses the empty-population boundary that the slack=None case
    # above cannot reach.
    unmatched_slack = {(99, b"z"): 1.0}
    assert policy.purge(installed, dual=None, slack=unmatched_slack, iteration=3) == ()

    # An empty slack map is the same boundary reached a different way: a
    # non-None signal that names no cut at all. Still retires nothing.
    assert policy.purge(installed, dual=None, slack={}, iteration=3) == ()


def test_slack_strip_retires_loosest_by_percentile() -> None:
    # SlackStrip strips cuts whose looseness is strictly above the given
    # percentile of the signalled population. Population = {loose: 5.0,
    # tight: 0.0}; np.percentile([5.0, 0.0], 50) = 2.5 (linear interp).
    # keep = looseness <= 2.5 -> {loose: strip, tight: keep}. The unkeyed
    # row has no slack reading, so it is neither population nor candidate
    # and cannot be retired. Expected retired = (loose,), derived from the
    # percentile arithmetic above rather than from combrum output.
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

    # Unkeyed rows must be excluded from the percentile POPULATION, not merely
    # spared from retirement. Pin that with a fixture where folding the two
    # unkeyed rows into the population at any default looseness would move the
    # cutoff enough to flip a keyed row: keyed population {near: 3.0, far: 5.0}
    # gives np.percentile([3.0, 5.0], 50) = 4.0, so keep = looseness <= 4.0
    # keeps ``near`` and strips only ``far`` -> retired = (far,). If the two
    # unkeyed rows were counted at looseness 0.0 the cutoff would drop to
    # np.percentile([3.0, 5.0, 0.0, 0.0], 50) = 1.5 and BOTH keyed rows would
    # be stripped -> (near, far). The single-row (far,) result is what proves
    # the unkeyed rows never entered the population.
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
    # The class promises "ties at the cutoff are kept": a cut is stripped iff
    # its looseness is STRICTLY above the percentile. Population slacks
    # {1.0, 1.0, 3.0}; np.percentile([1.0, 1.0, 3.0], 50) = 1.0, landing
    # exactly on the two 1.0 rows. keep = looseness <= 1.0 keeps both tied
    # rows and strips only the 3.0 row. Expected retired = (loosest,) alone,
    # derived from the percentile arithmetic. A strict-< cutoff would strip
    # all three (an over-purge); this pins the boundary.
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
    # The stripped set must actually derive from self._percentile, not a
    # baked-in constant. Over the SAME slack population {0,1,2,3,4} two
    # percentiles produce different cutoffs and thus different retired sets:
    #   np.percentile([0,1,2,3,4], 25) = 1.0 -> keep looseness <= 1.0 = {0,1},
    #       strip {2,3,4};
    #   np.percentile([0,1,2,3,4], 75) = 3.0 -> keep looseness <= 3.0 = {0,1,2,3},
    #       strip {4}.
    # Both expected sets are computed by hand from np.percentile, and they
    # DIVERGE, so a policy that hardcoded the percentile (any single constant
    # applied to both instances) could not satisfy both assertions.
    def population():
        rows = [make_row(i, bytes([ord("a") + i])) for i in range(5)]
        slack = {(i, bytes([ord("a") + i])): float(i) for i in range(5)}
        return rows, slack

    rows_25, slack_25 = population()
    lenient = SlackStrip(percentile=25.0)
    retired_25 = lenient.purge(rows_25, dual=None, slack=slack_25, iteration=3)
    assert retired_25 == (rows_25[2], rows_25[3], rows_25[4])
    assert retired_25 == slack_strip_oracle(rows_25, slack_25, 25.0)

    rows_75, slack_75 = population()
    strict = SlackStrip(percentile=75.0)
    retired_75 = strict.purge(rows_75, dual=None, slack=slack_75, iteration=3)
    assert retired_75 == (rows_75[4],)
    assert retired_75 == slack_strip_oracle(rows_75, slack_75, 75.0)

    # Guard the discrimination itself: if the two cutoffs ever coincided the
    # test would silently stop pinning the flow-through. Keep the retired
    # counts distinct so a refactor that made the fixture percentile-invariant
    # (e.g. dropping some population points) trips here rather than passing.
    assert len(retired_25) != len(retired_75)

    # Retired rows come back in INSTALLED order, not sorted by slack. The two
    # fixtures above build installed order to coincide with slack-ascending
    # order (slack[i] = i), so a regression that re-sorted the retired output by
    # slack would still produce the same tuples and slip through. Use a fixture
    # whose installed order DIVERGES from slack order and retires more than one
    # row so the two orderings are distinguishable:
    #   installed = [high, low, mid] with slacks {high: 5.0, low: 1.0, mid: 3.0};
    #   np.percentile([5,1,3], 25) = 2.0 -> keep looseness <= 2.0 = {low}, strip
    #   {high, mid}. In installed order that is (high, mid); a slack-ascending
    #   reorder-on-return would yield (mid, high) and fail. The whole expected
    #   tuple is rebuilt by the independent oracle, so ordering, cutoff, and
    #   population mutations are all pinned.
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
    assert diverging_retired == slack_strip_oracle(
        diverging_installed, diverging_slack, 25.0
    )


def test_purge_consumes_dual_only_signal() -> None:
    # A populated dual with slack=None must actually drive the decision.
    # PurgeInactive(max_age=1) retires a cut whose dual is (near-)zero for
    # one signalled call; a nonzero dual resets and retires nothing. Both
    # calls pass slack=None, so the outcome depends solely on the dual
    # values -- if dual were dropped the near-zero call could not retire.
    row = make_row(1, b"a")

    zero_dual = PurgeInactive(max_age=1)
    assert zero_dual.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == (
        row,
    )

    nonzero_dual = PurgeInactive(max_age=1)
    assert (
        nonzero_dual.purge([row], dual={(1, b"a"): 0.5}, slack=None, iteration=0) == ()
    )

    # Symmetric to SlackStrip's slack=None degrade: when the caller supplies no
    # dual signal, a dual-driven policy must retire nothing and move no counter.
    # Without this a policy that returned the whole installed set on dual=None
    # (retiring everything the moment the signal drops out) would pass. Feed a
    # nonempty installed set so "retires nothing" is a real claim, and confirm
    # the signal-free call did not seed a streak: a following zero-dual call is
    # the FIRST zero reading (streak 1, still no retire under max_age=1... which
    # would retire), so use max_age=2 to observe that the dual=None call left
    # the counter at 0.
    dual_none = PurgeInactive(max_age=2)
    other = make_row(2, b"b")
    assert dual_none.purge([row, other], dual=None, slack=None, iteration=0) == ()
    # The dual=None call created no streak, so this first real zero reading is
    # streak 1 (< max_age=2) and retires nothing; had dual=None counted as a
    # zero, this would already retire.
    assert dual_none.purge([row], dual={(1, b"a"): 0.0}, slack=None, iteration=1) == ()


def test_purge_inactive_requires_consecutive_zero_streak() -> None:
    # max_age=2 means retire only after TWO consecutive signalled zero-dual
    # calls. One zero reading is not enough (streak of 1 < max_age), so the
    # first call retires nothing; the second consecutive zero pushes the
    # streak to 2 and retires. Expectations derived from the counting rule
    # ("retire when streak >= max_age"), not from combrum. A policy that
    # ignores max_age (or that retires on the first zero) fails call 1.
    row = make_row(1, b"a")
    zero = {(1, b"a"): 0.0}

    policy = PurgeInactive(max_age=2)
    assert policy.purge([row], dual=zero, slack=None, iteration=0) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=1) == (row,)
    # A cut that stays dead PAST max_age must keep retiring, not come back to
    # life. This third consecutive zero (streak 3 > max_age=2) still retires;
    # a boundary that only fired at streak == max_age would silently resurrect
    # a long-dead cut on every call after the first retirement.
    assert policy.purge([row], dual=zero, slack=None, iteration=2) == (row,)


def test_purge_inactive_nonzero_reading_resets_streak() -> None:
    # A nonzero dual mid-streak resets the counter, so an accumulated streak
    # cannot survive a live reading. Sequence with max_age=2:
    #   zero    -> streak 1, no retire
    #   nonzero -> reset to 0, no retire
    #   zero    -> streak 1 (NOT 2), no retire  <- proves the reset happened
    #   zero    -> streak 2, retire
    # Without the reset the third call would already retire; the () assertion
    # on that call is what pins the reset semantics.
    row = make_row(1, b"a")
    zero = {(1, b"a"): 0.0}
    nonzero = {(1, b"a"): 0.5}

    policy = PurgeInactive(max_age=2)
    assert policy.purge([row], dual=zero, slack=None, iteration=0) == ()
    assert policy.purge([row], dual=nonzero, slack=None, iteration=1) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=2) == ()
    assert policy.purge([row], dual=zero, slack=None, iteration=3) == (row,)


def test_purge_inactive_prunes_counter_when_cut_leaves_installed() -> None:
    # Docstring: "Counters of cuts absent from installed are pruned each call,
    # so a re-entering cut starts a fresh streak." Build a partial streak on
    # ``a`` (one zero reading, max_age=2), then issue a call where ``a`` is not
    # installed (only ``b`` is) so ``a``'s counter is pruned, then re-add ``a``
    # with a zero reading. If the counter survived the absence, this re-entry
    # would land at streak 2 and retire; the pruned-counter contract requires a
    # fresh streak of 1, so it retires NOTHING. The full three-call sequence is
    # pinned so a policy that kept stale counters trips on the last call.
    a = make_row(1, b"a")
    b = make_row(2, b"b")

    policy = PurgeInactive(max_age=2)
    # streak(a) -> 1
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == ()
    # a is absent this call: its counter must be pruned. b's live reading keeps
    # b out of any streak.
    assert policy.purge([b], dual={(2, b"b"): 0.5}, slack=None, iteration=1) == ()
    # a re-enters with a zero reading: fresh streak of 1 (< max_age), retires
    # nothing. Un-pruned counters would make this streak 2 and wrongly retire a.
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=2) == ()
    # One more zero to confirm the fresh streak now reaches max_age here (streak
    # 2), pinning that pruning restarts rather than disables the counter.
    assert policy.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=3) == (a,)

    # The docstring says counters are pruned "each call", which includes
    # signal-free (dual=None) calls. The sequence above only prunes on a call
    # carrying a real dual, so a policy that pruned only AFTER the dual=None
    # early return would still pass it. Pin the dual=None branch directly:
    # seed a partial streak on ``a``, drop ``a`` on a dual=None call (only ``b``
    # installed), then re-add ``a`` with a zero. If the dual=None call pruned
    # a's counter, the re-entry is a fresh streak of 1 and retires nothing; a
    # prune-after-None-check regression keeps the stale streak 2 and wrongly
    # retires a on re-entry.
    none_pruned = PurgeInactive(max_age=2)
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=0) == ()
    assert none_pruned.purge([b], dual=None, slack=None, iteration=1) == ()
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=2) == ()
    # The fresh streak reaches max_age on this next zero, confirming the
    # dual=None call restarted rather than destroyed a's counter.
    assert none_pruned.purge([a], dual={(1, b"a"): 0.0}, slack=None, iteration=3) == (
        a,
    )


def test_purge_inactive_treats_near_zero_dual_within_tolerance_as_inactive() -> None:
    # The class counts a dual reading as inactive when it is within _DUAL_ATOL
    # (1e-10) of zero, not only at exact zero. Pin both sides of that band with
    # a single-call (max_age=1) policy so the decision is the tolerance test
    # alone, with no streak arithmetic in the way:
    #   1e-11 sits INSIDE the tolerance -> counts as inactive -> retires;
    #   1e-9 sits OUTSIDE the tolerance -> counts as active -> resets, retires
    #       nothing.
    # These magnitudes straddle _DUAL_ATOL=1e-10 by an order of magnitude each,
    # so a policy that collapsed the tolerance to exact-zero would fail to
    # retire the 1e-11 cut.
    row = make_row(1, b"a")

    within_tol = PurgeInactive(max_age=1)
    assert within_tol.purge(
        [row], dual={(1, b"a"): 1e-11}, slack=None, iteration=0
    ) == (row,)

    outside_tol = PurgeInactive(max_age=1)
    assert (
        outside_tol.purge([row], dual={(1, b"a"): 1e-9}, slack=None, iteration=0) == ()
    )

    # The near-zero reading must also FEED the streak, not merely retire once:
    # a reading inside the tolerance advances the counter exactly like an exact
    # zero would. With max_age=2 the first 1e-11 call retires nothing (streak
    # 1) and the second retires (streak 2), proving the tolerance-classified
    # reading is counted, not just short-circuited.
    streaked = PurgeInactive(max_age=2)
    near_zero = {(1, b"a"): 1e-11}
    assert streaked.purge([row], dual=near_zero, slack=None, iteration=0) == ()
    assert streaked.purge([row], dual=near_zero, slack=None, iteration=1) == (row,)

    # Pin the exact boundary, not just an order of magnitude on each side. The
    # comparison is `abs(pi) <= _DUAL_ATOL`, so a reading EXACTLY at the
    # threshold counts as inactive, while the next representable double above it
    # is active. Both magnitudes are built from _DUAL_ATOL itself (imported from
    # combrum.cut_policies) via np.nextafter, so the boundary is an independent
    # oracle rather than a copied constant. This kills the `<=`->`<` regression
    # (which would spare the exact-threshold cut) and any threshold shift off
    # _DUAL_ATOL by one ULP in either direction.
    for sign in (+1.0, -1.0):
        at_boundary = PurgeInactive(max_age=1)
        assert at_boundary.purge(
            [row], dual={(1, b"a"): sign * _DUAL_ATOL}, slack=None, iteration=0
        ) == (row,)

    just_above = np.nextafter(_DUAL_ATOL, np.inf)
    assert just_above > _DUAL_ATOL  # guard: the fixture really straddles the edge
    for sign in (+1.0, -1.0):
        above_boundary = PurgeInactive(max_age=1)
        assert (
            above_boundary.purge(
                [row], dual={(1, b"a"): sign * just_above}, slack=None, iteration=0
            )
            == ()
        )

    # The exact-threshold reading must feed the streak like any inactive one:
    # with max_age=2 the first boundary call is streak 1 (no retire) and the
    # second is streak 2 (retire), so the boundary classification flows into the
    # counter, not a one-shot short-circuit.
    boundary_streak = PurgeInactive(max_age=2)
    at_edge = {(1, b"a"): _DUAL_ATOL}
    assert boundary_streak.purge([row], dual=at_edge, slack=None, iteration=0) == ()
    assert boundary_streak.purge([row], dual=at_edge, slack=None, iteration=1) == (row,)
