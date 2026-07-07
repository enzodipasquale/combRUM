from __future__ import annotations

import numpy as np
import pytest

from combrum.certification import Certification
from combrum.transport.base import _pack_bundle, _unpack_bundle
from _support.synthetic import (
    STRIP_HARD_THRESHOLD,
    _CERT_TAG,
    _LARGE_M_TAG,
    _STRIP_TAG,
    _fixture_rng,
    inexact_certification_fixture,
    large_m_generator,
    stripping_snapshot_fixture,
)


_STRIP_FIELDS = ("agent_ids", "bundle_keys", "slacks", "expected_keep")

#: The documented keep percentile, restated as a test-owned literal so drift in
#: the fixture's STRIP_PERCENTILE is measured against a value that does not move
#: with the edit. Do NOT import syn.STRIP_PERCENTILE here -- that would track the
#: regression instead of catching it.
_DOCUMENTED_KEEP_PCT = 95.0

#: The documented low end of the certification injection band (1e-4, 1e-2],
#: restated as a test-owned literal. Line 70's single-seed lower-band check
#: cannot see a floor decoupled DOWN from this (at the fixed seed the sampled
#: minimum sits well above 1e-4), so the sweep below measures the empirical
#: minimum injected gap over many seeds against this literal. Do NOT import
#: syn._CERT_GAP_FLOOR -- that tracks the regression instead of catching it.
_DOCUMENTED_CERT_GAP_FLOOR = 1e-4
#: The documented high end of the certification injection band (1e-4, 1e-2],
#: restated as a test-owned literal. The single-seed upper-band check (line 84,
#: injected <= 1e-2) cannot see the band top SHRUNK (e.g. 1e-2 -> 2e-4): at that
#: one seed every draw stays below 1e-2, so the one-sided guard never fires. The
#: sweep below measures the empirical MAXIMUM injected gap against this literal.
#: Do NOT import syn._MAX_MEANINGFUL_GAP -- that tracks the changed value.
_DOCUMENTED_CERT_GAP_HIGH = 1e-2
#: Seed budget + agent count for the two-sided sweep. With ~4000 injected draws
#: the empirical minimum hugs the band's true low to ~0.3% and the empirical
#: maximum hugs the band top to ~0.04%, so tight windows clear the honest
#: fixture yet trip on either end decoupled.
_CERT_FLOOR_SWEEP_SEEDS = 40
_CERT_FLOOR_SWEEP_N_AGENTS = 300
#: Ceiling on the empirical minimum, as a fraction of the floor. Matched to the
#: src self-check's _CERT_OBS_CEILING (0.8%); the honest sweep_min lands at
#: ~1.003x, so this clears it yet trips on a low end decoupled UP by >=1% (its
#: sweep_min lands at ~1.016x). A looser 2% ceiling let that ~1% upward
#: decoupling slip through, so the two-sided window was not actually two-sided.
_CERT_FLOOR_SWEEP_CEILING = 1.008
#: Floor on the empirical maximum, as a fraction of the band top. The honest
#: sweep_max hugs the top at ~0.9996x, so a 0.5% shortfall floor clears it yet
#: trips on a band top shrunk by >=1% (a 1e-2 -> 2e-4 collapse drops sweep_max
#: to ~0.02x). Mirrors the src self-check's _CERT_HIGH_OBS_FLOOR.
_CERT_HIGH_SWEEP_FLOOR = 0.995


def _percentile_threshold(values: np.ndarray, pct: float) -> float:
    # numpy 'linear' percentile from its definition, without calling
    # np.percentile (which is exactly the op the fixture uses). Threshold sits
    # at virtual index q=(n-1)*pct/100 over the sorted values.
    ordered = np.sort(values)
    n = ordered.size
    q = (n - 1) * (pct / 100.0)
    lo = int(np.floor(q))
    frac = q - lo
    if frac == 0.0:
        return float(ordered[lo])
    return float(ordered[lo] + frac * (ordered[lo + 1] - ordered[lo]))


# ---- inexact certification fixture -----------------------------------------


def test_cert_expected_triple_matches_the_injected_gaps() -> None:
    # Pin the triple against facts derived independently of the fixture's own
    # reductions. n_inexact follows from the documented input rule
    # max(1, n_agents//3); worst_gap must be the largest injected gap AND fall
    # in the documented injection band (1e-4, 1e-2]; the zero positions are
    # exactly the exact agents. Recomputing count_nonzero/gaps.max() -- the very
    # ops the source uses -- would only restate the implementation.
    fix = inexact_certification_fixture(9, 7)
    gaps = fix["gaps"]
    n_priced, n_inexact, worst_gap = fix["expected"]
    inexact_ids = fix["inexact_ids"]

    assert n_priced == gaps.size == 9
    # Independent count rule from the input, not from the output array.
    assert n_inexact == max(1, 9 // 3) == 3

    # inexact_ids is a sorted, unique subset of exactly n_inexact valid agents.
    assert inexact_ids.tolist() == sorted(set(inexact_ids.tolist()))
    assert inexact_ids.size == n_inexact
    assert np.all((inexact_ids >= 0) & (inexact_ids < n_priced))

    # Every injected gap lies strictly inside the documented (1e-4, 1e-2] band,
    # and every other agent is exactly zero -- so gaps>0 marks exactly the ids.
    injected = gaps[inexact_ids]
    assert np.all((injected > 1e-4 - 1e-12) & (injected <= 1e-2))
    mask = np.zeros(n_priced, dtype=bool)
    mask[inexact_ids] = True
    assert np.array_equal(gaps > 0.0, mask)
    assert np.all(gaps[~mask] == 0.0)

    # worst_gap is the largest injected gap: check it equals the max over the
    # injected subset and dominates every gap, without reusing gaps.max().
    assert worst_gap == float(np.max(injected))
    assert all(g <= worst_gap for g in gaps.tolist())
    assert worst_gap in set(injected.tolist())

    # The single-seed band checks above are one-sided at each end: the lower
    # check (line 84) cannot see a floor decoupled DOWN or UP from 1e-4, and the
    # upper check cannot see the band top SHRUNK from 1e-2 -- at this one seed the
    # sampled min sits well above the floor and the sampled max well below the
    # top, so neither guard fires. Sweep many seeds and pin BOTH the empirical
    # minimum and the empirical maximum into test-owned two-sided windows around
    # the documented band ends. Over ~4000 draws the honest min hugs 1e-4 to
    # ~0.3% and the honest max hugs 1e-2 to ~0.04%; the literals are test-owned,
    # so the oracle does not move with a src edit to the band constants.
    sweep_min = float("inf")
    sweep_max = 0.0
    for sweep_seed in range(_CERT_FLOOR_SWEEP_SEEDS):
        swept = inexact_certification_fixture(
            _CERT_FLOOR_SWEEP_N_AGENTS, sweep_seed
        )
        sweep_injected = swept["gaps"][swept["inexact_ids"]]
        sweep_min = min(sweep_min, float(sweep_injected.min()))
        sweep_max = max(sweep_max, float(sweep_injected.max()))
    # Low end, two-sided: catches a floor decoupled DOWN (min falls below the
    # documented floor) and a floor decoupled UP by >=1% (min rises past the
    # tightened ceiling).
    ceiling = _DOCUMENTED_CERT_GAP_FLOOR * _CERT_FLOOR_SWEEP_CEILING
    assert _DOCUMENTED_CERT_GAP_FLOOR <= sweep_min <= ceiling
    # High end: catches the band top SHRUNK (max drops below the shortfall
    # floor) -- e.g. 1e-2 -> 2e-4 -- and pins the max at/below the documented top
    # so a widened top is bounded from above.
    high_floor = _DOCUMENTED_CERT_GAP_HIGH * _CERT_HIGH_SWEEP_FLOOR
    assert high_floor <= sweep_max <= _DOCUMENTED_CERT_GAP_HIGH


def test_cert_fixture_mixes_exact_and_inexact() -> None:
    # A (0, 0) result pins nothing, so the inexact subset must be
    # nonempty and proper to exercise both counting paths.
    for n_agents in (2, 3, 9, 40):
        fix = inexact_certification_fixture(n_agents, 11)
        n_priced, n_inexact, worst_gap = fix["expected"]
        assert 0 < n_inexact < n_priced
        # Pin the documented one-third rule at every size, computed from the
        # input alone. A count cap/scaling that still leaves a proper subset
        # (e.g. min(.., 5)) passes the loose bound above but not this: at
        # n_agents=40 the rule wants 13, a cap-at-5 yields 5.
        assert n_inexact == max(1, n_agents // 3)
        assert worst_gap > 0.0
        assert np.any(fix["gaps"] == 0.0)


def test_cert_expected_triple_satisfies_the_certification_contract() -> None:
    # The triple round-trips through Certification and satisfies its cross-field
    # invariant (worst_gap > 0 iff n_inexact > 0), and Certification enforces
    # that invariant: violating triples are rejected below.
    fix = inexact_certification_fixture(12, 3)
    exp_priced, exp_inexact, exp_worst = fix["expected"]
    report = Certification(*fix["expected"])

    assert report.n_priced == exp_priced == 12
    assert report.n_inexact == exp_inexact
    assert report.worst_gap == exp_worst
    # Independent input rule: a third of 12 agents are inexact.
    assert report.n_inexact == max(1, 12 // 3) == 4
    assert 0 < report.n_inexact <= report.n_priced
    # The invariant tied to the concrete injected value, not a self-satisfying
    # biconditional: this triple is inexact, so worst_gap must be the strictly
    # positive gap the fixture injected, in the documented band.
    assert report.n_inexact > 0
    assert report.worst_gap == exp_worst > 0.0
    assert 1e-4 - 1e-12 < report.worst_gap <= 1e-2

    # Enforcement, not just consistency: Certification must REJECT a triple that
    # breaks either side of the "worst_gap > 0 iff n_inexact > 0" contract.
    # Values are forced by this test (n_inexact -> 0, worst_gap -> 0.0), not
    # taken from the source; each match pins the specific guard that must fire.
    with pytest.raises(ValueError, match="every call was exact"):
        Certification(exp_priced, 0, exp_worst)
    with pytest.raises(ValueError, match="some call was inexact"):
        Certification(exp_priced, exp_inexact, 0.0)


def test_cert_fixture_is_deterministic_and_seed_sensitive() -> None:
    a = inexact_certification_fixture(9, 7)
    b = inexact_certification_fixture(9, 7)
    assert np.array_equal(a["gaps"], b["gaps"])
    assert np.array_equal(a["inexact_ids"], b["inexact_ids"])
    assert a["expected"] == b["expected"]
    c = inexact_certification_fixture(9, 8)
    assert not np.array_equal(a["gaps"], c["gaps"])


def test_cert_fixture_validation_and_read_only() -> None:
    with pytest.raises(ValueError, match="n_agents"):
        inexact_certification_fixture(1, 7)
    with pytest.raises(ValueError, match="seed"):
        inexact_certification_fixture(9, -1)
    fix = inexact_certification_fixture(9, 7)
    with pytest.raises(ValueError, match="read-only"):
        fix["gaps"][0] = 1.0


# ---- stripping snapshot fixture ---------------------------------------------


def test_strip_expected_keep_matches_the_documented_rule() -> None:
    # n_cuts=200, seed=7 is the case that trips the hard cap. Pin it against
    # facts derived from the documented rule and the fixture's slack model,
    # NOT by rerunning the source's percentile/argsort sequence.
    fix = stripping_snapshot_fixture(200, 7)
    slacks = fix["slacks"]
    threshold = fix["threshold"]
    final_keep = fix["expected_keep"]

    # 20% of rows (n_loose = 200//5 = 40) are loose outliers in [3, 8]; the
    # remaining 160 form the tight body in [0, 1]. The 95th-percentile cutoff
    # of a set that is 20% >= 3.0 must itself land in the loose range.
    n_loose = max(1, 200 // 5)
    assert n_loose == 40
    assert int(np.count_nonzero(slacks >= 3.0)) == n_loose
    assert 3.0 <= threshold <= 8.0
    # Pin the exact cutoff to the documented 95th percentile, recomputed from
    # the percentile definition rather than the fixture's own call. This catches
    # a STRIP_PERCENTILE drift (e.g. 95->90) that stays inside the loose band
    # above but moves the numeric threshold.
    assert threshold == _percentile_threshold(slacks, _DOCUMENTED_KEEP_PCT)

    # The percentile alone keeps ~95% (190) rows, above the 150 cap, so the
    # cap must fire and keep exactly STRIP_HARD_THRESHOLD rows.
    assert fix["hard_cap_active"] is True
    assert int(np.count_nonzero(slacks <= threshold)) > STRIP_HARD_THRESHOLD
    assert int(final_keep.sum()) == STRIP_HARD_THRESHOLD == 150

    # Kept rows are the value-smallest slacks: every kept slack <= every
    # dropped slack (a set property, independent of argsort tie-breaking).
    kept_slacks = slacks[final_keep]
    dropped_slacks = slacks[~final_keep]
    assert kept_slacks.max() <= dropped_slacks.min()

    # With only 150 of 160 body rows kept, all 40 loose outliers are dropped,
    # so every kept slack is below the loose floor and every loose row is out.
    assert kept_slacks.max() < 3.0
    assert np.all(~final_keep[slacks >= 3.0])


def test_strip_snapshot_keeps_a_proper_subset() -> None:
    # These sizes never trip the hard cap, so the keep count is governed purely
    # by the documented 95th-percentile rule. Pin the exact count against an
    # independent percentile recompute (the linear-interpolation definition, not
    # the fixture's np.percentile(.., STRIP_PERCENTILE) call): a percentile-
    # constant drift changes the fixture but not this oracle, so 95->90 is caught
    # here rather than sailing through a bare 0 < kept < n_cuts bound.
    for n_cuts in (2, 5, 20, 101):
        fix = stripping_snapshot_fixture(n_cuts, 13)
        slacks = fix["slacks"]
        keep = fix["expected_keep"]
        kept = int(keep.sum())
        assert 0 < kept < n_cuts
        assert fix["hard_cap_active"] is False

        threshold = _percentile_threshold(slacks, _DOCUMENTED_KEEP_PCT)
        assert kept == int(np.count_nonzero(slacks <= threshold))

        # Set property: every kept slack is at or below every dropped slack, so
        # the keep set is exactly the value-smallest rows the rule selects.
        assert slacks[keep].max() <= slacks[~keep].min()

    # Boundary case for the hard-cap comparison. At n_cuts=158 the 95th
    # percentile keeps EXACTLY STRIP_HARD_THRESHOLD (150) rows, so the strict
    # `> STRIP_HARD_THRESHOLD` trigger must stay False (the percentile keep is
    # not "more than" the cap). A `>=` off-by-one would flip hard_cap_active
    # True here while the keep set itself is unchanged, so the flag is the only
    # observable. Seed-independent: ceil((158-1)*0.05)=8 rows drop at any seed.
    fix = stripping_snapshot_fixture(158, 0)
    slacks = fix["slacks"]
    keep = fix["expected_keep"]
    threshold = _percentile_threshold(slacks, _DOCUMENTED_KEEP_PCT)
    assert int(np.count_nonzero(slacks <= threshold)) == STRIP_HARD_THRESHOLD
    assert int(keep.sum()) == STRIP_HARD_THRESHOLD == 150
    assert fix["hard_cap_active"] is False


def test_strip_snapshot_schema() -> None:
    fix = stripping_snapshot_fixture(20, 7)
    assert fix["agent_ids"].dtype == np.int64
    assert np.all(fix["agent_ids"] >= 0)
    assert np.all(fix["slacks"] >= 0.0)
    keys = fix["bundle_keys"]
    assert keys.shape == (20,)
    # These keys are the store's dedup identities, so they must actually
    # distinguish cuts. A degenerate all-constant bundle payload collapses every
    # row to one key (all cuts share a dedup identity), which would silently
    # merge under real dedup; require the per-row keys to carry variation.
    assert len(set(keys.tolist())) > 1
    # And the underlying bundle payload itself must carry both truth values --
    # recovered from the keys, so an all-False/all-True payload is rejected even
    # if it somehow produced distinct keys.
    recovered_stack = np.stack([_unpack_bundle(k) for k in keys])
    assert recovered_stack.any() and not recovered_stack.all()
    # Pin the FULL bundle payload against an independent replay of the fixture's
    # stream, not just its mixedness. Reconstruct SeedSequence((seed, _STRIP_TAG)),
    # consume the agent_ids integer draw (same low/high/size) to reach the bundle
    # draw, then binarize the (n_cuts, 12) uniform at the documented 0.5. A
    # threshold drift (0.5->0.7) keeps keys distinct and the payload mixed but
    # changes these exact bits, so the whole payload -- not one row -- is the
    # oracle. This kills the class of payload regressions the dedup-key checks let
    # through.
    _replay = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((7, _STRIP_TAG)))
    )
    # agent_ids is the first draw off the stream: the per-cut owning agent,
    # documented as integers(0, max(2, n_cuts//2)). Pin its FULL vector against
    # this independent replay (not just dtype/non-negativity, which all-zeros or
    # an out-of-range high bound trivially satisfy). This kills the class of
    # agent_ids mutations -- all-zeros, wrong upper bound, distribution swap --
    # the schema's dtype/>=0 checks let through, and the max() bound below pins
    # the documented owner range so ~2 cuts share each agent (the collision
    # structure the field models).
    expected_agent_ids = _replay.integers(0, max(2, 20 // 2), size=20)
    assert np.array_equal(fix["agent_ids"], expected_agent_ids)
    assert int(fix["agent_ids"].max()) < max(2, 20 // 2)
    expected_bundles = _replay.random((20, 12)) < 0.5
    assert np.array_equal(recovered_stack, expected_bundles)
    # These must be the exact byte-level keys a cut store dedups on. Confirm
    # every key round-trips to a bool bundle of the documented item width, with
    # no trailing bytes lost (a bundle ending in False must survive).
    assert all(isinstance(key, bytes) for key in keys)
    for key in keys:
        recovered = _unpack_bundle(key)
        assert recovered.dtype == np.bool_
        assert recovered.shape == (12,)
        # The codec is invertible: repacking the recovered bundle must
        # reproduce the stored key byte-for-byte, so the fixture emits exactly
        # the store's dedup byte-key rather than a lossy stand-in.
        assert _pack_bundle(recovered) == key
    for name in _STRIP_FIELDS:
        assert fix[name].shape == (20,)


def test_strip_snapshot_is_deterministic_and_validated() -> None:
    a = stripping_snapshot_fixture(20, 7)
    b = stripping_snapshot_fixture(20, 7)
    for name in _STRIP_FIELDS:
        assert np.array_equal(a[name], b[name])
    # The dedup keys must change with the seed; a constant bundle payload is
    # seed-invariant and would leave every snapshot sharing one dedup identity.
    other = stripping_snapshot_fixture(20, 8)
    assert not np.array_equal(a["bundle_keys"], other["bundle_keys"])
    with pytest.raises(ValueError, match="n_cuts"):
        stripping_snapshot_fixture(1, 7)
    with pytest.raises(ValueError, match="read-only"):
        a["slacks"][0] = 9.0


# ---- large-M generator ------------------------------------------------------


def test_large_m_columns_are_the_directed_pair_space() -> None:
    fix = large_m_generator(T=6, n_rows=3, seed=5)
    assert fix["M"] == 6 * 5
    assert fix["bundles"].shape == (3, 30)
    assert fix["bundles"].dtype == np.bool_
    # The fixture exists to move real disk bytes, so the payload must carry
    # signal, not a degenerate constant. A bundle that is all-False (or all-
    # True) has zero IO/dedup value; assert both truth values are present.
    assert fix["bundles"].any() and not fix["bundles"].all()
    assert fix["phi"].shape == (3, 2 * 6 + 1)
    assert fix["phi"].dtype == np.float64
    # Pin bundles AND phi's exact VALUES against an independently reconstructed
    # generator, not just shape/dtype/mixedness. Replay the same
    # SeedSequence((seed, tag)) stream: draw the (n_rows, M) uniform and
    # binarize it at the documented 0.5 threshold, then draw phi via
    # standard_normal. Pinning the whole bundle payload kills a class of payload
    # mutations at once -- a threshold drift (0.5->0.7) that keeps the payload
    # mixed, a distribution swap, a shape-preserving reshuffle -- none of which
    # the any()/not all() mixedness check above can see. The phi replay is the
    # same discipline for the feature block: a distribution swap
    # (normal->uniform) of matching shape/dtype passes every shape/determinism
    # check but not this.
    _replay_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((5, _LARGE_M_TAG)))
    )
    expected_bundles = _replay_rng.random((3, 6 * 5)) < 0.5
    assert np.array_equal(fix["bundles"], expected_bundles)
    expected_phi = _replay_rng.standard_normal((3, 2 * 6 + 1))
    assert np.array_equal(fix["phi"], expected_phi)
    # Every ordered pair with sender != receiver appears exactly once, in the
    # documented row-major order. Pin the full ORDERED sequence, not just the
    # set: set equality survives any column permutation (a column-major reorder,
    # a sender/receiver swap), so an independent row-major enumeration is the
    # backstop for the ordered contract the fixture's own guard also protects.
    pairs = list(zip(fix["senders"].tolist(), fix["receivers"].tolist()))
    assert len(pairs) == 30
    expected_pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    assert pairs == expected_pairs
    # Symmetry-breaking check on the sender/receiver alignment: the first
    # off-diagonal pair is (0, 1), so senders lead with 0 and receivers with 1.
    # A swap of the two columns leaves the set intact but fails here.
    assert fix["senders"][0] == 0 and fix["receivers"][0] == 1


def test_large_m_grows_quadratically() -> None:
    # The point of the fixture: byte volume driven by M = T*(T-1).
    for t in (2, 4, 9):
        fix = large_m_generator(T=t, n_rows=2, seed=5)
        assert fix["M"] == t * (t - 1)
        assert fix["bundles"].shape[1] == fix["M"]


def test_large_m_is_deterministic_and_validated() -> None:
    a = large_m_generator(T=5, n_rows=4, seed=21)
    b = large_m_generator(T=5, n_rows=4, seed=21)
    for name in ("senders", "receivers", "bundles", "phi"):
        assert np.array_equal(a[name], b[name])
    # Cross-seed inequality must hold for the bundle payload too, not just phi.
    # A constant/degenerate bundle is seed-invariant and would slip past a
    # phi-only guard while defeating the fixture's disk-bytes purpose.
    other = large_m_generator(T=5, n_rows=4, seed=22)
    assert not np.array_equal(a["phi"], other["phi"])
    assert not np.array_equal(a["bundles"], other["bundles"])
    with pytest.raises(ValueError, match="T must be >= 2"):
        large_m_generator(T=1, n_rows=4, seed=21)
    with pytest.raises(ValueError, match="n_rows"):
        large_m_generator(T=5, n_rows=0, seed=21)
    with pytest.raises(ValueError, match="seed"):
        large_m_generator(T=5, n_rows=4, seed=-1)
    with pytest.raises(ValueError, match="read-only"):
        a["bundles"][0, 0] = True


def test_fixture_kinds_draw_disjoint_streams() -> None:
    # Same user seed, different fixture tags: no two kinds may share a
    # stream, or fixtures would correlate across gates. Compare the RAW
    # uniform draws each fixture's stream emits at the same offset -- derived
    # masks differ by construction and would pass even with a shared tag.
    seed = 7
    draws = {
        tag: _fixture_rng(seed, tag).random(64)
        for tag in (_CERT_TAG, _STRIP_TAG, _LARGE_M_TAG)
    }
    tags = sorted(draws)
    assert len({_CERT_TAG, _STRIP_TAG, _LARGE_M_TAG}) == 3
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            assert not np.array_equal(draws[a], draws[b])
