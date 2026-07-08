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

#: Documented fixture constants, restated as test-owned literals. Deliberately
#: not imported from _support.synthetic: an edit to the fixture's constants
#: has to be measured against values that do not move with it.
_DOCUMENTED_KEEP_PCT = 95.0
_DOCUMENTED_CERT_GAP_FLOOR = 1e-4
_DOCUMENTED_CERT_GAP_HIGH = 1e-2

#: Sweep budget for the two-sided band check. ~4000 injected draws pull the
#: empirical minimum within ~0.3% of the band floor and the empirical maximum
#: within ~0.04% of the top, so the windows below clear the honest fixture yet
#: trip on either band end moved by >=1% (a decoupled floor lands the minimum
#: at ~1.016x; a 1e-2 -> 2e-4 top collapse drops the maximum to ~0.02x).
_CERT_FLOOR_SWEEP_SEEDS = 40
_CERT_FLOOR_SWEEP_N_AGENTS = 300
_CERT_FLOOR_SWEEP_CEILING = 1.008  # matches the fixture self-check's window
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
    fix = inexact_certification_fixture(9, 7)
    gaps = fix["gaps"]
    n_priced, n_inexact, worst_gap = fix["expected"]
    inexact_ids = fix["inexact_ids"]

    assert n_priced == gaps.size == 9
    # Documented rule, computed from the input rather than the output array.
    assert n_inexact == max(1, 9 // 3) == 3

    # inexact_ids: sorted, unique, in range, exactly n_inexact of them.
    assert inexact_ids.tolist() == sorted(set(inexact_ids.tolist()))
    assert inexact_ids.size == n_inexact
    assert np.all((inexact_ids >= 0) & (inexact_ids < n_priced))

    # Injected gaps sit in the documented (1e-4, 1e-2] band; every other agent
    # is exactly zero, so gaps > 0 marks precisely the inexact ids.
    injected = gaps[inexact_ids]
    assert np.all((injected > 1e-4 - 1e-12) & (injected <= 1e-2))
    mask = np.zeros(n_priced, dtype=bool)
    mask[inexact_ids] = True
    assert np.array_equal(gaps > 0.0, mask)
    assert np.all(gaps[~mask] == 0.0)

    # worst_gap dominates every gap and is itself one of the injected values.
    assert worst_gap == float(np.max(injected))
    assert all(g <= worst_gap for g in gaps.tolist())
    assert worst_gap in set(injected.tolist())

    # One seed cannot pin the band ends: its sampled minimum sits well above
    # the floor and its maximum well below the top. Sweep seeds and hold the
    # empirical extremes inside two-sided windows around the documented band.
    sweep_min = float("inf")
    sweep_max = 0.0
    for sweep_seed in range(_CERT_FLOOR_SWEEP_SEEDS):
        swept = inexact_certification_fixture(
            _CERT_FLOOR_SWEEP_N_AGENTS, sweep_seed
        )
        sweep_injected = swept["gaps"][swept["inexact_ids"]]
        sweep_min = min(sweep_min, float(sweep_injected.min()))
        sweep_max = max(sweep_max, float(sweep_injected.max()))
    ceiling = _DOCUMENTED_CERT_GAP_FLOOR * _CERT_FLOOR_SWEEP_CEILING
    assert _DOCUMENTED_CERT_GAP_FLOOR <= sweep_min <= ceiling
    high_floor = _DOCUMENTED_CERT_GAP_HIGH * _CERT_HIGH_SWEEP_FLOOR
    assert high_floor <= sweep_max <= _DOCUMENTED_CERT_GAP_HIGH


def test_cert_fixture_mixes_exact_and_inexact() -> None:
    # The inexact subset must be nonempty and proper at every size, and must
    # follow the documented one-third rule: at n_agents=40 the rule wants 13,
    # so a count that merely stays a proper subset (say a cap at 5) fails.
    for n_agents in (2, 3, 9, 40):
        fix = inexact_certification_fixture(n_agents, 11)
        n_priced, n_inexact, worst_gap = fix["expected"]
        assert 0 < n_inexact < n_priced
        assert n_inexact == max(1, n_agents // 3)
        assert worst_gap > 0.0
        assert np.any(fix["gaps"] == 0.0)


def test_cert_expected_triple_satisfies_the_certification_contract() -> None:
    fix = inexact_certification_fixture(12, 3)
    exp_priced, exp_inexact, exp_worst = fix["expected"]
    report = Certification(*fix["expected"])

    assert report.n_priced == exp_priced == 12
    assert report.n_inexact == exp_inexact
    assert report.worst_gap == exp_worst
    # A third of 12 agents are inexact, per the input rule.
    assert report.n_inexact == max(1, 12 // 3) == 4
    assert 0 < report.n_inexact <= report.n_priced
    # This triple is inexact, so worst_gap must be the strictly positive
    # injected gap, in the documented band.
    assert report.n_inexact > 0
    assert report.worst_gap == exp_worst > 0.0
    assert 1e-4 - 1e-12 < report.worst_gap <= 1e-2

    # Certification enforces "worst_gap > 0 iff n_inexact > 0" in both
    # directions; each match names the guard that must fire.
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
    # n_cuts=200, seed=7 trips the hard cap. 20% of rows (200//5 = 40) are
    # loose outliers in [3, 8], the other 160 form the tight body in [0, 1],
    # and the 95th percentile of a set that is 20% >= 3.0 lands in the loose
    # range.
    fix = stripping_snapshot_fixture(200, 7)
    slacks = fix["slacks"]
    threshold = fix["threshold"]
    final_keep = fix["expected_keep"]

    n_loose = max(1, 200 // 5)
    assert n_loose == 40
    assert int(np.count_nonzero(slacks >= 3.0)) == n_loose
    assert 3.0 <= threshold <= 8.0
    # Exact cutoff recomputed from the percentile definition, so a
    # STRIP_PERCENTILE drift (95 -> 90) that stays inside the loose band
    # still fails.
    assert threshold == _percentile_threshold(slacks, _DOCUMENTED_KEEP_PCT)

    # The percentile alone would keep ~190 rows, above the 150 cap.
    assert fix["hard_cap_active"] is True
    assert int(np.count_nonzero(slacks <= threshold)) > STRIP_HARD_THRESHOLD
    assert int(final_keep.sum()) == STRIP_HARD_THRESHOLD == 150

    # Kept rows are the smallest slacks, stated as a set property so argsort
    # tie-breaking never enters.
    kept_slacks = slacks[final_keep]
    dropped_slacks = slacks[~final_keep]
    assert kept_slacks.max() <= dropped_slacks.min()

    # Only 150 of the 160 body rows fit under the cap, so every loose
    # outlier is dropped.
    assert kept_slacks.max() < 3.0
    assert np.all(~final_keep[slacks >= 3.0])


def test_strip_snapshot_keeps_a_proper_subset() -> None:
    # Sizes below the hard cap: the keep count is governed by the percentile
    # rule alone, pinned against the recomputed cutoff.
    for n_cuts in (2, 5, 20, 101):
        fix = stripping_snapshot_fixture(n_cuts, 13)
        slacks = fix["slacks"]
        keep = fix["expected_keep"]
        kept = int(keep.sum())
        assert 0 < kept < n_cuts
        assert fix["hard_cap_active"] is False

        threshold = _percentile_threshold(slacks, _DOCUMENTED_KEEP_PCT)
        assert kept == int(np.count_nonzero(slacks <= threshold))
        assert slacks[keep].max() <= slacks[~keep].min()

    # Boundary of the hard-cap comparison: at n_cuts=158 the percentile keeps
    # EXACTLY 150 rows (ceil((158-1)*0.05) = 8 drop, at any seed), which is
    # not "more than" the cap, so the strict `>` trigger must stay False. A
    # `>=` off-by-one flips hard_cap_active here with the keep set unchanged.
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
    # The keys are the store's dedup identities, so they must distinguish
    # cuts, and the payload behind them must carry both truth values.
    assert len(set(keys.tolist())) > 1
    recovered_stack = np.stack([_unpack_bundle(k) for k in keys])
    assert recovered_stack.any() and not recovered_stack.all()
    # Replay the fixture's stream from SeedSequence((seed, _STRIP_TAG)):
    # agent_ids is the first draw (integers(0, max(2, n_cuts//2)), so ~2 cuts
    # share each owning agent), the bundle payload is the next, binarized at
    # the documented 0.5. Pinning the full vectors catches what dtype and
    # mixedness checks cannot — all-zeros ids, a wrong owner bound, a
    # distribution swap, a binarization drift (0.5 -> 0.7).
    _replay = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((7, _STRIP_TAG)))
    )
    expected_agent_ids = _replay.integers(0, max(2, 20 // 2), size=20)
    assert np.array_equal(fix["agent_ids"], expected_agent_ids)
    assert int(fix["agent_ids"].max()) < max(2, 20 // 2)
    expected_bundles = _replay.random((20, 12)) < 0.5
    assert np.array_equal(recovered_stack, expected_bundles)
    # Each key must be the exact byte-level store key: bytes, documented item
    # width, and invertible with no trailing bytes lost (a bundle ending in
    # False must round-trip).
    assert all(isinstance(key, bytes) for key in keys)
    for key in keys:
        recovered = _unpack_bundle(key)
        assert recovered.dtype == np.bool_
        assert recovered.shape == (12,)
        assert _pack_bundle(recovered) == key
    for name in _STRIP_FIELDS:
        assert fix[name].shape == (20,)


def test_strip_snapshot_is_deterministic_and_validated() -> None:
    a = stripping_snapshot_fixture(20, 7)
    b = stripping_snapshot_fixture(20, 7)
    for name in _STRIP_FIELDS:
        assert np.array_equal(a[name], b[name])
    # The dedup keys must vary with the seed; a constant payload would give
    # every snapshot one shared dedup identity.
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
    # The fixture exists to move real disk bytes; an all-False or all-True
    # payload has no IO/dedup value.
    assert fix["bundles"].any() and not fix["bundles"].all()
    assert fix["phi"].shape == (3, 2 * 6 + 1)
    assert fix["phi"].dtype == np.float64
    # Replay the SeedSequence((seed, tag)) stream and pin the exact values:
    # the (n_rows, M) uniform binarized at the documented 0.5, then phi via
    # standard_normal. Shape/dtype/mixedness alone would admit a threshold
    # drift (0.5 -> 0.7), a distribution swap (normal -> uniform), or a
    # shape-preserving reshuffle.
    _replay_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((5, _LARGE_M_TAG)))
    )
    expected_bundles = _replay_rng.random((3, 6 * 5)) < 0.5
    assert np.array_equal(fix["bundles"], expected_bundles)
    expected_phi = _replay_rng.standard_normal((3, 2 * 6 + 1))
    assert np.array_equal(fix["phi"], expected_phi)
    # Columns are the row-major enumeration of ordered off-diagonal pairs.
    # The ORDER is the contract: set equality would accept a column-major
    # reorder or a sender/receiver swap.
    pairs = list(zip(fix["senders"].tolist(), fix["receivers"].tolist()))
    assert len(pairs) == 30
    expected_pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    assert pairs == expected_pairs
    # First off-diagonal pair is (0, 1): senders lead with 0, receivers with
    # 1, so swapping the two columns fails here.
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
    # The bundle payload must vary with the seed too, not just phi; a constant
    # payload defeats the fixture's disk-bytes purpose.
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
    # Same user seed, different fixture tags: no two kinds may share a stream.
    # Compare the raw uniform draws at the same offset — derived masks would
    # differ even with a shared tag.
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
