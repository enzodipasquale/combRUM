"""Cost bounds for :func:`combrum.reductions.canonical_sum`.

The determinizing argsort is a real O(n log n) cost the native O(n) reduce
does not pay, so canonical_sum legitimately runs a small multiple of the
native reduce. These tests reject cliffs, not the multiple: a per-element
Python loop, a quadratic reduction, or a repeated sort breaches the bounds
by orders of magnitude. Timing is noisy, so each measurement is the minimum
over several repetitions (load spikes inflate a sample, never deflate it)
and the ceilings are generous; structural checks on the reduce and gather
schedules cover the constant-factor regressions wall-clock ratios dilute.
A memory check bounds the determinism bookkeeping at O(1) per row.
"""

from __future__ import annotations

import importlib
import math
import time
import tracemalloc
import unittest.mock as mock

import numpy as np
import pytest

import combrum.reductions as reductions
from combrum.reductions import canonical_sum, canonical_sum_window_rows

pytestmark = pytest.mark.slow

#: The real ``np.add.reduce`` captured once, before any test wraps it.
_REAL_ADD_REDUCE = np.add.reduce

#: The real ``np.asarray`` / ``np.take``, captured before
#: :func:`_record_value_gathers` wraps them.
_REAL_ASARRAY = np.asarray
_REAL_TAKE = np.take


def _numpy_methods_module():
    for methods_path in ("numpy._core._methods", "numpy.core._methods"):
        try:
            module = importlib.import_module(methods_path)
        except ImportError:
            continue
        if hasattr(module, "umr_sum"):
            return module
    pytest.skip("numpy's private umr_sum helper is unavailable")


def _record_reduce_operands(fn):
    """Run ``fn`` and return the ``.shape`` of the first positional operand of
    every reduce it issues, in call order.

    The reduce is a small fraction of an argsort-dominated call, so a repeated
    reduce barely moves the wall-clock ratios; counting reduce operations
    catches it structurally. Patching both ``np.add.reduce`` (also the reduce
    ``np.sum`` calls internally) and numpy's private ``umr_sum`` (what
    ``ndarray.sum(axis=0)`` delegates to) makes the count independent of how a
    reduce is spelled, with no double counting.
    """
    shapes: list[tuple[int, ...]] = []
    methods_module = _numpy_methods_module()
    real_umr_sum = methods_module.umr_sum

    def _rec_reduce(a, *args, **kwargs):
        shapes.append(np.asarray(a).shape)
        return _REAL_ADD_REDUCE(a, *args, **kwargs)

    def _rec_umr_sum(a, *args, **kwargs):
        shapes.append(np.asarray(a).shape)
        return real_umr_sum(a, *args, **kwargs)

    with mock.patch.object(np.add, "reduce", _rec_reduce), mock.patch.object(
        methods_module, "umr_sum", _rec_umr_sum
    ):
        fn()
    return shapes


def _record_value_gathers(fn):
    """Run ``fn`` and return the ``(rows, WIDTH)`` shape of every fancy-index
    gather of the kernel's float64 value block, in call order.

    The sort path pays exactly one gather of the ``(n, WIDTH)`` values to
    materialize the contiguous copy the reduce sees. A duplicated gather is a
    copy, not a reduce, so the reduce recorder above cannot see it; it is a
    fraction of the argsort-dominated call, so wall-clock ratios dilute it;
    and a transient duplicate is freed before the tracemalloc peak. Counting
    gathers covers all of that, whether the extra ``vals[order]`` is retained,
    transient, or spelled ``np.take``.

    The value block is the only 2D float64 array the kernel builds (one
    ``np.asarray(values, dtype=np.float64)`` per call), so wrapping
    ``np.asarray`` to return an index-counting view of 2D float64 results
    instruments exactly that block. Windowed-branch slice reduces
    (``canonical[left:right]``) use slice keys, not integer index arrays, and
    are not counted.
    """
    shapes: list[tuple[int, ...]] = []

    class _GatherCounter(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype.kind in "iu":
                shapes.append(self.shape)
            return super().__getitem__(key)

    def _rec_asarray(a, *args, **kwargs):
        out = _REAL_ASARRAY(a, *args, **kwargs)
        if out.ndim == 2 and out.dtype == np.float64:
            return out.view(_GatherCounter)
        return out

    def _rec_take(a, indices, *args, **kwargs):
        arr = _REAL_ASARRAY(a)
        if arr.ndim == 2 and arr.dtype == np.float64:
            shapes.append(arr.shape)
        return _REAL_TAKE(a, indices, *args, **kwargs)

    with mock.patch.object(reductions.np, "asarray", _rec_asarray), mock.patch.object(
        reductions.np, "take", _rec_take
    ):
        fn()
    return shapes


def _expected_window_block_rows(ids: np.ndarray, window_rows: int) -> list[int]:
    """Row count of every non-empty window, re-derived from the ids alone:
    sort, then walk id-range windows ``[lo, lo + window_rows)`` from the
    minimum id. Plain ``np.sort`` + ``searchsorted`` with no gather, so it
    shares no code with the kernel's while-loop."""
    sids = np.sort(np.asarray(ids))
    lo = int(sids[0])
    stop = int(sids[-1]) + 1
    left = 0
    blocks: list[int] = []
    while lo < stop:
        hi = lo + window_rows
        right = int(np.searchsorted(sids, hi, side="left"))
        if right > left:
            blocks.append(right - left)
        left = right
        lo = hi
    return blocks


# Representative reduced shard: thousands of per-agent contributions of a
# K-wide moment, the shape sum_reproducible pools each iteration.
_N_CONTRIB = 20_000
_WIDTH = 8
_REPS = 15
#: Cliff-detector, not a band. A healthy argsort-and-reduce ratio sits in
#: the low single digits; 25 rejects a pathological implementation (a
#: Python-level loop runs hundreds to thousands x) while absorbing run
#: jitter.
_CLIFF_CEILING = 25.0
#: Ceiling against the single-argsort reference. canonical/native is a large
#: noisy magnitude (healthy ~14x) that the 25x dead-band barely constrains;
#: the reference pays the SAME dominant argsort once, so healthy canonical
#: sits at 1.3-1.5x it (n=20_000) while a doubled argsort lands ~2.1x. 1.7
#: sits in that gap.
_OVERHEAD_REFERENCE_CEILING = 1.7


def _min_seconds(fn, reps: int = _REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _single_sort_reduce(vals: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """One argsort, one gather, one reduce — the kernel's dominant cost paid
    exactly once, minus canonical_sum's uniqueness/validation bookkeeping.
    A healthy kernel sits just above this; a redundant argsort shows up as a
    near-doubling."""
    order = np.argsort(ids, kind="stable")
    return np.add.reduce(vals[order], axis=0)


def test_canonical_sum_overhead_is_bounded_not_a_cliff() -> None:
    rng = np.random.default_rng(20260613)
    # Scrambled unique global ids: the case where the sort actually reorders.
    global_ids = rng.permutation(_N_CONTRIB).astype(np.int64)
    values = rng.standard_normal((_N_CONTRIB, _WIDTH))

    canonical = _min_seconds(lambda: canonical_sum(values, global_ids))
    # Native baseline: the same reduction without the determinizing sort.
    native = _min_seconds(lambda: np.add.reduce(values, axis=0))
    reference = _min_seconds(lambda: _single_sort_reduce(values, global_ids))

    # Numeric equality is covered by test_reductions; this is purely cost.
    assert canonical < _CLIFF_CEILING * native, (
        f"canonical_sum took {canonical / native:.1f}x the native sum"
        f" (ceiling {_CLIFF_CEILING}x): the determinizing sort has become"
        " a throughput cliff, not bounded overhead"
    )
    # A doubled argsort stays under the 25x native dead-band; against the
    # single-sort reference it shows up as a near-doubling.
    sort_overhead = canonical / reference
    assert sort_overhead < _OVERHEAD_REFERENCE_CEILING, (
        f"canonical_sum ran {sort_overhead:.2f}x a single sort-and-reduce"
        f" (ceiling {_OVERHEAD_REFERENCE_CEILING}x): the determinizing sort is"
        " being paid more than once"
    )

    # The non-windowed branch must reduce the whole (n, WIDTH) block exactly
    # once; see _record_reduce_operands for why timing cannot see a repeat.
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(values, global_ids)
    )
    assert reduce_shapes == [(_N_CONTRIB, _WIDTH)], (
        f"canonical_sum issued {len(reduce_shapes)} full-block reduces"
        f" {reduce_shapes} on the non-windowed path (expected one over the"
        f" whole ({_N_CONTRIB}, {_WIDTH}) block)"
    )

    # Likewise one fancy-index gather of the value block; see
    # _record_value_gathers for the regressions only this count can see.
    gather_shapes = _record_value_gathers(
        lambda: canonical_sum(values, global_ids)
    )
    assert gather_shapes == [(_N_CONTRIB, _WIDTH)], (
        f"canonical_sum issued {len(gather_shapes)} full-block value gathers"
        f" {gather_shapes} on the non-windowed path (expected exactly one"
        f" gather of the ({_N_CONTRIB}, {_WIDTH}) block)"
    )


#: Base size for the growth check. Large enough that the O(n log n) sort
#: work dominates fixed per-call overhead (argument validation, the
#: ascending-scan, small-array numpy dispatch), so the measured ratio
#: reflects the asymptotic cost rather than a per-call floor.
_GROWTH_BASE = 100_000
#: Span for the growth check: measure at base and 8x base. A dense id
#: permutation has span == n, and 8 * base = 800_000 stays under
#: ``canonical_sum_window_rows(_WIDTH)`` (932067 for width 8), so both points
#: run the SAME non-windowed branch; if they straddled it, a common-path
#: regression would inflate only the smaller point and LOWER the fitted
#: exponent. Over 8x an O(n log n) reduce grows ~8.9x (exponent ~1.15); a
#: quadratic one grows ~64x (exponent 2.0).
_GROWTH_SPAN = 8
#: Fitted-exponent ceiling: log(large/small)/log(span). Log-linear scaling
#: sits well under 1.5; O(n^2) sits at ~1.8-2.0.
_GROWTH_EXPONENT_CEILING = 1.5
#: Magnitude ceiling on large-point canonical / native. A same-order constant
#: cliff (a per-element Python loop grows O(n) like the native reduce; k>=3
#: redundant argsorts grow O(n log n) like the healthy sort) leaves the
#: exponent flat, so magnitude needs its own bound. Healthy canonical sits
#: ~34x native at this size; an order-of-magnitude cliff blows past 60x. 45
#: leaves jitter headroom but is coarse: a doubled sort (~1.6x) hides under
#: it, which is what the reference ratio below is for.
_GROWTH_OVERHEAD_CEILING = 45.0
#: Ceiling on large-point canonical / single-sort-reference. The reference
#: pays the same dominant argsort (one stable argsort + one fancy-index + one
#: reduce), so healthy canonical sits just over it — measured 1.14-1.24x, the
#: excess being np.unique's duplicate check plus validation the reference
#: skips. A single redundant argsort pushes the ratio to ~1.8-1.95x; 1.5 sits
#: in that gap.
_REFERENCE_CEILING = 1.5


def test_canonical_sum_cost_grows_subquadratically() -> None:
    # Time the scrambled-id sort-and-reduce at two sizes on the same
    # non-windowed branch, then check three things: the fitted growth exponent
    # (a super-linear order change), overhead over the native reduce (coarse
    # magnitude), and the ratio over the single-argsort reference (a
    # same-order constant multiplier the first two are blind to).
    rng = np.random.default_rng(20260614)

    def cost(n: int) -> tuple[float, float, float]:
        # span == n keeps both sizes non-windowed.
        ids = rng.permutation(n).astype(np.int64)
        vals = rng.standard_normal((n, _WIDTH))
        canonical = _min_seconds(lambda: canonical_sum(vals, ids))
        native = _min_seconds(lambda: np.add.reduce(vals, axis=0))
        reference = _min_seconds(lambda: _single_sort_reduce(vals, ids))
        return canonical, native, reference

    small, _, _ = cost(_GROWTH_BASE)
    large, large_native, large_reference = cost(_GROWTH_SPAN * _GROWTH_BASE)
    ratio = large / small
    exponent = math.log(ratio) / math.log(_GROWTH_SPAN)
    assert exponent < _GROWTH_EXPONENT_CEILING, (
        f"canonical_sum cost grew {ratio:.1f}x over a {_GROWTH_SPAN}x size"
        f" span (fitted exponent {exponent:.2f} >= {_GROWTH_EXPONENT_CEILING}):"
        " the reduction is scaling super-linearly, worse than the argsort allows"
    )
    overhead = large / large_native
    assert overhead < _GROWTH_OVERHEAD_CEILING, (
        f"canonical_sum ran {overhead:.1f}x the native reduce at"
        f" n={_GROWTH_SPAN * _GROWTH_BASE} (ceiling {_GROWTH_OVERHEAD_CEILING}x)"
    )
    sort_overhead = large / large_reference
    assert sort_overhead < _REFERENCE_CEILING, (
        f"canonical_sum ran {sort_overhead:.2f}x a single sort-and-reduce"
        f" at n={_GROWTH_SPAN * _GROWTH_BASE} (ceiling {_REFERENCE_CEILING}x):"
        " the dominant argsort is being paid more than once"
    )

    # At n=800_000 the reduce is ~5% of the call, so even an 8x redundant
    # reduce passes all three ratios above (measured: exponent flat, native
    # ~37x, reference ~1.48x). The operand count is what catches it.
    large_n = _GROWTH_SPAN * _GROWTH_BASE
    check_ids = np.random.default_rng(20260614001).permutation(large_n).astype(
        np.int64
    )
    check_vals = np.random.default_rng(20260614002).standard_normal(
        (large_n, _WIDTH)
    )
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(check_vals, check_ids)
    )
    assert reduce_shapes == [(large_n, _WIDTH)], (
        f"canonical_sum issued {len(reduce_shapes)} full-block reduces"
        f" {reduce_shapes} at n={large_n} (expected one over the whole"
        f" ({large_n}, {_WIDTH}) block)"
    )

    # Same for the gather: a doubled one measures ~1.5x reference, right at
    # the ceiling, and a transient copy is invisible to the memory check.
    gather_shapes = _record_value_gathers(
        lambda: canonical_sum(check_vals, check_ids)
    )
    assert gather_shapes == [(large_n, _WIDTH)], (
        f"canonical_sum issued {len(gather_shapes)} full-block value gathers"
        f" {gather_shapes} at n={large_n} (expected exactly one gather of the"
        f" ({large_n}, {_WIDTH}) block)"
    )


def _peak_scratch_bytes(n: int, rng: np.random.Generator) -> int:
    """Peak bytes allocated inside a scrambled-id canonical_sum call.

    ``values`` and ``global_ids`` are built before tracing starts, so the
    traced peak is the kernel's own auxiliary footprint: the float64 copy of
    ``values``, the argsort index, and the fancy-indexed canonical copies.
    """
    ids = rng.permutation(n).astype(np.int64)
    vals = rng.standard_normal((n, _WIDTH))
    tracemalloc.start()
    try:
        canonical_sum(vals, ids)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


#: A size past canonical_sum_window_rows(_WIDTH) == 932067, so a dense id
#: permutation (span == n) drives the windowed while-loop. Healthy windowed
#: scratch is a slice view plus the fixed accumulator, matching the
#: non-windowed bytes/row; buffering each window's copy would inflate it.
#: Without this size the windowed branch is never memory-traced.
_WINDOWED_N = 1_200_000

#: Per-row scratch in bytes, hand-counted from the kernel's intermediates
#: (values arrives float64, so its cast is a no-copy view): argsort index (8)
#: + canonical id copy (8) + fancy-indexed float64 value copy (8 * _WIDTH) +
#: np.unique's transient unique+counts int64 pair (16). 96 for _WIDTH == 8,
#: matching the ~96-99 measured.
_SCRATCH_PER_ROW = 16 + 8 * _WIDTH + 16
#: Allocator slack on the hand-counted coefficient. 1.2x clears the measured
#: ~96-99 bytes/row with ~16 of headroom while still tripping on a retained
#: float32 copy of the values (+4 * _WIDTH == 32 bytes/row -> ~128 > 115); a
#: 1.4x band (134.4) would let that through.
_SCRATCH_SLACK = 1.2


def test_global_id_tag_is_o1_per_row() -> None:
    # Auxiliary footprint must be a constant number of bytes per contribution.
    # Drive the real kernel across both branches and require the measured
    # bytes/row to stay bounded and roughly flat: quadratic scratch or a
    # retained copy of the values array pushes it up.
    rng = np.random.default_rng(20260615)
    sizes = (_N_CONTRIB, 4 * _N_CONTRIB, 16 * _N_CONTRIB, _WINDOWED_N)
    bytes_per_row = [_peak_scratch_bytes(n, rng) / n for n in sizes]

    # The band must sit above the hand-counted coefficient yet below it plus a
    # float32 retention (+4 * _WIDTH bytes/row), or it stops catching
    # sub-float64 scratch bloat.
    ceiling = _SCRATCH_SLACK * _SCRATCH_PER_ROW
    assert _SCRATCH_PER_ROW < ceiling < _SCRATCH_PER_ROW + 4 * _WIDTH, (
        f"scratch band {ceiling:.0f} is miscalibrated against the"
        f" {_SCRATCH_PER_ROW}-byte coefficient and a +{4 * _WIDTH}-byte float32"
        " retention"
    )
    for n, per_row in zip(sizes, bytes_per_row):
        assert per_row < ceiling, (
            f"canonical_sum used {per_row:.0f} bytes/row of scratch at n={n}"
            f" (ceiling {ceiling:.0f}): auxiliary footprint is not O(1)/row"
        )

    # Flat across the span: an O(n^2) buffer would inflate the largest size.
    spread = max(bytes_per_row) / min(bytes_per_row)
    assert spread < 1.5, (
        f"canonical_sum scratch bytes/row varied {spread:.2f}x across the"
        f" size span ({[round(b) for b in bytes_per_row]}): the per-row"
        " coefficient is growing with n, not constant"
    )


#: Fixed window row count for the windowed-branch timing check (the default
#: 932067-row window would need >1.2M rows). Held fixed as n grows so the
#: number of windows scales with n: a per-window full-array reduce then costs
#: O(windows * n) == O(n^2), while a per-window slice reduce keeps total block
#: work at O(n).
_WINDOWED_ROWS = 1_000
#: Base and 8x span for the windowed growth check; both past _WINDOWED_ROWS so
#: both take the while-loop. 8x n means 8x windows.
_WINDOWED_BASE = 120_000
_WINDOWED_SPAN = 8
#: A healthy windowed reduce grows ~log-linearly (measured exponent ~1.15); a
#: per-window full reduce grows ~quadratically (~1.95).
_WINDOWED_EXPONENT_CEILING = 1.5
#: Ceiling on large-point canonical / windowed reference. The reference runs
#: the same block structure with one slice reduce per block, so healthy
#: canonical sits just above it (measured 1.18-1.21x); a per-window full
#: reduce blows the ratio to ~35-40x.
_WINDOWED_REFERENCE_CEILING = 1.5


def _windowed_reference(
    vals: np.ndarray, ids: np.ndarray, window_rows: int
) -> np.ndarray:
    """Memory-bounded windowed sum paying one slice reduce per block: sort by
    id, walk id-range blocks ``[lo, lo + window_rows)`` from the minimum id.
    A healthy kernel sits just above this; a per-window full-array reduce
    (O(windows * n)) lands an order of magnitude over."""
    vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(ids, kind="stable")
    sids = ids[order]
    svals = vals[order]
    total = np.zeros(svals.shape[1], dtype=np.float64)
    lo = int(sids[0])
    stop = int(sids[-1]) + 1
    left = 0
    while lo < stop:
        hi = lo + window_rows
        right = int(np.searchsorted(sids, hi, side="left"))
        if right > left:
            total += np.add.reduce(svals[left:right], axis=0)
        left = right
        lo = hi
    return total


def test_windowed_branch_cost_grows_subquadratically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The only other test on the windowed while-loop (the O(1)/row memory
    # check) times nothing, so a cliff inside the loop needs its own bound.
    # Shrink the window budget to force the loop, then apply the same two
    # checks as the non-windowed growth test: fitted exponent plus a ratio
    # over the windowed reference.
    monkeypatch.setattr(
        reductions,
        "_CANONICAL_SUM_WINDOW_BYTES",
        8 * (_WIDTH + 1) * _WINDOWED_ROWS,
    )
    window_rows = canonical_sum_window_rows(_WIDTH)
    assert window_rows == _WINDOWED_ROWS  # budget inverts exactly
    rng = np.random.default_rng(20260616)

    def cost(n: int) -> tuple[float, float]:
        ids = rng.permutation(n).astype(np.int64)  # span == n
        vals = rng.standard_normal((n, _WIDTH))
        assert n > window_rows  # windowed branch, not the fast path
        assert n // window_rows >= 8  # many blocks
        canonical = _min_seconds(lambda: canonical_sum(vals, ids))
        reference = _min_seconds(
            lambda: _windowed_reference(vals, ids, window_rows)
        )
        return canonical, reference

    small, _ = cost(_WINDOWED_BASE)
    large, large_reference = cost(_WINDOWED_SPAN * _WINDOWED_BASE)
    ratio = large / small
    exponent = math.log(ratio) / math.log(_WINDOWED_SPAN)
    assert exponent < _WINDOWED_EXPONENT_CEILING, (
        f"canonical_sum windowed cost grew {ratio:.1f}x over a {_WINDOWED_SPAN}x"
        f" span (fitted exponent {exponent:.2f} >= {_WINDOWED_EXPONENT_CEILING}):"
        " the windowed reduce is scaling super-linearly — a per-window full-array"
        " reduce or other quadratic regression in the while-loop"
    )
    sort_overhead = large / large_reference
    assert sort_overhead < _WINDOWED_REFERENCE_CEILING, (
        f"canonical_sum ran {sort_overhead:.2f}x the windowed block-loop"
        f" reference at n={_WINDOWED_SPAN * _WINDOWED_BASE}"
        f" (ceiling {_WINDOWED_REFERENCE_CEILING}x): more than one reduce per"
        " block on the memory-bounded path"
    )

    # The per-block slice reduce is tiny next to the shared argsort, so a
    # per-block constant multiplier can stay under the reference ceiling.
    # Check the reduce schedule instead: one reduce per non-empty block, with
    # operand row counts exactly the block partition (so they sum to n).
    # Five reduces per block gives 5x the calls; a per-window full-array
    # reduce makes each operand n rows.
    check_n = _WINDOWED_SPAN * _WINDOWED_BASE
    check_ids = np.random.default_rng(20260616001).permutation(check_n).astype(
        np.int64
    )
    check_vals = np.random.default_rng(20260616002).standard_normal(
        (check_n, _WIDTH)
    )
    expected_blocks = _expected_window_block_rows(check_ids, window_rows)
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(check_vals, check_ids)
    )
    assert all(shape[1:] == (_WIDTH,) for shape in reduce_shapes), (
        f"canonical_sum reduced non-({_WIDTH},)-wide operands {reduce_shapes}:"
        " the windowed branch must reduce (block_rows, WIDTH) slices"
    )
    block_rows = [int(shape[0]) for shape in reduce_shapes]
    assert block_rows == expected_blocks, (
        f"canonical_sum's windowed reduce schedule ({len(block_rows)} calls,"
        f" {sum(block_rows)} total rows) does not match the expected partition"
        f" ({len(expected_blocks)} blocks, {sum(expected_blocks)} rows == n)"
    )
