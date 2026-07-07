"""The determinism-cost gate.

:func:`combrum.reductions.canonical_sum` makes a cross-rank sum a pure
function of the (id, value) set by sorting on the global id before
reducing. That sort must not cost the goals, in two parts:

* **no throughput cliff** — the argsort is a real O(n log n) cost the
  native O(n) reduce does not pay, so canonical runs a small multiple of
  native; that is the price of the sort, not a regression. The gate
  forbids a cliff — a per-element Python loop or quadratic reduction —
  which breaches the bound by orders of magnitude and grows
  super-log-linearly. It asserts the surrogate a cliff cannot satisfy: a
  bounded ratio and ~log-linear growth.
* **O(1)/row tag** — each contribution carries exactly one integer id, so
  the determinism bookkeeping is structurally constant per row.

Timing is noisy, so the gate is tolerant, not tight: each reduction is the
minimum over several repetitions (load spikes only inflate a sample,
never deflate it), and the pass conditions are a generous cliff-ceiling
plus a growth bound, not a calibrated band. Run jitter breaches neither;
a cliff breaches both.
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

#: The real ``np.asarray`` / ``np.take`` captured once, before the gather oracle
#: wraps them. The oracle re-routes the kernel's value-block coercion through an
#: index-counting view; these preserve the true behaviour underneath.
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
    """Run ``fn`` with every observable reduce phrasing wrapped and return the
    ``.shape`` of the first positional operand of every reduce, in call order.

    Wall-clock gates dilute a reduce-only multiplier below threshold because the
    argsort+gather dominates the call; this counts the reduce operations
    directly instead, so a redundant reduce is caught by structure, not timing.
    The recorded operand shapes are the full per-call output: their count and
    row totals pin the reduce schedule against an independently derived one.

    The hook is phrasing-independent: it patches both ``np.add.reduce`` (the
    kernel's own reduce, and the reduce ``np.sum`` calls internally) and
    numpy's private ``umr_sum`` (what ``ndarray.sum(axis=0)`` delegates to).
    A redundant full-block reduce inflates the recorded schedule whether it is
    written as ``np.add.reduce``, ``np.sum``, or ``arr.sum(axis=0)`` — each of
    those records exactly one operand shape, with no double counting.
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
    """Run ``fn`` with the kernel's float64 value block instrumented and return
    the ``(rows, WIDTH)`` shape of every fancy-index gather of that block, in
    call order.

    The determinizing sort path pays exactly one fancy-index gather of the
    ``(n, WIDTH)`` values to materialize the sorted, contiguous copy the reduce
    sees. That gather is a real O(n*WIDTH) copy but is not a reduce, so the
    reduce-schedule oracle above is blind to a duplicated one; the wall-clock
    gates dilute it (a doubled gather adds only a fraction of the argsort-
    dominated call) and a transient duplicate is discarded before the
    tracemalloc peak, so the memory gate misses it too. This counts the gather
    directly: a redundant ``vals[order]`` — whether retained, transient, or
    spelled ``np.take`` — inflates the recorded schedule by structure.

    The only 2D float64 array the kernel builds is the value block (verified:
    one ``np.asarray(values, dtype=np.float64)`` per call), so wrapping
    ``np.asarray`` to return an integer-index-counting view of every 2D float64
    result instruments exactly that block and nothing else. Windowed-branch
    slice reduces (``canonical[left:right]``) use slice keys, not integer index
    arrays, so they are not gathers and are not counted.
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
    """Independently re-derive the windowed reduce schedule: sort the ids, walk
    id-range windows ``[lo, lo + window_rows)`` from the minimum id, and record
    the row count of every non-empty window in order. A structurally different
    twin of the kernel's while-loop (plain ``np.sort`` + ``searchsorted``, no
    fancy-index gather), used as the oracle for the block partition the kernel
    reduces over."""
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
#: Sharp ceiling for the same test against an independent single-argsort
#: reference. The native reduce pays no sort, so canonical/native is a large
#: noisy magnitude the 25x dead-band barely constrains (healthy ~14x). The
#: reference below pays the SAME dominant argsort once, so healthy canonical
#: sits just over it (measured 1.3-1.5x at n=20_000) while a doubled argsort
#: pushes it to ~2.1x. 1.7 sits in that gap, so a constant-factor sort
#: regression trips this test on its own rather than hiding in the dead-band.
_OVERHEAD_REFERENCE_CEILING = 1.7


def _min_seconds(fn, reps: int = _REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_canonical_sum_overhead_is_bounded_not_a_cliff() -> None:
    rng = np.random.default_rng(20260613)
    # Scrambled unique global ids: the realistic case where the sort
    # actually reorders (the kernel's reason for existing).
    global_ids = rng.permutation(_N_CONTRIB).astype(np.int64)
    values = rng.standard_normal((_N_CONTRIB, _WIDTH))

    canonical = _min_seconds(lambda: canonical_sum(values, global_ids))
    # Native baseline: the same reduction without the determinizing sort.
    native = _min_seconds(lambda: np.add.reduce(values, axis=0))
    # Independent reference paying the dominant argsort exactly once (see
    # _single_sort_reduce): a doubled sort stands out against it as a near
    # -doubling that the coarse native dead-band absorbs.
    reference = _min_seconds(lambda: _single_sort_reduce(values, global_ids))

    # Numeric equality is covered by test_reductions; this is purely cost.
    assert canonical < _CLIFF_CEILING * native, (
        f"canonical_sum took {canonical / native:.1f}x the native sum"
        f" (ceiling {_CLIFF_CEILING}x): the determinizing sort has become"
        " a throughput cliff, not bounded overhead"
    )
    # Sharper signal: pin the overhead against the single-argsort reference so a
    # constant-factor sort regression (e.g. a doubled argsort) that stays under
    # the 25x native dead-band still trips here.
    sort_overhead = canonical / reference
    assert sort_overhead < _OVERHEAD_REFERENCE_CEILING, (
        f"canonical_sum ran {sort_overhead:.2f}x a single honest sort-and-reduce"
        f" (ceiling {_OVERHEAD_REFERENCE_CEILING}x): the determinizing sort is"
        " being paid more than once — a constant-multiplier cliff on the sort"
        " path the native-ratio gate is too coarse to see"
    )

    # Timing-free signal for the reduce path. The reduce is a small, gather
    # -dominated fraction of the call, so a doubled/repeated reduce that keeps
    # the sum bitwise-identical hides under every wall-clock ratio above. Count
    # the reduce operations directly: the non-windowed branch must reduce the
    # whole (n, WIDTH) block exactly once. The oracle records every reduce
    # phrasing (np.add.reduce, np.sum, and arr.sum(axis=0) via umr_sum), so a
    # k-fold redundant reduce shows up as k+1 operands however it is written,
    # regardless of how cheap each one is.
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(values, global_ids)
    )
    assert reduce_shapes == [(_N_CONTRIB, _WIDTH)], (
        f"canonical_sum issued {len(reduce_shapes)} full-block reduces"
        f" {reduce_shapes} on the non-windowed path (expected one over the whole"
        f" ({_N_CONTRIB}, {_WIDTH}) block): a repeated reduce is a"
        " constant-multiplier cliff on the reduce path the wall-clock gates dilute"
    )

    # Timing-free signal for the gather path. Materializing the sorted values is a
    # fancy-index gather of the whole (n, WIDTH) block — a real O(n*WIDTH) copy,
    # but a copy, not a reduce, so the reduce oracle above never sees a doubled
    # one. It is also a fraction of the argsort-dominated call, so every
    # wall-clock ratio dilutes it, and a transient duplicate is freed before the
    # tracemalloc peak so the memory gate misses it. Pin the whole gather
    # schedule: the sort path gathers the block exactly once. A redundant
    # vals[order] (retained, transient, or spelled np.take) records an extra
    # operand however cheap or short-lived it is.
    gather_shapes = _record_value_gathers(
        lambda: canonical_sum(values, global_ids)
    )
    assert gather_shapes == [(_N_CONTRIB, _WIDTH)], (
        f"canonical_sum issued {len(gather_shapes)} full-block value gathers"
        f" {gather_shapes} on the non-windowed path (expected exactly one gather"
        f" of the ({_N_CONTRIB}, {_WIDTH}) block): the determinizing sort's"
        " gather is being paid more than once — a constant-multiplier cliff the"
        " reduce, timing, and memory gates all miss"
    )


#: Base size for the growth check. Large enough that the O(n log n) sort
#: work dominates fixed per-call overhead (argument validation, the
#: ascending-scan, small-array numpy dispatch), so the measured ratio
#: reflects the asymptotic cost rather than a per-call floor.
_GROWTH_BASE = 100_000
#: Doubling span for the growth check: measure at base and 8x base. Both
#: points stay under ``canonical_sum_window_rows(_WIDTH)`` (932067 for width 8)
#: so both exercise the SAME non-windowed reduce branch. A dense id permutation
#: has span == n, so 8 * base = 800_000 is still below the window threshold; if
#: the two points straddled it, a common-path regression would inflate only the
#: smaller point (the larger runs the windowed loop) and perversely LOWER the
#: fitted exponent instead of raising it. Over an 8x span an O(n log n) reduce
#: grows ~8.9x, giving a fitted exponent ~1.15; a quadratic regression grows
#: ~64x, exponent 2.0.
_GROWTH_SPAN = 8
#: Fitted-exponent ceiling: log(large/small)/log(span). Log-linear scaling
#: sits well under 1.5; O(n^2) sits at ~1.8-2.0. Rejects a super-linear
#: regression while absorbing the run-to-run jitter that nudges the healthy
#: exponent above its ~1.15 mean.
_GROWTH_EXPONENT_CEILING = 1.5
#: Magnitude ceiling on the large-point overhead ratio (canonical / native
#: reduce). The exponent gate only sees a change of asymptotic ORDER; a cliff
#: that keeps the same order but multiplies the constant — a per-element Python
#: loop reduce grows O(n) like the native reduce, k>=3 redundant argsorts grow
#: O(n log n) like the healthy sort — leaves the exponent flat. The native
#: reduce is an independent reference (no determinizing sort); healthy canonical
#: sits ~34x it at this size, an order-of-magnitude cliff blows past 60x. 45
#: rejects the cliff with headroom for the ~34x healthy mean and its run jitter.
#: This magnitude gate is a coarse dead-band that only catches k>=3 redundant
#: argsorts; a doubled sort (~1.6x constant regression) hides under it, so the
#: reference-ratio gate below is what supplies the sharper signal.
_GROWTH_OVERHEAD_CEILING = 45.0
#: Sharp ceiling on canonical / single-sort-reference at the large point. The
#: native reduce mixes only the reduce cost, so canonical/native is a large
#: noisy magnitude that needs a coarse dead-band. The independent reference
#: below pays the SAME dominant argsort the healthy kernel does (one stable
#: argsort + one fancy-index + one reduce), so healthy canonical sits just over
#: it (measured 1.14-1.24x: the excess is np.unique's duplicate check plus
#: argument validation the reference skips). A single redundant argsort roughly
#: doubles the dominant cost and pushes the ratio to ~1.8-1.95x. 1.5 sits in
#: that gap — ~1.2x over the healthy max, well under a doubled sort — so it
#: trips on the k=2 cliff the coarse magnitude gate misses.
_REFERENCE_CEILING = 1.5


def _single_sort_reduce(vals: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Minimal honest sort-and-reduce: exactly one argsort, one gather, one
    reduce. Independent oracle for the kernel's dominant cost — it pays the
    determinizing sort once and skips canonical_sum's uniqueness/validation
    bookkeeping, so a healthy kernel sits just above it and a redundant argsort
    stands out as a near-doubling."""
    order = np.argsort(ids, kind="stable")
    return np.add.reduce(vals[order], axis=0)


def test_canonical_sum_cost_grows_subquadratically() -> None:
    # Structural backstop. Measure the scrambled-id sort-and-reduce cost at
    # _GROWTH_BASE and _GROWTH_SPAN x _GROWTH_BASE, BOTH on the non-windowed
    # branch, then check three independent things: the fitted growth exponent
    # (catches a super-linear order change), the large-point overhead over the
    # native reduce (coarse magnitude cliff), and the large-point ratio over an
    # independent single-argsort reference (catches a same-order constant
    # multiplier — a doubled argsort — the first two are blind to).
    rng = np.random.default_rng(20260614)

    def cost(n: int) -> tuple[float, float, float]:
        # Scrambled unique ids so the argsort branch actually runs. span == n
        # keeps both sizes non-windowed. Pair each canonical timing with the
        # native reduce (an independent reference paying NO sort) and a single
        # honest sort-and-reduce (an independent reference paying the sort ONCE).
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
        f" n={_GROWTH_SPAN * _GROWTH_BASE} (ceiling {_GROWTH_OVERHEAD_CEILING}x):"
        " the sort-and-reduce has a constant-multiplier cliff on the common"
        " reduce path that leaves the growth exponent flat"
    )
    sort_overhead = large / large_reference
    assert sort_overhead < _REFERENCE_CEILING, (
        f"canonical_sum ran {sort_overhead:.2f}x a single honest sort-and-reduce"
        f" at n={_GROWTH_SPAN * _GROWTH_BASE} (ceiling {_REFERENCE_CEILING}x):"
        " the kernel is paying the dominant argsort more than once — a"
        " constant-multiplier cliff on the sort path"
    )

    # Timing-free reduce-path signal. At n=800_000 the reduce is ~5% of an
    # argsort-dominated call, so an 8x redundant reduce only nudges the ratios
    # above (measured k=8: exponent flat, native ~37x, reference ~1.48x) and all
    # three pass. Count the reduce operations instead: the non-windowed branch
    # reduces the whole (n, WIDTH) block exactly once, and the oracle records
    # every reduce phrasing (np.add.reduce, np.sum, arr.sum), so a repeated
    # reduce shows up as extra operands whatever it costs or however it is spelled.
    large_n = _GROWTH_SPAN * _GROWTH_BASE
    oracle_ids = np.random.default_rng(20260614001).permutation(large_n).astype(
        np.int64
    )
    oracle_vals = np.random.default_rng(20260614002).standard_normal(
        (large_n, _WIDTH)
    )
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(oracle_vals, oracle_ids)
    )
    assert reduce_shapes == [(large_n, _WIDTH)], (
        f"canonical_sum issued {len(reduce_shapes)} full-block reduces"
        f" {reduce_shapes} at n={large_n} (expected one over the whole"
        f" ({large_n}, {_WIDTH}) block): a repeated reduce is a"
        " constant-multiplier cliff the wall-clock growth gates dilute"
    )

    # Timing-free gather-path signal at the large point. The sorted-value gather
    # is a fraction of the argsort-dominated call, so a doubled gather stays
    # inside every wall-clock ratio (measured ~1.5x reference vs the 1.5 ceiling)
    # and a transient copy evades the memory gate. Pin the gather schedule: one
    # full-block gather over the whole (large_n, WIDTH) values, whatever it costs.
    gather_shapes = _record_value_gathers(
        lambda: canonical_sum(oracle_vals, oracle_ids)
    )
    assert gather_shapes == [(large_n, _WIDTH)], (
        f"canonical_sum issued {len(gather_shapes)} full-block value gathers"
        f" {gather_shapes} at n={large_n} (expected exactly one gather of the"
        f" ({large_n}, {_WIDTH}) block): the determinizing sort's gather is being"
        " paid more than once — a constant-multiplier cliff the timing, reduce,"
        " and memory gates all dilute"
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
#: permutation (span == n) drives the windowed while-loop rather than the single
#: np.add.reduce. Its scratch is a slice view of the sorted values plus the
#: fixed accumulator, so healthy bytes/row match the non-windowed sizes; a
#: regression that buffered each window's copy in a growing list would inflate
#: it. Without this size the windowed branch is never memory-traced.
_WINDOWED_N = 1_200_000

#: The exact per-row scratch coefficient, in bytes, derived from the kernel's
#: named intermediates (see the comment in the test): argsort index (8) +
#: canonical id copy (8) + fancy-indexed float64 value copy (8 * _WIDTH) +
#: np.unique's transient unique+counts int64 pair (16). Independent of any
#: measurement — it is hand-counted from the algorithm.
_SCRATCH_PER_ROW = 16 + 8 * _WIDTH + 16
#: Allocator-slack multiplier on the hand-counted coefficient. Measured healthy
#: bytes/row is ~96-99 (spread only ~1.03), so a 1.2x band clears healthy with
#: ~16 bytes/row of headroom while tripping on a retained float32 copy of the
#: values array (+4 * _WIDTH == 32 bytes/row -> ~128 > 115), not just a retained
#: float64 copy (+64). The old 1.4x band (134.4) let a float32 retention through.
_SCRATCH_SLACK = 1.2


def test_global_id_tag_is_o1_per_row() -> None:
    # canonical_sum's auxiliary footprint must be O(1)/row: a constant number
    # of bytes per contribution (the argsort index is one int64 per row, plus
    # the float64 value copies), so total scratch is linear in n. Drive the
    # real kernel across both branches and require the measured bytes/row to
    # stay bounded and roughly constant. Quadratic scratch (an (n, n) buffer, a
    # per-row coefficient that grows with n) or a constant multiplicative bloat
    # (an accidental retained copy of the values array) push bytes/row up.
    rng = np.random.default_rng(20260615)
    sizes = (_N_CONTRIB, 4 * _N_CONTRIB, 16 * _N_CONTRIB, _WINDOWED_N)
    bytes_per_row = [_peak_scratch_bytes(n, rng) / n for n in sizes]

    # Independent upper bound on the per-row coefficient. ``values`` is passed in
    # as float64, so the float64 cast inside the kernel is a no-copy view. The
    # traced peak is the argsort index (int64, 8 bytes/row), the canonical id
    # copy (int64, 8 bytes/row), the fancy-indexed float64 value copy
    # (8 * _WIDTH bytes/row), plus np.unique's transient unique+counts int64
    # pair (up to 16 bytes/row) that coexists with them: 8 + 8 + 8 * _WIDTH +
    # 16 == 96 bytes/row for _WIDTH == 8, matching the ~96 measured. The 1.2x
    # slack trips on a retained float32 copy of the values (+4 * _WIDTH == 32
    # bytes/row -> ~128 > 115), not only a retained float64 copy (+64).
    # Guard the calibration: the band must sit above the hand-counted healthy
    # coefficient yet below it plus a 32-byte float32 retention, or the gate
    # stops catching sub-float64 scratch bloat.
    ceiling = _SCRATCH_SLACK * _SCRATCH_PER_ROW
    assert _SCRATCH_PER_ROW < ceiling < _SCRATCH_PER_ROW + 4 * _WIDTH, (
        f"scratch band {ceiling:.0f} is miscalibrated against the"
        f" {_SCRATCH_PER_ROW}-byte coefficient and a +{4 * _WIDTH}-byte float32"
        " retention: loosening it past this range reopens the coverage gap"
    )
    for n, per_row in zip(sizes, bytes_per_row):
        assert per_row < ceiling, (
            f"canonical_sum used {per_row:.0f} bytes/row of scratch at n={n}"
            f" (ceiling {ceiling:.0f}): auxiliary footprint is not O(1)/row"
        )

    # O(1)/row means the coefficient is flat across the span. A footprint that
    # grew with n (e.g. an O(n^2) buffer) would inflate the largest size.
    spread = max(bytes_per_row) / min(bytes_per_row)
    assert spread < 1.5, (
        f"canonical_sum scratch bytes/row varied {spread:.2f}x across the"
        f" size span ({[round(b) for b in bytes_per_row]}): the per-row"
        " coefficient is growing with n, not constant"
    )


#: Fixed window row count for the windowed-branch timing check. Shrinking the
#: memory budget to this many rows forces even a modest span through the
#: while-loop (the default 932067-row window needs >1.2M rows). Held fixed as n
#: grows so the number of windows scales with n: a per-window full-array reduce
#: then costs O(windows * n) == O(n^2), tripping the exponent gate; a healthy
#: per-window slice reduce keeps total block work at O(n).
_WINDOWED_ROWS = 1_000
#: Base and 8x span for the windowed growth check; both past _WINDOWED_ROWS so
#: both take the while-loop. 8x n means 8x windows.
_WINDOWED_BASE = 120_000
_WINDOWED_SPAN = 8
#: Exponent ceiling for the windowed branch. A healthy windowed reduce grows
#: ~log-linearly (measured exponent ~1.15); a per-window full reduce grows
#: ~quadratically (~1.95). 1.5 rejects the quadratic order change with headroom.
_WINDOWED_EXPONENT_CEILING = 1.5
#: Sharp ceiling on canonical / honest-windowed-reference at the large point.
#: The reference below runs the SAME block structure with an honest per-block
#: slice reduce, so healthy canonical sits just above it (measured 1.18-1.21x);
#: a per-window full reduce blows the ratio to ~35-40x. 1.5 trips on that cliff
#: while clearing healthy with margin.
_WINDOWED_REFERENCE_CEILING = 1.5


def _honest_windowed_reference(
    vals: np.ndarray, ids: np.ndarray, window_rows: int
) -> np.ndarray:
    """Independent memory-bounded windowed sum: sort by id, then walk id-range
    blocks ``[lo, lo + window_rows)`` from the minimum id, reducing each block's
    slice once. Pays the same per-block work the kernel should, so a healthy
    kernel sits just above it and a per-window full-array reduce (O(windows * n))
    stands out as an order-of-magnitude cliff."""
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
    # The only other test touching the windowed while-loop (the O(1)/row memory
    # check) times nothing, so a quadratic/cliff regression inside the loop —
    # exactly the throughput cliff the file forbids — otherwise survives. Shrink
    # the window budget to force the loop, then gate the same way the
    # non-windowed growth test does: a fitted exponent (catches a super-linear
    # order change) plus a ratio over an independent honest windowed reference
    # (catches a per-window constant multiplier the exponent may not see cleanly).
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
            lambda: _honest_windowed_reference(vals, ids, window_rows)
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
        f"canonical_sum ran {sort_overhead:.2f}x an honest windowed block-loop"
        f" at n={_WINDOWED_SPAN * _WINDOWED_BASE}"
        f" (ceiling {_WINDOWED_REFERENCE_CEILING}x): the windowed branch is doing"
        " more than one reduce per block — a constant-multiplier cliff on the"
        " memory-bounded path"
    )

    # Timing-free signal for the memory-bounded path. The per-block slice reduce
    # is tiny next to the shared argsort, so a per-block constant multiplier
    # (5 reduces per block) or a per-window full-array reduce that keeps the sum
    # bitwise-identical stays under the reference ceiling above. Pin the whole
    # reduce schedule instead: record the operand shape of every reduce the
    # kernel issues — in any phrasing, np.add.reduce or a per-block
    # slice.sum(axis=0) via umr_sum — and require the exact sequence of block
    # row counts an independent sort+searchsorted walk produces. One honest
    # reduce per non-empty block; the row counts partition n. Five reduces per
    # block gives 5x the calls; a per-window full-array reduce makes each operand
    # n rows so the recorded counts stop summing to n.
    oracle_n = _WINDOWED_SPAN * _WINDOWED_BASE
    oracle_ids = np.random.default_rng(20260616001).permutation(oracle_n).astype(
        np.int64
    )
    oracle_vals = np.random.default_rng(20260616002).standard_normal(
        (oracle_n, _WIDTH)
    )
    expected_blocks = _expected_window_block_rows(oracle_ids, window_rows)
    reduce_shapes = _record_reduce_operands(
        lambda: canonical_sum(oracle_vals, oracle_ids)
    )
    assert all(shape[1:] == (_WIDTH,) for shape in reduce_shapes), (
        f"canonical_sum reduced non-({_WIDTH},)-wide operands {reduce_shapes}:"
        " the windowed branch must reduce (block_rows, WIDTH) slices"
    )
    block_rows = [int(shape[0]) for shape in reduce_shapes]
    assert block_rows == expected_blocks, (
        f"canonical_sum's windowed reduce schedule ({len(block_rows)} calls,"
        f" {sum(block_rows)} total rows) does not match the honest partition"
        f" ({len(expected_blocks)} blocks, {sum(expected_blocks)} rows == n):"
        " the memory-bounded loop is reducing more than once per block or over"
        " more than the block slice"
    )
