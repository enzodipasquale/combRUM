from __future__ import annotations

import numpy as np
import pytest

import combrum.reductions as reductions
from combrum.reductions import canonical_sum, canonical_sum_window_rows


def float_bytes(x: object) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def windowed_oracle(
    values: np.ndarray, ids: np.ndarray, window_rows: int
) -> np.ndarray:
    """Independent windowed sum: a plain-Python block loop keyed to window_rows.

    Deliberately structured differently from the numpy-vectorized kernel: it
    sorts by id, walks id-range blocks ``[lo, lo + window_rows)`` starting from
    the minimum id, sums each block's rows element by element in Python, then
    accumulates block totals with ``+=``. Pinning ``float_bytes`` against this
    oracle fixes the exact block boundaries, so removing the windowed branch
    (whole-array reduce) or shifting a boundary produces different bytes.
    """
    vals = np.asarray(values, dtype=np.float64)
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    sids = [int(x) for x in ids[order].tolist()]
    svals = vals[order]
    ndim1 = vals.ndim == 1
    width = 1 if ndim1 else int(vals.shape[1])
    rows = (
        [[float(v)] for v in svals.tolist()]
        if ndim1
        else [[float(x) for x in r] for r in svals.tolist()]
    )
    total = [0.0] * width
    if not sids:
        return np.array(total, dtype=np.float64)
    lo = sids[0]
    stop = sids[-1] + 1
    left = 0
    n = len(sids)
    while lo < stop:
        hi = lo + window_rows
        right = left
        while right < n and sids[right] < hi:
            right += 1
        if right > left:
            block = [0.0] * width
            for r in range(left, right):
                row = rows[r]
                for c in range(width):
                    block[c] = block[c] + row[c]
            for c in range(width):
                total[c] = total[c] + block[c]
        left = right
        lo = hi
    return np.array(total, dtype=np.float64)


def ascending_reduce_oracle(
    values: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Independent ascending-id reduce for the fast (span <= window) path.

    Sorts by id with its own ``argsort`` and reduces the sorted rows in
    ascending order. The independence is in the ordering: it never reads the
    kernel's canonical array, it re-derives ascending order from the ids. On
    magnitude-spread data the ascending reduce differs bitwise from any
    reversed or reordered reduce, so pinning ``float_bytes`` against this
    fixes the fast-path summation order at ascending specifically -- not merely
    "some consistent order".
    """
    vals = np.asarray(values, dtype=np.float64)
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    return np.add.reduce(vals[order], axis=0)


def spread_values(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    # Magnitudes across many orders of magnitude: the regime where addition
    # order visibly moves the result, giving the bitwise assertions a real signal.
    magnitude = rng.uniform(-10.0, 10.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


def block_material_values(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    # Magnitudes over ~6 orders: wide enough that block-summation order moves
    # the low bits (windowed output differs from a whole-array reduce), yet
    # bounded so every block stays material relative to the running total.
    # A dropped or misaccumulated block therefore shifts the bytes rather than
    # vanishing under a single dominant term.
    magnitude = rng.uniform(0.0, 6.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


def test_scalar_value_and_type() -> None:
    result = canonical_sum(np.array([1.0, 2.0, 4.0]), np.array([2, 0, 1]))
    # Exact type, not isinstance: numpy.float64 is a float subclass, so the
    # docstring's "Returns float for (n,) input" contract only bites if the
    # float(...) coercion is actually applied.
    assert type(result) is float
    assert result == 7.0


def test_matrix_shape_dtype_and_values() -> None:
    values = np.arange(6.0).reshape(3, 2)
    result = canonical_sum(values, np.array([2, 0, 1]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.array([6.0, 9.0]))


def test_integer_input_coerced_to_float64() -> None:
    result = canonical_sum(np.array([1, 2, 3]), np.array([0, 1, 2]))
    assert type(result) is float
    assert result == 6.0
    matrix = canonical_sum(
        np.array([[1, 2], [3, 4]]), np.array([1, 0])
    )
    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == np.float64


def test_permutation_invariance_scalar() -> None:
    # Fast (single-shot np.add.reduce) path: span stays well under window_rows.
    rng = np.random.default_rng(20260612)
    n = 257
    ids = rng.permutation(n)
    assert int(ids.max()) - int(ids.min()) + 1 <= canonical_sum_window_rows(1)
    values = spread_values(rng, (n,))
    reference = float_bytes(canonical_sum(values, ids))
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_permutation_invariance_matrix() -> None:
    # Fast path, width>1: span stays under the (smaller) width-5 window_rows.
    rng = np.random.default_rng(987)
    n, m = 300, 5
    ids = rng.permutation(n)
    assert int(ids.max()) - int(ids.min()) + 1 <= canonical_sum_window_rows(m)
    values = spread_values(rng, (n, m))
    reference = float_bytes(canonical_sum(values, ids))
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_fast_path_ascending_order_scalar() -> None:
    # Fast path (span <= window_rows) with magnitude-spread values, pinned
    # bytewise against an independent ascending-id reduce. The ids are strictly
    # ascending so canonical_sum takes the sorted_unique fast branch (src
    # reductions.py lines 61-63) that skips argsort and reduces the input rows
    # as-is -- exactly the already-sorted single-window case. The invariance
    # tests only fix that every permutation agrees with a self-reference, so a
    # regression reversing the fast-branch reduce (e.g. reducing canonical[::-1]
    # when ids arrive strictly ascending) stays permutation-invariant and
    # survives them. This pins the order at ascending: a reversed fast-branch
    # reduce differs by at least one ULP on this data.
    rng = np.random.default_rng(20260612)
    n = 257
    ids = np.arange(n)  # strictly ascending -> sorted_unique fast branch
    assert bool(np.all(ids[1:] > ids[:-1]))  # guarantees the fast branch
    span = int(ids.max()) - int(ids.min()) + 1
    assert span <= canonical_sum_window_rows(1)  # fast path, not windowed
    values = spread_values(rng, (n,))

    # Self-check on the SAME rows the fast branch reduces (ids already
    # ascending): ascending vs reversed accumulation genuinely disagree here,
    # so the byte pin below has a meaningful signal (mirrors test_naive_summation_order).
    ascending = np.add.reduce(values, axis=0)
    descending = np.add.reduce(values[::-1], axis=0)
    assert float_bytes(ascending) != float_bytes(descending)

    assert float_bytes(canonical_sum(values, ids)) == float_bytes(
        ascending_reduce_oracle(values, ids)
    )


def test_fast_path_ascending_order_matrix() -> None:
    # Matrix analogue of the fast-path ascending pin: strictly ascending ids
    # drive the sorted_unique fast branch, and each column's reduce must follow
    # ascending id order, caught bytewise against the independent oracle.
    rng = np.random.default_rng(987)
    n, m = 300, 5
    ids = np.arange(n)  # strictly ascending -> sorted_unique fast branch
    assert bool(np.all(ids[1:] > ids[:-1]))  # guarantees the fast branch
    span = int(ids.max()) - int(ids.min()) + 1
    assert span <= canonical_sum_window_rows(m)  # fast path, not windowed
    values = spread_values(rng, (n, m))

    ascending = np.add.reduce(values, axis=0)
    descending = np.add.reduce(values[::-1], axis=0)
    # Every column's accumulation order matters on this data, so a reversed
    # fast-branch reduce shifts all columns, not just one.
    assert bool(np.all(ascending != descending))
    assert float_bytes(ascending) != float_bytes(descending)

    result = canonical_sum(values, ids)
    assert result.shape == (m,)
    assert float_bytes(result) == float_bytes(
        ascending_reduce_oracle(values, ids)
    )


@pytest.fixture()
def tiny_window(monkeypatch: pytest.MonkeyPatch):
    """Shrink the memory window so modest id spans take the windowed branch.

    Returns a callable that pins ``window_rows`` to an exact value for a given
    contribution width, so tests can force multi-row blocks (window_rows > 1)
    and assert the loop steps through many blocks rather than one. The byte
    budget inverts ``canonical_sum_window_rows`` exactly:
    ``rows == bytes // (8 * (width + 1))``.
    """

    def _patch(window_rows: int, width: int = 1) -> None:
        monkeypatch.setattr(
            reductions,
            "_CANONICAL_SUM_WINDOW_BYTES",
            8 * (width + 1) * window_rows,
        )

    return _patch


def test_windowed_path_matches_exact_sum_scalar(tiny_window) -> None:
    # Force the memory-bounded windowing branch (src reductions.py lines 82-93)
    # with many multi-row blocks, then pin the exact bytes against an
    # independent block-loop oracle keyed to the same window_rows. Spread
    # magnitudes make addition order matter, so the windowed block boundaries
    # produce bytes distinct from a whole-array reduce or a shifted boundary.
    tiny_window(4)
    rng = np.random.default_rng(0)
    n = 257
    ids = rng.permutation(n).astype(np.int64)
    span = int(ids.max()) - int(ids.min()) + 1
    window_rows = canonical_sum_window_rows(1)
    assert window_rows > 1  # blocks span multiple rows, not one id per block
    assert span > window_rows  # windowed branch, not the fast path
    assert span // window_rows >= 8  # exercised across many blocks
    assert (span - 1) % window_rows == 0  # last id lands on a block boundary

    values = block_material_values(rng, (n,))
    result = canonical_sum(values, ids)
    assert float_bytes(result) == float_bytes(
        windowed_oracle(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_path_matches_exact_sum_matrix(tiny_window) -> None:
    n, m = 301, 5
    tiny_window(4, width=m)
    rng = np.random.default_rng(0)
    ids = rng.permutation(n).astype(np.int64)
    span = int(ids.max()) - int(ids.min()) + 1
    window_rows = canonical_sum_window_rows(m)
    assert window_rows > 1  # blocks span multiple rows across all columns
    assert span > window_rows
    assert span // window_rows >= 8
    assert (span - 1) % window_rows == 0  # last id lands on a block boundary

    values = block_material_values(rng, (n, m))
    result = canonical_sum(values, ids)
    assert result.shape == (m,)
    assert float_bytes(result) == float_bytes(
        windowed_oracle(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_path_large_real_id_span_scalar() -> None:
    # Trigger the windowed branch with the real default window (no monkeypatch).
    # Two ids sit inside the first window and two inside a later window, so each
    # block spans multiple rows and the cancellation is confined to a block.
    window_rows = canonical_sum_window_rows(1)
    ids = np.array([0, 1, 2 * window_rows + 50, 2 * window_rows + 51], dtype=np.int64)
    assert int(ids.max()) - int(ids.min()) + 1 > window_rows

    # Canonical (ascending-id) values are [1e16, 1.0, -1e16, 2.5]. The windowed
    # branch groups {1e16, 1.0} then {-1e16, 2.5}. Hand-derived block totals in
    # float64 (ULP at 1e16 is 2): (1e16 + 1.0) == 1e16 (the 1.0 is absorbed);
    # (-1e16 + 2.5) == -9999999999999998.0 (2.5 rounds down by one ULP); their
    # sum is 2.0. A whole-array reduce instead yields 2.5, so this value pins
    # the windowed block boundaries, not merely any correct accumulation.
    values = np.array([1e16, 1.0, -1e16, 2.5])
    expected = (1e16 + 1.0) + (-1e16 + 2.5)
    assert expected == 2.0
    assert float_bytes(canonical_sum(values, ids)) == float_bytes(expected)

    # Permutation invariance must still hold across the windowed branch.
    reference = float_bytes(canonical_sum(values, ids))
    rng = np.random.default_rng(4)
    for _ in range(20):
        perm = rng.permutation(ids.size)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_origin_anchored_at_min_id_scalar(tiny_window) -> None:
    # Same construction as the scalar windowed test, but the ids start at a
    # strictly positive minimum that is NOT a multiple of window_rows. The
    # window origin (src anchors the first block at min id, not absolute 0)
    # then determines which ids share a block. Pinning against the oracle,
    # which also anchors at the min id, kills a src bug that starts windows at
    # absolute 0: that regroups every block and shifts the bytes.
    tiny_window(4)
    rng = np.random.default_rng(0)
    n = 257
    window_rows = canonical_sum_window_rows(1)
    offset = 2 * window_rows + 1  # min id % window_rows == 1, strictly > 0
    ids = rng.permutation(n).astype(np.int64) + offset
    span = int(ids.max()) - int(ids.min()) + 1
    assert window_rows > 1
    assert span > window_rows  # windowed branch, not the fast path
    assert span // window_rows >= 8  # many blocks
    assert int(ids.min()) % window_rows != 0  # origin at 0 vs min id diverges

    values = block_material_values(rng, (n,))
    result = canonical_sum(values, ids)
    assert float_bytes(result) == float_bytes(
        windowed_oracle(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_origin_anchored_at_min_id_matrix(tiny_window) -> None:
    n, m = 301, 5
    tiny_window(4, width=m)
    rng = np.random.default_rng(0)
    window_rows = canonical_sum_window_rows(m)
    offset = 2 * window_rows + 1  # min id % window_rows == 1, strictly > 0
    ids = rng.permutation(n).astype(np.int64) + offset
    span = int(ids.max()) - int(ids.min()) + 1
    assert window_rows > 1
    assert span > window_rows
    assert span // window_rows >= 8
    assert int(ids.min()) % window_rows != 0

    values = block_material_values(rng, (n, m))
    result = canonical_sum(values, ids)
    assert result.shape == (m,)
    assert float_bytes(result) == float_bytes(
        windowed_oracle(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def window_rows_ceiling(budget_bytes: int, width: int) -> int:
    """Independent, layout-driven oracle for ``canonical_sum_window_rows``.

    Derives the per-row cost by measuring the byte size of a real window
    buffer instead of reusing the kernel's ``8 * (width + 1)`` expression: a
    window holds ``rows`` entries, each carrying ``width`` float64 value
    columns plus one float64 id column. Building a one-row buffer and reading
    ``numpy.nbytes`` gives the per-row cost from the memory layout itself, so
    the largest number of rows fitting ``budget_bytes`` is a distinct
    invariant, not the code under test spelled backwards.
    """
    value_cols = np.empty((1, max(1, int(width))), dtype=np.float64)
    id_col = np.empty((1,), dtype=np.float64)
    per_row = value_cols.nbytes + id_col.nbytes
    return max(1, budget_bytes // per_row)


def test_window_rows_width_scaling_layout_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the width->window_rows scaling directly. The windowed tests read
    # window_rows from canonical_sum_window_rows and hand the SAME value to the
    # oracle, so a wrong width scaling cancels out of the byte comparison. Here
    # the expected window_rows is derived independently from the memory layout
    # (value columns + one id column, measured via numpy.nbytes), so shrinking
    # the (width+1) term to (2*width) -- a no-op at width 1 but wrong for
    # width>1 -- is caught.
    for width, rows in ((3, 100), (7, 250), (5, 137)):
        # Exact-fit budget so the floor is unambiguous, sized from the layout
        # oracle's per-row cost (value columns + one id column) rather than the
        # kernel's own expression.
        one_row_bytes = (
            np.empty((1, width), dtype=np.float64).nbytes
            + np.empty((1,), dtype=np.float64).nbytes
        )
        budget = rows * one_row_bytes
        monkeypatch.setattr(
            reductions, "_CANONICAL_SUM_WINDOW_BYTES", budget
        )
        expected = window_rows_ceiling(budget, width)
        # Self-check: the exact-fit construction recovers the chosen row count,
        # and the oracle genuinely separates the correct scaling from the
        # (2*width) regression -- otherwise this pin would have no meaningful signal.
        assert expected == rows
        mutated = max(1, budget // (8 * (2 * width)))
        assert mutated < expected
        assert canonical_sum_window_rows(width) == expected


def test_window_rows_clamps_to_one_below_row_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The outer max(1, ...) guard on canonical_sum_window_rows keeps window_rows
    # >= 1 when the byte budget is smaller than a single row's cost. Without it
    # the floor is 0, and canonical_sum's windowed loop advances by hi = lo + 0,
    # so lo never moves and the sum hangs. No other test drives the budget below
    # one row, so this pins the degenerate boundary directly.
    for width in (1, 3, 7):
        one_row_bytes = 8 * (width + 1)
        # Budgets strictly below one row's cost: the unclamped floor is 0.
        for budget in (1, one_row_bytes - 1):
            monkeypatch.setattr(
                reductions, "_CANONICAL_SUM_WINDOW_BYTES", budget
            )
            assert budget // one_row_bytes == 0  # unclamped floor would be 0
            assert canonical_sum_window_rows(width) == 1

    # canonical_sum must still terminate and return the whole-array reduce when
    # window_rows == 1 but the id span exceeds it (windowed branch, blocks of
    # one id). A dropped outer clamp would make this hang instead.
    monkeypatch.setattr(reductions, "_CANONICAL_SUM_WINDOW_BYTES", 1)
    assert canonical_sum_window_rows(1) == 1
    ids = np.arange(6, dtype=np.int64)  # span 6 > window_rows 1
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    assert float_bytes(canonical_sum(values, ids)) == float_bytes(
        np.add.reduce(values, axis=0)
    )


def test_window_rows_width_clamp_at_nonpositive_width() -> None:
    # The inner w = max(1, int(width)) guard treats width <= 0 as width 1.
    # Without it, width 0 would divide by (0 + 1) and width -1 by (-1 + 1) == 0
    # (a ZeroDivisionError). Pin the clamp: non-positive widths match width 1.
    baseline = canonical_sum_window_rows(1)
    assert canonical_sum_window_rows(0) == baseline
    assert canonical_sum_window_rows(-5) == baseline


def test_windowed_origin_min_id_hand_derived(tiny_window) -> None:
    # Sharp, oracle-free pin of the window origin. With window_rows == 4 and
    # ids [2, 3, 5, 6] (min id 2, not a multiple of 4), src anchors the first
    # window at 2: block [2, 6) holds ids {2, 3, 5}, block [6, 10) holds {6}.
    # Canonical values are [1e16, 1.0, -1e16, 2.5]:
    #   block1 = (1e16 + 1.0) + (-1e16) = 0.0  (the 1.0 is absorbed at 1e16)
    #   block2 = 2.5 ; total = 2.5.
    # Anchoring windows at absolute 0 instead would group [0, 4) -> {2, 3} and
    # [4, 8) -> {5, 6}, giving (1e16 + 1.0) + (-1e16 + 2.5) = 2.0. The value 2.5
    # therefore pins the origin, not merely a correct accumulation.
    tiny_window(4)
    window_rows = canonical_sum_window_rows(1)
    ids = np.array([2, 3, 5, 6], dtype=np.int64)
    span = int(ids.max()) - int(ids.min()) + 1
    assert span > window_rows  # windowed branch
    assert int(ids.min()) % window_rows != 0

    values = np.array([1e16, 1.0, -1e16, 2.5])
    expected = ((1e16 + 1.0) + (-1e16)) + 2.5
    assert expected == 2.5
    assert float_bytes(canonical_sum(values, ids)) == float_bytes(expected)

    reference = float_bytes(canonical_sum(values, ids))
    rng = np.random.default_rng(11)
    for _ in range(20):
        perm = rng.permutation(ids.size)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_naive_summation_order_disagrees() -> None:
    # Naive order-dependent addition genuinely disagrees here (1.0 is absorbed
    # into 1e16 in one ordering), so the invariance assertions above are exercised.
    a = np.array([1e16, 1.0, -1e16])
    b = np.array([1e16, -1e16, 1.0])
    assert float(np.add.reduce(a)) != float(np.add.reduce(b))
    ids_a = np.array([0, 1, 2])
    ids_b = np.array([0, 2, 1])
    assert float_bytes(canonical_sum(a, ids_a)) == float_bytes(canonical_sum(b, ids_b))


def test_duplicate_ids_rejected_and_error_lists_them() -> None:
    with pytest.raises(ValueError, match=r"\[1\]"):
        canonical_sum(np.ones(4), np.array([0, 1, 1, 3]))
    with pytest.raises(ValueError, match=r"\[2, 5\]"):
        canonical_sum(np.ones(4), np.array([5, 5, 2, 2]))


def test_shape_and_dtype_validation() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        canonical_sum(np.ones(4), np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="integer dtype"):
        canonical_sum(np.ones(3), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match=r"\(n,\) or \(n, M\)"):
        canonical_sum(np.ones((2, 2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="one id per contribution row"):
        canonical_sum(np.ones(3), np.array([0, 1]))


def test_empty_sum_is_additive_identity() -> None:
    empty_ids = np.array([], dtype=np.int64)
    scalar = canonical_sum(np.array([], dtype=np.float64), empty_ids)
    assert type(scalar) is float
    assert scalar == 0.0
    matrix = canonical_sum(np.empty((0, 4)), empty_ids)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4,)
    np.testing.assert_array_equal(matrix, np.zeros(4))
