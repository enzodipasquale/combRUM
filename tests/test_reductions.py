from __future__ import annotations

import numpy as np
import pytest

import combrum.reductions as reductions
from _support.commprobe import spread_values
from combrum.reductions import canonical_sum, canonical_sum_window_rows


def float_bytes(x: object) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def expected_windowed_sum(
    values: np.ndarray, ids: np.ndarray, window_rows: int
) -> np.ndarray:
    """Plain-Python windowed sum -- sort by id, sum id-range blocks
    ``[lo, lo + window_rows)`` starting from the minimum id, accumulate
    block totals."""
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


def ascending_reduce(
    values: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Reduce rows in ascending id order -- the summation order of the fast
    (span <= window) path."""
    vals = np.asarray(values, dtype=np.float64)
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    return np.add.reduce(vals[order], axis=0)


def block_material_values(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    # Magnitudes over ~6 orders: wide enough that block-summation order moves
    # the low bits, bounded so every block stays material relative to the
    # running total (no single dominant term).
    magnitude = rng.uniform(0.0, 6.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


def test_scalar_value_and_type() -> None:
    result = canonical_sum(np.array([1.0, 2.0, 4.0]), np.array([2, 0, 1]))
    # numpy.float64 subclasses float, so isinstance would miss a dropped
    # float() coercion.
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
    # Strictly ascending ids skip the argsort and reduce rows as-is, so the
    # reduce order on that branch needs its own bytewise check: permutation
    # invariance cannot see a reversed reduce there.
    rng = np.random.default_rng(20260612)
    n = 257
    ids = np.arange(n)  # strictly ascending -> sorted_unique fast branch
    assert bool(np.all(ids[1:] > ids[:-1]))
    span = int(ids.max()) - int(ids.min()) + 1
    assert span <= canonical_sum_window_rows(1)  # fast path, not windowed
    values = spread_values(rng, (n,))

    # ascending and reversed accumulation disagree on this data
    ascending = np.add.reduce(values, axis=0)
    descending = np.add.reduce(values[::-1], axis=0)
    assert float_bytes(ascending) != float_bytes(descending)

    assert float_bytes(canonical_sum(values, ids)) == float_bytes(
        ascending_reduce(values, ids)
    )


def test_fast_path_ascending_order_matrix() -> None:
    # Matrix analogue: ascending reduce order per column.
    rng = np.random.default_rng(987)
    n, m = 300, 5
    ids = np.arange(n)  # strictly ascending -> sorted_unique fast branch
    assert bool(np.all(ids[1:] > ids[:-1]))
    span = int(ids.max()) - int(ids.min()) + 1
    assert span <= canonical_sum_window_rows(m)  # fast path, not windowed
    values = spread_values(rng, (n, m))

    ascending = np.add.reduce(values, axis=0)
    descending = np.add.reduce(values[::-1], axis=0)
    # accumulation order matters in every column on this data
    assert bool(np.all(ascending != descending))
    assert float_bytes(ascending) != float_bytes(descending)

    result = canonical_sum(values, ids)
    assert result.shape == (m,)
    assert float_bytes(result) == float_bytes(
        ascending_reduce(values, ids)
    )


@pytest.fixture()
def tiny_window(monkeypatch: pytest.MonkeyPatch):
    """Shrink the memory window so modest id spans take the windowed branch.

    Returns a callable that sets ``window_rows`` to an exact value for a
    given contribution width -- the byte budget inverts
    ``canonical_sum_window_rows``: ``rows == bytes // (8 * (width + 1))``.
    """

    def _patch(window_rows: int, width: int = 1) -> None:
        monkeypatch.setattr(
            reductions,
            "_CANONICAL_SUM_WINDOW_BYTES",
            8 * (width + 1) * window_rows,
        )

    return _patch


def test_windowed_path_matches_exact_sum_scalar(tiny_window) -> None:
    # Many multi-row blocks through the windowed branch, compared bytewise
    # against the plain-Python windowed sum at the same window_rows.
    tiny_window(4)
    rng = np.random.default_rng(0)
    n = 257
    ids = rng.permutation(n).astype(np.int64)
    span = int(ids.max()) - int(ids.min()) + 1
    window_rows = canonical_sum_window_rows(1)
    assert window_rows > 1  # multi-row blocks, not one id per block
    assert span > window_rows  # windowed branch, not the fast path
    assert span // window_rows >= 8  # many blocks
    assert (span - 1) % window_rows == 0  # last id lands on a block boundary

    values = block_material_values(rng, (n,))
    result = canonical_sum(values, ids)
    assert float_bytes(result) == float_bytes(
        expected_windowed_sum(values, ids, window_rows)
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
        expected_windowed_sum(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_path_large_real_id_span_scalar() -> None:
    # The real default window, no monkeypatch: two ids in the first window,
    # two in a later one, so each block spans multiple rows and the
    # cancellation is confined to a block.
    window_rows = canonical_sum_window_rows(1)
    ids = np.array([0, 1, 2 * window_rows + 50, 2 * window_rows + 51], dtype=np.int64)
    assert int(ids.max()) - int(ids.min()) + 1 > window_rows

    # Canonical (ascending-id) values are [1e16, 1.0, -1e16, 2.5]. The windowed
    # branch groups {1e16, 1.0} then {-1e16, 2.5}. Block totals in float64
    # (ULP at 1e16 is 2): (1e16 + 1.0) == 1e16 (the 1.0 is absorbed);
    # (-1e16 + 2.5) == -9999999999999998.0 (2.5 rounds down by one ULP);
    # their sum is 2.0. A whole-array reduce would yield 2.5.
    values = np.array([1e16, 1.0, -1e16, 2.5])
    expected = (1e16 + 1.0) + (-1e16 + 2.5)
    assert expected == 2.0
    assert float_bytes(canonical_sum(values, ids)) == float_bytes(expected)

    # permutation invariance still holds on the windowed branch
    reference = float_bytes(canonical_sum(values, ids))
    rng = np.random.default_rng(4)
    for _ in range(20):
        perm = rng.permutation(ids.size)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def test_windowed_origin_anchored_at_min_id_scalar(tiny_window) -> None:
    # Same construction as the scalar windowed test, but the minimum id is
    # strictly positive and not a multiple of window_rows: a first window
    # anchored at the min id groups ids differently than one anchored at
    # absolute 0.
    tiny_window(4)
    rng = np.random.default_rng(0)
    n = 257
    window_rows = canonical_sum_window_rows(1)
    offset = 2 * window_rows + 1  # min id % window_rows == 1, strictly > 0
    ids = rng.permutation(n).astype(np.int64) + offset
    span = int(ids.max()) - int(ids.min()) + 1
    assert window_rows > 1
    assert span > window_rows
    assert span // window_rows >= 8
    assert int(ids.min()) % window_rows != 0  # origin at 0 vs min id diverges

    values = block_material_values(rng, (n,))
    result = canonical_sum(values, ids)
    assert float_bytes(result) == float_bytes(
        expected_windowed_sum(values, ids, window_rows)
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
        expected_windowed_sum(values, ids, window_rows)
    )

    reference = float_bytes(result)
    for _ in range(20):
        perm = rng.permutation(n)
        assert float_bytes(canonical_sum(values[perm], ids[perm])) == reference


def window_rows_ceiling(budget_bytes: int, width: int) -> int:
    """Expected ``canonical_sum_window_rows``, derived from the memory layout.

    A window row carries ``width`` float64 value columns plus one float64 id
    column; the per-row cost is read off a one-row buffer via ``numpy.nbytes``
    rather than the kernel's ``8 * (width + 1)`` expression, so this is not
    the code under test spelled backwards.
    """
    value_cols = np.empty((1, max(1, int(width))), dtype=np.float64)
    id_col = np.empty((1,), dtype=np.float64)
    per_row = value_cols.nbytes + id_col.nbytes
    return max(1, budget_bytes // per_row)


def test_window_rows_width_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The windowed tests hand canonical_sum_window_rows's own value to the
    # reference sum, so a wrong width scaling would cancel out of those byte
    # comparisons. Here the expected value comes from the measured memory
    # layout instead, which distinguishes (width + 1) from e.g. (2 * width)
    # -- identical at width 1, wrong above it.
    for width, rows in ((3, 100), (7, 250), (5, 137)):
        # Exact-fit budget sized from the measured per-row cost, so the
        # floor is unambiguous.
        one_row_bytes = (
            np.empty((1, width), dtype=np.float64).nbytes
            + np.empty((1,), dtype=np.float64).nbytes
        )
        budget = rows * one_row_bytes
        monkeypatch.setattr(
            reductions, "_CANONICAL_SUM_WINDOW_BYTES", budget
        )
        expected = window_rows_ceiling(budget, width)
        # The exact fit recovers the chosen row count, and the wrong
        # (2 * width) scaling gives a strictly smaller one.
        assert expected == rows
        wrong_scaling = max(1, budget // (8 * (2 * width)))
        assert wrong_scaling < expected
        assert canonical_sum_window_rows(width) == expected


def test_window_rows_clamps_to_one_below_row_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Below one row's byte cost the unclamped floor is 0, and canonical_sum's
    # windowed loop would advance by hi = lo + 0 and never terminate; the
    # outer max(1, ...) keeps window_rows >= 1.
    for width in (1, 3, 7):
        one_row_bytes = 8 * (width + 1)
        # Budgets strictly below one row's cost: the unclamped floor is 0.
        for budget in (1, one_row_bytes - 1):
            monkeypatch.setattr(
                reductions, "_CANONICAL_SUM_WINDOW_BYTES", budget
            )
            assert budget // one_row_bytes == 0  # unclamped floor would be 0
            assert canonical_sum_window_rows(width) == 1

    # window_rows == 1 with a larger span: blocks of one id, and the loop
    # must still terminate with the whole-array reduce.
    monkeypatch.setattr(reductions, "_CANONICAL_SUM_WINDOW_BYTES", 1)
    assert canonical_sum_window_rows(1) == 1
    ids = np.arange(6, dtype=np.int64)  # span 6 > window_rows 1
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    assert float_bytes(canonical_sum(values, ids)) == float_bytes(
        np.add.reduce(values, axis=0)
    )


def test_window_rows_width_clamp_at_nonpositive_width() -> None:
    # w = max(1, int(width)) treats width <= 0 as width 1; unclamped,
    # width 0 would double the row budget and width -1 would divide by zero.
    baseline = canonical_sum_window_rows(1)
    assert canonical_sum_window_rows(0) == baseline
    assert canonical_sum_window_rows(-5) == baseline


def test_windowed_origin_min_id_hand_derived(tiny_window) -> None:
    # Window origin, values worked by hand. window_rows == 4, ids [2, 3, 5, 6]
    # (min id 2, not a multiple of 4): the first window starts at 2, so block
    # [2, 6) holds {2, 3, 5} and block [6, 10) holds {6}. Canonical values
    # [1e16, 1.0, -1e16, 2.5]:
    #   block1 = (1e16 + 1.0) + (-1e16) = 0.0  (the 1.0 is absorbed at 1e16)
    #   block2 = 2.5 ; total = 2.5.
    # Windows anchored at absolute 0 would group {2, 3} and {5, 6} instead,
    # giving (1e16 + 1.0) + (-1e16 + 2.5) = 2.0.
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
    # Order-dependent addition disagrees on this data (1.0 is absorbed into
    # 1e16 in one ordering); canonical_sum agrees regardless of row order.
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
