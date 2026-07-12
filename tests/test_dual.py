from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from combrum.dual import CutDualRow, DualSolution
from combrum.dualstore import DualStoreReader, DualStoreWriter, equal


def make_dual(**overrides: object) -> DualSolution:
    kwargs: dict[str, object] = dict(
        rep_id=7,
        agent_ids=np.array([0, 1, 1], dtype=np.int64),
        bundle_row_ids=np.array([0, 1, 0], dtype=np.int64),
        pis=np.array([0.25, 0.5, 0.125]),
        bundle_table=np.array(
            [[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.25]]
        ),
        bound_duals={},
    )
    kwargs.update(overrides)
    return DualSolution(**kwargs)  # type: ignore[arg-type]


def test_parallel_array_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="parallel"):
        make_dual(pis=np.array([0.25, 0.5]))
    with pytest.raises(ValueError, match="parallel"):
        make_dual(bundle_row_ids=np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="parallel"):
        make_dual(agent_ids=np.array([0], dtype=np.int64))


def test_bundle_row_ids_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"rows in \[0, 2\)"):
        make_dual(bundle_row_ids=np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match=r"rows in \[0, 2\)"):
        make_dual(bundle_row_ids=np.array([-1, 0, 1], dtype=np.int64))


def test_nonfinite_pis_rejected() -> None:
    with pytest.raises(ValueError, match="pis must be finite"):
        make_dual(pis=np.array([0.25, np.nan, 0.125]))
    with pytest.raises(ValueError, match="pis must be finite"):
        make_dual(pis=np.array([0.25, np.inf, 0.125]))


def test_bound_duals_validation() -> None:
    with pytest.raises(ValueError, match=r"bound_duals\[3\] must be finite"):
        make_dual(bound_duals={3: float("nan")})
    with pytest.raises(ValueError, match=r"bound_duals\[3\] must be finite"):
        make_dual(bound_duals={3: float("inf")})
    with pytest.raises(ValueError, match="must be >= 0"):
        make_dual(bound_duals={-1: 0.5})


def test_negative_rep_id_and_agent_ids_rejected() -> None:
    with pytest.raises(ValueError, match="rep_id must be >= 0"):
        make_dual(rep_id=-1)
    with pytest.raises(ValueError, match="agent_ids must be >= 0"):
        make_dual(agent_ids=np.array([0, -2, 1], dtype=np.int64))


def test_float_rep_id_rejected_not_truncated() -> None:
    # rep_id goes through operator.index: a float must raise, never
    # int()-truncate (7.9 -> 7 would collide with rep 7). The constructor and
    # with_rep_id share the check.
    with pytest.raises(TypeError):
        make_dual(rep_id=7.9)
    with pytest.raises(TypeError):
        make_dual(rep_id=7.0)
    with pytest.raises(TypeError):
        make_dual(rep_id=np.float64(7.0))
    dual = make_dual()
    with pytest.raises(TypeError):
        dual.with_rep_id(11.9)
    with pytest.raises(TypeError):
        dual.with_rep_id(11.0)


def test_non_integer_id_arrays_rejected() -> None:
    # Integer dtype is required, not coerced: float ids would truncate silently.
    with pytest.raises(ValueError, match="expected a 1-D integer array for agent_ids"):
        make_dual(agent_ids=np.array([0.0, 1.0, 1.0]))
    with pytest.raises(
        ValueError, match="expected a 1-D integer array for bundle_row_ids"
    ):
        make_dual(bundle_row_ids=np.array([0.0, 1.0, 0.0]))


def test_bundle_table_must_be_2d() -> None:
    with pytest.raises(ValueError, match=r"expected a 2-D \(n_bundles, M\) bundle_table"):
        make_dual(bundle_table=np.zeros(4))
    with pytest.raises(ValueError, match=r"expected a 2-D \(n_bundles, M\) bundle_table"):
        make_dual(bundle_table=np.zeros((2, 2, 2)))


def test_payload_arrays_read_only() -> None:
    dual = make_dual()
    arrays = (dual.agent_ids, dual.bundle_row_ids, dual.pis, dual.bundle_table)
    for arr in arrays:
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[(0,) * arr.ndim] = 99


def test_payload_owns_its_data() -> None:
    # Construction must copy the caller's arrays, not alias them. A full-slice
    # view taken before construction stays writeable even after the payload
    # freezes its own buffer, so writing through it exposes shared storage.
    agent_ids = np.array([0, 1, 1], dtype=np.int64)
    pis = np.array([0.25, 0.5, 0.125])
    bundle_row_ids = np.array([0, 1, 0], dtype=np.int64)
    bundle_table = np.array([[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.25]])
    bound_duals: dict[int, float] = {2: 1.5}
    views = (
        (agent_ids[:], (0,), 99),
        (pis[:], (0,), 99.0),
        (bundle_row_ids[:], (0,), 1),
        (bundle_table[:], (0, 0), 99.0),
    )
    dual = make_dual(
        agent_ids=agent_ids,
        pis=pis,
        bundle_row_ids=bundle_row_ids,
        bundle_table=bundle_table,
        bound_duals=bound_duals,
    )
    for view, idx, value in views:
        # A frozen view means the caller array was aliased into the payload.
        if view.flags.writeable:
            view[idx] = value
    bound_duals[5] = -1.0
    assert dual.agent_ids[0] == 0
    assert dual.pis[0] == 0.25
    assert dual.bundle_row_ids[0] == 0
    assert dual.bundle_table[0, 0] == 1.0
    assert dual.bound_duals == {2: 1.5}


def test_with_rep_id_reuses_validated_payload_storage() -> None:
    dual = make_dual(bound_duals={0: 1.0, 2: -0.5})

    restamped = dual.with_rep_id(11)

    assert restamped.rep_id == 11
    assert restamped.agent_ids is dual.agent_ids
    assert restamped.bundle_row_ids is dual.bundle_row_ids
    assert restamped.pis is dual.pis
    assert restamped.bundle_table is dual.bundle_table
    assert restamped.bound_duals is dual.bound_duals
    assert not restamped.agent_ids.flags.writeable
    assert not restamped.bundle_row_ids.flags.writeable
    assert not restamped.pis.flags.writeable
    assert not restamped.bundle_table.flags.writeable
    with pytest.raises(TypeError):
        restamped.bound_duals[0] = 99.0  # type: ignore[index]
    # The reused arrays still price to the moment computed by hand.
    np.testing.assert_array_equal(
        restamped.moment(), np.array([0.375, 0.5, 1.25, 0.3125])
    )


def test_with_rep_id_rejects_invalid_rep_id() -> None:
    dual = make_dual()
    with pytest.raises(ValueError, match="rep_id must be >= 0"):
        dual.with_rep_id(-1)


def test_moment_matches_hand_computed_sum() -> None:
    dual = make_dual()
    # Rows priced: 0.25 * t0 + 0.5 * t1 + 0.125 * t0 = 0.375 t0 + 0.5 t1.
    # All literals are dyadic, so the comparison is exact in float64.
    expected = np.array([0.375, 0.5, 1.25, 0.3125])
    moment = dual.moment()
    assert moment.shape == (4,)
    assert moment.dtype == np.float64
    assert np.array_equal(moment, expected)


def test_moment_promotes_non_float_tables() -> None:
    dual = make_dual(
        agent_ids=np.array([0, 1], dtype=np.int64),
        bundle_row_ids=np.array([0, 1], dtype=np.int64),
        pis=np.array([0.5, 0.25]),
        bundle_table=np.array([[True, False], [True, True]]),
    )
    moment = dual.moment()
    assert moment.dtype == np.float64
    assert np.array_equal(moment, np.array([0.75, 0.25]))


def test_moment_empty_payload_is_zeros() -> None:
    empty = dict(
        agent_ids=np.array([], dtype=np.int64),
        bundle_row_ids=np.array([], dtype=np.int64),
        pis=np.array([], dtype=np.float64),
    )
    # The moment's width comes from the stored table even with no bundle rows.
    no_bundles = make_dual(bundle_table=np.empty((0, 4)), **empty)
    with_bundles = make_dual(**empty)
    for dual in (no_bundles, with_bundles):
        moment = dual.moment()
        assert moment.shape == (4,)
        assert moment.dtype == np.float64
        assert np.array_equal(moment, np.zeros(4))


def test_rows_decode_agent_ids_and_preserve_bundle_rows() -> None:
    dual = make_dual(
        agent_ids=np.array([0, 3, 4], dtype=np.int64),
        bundle_row_ids=np.array([0, 1, 0], dtype=np.int64),
        pis=np.array([0.25, 0.5, 0.125]),
    )

    rows = list(dual.rows(n_obs=2))

    assert all(isinstance(row, CutDualRow) for row in rows)
    assert [
        (row.agent_id, row.observation_id, row.simulation_id, row.pi)
        for row in rows
    ] == [(0, 0, 0, 0.25), (3, 1, 1, 0.5), (4, 0, 2, 0.125)]
    # bundle_row_ids == [0, 1, 0]: table row 0 decodes twice, so every decoded
    # row is compared against a literal table, not dual.bundle_table[...].
    expected_bundles = np.array(
        [[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.25], [1.0, 0.0, 2.0, 0.5]]
    )
    np.testing.assert_array_equal(
        np.array([row.generated_bundle for row in rows]), expected_bundles
    )
    assert not np.array_equal(
        rows[0].generated_bundle, rows[1].generated_bundle
    )
    for row in rows:
        assert row.generated_bundle.dtype == dual.bundle_table.dtype
        assert not row.generated_bundle.flags.writeable


@pytest.mark.parametrize(
    ("table_dtype", "table"),
    [
        (
            np.bool_,
            np.array([[True, False, True, False], [False, True, True, True]]),
        ),
        (
            np.int8,
            np.array([[1, 0, 2, 0], [0, 3, 1, 4]], dtype=np.int8),
        ),
    ],
)
def test_rows_preserve_non_float_bundle_dtype(
    table_dtype: type, table: np.ndarray
) -> None:
    # A silent astype-to-float in rows() is a no-op on the float64 default
    # table; only a bool/int8 table shows it.
    dual = make_dual(
        agent_ids=np.array([0, 3, 4], dtype=np.int64),
        bundle_row_ids=np.array([0, 1, 0], dtype=np.int64),
        pis=np.array([0.25, 0.5, 0.125]),
        bundle_table=table.astype(table_dtype),
    )

    rows = list(dual.rows(n_obs=2))

    decoded = np.array([row.generated_bundle for row in rows])
    expected = np.array([table[0], table[1], table[0]], dtype=table_dtype)
    # Compare against the literal dtype, not dual.bundle_table.dtype, which a
    # promotion would drag along with it.
    assert decoded.dtype == np.dtype(table_dtype)
    np.testing.assert_array_equal(decoded, expected)
    for row in rows:
        assert row.generated_bundle.dtype == np.dtype(table_dtype)
        assert not row.generated_bundle.flags.writeable


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
def test_rows_reject_bad_n_obs(bad: object) -> None:
    with pytest.raises(ValueError, match="n_obs"):
        make_dual().rows(n_obs=bad)  # type: ignore[arg-type]


def test_rows_empty_payload_yields_no_rows() -> None:
    dual = make_dual(
        agent_ids=np.array([], dtype=np.int64),
        bundle_row_ids=np.array([], dtype=np.int64),
        pis=np.array([], dtype=np.float64),
        bundle_table=np.empty((0, 4), dtype=bool),
    )

    assert list(dual.rows(n_obs=3)) == []


def test_moment_identical_across_store_round_trip(tmp_path: Path) -> None:
    # A reload must be content-bitwise identical, not merely moment-equal:
    # moment() ignores bound_duals and rep_id.
    dual = make_dual(rep_id=7, bound_duals={0: -0.5, 3: 1.25})
    DualStoreWriter(tmp_path).write(dual)
    loaded = DualStoreReader(tmp_path).load(dual.rep_id)
    assert equal(dual, loaded)
    assert loaded.rep_id == 7
    assert dict(loaded.bound_duals) == {0: -0.5, 3: 1.25}
    # The reloaded payload prices to the dyadic moment computed by hand.
    expected_bytes = np.array([0.375, 0.5, 1.25, 0.3125]).tobytes()
    assert loaded.moment().tobytes() == expected_bytes


def test_bound_duals_mapping_is_immutable() -> None:
    dual = make_dual(bound_duals={0: 1.0, 2: -0.5})
    with pytest.raises(TypeError):
        dual.bound_duals[0] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        del dual.bound_duals[2]  # type: ignore[attr-defined]
    assert dual.bound_duals == {0: 1.0, 2: -0.5}


def test_pickle_and_deepcopy_round_trip() -> None:
    # The bound-duals proxy and the read-only array flags do not survive a
    # default round trip; both invariants must be restored on the clone.
    dual = make_dual(bound_duals={0: 1.5, 2: -0.75})
    for clone in (pickle.loads(pickle.dumps(dual)), copy.deepcopy(dual)):
        assert clone.rep_id == dual.rep_id
        np.testing.assert_array_equal(clone.agent_ids, dual.agent_ids)
        np.testing.assert_array_equal(clone.bundle_row_ids, dual.bundle_row_ids)
        np.testing.assert_array_equal(clone.pis, dual.pis)
        np.testing.assert_array_equal(clone.bundle_table, dual.bundle_table)
        assert clone.bound_duals == {0: 1.5, 2: -0.75}
        with pytest.raises(TypeError):
            clone.bound_duals[0] = 99.0  # type: ignore[index]
        for arr in (
            clone.agent_ids,
            clone.bundle_row_ids,
            clone.pis,
            clone.bundle_table,
        ):
            assert not arr.flags.writeable
