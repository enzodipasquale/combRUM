from __future__ import annotations

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
    # rep_id is replication provenance: it must go through operator.index, which
    # rejects a non-integer, never int()-truncate it (7.9 -> 7 would mis-key a
    # replication and silently collide with rep 7). Both the constructor and
    # with_rep_id share the contract, so both paths are pinned. -1 alone can't
    # distinguish operator.index from int(), since int(-1) is still rejected as
    # negative; a fractional value is what separates them.
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
    # Requiring an integer dtype (instead of coercing) is the contract:
    # a silent float->int truncation would corrupt ids.
    with pytest.raises(ValueError, match="agent_ids must be a 1-D integer"):
        make_dual(agent_ids=np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="bundle_row_ids must be a 1-D integer"):
        make_dual(bundle_row_ids=np.array([0.0, 1.0, 0.0]))


def test_bundle_table_must_be_2d() -> None:
    with pytest.raises(ValueError, match=r"bundle_table must be 2-D"):
        make_dual(bundle_table=np.zeros(4))
    with pytest.raises(ValueError, match=r"bundle_table must be 2-D"):
        make_dual(bundle_table=np.zeros((2, 2, 2)))


def test_payload_arrays_read_only() -> None:
    dual = make_dual()
    arrays = (dual.agent_ids, dual.bundle_row_ids, dual.pis, dual.bundle_table)
    for arr in arrays:
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[(0,) * arr.ndim] = 99


def test_payload_owns_its_data() -> None:
    # Ownership, not just read-only flags: construction must copy every
    # caller array, so a view the caller took *before* construction cannot
    # write through into the payload. A full-slice view is a distinct ndarray
    # that keeps write=True even after the payload freezes its own buffer, so
    # it exposes an asarray-alias (shared storage) on any of the four arrays,
    # not just the two the old assertions touched.
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
        # A copy-on-construct payload leaves these views writeable; if a view
        # is frozen the caller array was aliased, which the asserts below fail.
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
    # The reused (not copied) arrays still price to the hand-computed moment.
    # Pin to the external dyadic vector, not to dual.moment(), so a broken
    # moment() can't hide by scaling both sides equally.
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
    # Pin the decoded bundle for every row against a hand-written table, not
    # against dual.bundle_table[...]. bundle_row_ids == [0, 1, 0], so row 0 is
    # decoded twice; a "default-when-zero" indexing bug (bundle_table[row or 1])
    # or any constant-row substitution corrupts rows[0]/rows[2] while leaving
    # rows[1] intact, so checking only rows[1] would miss it.
    expected_bundles = np.array(
        [[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.25], [1.0, 0.0, 2.0, 0.5]]
    )
    np.testing.assert_array_equal(
        np.array([row.generated_bundle for row in rows]), expected_bundles
    )
    # The bundle_row_id == 0 rows must decode to a distinct bundle from the
    # bundle_row_id == 1 row (guards against a collapse-to-one-row bug).
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
    # rows() must hand back each snapshot in the table's own dtype; a silent
    # astype-to-float promotion is a no-op on the float64 default table used by
    # the other rows() test, so it can only be caught on a bool/int8 table.
    # bundle_row_ids == [0, 1, 0] decodes row 0 twice, so the oracle below pins
    # every decoded row wholesale against a hand-written table rather than one
    # element of one row.
    dual = make_dual(
        agent_ids=np.array([0, 3, 4], dtype=np.int64),
        bundle_row_ids=np.array([0, 1, 0], dtype=np.int64),
        pis=np.array([0.25, 0.5, 0.125]),
        bundle_table=table.astype(table_dtype),
    )

    rows = list(dual.rows(n_obs=2))

    decoded = np.array([row.generated_bundle for row in rows])
    expected = np.array([table[0], table[1], table[0]], dtype=table_dtype)
    # Independent oracle: exact dtype AND exact values, pinned to a literal
    # dtype (not dual.bundle_table.dtype) so a float promotion can't hide by
    # dragging both sides along. astype(float64) on a bool/int8 table both
    # changes the dtype and forces a writeable copy, so the two asserts below
    # kill the promotion twice over.
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
    # Self-containedness in practice: a payload reloaded from disk, with its
    # producing context gone, is content-bitwise identical -- every array,
    # the bound-duals mapping, and rep_id -- not merely moment-equal (moment()
    # ignores bound_duals and rep_id, so a dropped bound multiplier or a
    # mis-keyed rep would slip past a moment-only check).
    dual = make_dual(rep_id=7, bound_duals={0: -0.5, 3: 1.25})
    DualStoreWriter(tmp_path).write(dual)
    loaded = DualStoreReader(tmp_path).load(dual.rep_id)
    assert equal(dual, loaded)
    assert loaded.rep_id == 7
    assert dict(loaded.bound_duals) == {0: -0.5, 3: 1.25}
    # Supplementary: the reloaded payload prices to the hand-computed dyadic
    # moment, bit-for-bit. Comparing to the external byte pattern (not to
    # dual.moment()) keeps a broken moment() from passing by breaking both
    # sides identically.
    expected_bytes = np.array([0.375, 0.5, 1.25, 0.3125]).tobytes()
    assert loaded.moment().tobytes() == expected_bytes


def test_bound_duals_mapping_is_immutable() -> None:
    # Like the read-only arrays, the bound-duals mapping must reject in-place edits.
    dual = make_dual(bound_duals={0: 1.0, 2: -0.5})
    with pytest.raises(TypeError):
        dual.bound_duals[0] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        del dual.bound_duals[2]  # type: ignore[attr-defined]
    assert dual.bound_duals == {0: 1.0, 2: -0.5}
