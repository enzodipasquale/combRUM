from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from combrum.dual import DualSolution
from combrum.dualstore import DualStoreReader, DualStoreWriter, equal

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dualstore"


# ---------------------------------------------------------------------------
# Golden fixtures: canonical literals + regeneration script.
#
# The checked-in npz files under FIXTURE_DIR were written by
# write_golden_fixtures() from these literals. Loading them must reproduce
# the payloads content-bitwise: bytes written today parse identically
# forever. Every literal is dyadic with an explicit dtype, so there is no
# RNG or platform-dependent default in play.
# ---------------------------------------------------------------------------


def canonical_interior() -> DualSolution:
    """Interior solution: empty bound_duals (n=3 rows, n_bundles=2, M=4)."""
    return DualSolution(
        rep_id=0,
        agent_ids=np.array([0, 1, 1], dtype=np.int64),
        bundle_row_ids=np.array([0, 1, 0], dtype=np.int64),
        pis=np.array([0.25, 0.5, 0.125], dtype=np.float64),
        bundle_table=np.array(
            [[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.25]], dtype=np.float64
        ),
        bound_duals={},
    )


def canonical_on_bound() -> DualSolution:
    """Theta-on-bound solution: nonempty, nonzero bound multipliers."""
    return DualSolution(
        rep_id=1,
        agent_ids=np.array([0, 2], dtype=np.int64),
        bundle_row_ids=np.array([1, 0], dtype=np.int64),
        pis=np.array([0.75, 1.5], dtype=np.float64),
        bundle_table=np.array(
            [[0.5, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 2.0]], dtype=np.float64
        ),
        bound_duals={0: -0.5, 3: 1.25},
    )


def write_golden_fixtures(target: Path = FIXTURE_DIR) -> tuple[Path, Path]:
    """Regenerate the checked-in goldens from the canonical literals.

    Not a test. Delete the two files first: the append-only guard
    refuses to overwrite them.
    """
    writer = DualStoreWriter(target)
    return writer.write(canonical_interior()), writer.write(canonical_on_bound())


def make_dual(rep_id: int = 0, **overrides: object) -> DualSolution:
    # Deliberately mixed dtypes: the store must preserve them natively.
    kwargs: dict[str, object] = dict(
        rep_id=rep_id,
        agent_ids=np.array([3, 5], dtype=np.int32),
        bundle_row_ids=np.array([1, 0], dtype=np.int64),
        pis=np.array([0.5, 2.0], dtype=np.float64),
        bundle_table=np.array([[1, 0, 2], [3, 1, 0]], dtype=np.int8),
        bound_duals={2: -1.5, 0: 0.25},
    )
    kwargs.update(overrides)
    return DualSolution(**kwargs)  # type: ignore[arg-type]


def test_write_load_round_trips_content_bitwise(tmp_path: Path) -> None:
    dual = make_dual()
    path = DualStoreWriter(tmp_path).write(dual)
    assert path == tmp_path / "rep-00000000.npz"
    loaded = DualStoreReader(tmp_path).load(0)
    assert equal(loaded, dual)
    # Assert dtypes explicitly: non-default int32 ids and int8 table
    # must survive natively, which equal() alone would not catch.
    assert loaded.agent_ids.dtype == np.dtype(np.int32)
    assert loaded.bundle_row_ids.dtype == np.dtype(np.int64)
    assert loaded.pis.dtype == np.dtype(np.float64)
    assert loaded.bundle_table.dtype == np.dtype(np.int8)
    assert loaded.rep_id == 0
    assert loaded.bound_duals == {2: -1.5, 0: 0.25}


def test_writer_canonicalizes_bound_coords_on_disk(tmp_path: Path) -> None:
    # The writer promises one encoding per payload regardless of dict order.
    # Feed unsorted insertion order and read the raw npz: bound_coords must
    # land sorted ascending, with bound_values carried alongside their coord.
    path = DualStoreWriter(tmp_path).write(make_dual(bound_duals={3: 1.0, 0: 2.0}))
    with np.load(path) as npz:
        assert npz["bound_coords"].tolist() == [0, 3]
        assert npz["bound_values"].tolist() == [2.0, 1.0]


def test_bool_bundle_table_dtype_preserved(tmp_path: Path) -> None:
    dual = make_dual(bundle_table=np.array([[True, False], [False, True]]))
    DualStoreWriter(tmp_path).write(dual)
    loaded = DualStoreReader(tmp_path).load(0)
    assert loaded.bundle_table.dtype == np.dtype(bool)
    assert equal(loaded, dual)


def test_duplicate_rep_write_raises(tmp_path: Path) -> None:
    writer = DualStoreWriter(tmp_path)
    writer.write(make_dual(rep_id=4))
    with pytest.raises(FileExistsError, match="replication 4 .*append-only"):
        writer.write(make_dual(rep_id=4))


def test_missing_rep_load_names_rep_and_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        DualStoreReader(tmp_path).load(42)
    message = str(excinfo.value)
    # Pin the raw rep id as a token, not the bare "42" that the zero-padded
    # filename rep-00000042.npz also carries: a message that only echoed the
    # resolved path would still contain "42" and str(tmp_path) yet drop the
    # "replication 42" wording this contract promises.
    assert "replication 42" in message
    # The unpadded rep id, not the padded filename form, must name the rep.
    assert "rep-00000042" not in message
    assert str(tmp_path) in message


def test_rep_ids_sorted_and_files_zero_padded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = DualStoreWriter(tmp_path)
    for rep_id in (5, 3, 9):
        writer.write(make_dual(rep_id=rep_id))
    # Force iterdir() to hand back the rep files in descending name order so
    # rep_ids() cannot pass by inheriting the filesystem's incidental sort:
    # a reader that skips its own sort would return (9, 5, 3) here.
    real_iterdir = Path.iterdir

    def reversed_iterdir(self: Path) -> Iterator[Path]:
        return iter(sorted(real_iterdir(self), key=lambda p: p.name, reverse=True))

    monkeypatch.setattr(Path, "iterdir", reversed_iterdir)
    assert DualStoreReader(tmp_path).rep_ids() == (3, 5, 9)
    # Zero-padded names keep lexicographic listing in numeric order.
    assert (tmp_path / "rep-00000005.npz").exists()


def test_iteration_ascending_and_lazy(tmp_path: Path) -> None:
    writer = DualStoreWriter(tmp_path)
    for rep_id in (2, 0, 1):
        writer.write(make_dual(rep_id=rep_id))
    reader = DualStoreReader(tmp_path)

    loads: list[int] = []
    real_load = reader.load

    def counting_load(rep_id: int) -> DualSolution:
        loads.append(rep_id)
        return real_load(rep_id)

    reader.load = counting_load  # type: ignore[method-assign]
    iterator = iter(reader)
    first = next(iterator)
    assert first.rep_id == 0
    # The streaming contract: consuming the first rep loaded exactly one.
    assert loads == [0]
    assert [dual.rep_id for dual in iterator] == [1, 2]
    assert loads == [0, 1, 2]


def test_write_leaves_no_tmp_residue(tmp_path: Path) -> None:
    DualStoreWriter(tmp_path).write(make_dual())
    assert [entry.name for entry in tmp_path.iterdir()] == ["rep-00000000.npz"]


def test_torn_write_propagates_and_leaves_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the rename-into-place to fail after the .tmp is written: this is
    # the only path where the writer's finally-cleanup is load-bearing. On
    # success os.replace consumes the .tmp, so the success-path test above
    # cannot see whether the cleanup exists.
    import combrum.dualstore as dualstore

    boom = OSError("disk full")

    def failing_replace(src: str, dst: str) -> None:
        raise boom

    monkeypatch.setattr(dualstore.os, "replace", failing_replace)
    with pytest.raises(OSError) as excinfo:
        DualStoreWriter(tmp_path).write(make_dual())
    assert excinfo.value is boom
    # No final file and, critically, no torn .tmp sibling left behind.
    assert list(tmp_path.iterdir()) == []


def test_rep_ids_excludes_foreign_and_torn_write_siblings(tmp_path: Path) -> None:
    # rep_ids() must list exactly the well-formed rep files. Drop siblings that
    # share the rep prefix but are not rep files: a torn-write .tmp, a stray
    # .bak, a foreign .txt, and a prefix-only rep-.npz. A reader that anchored
    # only the prefix (fullmatch -> match, or dropping the trailing \.npz)
    # would count rep-000000NN.npz.tmp/.bak as phantom duplicate reps.
    writer = DualStoreWriter(tmp_path)
    for rep_id in (7, 2):
        writer.write(make_dual(rep_id=rep_id))
    (tmp_path / "rep-00000007.npz.tmp").write_bytes(b"x")
    (tmp_path / "rep-00000002.npz.bak").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "rep-.npz").write_bytes(b"x")
    reader = DualStoreReader(tmp_path)
    # Full set, ascending, no phantom or duplicate: the faulty implementation returns
    # (2, 2, 7, 7); a filter that also swallowed rep-.npz would add a stray.
    assert reader.rep_ids() == (2, 7)
    # Iterating loads only the two real reps, in order — the siblings never
    # reach load() (which would fail on their non-npz bytes).
    assert [dual.rep_id for dual in reader] == [2, 7]


def test_writer_creates_missing_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "store"
    path = DualStoreWriter(nested).write(make_dual())
    assert path.exists()
    assert DualStoreReader(nested).rep_ids() == (0,)


def test_reader_requires_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="nope"):
        DualStoreReader(missing).rep_ids()


def test_load_rejects_mislabeled_file(tmp_path: Path) -> None:
    # A renamed file must not masquerade as another replication: the
    # rep_id inside the payload is authoritative.
    DualStoreWriter(tmp_path).write(make_dual(rep_id=1))
    (tmp_path / "rep-00000001.npz").rename(tmp_path / "rep-00000002.npz")
    with pytest.raises(ValueError, match="carries rep_id 1"):
        DualStoreReader(tmp_path).load(2)


def test_golden_interior_fixture_is_frozen_format() -> None:
    loaded = DualStoreReader(FIXTURE_DIR).load(0)
    assert equal(loaded, canonical_interior())
    assert loaded.bound_duals == {}


def test_golden_on_bound_fixture_preserves_bound_duals() -> None:
    loaded = DualStoreReader(FIXTURE_DIR).load(1)
    assert equal(loaded, canonical_on_bound())
    # Assert nonempty and exact on their own: a store that drops or
    # zeroes bound multipliers must fail here, not only via equal().
    assert all(value != 0.0 for value in loaded.bound_duals.values())
    assert loaded.bound_duals == {0: -0.5, 3: 1.25}


def test_golden_payloads_round_trip_through_fresh_store(tmp_path: Path) -> None:
    write_golden_fixtures(tmp_path)
    reader = DualStoreReader(tmp_path)
    assert reader.rep_ids() == (0, 1)
    assert equal(reader.load(0), canonical_interior())
    assert equal(reader.load(1), canonical_on_bound())


def test_equal_is_bitwise_on_signed_zero(tmp_path: Path) -> None:
    # Value equality would call +0.0 and -0.0 equal; the round-trip
    # predicate must not, since flipping bits while preserving values
    # still changes the frozen format.
    plus = make_dual(pis=np.array([0.0, 0.5]))
    minus = make_dual(pis=np.array([-0.0, 0.5]))
    assert not equal(plus, minus)
    bound_plus = make_dual(bound_duals={1: 0.0})
    bound_minus = make_dual(bound_duals={1: -0.0})
    assert not equal(bound_plus, bound_minus)
    # Positive control: bitwise-identical payloads still compare equal.
    assert equal(plus, make_dual(pis=np.array([0.0, 0.5])))
    # rep_id is part of the contract: payloads identical in every array and
    # bound multiplier but differing only in rep_id must not compare equal.
    assert not equal(make_dual(rep_id=0), make_dual(rep_id=1))
    # equal() must compare ALL four arrays, not just pis. Each control differs
    # from the baseline in exactly one array (value, then dtype), so an equal()
    # that quietly stopped comparing agent_ids / bundle_row_ids / bundle_table
    # would wrongly call these equal.
    base = make_dual()
    assert not equal(base, make_dual(agent_ids=np.array([3, 6], dtype=np.int32)))
    assert not equal(base, make_dual(bundle_row_ids=np.array([1, 1], dtype=np.int64)))
    assert not equal(
        base, make_dual(bundle_table=np.array([[1, 0, 2], [3, 1, 9]], dtype=np.int8))
    )
    # Same values, wider dtype: the bit-format comparison must reject it too.
    assert not equal(base, make_dual(agent_ids=np.array([3, 5], dtype=np.int64)))


def _rewrite_bound_arrays(
    path: Path, coords: np.ndarray, values: np.ndarray
) -> None:
    # Corrupt the file the way a buggy producer would: valid envelope,
    # malformed bound arrays — bypassing the writer on purpose.
    with np.load(path) as npz:
        payload = {name: npz[name] for name in npz.files}
    payload["bound_coords"] = coords
    payload["bound_values"] = values
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


def test_load_rejects_non_parallel_bound_arrays(tmp_path: Path) -> None:
    writer = DualStoreWriter(tmp_path)
    path = writer.write(make_dual(bound_duals={0: 1.0, 3: 2.0}))
    _rewrite_bound_arrays(
        path,
        coords=np.array([0, 3], dtype=np.int64),
        values=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="must be parallel"):
        DualStoreReader(tmp_path).load(0)


def test_load_rejects_duplicate_bound_coordinates(tmp_path: Path) -> None:
    writer = DualStoreWriter(tmp_path)
    path = writer.write(make_dual(bound_duals={0: 1.0, 3: 2.0}))
    _rewrite_bound_arrays(
        path,
        coords=np.array([3, 3], dtype=np.int64),
        values=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="duplicate bound"):
        DualStoreReader(tmp_path).load(0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128])
def test_load_rejects_non_integer_bound_coordinates(
    tmp_path: Path, dtype: type
) -> None:
    # Non-integer coords must be rejected outright: without the dtype guard,
    # int(c) would silently truncate float coords [0.7, 3.2] into keys {0, 3},
    # accepting a wrong-coordinate payload as valid. Parametrizing over the
    # whole non-integer class (float64/float32/complex) kills a regression that
    # narrows the guard to reject only one of them.
    writer = DualStoreWriter(tmp_path)
    path = writer.write(make_dual(bound_duals={0: 1.0, 3: 2.0}))
    _rewrite_bound_arrays(
        path,
        coords=np.array([0.7, 3.2], dtype=dtype),
        values=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="must be integers"):
        DualStoreReader(tmp_path).load(0)


def test_load_accepts_non_int64_integer_bound_coordinates(tmp_path: Path) -> None:
    # The guard rejects the non-integer class, not everything but int64: a
    # coords array in another integer dtype (int32) is still a valid flattened
    # mapping and must load, decoding to the exact {coord: value} payload.
    # This kills a regression that over-narrows the guard to `dtype == int64`.
    writer = DualStoreWriter(tmp_path)
    path = writer.write(make_dual(bound_duals={0: 1.0, 3: 2.0}))
    _rewrite_bound_arrays(
        path,
        coords=np.array([0, 3], dtype=np.int32),
        values=np.array([-0.5, 1.25]),
    )
    loaded = DualStoreReader(tmp_path).load(0)
    assert loaded.bound_duals == {0: -0.5, 3: 1.25}
