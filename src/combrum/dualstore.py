"""Per-replication dual store: one ``rep-{rep_id:08d}.npz`` file per replication.

The serialization is a frozen contract: the round-trip ``load(write(x))``
must be content-bitwise equal to ``x`` (see :func:`equal`). Parsed content
is frozen, not file bytes; npz is a zip and zip members carry timestamps,
so identical payloads may differ on disk yet parse identically.
"""

from __future__ import annotations

import os
import re
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from combrum.dual import DualSolution

_REP_FILE = re.compile(r"rep-(\d+)\.npz")


def _rep_filename(rep_id: int) -> str:
    return f"rep-{rep_id:08d}.npz"


def _same_array_bits(x: np.ndarray, y: np.ndarray) -> bool:
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def _same_float_bits(a: Mapping[int, float], b: Mapping[int, float]) -> bool:
    if sorted(a) != sorted(b):
        return False
    return all(struct.pack("<d", a[k]) == struct.pack("<d", b[k]) for k in a)


def equal(a: DualSolution, b: DualSolution) -> bool:
    """Content-bitwise equality: dtype, shape, and raw bytes of every array
    and bound multiplier, plus ``rep_id`` (signed zeros differ; NaNs compare
    by bit pattern).
    """
    if a.rep_id != b.rep_id:
        return False
    if not _same_float_bits(a.bound_duals, b.bound_duals):
        return False
    pairs = (
        (a.agent_ids, b.agent_ids),
        (a.bundle_row_ids, b.bundle_row_ids),
        (a.pis, b.pis),
        (a.bundle_table, b.bundle_table),
    )
    return all(_same_array_bits(x, y) for x, y in pairs)


class DualStoreWriter:
    """Appends whole replications to a store directory, one file each."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def write(self, dual: DualSolution) -> Path:
        """Persist one replication and return its final path.

        Append-only: rewriting an existing ``rep_id`` raises
        ``FileExistsError``. The payload lands via ``os.replace`` from a
        ``.tmp`` sibling, so a torn write cannot parse as a valid
        replication file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / _rep_filename(dual.rep_id)
        if path.exists():
            raise FileExistsError(
                f"replication {dual.rep_id} already exists in dual store"
                f" {self._dir}; the store is append-only evidence and"
                " never overwrites"
            )
        coords = np.array(sorted(dual.bound_duals), dtype=np.int64)
        values = np.array(
            [dual.bound_duals[c] for c in coords.tolist()], dtype=np.float64
        )
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    rep_id=np.asarray(dual.rep_id, dtype=np.int64),
                    agent_ids=dual.agent_ids,
                    bundle_row_ids=dual.bundle_row_ids,
                    pis=dual.pis,
                    bundle_table=dual.bundle_table,
                    bound_coords=coords,
                    bound_values=values,
                )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


class DualStoreReader:
    """Streaming reads over a store directory, one replication in memory."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def rep_ids(self) -> tuple[int, ...]:
        """All replication ids present, ascending."""
        if not self._dir.is_dir():
            raise FileNotFoundError(f"dual store directory {self._dir} does not exist")
        return tuple(
            sorted(
                int(match.group(1))
                for entry in self._dir.iterdir()
                if (match := _REP_FILE.fullmatch(entry.name))
            )
        )

    def load(self, rep_id: int) -> DualSolution:
        """Load one replication via the :class:`DualSolution` constructor.

        Raises:
            ValueError: if the file is mislabeled or corrupt.
        """
        path = self._dir / _rep_filename(rep_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no replication {rep_id} in dual store {self._dir}"
            )
        with np.load(path) as npz:
            stored_rep_id = int(npz["rep_id"][()])
            if stored_rep_id != rep_id:
                raise ValueError(
                    f"dual store file {path} carries rep_id"
                    f" {stored_rep_id}; a renamed file must not"
                    f" masquerade as replication {rep_id}"
                )
            coords = npz["bound_coords"]
            values = npz["bound_values"]
            if coords.ndim != 1 or values.ndim != 1:
                raise ValueError(
                    f"dual store file {path} is corrupt: expected 1-D"
                    f" bound_coords/bound_values, got shapes"
                    f" {coords.shape}, {values.shape}"
                )
            if coords.shape != values.shape:
                raise ValueError(
                    f"dual store file {path} is corrupt: bound_coords"
                    f" and bound_values must be parallel; got"
                    f" {coords.shape[0]} coordinates,"
                    f" {values.shape[0]} values"
                )
            if not np.issubdtype(coords.dtype, np.integer):
                raise ValueError(
                    f"dual store file {path} is corrupt: bound_coords"
                    f" must be integers; got dtype {coords.dtype}"
                )
            unique, counts = np.unique(coords, return_counts=True)
            if unique.size != coords.size:
                raise ValueError(
                    f"dual store file {path} is corrupt: duplicate bound"
                    f" coordinates {unique[counts > 1].tolist()}; a"
                    " mapping flattened to parallel arrays cannot carry"
                    " a coordinate twice"
                )
            return DualSolution(
                rep_id=stored_rep_id,
                agent_ids=npz["agent_ids"],
                bundle_row_ids=npz["bundle_row_ids"],
                pis=npz["pis"],
                bundle_table=npz["bundle_table"],
                bound_duals={
                    int(c): float(v) for c, v in zip(coords.tolist(), values.tolist())
                },
            )

    def __iter__(self) -> Iterator[DualSolution]:
        for rep_id in self.rep_ids():
            yield self.load(rep_id)
