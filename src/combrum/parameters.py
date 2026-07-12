"""Ordered named parameter blocks: the layout of the flat theta vector.

A :class:`Parameters` layout determines ``K``, the bound vectors, and the
named accessors on every result type.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

BlockSpec = tuple[float, float, int]


@dataclass(frozen=True)
class _Block:
    name: str
    lb: float
    ub: float
    size: int
    offset: int

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


def _parse_spec(name: str, spec: BlockSpec, offset: int) -> _Block:
    if not (isinstance(spec, tuple) and len(spec) == 3):
        raise ValueError(
            f"parameter block {name!r}: expected an (lb, ub, k) spec, got {spec!r}"
        )
    lb, ub = float(spec[0]), float(spec[1])
    k = spec[2]
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(
            f"parameter block {name!r}: expected integer k >= 1, got {k!r}"
        )
    # "not lb <= ub" also rejects NaN bounds, which compare False both ways.
    if not lb <= ub:
        raise ValueError(
            f"parameter block {name!r}: lb <= ub required; got lb={lb}, ub={ub}"
        )
    return _Block(name=name, lb=lb, ub=ub, size=int(k), offset=offset)


class Parameters:
    """Ordered named blocks of the flat theta vector.

    Built from ``{name: (lb, ub, k)}`` or an iterable of ``(name, (lb, ub, k))``
    pairs; insertion order defines the theta order. Duplicate names are rejected.
    Immutable after construction.
    """

    __slots__ = ("_blocks", "_by_name")

    _blocks: tuple[_Block, ...]
    _by_name: dict[str, _Block]

    def __init__(
        self,
        blocks: Mapping[str, BlockSpec] | Iterable[tuple[str, BlockSpec]],
    ) -> None:
        items = blocks.items() if isinstance(blocks, Mapping) else blocks
        parsed: list[_Block] = []
        by_name: dict[str, _Block] = {}
        offset = 0
        for name, spec in items:
            if name in by_name:
                raise ValueError(f"duplicate parameter block name {name!r}")
            block = _parse_spec(name, spec, offset)
            parsed.append(block)
            by_name[name] = block
            offset += block.size
        if not parsed:
            raise ValueError("Parameters requires at least one block")
        object.__setattr__(self, "_blocks", tuple(parsed))
        object.__setattr__(self, "_by_name", by_name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Parameters is immutable after construction")

    @property
    def K(self) -> int:
        """Total length of the flat theta vector."""
        last = self._blocks[-1]
        return last.offset + last.size

    @property
    def names(self) -> tuple[str, ...]:
        """Block names in theta order."""
        return tuple(block.name for block in self._blocks)

    def block(self, name: str) -> slice:
        """Slice into the flat theta vector for the named block."""
        try:
            return self._by_name[name].slice
        except KeyError:
            raise KeyError(
                f"unknown parameter block {name!r}; known blocks: {list(self._by_name)}"
            ) from None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize ``(lb_vec, ub_vec)`` as length-K float arrays."""
        lb = np.empty(self.K, dtype=np.float64)
        ub = np.empty(self.K, dtype=np.float64)
        for block in self._blocks:
            lb[block.slice] = block.lb
            ub[block.slice] = block.ub
        return lb, ub

    def unpack(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat length-K theta into ``{name: block_values}``."""
        theta = np.asarray(theta)
        if theta.shape != (self.K,):
            raise ValueError(
                f"expected theta of shape (K,) = ({self.K},), got {theta.shape}"
            )
        return {block.name: theta[block.slice] for block in self._blocks}

    def pack(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Inverse of :meth:`unpack`: assemble a flat length-K theta."""
        missing = [name for name in self.names if name not in values]
        if missing:
            raise ValueError(
                f"pack is missing block(s) {missing};"
                f" expected exactly {list(self.names)}"
            )
        extra = [name for name in values if name not in self._by_name]
        if extra:
            raise ValueError(
                f"pack got unknown block name(s) {extra};"
                f" expected exactly {list(self.names)}"
            )
        theta = np.empty(self.K, dtype=np.float64)
        for block in self._blocks:
            block_values = np.atleast_1d(
                np.asarray(values[block.name], dtype=np.float64)
            )
            if block_values.shape != (block.size,):
                raise ValueError(
                    f"pack: expected length {block.size} for block"
                    f" {block.name!r}, got shape {block_values.shape}"
                )
            theta[block.slice] = block_values
        return theta

    def _spec(self) -> tuple[tuple[str, float, float, int], ...]:
        return tuple((b.name, b.lb, b.ub, b.size) for b in self._blocks)

    def __getstate__(self) -> tuple[tuple[str, BlockSpec], ...]:
        """Pickle payload; __reduce__ rebuilds through __init__, not __setstate__."""
        return tuple((b.name, (b.lb, b.ub, b.size)) for b in self._blocks)

    def __reduce__(
        self,
    ) -> tuple[type[Parameters], tuple[tuple[tuple[str, BlockSpec], ...]]]:
        return (type(self), (self.__getstate__(),))

    def __eq__(self, other: object) -> bool:
        # Layout equality gates result merging: theta entries must align exactly.
        if not isinstance(other, Parameters):
            return NotImplemented
        return self._spec() == other._spec()

    def __hash__(self) -> int:
        return hash(self._spec())

    def __repr__(self) -> str:
        parts = ", ".join(f"{b.name}[{b.size}]" for b in self._blocks)
        return f"Parameters(K={self.K}: {parts})"
