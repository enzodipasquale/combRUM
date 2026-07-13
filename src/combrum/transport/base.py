"""Cross-rank communication contract for the distributed estimator.

:class:`Transport` is the only interface through which ranks exchange data.
Reference implementations live in
:mod:`combrum.transport.reference`; conforming implementations must match them
under the conformance suite, bitwise wherever a docstring says so.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from combrum._bundle_key import (
    pack_bundle as _pack_bundle,
    unpack_bundle as _unpack_bundle,
)

_T = TypeVar("_T")
_CUT_ROW_WIRE_HEADER_BYTES = 5 * np.dtype(np.float64).itemsize


@dataclass(frozen=True)
class NodeTopology:
    """One rank's place in the node layout.

    ``node_rank == 0`` marks the node's publishing member for
    :meth:`Transport.node_shared`.
    """

    node_id: int
    node_rank: int
    node_size: int
    n_nodes: int

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"expected n_nodes >= 1, got {self.n_nodes}")
        if self.node_size < 1:
            raise ValueError(f"expected node_size >= 1, got {self.node_size}")
        if not 0 <= self.node_id < self.n_nodes:
            raise ValueError(
                f"expected node_id in [0, n_nodes) = [0, {self.n_nodes}),"
                f" got {self.node_id}"
            )
        if not 0 <= self.node_rank < self.node_size:
            raise ValueError(
                "expected node_rank in [0, node_size) ="
                f" [0, {self.node_size}), got {self.node_rank}"
            )


class TransportError(RuntimeError):
    """A transport-level failure agreed across ranks.

    ``rank`` is the origin rank; every rank of a collective raises the same
    origin.
    """

    def __init__(self, rank: int, message: str) -> None:
        super().__init__(f"[rank {rank}] {message}")
        self.rank = int(rank)
        self.message = str(message)


@dataclass(frozen=True, eq=False)
class CutRow:
    """One generated cut: the frozen envelope rows travel in.

    ``rep_id`` keys the live replication; it is part of the envelope because
    several replications can share one owning rank, so a rep-less row could not
    be routed to the right master problem. ``agent_id`` is the global agent id,
    ``phi`` the length-K moment payload (float64, read-only), ``epsilon`` the
    cut's scalar term, and ``bundle_key`` the cut's identity key. For
    bundle-carrying formulations (e.g. NSlack) the generating bundle is
    recoverable via :attr:`bundle`.

    Identity is by object (``eq=False``): the ndarray payload makes structural
    equality ambiguous; consumers dedup by :attr:`canonical_key`.
    """

    rep_id: int
    agent_id: int
    phi: np.ndarray
    epsilon: float
    bundle_key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.rep_id, (int, np.integer)) or self.rep_id < 0:
            raise ValueError(
                f"expected rep_id to be an integer >= 0, got {self.rep_id!r}"
            )
        object.__setattr__(self, "rep_id", int(self.rep_id))
        if not isinstance(self.agent_id, (int, np.integer)) or self.agent_id < 0:
            raise ValueError(
                f"expected agent_id to be an integer >= 0, got {self.agent_id!r}"
            )
        object.__setattr__(self, "agent_id", int(self.agent_id))
        phi = np.array(self.phi, dtype=np.float64, copy=True, order="C")
        if phi.ndim != 1:
            raise ValueError(
                f"expected one-dimensional (K,) phi, got shape {phi.shape}"
            )
        phi.setflags(write=False)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "epsilon", float(self.epsilon))
        if not isinstance(self.bundle_key, bytes):
            raise ValueError(
                f"expected bytes bundle_key, got {type(self.bundle_key).__name__}"
            )
        if not self.bundle_key:
            raise ValueError("bundle_key must be nonempty")

    @classmethod
    def _from_parts(
        cls,
        *,
        rep_id: int,
        agent_id: int,
        phi: np.ndarray,
        epsilon: float,
        bundle_key: bytes,
    ) -> "CutRow":
        if not isinstance(rep_id, (int, np.integer)) or rep_id < 0:
            raise ValueError(f"expected rep_id to be an integer >= 0, got {rep_id!r}")
        if not isinstance(agent_id, (int, np.integer)) or agent_id < 0:
            raise ValueError(
                f"expected agent_id to be an integer >= 0, got {agent_id!r}"
            )
        phi = np.asarray(phi, dtype=np.float64, order="C")
        if phi.ndim != 1:
            raise ValueError(
                f"expected one-dimensional (K,) phi, got shape {phi.shape}"
            )
        phi.setflags(write=False)
        if not isinstance(bundle_key, bytes):
            raise ValueError(
                f"expected bytes bundle_key, got {type(bundle_key).__name__}"
            )
        if not bundle_key:
            raise ValueError("bundle_key must be nonempty")
        row = object.__new__(cls)
        object.__setattr__(row, "rep_id", int(rep_id))
        object.__setattr__(row, "agent_id", int(agent_id))
        object.__setattr__(row, "phi", phi)
        object.__setattr__(row, "epsilon", float(epsilon))
        object.__setattr__(row, "bundle_key", bundle_key)
        return row

    def _replace(
        self,
        *,
        rep_id: int | None = None,
        epsilon: float | None = None,
        bundle_key: bytes | None = None,
    ) -> "CutRow":
        new_rep_id = self.rep_id if rep_id is None else rep_id
        if not isinstance(new_rep_id, (int, np.integer)) or new_rep_id < 0:
            raise ValueError(
                f"expected rep_id to be an integer >= 0, got {new_rep_id!r}"
            )
        new_eps = self.epsilon if epsilon is None else float(epsilon)
        new_key = self.bundle_key if bundle_key is None else bundle_key
        if not isinstance(new_key, bytes) or not new_key:
            raise ValueError("bundle_key must be nonempty bytes")
        if (
            int(new_rep_id) == self.rep_id
            and new_eps == self.epsilon
            and new_key == self.bundle_key
        ):
            return self
        row = self._from_parts(
            rep_id=int(new_rep_id),
            agent_id=self.agent_id,
            phi=self.phi,
            epsilon=new_eps,
            bundle_key=new_key,
        )
        if new_key == self.bundle_key:
            cached = self.__dict__.get("_bundle")
            if cached is not None:
                object.__setattr__(row, "_bundle", cached)
        return row

    @property
    def canonical_key(self) -> tuple[int, int, bytes]:
        """The total exchange order: ``(rep_id, agent_id, bundle_key)``."""
        return (self.rep_id, self.agent_id, self.bundle_key)

    @property
    def bundle(self) -> np.ndarray:
        """The generating bundle as a read-only array.

        Defined for cuts that pack an explicit bundle into ``bundle_key``
        (those produced by :class:`~combrum.NSlack`). Raises ``ValueError``
        for cuts whose key is an opaque aggregate digest
        (:class:`~combrum.OneSlack`), which pool many bundles and so carry no
        single generating bundle.
        """
        cached = self.__dict__.get("_bundle")
        if cached is None:
            cached = _unpack_bundle(self.bundle_key)
            object.__setattr__(self, "_bundle", cached)
        return cached

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state.pop("_bundle", None)
        return state


def _cut_row_nbytes(row: CutRow) -> int:
    return _CUT_ROW_WIRE_HEADER_BYTES + int(row.phi.nbytes) + len(row.bundle_key)


def canonical_cut_order(rows: Iterable[CutRow]) -> tuple[CutRow, ...]:
    """Rows sorted by :attr:`CutRow.canonical_key` (stable sort).

    Stability matters for permitted duplicate keys: a deterministic pooled
    input order (rank-major, within-rank contribution order) keeps duplicates
    deterministically ordered.
    """
    return tuple(sorted(rows, key=lambda row: row.canonical_key))


class Transport(ABC):
    """Cross-rank communication contract (SPMD discipline).

    Every rank runs the same program and reaches the same collectives in the
    same order; a collective returns only once every rank has contributed.
    Wherever a float-valued result could depend on combination order, the
    method's own docstring names the canonical order. Arrival order must never
    affect a conforming implementation.
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """This rank's id, in ``[0, size)``."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of ranks."""

    @property
    @abstractmethod
    def node(self) -> NodeTopology:
        """This rank's place in the node layout."""

    @abstractmethod
    def collective(self) -> AbstractContextManager[None]:
        """Guard rank-local work so a divergent failure is agreed, not hung.

        Usage::

            with transport.collective():
                ...  # rank-local work that may fail on some ranks only

        If the body raises on any subset of ranks, EVERY rank raises
        :class:`TransportError` at the guard exit, carrying the origin rank and
        original message (with several failing ranks, the lowest-ranked report
        is the agreed verdict); no rank is left waiting in a later collective for
        a dead peer. If no rank fails, the guard is transparent.

        The agreement is one fixed word-sized round at the guard exit, reached
        on success and failure alike. An implementation MAY fold it into a data
        collective that is the body's final operation, but correctness must never
        depend on the body reaching any collective.
        """

    @abstractmethod
    def bcast(self, obj: _T | None, root: int = 0) -> _T:
        """Deliver ``obj`` from rank ``root`` to every rank.

        Non-root ranks' ``obj`` argument is ignored (pass ``None``).
        Returns root's object on root and a private copy elsewhere:
        mutating a received copy must never alias root state.
        """

    @abstractmethod
    def send_to_root(self, obj: _T | None, *, source: int, root: int = 0) -> _T | None:
        """Deliver one object from ``source`` to ``root`` only.

        Only ``source``'s ``obj`` is meaningful and only ``root`` receives it;
        non-root ranks return ``None``. For final artifacts that ``root``
        persists and that should not be broadcast to every worker.
        """

    @abstractmethod
    def allreduce_max(self, value: float) -> float:
        """Max of ``value`` over all ranks, returned on every rank.

        Max is order-independent, so a native reduction is permitted (no
        canonical-order requirement). A NaN on any rank yields NaN on
        every rank.
        """

    @abstractmethod
    def sum_reproducible(
        self, values: np.ndarray, global_ids: np.ndarray
    ) -> np.ndarray | float:
        """Deterministic global sum of per-rank contributions.

        Each rank passes its local contribution rows: ``values`` shaped
        ``(n_r,)`` or ``(n_r, M)`` with one integer global row id per
        row, and every rank returns the identical global sum. A rank
        with nothing to contribute passes the empty pair with
        integer-dtype ids.

        Bitwise contract: the result equals
        :func:`combrum.reductions.canonical_sum` of the concatenation of
        all ranks' contributions, for any rank count and any
        distribution of the rows across ranks. The union of ids must be
        globally unique (one contribution per global row id per call; pre-combine
        rank-locally otherwise).
        """

    @abstractmethod
    def sum_vectors_reproducible(self, values: np.ndarray) -> np.ndarray:
        """Deterministic elementwise sum of one local aggregate per rank.

        Each rank passes a same-shaped float64 array, either ``(M,)`` or
        ``(B, M)``. The returned array has that shape and equals the
        rank-ordered sum over ranks. This primitive is for data whose intended
        contract is tied to a fixed rank layout; row-keyed quantities such as
        observed moments, ``c_theta``, and bootstrap normalizers must use
        :meth:`sum_reproducible`.

        Bitwise contract: the result equals
        :func:`combrum.reductions.canonical_sum` of the flattened per-rank
        arrays keyed by rank index. This is a fixed-rank aggregate contract, not
        the row-distribution-invariant contract of :meth:`sum_reproducible`.
        """

    @abstractmethod
    def scatter_by_agent(
        self,
        arrays: dict[str, np.ndarray] | None,
        agent_ids: np.ndarray,
        *,
        root: int = 0,
    ) -> dict[str, np.ndarray]:
        """Distribute rows of root-held arrays to the ranks owning them.

        Rank ``root`` passes ``arrays``: a dict of arrays whose axis 0 is
        indexed by global agent id (every value shares the same axis-0
        length); every other rank passes ``None``. Each rank passes its own
        ``agent_ids`` (1-D integer global ids) and receives exactly
        ``{key: arrays[key][agent_ids]}``, its rows in ``agent_ids`` order,
        as read-only arrays.

        The mapping is the whole contract; how the rows travel is each
        implementation's concern.
        """

    @abstractmethod
    def gather_agent_values(
        self,
        values: np.ndarray,
        global_ids: np.ndarray,
        n_global: int,
        *,
        root: int = 0,
    ) -> np.ndarray | None:
        """Gather shard-local agent values into a dense root vector.

        Each rank passes ``values`` shaped ``(n_r,)`` and matching integer
        ``global_ids``. The root receives a dense read-only ``(n_global,)``
        vector filled with zeros for absent ids and the contributed value at
        every supplied id; non-root ranks return ``None``. A result-publication
        primitive: it moves the agent axis once, only to the persisting rank.
        """

    @abstractmethod
    def route_agent_values(
        self,
        values: Mapping[int, float] | None,
        agent_ids: np.ndarray,
        *,
        source: int,
        n_agents: int,
    ) -> dict[int, float]:
        """Route sparse source-held agent values to pricing-agent owners.

        ``source`` passes a mapping keyed by global agent id; other ranks pass
        ``None``. Returned values belong to this rank's contiguous agent shard.
        Implementations must route sparse payloads without materializing the
        full agent axis. Values are always keyed by global agent id.
        """

    def route_agent_values_batched(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        agent_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> dict[int, dict[int, float]]:
        """Route many sparse owner-held agent maps in one super-step.

        For replication ``b``, only rank ``owners[b]`` may pass
        ``values_by_rep[b]``. The return maps delivered replication ids to this
        rank's sparse local values. Distributed bootstrap requires a transport
        implementation that provides this batched primitive.
        """
        raise NotImplementedError

    @abstractmethod
    def node_shared(self, arrays: dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Publish read-only data once per node.

        The node's publishing member (``node.node_rank == 0``) provides
        the content: its ``arrays`` become the node's single copy; the
        argument of the node's other ranks is ignored (they may pass an
        empty dict). Returns a read-only mapping; all ranks of one node
        read one node-local copy (in-process transports may expose
        ``np.shares_memory``; MPI transports expose the same shared-window
        contents through process-local arrays), and distinct nodes hold
        distinct copies. Publishing copies the input once, so later mutation of
        the source cannot leak into the shared data.
        """

    @abstractmethod
    def batched_max(self, values: np.ndarray) -> np.ndarray:
        """Elementwise max across ranks of a ``(B,)`` batch in one round.

        Every rank passes the same shape ``(B,)`` and receives the
        identical ``(B,)`` float64 result. One round for the whole batch:
        per-entry rounds would multiply latency by the number of live
        replications.
        """

    def owner_broadcast(self, values: np.ndarray, owners: np.ndarray) -> np.ndarray:
        """Broadcast owner-held rows to every rank in one batched round.

        Row ``b`` is meaningful only on rank ``owners[b]``; every rank receives
        the complete ``(B, M)`` block. This primitive is for small per-master
        state. Distributed bootstrap requires a transport implementation that
        provides this batched primitive.
        """
        raise NotImplementedError

    @abstractmethod
    def exchange_cuts(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[CutRow, ...]:
        """Route generated cut rows to their owning ranks in one batched
        super-step (a counts exchange plus a payload exchange), constant in the
        number of live replications, never a per-replication round.

        Every rank contributes ``rows`` for any replication; ``owners``
        is ``(B,)`` integer ranks, identical on every rank, and a row
        with ``rep_id == b`` belongs to rank ``owners[b]``; every
        ``rep_id`` must lie in ``[0, B)``. Each rank receives exactly the
        rows of the replications it owns from all ranks, its own included,
        sorted in canonical ``(rep_id, agent_id, bundle_key)``
        order. One super-step serves all live replications together.

        Rows with equal canonical keys contributed by different ranks are
        permitted and all delivered (the consumer dedups against its
        constraint set); among equal keys the delivery order is
        rank-major, then within-rank contribution order, so even
        duplicates arrive deterministically.
        """
