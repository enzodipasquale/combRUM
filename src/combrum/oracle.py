"""Pricing oracle: where a model's subproblem plugs into the engine.

Oracles must be
deterministic functions of ``(theta, agent_id(s))`` given their data:
randomness or hidden mutable state breaks every parity/determinism gate,
since a rerun or re-sharded run must price every agent identically.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping

import numpy as np

from combrum.demand import Demand, DemandBatch
from combrum.transport.base import Transport


class Oracle(ABC):
    """User-implemented pricing oracle.

    Pricing must be a
    deterministic function of ``(theta, agent_id(s))``. An oracle must
    override at least one of :meth:`price` / :meth:`price_batch`. For large
    or sharded applications, ``price_batch`` is the main path: the engine
    passes exactly the global ids owned by the rank.
    """

    def setup(self, transport: Transport, agent_ids: np.ndarray) -> None:
        """Optional one-time hook called before pricing begins.

        ``agent_ids`` are the global agent ids owned by this rank, so
        stateful oracles can build per-agent state only for their own shard.
        Publish big read-only structure (feature tables, graphs, choice
        sets) once per node via
        :meth:`combrum.transport.base.Transport.node_shared` so memory
        scales with nodes, not ranks.
        """

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        """Solve one agent's subproblem at ``theta``.

        ``agent_id`` is the global id; the same id prices the same agent
        under any sharding. Approximate solves with a finite certified gap
        return :meth:`Demand.inexact`; feasible incumbents without a usable
        certificate return :meth:`Demand.uncertified`.
        """
        raise NotImplementedError(
            "Oracle.price is not overridden; override price or price_batch"
        )

    def price_batch(
        self, theta: np.ndarray, agent_ids: np.ndarray
    ) -> Mapping[int, Demand] | DemandBatch:
        """Solve a whole shard's subproblems at ``theta`` in one call.

        ``agent_ids`` are the global agent ids to price; the result carries
        one :class:`~combrum.demand.Demand` per id, keyed by that global id.
        Vectorized oracles may instead return
        :class:`~combrum.demand.DemandBatch` (bundles/payoffs/gaps as arrays)
        under the same mapping contract when every gap is finite. If any priced
        incumbent lacks a usable finite certificate, return a scalar mapping
        with :meth:`Demand.uncertified` for that id rather than a
        :class:`DemandBatch`.

        When an oracle overrides both, ``price_batch(theta, ids)[id]`` must
        match ``price(theta, id)`` for every id (discrete fields
        byte-identical, continuous fields within ``1e-13``), with the
        per-agent path the deterministic reference.
        """
        raise NotImplementedError(
            "Oracle.price_batch is not overridden; override price or price_batch"
        )

    def teardown(self) -> None:
        """Release per-rank resources; default no-op."""
