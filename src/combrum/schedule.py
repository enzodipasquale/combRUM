"""Re-pricing schedules: which agents to re-price each iteration.

Schedules return a boolean ndarray mask of length ``n_agents`` (not index
lists). The built-in schedules are pure functions of ``(iteration,
n_agents)``, so every rank computes the identical mask locally with zero
communication. Schedules depending on root-resident state read the
``dual``/``last_resolved`` arguments; their payloads must stay O(local
shard)/O(support), never a full O(n_agents) broadcast per iteration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RepricingSchedule(ABC):
    """Which agents to re-price this iteration, as a boolean mask."""

    @abstractmethod
    def select(
        self,
        iteration: int,
        n_agents: int,
        dual: object | None = None,
        last_resolved: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boolean mask of shape ``(n_agents,)``: True = re-price.

        The optional arguments are signals an informed schedule may read;
        ``None`` means the caller cannot supply that signal this iteration.
        """


class ResolveAll(RepricingSchedule):
    """Every agent, every iteration: the exhaustive reference schedule."""

    def select(
        self,
        iteration: int,
        n_agents: int,
        dual: object | None = None,
        last_resolved: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.ones(int(n_agents), dtype=bool)

    def __repr__(self) -> str:
        return "ResolveAll()"


class RoundRobin(RepricingSchedule):
    """Cycle through ``chunks`` balanced contiguous slices of the agents.

    Iteration ``i`` selects the ``i mod chunks``-th slice; slice sizes
    differ by at most one, so ``chunks`` consecutive iterations cover
    every agent exactly once.
    """

    def __init__(self, chunks: int) -> None:
        if not isinstance(chunks, (int, np.integer)) or chunks < 1:
            raise ValueError(f"chunks must be an integer >= 1; got {chunks!r}")
        self._chunks = int(chunks)

    def select(
        self,
        iteration: int,
        n_agents: int,
        dual: object | None = None,
        last_resolved: np.ndarray | None = None,
    ) -> np.ndarray:
        n_agents = int(n_agents)
        c = int(iteration) % self._chunks
        # First `extra` slices carry one agent more, covering [0, n_agents).
        base, extra = divmod(n_agents, self._chunks)
        start = c * base + min(c, extra)
        stop = start + base + (1 if c < extra else 0)
        mask = np.zeros(n_agents, dtype=bool)
        mask[start:stop] = True
        return mask

    def __repr__(self) -> str:
        return f"RoundRobin(chunks={self._chunks})"
