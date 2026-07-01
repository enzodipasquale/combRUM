"""Dual-informed re-pricing schedule and its compact payload.

:class:`DualInformed` skips agents whose dual mass has concentrated on a
single bundle, since re-pricing them is unlikely to yield a new cut.
Skipping is bounded: agents the payload cannot vouch for are always
re-priced, and every agent is force-revisited after
``min_revisit_period`` iterations, because concentration is a heuristic
over the last solve, not a proof about the next.

The root-resident dual state crosses ranks as
:class:`DualConcentration`: parallel arrays sized by the dual support
(O(|support|), never an O(n_agents) broadcast). Each rank derives the
full ``(n_agents,)`` mask locally from the payload, the iteration
counter, and its own ``last_resolved`` bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from combrum.schedule import RepricingSchedule

# dual mass at or below this is solver noise, not support
# (shared with the retirement policies)
_SUPPORT_ATOL = 1e-10


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class DualConcentration:
    """Per-agent dual concentration, sized by the dual support.

    ``agent_ids`` (strictly increasing ``int64``) lists the agents
    holding dual mass above :data:`_SUPPORT_ATOL`; ``max_weights`` is a
    parallel ``float64`` array of each agent's largest normalized weight
    (single-cut share of its dual mass, in ``(0, 1]``). Absence from the
    payload means "no evidence", never "settled". Both arrays are copied
    and frozen so the broadcast payload cannot be mutated via an alias.
    """

    agent_ids: np.ndarray
    max_weights: np.ndarray

    def __post_init__(self) -> None:
        agent_ids = np.array(self.agent_ids)
        if agent_ids.ndim != 1 or not np.issubdtype(agent_ids.dtype, np.integer):
            raise ValueError(
                "agent_ids must be a 1-D integer array;"
                f" got shape {agent_ids.shape}, dtype {agent_ids.dtype}"
            )
        if agent_ids.size:
            if int(agent_ids.min()) < 0:
                raise ValueError(
                    f"agent_ids must be >= 0; got {agent_ids[agent_ids < 0].tolist()}"
                )
            if np.any(np.diff(agent_ids) <= 0):
                # Strictly increasing ids keep serialization deterministic:
                # equal dual states produce equal payload bytes.
                raise ValueError(
                    f"agent_ids must be strictly increasing; got {agent_ids.tolist()}"
                )
        max_weights = np.array(self.max_weights, dtype=np.float64)
        if max_weights.shape != agent_ids.shape:
            raise ValueError(
                "max_weights must be parallel to agent_ids;"
                f" got shapes {max_weights.shape}, {agent_ids.shape}"
            )
        if max_weights.size and not (
            np.all(max_weights > 0.0) and np.all(max_weights <= 1.0)
        ):
            raise ValueError(
                "max_weights are normalized single-cut shares and must"
                f" lie in (0, 1]; got {max_weights.tolist()}"
            )
        object.__setattr__(self, "agent_ids", _frozen(agent_ids.astype(np.int64)))
        object.__setattr__(self, "max_weights", _frozen(max_weights))

    @classmethod
    def from_cut_duals(
        cls, duals: Mapping[tuple[int, bytes], float]
    ) -> DualConcentration:
        """Condense per-cut duals into the per-agent concentration payload.

        ``duals`` is the per-cut mapping keyed by ``(agent_id,
        bundle_key)``. Each agent's support is its cuts with dual mass
        above :data:`_SUPPORT_ATOL`; the recorded weight is the largest
        support mass normalized by the agent's total support mass. Agents
        without support are omitted, so the payload scales with the
        support, not the agent space.
        """
        per_agent: dict[int, list[float]] = {}
        for (agent_id, _bundle_key), pi in duals.items():
            if pi > _SUPPORT_ATOL:
                per_agent.setdefault(int(agent_id), []).append(float(pi))
        agent_ids = sorted(per_agent)
        weights = [max(per_agent[agent]) / sum(per_agent[agent]) for agent in agent_ids]
        return cls(
            agent_ids=np.asarray(agent_ids, dtype=np.int64),
            max_weights=np.asarray(weights, dtype=np.float64),
        )


class DualInformed(RepricingSchedule):
    """Skip dual-settled agents, with bounded staleness.

    An agent is skipped only when the payload shows its dual concentrated
    (max normalized weight ``>= concentration_threshold``) and it was
    re-priced within the last ``min_revisit_period`` iterations.
    Everything else re-prices: iteration 0, agents absent from the
    payload, agents overdue per ``last_resolved``, and every agent when
    ``dual`` or ``last_resolved`` is missing.
    """

    def __init__(
        self,
        concentration_threshold: float = 0.9,
        min_revisit_period: int = 5,
    ) -> None:
        threshold = float(concentration_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "concentration_threshold must lie in (0, 1];"
                f" got {concentration_threshold!r}"
            )
        if (
            not isinstance(min_revisit_period, (int, np.integer))
            or min_revisit_period < 1
        ):
            raise ValueError(
                "min_revisit_period must be an integer >= 1;"
                f" got {min_revisit_period!r}"
            )
        self._threshold = threshold
        self._period = int(min_revisit_period)

    def select(
        self,
        iteration: int,
        n_agents: int,
        dual: object | None = None,
        last_resolved: np.ndarray | None = None,
    ) -> np.ndarray:
        n_agents = int(n_agents)
        if iteration == 0 or dual is None or last_resolved is None:
            return np.ones(n_agents, dtype=bool)
        if not isinstance(dual, DualConcentration):
            raise ValueError(
                f"dual must be a DualConcentration payload; got {type(dual).__name__}"
            )
        last = np.asarray(last_resolved)
        if last.shape != (n_agents,) or not np.issubdtype(last.dtype, np.integer):
            raise ValueError(
                "last_resolved must be a (n_agents,) integer array;"
                f" got shape {last.shape}, dtype {last.dtype}"
            )
        mask = np.ones(n_agents, dtype=bool)
        mask[dual.agent_ids[dual.max_weights >= self._threshold]] = False
        # Forced revisit overrides concentration, never the reverse.
        return mask | ((int(iteration) - last) >= self._period)

    def __repr__(self) -> str:
        return (
            f"DualInformed(concentration_threshold={self._threshold},"
            f" min_revisit_period={self._period})"
        )
