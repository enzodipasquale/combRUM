"""Agent-space context shared by every estimation component.

The global agent index is sim-major: ``a = s * N + i``, with observation
index ``i`` in ``[0, N)``, simulation index ``s`` in ``[0, S)``, and global
agent id ``a`` in ``[0, N * S)``. Every cut, dual, and reduction is keyed by
this id, so the convention lives here as code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Literal

import numpy as np

from combrum.master import MasterBackend
from combrum.policies import CutPolicy
from combrum.schedule import RepricingSchedule
from combrum.transport.base import Transport


class ResultPublication(IntFlag):
    """Final artifacts a formulation should publish after convergence."""

    SUMMARY = 0
    SLACK = auto()
    ACTIVE_SET = auto()
    DUAL = auto()
    BROADCAST = auto()
    FULL = SLACK | ACTIVE_SET | DUAL | BROADCAST


_RESULT_PUBLICATION_NAMES: dict[str, ResultPublication] = {
    "summary": ResultPublication.SUMMARY,
    "slack": ResultPublication.SLACK,
    "active_set": ResultPublication.ACTIVE_SET,
    "dual": ResultPublication.DUAL,
    "full": ResultPublication.FULL,
}


def _coerce_result_publication(value: object) -> ResultPublication:
    if isinstance(value, ResultPublication):
        return value
    if isinstance(value, str):
        try:
            return _RESULT_PUBLICATION_NAMES[value]
        except KeyError as exc:
            raise ValueError(
                f"unknown result_publication mode {value!r}:"
                f" expected one of {sorted(_RESULT_PUBLICATION_NAMES)}"
            ) from exc
    if isinstance(value, Iterable):
        publication = ResultPublication.SUMMARY
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "result_publication iterables must contain mode-name"
                    f" strings, got {item!r}"
                )
            publication |= _coerce_result_publication(item)
        return publication
    raise ValueError(
        "result_publication must be a ResultPublication, a mode string, or an"
        f" iterable of mode strings; got {value!r}"
    )


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _coerce_agent_vector(ctx: FitContext, name: str) -> None:
    arr = np.asarray(getattr(ctx, name), dtype=np.float64)
    if arr.shape != (ctx.n_agents,):
        raise ValueError(
            f"expected {name} of shape (n_agents,) ="
            f" ({ctx.n_agents},) with n_agents = N * S,"
            f" got {arr.shape}"
        )
    object.__setattr__(ctx, name, _readonly(arr))


@dataclass(frozen=True)
class FitContext:
    """Geometry and interfaces handed from the driver to a formulation for one fit."""

    K: int
    N: int
    S: int
    theta_bounds: tuple[np.ndarray, np.ndarray]
    theta_coef: np.ndarray | None
    agent_weights: np.ndarray | None
    local_ids: np.ndarray
    transport: Transport
    tolerance: float
    slack_coef: Callable[[int], float] | None = None
    theta_init: np.ndarray | None = None
    master_backend: MasterBackend | None = None
    cut_policy: CutPolicy | None = None
    schedule: RepricingSchedule | None = None
    master_params: dict[str, object] = field(default_factory=dict)
    owner_rank: int = 0
    result_publication: ResultPublication = ResultPublication.FULL
    weight_mode: Literal["dense", "distributed"] = "dense"

    @property
    def n_agents(self) -> int:
        return self.N * self.S

    def slack_weight(self, agent_id: int) -> float:
        if self.weight_mode == "dense":
            return float(self.agent_weights[int(agent_id)])  # type: ignore[index]
        assert self.slack_coef is not None
        return float(self.slack_coef(int(agent_id)))

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K (parameter dimension) must be >= 1, got {self.K}")
        if self.N < 1:
            raise ValueError(f"N (observations) must be >= 1, got {self.N}")
        if self.S < 1:
            raise ValueError(f"S (simulations) must be >= 1, got {self.S}")
        if not self.tolerance > 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")

        bounds = self.theta_bounds
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError(
                f"theta_bounds must be a (lower, upper) 2-tuple; got {bounds!r}"
            )
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        if lower.shape != (self.K,):
            raise ValueError(
                f"expected theta_bounds lower of shape (K,) = ({self.K},),"
                f" got {lower.shape}"
            )
        if upper.shape != (self.K,):
            raise ValueError(
                f"expected theta_bounds upper of shape (K,) = ({self.K},),"
                f" got {upper.shape}"
            )
        if np.any(lower > upper):
            bad = np.flatnonzero(lower > upper)
            raise ValueError(
                "theta_bounds must satisfy lower <= upper elementwise;"
                f" violated at indices {bad.tolist()}"
            )
        object.__setattr__(self, "theta_bounds", (_readonly(lower), _readonly(upper)))

        if self.weight_mode not in ("dense", "distributed"):
            raise ValueError(
                "weight_mode must be 'dense' or 'distributed',"
                f" got {self.weight_mode!r}"
            )

        if self.weight_mode == "dense":
            if self.theta_coef is None or self.agent_weights is None:
                raise ValueError(
                    "dense FitContext requires theta_coef and agent_weights"
                )
            if self.slack_coef is not None:
                raise ValueError(
                    "dense FitContext must not set slack_coef;"
                    " pass agent_weights instead"
                )
            _coerce_agent_vector(self, "theta_coef")
            _coerce_agent_vector(self, "agent_weights")
        else:
            if self.theta_coef is not None or self.agent_weights is not None:
                raise ValueError(
                    "distributed FitContext stores no dense theta_coef or"
                    " agent_weights arrays"
                )
            if not callable(self.slack_coef):
                raise ValueError(
                    "distributed FitContext requires a callable slack_coef"
                )

        local_ids = np.asarray(self.local_ids)
        if local_ids.ndim != 1:
            raise ValueError(
                f"expected 1-D local_ids, got shape {local_ids.shape}"
            )
        if not np.issubdtype(local_ids.dtype, np.integer):
            raise ValueError(
                "local_ids must have an integer dtype (global agent ids);"
                f" got dtype {local_ids.dtype}"
            )
        unique, counts = np.unique(local_ids, return_counts=True)
        if unique.size != local_ids.size:
            raise ValueError(
                "local_ids must be unique; duplicated ids:"
                f" {unique[counts > 1].tolist()}"
            )
        if local_ids.size and (
            int(local_ids.min()) < 0 or int(local_ids.max()) >= self.n_agents
        ):
            raise ValueError(
                f"local_ids must lie in [0, n_agents) = [0, {self.n_agents});"
                f" got range [{int(local_ids.min())}, {int(local_ids.max())}]"
            )
        object.__setattr__(self, "local_ids", _readonly(local_ids))

        if self.theta_init is not None:
            theta_init = np.asarray(self.theta_init, dtype=np.float64)
            if theta_init.shape != (self.K,):
                raise ValueError(
                    f"expected theta_init of shape (K,) = ({self.K},),"
                    f" got {theta_init.shape}"
                )
            if np.any(~np.isfinite(theta_init)):
                raise ValueError("theta_init must be finite")
            object.__setattr__(self, "theta_init", _readonly(theta_init))

        if not isinstance(self.transport, Transport):
            raise ValueError(
                "transport must implement combrum.transport.base.Transport;"
                f" got {type(self.transport).__name__}"
            )
        if not isinstance(self.master_params, dict):
            raise ValueError(
                "master_params must be a dict of backend-owned knobs;"
                f" got {type(self.master_params).__name__}"
            )
        if not isinstance(self.owner_rank, int) or not (
            0 <= self.owner_rank < self.transport.size
        ):
            raise ValueError(
                "owner_rank must be an int in [0, size) ="
                f" [0, {self.transport.size}); got {self.owner_rank!r}"
            )
        object.__setattr__(
            self,
            "result_publication",
            _coerce_result_publication(self.result_publication),
        )
