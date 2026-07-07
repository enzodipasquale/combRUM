"""Distributed fit-context assembly without root-held data arrays."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from combrum.context import (
    FitContext,
    ResultPublication,
    _coerce_result_publication,
)
from combrum.engine.agreement import agree_public_int, require_public_object_agreement
from combrum.engine.context_builder import BuiltContext, prepare_warm_cuts
from combrum.formulations import NSlack
from combrum.masters import make_master
from combrum.model import Model
from combrum.transport._common import owned_agent_ids
from combrum.transport.base import CutRow, Transport


@dataclass(frozen=True)
class DistributedObservedPrep:
    """Observation-owned distributed geometry and observed feature rows."""

    K: int
    N: int
    S: int
    n_agents: int
    owned_obs: np.ndarray
    local_ids: np.ndarray
    phi_obs_local: np.ndarray
    empirical_moment: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "owned_obs",
            "local_ids",
            "phi_obs_local",
            "empirical_moment",
        ):
            arr = getattr(self, name)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)


def owned_observation_ids(n_observations: int, rank: int, size: int) -> np.ndarray:
    """Contiguous observation shard owned by ``rank``."""
    N = int(n_observations)
    if N < 1:
        raise ValueError(f"n_observations must be >= 1; got {n_observations}")
    if not 0 <= int(rank) < int(size):
        raise ValueError(f"rank must lie in [0, {size}); got {rank}")
    base, extra = divmod(N, int(size))
    start = int(rank) * base + min(int(rank), extra)
    stop = start + base + (1 if int(rank) < extra else 0)
    return np.arange(start, stop, dtype=np.int64)


def _surface_token(model: Model) -> tuple[str, str, str, bool, bool, bool]:
    if model.observed_features is not None:
        source = "observed_features"
        surface = model.observed_features
    elif model.features is not None:
        source = "features"
        surface = model.features
    else:
        return ("none", "", "", False, False, False)
    return (
        source,
        type(surface).__module__,
        type(surface).__qualname__,
        callable(getattr(surface, "setup_observed", None)),
        callable(getattr(surface, "observed_features_batch", None)),
        callable(getattr(surface, "observed_objective", None)),
    )


def _pricing_surface_token(model: Model) -> tuple[str, str, bool]:
    surface = model.features
    if surface is None:
        return ("none", "", False)
    return (
        type(surface).__module__,
        type(surface).__qualname__,
        callable(getattr(surface, "setup_pricing_agents", None)),
    )


def _distributed_observed_surface(model: Model, transport: Transport) -> object:
    token = _surface_token(model)
    with transport.collective():
        root_token = transport.bcast(token if transport.rank == 0 else None)
        if token != root_token:
            raise ValueError(
                "distributed observed-feature surface must be identical on every rank"
            )
    (
        source,
        _module,
        _qualname,
        _has_obs_setup,
        has_batch,
        has_objective,
    ) = token
    if source == "none" or not has_batch or has_objective:
        raise ValueError(
            "distributed estimation requires an observed-feature surface with"
            " observed_features_batch(observation_ids). setup_observed is"
            " optional; observed_objective belongs to the single-process data"
            " path"
        )
    return model.observed_features if source == "observed_features" else model.features


def _distributed_pricing_surface(model: Model, transport: Transport) -> object | None:
    token = _pricing_surface_token(model)
    with transport.collective():
        root_token = transport.bcast(token if transport.rank == 0 else None)
        if token != root_token:
            raise ValueError(
                "distributed pricing feature surface must be identical on every rank"
            )
    return model.features


def _checked_distributed_phi(value: object, *, n_rows: int, K: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(
            "observed_features_batch must return a numpy.ndarray;"
            f" got {type(value).__name__}"
        )
    expected = (int(n_rows), int(K))
    if value.shape != expected:
        raise ValueError(
            f"observed_features_batch returned shape {value.shape}; expected {expected}"
        )
    if value.dtype != np.float64:
        raise ValueError(
            f"observed_features_batch must return float64 rows; got {value.dtype}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("observed_features_batch must return a C-contiguous array")
    out = value.view()
    out.setflags(write=False)
    return out


def prepare_distributed_observed(
    model: Model,
    *,
    n_observations: int,
    n_simulations: int,
    transport: Transport,
) -> DistributedObservedPrep:
    """Prepare observation-owned feature rows and agent ids for distributed fits."""
    N = agree_public_int("n_observations", n_observations, transport, lower=1)
    S = agree_public_int("n_simulations", n_simulations, transport, lower=1)
    parameters = require_public_object_agreement(
        "model.parameters", model.parameters, transport
    )
    K = agree_public_int("model.parameters.K", parameters.K, transport, lower=1)
    owned_obs = owned_observation_ids(N, transport.rank, transport.size)
    local_ids = owned_agent_ids(N * S, transport.rank, transport.size)
    surface = _distributed_observed_surface(model, transport)
    pricing_surface = _distributed_pricing_surface(model, transport)
    setup_observed = getattr(surface, "setup_observed", None)
    setup_pricing_agents = getattr(pricing_surface, "setup_pricing_agents", None)
    if callable(setup_observed) or callable(setup_pricing_agents):
        with transport.collective():
            if callable(setup_observed):
                setup_observed(transport, owned_obs)
            if callable(setup_pricing_agents):
                setup_pricing_agents(transport, local_ids)
    with transport.collective():
        phi_obs_local = _checked_distributed_phi(
            surface.observed_features_batch(owned_obs),
            n_rows=int(owned_obs.size),
            K=K,
        )
    empirical_sum = np.asarray(
        transport.sum_reproducible(phi_obs_local, owned_obs),
        dtype=np.float64,
    )
    if empirical_sum.shape != (K,):
        raise ValueError(
            "observed feature reduction returned shape"
            f" {empirical_sum.shape}; expected ({K},)"
        )
    return DistributedObservedPrep(
        K=K,
        N=N,
        S=S,
        n_agents=N * S,
        owned_obs=owned_obs,
        local_ids=local_ids,
        phi_obs_local=phi_obs_local,
        empirical_moment=empirical_sum / float(N),
    )


def distributed_c_theta(
    prep: DistributedObservedPrep,
    *,
    obs_weights_local: np.ndarray | None = None,
    transport: Transport,
) -> np.ndarray:
    """Observed-axis ``c_theta`` reduction for one distributed fit."""
    weights = (
        np.ones(prep.owned_obs.size, dtype=np.float64)
        if obs_weights_local is None
        else np.asarray(obs_weights_local, dtype=np.float64)
    )
    if weights.shape != (prep.owned_obs.size,):
        raise ValueError(
            "obs_weights_local must have shape (len(owned_obs),) ="
            f" ({prep.owned_obs.size},); got {weights.shape}"
        )
    local_rows = -float(prep.S) * (weights[:, None] * prep.phi_obs_local)
    c_theta = np.asarray(
        transport.sum_reproducible(local_rows, prep.owned_obs),
        dtype=np.float64,
    )
    if c_theta.shape != (prep.K,):
        raise ValueError(
            f"distributed c_theta has shape {c_theta.shape}; expected ({prep.K},)"
        )
    return c_theta


def build_distributed_fit_context(
    prep: DistributedObservedPrep,
    *,
    model: Model,
    formulation: Any | None = None,
    c_theta: np.ndarray,
    slack_coef: Callable[[int], float],
    transport: Transport,
    owner_rank: int,
    master_backend: str,
    master_params: dict[str, object] | None,
    tolerance: float,
    theta_init: np.ndarray | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    cut_policy: Any | None = None,
    result_publication: ResultPublication | str | Iterable[str],
    guard_master: bool = True,
) -> BuiltContext:
    """Build an NSlack distributed context without dense agent-weight arrays."""
    publication = _coerce_result_publication(result_publication)
    if publication & ResultPublication.BROADCAST:
        raise ValueError(
            "distributed contexts do not support broadcast result publication;"
            " request root-gathered slack, active_set, or dual payloads instead"
        )
    formulation = (
        model.formulation(model.features) if formulation is None else formulation
    )
    if not isinstance(formulation, NSlack):
        raise ValueError("distributed contexts currently support NSlack only")
    c = np.asarray(c_theta, dtype=np.float64)
    if c.shape != (prep.K,):
        raise ValueError(f"c_theta must have shape ({prep.K},); got {c.shape}")
    if not callable(slack_coef):
        raise ValueError("slack_coef must be callable")
    owner = int(owner_rank)
    if not 0 <= owner < transport.size:
        raise ValueError(
            f"owner_rank must lie in [0, {transport.size}); got {owner_rank}"
        )

    def _owner_master() -> object:
        params = master_params
        if params is None and master_backend == "gurobi":
            params = {"Method": 0, "LPWarmStart": 2}
        master_obj = make_master(
            prep.K,
            model.parameters.bounds(),
            c,
            lambda agent_id: float(slack_coef(int(agent_id))),
            backend=master_backend,
            params=params,
            n_agents=None,
        )
        try:
            if warm_cuts is not None:
                master_obj.reinstall(prepare_warm_cuts(formulation, warm_cuts))
            return master_obj
        except Exception:
            master_obj.close()
            raise

    master_obj = None
    if guard_master:
        with transport.collective():
            if transport.rank == owner:
                master_obj = _owner_master()
    elif transport.rank == owner:
        master_obj = _owner_master()
    try:
        ctx = FitContext(
            K=prep.K,
            N=prep.N,
            S=prep.S,
            theta_bounds=model.parameters.bounds(),
            theta_coef=None,
            agent_weights=None,
            slack_coef=slack_coef,
            local_ids=prep.local_ids,
            transport=transport,
            tolerance=tolerance,
            theta_init=theta_init,
            master_backend=master_obj,
            cut_policy=cut_policy,
            master_params=master_params or {},
            owner_rank=owner,
            result_publication=publication,
            weight_mode="distributed",
        )
    except Exception:
        if master_obj is not None:
            master_obj.close()
        raise
    return BuiltContext(ctx=ctx, c_theta=c, empirical_moment=prep.empirical_moment)
