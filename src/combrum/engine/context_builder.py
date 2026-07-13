"""Shared fit-context builder: single owner of estimation context assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from combrum.context import FitContext, ResultPublication
from combrum.engine.observed import ObservedObjectiveCache, observed_objective
from combrum.formulations import OneSlack
from combrum.masters import make_master, master_environment, resolve_master_backend
from combrum.parameters import Parameters
from combrum.policies import CutPolicy
from combrum.result import FitResult
from combrum.transport.base import CutRow, Transport

__all__ = [
    "BuiltContext",
    "build_fit_context",
    "master_environment",
    "prepare_warm_cuts",
    "resolve_master_backend",
]


@dataclass(frozen=True)
class BuiltContext:
    """Assembled fit context plus observed-data quantities folded into a result."""

    ctx: FitContext
    c_theta: np.ndarray
    empirical_moment: np.ndarray


def prepare_warm_cuts(formulation: Any, rows: Sequence[CutRow]) -> tuple[CutRow, ...]:
    """Let a formulation normalize persisted cut rows before reinstall."""

    cut_rows = tuple(rows)
    prepare = getattr(formulation, "prepare_warm_cuts", None)
    if not callable(prepare):
        return cut_rows
    return tuple(prepare(cut_rows))


def _master_params_for_backend(
    backend: str, master_params: dict[str, object] | None
) -> dict[str, object] | None:
    if backend != "gurobi":
        return master_params
    params: dict[str, object] = {"Method": 0, "LPWarmStart": 2}
    if master_params is not None:
        params.update(master_params)
    return params


def build_fit_context(
    parameters: Parameters,
    *,
    observables: Sequence[Any],
    observed_bundles: np.ndarray,
    shocks: np.ndarray,
    formulation: Any,
    features: object,
    observed_features: object | None,
    transport: Transport,
    master_backend: str = "auto",
    resolved_master_backend: str | None = None,
    master_params: dict[str, object] | None = None,
    tolerance: float = 1e-6,
    schedule: Any | None = None,
    weights: np.ndarray | None = None,
    warm_start: FitResult | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    cut_policy: CutPolicy | None = None,
    master: Any = None,
    master_env: object | None = None,
    result_publication: ResultPublication | str | Iterable[str] = (
        ResultPublication.FULL
    ),
    observed_cache: ObservedObjectiveCache | None = None,
) -> BuiltContext:
    """Assemble the per-rank fit context and master.

    Args:
        weights: Per-observation ``(N,)`` row, or ``None`` for unit weights.
            Expanded to the ``(N*S,)`` agent space under ``a = s*N + i``
            (``theta_coef[a] = agent_weights[a] = weights[a % N]``) and applied
            to both ``c_theta`` and the per-agent epigraph/aggregate coefficients.
        warm_start: ``FitResult`` whose ``theta_hat`` becomes the proximal
            anchor ``theta_init``, or ``None`` for a cold start.
        warm_cuts: Cut rows reinstalled onto the fresh master via
            :meth:`MasterBackend.reinstall` before the formulation's setup solve,
            or ``None`` for a fresh master. ``reinstall`` replaces the installed
            set, so setup rebuilds bookkeeping from the warm relaxation.
        master: Live ``MasterBackend`` to reuse on rank 0 (skips ``make_master``
            and ``reinstall``), or ``None`` to build fresh. ``c_theta`` /
            ``empirical_moment`` are recomputed either way.
        master_env: Caller-owned solver environment from
            :func:`combrum.masters.master_environment`, shared by sequential
            masters (one license checkout per run, not per build), or ``None``
            for a master-owned environment.
    """
    backend_for_master = resolved_master_backend or master_backend
    K = parameters.K
    observed_bundles = np.asarray(observed_bundles)
    shocks = np.asarray(shocks)
    N = len(observables)
    if observed_bundles.ndim != 2 or observed_bundles.shape[0] != N:
        raise ValueError(
            "observed_bundles must be 2-D (N, M) with N = len(observables) ="
            f" {N}; got shape {observed_bundles.shape}"
        )
    if shocks.ndim < 2 or shocks.shape[0] != N:
        raise ValueError(
            f"shocks must have shape (N, S, ...) with N = {N}; got shape {shocks.shape}"
        )
    S = int(shocks.shape[1])
    n_agents = N * S

    local_ids = np.arange(transport.rank, n_agents, transport.size, dtype=np.int64)

    if weights is None:
        theta_coef = np.ones(n_agents, dtype=np.float64)
        agent_weights = np.ones(n_agents, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (N,):
            raise ValueError(
                "weights are PER-OBSERVATION and must have shape (N,) ="
                f" ({N},); got {w.shape}. The builder expands them to the"
                " (N*S,) agent space, so a caller passes the (N,) observation"
                " weight row, never a pre-expanded vector."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        agent_w = np.tile(w, S)
        theta_coef = agent_w
        agent_weights = agent_w

    theta_init = (
        None
        if warm_start is None
        else np.asarray(warm_start.theta_hat, dtype=np.float64)
    )

    c_theta, empirical_moment = observed_objective(
        K=K,
        N=N,
        theta_coef=theta_coef,
        observed_bundles=observed_bundles,
        local_ids=local_ids,
        transport=transport,
        features=features,
        observed_features=observed_features,
        cache=observed_cache,
    )

    def _rank0_master() -> Any:
        if master is not None:
            return master
        u_coef = (
            (lambda agent_id: 1.0)
            if isinstance(formulation, OneSlack)
            else agent_weights
        )
        params = _master_params_for_backend(backend_for_master, master_params)
        master_obj = make_master(
            K,
            parameters.bounds(),
            c_theta,
            u_coef,
            backend=backend_for_master,
            params=params,
            n_agents=None if isinstance(formulation, OneSlack) else n_agents,
            env=master_env,
        )
        if warm_cuts is not None:
            master_obj.reinstall(prepare_warm_cuts(formulation, warm_cuts))
        return master_obj

    master_obj = None
    if transport.size == 1:
        master_obj = _rank0_master()
    else:
        with transport.collective():
            if transport.rank == 0:
                master_obj = _rank0_master()
    ctx = FitContext(
        K=K,
        N=N,
        S=S,
        theta_bounds=parameters.bounds(),
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=tolerance,
        theta_init=theta_init,
        master_backend=master_obj,
        schedule=schedule,
        cut_policy=cut_policy,
        master_params=master_params or {},
        result_publication=result_publication,
    )
    return BuiltContext(ctx=ctx, c_theta=c_theta, empirical_moment=empirical_moment)
