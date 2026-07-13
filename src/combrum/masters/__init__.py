"""Real master-problem backends behind the frozen contract.

:func:`make_master` builds one replication's relaxation host

    minimize    c_theta . theta  +  sum_a u_coef(a) * u_a
    subject to  u_a >= phi_r . theta + epsilon_r   for every installed
                                                   cut row r of agent a
                lower <= theta <= upper,  u_a >= u_lower

on an actual solver, behind :class:`~combrum.master.MasterBackend`.
Loading is lazy: each backend module pulls its solver in only when a
master is built or its availability is probed, so the package resolves
on machines that hold neither solver.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager

import numpy as np

from combrum.master import MasterBackend
from combrum.transport.base import Transport

__all__ = [
    "MasterBackend",
    "make_master",
    "master_environment",
    "resolve_master_backend",
]


def _validate_backend_name(requested: str) -> None:
    if requested not in ("auto", "gurobi", "highs"):
        raise ValueError(
            f"unknown master backend {requested!r};"
            " valid backends: 'auto', 'gurobi', 'highs'"
        )


def _resolve_master_backend_local(
    requested: str,
    *,
    require_quadratic: bool = False,
) -> str:
    if requested == "gurobi":
        return "gurobi"
    if requested == "highs":
        return "highs"

    import combrum.masters.gurobi as gurobi
    import combrum.masters.highs as highs

    if gurobi.available():
        return "gurobi"
    if require_quadratic:
        raise RuntimeError(
            "no quadratic-capable master backend is available:"
            " gurobi could not import and start an environment"
        )
    if highs.available():
        return "highs"
    raise RuntimeError(
        "no master backend is available: neither gurobi nor"
        " highs could import and start an environment"
    )


def _normalize_owner_ranks(
    *,
    size: int,
    owner_rank: int,
    owner_ranks: Iterable[int] | None,
) -> tuple[int, ...]:
    ranks = (owner_rank,) if owner_ranks is None else tuple(owner_ranks)
    if not ranks:
        raise ValueError("owner_ranks must contain at least one rank")
    normalized: list[int] = []
    for rank in ranks:
        r = int(rank)
        if not 0 <= r < size:
            raise ValueError(f"invalid owner rank {rank}: must lie in [0, {size})")
        if r not in normalized:
            normalized.append(r)
    return tuple(normalized)


def _unavailable_flags(
    requested: str,
    *,
    require_quadratic: bool,
) -> np.ndarray:
    unavailable = np.zeros(2, dtype=np.float64)
    if requested == "gurobi":
        import combrum.masters.gurobi as gurobi

        unavailable[0] = 0.0 if gurobi.available() else 1.0
        return unavailable
    if requested == "highs":
        import combrum.masters.highs as highs

        unavailable[1] = 0.0 if highs.available() else 1.0
        return unavailable

    import combrum.masters.gurobi as gurobi

    unavailable[0] = 0.0 if gurobi.available() else 1.0
    if not require_quadratic:
        import combrum.masters.highs as highs

        unavailable[1] = 0.0 if highs.available() else 1.0
    return unavailable


def _choose_from_unavailable(
    requested: str,
    unavailable: np.ndarray,
    *,
    require_quadratic: bool,
) -> str:
    gurobi_unavailable = bool(unavailable[0] > 0.0)
    highs_unavailable = bool(unavailable[1] > 0.0)

    if requested == "gurobi":
        if gurobi_unavailable:
            raise RuntimeError(
                "master backend 'gurobi' is not available on every owner rank"
            )
        return "gurobi"
    if requested == "highs":
        if highs_unavailable:
            raise RuntimeError(
                "master backend 'highs' is not available on every owner rank"
            )
        return "highs"

    if not gurobi_unavailable:
        return "gurobi"
    if require_quadratic:
        raise RuntimeError(
            "no quadratic-capable master backend is available on every"
            " owner rank: gurobi could not import and start an environment"
        )
    if not highs_unavailable:
        return "highs"
    raise RuntimeError(
        "no master backend is available on every owner rank: neither gurobi"
        " nor highs could import and start an environment"
    )


def resolve_master_backend(
    requested: str,
    *,
    require_quadratic: bool = False,
    transport: Transport | None = None,
    owner_rank: int = 0,
    owner_ranks: Iterable[int] | None = None,
) -> str:
    """Resolve a requested backend to one concrete backend string.

    With a multirank transport, only owner ranks probe solver availability and
    every rank derives the same concrete backend from the owner intersection.
    Serial calls keep the original exception types by resolving locally.
    """
    _validate_backend_name(requested)
    if requested == "highs" and require_quadratic:
        raise RuntimeError(
            "master backend 'highs' does not support quadratic penalties"
        )
    if transport is None or transport.size == 1:
        return _resolve_master_backend_local(
            requested, require_quadratic=require_quadratic
        )
    owners = _normalize_owner_ranks(
        size=transport.size, owner_rank=owner_rank, owner_ranks=owner_ranks
    )
    owner_set = set(owners)
    unavailable = np.zeros(2, dtype=np.float64)
    with transport.collective():
        if transport.rank in owner_set:
            unavailable = _unavailable_flags(
                requested, require_quadratic=require_quadratic
            )
    agreed_unavailable = np.asarray(
        transport.batched_max(unavailable), dtype=np.float64
    )
    return _choose_from_unavailable(
        requested,
        agreed_unavailable,
        require_quadratic=require_quadratic,
    )


@contextmanager
def master_environment(backend: str) -> Iterator[object | None]:
    """One caller-owned solver environment for sequential masters.

    Gurobi's environment start is the license checkout, so a run that
    builds one master per bootstrap replication can hold a single
    checkout by passing the yielded environment to every
    :func:`make_master` call. Yields ``None`` for backends without a
    shareable environment; passing that through is a no-op.
    """
    _validate_backend_name(backend)
    if backend != "gurobi":
        yield None
        return
    import combrum.masters.gurobi as gurobi
    import gurobipy

    env = gurobi._started_env(gurobipy)
    try:
        yield env
    finally:
        env.dispose()


def make_master(
    K: int,
    theta_bounds: tuple[np.ndarray, np.ndarray],
    c_theta: np.ndarray,
    u_coef: Callable[[int], float] | np.ndarray,
    *,
    backend: str = "auto",
    params: Mapping[str, object] | None = None,
    n_agents: int | None = None,
    env: object | None = None,
) -> MasterBackend:
    """One relaxation host on a real solver.

    ``K`` is the theta dimension; ``theta_bounds`` is the ``(lower,
    upper)`` box pair, each shaped ``(K,)``; ``c_theta`` is the ``(K,)``
    linear theta objective. ``u_coef`` gives each agent's slack objective
    coefficient — a callable from agent id, or a 1-D array indexed by
    agent id, which the backends read without one Python call per agent.
    Either form is applied verbatim; the caller owns the weighting.
    ``params`` are backend-owned solver knobs,
    passed through opaquely to whichever solver hosts the master. The
    optional ``params["u_lower_bound"]`` defaults to ``0.0``; set it to
    ``None`` for a solver-native free epigraph variable.

    ``backend`` picks the host: ``"gurobi"`` or ``"highs"`` bind that
    solver; ``"auto"`` takes the first available of (gurobi, highs).
    Available means the solver package imports and an environment
    actually starts; an installed gurobipy whose license cannot start
    an environment reads as unavailable, so auto falls through to a
    backend that can actually solve.

    ``n_agents`` pre-declares that many slack (epigraph) columns up front for
    a fixed, deterministic column structure; ``None`` adds each agent's column
    lazily on its first cut.

    ``env`` is an optional caller-owned solver environment from
    :func:`master_environment`; the master then skips its own environment
    start and never disposes the shared one. Only gurobi has a shareable
    environment.
    """
    _validate_backend_name(backend)
    if backend == "auto":
        backend = _resolve_master_backend_local(backend)
    if backend == "gurobi":
        import combrum.masters.gurobi as gurobi

        return gurobi.GurobiMaster(
            K, theta_bounds, c_theta, u_coef, params=params, n_agents=n_agents, env=env
        )
    if env is not None:
        raise ValueError("env is only meaningful for the gurobi backend")
    import combrum.masters.highs as highs

    return highs.HighsMaster(
        K, theta_bounds, c_theta, u_coef, params=params, n_agents=n_agents
    )
