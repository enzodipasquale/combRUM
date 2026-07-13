"""The B=1 convergence loop over the engine phase path.

The driver runs the per-rank iterate loop (setup, schedule mask, static-anchor
penalty schedule, full-sweep + pure-LP convergence, warm-start) and drives the
engine phases ``contribute`` -> reduce+exchange -> ``finalise`` ->
``apply_step``. A live-set mask over the replication set lets a converged
replication retire independently.

The driver owns the cross-rank collectives (through
:func:`~combrum.engine.fitstep.fit_step`) so the formulation stays
transport-passive.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from combrum.activity import (
    ActivityRun,
    RowGenFinal,
    RowGenIteration,
    RowGenStart,
    _activity_details,
    _object_name,
)
from combrum.context import FitContext
from combrum.engine.agreement import (
    callback_convergence_floor,
    collective_call,
)
from combrum.engine.fitstep import fit_step
from combrum.formulation import FormulationResult
from combrum.informed_schedule import DualConcentration
from combrum.interface_resolution import Resolution, resolve
from combrum.oracle import Oracle
from combrum.rowgen import RowGenStep
from combrum.schedule import RepricingSchedule, ResolveAll


def _validate_schedule_formulation(
    formulation: object, schedule: RepricingSchedule | None
) -> None:
    if schedule is None:
        return
    if not isinstance(schedule, RepricingSchedule):
        raise TypeError(
            "schedule must be a RepricingSchedule such as ResolveAll,"
            " RoundRobin, or DualInformed; combrum.TimeoutSchedule is for"
            " timeout callbacks passed through iteration_callback"
        )
    cls = type(formulation)
    if (
        cls.__module__ != "combrum.formulations.oneslack"
        or cls.__qualname__ != "OneSlack"
    ):
        return
    if isinstance(schedule, ResolveAll):
        return
    raise ValueError(
        "OneSlack requires full re-pricing every iteration;"
        " schedule must be None or ResolveAll"
    )


def _coerce_iteration_count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"expected {name} to be an integer >= {minimum}, got bool")
    try:
        coerced = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(
            f"expected {name} to be an integer >= {minimum},"
            f" got {type(value).__name__}"
        ) from exc
    if coerced < minimum:
        raise ValueError(f"expected {name} >= {minimum}, got {value}")
    return coerced


def _validate_loop_controls(
    max_iterations: int,
    qp_weight: float,
    qp_iterations: int,
    penalty_ref: str,
    min_iterations: int = 0,
) -> None:
    max_value = _coerce_iteration_count("max_iterations", max_iterations, 1)
    min_value = _coerce_iteration_count("min_iterations", min_iterations, 0)
    if min_value > max_value:
        raise ValueError(
            "min_iterations must be <= max_iterations;"
            f" got min_iterations={min_iterations!r},"
            f" max_iterations={max_iterations!r}"
        )
    if isinstance(qp_weight, (bool, np.bool_)):
        raise TypeError("qp_weight must be a finite float >= 0; got bool")
    try:
        qp_value = float(qp_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"qp_weight must be a finite float >= 0; got {type(qp_weight).__name__}"
        ) from exc
    if not np.isfinite(qp_value):
        raise ValueError(f"qp_weight must be finite; got {qp_weight!r}")
    if qp_value < 0.0:
        raise ValueError(f"qp_weight must be >= 0; got {qp_weight!r}")
    if qp_value > 0.0 and qp_iterations <= 0:
        raise ValueError(
            "qp_weight>0 needs qp_iterations>=1 so the weight reaches 0"
            f" (got qp_iterations={qp_iterations!r})"
        )
    if penalty_ref not in ("dynamic", "static"):
        raise ValueError(
            f"penalty_ref must be 'dynamic' or 'static' (got {penalty_ref!r})"
        )


@dataclass(frozen=True)
class LoopConfig:
    """Loop controls for the driver (not the agent space, which lives on
    :class:`~combrum.context.FitContext`).
    """

    max_iterations: int
    schedule: RepricingSchedule | None = None
    qp_weight: float = 0.0
    qp_iterations: int = 0
    penalty_ref: str = "static"
    min_iterations: int = 0
    iteration_callback: Callable[[int, Oracle], int | None] | None = None
    activity: ActivityRun | None = None

    def __post_init__(self) -> None:
        _validate_loop_controls(
            self.max_iterations,
            self.qp_weight,
            self.qp_iterations,
            self.penalty_ref,
            self.min_iterations,
        )


@dataclass(frozen=True)
class LoopDiagnostics:
    converged: bool
    iterations: int
    cuts_admitted: int
    final_penalty_weight: float = 0.0


@dataclass(frozen=True)
class LoopOutcome:
    result: FormulationResult
    diagnostics: LoopDiagnostics


def _resolve_price(oracle: Oracle, transport) -> Resolution:
    return resolve(
        oracle,
        surface="price",
        default_name="price",
        optimized_name="price_batch",
        default_func=Oracle.price,
        optimized_func=Oracle.price_batch,
        transport=transport,
    )


def run_fit(
    ctx: FitContext,
    oracle: Oracle,
    formulation: RowGenStep,
    config: LoopConfig,
    *,
    demand_sink: Callable[..., None] | None = None,
    suppress_close: bool = False,
) -> LoopOutcome:
    """Run the B=1 phase-path convergence loop to a published answer.

    ``ctx`` is a built :class:`~combrum.context.FitContext` (the master, if
    any, already on ``ctx.master_backend`` on ``ctx.owner_rank``); ``oracle``
    and ``formulation`` are constructed but not yet set up. The driver runs::

        setup (oracle + formulation + resolve the price interface)
        repeat: solve -> [schedule mask] -> fit_step (price/contribute/
                 reduce+exchange/finalise/apply_step) -> stop check
        result()

    ``demand_sink`` is an optional read-only observer of every iteration's
    priced demands. When present it only reads the demands the price phase
    already produced; attaching one adds no communication and leaves the
    reductions untouched.

    ``suppress_close`` keeps the master alive past this fit. Default ``False``
    closes ``ctx.master_backend`` in the ``finally``; ``True`` skips that close
    so a persistent driver can hold one master across an outer search and
    warm-solve it again, taking over the close obligation. Oracle teardown and
    formulation dispose run on both paths.
    """
    transport = ctx.transport
    owner_rank = ctx.owner_rank
    n_agents = ctx.n_agents
    local_ids = np.asarray(ctx.local_ids, dtype=np.int64)
    schedule = config.schedule
    _validate_schedule_formulation(formulation, schedule)
    max_iterations = int(config.max_iterations)
    tolerance = ctx.tolerance

    n_reps = 1
    live = np.ones(n_reps, dtype=bool)

    static_ref = (
        np.zeros(ctx.K, dtype=np.float64)
        if ctx.theta_init is None
        else np.asarray(ctx.theta_init, dtype=np.float64)
    )

    penalty_on = config.qp_weight > 0.0 and config.qp_iterations > 0
    master = ctx.master_backend

    base_convergence_floor = (
        max(config.min_iterations, config.qp_iterations + 1)
        if penalty_on
        else config.min_iterations
    )
    iteration_callback = config.iteration_callback

    converged = False
    iterations = 0
    cuts_admitted = 0
    activity = config.activity
    activity_enabled = activity is not None and activity.enabled
    activity_details = activity_enabled and _activity_details(activity.config.level)
    run_t0 = time.perf_counter() if activity_enabled else None
    if activity_enabled:
        activity.emit(
            RowGenStart(
                run_id=activity.config.run_id,
                label=activity.config.label,
                n_obs=ctx.N,
                n_simulations=ctx.S,
                n_parameters=ctx.K,
                n_agents=ctx.n_agents,
                tolerance=tolerance,
                max_iterations=max_iterations,
                min_iterations=config.min_iterations,
                schedule=_object_name(schedule),
                cut_policy=_object_name(ctx.cut_policy),
                rank=transport.rank,
                world_size=transport.size,
                transport=type(transport).__name__,
                activity_level=activity.config.level,
            )
        )
    last_resolved = (
        np.full(n_agents, -1, dtype=np.int64) if schedule is not None else None
    )
    force_full = True
    priced_weight = 0.0
    last_solve_weight = 0.0
    last_violation: float | None = None
    last_logged_gap: float | None = None
    try:
        collective_call(transport, lambda: oracle.setup(transport, local_ids))
        formulation.setup(ctx)
        price_resolution = _resolve_price(oracle, transport)
        for it in range(max_iterations):
            if not live.any():
                break
            convergence_floor = callback_convergence_floor(
                name="iteration_callback floor",
                callback=iteration_callback,
                iteration=it,
                oracle=oracle,
                base_floor=base_convergence_floor,
                transport=transport,
            )
            iter_t0 = time.perf_counter() if activity_details else 0.0
            theta = formulation.solve()
            if schedule is None:
                this_full = True
                scheduled_local_ids = local_ids
            else:
                if force_full or priced_weight > 0.0:
                    mask = np.ones(n_agents, dtype=bool)
                else:

                    def _dual_payload() -> DualConcentration | None:
                        if transport.rank != owner_rank:
                            return None
                        assert master is not None
                        return DualConcentration.from_cut_duals(master.dual_values())

                    payload = collective_call(transport, _dual_payload)
                    payload = transport.bcast(payload, root=owner_rank)
                    mask = schedule.select(
                        it,
                        n_agents,
                        dual=payload,
                        last_resolved=last_resolved,
                    )
                this_full = bool(mask.all())
                scheduled_local_ids = (
                    local_ids if this_full else local_ids[mask[local_ids]]
                )

            weight_t = (
                config.qp_weight if penalty_on and it < config.qp_iterations else 0.0
            )

            needs_penalty_solve = penalty_on and (
                weight_t > 0.0 or last_solve_weight > 0.0
            )
            before_apply = None
            if needs_penalty_solve:
                ref_t = theta if config.penalty_ref == "dynamic" else static_ref
                prepare_penalty = getattr(formulation, "prepare_penalty_solve", None)
                if callable(prepare_penalty):
                    prepare_penalty(ref_t, weight_t)
                else:

                    def _penalty_solve(
                        weight: float = weight_t,
                        ref: np.ndarray = ref_t,
                    ) -> None:
                        if transport.rank != owner_rank:
                            return
                        assert master is not None
                        master.set_penalty(ref, weight)
                        master.solve()

                    def _before_apply() -> None:
                        collective_call(transport, _penalty_solve)

                    before_apply = _before_apply

            step = fit_step(
                formulation,
                transport=transport,
                price_resolution=price_resolution,
                theta=theta,
                scheduled_local_ids=scheduled_local_ids,
                owner_rank=ctx.owner_rank,
                before_apply=before_apply,
                demand_sink=demand_sink,
            )
            if needs_penalty_solve:
                last_solve_weight = weight_t
            cuts_admitted += step.progressed
            if schedule is not None:
                last_resolved[mask] = it
            iterations += 1
            last_violation = float(step.violation)
            hit_tolerance = step.violation <= tolerance
            accepted = (
                hit_tolerance
                and this_full
                and iterations >= convergence_floor
                and priced_weight == 0.0
            )

            if activity_details:
                now = time.perf_counter()
                objective = (
                    master.objective()
                    if transport.rank == owner_rank and master is not None
                    else None
                )
                active_cuts = (
                    master.n_active_cuts
                    if transport.rank == owner_rank and master is not None
                    else cuts_admitted
                )
                activity.emit(
                    RowGenIteration(
                        run_id=activity.config.run_id,
                        label=activity.config.label,
                        iteration=it,
                        gap=float(step.violation),
                        gap_delta=(
                            None
                            if last_logged_gap is None
                            else float(step.violation) - last_logged_gap
                        ),
                        objective=(None if objective is None else float(objective)),
                        active_cuts=int(active_cuts),
                        cuts_added=int(step.progressed),
                        violation_count=int(step.n_candidates),
                        n_priced_local=int(step.n_priced),
                        n_inexact_local=int(step.n_inexact),
                        reduce_rounds=int(step.reduce_rounds),
                        exchange_rounds=int(step.exchange_rounds),
                        full_sweep=this_full,
                        convergence_candidate=accepted,
                        price_seconds=float(step.pricing_seconds),
                        master_seconds=float(step.master_seconds),
                        iteration_seconds=(float(now - iter_t0) if iter_t0 else None),
                        total_seconds=(
                            float(now - run_t0) if run_t0 is not None else None
                        ),
                    )
                )
                last_logged_gap = float(step.violation)

            if accepted:
                live[:] = False
                converged = True
                break
            force_full = hit_tolerance
            priced_weight = weight_t
        result = formulation.result()
        if activity_enabled:
            activity.emit(
                RowGenFinal(
                    run_id=activity.config.run_id,
                    label=activity.config.label,
                    converged=bool(converged),
                    termination_reason=(
                        "converged"
                        if converged
                        else (
                            "max_iterations"
                            if iterations >= max_iterations
                            else "stopped"
                        )
                    ),
                    iterations=int(iterations),
                    final_gap=last_violation,
                    objective=float(result.objective),
                    active_cuts=int(result.n_active_cuts),
                    wall_seconds=(
                        float(time.perf_counter() - run_t0)
                        if run_t0 is not None
                        else None
                    ),
                )
            )
    finally:
        oracle.teardown()
        formulation.dispose()
        if master is not None and not suppress_close:
            master.close()
    return LoopOutcome(
        result=result,
        diagnostics=LoopDiagnostics(
            converged=converged,
            iterations=iterations,
            cuts_admitted=cuts_admitted,
            final_penalty_weight=last_solve_weight,
        ),
    )
