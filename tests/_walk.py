"""Test-local end-to-end driver for the row-generation formulations.

One walk, both methods, any transport: build the per-rank fit context
over an interleaved shard (``a % size == rank``), construct the master
on rank 0 only (every other rank passes ``master_backend=None``), then
run setup -> loop{solve -> price local shard -> evaluate -> update} ->
result, stopping at ``violation <= tolerance``.

All reference-comparison arithmetic lives here, never in the
formulations: the master's linear theta objective is
``c_theta = -sum_a theta_coef_a * phi_a(observed_a)`` and the published
criterion is the master objective itself.

Underscore-prefixed module: test support, never collected by pytest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from _family_oracles import FamilyProblem
from combrum.context import FitContext
from combrum.formulation import Formulation, FormulationResult
from combrum.formulations import OneSlack
from _support.constants import MAX_ITERATIONS, TOLERANCE
from combrum.informed_schedule import DualConcentration
from combrum.masters import make_master
from combrum.policies import CutPolicy
from combrum.schedule import RepricingSchedule
from combrum.steprecord import ListTraceSink, StepRecord
from combrum.transport.base import CutRow, Transport


@dataclass(frozen=True)
class WalkOutcome:
    """One walk's published answer plus the walk-level bookkeeping."""

    result: FormulationResult
    objective: float  # reference-comparable row-generation objective
    converged: bool
    iterations: int
    cuts_admitted: int
    #: Total agent-prices over the run (sum of the global re-pricing mask
    #: across iterations) — the schedule's pricing-budget metric. Equals
    #: ``iterations * n_agents`` under the default full-sweep schedule.
    pricing_calls: int = 0
    #: Per-iteration global re-pricing mask (only recorded when a schedule
    #: is supplied), so a test can verify pricing reduction and the forced
    #: revisit bound from the trace.
    schedule_masks: tuple[np.ndarray, ...] = ()
    #: Per-iteration size of the bcast dual-concentration payload — the
    #: support count, never ``n_agents`` (the O(support) payload claim).
    payload_supports: tuple[int, ...] = ()
    #: Penalty weight in effect for the master solve that produced the
    #: published theta. Zero whenever no penalty ran, and — by the decay
    #: floor — zero at every accepted convergence, so a gate can assert
    #: the terminating solve was a pure LP from the outcome alone.
    final_penalty_weight: float = 0.0
    #: Max installed-cut count the master held over the run (root reading
    #: of ``len(master.extract_cuts())`` each iteration). A stripping gate
    #: bounds it: a retirement policy that retires deeply-slack cuts mid-run
    #: holds a lower peak than the unstripped accumulation, at an unchanged
    #: estimate. Zero on every non-root rank and on a run that never reaches
    #: the loop.
    peak_installed_cuts: int = 0
    #: Per-iteration snapshot of the master's installed ``CutRow`` tuple
    #: (root-only, opt-in via ``capture_installed``), one entry per iteration
    #: in loop order, taken after that iteration's ``update``. Empty by default.
    #: The hard-clause gate replays the per-agent
    #: vs batched features path and asserts these snapshots identical.
    installed_snapshots: tuple[tuple[CutRow, ...], ...] = ()
    #: Per-iteration ``StepRecord`` stream (root-only, opt-in via
    #: ``capture_steprecords``), captured via a ``ListTraceSink`` on the
    #: formulation. Each record holds one iteration's filter-chain inputs over
    #: their full pre-filter domain (reduced costs, violations, purge inputs,
    #: install key sets, aggregate raw/bytes, priced demand/feature stream).
    #: Empty by default and on every non-root rank. The wholesale-capture gate
    #: replays the per-agent vs batched path across shard permutations and
    #: asserts these field-by-field: discrete identical, continuous within
    #: ``1e-13``.
    step_records: tuple[StepRecord, ...] = ()
    #: Per-iteration ``DualConcentration`` payload (support ``agent_ids`` +
    #: per-agent ``max_weights``) the schedule branch reads. Captured here
    #: because it is driver-owned — computed in this loop, not inside a
    #: formulation — so it is the one such field visible in run_walk rather than
    #: formulation-internal. Bcast on root, so identical on every rank;
    #: populated only under ``capture_steprecords`` with a ``schedule`` active.
    #: The wholesale gate compares it across feature + shard axes: support
    #: identical, max_weights within ``1e-13``.
    schedule_concentrations: tuple[DualConcentration, ...] = ()


def run_walk(
    arrays: Mapping[str, np.ndarray],
    problem: FamilyProblem,
    formulation_cls: type[Formulation],
    transport: Transport,
    *,
    backend: str,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    cut_policy: CutPolicy | None = None,
    schedule: RepricingSchedule | None = None,
    qp_weight: float = 0.0,
    decay: int = 0,
    penalty_ref: str = "dynamic",
    min_iterations: int = 0,
    theta_init: np.ndarray | None = None,
    warm_start: FormulationResult | None = None,
    capture_installed: bool = False,
    capture_steprecords: bool = False,
) -> WalkOutcome:
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    # Warm-start: a prior fit's published theta is the proximal anchor
    # of the refit unless the caller pins one explicitly. The cut set the
    # prior fit installed is replayed onto the master below (root-only);
    # this only chooses where a penalty would be aimed, so it is harmless
    # on the pure-LP path that ignores the anchor entirely.
    if warm_start is not None and theta_init is None:
        theta_init = warm_start.theta_hat
    local_ids = np.arange(
        transport.rank, n_agents, transport.size, dtype=np.int64
    )
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    master = None
    if transport.rank == 0:
        c_theta = np.zeros(problem.K, dtype=np.float64)
        for a in range(n_agents):
            phi_obs = problem.observed_features(a, observed[a])
            c_theta -= theta_coef[a] * np.asarray(phi_obs, dtype=np.float64)
        u_coef = (
            (lambda agent_id: 1.0)
            if formulation_cls is OneSlack
            else (lambda agent_id: float(agent_weights[agent_id]))
        )
        # The captured references solve every gurobi master with
        # warm-started primal simplex; on a degenerate optimal face the
        # published vertex is a function of that configuration, so the
        # walk mirrors it (their highs masters run stock options).
        params = (
            {"Method": 0, "LPWarmStart": 2} if backend == "gurobi" else None
        )
        master = make_master(
            problem.K,
            problem.theta_bounds,
            c_theta,
            u_coef,
            backend=backend,
            params=params,
            # Pre-declare per-agent u-columns only for the per-agent-slack
            # formulation; OneSlack carries one aggregate slack, so n_agents
            # columns would be spurious (and degeneracy-inducing).
            n_agents=None if formulation_cls is OneSlack else n_agents,
        )
        # Warm-start: replay the prior fit's installed cuts onto the
        # fresh master before setup. reinstall replaces the installed set,
        # and NSlack.setup() then solves and rebuilds its bookkeeping
        # entirely from extract_cuts()/theta(), so it adopts the
        # pre-installed relaxation with no change of its own. The rows
        # arrive in the canonical order extract_cuts() emitted, so the
        # warm master is the byte-identical relaxation the prior fit ended
        # on. Only a cut-carrying method publishes an active_set; a cutless
        # prior fit (active_set is None) seeds nothing.
        if warm_start is not None and warm_start.active_set is not None:
            master.reinstall(warm_start.active_set)
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=tolerance,
        master_backend=master,
        cut_policy=cut_policy,
    )
    oracle = problem.oracle
    formulation = formulation_cls(problem.features)
    # Opt-in: attach a capturing sink so the formulation emits one StepRecord
    # per iteration into `step_sink.records`. Default off, and set_trace_sink
    # is a no-op-when-None write, so capture_steprecords=False stays identical
    # to the pre-capture run. The sink attaches on every rank, but only the
    # rank holding the master populates the admit/purge/install fields; a
    # non-root record carries just the rank-local contribute fields.
    step_sink: ListTraceSink | None = (
        ListTraceSink() if capture_steprecords else None
    )
    if step_sink is not None:
        formulation.set_trace_sink(step_sink)
    converged = False
    iterations = 0
    cuts_admitted = 0
    pricing_calls = 0
    # Max installed-cut count over the run, tracked root-only — the master
    # lives on rank 0 alone, so no other rank can read it. extract_cuts()
    # is a pure accessor (it sorts the installed keys into a tuple and
    # touches no solver state), so reading it each iteration cannot move
    # theta or the solve path: the no-warm/no-strip run stays byte-identical
    # in its published answer, which the parity and rank-invariance gates
    # confirm. The metric earns its keep on the stripping path, where a
    # retired cut lowers this peak below the unstripped accumulation.
    peak_installed_cuts = 0
    # Per-iteration installed-row snapshots, captured only when the caller
    # opts in. Same pure extract_cuts() accessor as peak_installed_cuts, so
    # the default capture_installed=False path stays byte-identical and never
    # builds this list.
    installed_snapshots: list[tuple[CutRow, ...]] = []
    masks: list[np.ndarray] = []
    supports: list[int] = []
    # Per-iteration Schedule field, captured only under capture_steprecords
    # (and only when a schedule is active, where the payload is computed). Built
    # nowhere on the default path, so it never moves the byte-identical run.
    schedule_concentrations: list[DualConcentration] = []
    # Last iteration each agent was re-priced; identical on every rank (the
    # mask is global and deterministic), so the schedule needs no extra comm.
    last_resolved = np.full(n_agents, -1, dtype=np.int64)
    # The first sweep is always full: no solve has produced a dual signal
    # yet, and a partial sweep that reports convergence is re-certified by
    # a full sweep before the walk stops (the unpriced agents may still be
    # violated). force_full carries that "certify next" obligation.
    force_full = True
    # Quadratic-penalty decay schedule. qp_weight == 0 disables it
    # entirely: not one set_penalty call is made and the walk is the
    # byte-identical pure-LP run every existing caller already gets. When
    # active the weight decays linearly to exactly zero over `decay`
    # iterations, so the terminating solve is a pure LP with valid LP
    # duals (the whole point of the static-anchor schedule).
    penalty_on = qp_weight > 0.0 and decay > 0
    if qp_weight > 0.0 and decay <= 0:
        raise ValueError(
            f"qp_weight>0 needs decay>=1 so the weight reaches 0; got"
            f" decay={decay!r}"
        )
    if penalty_ref not in ("dynamic", "static"):
        raise ValueError(
            f"penalty_ref must be 'dynamic' or 'static'; got {penalty_ref!r}"
        )
    # The static anchor is fixed for the whole walk: the seed if given,
    # else the origin. The dynamic ref is recomputed each iteration from
    # the current theta (the proximal point) and so needs no
    # precomputation here.
    static_ref = (
        np.zeros(problem.K, dtype=np.float64)
        if theta_init is None
        else np.asarray(theta_init, dtype=np.float64)
    )
    # The decay floor enforces effective_min_iters >= decay+1: the
    # walk may not accept convergence before the weight has fully decayed
    # to 0. The real correctness guard, though, is the priced_weight==0
    # check at the stop rule (below) — it refuses any convergence whose
    # theta did not come from a pure-LP solve, whatever the floor constant.
    convergence_floor = (
        max(min_iterations, decay + 1) if penalty_on else min_iterations
    )
    # Weight behind the theta the current iteration prices: set by the
    # previous iteration's solve, 0 for the setup solve (no penalty) and
    # for every non-penalty walk. The stop rule certifies convergence
    # only when this is 0, so the published answer is always a pure-LP
    # solution with valid LP duals.
    priced_weight = 0.0
    # Weight behind the published theta, surfaced in the outcome. Set on
    # every penalty solve and re-affirmed on the accepted convergence.
    last_solve_weight = 0.0
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        if master is not None:
            # Seed the peak with the post-setup count: a warm-started run
            # already holds the prior fit's whole cut set here, before the
            # first iteration ever prices.
            peak_installed_cuts = len(master.extract_cuts())
        for it in range(max_iterations):
            theta = formulation.solve()
            if schedule is None:
                mask = np.ones(n_agents, dtype=bool)
            else:
                # Condense the master's per-cut duals into the O(support)
                # concentration payload on root, broadcast it, and let every
                # rank derive the identical mask rank-locally.
                payload = None
                if transport.rank == 0:
                    payload = DualConcentration.from_cut_duals(
                        master.dual_values()
                    )
                payload = transport.bcast(payload, root=0)
                supports.append(int(payload.agent_ids.size))
                if capture_steprecords:
                    # The full Schedule field (not just its size): the
                    # DualConcentration the gate compares per support agent.
                    schedule_concentrations.append(payload)
                mask = (
                    np.ones(n_agents, dtype=bool)
                    if force_full
                    else schedule.select(
                        it,
                        n_agents,
                        dual=payload,
                        last_resolved=last_resolved,
                    )
                )
            this_full = bool(mask.all())
            local_masked = local_ids[mask[local_ids]]
            demands = {int(a): oracle.price(theta, int(a)) for a in local_masked}
            pricing_calls += int(mask.sum())
            evaluated = formulation.evaluate(demands)
            weight_t = (
                qp_weight * max(0.0, 1.0 - it / decay) if penalty_on else 0.0
            )
            if penalty_on and transport.rank == 0:
                # Re-aim the master objective at the decayed weight and
                # solve it here, before update(). weight_t hits exactly 0
                # at it >= decay, so from then on this solve is a pure LP
                # (set_penalty with weight 0 reverts the quadratic term
                # entirely, never zeroes it approximately). ref_t is the
                # current theta (dynamic proximal point) or the fixed
                # anchor (static). Solving here — not leaving it to
                # update()'s cut-gated solve — is what makes the weight
                # actually move theta: a penalized iterate re-prices
                # already-installed bundles, so update() would admit no
                # cut and never re-solve, freezing theta at a stale weight.
                # update()'s _state() then reads this solved theta and
                # broadcasts it (its own solve only fires, redundantly at
                # the same weight, when a genuinely new cut is admitted).
                ref_t = theta if penalty_ref == "dynamic" else static_ref
                master.set_penalty(ref_t, weight_t)
                master.solve()
                # This solve produced the theta update() is about to adopt
                # and publish, so it is the weight behind the next answer.
                last_solve_weight = weight_t
            progressed = formulation.update(evaluated)
            cuts_admitted += progressed
            if master is not None:
                # Post-update installed set: update() has already added this
                # step's cuts and run any policy retirement on root, so this
                # reading is the master's true installed rows this iteration —
                # the width a stripping policy holds down.
                installed = master.extract_cuts()
                peak_installed_cuts = max(peak_installed_cuts, len(installed))
                if capture_installed:
                    # One snapshot per iteration, in loop order, of the exact
                    # CutRow objects the master holds — the hard-clause gate
                    # compares these across the per-agent vs batched features
                    # paths.
                    installed_snapshots.append(installed)
            last_resolved[mask] = it
            masks.append(mask)
            iterations += 1
            if evaluated.violation <= tolerance:
                # priced_weight is the weight behind this iteration's priced
                # theta (set last iteration), so a violation<=tol certificate
                # is honest only when that theta came from a pure-LP solve. The
                # decay floor and priced_weight==0 gate refuse convergence until
                # the weight has fully decayed; below the floor the walk keeps
                # re-pricing the full sweep (and decaying) until it does. The
                # theta published on break equals this validated one: a
                # converged step ships no cut, so re-solving the unchanged cut
                # set at weight 0 reproduces the same pure-LP vertex.
                if (
                    this_full
                    and iterations >= convergence_floor
                    and priced_weight == 0.0
                ):
                    # last_solve_weight is already 0 here: this iteration's
                    # solve ran at weight_t, which is 0 once it >= decay
                    # (guaranteed by the floor), so the published theta is
                    # a pure LP.
                    converged = True
                    break
                # A partial sweep cannot certify: force a full sweep next.
                force_full = True
            else:
                force_full = False
            # weight_t is the weight behind the theta next iteration prices.
            priced_weight = weight_t
        result = formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()
        if master is not None:
            master.close()
    return WalkOutcome(
        result=result,
        objective=result.objective,
        converged=converged,
        iterations=iterations,
        cuts_admitted=cuts_admitted,
        pricing_calls=pricing_calls,
        schedule_masks=tuple(masks) if schedule is not None else (),
        payload_supports=tuple(supports),
        final_penalty_weight=last_solve_weight,
        peak_installed_cuts=peak_installed_cuts,
        installed_snapshots=tuple(installed_snapshots),
        step_records=(
            tuple(step_sink.records) if step_sink is not None else ()
        ),
        schedule_concentrations=tuple(schedule_concentrations),
    )
