from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest

from combrum.activity import (
    ActivityConfig,
    ActivityRun,
    RowGenIteration,
)
from combrum.context import FitContext
from combrum.engine.driver import LoopConfig, run_fit
from combrum.engine.fitstep import StepResult
from combrum.formulation import FormulationResult
from combrum.oracle import Oracle
from combrum.schedule import RepricingSchedule
from combrum.transport import LocalCluster, SerialTransport


class _Oracle(Oracle):
    pass


class _Formulation:
    def setup(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self.ctx = ctx

    def solve(self) -> np.ndarray:
        return np.zeros(self.ctx.K, dtype=np.float64)

    def result(self) -> FormulationResult:
        return FormulationResult(
            theta_hat=np.zeros(self.ctx.K, dtype=np.float64),
            objective=0.0,
            n_active_cuts=0,
        )

    def dispose(self) -> None:
        pass


def _zero_step() -> StepResult:
    return StepResult(
        violation=0.0,
        progressed=0,
        reduce_rounds=0,
        exchange_rounds=0,
        n_priced=5,
        n_inexact=0,
        n_candidates=0,
        pricing_seconds=0.0,
        master_seconds=0.0,
    )


def _ctx() -> FitContext:
    return FitContext(
        K=1,
        N=5,
        S=1,
        theta_bounds=(np.array([-1.0]), np.array([1.0])),
        theta_coef=np.ones(5, dtype=np.float64),
        agent_weights=np.ones(5, dtype=np.float64),
        local_ids=np.arange(5, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-6,
    )


def test_no_schedule_fit_allocates_no_last_resolved(monkeypatch) -> None:
    driver_mod = importlib.import_module("combrum.engine.driver")

    def fake_resolve_price(*args, **kwargs):  # type: ignore[no-untyped-def]
        return object()

    def fake_fit_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _zero_step()

    def fail_np_full(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("no-schedule run should not allocate last_resolved")

    monkeypatch.setattr(driver_mod, "_resolve_price", fake_resolve_price)
    monkeypatch.setattr(driver_mod, "fit_step", fake_fit_step)
    monkeypatch.setattr(driver_mod.np, "full", fail_np_full)

    outcome = run_fit(
        _ctx(),
        _Oracle(),
        _Formulation(),
        LoopConfig(max_iterations=2),
    )

    assert outcome.diagnostics.converged
    assert outcome.diagnostics.iterations == 1

    # Pin the no-penalty convergence floor to min_iterations exactly. The zero
    # step is convergent every iteration (violation 0, pure LP), so acceptance
    # is held off only by the floor. With min_iterations=2 the floor is 2, so
    # the loop accepts at the iteration whose post-increment count first reaches
    # 2: iterations lands on 2. A +/-1 drift of the floor (e.g. the no-penalty
    # branch reading min_iterations + 1) would shift acceptance to 3, so this
    # equality certifies the floor value the previous run cannot pin.
    floored = run_fit(
        _ctx(),
        _Oracle(),
        _Formulation(),
        LoopConfig(max_iterations=5, min_iterations=2),
    )

    assert floored.diagnostics.converged
    assert floored.diagnostics.iterations == 2


def test_loop_config_rejects_non_integer_and_contradictory_bounds() -> None:
    for kwargs, exc_type, message in (
        ({"max_iterations": True}, TypeError, "max_iterations"),
        ({"max_iterations": np.bool_(True)}, TypeError, "max_iterations"),
        ({"max_iterations": 1.5}, TypeError, "max_iterations"),
        ({"max_iterations": 1, "min_iterations": False}, TypeError, "min_iterations"),
        (
            {"max_iterations": 1, "min_iterations": np.bool_(False)},
            TypeError,
            "min_iterations",
        ),
        ({"max_iterations": 2, "min_iterations": 3}, ValueError, "min_iterations"),
        # Numeric lower bounds: an empty loop (max_iterations=0) and a negative
        # floor are both rejected. These pin the `< 1` / `< 0` guards, which the
        # type/contradiction cases above leave free to slide to `< 0` / dropped.
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"max_iterations": 1, "min_iterations": -1}, ValueError, "min_iterations"),
    ):
        with pytest.raises(exc_type, match=message):
            LoopConfig(**kwargs)

    config = LoopConfig(max_iterations=np.int64(2), min_iterations=np.int64(1))
    assert int(config.max_iterations) == 2
    assert int(config.min_iterations) == 1

    # The min == max boundary is legal: a floor equal to the cap just forces the
    # loop to run to the cap before accepting. Pinning this edge as valid stops
    # the `min > max` guard from sliding to `min >= max`, which the strict
    # min=3/max=2 case above cannot distinguish.
    equal_bounds = LoopConfig(max_iterations=2, min_iterations=2)
    assert int(equal_bounds.min_iterations) == int(equal_bounds.max_iterations) == 2


def test_driver_master_calls_only_on_owner_rank(monkeypatch) -> None:
    driver_mod = importlib.import_module("combrum.engine.driver")

    class _Master:
        def __init__(self) -> None:
            self.dual_calls = 0
            self.penalty_calls = 0
            self.solve_calls = 0
            self.objective_calls = 0
            self.n_active_cuts = 9
            self.weights: list[float] = []
            self.refs: list[tuple[float, ...]] = []

        def dual_values(self):
            self.dual_calls += 1
            return {}

        def set_penalty(self, ref, weight) -> None:
            self.penalty_calls += 1
            self.weights.append(float(weight))
            # Record the anchor theta the owner installs each iteration so the
            # static-vs-dynamic penalty_ref branch is observable, not discarded.
            self.refs.append(tuple(np.asarray(ref, dtype=np.float64).ravel()))

        def solve(self) -> None:
            self.solve_calls += 1

        def objective(self) -> float:
            self.objective_calls += 1
            return 7.5

        def close(self) -> None:
            pass

    class _Schedule(RepricingSchedule):
        def select(self, iteration, n_agents, *, dual, last_resolved):
            return np.ones(n_agents, dtype=bool)

    class _CapturingSink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event) -> None:  # type: ignore[no-untyped-def]
            self.events.append(event)

    def fake_resolve_price(*args, **kwargs):  # type: ignore[no-untyped-def]
        return object()

    # Every step admits CUTS_PER_STEP cuts and reports a nonzero violation, so
    # the loop never converges early and runs the full three iterations. The
    # nonzero, constant progress makes the non-owner `active_cuts` column a
    # running total (CUTS_PER_STEP, 2*CUTS_PER_STEP, 3*CUTS_PER_STEP) that is
    # distinct both from a hardcoded 0 and from the single-step progress, so the
    # `else cuts_admitted` fallback is pinned by identity, not by collapse to 0.
    CUTS_PER_STEP = 4

    def fake_fit_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["owner_rank"] == 1
        if kwargs["before_apply"] is not None:
            kwargs["before_apply"]()
        step = _zero_step()
        return replace(step, progressed=CUTS_PER_STEP, violation=1.0)

    monkeypatch.setattr(driver_mod, "_resolve_price", fake_resolve_price)
    monkeypatch.setattr(driver_mod, "fit_step", fake_fit_step)

    # A formulation whose LP solve returns a fixed nonzero theta, distinct from
    # the static anchor below. Keeping query != static_ref is what makes the
    # penalty_ref branch observable: with a zeros-solving formulation and no
    # theta_init both anchors collapse to the origin and the branch is dead.
    THETA_SOLVE = 0.5
    THETA_INIT = 0.25

    class _NonzeroFormulation(_Formulation):
        def solve(self) -> np.ndarray:
            return np.full(self.ctx.K, THETA_SOLVE, dtype=np.float64)

    def per_rank(transport, *, penalty_ref):
        # Every rank gets a real spy master so the non-owner half of the
        # "master calls only on owner rank" claim is actually observed:
        # rank 0 (non-owner) must record zero dual/penalty/objective calls.
        # A driver bug that dropped the `rank != owner_rank` guard in
        # _dual_payload/_penalty_solve would tick rank 0's counters here.
        #
        # Activity is set to "iterations" with a capturing sink so the
        # per-iteration objective / active_cuts read path actually runs. That
        # path also owner-guards the master reads, so a bug that dropped the
        # rank guard there would either tick rank 0's objective_calls or leak
        # the owner's objective/active_cuts onto the non-owner's emitted row.
        master = _Master()
        sink = _CapturingSink()
        activity = ActivityRun(
            config=ActivityConfig(level="iterations"), sink=sink
        )
        ctx = FitContext(
            K=1,
            N=5,
            S=1,
            theta_bounds=(np.array([-1.0]), np.array([1.0])),
            theta_coef=np.ones(5, dtype=np.float64),
            agent_weights=np.ones(5, dtype=np.float64),
            local_ids=np.arange(transport.rank, 5, transport.size, dtype=np.int64),
            transport=transport,
            tolerance=1e-6,
            master_backend=master,
            owner_rank=1,
            theta_init=np.array([THETA_INIT], dtype=np.float64),
        )
        outcome = run_fit(
            ctx,
            _Oracle(),
            _NonzeroFormulation(),
            LoopConfig(
                max_iterations=3,
                schedule=_Schedule(),
                qp_weight=1.0,
                decay=1,
                penalty_ref=penalty_ref,
                activity=activity,
            ),
        )
        rows = [e for e in sink.events if isinstance(e, RowGenIteration)]
        return (
            master.dual_calls,
            master.penalty_calls,
            master.solve_calls,
            master.objective_calls,
            tuple(row.objective for row in rows),
            tuple(row.active_cuts for row in rows),
            tuple(master.weights),
            outcome.diagnostics.final_penalty_weight,
            tuple(master.refs),
        )

    # Independent oracle for the decayed penalty schedule. With qp_weight=1.0 and
    # decay=1 the linear-to-zero rule weight(it) = qp_weight * max(0, 1 - it/decay)
    # gives 1.0 at it=0 and 0.0 from it=1 on, so over three iterations the owner
    # installs exactly (1.0, 0.0): the positive QP solve, then the one necessary
    # revert solve to restore a pure LP. Later zero-weight iterations must not
    # keep solving an already-pure master. Recompute both from the config values
    # here rather than reading any driver local, so a non-decaying
    # `weight_t = qp_weight` constant (owner sees more positive weights), an
    # off-by-one `it + 1` shift (owner sees no positive solve), and a dropped
    # post-decay guard (owner sees an extra 0.0) all diverge from this.
    qp_weight, decay = 1.0, 1
    expected_weights = tuple(
        qp_weight * max(0.0, 1.0 - it / decay)
        for it in range(2)
    )
    final_weight = expected_weights[-1]

    # Independent oracle for the installed penalty anchor (ref) schedule. The
    # driver picks ref_t = query (the LP-solved theta) when penalty_ref ==
    # "dynamic", else static_ref = theta_init. With the fixtures above those two
    # anchors are deliberately distinct: solve() returns THETA_SOLVE and
    # theta_init is THETA_INIT. So on the owner the static run installs
    # (THETA_INIT,)*2 and the dynamic run installs (THETA_SOLVE,)*2; the
    # non-owner installs nothing. Hand-build both schedules from the fixture
    # scalars, never from a driver local. A branch collapse -- ref_t = query
    # always (static run drifts onto THETA_SOLVE) or ref_t = static_ref always
    # (dynamic run drifts onto THETA_INIT) -- makes exactly one of the two runs
    # diverge, so pinning both anchors kills the whole penalty_ref branch.
    static_ref_owner = tuple((THETA_INIT,) for _ in range(2))
    dynamic_ref_owner = tuple((THETA_SOLVE,) for _ in range(2))

    # Three iterations run (violation stays 1.0, so the loop hits max_iterations).
    # Rank 0 (non-owner) records no master calls; every emitted row carries
    # objective=None (the owner-guarded read yields None off-owner) and an
    # active_cuts that is the running cuts_admitted total: 4, 8, 12 for
    # CUTS_PER_STEP=4. That sequence is the independent oracle -- distinct from a
    # hardcoded (0, 0, 0) and from an `else step.progressed` fallback (4, 4, 4),
    # so both siblings of the fallback regression die. It installs no penalty
    # (empty weights, empty refs) and reports final_penalty_weight 0.0. Rank 1
    # (owner) makes exactly one dual_values call: the first sweep is forced
    # full, the second prices a QP-derived theta and is forced full again, and
    # only the third may consult LP duals. It makes two set_penalty/solve calls
    # (QP, then LP revert), and three objective reads (=7.5 each, the master
    # fixture's value); every owner row carries active_cuts=9 (the master
    # fixture's n_active_cuts, read instead of cuts_admitted); it installs the
    # decayed weights above at the branch-picked anchor and publishes at
    # final_weight.
    # Dropping either owner guard would flip rank 0's objective off None or its
    # active_cuts onto 9; dropping the owner's n_active_cuts read would slide
    # rank 1 onto the (4, 8, 12) totals.
    assert LocalCluster(2).run(lambda t: per_rank(t, penalty_ref="static")) == [
        (0, 0, 0, 0, (None, None, None), (4, 8, 12), (), 0.0, ()),
        (
            1,
            2,
            2,
            3,
            (7.5, 7.5, 7.5),
            (9, 9, 9),
            expected_weights,
            final_weight,
            static_ref_owner,
        ),
    ]
    assert LocalCluster(2).run(lambda t: per_rank(t, penalty_ref="dynamic")) == [
        (0, 0, 0, 0, (None, None, None), (4, 8, 12), (), 0.0, ()),
        (
            1,
            2,
            2,
            3,
            (7.5, 7.5, 7.5),
            (9, 9, 9),
            expected_weights,
            final_weight,
            dynamic_ref_owner,
        ),
    ]


def test_driver_stages_penalty_on_formulation_when_supported(monkeypatch) -> None:
    driver_mod = importlib.import_module("combrum.engine.driver")

    def fake_resolve_price(*args, **kwargs):  # type: ignore[no-untyped-def]
        return object()

    before_apply_values: list[object] = []

    def fake_fit_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        before_apply_values.append(kwargs["before_apply"])
        return replace(_zero_step(), violation=1.0)

    monkeypatch.setattr(driver_mod, "_resolve_price", fake_resolve_price)
    monkeypatch.setattr(driver_mod, "fit_step", fake_fit_step)

    class _StagingFormulation(_Formulation):
        def __init__(self) -> None:
            self.prepared: list[tuple[tuple[float, ...], float]] = []

        def solve(self) -> np.ndarray:
            return np.array([0.25], dtype=np.float64)

        def prepare_penalty_solve(self, ref, weight) -> None:  # type: ignore[no-untyped-def]
            self.prepared.append(
                (tuple(np.asarray(ref, dtype=np.float64).ravel()), float(weight))
            )

    formulation = _StagingFormulation()
    outcome = run_fit(
        _ctx(),
        _Oracle(),
        formulation,
        LoopConfig(
            max_iterations=2,
            qp_weight=1.0,
            decay=1,
            penalty_ref="dynamic",
        ),
    )

    assert before_apply_values == [None, None]
    assert formulation.prepared == [((0.25,), 1.0), ((0.25,), 0.0)]
    assert outcome.diagnostics.final_penalty_weight == 0.0
