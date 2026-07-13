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

    # The zero step is convergent every iteration (violation 0, pure LP), so
    # only min_iterations holds acceptance off: with a floor of 2 the loop
    # must accept exactly when the post-increment count reaches 2.
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
        # An empty loop and a negative floor are rejected on value, not type.
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"max_iterations": 1, "min_iterations": -1}, ValueError, "min_iterations"),
    ):
        with pytest.raises(exc_type, match=message):
            LoopConfig(**kwargs)

    config = LoopConfig(max_iterations=np.int64(2), min_iterations=np.int64(1))
    assert int(config.max_iterations) == 2
    assert int(config.min_iterations) == 1

    # min == max is legal: a floor equal to the cap just forces the loop to
    # run to the cap before accepting.
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

    # Every step admits CUTS_PER_STEP cuts at nonzero violation, so the loop
    # never converges early and the non-owner active_cuts column becomes a
    # running total (4, 8, 12) — distinct from both 0 and single-step progress.
    CUTS_PER_STEP = 4

    def fake_fit_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["owner_rank"] == 1
        if kwargs["before_apply"] is not None:
            kwargs["before_apply"]()
        step = _zero_step()
        return replace(step, progressed=CUTS_PER_STEP, violation=1.0)

    monkeypatch.setattr(driver_mod, "_resolve_price", fake_resolve_price)
    monkeypatch.setattr(driver_mod, "fit_step", fake_fit_step)

    # solve() returns THETA_SOLVE while theta_init is THETA_INIT; keeping the
    # two anchors distinct is what makes the static-vs-dynamic penalty_ref
    # branch observable (with both at the origin the branch would be dead).
    THETA_SOLVE = 0.5
    THETA_INIT = 0.25

    class _NonzeroFormulation(_Formulation):
        def solve(self) -> np.ndarray:
            return np.full(self.ctx.K, THETA_SOLVE, dtype=np.float64)

    def per_rank(transport, *, penalty_ref):
        # Both ranks get a spy master, so rank 0 (non-owner) recording zero
        # dual/penalty/objective calls is observed, not assumed. The
        # "iterations" activity sink exercises the per-iteration
        # objective/active_cuts read path, which is owner-guarded too.
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
                qp_iterations=1,
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

    # weight(it) = qp_weight while it < qp_iterations, else exactly 0,
    # recomputed here from the config values rather than read from any driver
    # local. With qp_weight=1.0 and qp_iterations=1 the owner installs exactly
    # (1.0, 0.0): the QP solve, then one revert solve back to a pure LP —
    # later zero-weight iterations must not keep re-solving an already-pure
    # master.
    qp_weight, qp_iterations = 1.0, 1
    expected_weights = tuple(
        qp_weight if it < qp_iterations else 0.0 for it in range(2)
    )
    final_weight = expected_weights[-1]

    # The driver anchors the penalty at the LP-solved theta when penalty_ref
    # is "dynamic" and at theta_init when "static". Hand-build both expected
    # anchor schedules from the fixture scalars: the static run must install
    # (THETA_INIT,)*2, the dynamic run (THETA_SOLVE,)*2, the non-owner nothing.
    static_ref_owner = tuple((THETA_INIT,) for _ in range(2))
    dynamic_ref_owner = tuple((THETA_SOLVE,) for _ in range(2))

    # Violation stays 1.0, so all three iterations run. Rank 0 (non-owner):
    # no master calls, objective None on every row, active_cuts is the running
    # cuts_admitted total (4, 8, 12), no penalty installed. Rank 1 (owner):
    # exactly one dual_values call — the first sweep is forced full and the
    # second reprices a QP-derived theta, so only the third consults LP duals —
    # two set_penalty/solve calls (QP, then LP revert), three objective reads
    # of the fixture's 7.5, and active_cuts=9 from the master rather than the
    # cuts_admitted total.
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
            qp_iterations=1,
            penalty_ref="dynamic",
        ),
    )

    assert before_apply_values == [None, None]
    assert formulation.prepared == [((0.25,), 1.0), ((0.25,), 0.0)]
    assert outcome.diagnostics.final_penalty_weight == 0.0
