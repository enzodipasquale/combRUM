from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from combrum.context import FitContext
from combrum.demand import Demand
from combrum.formulation import Evaluation, Formulation, FormulationResult
from combrum.transport import SerialTransport

K, N, S = 2, 3, 1


def make_ctx() -> FitContext:
    # Master-free on purpose: master_backend/cut_policy/schedule stay None.
    return FitContext(
        K=K,
        N=N,
        S=S,
        theta_bounds=(np.full(K, -10.0), np.full(K, 10.0)),
        theta_coef=np.zeros(N * S),
        agent_weights=np.full(N * S, 1.0 / (N * S)),
        local_ids=np.arange(N * S, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-6,
    )


# --- Evaluation --------------------------------------------------------------


def test_evaluation_accepts_nonnegative_violation() -> None:
    assert Evaluation(violation=0.0).violation == 0.0  # converged is valid
    ev = Evaluation(violation=np.float64(2.5))
    assert type(ev.violation) is float and ev.violation == 2.5


def test_evaluation_payload_is_optional_and_opaque() -> None:
    # The caller reads violation only; payload is method-owned state that
    # rides the same object into update().
    assert Evaluation(violation=1.0).payload is None
    marker = {"cuts": [1, 2, 3]}
    assert Evaluation(violation=1.0, payload=marker).payload is marker


def test_evaluation_rejects_negative_and_nan() -> None:
    for bad in (-1e-12, float("nan")):
        with pytest.raises(ValueError, match="violation must be >= 0"):
            Evaluation(violation=bad)


# --- FormulationResult --------------------------------------------------------


def test_result_theta_hat_and_objective_required_by_position() -> None:
    with pytest.raises(TypeError):
        FormulationResult()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        FormulationResult(np.zeros(K))  # type: ignore[call-arg]


def test_result_zero_active_cuts_valid() -> None:
    # objective is coerced to a plain float, mirroring Evaluation.violation:
    # np.float64 passes isinstance(_, float) but not `type(_) is float`.
    res = FormulationResult(
        theta_hat=np.zeros(K), objective=np.float64(1.0), n_active_cuts=0
    )
    assert type(res.objective) is float and res.objective == 1.0
    assert res.n_active_cuts == 0  # cutless methods exist
    assert res.slack is None and res.active_set is None and res.dual is None
    assert res.metadata == {}

    # objective has no sign constraint; a minimized loss can be negative.
    neg = FormulationResult(
        theta_hat=np.zeros(K), objective=-3.5, n_active_cuts=0
    )
    assert neg.objective == -3.5


def test_result_carries_opaque_active_set() -> None:
    # The installed-cut collection a row-generation method publishes for
    # retirement/persistence/warm-start; its concrete type is method-owned.
    installed = ("cut-a", "cut-b")
    dual_marker = object()
    res = FormulationResult(
        theta_hat=np.full(K, 2.0),
        objective=-3.5,
        n_active_cuts=2,
        active_set=installed,
        dual=dual_marker,
    )
    np.testing.assert_array_equal(res.theta_hat, np.full(K, 2.0))
    assert res.objective == -3.5
    assert res.n_active_cuts == 2
    assert res.active_set is installed
    assert res.dual is dual_marker
    assert res.slack is None and res.metadata == {}


def test_result_default_metadata_is_per_instance() -> None:
    # Default metadata must be a fresh dict per result, not a shared mutable
    # default: one method's diagnostics must never leak into another's.
    r1 = FormulationResult(theta_hat=np.zeros(K), objective=0.0, n_active_cuts=0)
    r2 = FormulationResult(theta_hat=np.zeros(K), objective=0.0, n_active_cuts=0)
    assert r1.metadata == {} and r2.metadata == {}
    assert r1.metadata is not r2.metadata
    r1.metadata["cuts_retired"] = 4
    assert "cuts_retired" not in r2.metadata
    assert r2.metadata == {}


def test_result_rejects_negative_active_cuts() -> None:
    with pytest.raises(ValueError, match="n_active_cuts must be >= 0"):
        FormulationResult(
            theta_hat=np.zeros(K), objective=0.0, n_active_cuts=-1
        )


def test_result_rejects_non_vector_theta_hat() -> None:
    # Neither a 2-D matrix nor a 0-D scalar is a (K,) vector.
    for bad in (np.zeros((K, K)), np.float64(3.0)):
        with pytest.raises(
            ValueError, match=r"expected one-dimensional \(K,\) theta_hat"
        ):
            FormulationResult(
                theta_hat=bad, objective=0.0, n_active_cuts=0
            )


def test_result_rejects_non_finite_theta_hat() -> None:
    for bad in (
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, -np.inf]),
    ):
        with pytest.raises(ValueError, match="theta_hat must be finite"):
            FormulationResult(theta_hat=bad, objective=0.0, n_active_cuts=0)


def test_result_arrays_read_only() -> None:
    res = FormulationResult(
        theta_hat=np.arange(K, dtype=np.float64),
        objective=0.0,
        n_active_cuts=2,
        slack=np.ones(N * S),
    )
    for arr in (res.theta_hat, res.slack):
        assert arr is not None
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 9.0

    # Freezing must not alter the values.
    np.testing.assert_array_equal(res.theta_hat, np.arange(K, dtype=np.float64))
    np.testing.assert_array_equal(res.slack, np.ones(N * S))
    assert res.n_active_cuts == 2


# --- master-free formulation double ---------------------------------------


def test_formulation_abc_not_instantiable() -> None:
    with pytest.raises(TypeError):
        Formulation()  # type: ignore[abstract]


class BestThetaFormulation(Formulation):
    """Master-free double -- no master, no cuts.

    Walks a fixed query path and publishes its best-evaluated theta, not
    the last solve() query. Assumes payoffs are <= 0, so the negated mean
    payoff is a nonnegative violation.
    """

    _PATH = (0.0, 1.0, 2.0, 3.0)  # query scales; overshoots past the optimum

    def __init__(self) -> None:
        self._K = 0
        self._step = 0
        self._query: np.ndarray | None = None
        self._best_violation = float("inf")
        self._best_theta: np.ndarray | None = None
        self._best_objective = float("-inf")

    def setup(self, ctx: FitContext) -> None:
        self._K = ctx.K

    def solve(self) -> np.ndarray:
        scale = self._PATH[min(self._step, len(self._PATH) - 1)]
        self._query = np.full(self._K, scale, dtype=np.float64)
        return self._query

    def evaluate(self, demands: Mapping[int, Demand]) -> Evaluation:
        # Evaluation only measures; state advances in update() via the payload.
        assert self._query is not None
        mean_payoff = float(
            np.mean([demand.payoff for demand in demands.values()])
        )
        return Evaluation(
            violation=-mean_payoff,
            payload=(self._query.copy(), mean_payoff),
        )

    def update(self, step: Evaluation) -> int:
        query, mean_payoff = step.payload  # type: ignore[misc]
        self._step += 1
        if step.violation < self._best_violation:
            self._best_violation = step.violation
            self._best_theta = query
            self._best_objective = mean_payoff
            return 1
        return 0

    def result(self) -> FormulationResult:
        assert self._best_theta is not None
        return FormulationResult(
            theta_hat=self._best_theta,
            objective=self._best_objective,
            n_active_cuts=0,
        )


TARGET_SCALE = 1.0  # payoffs peak where the query equals TARGET_SCALE * ones


def price_all(theta: np.ndarray, ctx: FitContext) -> dict[int, Demand]:
    # Stand-in pricing: one exact Demand per id, payoff peaking at zero
    # when theta hits the target.
    target = np.full(ctx.K, TARGET_SCALE)
    payoff = -float(np.sum((theta - target) ** 2))
    return {
        int(a): Demand.exact(np.array([1.0]), payoff) for a in ctx.local_ids
    }


def test_master_free_walk_publishes_best_not_last() -> None:
    ctx = make_ctx()
    formulation = BestThetaFormulation()
    formulation.setup(ctx)

    queries: list[np.ndarray] = []
    violations: list[float] = []
    progress: list[int] = []
    for _ in range(4):  # enough steps for the path to overshoot
        theta = formulation.solve()
        queries.append(np.array(theta, copy=True))
        ev = formulation.evaluate(price_all(theta, ctx))
        assert isinstance(ev, Evaluation)
        violations.append(ev.violation)
        progress.append(formulation.update(ev))

    res = formulation.result()
    formulation.dispose()  # default no-op must be callable without override

    # The walk overshot: the last query is strictly worse than the best.
    best = int(np.argmin(violations))
    assert best == 1
    assert violations[-1] > violations[best]

    np.testing.assert_array_equal(res.theta_hat, queries[best])
    assert not np.array_equal(res.theta_hat, queries[-1])
    assert res.objective == 0.0  # the best query hit the payoff peak

    # No row-generation assumption leaked into the published result.
    assert res.n_active_cuts == 0
    assert res.dual is None and res.slack is None and res.active_set is None

    # update() returned both 0 and 1 over the walk; the caller ignored it.
    assert 0 in progress and 1 in progress
