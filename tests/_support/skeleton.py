"""End-to-end vehicle exercising every frozen contract once on family fixtures.

A :class:`SkeletonProblem` adapter owns everything family-specific; the
walk is family-agnostic. Two adapters ship:

* :class:`ToyProblem` — agent ``i`` values bundle ``d`` at
  ``sum_k d_k * (r[i, k] * theta_k + nu[i, k])``, so exact demand takes
  item ``k`` iff its score ``r[i, k] * theta_k + nu[i, k]`` is positive.
* :class:`QkpProblem` — agent ``i`` maximises
  ``alpha * x_i·b - delta·b + 0.5 * lambda * b'Qb + nu_i·b`` subject to
  ``weights·b <= capacity_i``, with ``theta = [alpha, delta, lambda]``.

Each is the same rule its fixture generator used, making the data
rationalisable (regret exactly 0 at ``theta_true``).

The walk is subgradient descent on total regret
``sum_i [payoff(chosen_i) - payoff(observed_i)] >= 0``. Each violated
agent ships a cut whose moment ``phi(observed_i) - phi(chosen_i)`` is the
negative subgradient of its regret, so rank 0 steps
``theta += eta * sum(moments)`` toward the rationalisability cone, where
every chosen bundle equals its observed bundle bitwise and max regret is
exactly ``0.0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import pytest

from combrum.context import FitContext
from combrum.demand import Demand
from combrum.formulation import Evaluation, Formulation, FormulationResult
from combrum.oracle import Oracle
from combrum.transport.base import CutRow, Transport, TransportError
from combrum.transport.reference import SerialTransport

#: Single replication owned by rank 0; routes every cut through the exchange.
_OWNERS: np.ndarray = np.zeros(1, dtype=np.int64)
_OWNERS.setflags(write=False)

#: ``(agent_id, observed_row, chosen_row, regret)`` for one violated agent.
_Violated = tuple[int, np.ndarray, np.ndarray, float]

#: (step_size, max_iterations) per family.
_FAMILY_DEFAULTS: dict[str, tuple[float, int]] = {
    "toy": (0.1, 60),
    "qkp": (0.05, 400),
}


class SkeletonProblem(ABC):
    """Family-specific pricing, payoff, and cut moment for the walk."""

    @property
    @abstractmethod
    def K(self) -> int:
        """The parameter-vector length for this family."""

    @abstractmethod
    def setup(self, local_ids: np.ndarray) -> None:
        """Precompute per-agent state for the globally-indexed ids owned here."""

    @abstractmethod
    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        """Exact demand at ``theta`` for one agent (the argmax bundle)."""

    @abstractmethod
    def payoff(self, bundle: np.ndarray, agent_id: int, theta: np.ndarray) -> float:
        """The shared payoff expression; chosen == observed gives regret 0.0."""

    @abstractmethod
    def observed_bundle(self, agent_id: int) -> np.ndarray:
        """The fixture's observed bundle for one agent."""

    @abstractmethod
    def cut_moment(
        self, observed: np.ndarray, chosen: np.ndarray, agent_id: int
    ) -> np.ndarray:
        """``phi(observed) - phi(chosen)``: the negative subgradient of regret."""


def _bundle_payoff(scores: np.ndarray, bundle: np.ndarray) -> float:
    return float(np.where(bundle, scores, 0.0).sum())


class ToyProblem(SkeletonProblem):
    """Item-separable toy: take every item whose score is positive."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._observables = np.asarray(arrays["observables"], dtype=np.float64)
        self._shocks = np.asarray(arrays["shocks"], dtype=np.float64)
        self._observed = np.asarray(arrays["observed"], dtype=bool)
        self._rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def K(self) -> int:
        return self._observables.shape[1]

    def setup(self, local_ids: np.ndarray) -> None:
        # S == 1, so the global id indexes the arrays directly.
        self._rows = {
            int(a): (self._observables[int(a)], self._shocks[int(a), 0, :])
            for a in local_ids
        }

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        r, nu = self._rows[agent_id]
        scores = r * theta + nu
        # Strict ">" matches the fixture generator: pricing at theta_true must
        # reproduce every observed bundle bitwise.
        bundle = scores > 0.0
        return Demand.exact(bundle=bundle, payoff=_bundle_payoff(scores, bundle))

    def payoff(self, bundle: np.ndarray, agent_id: int, theta: np.ndarray) -> float:
        r, nu = self._rows[agent_id]
        return _bundle_payoff(r * theta + nu, bundle)

    def observed_bundle(self, agent_id: int) -> np.ndarray:
        return self._observed[agent_id]

    def cut_moment(
        self, observed: np.ndarray, chosen: np.ndarray, agent_id: int
    ) -> np.ndarray:
        r, _ = self._rows[agent_id]
        # phi(d) = d * r, so phi(observed) - phi(chosen).
        return (observed.astype(np.float64) - chosen.astype(np.float64)) * r


def _enumerate_bundles(n_items: int) -> np.ndarray:
    """All ``2**n_items`` bundles as ``(2**n_items, n_items)`` float 0/1."""
    index = np.arange(1 << n_items, dtype=np.int64)
    bits = (index[:, None] >> np.arange(n_items)[None, :]) & 1
    return bits.astype(np.float64)


class QkpProblem(SkeletonProblem):
    """Capacity-constrained quadratic knapsack; theta = [alpha, delta, lambda].

    Pricing is exact enumeration over the ``2**M`` bundles (M small). The
    feature map ``phi(b, i) = [x_i·b, -b, 0.5 b'Qb]`` makes
    ``payoff = theta·phi + nu_i·b`` match the generator's utility.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._x = np.asarray(arrays["x"], dtype=np.float64)
        self._Q = np.asarray(arrays["Q"], dtype=np.float64)
        self._weights = np.asarray(arrays["weights"], dtype=np.float64)
        self._cap = np.asarray(arrays["capacities"], dtype=np.float64)
        self._nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
        self._observed = np.asarray(arrays["observed"], dtype=bool)
        self._M = self._x.shape[1]
        # phi's quadratic term is agent-independent, so enumerate once.
        self._bundles = _enumerate_bundles(self._M)
        self._loads = self._bundles @ self._weights
        self._half_bQb = 0.5 * np.einsum(
            "bj,jk,bk->b", self._bundles, self._Q, self._bundles
        )
        self._rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def K(self) -> int:
        return self._M + 2

    def setup(self, local_ids: np.ndarray) -> None:
        for a in local_ids:
            a = int(a)
            feasible = self._loads <= self._cap[a]
            self._rows[a] = (self._x[a], self._nu[a], feasible)

    def _phi(self, bundle: np.ndarray, agent_id: int) -> np.ndarray:
        x_a, _, _ = self._rows[agent_id]
        b = bundle.astype(np.float64)
        return np.concatenate(
            ([float(x_a @ b)], -b, [0.5 * float(b @ self._Q @ b)])
        )

    def payoff(self, bundle: np.ndarray, agent_id: int, theta: np.ndarray) -> float:
        _, nu_a, _ = self._rows[agent_id]
        b = bundle.astype(np.float64)
        return float(self._phi(bundle, agent_id) @ theta + nu_a @ b)

    def observed_bundle(self, agent_id: int) -> np.ndarray:
        return self._observed[agent_id]

    def cut_moment(
        self, observed: np.ndarray, chosen: np.ndarray, agent_id: int
    ) -> np.ndarray:
        return self._phi(observed, agent_id) - self._phi(chosen, agent_id)

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        x_a, nu_a, feasible = self._rows[agent_id]
        bundles = self._bundles
        phi = np.concatenate(
            [
                (bundles @ x_a)[:, None],
                -bundles,
                self._half_bQb[:, None],
            ],
            axis=1,
        )
        payoffs = np.where(feasible, phi @ theta + bundles @ nu_a, -np.inf)
        best = bundles[int(np.argmax(payoffs))] > 0.5
        # Recompute payoff via the shared expression so it matches
        # payoff(observed) bit-for-bit when the bundles agree.
        return Demand.exact(bundle=best, payoff=self.payoff(best, agent_id, theta))


def _build_problem(family: str, arrays: Mapping[str, np.ndarray]) -> SkeletonProblem:
    if family == "toy":
        return ToyProblem(arrays)
    if family == "qkp":
        return QkpProblem(arrays)
    raise ValueError(f"unknown skeleton family {family!r}; expected toy or qkp")


class SkeletonOracle(Oracle):
    """Delegates exact pricing to a SkeletonProblem."""

    def __init__(self, problem: SkeletonProblem) -> None:
        self._problem = problem

    def setup(self, transport: Transport, local_ids: np.ndarray) -> None:
        self._problem.setup(local_ids)

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        return self._problem.price(theta, agent_id)


class SkeletonFormulation(Formulation):
    """Family-agnostic cut-collecting subgradient walk."""

    def __init__(self, problem: SkeletonProblem, step_size: float) -> None:
        if not step_size > 0.0:
            raise ValueError(f"step_size must be > 0; got {step_size}")
        self._problem = problem
        self._eta = float(step_size)

    def setup(self, ctx: FitContext) -> None:
        self._transport = ctx.transport
        self._lower, self._upper = ctx.theta_bounds
        self._ids = np.asarray(ctx.local_ids, dtype=np.int64)
        init = ctx.theta_init if ctx.theta_init is not None else np.zeros(ctx.K)
        self._theta = np.array(init, dtype=np.float64)
        self._best_violation = float("inf")
        self._best_theta: np.ndarray | None = None
        self._best_total = float("inf")
        self._cuts_received = 0
        self._n_iterations = 0

    def solve(self) -> np.ndarray:
        return self._theta.copy()

    def evaluate(self, demands: Mapping[int, Demand]) -> Evaluation:
        regrets = np.zeros(self._ids.shape[0], dtype=np.float64)
        violated: list[_Violated] = []
        for j, a in enumerate(self._ids):
            a = int(a)
            demand = demands[a]
            observed = self._problem.observed_bundle(a)
            raw = demand.payoff - self._problem.payoff(observed, a, self._theta)
            # Regret is >= 0 (oracle maximises); floor the float-cancellation
            # residue so the stop rule never sees a tiny negative.
            regret = raw if raw > 0.0 else 0.0
            regrets[j] = regret
            if regret > 0.0:
                chosen = np.asarray(demand.bundle, dtype=bool)
                violated.append((a, observed, chosen, regret))
        # Reproducible total keyed by global agent ids; stop rule via
        # allreduce_max. An empty shard contributes 0.0.
        total = float(self._transport.sum_reproducible(regrets, self._ids))
        local_max = float(regrets.max()) if regrets.size else 0.0
        violation = self._transport.allreduce_max(local_max)
        return Evaluation(violation=violation, payload=(total, tuple(violated)))

    def update(self, step: Evaluation) -> int:
        total, violated = step.payload  # type: ignore[misc]
        # Record best-so-far before stepping: this violation belongs to the
        # current theta, not the post-step one.
        if step.violation < self._best_violation:
            self._best_violation = step.violation
            self._best_theta = self._theta.copy()
            self._best_total = float(total)
        rows = [
            CutRow(
                rep_id=0,
                agent_id=agent_id,
                phi=self._problem.cut_moment(observed, chosen, agent_id),
                epsilon=regret,
                bundle_key=np.packbits(chosen).tobytes(),
            )
            for agent_id, observed, chosen, regret in violated
        ]
        received = self._transport.exchange_cuts(rows, _OWNERS)
        packet: tuple[np.ndarray, int] | None = None
        with self._transport.collective():
            if self._transport.rank == 0:
                self._cuts_received += len(received)
                theta = self._theta
                if received:
                    # Sum in delivered (rep, agent, key) order; agent ids are
                    # unique, so this is bitwise invariant to sharding.
                    agg = np.add.reduce(
                        np.stack([row.phi for row in received]), axis=0
                    )
                    theta = np.clip(
                        self._theta + self._eta * agg, self._lower, self._upper
                    )
                packet = (theta, self._cuts_received)
        theta, cuts_received = self._transport.bcast(packet, root=0)
        self._theta = np.asarray(theta, dtype=np.float64)
        # All ranks adopt the rank-0 tally so result() agrees everywhere.
        self._cuts_received = int(cuts_received)
        self._n_iterations += 1
        return len(rows)

    def result(self) -> FormulationResult:
        if self._best_theta is None:
            raise RuntimeError("result() requires at least one evaluated step")
        return FormulationResult(
            theta_hat=self._best_theta,
            objective=-self._best_violation,
            n_active_cuts=self._cuts_received,
            metadata={
                "n_iterations": self._n_iterations,
                "best_total_regret": self._best_total,
            },
        )


def run_skeleton(
    arrays: Mapping[str, np.ndarray],
    transport: Transport,
    *,
    family: str = "toy",
    tolerance: float = 1e-9,
    max_iterations: int | None = None,
    step: float | None = None,
) -> FormulationResult:
    """Solve a family fixture end-to-end through every frozen contract.

    Every rank calls this with the same arrays; the shard is a rank/size
    round-robin (``a % size == rank``) so shards interleave. ``step`` and
    ``max_iterations`` fall back to the family defaults when unset.
    """
    default_step, default_iters = _FAMILY_DEFAULTS[family]
    step = default_step if step is None else step
    max_iterations = default_iters if max_iterations is None else max_iterations

    problem = _build_problem(family, arrays)
    n_obs = np.asarray(arrays["observed"]).shape[0]
    k = problem.K
    local_ids = np.arange(transport.rank, n_obs, transport.size, dtype=np.int64)
    ctx = FitContext(
        K=k,
        N=n_obs,
        S=1,
        theta_bounds=(np.full(k, -10.0), np.full(k, 10.0)),
        theta_coef=np.ones(n_obs),
        agent_weights=np.full(n_obs, 1.0 / n_obs),
        local_ids=local_ids,
        transport=transport,
        tolerance=tolerance,
    )
    oracle = SkeletonOracle(problem)
    formulation = SkeletonFormulation(problem, step_size=step)
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(max_iterations):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            evaluated = formulation.evaluate(demands)
            formulation.update(evaluated)
            if evaluated.violation <= tolerance:
                break
        return formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()


# --------------------------------------------------------------------------
# Skeleton-formulation contract tests.
#
# These cases exercise contracts that the end-to-end skeleton walk reaches but
# does not stress: active bounds, total-regret accounting, bundle-key identity,
# and collective failure agreement.
#
# Support-module tests are not collected by the directory suite; run them by
# naming this file, e.g.:
#   pytest -o python_files='*.py' tests/_support/skeleton.py
# --------------------------------------------------------------------------


def _single_step(
    arrays: Mapping[str, np.ndarray],
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    theta_init: np.ndarray,
    step: float,
    transport: Transport,
    family: str = "toy",
) -> SkeletonFormulation:
    """Set up a formulation and run exactly one evaluate/update on rank 0.

    Returns the formulation after the single step so callers can read
    ``_theta``, ``_best_total``, and the shipped cuts.
    """
    problem = _build_problem(family, arrays)
    k = problem.K
    n_obs = np.asarray(arrays["observed"]).shape[0]
    ids = np.arange(n_obs, dtype=np.int64)
    ctx = FitContext(
        K=k,
        N=n_obs,
        S=1,
        theta_bounds=bounds,
        theta_coef=np.ones(n_obs),
        agent_weights=np.full(n_obs, 1.0 / n_obs),
        local_ids=ids,
        transport=transport,
        tolerance=1e-12,
        theta_init=np.asarray(theta_init, dtype=np.float64),
    )
    oracle = SkeletonOracle(problem)
    formulation = SkeletonFormulation(problem, step_size=step)
    oracle.setup(transport, ids)
    formulation.setup(ctx)
    theta = formulation.solve()
    demands = {int(a): oracle.price(theta, int(a)) for a in ids}
    formulation.update(formulation.evaluate(demands))
    return formulation


def test_update_enforces_theta_bounds_clip() -> None:
    # run_skeleton converges to an interior optimum where the np.clip in update
    # never binds, so drive one step whose unconstrained iterate lands outside
    # a deliberately tight box.
    #
    # 1 agent, 2 items, r = [+1, -1], nu = [-0.5, -0.5], observed = both taken.
    # At theta_init = 0 the priced demand takes neither item (both scores
    # -0.5 < 0), so both are violated. The cut moment is
    # (observed - chosen) * r = [+1, -1]; one step of size 0.1 gives the
    # unconstrained iterate [0.1, -0.1].
    arrays = {
        "observables": np.array([[1.0, -1.0]]),
        "shocks": np.array([[[-0.5, -0.5]]]),
        "observed": np.array([[True, True]]),
        "theta_true": np.array([2.0, -2.0]),
    }
    theta_init = np.zeros(2)
    step = 0.1
    agg = np.array([1.0, -1.0])
    unclipped = theta_init + step * agg  # [0.1, -0.1]
    lower = np.array([-10.0, -0.05])
    upper = np.array([0.05, 10.0])
    expected = np.array([0.05, -0.05])  # unclipped clipped into the tight box

    form = _single_step(
        arrays,
        bounds=(lower, upper),
        theta_init=theta_init,
        step=step,
        transport=SerialTransport(),
    )
    # Both components must sit exactly on their bounds, not at the
    # unconstrained iterate.
    assert form._theta.tobytes() == expected.tobytes()
    assert form._theta[0] == upper[0]
    assert form._theta[1] == lower[1]
    assert not np.array_equal(form._theta, unclipped)


def test_best_total_regret_sums_all_violated_agents() -> None:
    # At convergence every per-agent regret is zero, so record the total at a
    # non-converged theta instead.
    #
    # 3 agents, 1 item, r = +1, nu = [-0.3, -0.7, -1.2], observed = taken. At
    # theta_init = 0 the priced demand takes nothing (score nu < 0), so each
    # agent's regret is |nu|; the first step records
    # best_total_regret = sum |nu| = 2.2 before stepping.
    nu = np.array([-0.3, -0.7, -1.2])
    arrays = {
        "observables": np.ones((3, 1)),
        "shocks": nu.reshape(3, 1, 1),
        "observed": np.ones((3, 1), dtype=bool),
        "theta_true": np.array([3.0]),
    }
    expected_total = float(abs(-0.3) + abs(-0.7) + abs(-1.2))

    form = _single_step(
        arrays,
        bounds=(np.full(1, -10.0), np.full(1, 10.0)),
        theta_init=np.zeros(1),
        step=0.1,
        transport=SerialTransport(),
    )
    assert form._best_total == expected_total


def test_update_ships_bundle_key_packing_the_chosen_bundle() -> None:
    # The walk itself never reads bundle_key (agent ids are unique), so capture
    # the shipped key and check its content directly.
    #
    # 1 agent, 5 items, r = +1, nu = [+1, -1, +1, +1, -1]. At theta_init = 0
    # the priced demand takes item k iff nu_k > 0, so chosen = [T, F, T, T, F].
    nu = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    arrays = {
        "observables": np.ones((1, 5)),
        "shocks": nu.reshape(1, 1, 5),
        "observed": np.ones((1, 5), dtype=bool),
        "theta_true": np.full(5, 5.0),
    }
    shipped: list[CutRow] = []

    class _Capture(SerialTransport):
        def exchange_cuts(self, rows, owners):  # type: ignore[override]
            shipped.extend(rows)
            return super().exchange_cuts(rows, owners)

    _single_step(
        arrays,
        bounds=(np.full(5, -10.0), np.full(5, 10.0)),
        theta_init=np.zeros(5),
        step=0.1,
        transport=_Capture(),
    )
    assert len(shipped) == 1
    # packbits is MSB-first: [T, F, T, T, F] -> bits 10110000.
    assert shipped[0].bundle_key == bytes([0b10110000])

    # combrum's own key encoder (_pack_bundle, what every NSlack cut ships
    # under) and its CutRow.bundle inverse.
    from combrum.transport.base import _pack_bundle

    chosen = np.array([True, False, True, True, False])
    key = _pack_bundle(chosen)
    # The decode must recover values, dtype, and the read-only flag -- not
    # merely round-trip its own encoder.
    recovered = CutRow(
        rep_id=0, agent_id=0, phi=np.array([1.0]), epsilon=1.0, bundle_key=key
    ).bundle
    np.testing.assert_array_equal(recovered, chosen)
    assert recovered.dtype == np.bool_
    assert not recovered.flags.writeable

    # bundle_key also breaks delivery-order ties among rows sharing
    # (rep_id, agent_id).
    key_hi = np.packbits([True, False, False, False, False]).tobytes()  # 0x80
    key_lo = np.packbits([False, True, False, False, False]).tobytes()  # 0x40
    row_hi = CutRow(rep_id=0, agent_id=7, phi=np.array([1.0]), epsilon=1.0,
                    bundle_key=key_hi)
    row_lo = CutRow(rep_id=0, agent_id=7, phi=np.array([2.0]), epsilon=2.0,
                    bundle_key=key_lo)
    ordered = SerialTransport().exchange_cuts(
        [row_hi, row_lo], np.zeros(1, dtype=np.int64)
    )
    # 0x40 < 0x80, so the input order is reversed by the key comparison.
    assert [row.bundle_key for row in ordered] == [key_lo, key_hi]


def test_collective_guard_agrees_a_body_failure() -> None:
    # The guard, used exactly as update() uses it, must convert a body failure
    # into a TransportError rather than let the raw error escape on one rank.
    transport = SerialTransport()

    def guarded_body() -> None:
        with transport.collective():
            if transport.rank == 0:
                raise ValueError("body failed on rank 0")

    with pytest.raises(TransportError) as excinfo:
        guarded_body()
    # The conversion must keep the origin rank and the original message.
    assert excinfo.value.rank == 0
    assert "body failed on rank 0" in excinfo.value.message
    assert "body failed on rank 0" in str(excinfo.value)

    # An already-guarded TransportError passes through unchanged: same origin
    # rank, same message.
    def reguarded_body() -> None:
        with transport.collective():
            raise TransportError(3, "already agreed on rank 3")

    with pytest.raises(TransportError) as reinfo:
        reguarded_body()
    assert reinfo.value.rank == 3
    assert reinfo.value.message == "already agreed on rank 3"
