"""Exact test-grade oracles + feature maps over the canonical families.

One :class:`FamilyProblem` per family bundles what the test-local walk
needs: an exact pricing oracle over the family arrays, the matching
feature map ``(agent_id, bundle) -> (phi, eps)``, and the theta geometry
the captured references solved under (K, box bounds). Both oracles are
solver-free: the toy demand rule is closed-form and the QKP subproblem is
brute-force enumerable at the fixture size, so exactness needs no solver
license or dependency.

Underscore-prefixed module: test support, never collected by pytest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from combrum.demand import Demand
from combrum.interface_resolution import FeatureMap
from _support.constants import THETA_BOUND
from combrum.oracle import Oracle
from combrum.transport.base import Transport


@dataclass(frozen=True)
class FamilyProblem:
    """One family's walk inputs: oracle, feature maps, theta geometry."""

    oracle: Oracle
    features: Callable[[int, np.ndarray], tuple[np.ndarray, float]]
    observed_features: Callable[[int, np.ndarray], np.ndarray]
    K: int
    theta_bounds: tuple[np.ndarray, np.ndarray]


class ToyOracle(Oracle):
    """Exact toy pricing: take item k iff ``r_k * theta_k + nu_k > 0``.

    Strict ">" mirrors the family generator, so pricing at theta_true
    reproduces every observed bundle bitwise. S == 1, so the global agent
    id indexes the family arrays directly.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._r = np.asarray(arrays["observables"], dtype=np.float64)
        self._nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]

    def setup(self, transport: Transport, local_ids: np.ndarray) -> None:
        pass

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        scores = self._r[agent_id] * theta + self._nu[agent_id]
        bundle = scores > 0.0
        return Demand.exact(
            bundle=bundle, payoff=float(np.where(bundle, scores, 0.0).sum())
        )

    def price_batch(
        self, theta: np.ndarray, local_ids: np.ndarray
    ) -> Mapping[int, Demand]:
        # The batched twin: scores each id with the same elementwise
        # expression and the same float64 reduction order, making
        # price_batch(theta, ids) bitwise equal to [price(theta, i) for i in
        # ids] — the conformance gate.
        ids = np.asarray(local_ids, dtype=np.int64)
        out: dict[int, Demand] = {}
        for agent_id in ids:
            a = int(agent_id)
            scores = self._r[a] * theta + self._nu[a]
            bundle = scores > 0.0
            out[a] = Demand.exact(
                bundle=bundle,
                payoff=float(np.where(bundle, scores, 0.0).sum()),
            )
        return out


class QKPOracle(Oracle):
    """Exact QKP pricing by enumeration over ``{0,1}^M`` under capacity.

    Maximizes ``alpha * x_a . b - delta . b + 0.5 * lambda * b'Qb
    + nu_a . b`` subject to ``weights . b <= capacity_a`` with
    ``theta = [alpha, delta_1..M, lambda]`` — the exact value expression
    the family generator enumerated, so pricing at theta_true reproduces
    every observed bundle. S == 1, so the global id indexes the arrays.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._x = np.asarray(arrays["x"], dtype=np.float64)
        self._nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
        self._cap = np.asarray(arrays["capacities"], dtype=np.float64)
        q = np.asarray(arrays["Q"], dtype=np.float64)
        weights = np.asarray(arrays["weights"], dtype=np.float64)
        m = weights.shape[0]
        count = np.arange(2**m)
        self._bundles = (
            (count[:, None] >> np.arange(m)[None, :]) & 1
        ).astype(np.float64)
        self._loads = self._bundles @ weights
        # The lambda-free half of the quadratic term, precomputed once:
        # scaling by lambda at price time keeps the per-call work linear
        # in the bundle count.
        self._quad = 0.5 * np.einsum(
            "bj,jk,bk->b", self._bundles, q, self._bundles
        )

    def setup(self, transport: Transport, local_ids: np.ndarray) -> None:
        pass

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        alpha = float(theta[0])
        delta = np.asarray(theta[1:-1], dtype=np.float64)
        lam = float(theta[-1])
        utility = (
            self._bundles @ (alpha * self._x[agent_id] - delta + self._nu[agent_id])
            + lam * self._quad
        )
        utility = np.where(self._loads <= self._cap[agent_id], utility, -np.inf)
        best = int(np.argmax(utility))
        return Demand.exact(
            bundle=self._bundles[best] > 0.5, payoff=float(utility[best])
        )

    def price_batch(
        self, theta: np.ndarray, local_ids: np.ndarray
    ) -> Mapping[int, Demand]:
        # The batched twin: the same per-agent enumeration, one id at a
        # time. The arithmetic per id is byte-identical to price (same
        # precomputed quad term, same capacity mask, same argmax), so the
        # batch result is bitwise equal to the per-agent result (the
        # conformance gate), while the shape is the batch-call contract.
        ids = np.asarray(local_ids, dtype=np.int64)
        out: dict[int, Demand] = {}
        for agent_id in ids:
            out[int(agent_id)] = self.price(theta, int(agent_id))
        return out


def toy_problem(arrays: Mapping[str, np.ndarray]) -> FamilyProblem:
    """Fresh toy oracle + features: ``phi = b * r_a``, ``eps = b . nu_a``."""
    r = np.asarray(arrays["observables"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    n_items = r.shape[1]

    def features(agent_id: int, bundle: np.ndarray) -> tuple[np.ndarray, float]:
        b = np.asarray(bundle, dtype=np.float64)
        return observed_features(agent_id, b), float(b @ nu[agent_id])

    def observed_features(agent_id: int, bundle: np.ndarray) -> np.ndarray:
        b = np.asarray(bundle, dtype=np.float64)
        return b * r[agent_id]

    return FamilyProblem(
        oracle=ToyOracle(arrays),
        features=features,
        observed_features=observed_features,
        K=n_items,
        theta_bounds=(
            np.full(n_items, -THETA_BOUND),
            np.full(n_items, THETA_BOUND),
        ),
    )


def qkp_problem(arrays: Mapping[str, np.ndarray]) -> FamilyProblem:
    """Fresh QKP oracle + features in the ``[alpha, delta, lambda]`` layout.

    ``phi = [x_a . b, -b, 0.5 * b'Qb]`` and ``eps = nu_a . b``, exactly
    the canonical parameterisation the captured references consumed; the
    box pins ``alpha >= 0`` and ``lambda >= 0`` like the references'
    parameter blocks.
    """
    x = np.asarray(arrays["x"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    q = np.asarray(arrays["Q"], dtype=np.float64)
    m = x.shape[1]

    def features(agent_id: int, bundle: np.ndarray) -> tuple[np.ndarray, float]:
        phi = observed_features(agent_id, bundle)
        b = np.asarray(bundle, dtype=np.float64)
        return phi, float(nu[agent_id] @ b)

    def observed_features(agent_id: int, bundle: np.ndarray) -> np.ndarray:
        b = np.asarray(bundle, dtype=np.float64)
        phi = np.empty(m + 2, dtype=np.float64)
        phi[0] = float(x[agent_id] @ b)
        phi[1 : m + 1] = -b
        phi[m + 1] = 0.5 * float(b @ (q @ b))
        return phi

    lower = np.concatenate([[0.0], np.full(m, -THETA_BOUND), [0.0]])
    upper = np.full(m + 2, THETA_BOUND)
    return FamilyProblem(
        oracle=QKPOracle(arrays),
        features=features,
        observed_features=observed_features,
        K=m + 2,
        theta_bounds=(lower, upper),
    )


# --- batched FeatureMap variants for the features either-one tests -----------
#
# Each wraps the same per-agent feature function the families inject and
# also overrides features_batch, so it is a both-supplied FeatureMap whose
# batch return is byte-identical to the per-agent path row-by-row. A
# subclass that drops `features` exercises the optimized-only path; the
# divergent subclass below exercises the both-supplied fail.


class _BatchedFeatureMap(FeatureMap):
    """A FeatureMap over a per-agent ``(agent_id, bundle) -> (phi, eps)``.

    Overrides both members: ``features`` forwards to the wrapped per-agent
    function, ``features_batch`` calls it row-by-row and stacks — so the
    optimized return equals the per-agent path bitwise (the conformance the
    both-supplied gate checks), while the call shape is the batch contract.
    """

    def __init__(
        self,
        per_agent: Callable[[int, np.ndarray], tuple[np.ndarray, float]],
    ) -> None:
        self._per_agent = per_agent

    def features(
        self, agent_id: int, bundle: np.ndarray
    ) -> tuple[np.ndarray, float]:
        return self._per_agent(int(agent_id), bundle)

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        id_arr = np.asarray(ids, dtype=np.int64)
        rows = [
            self._per_agent(int(a), np.asarray(bundles)[r])
            for r, a in enumerate(id_arr)
        ]
        phi = np.stack(
            [np.asarray(p, dtype=np.float64) for p, _ in rows], axis=0
        )
        eps = np.array([float(e) for _, e in rows], dtype=np.float64)
        return phi, eps

    def __call__(
        self, agent_id: int, bundle: np.ndarray
    ) -> tuple[np.ndarray, float]:
        # _walk builds the master's c_theta objective by calling the map per
        # agent for the observed bundles, independent of the formulation's
        # resolved path. Routing through features_batch (a one-row batch)
        # means the batch-only subclasses work here too. This is for the
        # c_theta build only; the formulation still resolves the either-one by
        # MRO (resolve_features checks isinstance FeatureMap first).
        phi, eps = self.features_batch(
            np.asarray([agent_id], dtype=np.int64),
            np.asarray([bundle]),
        )
        return np.ascontiguousarray(phi[0], dtype=np.float64), float(eps[0])


class _BatchOnlyFeatureMap(_BatchedFeatureMap):
    """A FeatureMap overriding only ``features_batch`` (the optimized path).

    Resolution must then pick the batch member alone — the optimized-only
    mode.
    """

    # Restore the base ABC raising default (the parent overrides features), so
    # this is the optimized-only either-one form.
    features = FeatureMap.features


class _DivergentBatchedFeatureMap(_BatchedFeatureMap):
    """A both-supplied FeatureMap whose batch disagrees with its per-agent.

    The batch perturbs eps by an above-tolerance delta on one row, so the
    both-supplied conformance gate must fail; the per-agent member is the
    documented fallback.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        if eps.size:
            eps = eps.copy()
            eps[0] += 1e-6  # above the 1e-13 comparison tolerance
        return phi, eps


def toy_feature_map(
    arrays: Mapping[str, np.ndarray],
) -> _BatchedFeatureMap:
    """A both-supplied FeatureMap matching :func:`toy_problem`'s features."""
    return _BatchedFeatureMap(toy_problem(arrays).features)


def qkp_feature_map(
    arrays: Mapping[str, np.ndarray],
) -> _BatchedFeatureMap:
    """A both-supplied FeatureMap matching :func:`qkp_problem`'s features."""
    return _BatchedFeatureMap(qkp_problem(arrays).features)


def toy_feature_map_batch_only(
    arrays: Mapping[str, np.ndarray],
) -> _BatchOnlyFeatureMap:
    """An optimized-only FeatureMap matching :func:`toy_problem`'s features."""
    return _BatchOnlyFeatureMap(toy_problem(arrays).features)


def qkp_feature_map_batch_only(
    arrays: Mapping[str, np.ndarray],
) -> _BatchOnlyFeatureMap:
    """An optimized-only FeatureMap matching :func:`qkp_problem`'s features."""
    return _BatchOnlyFeatureMap(qkp_problem(arrays).features)


def divergent_feature_map(
    arrays: Mapping[str, np.ndarray],
) -> _DivergentBatchedFeatureMap:
    """A both-supplied FeatureMap whose batch path violates conformance."""
    return _DivergentBatchedFeatureMap(toy_problem(arrays).features)


# --- divergent FeatureMaps for the wholesale-capture gate -------------------
#
# Each perturbation below is an optimized-only (``features_batch``-only)
# FeatureMap. It resolves to Mode.OPTIMIZED, so the divergent batch return
# flows straight through ``feature_rows`` into the formulation (the
# both-supplied conformance gate fires in Mode.BOTH only) and on into the
# ``StepRecord`` capture, where the wholesale comparator
# (``test_wholesale_capture``) must fail — one perturbation per filter stage.
# Each test first runs the unperturbed ``*_batch_only`` map through the same
# comparator, so a raise is attributable to the perturbation alone.
#
# Sizing: every continuous perturbation sits above the 1e-13 comparison
# tolerance (1e-6, 1e-9, or 1e-11); the support perturbation flips an exact
# zero, and the OneSlack ones straddle the aggregate identity or install gate.


class _PhiValuePerturbationMap(_BatchOnlyFeatureMap):
    """Lifts the first nonzero phi entry of every featurised row by ``1e-6``.

    The drift rides into the master objective and the installed cuts, moving
    the priced bundle keys and the admit-side violation. A nonzero target
    keeps it a pure value drift, distinct from the support flip below.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        phi = phi.copy()
        for r in range(phi.shape[0]):
            nz = np.flatnonzero(phi[r])
            if nz.size:
                phi[r, nz[0]] += 1e-6
        return phi, eps


class _PhiSupportPerturbationMap(_BatchOnlyFeatureMap):
    """Turns the first exact-zero phi entry of the first row into ``1e-12``.

    A discrete identity flip, however tiny the magnitude: it changes the
    zero mask that ``highs.py``'s ``np.flatnonzero(row.phi)`` keys the
    installed column set on, and on the gurobi master it also drifts the
    priced reduced costs.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        phi = phi.copy()
        for r in range(phi.shape[0]):
            z = np.flatnonzero(phi[r] == 0.0)
            if z.size:
                phi[r, z[0]] = 1e-12
                break
        return phi, eps


class _EpsPerturbationMap(_BatchOnlyFeatureMap):
    """Lifts the first featurised row's eps by ``1e-6``.

    eps enters the cut row and the admit-side violation
    ``phi.theta + eps - u``, so the drift surfaces in the captured admit
    violations and reduced costs.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        eps = eps.copy()
        if eps.size:
            eps[0] += 1e-6
        return phi, eps


class _AggregateBytesPerturbationMap(_BatchOnlyFeatureMap):
    """Lifts the first nonzero phi entry of the first row by ``1e-11``.

    Sized to flip aggregate bytes only: large enough to drift the summed
    aggregate by ~``1e-10`` and flip the SHA-256 over ``[phi_agg, eps_agg]``
    (``oneslack.py:_aggregate_key``), yet small enough to preserve the
    OneSlack convergence shape on these families, so the comparison reaches
    the aggregate fields instead of stopping at the stream-length check.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        phi = phi.copy()
        for r in range(phi.shape[0]):
            nz = np.flatnonzero(phi[r])
            if nz.size:
                phi[r, nz[0]] += 1e-11
                break
        return phi, eps


class _InstallGatePerturbationMap(_BatchOnlyFeatureMap):
    """Lifts the first nonzero phi entry of the first row by ``1e-6``.

    A single-row lift cannot be absorbed into theta (a uniform all-rows lift
    could), so the aggregate slack never settles to ``<= ctx.tolerance`` at
    the clean path's convergence iteration: the install gate
    ``violation > ctx.tolerance`` (``oneslack.py:260``) keeps firing and the
    convergence shapes diverge, which the stream-length check fails.
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        phi = phi.copy()
        for r in range(phi.shape[0]):
            nz = np.flatnonzero(phi[r])
            if nz.size:
                phi[r, nz[0]] += 1e-6
                break
        return phi, eps


class _SchedulePerturbationMap(_BatchOnlyFeatureMap):
    """Lifts every row's first nonzero phi entry by ``1e-6``.

    The drift moves the master duals behind the driver's
    ``DualConcentration`` schedule payload, so the NSlack dual-informed walk
    no longer converges on the clean path's iteration count and the
    schedule-concentration stream lengths diverge. (max_weights saturate at
    ``1.0`` on these families, so a continuous weight drift cannot move them;
    the divergence comes through the shape/support the dual shift induces.)
    """

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, eps = super().features_batch(ids, bundles)
        phi = phi.copy()
        for r in range(phi.shape[0]):
            nz = np.flatnonzero(phi[r])
            if nz.size:
                phi[r, nz[0]] += 1e-6
        return phi, eps


class _PerturbationPriceToyOracle(ToyOracle):
    """A ToyOracle that lifts agent 0's priced payoff by ``1e-6``.

    The chosen bundle stays byte-identical, so the discrete demand identity
    holds and only the continuous payoff (hence the certified gap) drifts.
    The priced-demand stream captured in ``priced_features`` then differs
    between a clean and a perturbed run, failing the wholesale comparator's
    payoff-drift check. (The batched price path gets its own divergence from
    ``_DivergentBatchToy`` in ``test_either_one.py``; this oracle drives the
    price-stage drift through the wholesale capture's demand stream instead.)
    """

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        demand = super().price(theta, int(agent_id))
        if int(agent_id) == 0:
            return Demand.exact(
                bundle=demand.bundle,
                payoff=demand.payoff + 1e-6,
            )
        return demand


def toy_perturbation_price_oracle(
    arrays: Mapping[str, np.ndarray],
) -> _PerturbationPriceToyOracle:
    """A toy oracle whose priced payoff diverges above-tolerance on agent 0."""
    return _PerturbationPriceToyOracle(arrays)


def toy_phi_value_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _PhiValuePerturbationMap:
    """Batch-only perturbation: above-tolerance phi value drift over toy."""
    return _PhiValuePerturbationMap(toy_problem(arrays).features)


def toy_phi_support_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _PhiSupportPerturbationMap:
    """Batch-only perturbation: exact-zero -> 1e-12 mask flip over toy."""
    return _PhiSupportPerturbationMap(toy_problem(arrays).features)


def toy_eps_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _EpsPerturbationMap:
    """Batch-only perturbation: above-tolerance eps drift over toy."""
    return _EpsPerturbationMap(toy_problem(arrays).features)


def toy_aggregate_bytes_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _AggregateBytesPerturbationMap:
    """Batch-only perturbation: shape-preserving aggregate-byte flip over toy."""
    return _AggregateBytesPerturbationMap(toy_problem(arrays).features)


def toy_install_gate_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _InstallGatePerturbationMap:
    """Batch-only perturbation: OneSlack install-gate straddle over toy."""
    return _InstallGatePerturbationMap(toy_problem(arrays).features)


def toy_schedule_perturbation(
    arrays: Mapping[str, np.ndarray],
) -> _SchedulePerturbationMap:
    """Batch-only perturbation: schedule DualConcentration divergence over toy."""
    return _SchedulePerturbationMap(toy_problem(arrays).features)
