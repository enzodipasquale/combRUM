"""OneSlack's pricing rounds: in-out stabilization and the scaled master."""

from __future__ import annotations

import numpy as np
import pytest

import combrum as cb
from combrum.demand import DemandBatch
from combrum.formulations import OneSlack

TOLERANCE = 1e-6


class _Additive(cb.Oracle, cb.FeatureMap):
    """Agents take every item of positive utility ``theta_j + theta_x x_i + nu``.

    The criterion sums many hinges, so it is smooth at the scale of the
    design: the regime in-out stabilization is meant for.
    """

    def __init__(self, n_obs: int, n_sim: int, n_items: int) -> None:
        rng = np.random.default_rng(0)
        self.N, self.M = n_obs, n_items
        self.x = rng.normal(size=n_obs)
        self.nu = rng.normal(size=(n_obs, n_sim, n_items))
        theta = np.append(rng.normal(scale=0.5, size=n_items), 0.5)
        utility = theta[:-1] + theta[-1] * self.x[:, None]
        self.observed = (utility + rng.normal(size=utility.shape) > 0).astype(float)

    def price_batch(self, theta, agent_ids):  # type: ignore[no-untyped-def]
        obs, sim = agent_ids % self.N, agent_ids // self.N
        utility = theta[:-1] + theta[-1] * self.x[obs, None] + self.nu[obs, sim]
        return DemandBatch.exact(
            agent_ids,
            (utility > 0).astype(float),
            np.maximum(utility, 0.0).sum(axis=1),
        )

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):  # type: ignore[no-untyped-def]
        obs, sim = ids % self.N, ids // self.N
        phi = np.column_stack([bundles, self.x[obs] * bundles.sum(axis=1)])
        eps = np.einsum("ij,ij->i", self.nu[obs, sim], bundles)
        if aggregate:
            return weights @ phi, float(weights @ eps)
        return phi, eps


def _fit(design: _Additive, formulation: type, *, weight: float, tolerance: float):  # type: ignore[no-untyped-def]
    parameters = cb.Parameters({"item": (-3.0, 3.0, design.M), "x": (-3.0, 3.0, 1)})
    return cb.estimate(
        cb.Model(design, parameters, features=design, formulation=formulation),
        cb.Data(
            observed_bundles=design.observed,
            shocks=design.nu,
            observables=np.arange(design.N),
        ),
        master_backend="highs",
        tolerance=tolerance,
        weights=np.full(design.N, weight),
    )


class _Kelley(OneSlack):
    stabilization = 0.0


def _traced(rounds: list[tuple[bool, float]]) -> type[OneSlack]:
    class Traced(OneSlack):
        def finalise(self, reduced):  # type: ignore[no-untyped-def]
            outcome = super().finalise(reduced)
            rounds.append((self._stabilized, outcome.violation))
            return outcome

    return Traced


def test_stabilization_reaches_kelleys_optimum_in_fewer_rounds() -> None:
    design = _Additive(200, 5, 8)
    weight = 1.0 / (design.N * 5)
    kelley = _fit(design, _Kelley, weight=weight, tolerance=TOLERANCE)
    stabilized = _fit(design, OneSlack, weight=weight, tolerance=TOLERANCE)

    assert kelley.metadata["converged"] and stabilized.metadata["converged"]
    # Without stabilization every round but the certifying one cuts.
    assert kelley.n_active_cuts == kelley.metadata["iterations"] - 1
    assert stabilized.objective == pytest.approx(kelley.objective, abs=TOLERANCE)
    assert 2 * stabilized.metadata["iterations"] < kelley.metadata["iterations"]


def test_only_rounds_priced_at_the_master_solution_certify() -> None:
    design = _Additive(200, 5, 8)
    rounds: list[tuple[bool, float]] = []
    fit = _fit(
        design, _traced(rounds), weight=1.0 / (design.N * 5), tolerance=TOLERANCE
    )

    assert fit.metadata["converged"]
    assert any(stabilized for stabilized, _ in rounds)
    assert all(violation > TOLERANCE for stabilized, violation in rounds if stabilized)
    assert rounds[-1] == (False, pytest.approx(0.0, abs=TOLERANCE))


@pytest.mark.parametrize("weight", [1e-6, 1e12])
def test_master_scale_makes_the_fit_invariant_to_weight_units(weight: float) -> None:
    # Unscaled, the master's coefficients follow the weights while solver
    # tolerances stay absolute: at 1e-6 the fit's tolerance sinks below the
    # solver's and the fit stalls, at 1e12 HiGHS rejects the rows. Scaled,
    # every weight solves the same master, reported in the caller's units.
    design = _Additive(200, 5, 8)
    tolerance = TOLERANCE * design.N * 5
    unit = _fit(design, OneSlack, weight=1.0, tolerance=tolerance)
    fit = _fit(design, OneSlack, weight=weight, tolerance=tolerance * weight)

    assert fit.metadata["converged"]
    assert fit.metadata["iterations"] == unit.metadata["iterations"]
    np.testing.assert_allclose(fit.theta_hat, unit.theta_hat, atol=1e-9)
    assert fit.objective == pytest.approx(weight * unit.objective, rel=1e-9)
