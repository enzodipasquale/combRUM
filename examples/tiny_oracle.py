"""Tiny distributed public-API example for a bundle-choice oracle."""

from __future__ import annotations

import itertools
import os

import numpy as np

import combrum as cb


def _transport() -> cb.Transport:
    mpi_env = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MV2_COMM_WORLD_SIZE",
    )
    if any(name in os.environ for name in mpi_env):
        return cb.MpiTransport()
    return cb.SerialTransport()


def main() -> None:
    transport: cb.Transport | None = None
    try:
        n_items = 3
        n_obs = 100
        n_simulations = 5
        rng = np.random.default_rng(17)
        transport = _transport()

        bundles = np.array(
            list(itertools.product([0.0, 1.0], repeat=n_items)),
            dtype=np.float64,
        )
        theta_true = np.array([0.2, 0.1, -0.1])
        shocks = rng.normal(scale=0.25, size=(n_obs, n_simulations, n_items))

        observed = np.zeros((n_obs, n_items), dtype=np.float64)
        for i in range(n_obs):
            scores = bundles @ (theta_true + shocks[i, 0])
            observed[i] = bundles[np.argmax(scores)]

        class TinyOracle(cb.Oracle):
            def __init__(self, shocks: np.ndarray, bundles: np.ndarray) -> None:
                self.shocks = shocks
                self.bundles = bundles
                self.n_obs = self.shocks.shape[0]

            def price(self, theta: np.ndarray, agent_id: int) -> cb.Demand:
                i = agent_id % self.n_obs
                s = agent_id // self.n_obs
                scores = self.bundles @ theta
                scores = scores + self.bundles @ self.shocks[i, s]
                j = np.argmax(scores)
                return cb.Demand.exact(self.bundles[j], scores[j])

            def price_batch(
                self, theta: np.ndarray, local_ids: np.ndarray
            ) -> dict[int, cb.Demand]:
                return {
                    agent_id: self.price(theta, agent_id)
                    for agent_id in local_ids
                }

        class BundleFeatures(cb.FeatureMap):
            def setup_observed(
                self, transport: cb.Transport, observation_ids: np.ndarray
            ) -> None:
                pass

            def features(
                self, agent_id: int, bundle: np.ndarray
            ) -> tuple[np.ndarray, float]:
                i = agent_id % n_obs
                s = agent_id // n_obs
                return bundle, shocks[i, s] @ bundle

            def observed_features_batch(
                self, observation_ids: np.ndarray
            ) -> np.ndarray:
                return np.ascontiguousarray(observed[observation_ids], dtype=np.float64)

        params = cb.Parameters({"taste": (-2.0, 2.0, n_items)})
        features = BundleFeatures()
        model = cb.Model(
            TinyOracle(shocks, bundles),
            params,
            features=features,
            observed_features=features,
            formulation=cb.NSlack,
        )

        fit = cb.estimate_distributed(
            model,
            n_observations=n_obs,
            n_simulations=n_simulations,
            transport=transport,
            master_backend="highs",
            tolerance=1e-8,
            max_iterations=60,
        )
        if transport.rank == 0:
            print("theta_hat:", fit.theta_hat.round(6).tolist())
            print("objective:", round(float(fit.objective), 6))
            print("converged:", fit.metadata["converged"])

        boot = cb.bootstrap_distributed(
            model,
            n_bootstrap=50,
            base_seed=23,
            n_observations=n_obs,
            n_simulations=n_simulations,
            transport=transport,
            warm_start=fit,
            master_backend="highs",
            tolerance=1e-8,
            max_iterations=60,
        )
        if transport.rank == 0:
            print("bootstrap se:", boot.se(only_converged=False).round(6).tolist())
    finally:
        if transport is not None:
            close = getattr(transport, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
