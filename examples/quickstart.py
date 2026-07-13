"""Small bundle-choice example."""

import itertools
import os

import numpy as np

import combrum as cb


N = 300
S = 5
M = 3
BUNDLES = np.array(list(itertools.product([0.0, 1.0], repeat=M)), dtype=np.float64)
BUNDLE_SIZES = BUNDLES.sum(axis=1)
THETA_TRUE = np.array([0.2, 0.1, -0.1, 0.5], dtype=np.float64)


# Oracle: solve each simulated agent's bundle problem.
class BundleOracle(cb.Oracle):
    def __init__(self, arrays):
        self.x = arrays["x"]
        self.shocks = arrays["shocks"]

    def price_batch(self, theta, agent_ids):
        obs = agent_ids % N
        sim = agent_ids // N
        scores = (
            np.einsum("bm,m->b", BUNDLES, theta[:M])
            + theta[-1] * self.x[obs, None] * BUNDLE_SIZES
            + np.einsum("nm,bm->nb", self.shocks[obs, sim], BUNDLES, optimize=True)
        )
        best = np.argmax(scores, axis=1)
        return cb.DemandBatch.exact(
            agent_ids,
            BUNDLES[best],
            scores[np.arange(agent_ids.size), best],
        )


# Feature map: compute phi_i(d) and epsilon_i(d).
class BundleFeatures(cb.FeatureMap):
    def __init__(self, arrays):
        self.x = arrays["x"]
        self.shocks = arrays["shocks"]
        self.observed = arrays["observed"]

    def observed_features_batch(self, observation_ids):
        Phi, _eps = self.features_batch(observation_ids, self.observed[observation_ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):
        obs = ids % N
        sim = ids // N
        Phi = np.empty((ids.size, M + 1), dtype=np.float64)
        Phi[:, :M] = bundles
        Phi[:, -1] = self.x[obs] * bundles.sum(axis=1)
        eps = np.einsum("ij,ij->i", self.shocks[obs, sim], bundles)
        if aggregate:
            return weights @ Phi, float(weights @ eps)
        return Phi, eps


# Use MPI when the script is launched with mpiexec.
transport = (
    cb.MpiTransport()
    if "OMPI_COMM_WORLD_SIZE" in os.environ
    else cb.SerialTransport()
)

# Build the data on each node leader, then share it with ranks on that node.
publish = {}
with transport.collective():
    if transport.node.node_rank == 0:
        rng = np.random.default_rng(17)
        x = rng.normal(size=N)
        shocks = rng.normal(scale=0.25, size=(N, S, M))
        dgp_utilities = (
            np.einsum("bm,m->b", BUNDLES, THETA_TRUE[:M])
            + THETA_TRUE[-1] * x[:, None] * BUNDLE_SIZES
            + np.einsum("nm,bm->nb", shocks[:, 0], BUNDLES, optimize=True)
        )
        publish = {
            "x": x,
            "shocks": shocks,
            "observed": BUNDLES[np.argmax(dgp_utilities, axis=1)],
        }

# combRUM model = oracle + parameters + feature map.
arrays = dict(transport.node_shared(publish))
features = BundleFeatures(arrays)
model = cb.Model(
    BundleOracle(arrays),
    cb.Parameters({"item": (-2.0, 2.0, M), "x_size": (-2.0, 2.0, 1)}),
    features=features,
    observed_features=features,
    formulation=cb.NSlack,
)

# Estimate by row generation, then run a multiplier bootstrap.
fit = cb.estimate_distributed(
    model,
    n_observations=N,
    n_simulations=S,
    transport=transport,
    master_backend="highs",
    tolerance=1e-8,
    max_iterations=60,
)
boot = cb.bootstrap_distributed(
    model,
    n_bootstrap=50,
    base_seed=23,
    n_observations=N,
    n_simulations=S,
    transport=transport,
    warm_start=fit,
    master_backend="highs",
    tolerance=1e-8,
    max_iterations=60,
)

if transport.rank == 0:
    print("theta_true:", THETA_TRUE.round(6).tolist())
    print("theta_hat:", fit.theta_hat.round(6).tolist())
    print("objective:", round(float(fit.objective), 6))
    print("converged:", bool(fit.metadata["converged"]))
    print("bootstrap se:", boot.se(only_converged=False).round(6).tolist())
