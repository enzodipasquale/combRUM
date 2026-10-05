"""Small bundle-choice example, explained step by step in notebooks/01_quickstart.ipynb."""

import itertools

import numpy as np

import combrum as cb

rng = np.random.default_rng(17)

N_ITEMS = 4
N_FEATURES = 3
N_OBS = 100
N_SIMULATIONS = 4
THETA_TRUE = np.array([0.9, -0.5, 0.35])
SHOCK_SCALE = 0.10

BUNDLES = np.array(list(itertools.product([0.0, 1.0], repeat=N_ITEMS)))
X = rng.normal(size=(N_OBS, N_ITEMS, N_FEATURES))
estimation_shocks = rng.normal(
    scale=SHOCK_SCALE,
    size=(N_OBS, N_SIMULATIONS, N_ITEMS),
)
observed_shocks = rng.normal(scale=SHOCK_SCALE, size=(N_OBS, N_ITEMS))


def simulate_observed(theta):
    observed = np.zeros((N_OBS, N_ITEMS))
    for i in range(N_OBS):
        utilities = X[i] @ theta + observed_shocks[i]
        scores = BUNDLES @ utilities
        observed[i] = BUNDLES[np.argmax(scores)]
    return observed


# Oracle: each simulated agent's best bundle, by enumeration.
class BundleOracle(cb.Oracle):
    def price_batch(self, theta, agent_ids):
        i = agent_ids % N_OBS
        s = agent_ids // N_OBS
        utilities = X[i] @ theta + estimation_shocks[i, s]
        scores = utilities @ BUNDLES.T
        choices = np.argmax(scores, axis=1)
        payoffs = scores[np.arange(agent_ids.size), choices]
        return cb.DemandBatch.exact(agent_ids, BUNDLES[choices], payoffs)


# Feature map: phi_i(d) and epsilon_i(d) for a batch of bundles.
class BundleFeatures(cb.FeatureMap):
    def features_batch(self, ids, bundles):
        i = ids % N_OBS
        s = ids // N_OBS
        return np.einsum("ij,ijk->ik", bundles, X[i]), np.einsum(
            "ij,ij->i", bundles, estimation_shocks[i, s]
        )


model = cb.Model(
    BundleOracle(),
    cb.Parameters({"taste": (-2.0, 2.0, N_FEATURES)}),
    features=BundleFeatures(),
)
data = cb.Data(
    observed_bundles=simulate_observed(THETA_TRUE),
    shocks=estimation_shocks,
    observables=X,
)

fit = cb.estimate(
    model,
    data,
    master_backend="highs",
    tolerance=1e-8,
    max_iterations=100,
)
boot = cb.bootstrap(
    model,
    data,
    n_bootstrap=10,
    weight_source=cb.ExponentialDraws(n_observations=N_OBS, base_seed=23),
    master_backend="highs",
    tolerance=1e-8,
    max_iterations=100,
)

print("theta_true:", THETA_TRUE.round(4).tolist())
print("theta_hat:", fit.theta_hat.round(4).tolist())
print("converged:", fit.metadata["converged"])
print("iterations:", fit.metadata["iterations"])
print("bootstrap se:", boot.se(only_converged=False).round(4).tolist())
