"""Network formation with one player fixed effect and node-shared MPI data."""

import argparse
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

import combrum as cb

K_BETA = 5
N_PLAYERS = 50
N_OBSERVATIONS = 1
N_SIMULATIONS = 1000
SHOCK_SEED = 4000
TOLERANCE = 1e-3
MAX_ITERATIONS = 2000
BETA_TRUE = np.array([0.8, 0.5, 1.0, 0.3, 0.4], dtype=np.float64)
GAMMA_TRUE = 0.5
FE_MEAN = -1.0
FE_SD = 0.5
DESIGN_SEED = 20260424


def min_cut_bundle(
    linear: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    pair: np.ndarray,
) -> np.ndarray:
    upper_rowsum = np.zeros(linear.size)
    np.add.at(upper_rowsum, edge_rows, pair)
    net = -linear - upper_rowsum
    magnitude = max(
        np.abs(net).max(initial=0.0),
        pair.max(initial=0.0),
    )
    scale = 1.0 if magnitude == 0.0 else 10.0 ** (
        8 - np.floor(np.log10(magnitude))
    )
    net_int = np.round(net * scale).astype(np.int64)
    pair_int = np.round(pair * scale).astype(np.int64)

    source, sink = 0, 1
    node_ids = np.arange(linear.size) + 2
    select = net_int < 0
    present = pair_int > 0
    rows = np.concatenate(
        [
            np.full(select.sum(), source),
            node_ids[~select],
            edge_rows[present] + 2,
        ]
    )
    cols = np.concatenate(
        [
            node_ids[select],
            np.full((~select).sum(), sink),
            edge_cols[present] + 2,
        ]
    )
    vals = np.concatenate([-net_int[select], net_int[~select], pair_int[present]])
    cap = csr_matrix(
        (vals, (rows, cols)),
        shape=(linear.size + 2, linear.size + 2),
        dtype=np.int64,
    )
    flow = maximum_flow(cap, source, sink).flow
    residual = (cap - flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    _, predecessors = breadth_first_order(
        residual,
        i_start=source,
        directed=True,
        return_predecessors=True,
    )
    reachable = predecessors != -9999
    reachable[source] = True
    return reachable[2:]


def make_arrays(
    T: int,
    N: int,
    S: int,
    *,
    design_seed: int,
    shock_seed: int,
) -> dict[str, np.ndarray]:
    T = int(T)
    N = int(N)
    S = int(S)
    M = T * (T - 1)
    rng = np.random.default_rng(design_seed)
    snd, rcv = np.nonzero(~np.eye(T, dtype=bool))
    pair_index = np.empty((T, T), dtype=np.int64)
    pair_index[snd, rcv] = np.arange(M, dtype=np.int64)
    recip = pair_index[rcv, snd]

    g1 = rng.integers(0, 2, size=T)
    g2 = rng.integers(0, 2, size=T)
    a = rng.uniform(0.0, 1.0, size=T)
    b = rng.normal(0.0, 1.0, size=T)
    z = rng.normal(0.0, 1.0, size=T)
    r = np.empty((M, K_BETA), dtype=np.float64)
    r[:, 0] = (g1[snd] == g1[rcv]).astype(np.float64)
    r[:, 1] = (g2[snd] == g2[rcv]).astype(np.float64)
    r[:, 2] = -np.abs(a[snd] - a[rcv])
    r[:, 3] = -np.abs(b[snd] - b[rcv])
    r[:, 4] = z[snd] * z[rcv]

    alpha = rng.normal(FE_MEAN, FE_SD, size=T)
    theta_true = np.r_[alpha, BETA_TRUE, GAMMA_TRUE]

    shocks = np.random.default_rng(shock_seed).standard_normal((N, S, M))
    edge_rows = np.flatnonzero(np.arange(M, dtype=np.int64) < recip)
    edge_cols = recip[edge_rows]
    xbeta = np.einsum("mk,k->m", r, BETA_TRUE, optimize=True)
    base = alpha[snd] + alpha[rcv] + xbeta
    pair = np.full(edge_rows.size, GAMMA_TRUE)
    observed = np.empty((N, M), dtype=bool)
    for obs_id in range(N):
        observed[obs_id] = min_cut_bundle(
            base + shocks[obs_id, 0],
            edge_rows,
            edge_cols,
            pair,
        )
    return {
        "r": r,
        "snd": snd,
        "rcv": rcv,
        "edge_rows": edge_rows,
        "edge_cols": edge_cols,
        "shocks": shocks,
        "observed": observed,
        "theta_true": theta_true,
    }


class NetworkFeatures(cb.FeatureMap):
    def __init__(self, arrays, T):
        self.T = int(T)
        self.N = arrays["observed"].shape[0]
        self.M = self.T * (self.T - 1)
        self.K = self.T + K_BETA + 1
        self.r = arrays["r"]
        self.edge_rows = arrays["edge_rows"]
        self.edge_cols = arrays["edge_cols"]
        self.shocks = arrays["shocks"]
        self.observed = arrays["observed"]

        snd = arrays["snd"]
        rcv = arrays["rcv"]
        cols = np.arange(self.M)
        ones = np.ones(self.M)
        sender = csr_matrix((ones, (snd, cols)), shape=(self.T, self.M))
        receiver = csr_matrix((ones, (rcv, cols)), shape=(self.T, self.M))
        self.incidence = sender + receiver

    def observed_features_batch(self, observation_ids):
        Phi, _eps = self.features_batch(observation_ids, self.observed[observation_ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):
        beta_start = self.T
        obs_ids = ids % self.N
        sim_ids = ids // self.N
        reciprocal = bundles[:, self.edge_rows] * bundles[:, self.edge_cols]
        Phi = np.zeros((bundles.shape[0], self.K), dtype=np.float64)
        Phi[:, : self.T] = (self.incidence @ bundles.T).T
        Phi[:, beta_start : beta_start + K_BETA] = np.einsum(
            "nm,mk->nk", bundles, self.r, optimize=True
        )
        Phi[:, -1] = reciprocal.sum(axis=1)
        eps = np.einsum("nm,nm->n", self.shocks[obs_ids, sim_ids], bundles, optimize=True)
        if aggregate:
            return weights @ Phi, float(weights @ eps)
        return Phi, eps


class NetworkDemandOracle(cb.Oracle):
    def __init__(self, arrays, T):
        self.T = int(T)
        self.N = arrays["observed"].shape[0]
        self.M = self.T * (self.T - 1)
        self.r = arrays["r"]
        self.snd = arrays["snd"]
        self.rcv = arrays["rcv"]
        self.edge_rows = arrays["edge_rows"]
        self.edge_cols = arrays["edge_cols"]
        self.shocks = arrays["shocks"]

    def price_batch(self, theta: np.ndarray, local_ids: np.ndarray) -> dict[int, cb.Demand]:
        alpha = theta[: self.T]
        beta = theta[self.T : self.T + K_BETA]
        gamma = theta[-1]
        base = alpha[self.snd] + alpha[self.rcv] + np.einsum(
            "mk,k->m", self.r, beta, optimize=True
        )
        pair = np.full(self.edge_rows.size, gamma)
        demands: dict[int, cb.Demand] = {}
        for agent_id in local_ids:
            obs_id = agent_id % self.N
            sim_id = agent_id // self.N
            linear = base + self.shocks[obs_id, sim_id]
            bundle = min_cut_bundle(linear, self.edge_rows, self.edge_cols, pair)
            reciprocal = np.count_nonzero(bundle[self.edge_rows] & bundle[self.edge_cols])
            payoff = np.sum(linear, dtype=np.float64, where=bundle) + gamma * reciprocal
            demands[agent_id] = cb.Demand.exact(bundle, payoff)
        return demands


def make_transport(kind):
    if kind == "mpi" or (kind == "auto" and "OMPI_COMM_WORLD_SIZE" in os.environ):
        return cb.MpiTransport()
    return cb.SerialTransport()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-backend", default="auto")
    parser.add_argument("--transport", default="auto")
    parser.add_argument("--activity-level", default="iterations")
    args = parser.parse_args()

    transport = make_transport(args.transport)
    publish = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            publish = make_arrays(
                N_PLAYERS,
                N_OBSERVATIONS,
                N_SIMULATIONS,
                design_seed=DESIGN_SEED,
                shock_seed=SHOCK_SEED,
            )
    arrays = dict(transport.node_shared(publish))

    parameters = cb.Parameters(
        {
            "alpha": (-8.0, 8.0, N_PLAYERS),
            "beta": (-8.0, 8.0, K_BETA),
            "gamma": (0.0, 8.0, 1),
        }
    )
    features = NetworkFeatures(arrays, N_PLAYERS)
    model = cb.Model(
        NetworkDemandOracle(arrays, N_PLAYERS),
        parameters,
        features=features,
        observed_features=features,
        formulation=cb.NSlack,
    )
    activity = None
    if args.activity_level.lower() != "off":
        activity = cb.ActivityConfig(label="network", level=args.activity_level, stdout=True)

    fit = cb.estimate_distributed(
        model,
        n_observations=arrays["observed"].shape[0],
        n_simulations=N_SIMULATIONS,
        transport=transport,
        master_backend=args.master_backend,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        cut_policy=cb.PurgeInactive(max_age=15),
        activity=activity,
    )
    if transport.rank != 0:
        return

    estimate = parameters.unpack(fit.theta_hat)
    truth = parameters.unpack(arrays["theta_true"])
    alpha_hat = np.asarray(estimate["alpha"], dtype=np.float64)
    alpha_true = np.asarray(truth["alpha"], dtype=np.float64)
    beta_hat = np.asarray(estimate["beta"], dtype=np.float64)
    gamma_hat = float(np.asarray(estimate["gamma"], dtype=np.float64)[0])
    gamma_true = float(np.asarray(truth["gamma"], dtype=np.float64)[0])
    M = N_PLAYERS * (N_PLAYERS - 1)

    print(
        "network:",
        f"T={N_PLAYERS}",
        f"N={arrays['observed'].shape[0]}",
        f"M={M}",
        f"S={N_SIMULATIONS}",
        f"K={parameters.K}",
        f"observed_links={int(arrays['observed'].sum())}",
    )
    print("beta_hat:", beta_hat.round(6).tolist())
    print("gamma_hat:", round(gamma_hat, 6))
    print("gamma_true:", round(gamma_true, 6))
    print("alpha_corr:", round(float(np.corrcoef(alpha_hat, alpha_true)[0, 1]), 6))
    print("objective:", round(float(fit.objective), 6))
    print("runtime seconds:", round(float(fit.runtime_seconds), 3))
    print("converged:", bool(fit.metadata["converged"]))
    print("iterations:", int(fit.metadata["iterations"]))
    print("active cuts:", int(fit.n_active_cuts))


if __name__ == "__main__":
    main()
