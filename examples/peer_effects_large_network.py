"""Peer effects with MPI row generation and a warm-solved sigma grid."""

import argparse
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

import combrum as cb

N_NODES = 500
IID_SIMULATIONS = 50
CORRELATED_N = 20
CORRELATED_SIMULATIONS = 20
SIGMA_GRID_SIZE = 50
SIGMA_MIN = 0.2
SIGMA_MAX = 0.8
TOLERANCE = 1e-3
MAX_ITERATIONS = 1000
AVG_DEGREE = 10.0
LINK_PROB = 0.75
GRAPH_SEED = 20260423
IID_SEED = 49892
CORRELATED_SEED = 2718

BETA_TRUE = np.array([-1.0, -0.5, -1.0, 0.5], dtype=np.float64)
DELTA_TRUE = 0.20
ALPHA_TRUE = 0.50
SIGMA_TRUE = 0.50


def make_transport(kind):
    if kind == "mpi" or (kind == "auto" and "OMPI_COMM_WORLD_SIZE" in os.environ):
        return cb.MpiTransport()
    return cb.SerialTransport()


def min_cut_choice(
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    pair: np.ndarray,
) -> np.ndarray:
    upper_rowsum = np.zeros(linear.size)
    np.add.at(upper_rowsum, edge_i, pair)
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
            edge_i[present] + 2,
        ]
    )
    cols = np.concatenate(
        [
            node_ids[select],
            np.full((~select).sum(), sink),
            edge_j[present] + 2,
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
    *,
    T: int,
    iid_s: int,
    correlated_n: int,
    correlated_s: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(GRAPH_SEED)
    side = np.sqrt(T)
    radius = np.sqrt(AVG_DEGREE / (LINK_PROB * np.pi))
    positions = rng.uniform(0.0, side, size=(T, 2))
    distance = np.sqrt(
        ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1)
    )

    link_shocks = np.triu(rng.logistic(size=(T, T)), k=1)
    link_shocks = link_shocks + link_shocks.T
    W = (
        (distance <= radius)
        & (np.log(LINK_PROB / (1.0 - LINK_PROB)) >= link_shocks)
    )
    np.fill_diagonal(W, 0)

    edge_i, edge_j = np.nonzero(np.triu(W, k=1))
    pair = np.full(edge_i.size, DELTA_TRUE)

    degree = W.sum(axis=1, dtype=np.float64)
    inv_sqrt_degree = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=inv_sqrt_degree, where=degree > 0)
    W_tilde = inv_sqrt_degree[:, None] * W * inv_sqrt_degree[None, :]

    X = np.column_stack(
        [
            rng.integers(0, 2, size=T),
            rng.integers(0, 2, size=T),
            rng.uniform(0.0, 1.0, size=T),
            rng.uniform(0.0, 1.0, size=T),
        ]
    )
    xbeta = np.einsum("tk,k->t", X, BETA_TRUE, optimize=True)

    rng_iid = np.random.default_rng(IID_SEED)
    iid_eta_dgp = rng_iid.standard_normal((1, T))
    iid_eta_est = rng_iid.standard_normal((1, iid_s, T))
    iid_y = min_cut_choice(xbeta + iid_eta_dgp[0], edge_i, edge_j, pair)[None, :]

    rng_corr = np.random.default_rng(CORRELATED_SEED)
    corr_z = np.zeros(correlated_n, dtype=np.float64)
    corr_z[correlated_n // 2 :] = 1.0
    corr_eta_dgp = rng_corr.standard_normal((correlated_n, T))
    corr_etaW_dgp = np.einsum(
        "ns,ts->nt",
        corr_eta_dgp,
        W_tilde,
        optimize=True,
    )
    corr_eta_est = rng_corr.standard_normal((correlated_n, correlated_s, T))
    corr_etaW_est = np.einsum(
        "nst,ut->nsu",
        corr_eta_est,
        W_tilde,
        optimize=True,
    )

    corr_y = np.empty((correlated_n, T), dtype=bool)
    for i, shock in enumerate(corr_eta_dgp + SIGMA_TRUE * corr_etaW_dgp):
        corr_y[i] = min_cut_choice(
            xbeta + ALPHA_TRUE * corr_z[i] + shock,
            edge_i,
            edge_j,
            pair,
        )

    return {
        "X": X,
        "edge_i": edge_i,
        "edge_j": edge_j,
        "degree": degree,
        "iid_eta_est": iid_eta_est,
        "iid_y": iid_y,
        "corr_z": corr_z,
        "corr_eta_est": corr_eta_est,
        "corr_etaW_est": corr_etaW_est,
        "corr_y": corr_y,
    }


class PeerGame(cb.Oracle, cb.FeatureMap):
    def __init__(
        self,
        *,
        X: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        z: np.ndarray,
        eta_est: np.ndarray,
        observed: np.ndarray | None = None,
        sigma: float = 0.0,
        etaW_est: np.ndarray | None = None,
        with_alpha: bool,
    ) -> None:
        self.X = X
        self.edge_i = edge_i
        self.edge_j = edge_j
        self.z = z
        self.eps = eta_est if etaW_est is None else eta_est + sigma * etaW_est
        self.observed = observed
        self.N, self.S, self.T = self.eps.shape
        self.with_alpha = with_alpha
        self.K = BETA_TRUE.size + 1 + self.with_alpha

    def observed_features_batch(self, observation_ids):
        Phi, _eps = self.features_batch(observation_ids, self.observed[observation_ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):
        obs = ids % self.N
        sim = ids // self.N

        Phi = np.empty((ids.size, self.K), dtype=np.float64)
        col = 0
        if self.with_alpha:
            Phi[:, 0] = self.z[obs] * bundles.sum(axis=1)
            col = 1
        Phi[:, col : col + BETA_TRUE.size] = np.einsum(
            "nt,tk->nk",
            bundles,
            self.X,
            optimize=True,
        )
        Phi[:, -1] = (
            bundles[:, self.edge_i] * bundles[:, self.edge_j]
        ).sum(axis=1)
        eps = np.einsum(
            "nt,nt->n",
            self.eps[obs, sim],
            bundles,
            optimize=True,
        )

        if aggregate:
            return weights @ Phi, float(weights @ eps)
        return Phi, eps

    def price_batch(self, theta, agent_ids):
        obs = agent_ids % self.N
        sim = agent_ids // self.N

        if self.with_alpha:
            alpha = theta[0]
            beta = theta[1 : 1 + BETA_TRUE.size]
        else:
            alpha = 0.0
            beta = theta[: BETA_TRUE.size]
        delta = theta[-1]
        pair = np.full(self.edge_i.size, delta)
        base = np.einsum("tk,k->t", self.X, beta, optimize=True)

        bundles = np.empty((agent_ids.size, self.T), dtype=bool)
        payoffs = np.empty(agent_ids.size)
        for r, (i, s) in enumerate(zip(obs, sim)):
            linear = base + alpha * self.z[i] + self.eps[i, s]
            bundle = min_cut_choice(linear, self.edge_i, self.edge_j, pair)
            bundles[r] = bundle
            payoffs[r] = (
                np.sum(linear, dtype=np.float64, where=bundle)
                + delta
                * np.count_nonzero(bundle[self.edge_i] & bundle[self.edge_j])
            )
        return cb.DemandBatch.exact(agent_ids, bundles, payoffs)


def fit_iid(
    arrays,
    *,
    transport: cb.Transport,
    master_backend: str,
    activity_level: str,
):
    game = PeerGame(
        X=arrays["X"],
        edge_i=arrays["edge_i"],
        edge_j=arrays["edge_j"],
        z=np.zeros(1, dtype=np.float64),
        eta_est=arrays["iid_eta_est"],
        observed=arrays["iid_y"],
        with_alpha=False,
    )
    params = cb.Parameters(
        {"beta": (-5.0, 5.0, BETA_TRUE.size), "delta": (0.0, 5.0, 1)}
    )
    activity = None
    if activity_level != "off":
        activity = cb.ActivityConfig(
            label="peer-iid",
            level=activity_level,
            stdout=True,
        )
    fit = cb.estimate_distributed(
        cb.Model(
            game,
            params,
            features=game,
            observed_features=game,
            formulation=cb.NSlack,
        ),
        n_observations=game.N,
        n_simulations=game.S,
        transport=transport,
        master_backend=master_backend,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        activity=activity,
    )
    if transport.rank != 0:
        return None
    theta_true = np.r_[BETA_TRUE, [DELTA_TRUE]]
    return {
        "max_abs_error": float(np.max(np.abs(fit.theta_hat - theta_true))),
        "iterations": int(fit.metadata["iterations"]),
        "active_cuts": int(fit.n_active_cuts),
        "converged": bool(fit.metadata["converged"]),
    }


def fit_sigma_grid(
    arrays,
    *,
    sigma_grid: np.ndarray,
    transport: cb.Transport,
    master_backend: str,
    progress: bool,
):
    X = arrays["X"]
    edge_i = arrays["edge_i"]
    edge_j = arrays["edge_j"]
    corr_z = arrays["corr_z"]
    corr_y = arrays["corr_y"]
    corr_eta_est = arrays["corr_eta_est"]
    corr_etaW_est = arrays["corr_etaW_est"]
    n_profiles = corr_eta_est.shape[0]
    observed_size2 = np.mean(corr_y.sum(axis=1) ** 2)

    def rhs_sigma(row, sigma):
        i = row.agent_id % n_profiles
        s = row.agent_id // n_profiles
        eps = corr_eta_est[i, s] + sigma * corr_etaW_est[i, s]
        return np.sum(eps, where=row.bundle)

    params = cb.Parameters(
        {
            "alpha": (-5.0, 5.0, 1),
            "beta": (-5.0, 5.0, BETA_TRUE.size),
            "delta": (0.0, 5.0, 1),
        }
    )
    # Reuse the same NSlack master over the sigma grid.
    driver = cb.PersistentMasterFit(
        params,
        observables=np.arange(n_profiles, dtype=np.int64),
        observed_bundles=corr_y,
        transport=transport,
        config=cb.LoopConfig(max_iterations=MAX_ITERATIONS),
        rhs_transform=rhs_sigma,
        master_backend=master_backend,
        tolerance=TOLERANCE,
    )

    rows = []
    try:
        for k, sigma in enumerate(sigma_grid):
            game = PeerGame(
                X=X,
                edge_i=edge_i,
                edge_j=edge_j,
                z=corr_z,
                eta_est=corr_eta_est,
                sigma=sigma,
                etaW_est=corr_etaW_est,
                with_alpha=True,
            )
            fit = driver.fit if k == 0 else driver.reevaluate
            result = fit(
                sigma,
                oracle=game,
                shocks=game.eps,
            )
            if not result.converged:
                raise RuntimeError(f"sigma={sigma:.6g} did not converge")
            if transport.rank != 0:
                continue

            dual = result.dual
            dual_bundles = dual.bundle_table[dual.bundle_row_ids]
            dual_pis = dual.pis.astype(np.longdouble)
            dual_sizes = dual_bundles.sum(axis=1, dtype=np.longdouble)
            dual_size2 = (
                dual_pis @ (dual_sizes * dual_sizes) / (game.N * game.S)
            )
            row = {
                "sigma": sigma,
                "start": "cold" if k == 0 else "warm",
                "criterion": ((dual_size2 - observed_size2) / observed_size2)
                ** 2,
                "theta_hat": result.theta_hat,
                "dual_size2": dual_size2,
                "iterations": result.iterations,
                "active_cuts": result.n_active_cuts,
            }
            rows.append(row)
            if progress:
                print(
                    f"sigma={row['sigma']:.3f}",
                    row["start"],
                    f"criterion={row['criterion']:.3e}",
                    f"iters={row['iterations']}",
                    f"cuts={row['active_cuts']}",
                    flush=True,
                )
    finally:
        driver.close()

    if transport.rank != 0:
        return None

    best_id = np.argmin([row["criterion"] for row in rows])
    best = rows[best_id]
    theta_true = np.r_[[ALPHA_TRUE], BETA_TRUE, [DELTA_TRUE]]
    theta_error = best["theta_hat"] - theta_true
    return {
        "theta_true": theta_true,
        "theta_hat": best["theta_hat"],
        "theta_error": theta_error,
        "max_abs_error": np.max(np.abs(theta_error)),
        "sigma_best_id": best_id,
        "sigma_best": best,
        "sigma_rows": rows,
        "observed_size2": observed_size2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-backend", default="auto")
    parser.add_argument("--transport", default="auto")
    parser.add_argument("--activity-level", default="summary")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    transport = make_transport(args.transport)
    publish = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            publish = make_arrays(
                T=N_NODES,
                iid_s=IID_SIMULATIONS,
                correlated_n=CORRELATED_N,
                correlated_s=CORRELATED_SIMULATIONS,
            )
    arrays = dict(transport.node_shared(publish))
    iid = fit_iid(
        arrays,
        transport=transport,
        master_backend=args.master_backend,
        activity_level=args.activity_level,
    )
    sigma_grid = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_GRID_SIZE)
    correlated = fit_sigma_grid(
        arrays,
        sigma_grid=sigma_grid,
        transport=transport,
        master_backend=args.master_backend,
        progress=bool(args.progress),
    )
    if transport.rank != 0:
        return

    print(
        "peer effects:",
        f"T={arrays['X'].shape[0]}",
        f"edges={arrays['edge_i'].size}",
        f"mean_degree={float(np.mean(arrays['degree'])):.3f}",
        f"ranks={int(transport.size)}",
        f"nodes={int(transport.node.n_nodes)}",
    )
    print(
        "iid:",
        f"converged={iid['converged']}",
        f"iterations={iid['iterations']}",
        f"cuts={iid['active_cuts']}",
        f"max_abs_error={iid['max_abs_error']:.4f}",
    )
    print("selected sigma:", f"{correlated['sigma_best']['sigma']:.3f}", f"(true {SIGMA_TRUE:.3f})")
    print(f"max parameter error: {correlated['max_abs_error']:.4f}")
    print()
    print("Sigma grid")
    print("  sigma  start  criterion  dual mean(size^2)  iterations  cuts")
    best_id = int(correlated["sigma_best_id"])
    for i, row in enumerate(correlated["sigma_rows"]):
        if abs(i - best_id) <= 5:
            mark = "*" if i == best_id else " "
            print(
                f"{mark} {row['sigma']:5.2f}  {row['start']:<5}"
                f"  {row['criterion']:9.2e}"
                f"  {row['dual_size2']:18.3f}"
                f"  {row['iterations']:10d}"
                f"  {row['active_cuts']:5d}"
            )
    print(f"observed mean(size^2): {correlated['observed_size2']:.3f}")
    print()
    print("Parameter recovery")
    print("  parameter       true   estimate      error")
    names = ["alpha"] + [f"beta[{j}]" for j in range(BETA_TRUE.size)] + ["delta"]
    for name, true, estimate, error in zip(
        names,
        correlated["theta_true"],
        correlated["theta_hat"],
        correlated["theta_error"],
    ):
        print(f"  {name:<9} {true:8.4f} {estimate:10.4f} {error:10.4f}")


if __name__ == "__main__":
    main()
