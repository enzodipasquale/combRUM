"""Peer effects on a large network with MPI row generation.

Launch with MPI to distribute pricing over the simulated profiles:

    mpiexec -n 4 python examples/peer_effects_large_network.py --transport mpi

Each node leader builds the graph, covariates, observed choices, and simulation
draws once, then publishes them with ``Transport.node_shared``. combRUM shards
the ``N*S`` simulated profiles across ranks. The sigma grid uses
``cb.PersistentMasterFit``: the first point is cold, and later points rewrite
cut RHS values and warm-solve the same NSlack master.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

import combrum as cb

AVG_DEGREE = 10.0
LINK_PROB = 0.75
GRAPH_SEED = 20260423
IID_SEED = 49892
CORRELATED_SEED = 2718

BETA_TRUE = np.array([-1.0, -0.5, -1.0, 0.5], dtype=np.float64)
DELTA_TRUE = 0.20
ALPHA_TRUE = 0.50
SIGMA_TRUE = 0.50


def make_transport(kind: str) -> cb.Transport:
    if kind == "serial":
        return cb.SerialTransport()
    if kind == "mpi":
        return cb.MpiTransport()
    mpi_env = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MV2_COMM_WORLD_SIZE",
    )
    if any(name in os.environ for name in mpi_env):
        try:
            return cb.MpiTransport()
        except Exception:
            return cb.SerialTransport()
    return cb.SerialTransport()


def min_cut_choice(
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    pair: np.ndarray,
) -> np.ndarray:
    linear = np.asarray(linear, dtype=np.float64)
    pair = np.maximum(np.asarray(pair, dtype=np.float64), 0.0)
    n = linear.size

    row_sum = np.zeros(n, dtype=np.float64)
    np.add.at(row_sum, edge_i, pair)
    net = -linear - row_sum

    magnitude = max(
        float(np.abs(net).max(initial=0.0)),
        float(pair.max(initial=0.0)),
    )
    scale = 1.0 if magnitude == 0.0 else 10.0 ** (
        8 - int(np.floor(np.log10(magnitude)))
    )
    net_i = np.round(net * scale).astype(np.int64)
    pair_i = np.round(pair * scale).astype(np.int64)

    source, sink = 0, 1
    nodes = np.arange(n, dtype=np.int64) + 2
    choose = net_i < 0
    linked = pair_i > 0

    rows = np.concatenate(
        [
            np.full(int(choose.sum()), source, dtype=np.int64),
            nodes[~choose],
            edge_i[linked] + 2,
        ]
    )
    cols = np.concatenate(
        [
            nodes[choose],
            np.full(int((~choose).sum()), sink, dtype=np.int64),
            edge_j[linked] + 2,
        ]
    )
    vals = np.concatenate([-net_i[choose], net_i[~choose], pair_i[linked]])

    cap = csr_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))
    residual = (cap - maximum_flow(cap, source, sink).flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()

    _, pred = breadth_first_order(
        residual,
        i_start=source,
        directed=True,
        return_predecessors=True,
    )
    selected = pred != -9999
    selected[source] = True
    return selected[2:]


def make_arrays(
    *,
    T: int,
    iid_s: int,
    correlated_n: int,
    correlated_s: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(GRAPH_SEED)
    side = np.sqrt(float(T))
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
    ).astype(np.int8)
    np.fill_diagonal(W, 0)

    edge_i, edge_j = np.nonzero(np.triu(W, k=1))
    edge_i = edge_i.astype(np.int64)
    edge_j = edge_j.astype(np.int64)
    pair = np.full(edge_i.size, DELTA_TRUE, dtype=np.float64)

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
    ).astype(np.float64)
    xbeta = np.einsum("tk,k->t", X, BETA_TRUE, optimize=True)

    rng_iid = np.random.default_rng(IID_SEED)
    iid_eta_dgp = rng_iid.standard_normal((1, T))
    iid_eta_est = rng_iid.standard_normal((1, iid_s, T))
    iid_y = np.asarray(
        [min_cut_choice(xbeta + iid_eta_dgp[0], edge_i, edge_j, pair)],
        dtype=bool,
    )

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


def build_arrays(
    *,
    T: int,
    iid_s: int,
    correlated_n: int,
    correlated_s: int,
    transport: cb.Transport,
) -> dict[str, np.ndarray]:
    publish: dict[str, np.ndarray] = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            publish = make_arrays(
                T=T,
                iid_s=iid_s,
                correlated_n=correlated_n,
                correlated_s=correlated_s,
            )
    return dict(transport.node_shared(publish))


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
        self.X = np.asarray(X, dtype=np.float64)
        self.edge_i = np.asarray(edge_i, dtype=np.int64)
        self.edge_j = np.asarray(edge_j, dtype=np.int64)
        self.z = np.asarray(z, dtype=np.float64)
        eta_est = np.asarray(eta_est, dtype=np.float64)
        self.eps = (
            eta_est
            if etaW_est is None
            else eta_est + float(sigma) * np.asarray(etaW_est, dtype=np.float64)
        )
        self.observed = (
            None if observed is None else np.asarray(observed, dtype=bool)
        )
        self.N, self.S, self.T = self.eps.shape
        self.with_alpha = bool(with_alpha)
        self.K = BETA_TRUE.size + 1 + int(self.with_alpha)

    def setup_observed(
        self, transport: cb.Transport, observation_ids: np.ndarray
    ) -> None:
        self.observation_ids = np.asarray(observation_ids, dtype=np.int64)

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        if self.observed is None:
            raise ValueError("observed choices are required for distributed fits")
        ids = np.asarray(observation_ids, dtype=np.int64)
        Phi, _eps = self.features_batch(ids, self.observed[ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(
        self,
        ids: np.ndarray,
        bundles: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        aggregate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        ids = np.asarray(ids, dtype=np.int64)
        bundles = np.asarray(bundles, dtype=np.float64)
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
            if weights is None:
                raise ValueError("weights are required when aggregate=True")
            w = np.asarray(weights, dtype=np.float64)
            return (
                np.einsum("n,nk->k", w, Phi, optimize=True),
                float(np.einsum("n,n->", w, eps, optimize=True)),
            )
        return Phi, eps

    def price_batch(
        self,
        theta: np.ndarray,
        local_ids: np.ndarray,
    ) -> cb.DemandBatch:
        theta = np.asarray(theta, dtype=np.float64)
        ids = np.asarray(local_ids, dtype=np.int64)
        obs = ids % self.N
        sim = ids // self.N

        if self.with_alpha:
            alpha = float(theta[0])
            beta = theta[1 : 1 + BETA_TRUE.size]
        else:
            alpha = 0.0
            beta = theta[: BETA_TRUE.size]
        delta = float(theta[-1])
        pair = np.full(self.edge_i.size, delta, dtype=np.float64)
        base = np.einsum("tk,k->t", self.X, beta, optimize=True)

        bundles = np.empty((ids.size, self.T), dtype=bool)
        payoffs = np.empty(ids.size, dtype=np.float64)
        for r, (i, s) in enumerate(zip(obs, sim)):
            linear = base + alpha * self.z[i] + self.eps[i, s]
            bundle = min_cut_choice(linear, self.edge_i, self.edge_j, pair)
            bundles[r] = bundle
            payoffs[r] = float(
                np.sum(linear, dtype=np.float64, where=bundle)
                + delta
                * np.count_nonzero(bundle[self.edge_i] & bundle[self.edge_j])
            )
        return cb.DemandBatch.exact(ids, bundles, payoffs)


def fit_iid(
    arrays: dict[str, np.ndarray],
    *,
    transport: cb.Transport,
    master_backend: str,
    tolerance: float,
    max_iterations: int,
    activity_level: str,
) -> tuple[cb.FitResult, dict[str, Any] | None]:
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
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        activity=activity,
    )
    if transport.rank != 0:
        return fit, None
    theta_true = np.r_[BETA_TRUE, [DELTA_TRUE]]
    return fit, {
        "theta_true": theta_true,
        "theta_hat": fit.theta_hat,
        "max_abs_error": float(np.max(np.abs(fit.theta_hat - theta_true))),
        "iterations": int(fit.metadata["iterations"]),
        "active_cuts": int(fit.n_active_cuts),
        "converged": bool(fit.metadata["converged"]),
    }


def fit_sigma_grid(
    arrays: dict[str, np.ndarray],
    *,
    sigma_grid: np.ndarray,
    transport: cb.Transport,
    master_backend: str,
    tolerance: float,
    max_iterations: int,
    progress: bool,
) -> dict[str, Any] | None:
    X = arrays["X"]
    edge_i = arrays["edge_i"]
    edge_j = arrays["edge_j"]
    corr_z = arrays["corr_z"]
    corr_y = arrays["corr_y"]
    corr_eta_est = arrays["corr_eta_est"]
    corr_etaW_est = arrays["corr_etaW_est"]
    n_profiles, n_simulations, _ = corr_eta_est.shape
    observed_size2 = float(np.mean(corr_y.sum(axis=1, dtype=np.float64) ** 2))

    def rhs_sigma(row, sigma):
        i = row.agent_id % n_profiles
        s = row.agent_id // n_profiles
        eps = corr_eta_est[i, s] + float(sigma) * corr_etaW_est[i, s]
        return float(np.sum(eps, dtype=np.float64, where=row.bundle))

    params = cb.Parameters(
        {
            "alpha": (-5.0, 5.0, 1),
            "beta": (-5.0, 5.0, BETA_TRUE.size),
            "delta": (0.0, 5.0, 1),
        }
    )
    # Persistent by design: the distributed NSlack master is reused across sigma.
    driver = cb.PersistentMasterFit(
        params,
        observables=np.arange(n_profiles, dtype=np.int64),
        observed_bundles=corr_y,
        transport=transport,
        config=cb.LoopConfig(max_iterations=int(max_iterations)),
        rhs_transform=rhs_sigma,
        master_backend=master_backend,
        tolerance=float(tolerance),
    )

    rows: list[dict[str, Any]] = []
    try:
        for k, sigma in enumerate(np.asarray(sigma_grid, dtype=np.float64)):
            game = PeerGame(
                X=X,
                edge_i=edge_i,
                edge_j=edge_j,
                z=corr_z,
                eta_est=corr_eta_est,
                sigma=float(sigma),
                etaW_est=corr_etaW_est,
                with_alpha=True,
            )
            fit = driver.fit if k == 0 else driver.reevaluate
            result = fit(
                float(sigma),
                oracle=game,
                shocks=game.eps,
            )
            if not result.converged:
                raise RuntimeError(f"sigma={float(sigma):.6g} did not converge")
            if transport.rank != 0:
                continue

            dual = result.dual
            dual_bundles = dual.bundle_table[dual.bundle_row_ids]
            dual_pis = dual.pis.astype(np.longdouble)
            dual_sizes = dual_bundles.sum(axis=1, dtype=np.longdouble)
            dual_size2 = float(
                dual_pis @ (dual_sizes * dual_sizes) / (game.N * game.S)
            )
            row = {
                "sigma": float(sigma),
                "start": "cold" if k == 0 else "warm",
                "criterion": ((dual_size2 - observed_size2) / observed_size2)
                ** 2,
                "theta_hat": result.theta_hat,
                "dual_size2": dual_size2,
                "iterations": int(result.iterations),
                "active_cuts": int(result.n_active_cuts),
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

    best_id = int(np.argmin([row["criterion"] for row in rows]))
    best = rows[best_id]
    theta_true = np.r_[[ALPHA_TRUE], BETA_TRUE, [DELTA_TRUE]]
    theta_error = best["theta_hat"] - theta_true
    return {
        "theta_true": theta_true,
        "theta_hat": best["theta_hat"],
        "theta_error": theta_error,
        "max_abs_error": float(np.max(np.abs(theta_error))),
        "sigma_true": SIGMA_TRUE,
        "sigma_best_id": best_id,
        "sigma_best": best,
        "sigma_rows": rows,
        "observed_size2": observed_size2,
    }


def run_example(
    *,
    T: int = 500,
    iid_s: int = 50,
    correlated_n: int = 20,
    correlated_s: int = 20,
    sigma_grid_size: int = 50,
    sigma_min: float = 0.2,
    sigma_max: float = 0.8,
    master_backend: str = "auto",
    tolerance: float = 1e-3,
    max_iterations: int = 1000,
    activity_level: str = "summary",
    progress: bool = False,
    transport: cb.Transport | None = None,
) -> dict[str, Any]:
    if T < 2:
        raise ValueError("T must be at least 2")
    if iid_s < 1:
        raise ValueError("iid_s must be positive")
    if correlated_n < 2:
        raise ValueError("correlated_n must be at least 2")
    if correlated_s < 1:
        raise ValueError("correlated_s must be positive")
    if sigma_grid_size < 2:
        raise ValueError("sigma_grid_size must be at least 2")

    transport = cb.SerialTransport() if transport is None else transport
    arrays = build_arrays(
        T=int(T),
        iid_s=int(iid_s),
        correlated_n=int(correlated_n),
        correlated_s=int(correlated_s),
        transport=transport,
    )
    _, iid = fit_iid(
        arrays,
        transport=transport,
        master_backend=master_backend,
        tolerance=tolerance,
        max_iterations=max_iterations,
        activity_level=activity_level,
    )
    sigma_grid = np.linspace(
        float(sigma_min),
        float(sigma_max),
        int(sigma_grid_size),
    )
    correlated = fit_sigma_grid(
        arrays,
        sigma_grid=sigma_grid,
        transport=transport,
        master_backend=master_backend,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress=progress,
    )
    return {
        "arrays": arrays,
        "iid": iid,
        "correlated": correlated,
        "ranks": int(transport.size),
        "nodes": int(transport.node.n_nodes),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--iid-S", type=int, default=50)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--S", type=int, default=20)
    parser.add_argument("--sigma-grid-size", type=int, default=50)
    parser.add_argument("--sigma-min", type=float, default=0.2)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument(
        "--master-backend",
        choices=("auto", "gurobi", "highs"),
        default="auto",
    )
    parser.add_argument("--transport", choices=("auto", "serial", "mpi"), default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--activity-level",
        choices=("off", "summary", "iterations", "diagnostic"),
        default="summary",
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def _print_summary(result: dict[str, Any]) -> None:
    arrays = result["arrays"]
    iid = result["iid"]
    correlated = result["correlated"]
    assert iid is not None
    assert correlated is not None

    print(
        "peer effects:",
        f"T={arrays['X'].shape[0]}",
        f"edges={arrays['edge_i'].size}",
        f"mean_degree={float(np.mean(arrays['degree'])):.3f}",
        f"ranks={result['ranks']}",
        f"nodes={result['nodes']}",
    )
    print(
        "iid:",
        f"converged={iid['converged']}",
        f"iterations={iid['iterations']}",
        f"cuts={iid['active_cuts']}",
        f"max_abs_error={iid['max_abs_error']:.4f}",
    )
    print(
        "selected sigma:",
        f"{correlated['sigma_best']['sigma']:.3f}",
        f"(true {correlated['sigma_true']:.3f})",
    )
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


def main() -> None:
    args = _parser().parse_args()
    transport = make_transport(args.transport)
    try:
        result = run_example(
            T=args.T,
            iid_s=args.iid_S,
            correlated_n=args.N,
            correlated_s=args.S,
            sigma_grid_size=args.sigma_grid_size,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            master_backend=args.master_backend,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            activity_level=args.activity_level,
            progress=bool(args.progress),
            transport=transport,
        )
        if transport.rank == 0:
            _print_summary(result)
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
