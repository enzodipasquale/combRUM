"""Network formation with one player fixed effect and node-shared MPI data."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

import combrum as cb

K_BETA = 5
BETA_TRUE_DEFAULT = np.array([0.8, 0.5, 1.0, 0.3, 0.4], dtype=np.float64)
GAMMA_TRUE_DEFAULT = 0.5
FE_MEAN_DEFAULT = -1.0
FE_SD_DEFAULT = 0.5
DESIGN_SEED_DEFAULT = 20260424


def nf_parameters(T: int, bound: float = 8.0) -> cb.Parameters:
    return cb.Parameters(
        {
            "alpha": (-bound, bound, int(T)),
            "beta": (-bound, bound, K_BETA),
            "gamma": (0.0, bound, 1),
        }
    )


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


def _build_arrays(
    T: int,
    N: int,
    S: int,
    *,
    design_seed: int,
    shock_seed: int,
    beta_true: np.ndarray = BETA_TRUE_DEFAULT,
    gamma_true: float = GAMMA_TRUE_DEFAULT,
    fe_mean: float = FE_MEAN_DEFAULT,
    fe_sd: float = FE_SD_DEFAULT,
) -> dict[str, np.ndarray]:
    T = int(T)
    N = int(N)
    S = int(S)
    if T < 2:
        raise ValueError("T must be at least 2")
    if N < 1:
        raise ValueError("N must be positive")
    if S < 1:
        raise ValueError("S must be positive")

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

    alpha = rng.normal(fe_mean, fe_sd, size=T)
    theta_true = np.r_[alpha, beta_true, gamma_true]

    shocks = np.random.default_rng(shock_seed).standard_normal((N, S, M))
    edge_rows = np.flatnonzero(np.arange(M, dtype=np.int64) < recip)
    edge_cols = recip[edge_rows]
    xbeta = np.einsum("mk,k->m", r, beta_true, optimize=True)
    base = alpha[snd] + alpha[rcv] + xbeta
    pair = np.full(edge_rows.size, gamma_true)
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


@dataclass(frozen=True)
class NetworkDesign:
    arrays: Mapping[str, np.ndarray]
    T: int
    n_simulations: int

    @property
    def M(self) -> int:
        return self.T * (self.T - 1)

    @property
    def N(self) -> int:
        return self.arrays["observed"].shape[0]

    @property
    def observed(self) -> np.ndarray:
        return self.arrays["observed"]

    @property
    def theta_true(self) -> np.ndarray:
        return self.arrays["theta_true"]


class NetworkFeatures(cb.FeatureMap):
    def __init__(self, design: NetworkDesign) -> None:
        self.design = design
        arrays = design.arrays
        self.T = int(design.T)
        self.N = int(design.N)
        self.M = self.T * (self.T - 1)
        self.K = self.T + K_BETA + 1
        self.r = arrays["r"]
        self.edge_rows = arrays["edge_rows"]
        self.edge_cols = arrays["edge_cols"]
        self.shocks = arrays["shocks"]

        snd = arrays["snd"]
        rcv = arrays["rcv"]
        cols = np.arange(self.M)
        ones = np.ones(self.M)
        sender = csr_matrix((ones, (snd, cols)), shape=(self.T, self.M))
        receiver = csr_matrix((ones, (rcv, cols)), shape=(self.T, self.M))
        self.incidence = sender + receiver

    def setup_observed(
        self, transport: cb.Transport, observation_ids: np.ndarray
    ) -> None:
        pass

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        Phi, _eps = self.features_batch(
            observation_ids,
            self.design.observed[observation_ids],
        )
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(
        self,
        ids: np.ndarray,
        bundles: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        K: int | None = None,
        aggregate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        beta_start = self.T
        obs_ids = ids % self.N
        sim_ids = ids // self.N
        reciprocal = bundles[:, self.edge_rows] * bundles[:, self.edge_cols]
        if aggregate:
            weighted_bundle = np.einsum("n,nm->m", weights, bundles, optimize=True)
            phi = np.zeros(self.K, dtype=np.float64)
            phi[: self.T] = self.incidence @ weighted_bundle
            phi[beta_start : beta_start + K_BETA] = np.einsum(
                "m,mk->k",
                weighted_bundle,
                self.r,
                optimize=True,
            )
            phi[-1] = np.einsum(
                "n,n->",
                weights,
                reciprocal.sum(axis=1),
                optimize=True,
            )
            eps = np.einsum(
                "n,nm,nm->",
                weights,
                self.shocks[obs_ids, sim_ids],
                bundles,
                optimize=True,
            )
            return phi, eps

        Phi = np.zeros((bundles.shape[0], self.K), dtype=np.float64)
        Phi[:, : self.T] = (self.incidence @ bundles.T).T
        Phi[:, beta_start : beta_start + K_BETA] = np.einsum(
            "nm,mk->nk",
            bundles,
            self.r,
            optimize=True,
        )
        Phi[:, -1] = reciprocal.sum(axis=1)
        eps = np.einsum(
            "nm,nm->n",
            self.shocks[obs_ids, sim_ids],
            bundles,
            optimize=True,
        )
        return Phi, eps


class NetworkDemandOracle(cb.Oracle):
    def __init__(self, design: NetworkDesign) -> None:
        arrays = design.arrays
        self.T = int(design.T)
        self.N = int(design.N)
        self.M = self.T * (self.T - 1)
        self.r = arrays["r"]
        self.snd = arrays["snd"]
        self.rcv = arrays["rcv"]
        self.edge_rows = arrays["edge_rows"]
        self.edge_cols = arrays["edge_cols"]
        self.shocks = arrays["shocks"]

    def _linear(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        beta_start = self.T
        alpha = theta[:beta_start]
        beta = theta[beta_start : beta_start + K_BETA]
        gamma = theta[-1]
        xbeta = np.einsum("mk,k->m", self.r, beta, optimize=True)
        return alpha[self.snd] + alpha[self.rcv] + xbeta, gamma

    def price_batch(
        self, theta: np.ndarray, local_ids: np.ndarray
    ) -> dict[int, cb.Demand]:
        base, gamma = self._linear(theta)
        pair = np.full(self.edge_rows.size, gamma)
        demands: dict[int, cb.Demand] = {}
        for agent_id in local_ids:
            obs_id = agent_id % self.N
            sim_id = agent_id // self.N
            linear = base + self.shocks[obs_id, sim_id]
            bundle = min_cut_bundle(linear, self.edge_rows, self.edge_cols, pair)
            payoff = (
                np.sum(linear, dtype=np.float64, where=bundle)
                + gamma
                * np.count_nonzero(
                    bundle[self.edge_rows] & bundle[self.edge_cols]
                )
            )
            demands[agent_id] = cb.Demand.exact(bundle, payoff)
        return demands


def make_transport(kind: str) -> cb.Transport:
    if kind == "serial":
        return cb.SerialTransport()
    if kind == "mpi":
        return cb.MpiTransport()
    if kind == "auto":
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
                pass
        return cb.SerialTransport()
    raise ValueError("transport must be one of auto, serial, mpi")


def build_design(
    T: int = 50,
    n_observations: int = 1,
    n_simulations: int = 1000,
    *,
    design_seed: int = DESIGN_SEED_DEFAULT,
    shock_seed: int = 4000,
    transport: cb.Transport | None = None,
) -> NetworkDesign:
    if transport is None:
        arrays = _build_arrays(
            T,
            n_observations,
            n_simulations,
            design_seed=int(design_seed),
            shock_seed=int(shock_seed),
        )
    else:
        publish: dict[str, np.ndarray] = {}
        with transport.collective():
            if transport.node.node_rank == 0:
                publish = _build_arrays(
                    T,
                    n_observations,
                    n_simulations,
                    design_seed=int(design_seed),
                    shock_seed=int(shock_seed),
                )
        arrays = dict(transport.node_shared(publish))
    return NetworkDesign(arrays=arrays, T=int(T), n_simulations=int(n_simulations))


def make_model(design: NetworkDesign) -> cb.Model:
    features = NetworkFeatures(design)
    return cb.Model(
        NetworkDemandOracle(design),
        nf_parameters(design.T),
        features=features,
        observed_features=features,
        formulation=cb.NSlack,
    )


def fit_network(
    design: NetworkDesign,
    *,
    transport: cb.Transport | None = None,
    master_backend: str = "auto",
    tolerance: float = 1e-3,
    max_iterations: int = 2000,
    activity_level: str = "iterations",
) -> cb.FitResult:
    transport = cb.SerialTransport() if transport is None else transport
    activity = None
    if activity_level.lower() != "off":
        activity = cb.ActivityConfig(
            label="network",
            level=activity_level,
            stdout=True,
        )
    return cb.estimate_distributed(
        make_model(design),
        n_observations=design.N,
        n_simulations=design.n_simulations,
        transport=transport,
        master_backend=master_backend,
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        cut_policy=cb.PurgeInactive(max_age=15),
        activity=activity,
    )


def summarize_fit(design: NetworkDesign, fit: cb.FitResult) -> dict[str, Any]:
    parameters = nf_parameters(design.T)
    estimate = parameters.unpack(fit.theta_hat)
    truth = parameters.unpack(design.theta_true)
    alpha_hat = np.asarray(estimate["alpha"], dtype=np.float64)
    alpha_true = np.asarray(truth["alpha"], dtype=np.float64)
    beta_hat = np.asarray(estimate["beta"], dtype=np.float64)
    beta_true = np.asarray(truth["beta"], dtype=np.float64)
    gamma_hat = float(np.asarray(estimate["gamma"], dtype=np.float64)[0])
    gamma_true = float(np.asarray(truth["gamma"], dtype=np.float64)[0])
    return {
        "T": design.T,
        "N": design.N,
        "M": design.M,
        "S": design.n_simulations,
        "K": parameters.K,
        "links_observed": int(design.observed.sum()),
        "alpha_corr": float(np.corrcoef(alpha_hat, alpha_true)[0, 1]),
        "beta_hat": beta_hat.tolist(),
        "beta_true": beta_true.tolist(),
        "gamma_hat": gamma_hat,
        "gamma_true": gamma_true,
        "converged": bool(fit.metadata["converged"]),
        "iterations": int(fit.metadata["iterations"]),
        "active_cuts": int(fit.n_active_cuts),
        "objective": float(fit.objective),
        "runtime_seconds": float(fit.runtime_seconds),
    }


def run_example(
    args: argparse.Namespace | None = None,
    *,
    transport: cb.Transport | None = None,
) -> dict[str, Any]:
    if args is None:
        args = _parser().parse_args([])
    if transport is None:
        transport = make_transport(getattr(args, "transport", "serial"))
    design = build_design(
        T=int(args.T),
        n_observations=int(args.N),
        n_simulations=int(args.S),
        design_seed=int(args.design_seed),
        shock_seed=int(args.shock_seed),
        transport=transport,
    )
    fit = fit_network(
        design,
        transport=transport,
        master_backend=args.master_backend,
        tolerance=float(args.tolerance),
        max_iterations=int(args.max_iterations),
        activity_level=str(args.activity_level),
    )
    return {"design": design, "fit": fit, "summary": summarize_fit(design, fit)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--S", type=int, default=1000)
    p.add_argument("--design-seed", type=int, default=DESIGN_SEED_DEFAULT)
    p.add_argument("--shock-seed", type=int, default=4000)
    p.add_argument(
        "--master-backend", choices=("auto", "gurobi", "highs"), default="auto"
    )
    p.add_argument("--transport", choices=("auto", "serial", "mpi"), default="auto")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--max-iterations", type=int, default=2000)
    p.add_argument(
        "--activity-level",
        choices=("off", "summary", "iterations", "diagnostic"),
        default="iterations",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    transport = make_transport(args.transport)
    try:
        result = run_example(args, transport=transport)
        if transport.rank != 0:
            return
        summary: dict[str, Any] = result["summary"]
        print(
            "network:",
            f"T={summary['T']}",
            f"N={summary['N']}",
            f"M={summary['M']}",
            f"S={summary['S']}",
            f"K={summary['K']}",
            f"observed_links={summary['links_observed']}",
        )
        print(
            "beta_hat:",
            np.asarray(summary["beta_hat"], dtype=np.float64).round(6).tolist(),
        )
        print("gamma_hat:", round(float(summary["gamma_hat"]), 6))
        print("gamma_true:", round(float(summary["gamma_true"]), 6))
        print("alpha_corr:", round(float(summary["alpha_corr"]), 6))
        print("objective:", round(float(summary["objective"]), 6))
        print("runtime seconds:", round(float(summary["runtime_seconds"]), 3))
        print("converged:", bool(summary["converged"]))
        print("iterations:", int(summary["iterations"]))
        print("active cuts:", int(summary["active_cuts"]))
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
