"""Market-wise OneSlack demand with MPI rank-level parallelism.

Each node leader builds only the markets assigned to that node and publishes
them once with ``cb.Transport.node_shared``. Ranks on the node then solve
disjoint markets locally.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from linearmodels.iv import IV2SLS

import combrum as cb

DEFAULT_N_OBS = 5_000_000
DEFAULT_N_SIMULATIONS = 1
DEFAULT_N_MARKETS = 50
DEFAULT_N_INSIDE_ITEMS = 10
DEFAULT_MAX_ITERATIONS = 5_000
DEFAULT_TOLERANCE = 1e-3
N_COVARIATES = 3
SHOCK_SCALE = 1.0
ALPHA_TRUE = 1.0
XI_LOADING = 0.5
XI_MEAN = 0.5


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


def price_sensitivity_alpha(
    delta: np.ndarray, prices: np.ndarray, instruments: np.ndarray
) -> tuple[float, float]:
    delta_flat = delta.ravel()
    price_flat = prices.ravel()
    instrument_flat = instruments.ravel()
    constant = np.ones(delta_flat.size)
    ols = IV2SLS(
        delta_flat,
        np.column_stack([constant, price_flat]),
        None,
        None,
    ).fit()
    tsls = IV2SLS(delta_flat, constant, price_flat, instrument_flat).fit()
    return -float(ols.params.iloc[-1]), -float(tsls.params.iloc[-1])


class MarketDemand(cb.Oracle, cb.FeatureMap):
    """One market: dense covariates plus local item fixed effects."""

    def __init__(self, *, x: np.ndarray, est_shocks: np.ndarray) -> None:
        self.x = x
        self.est_shocks = est_shocks
        self.N, self.J, self.C = self.x.shape
        self.K = self.C + self.J
        self.rows = np.arange(self.N)

    def batch(
        self, agent_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.est_shocks.shape[1] == 1
            and len(agent_ids) == self.N
            and np.array_equal(agent_ids, self.rows)
        ):
            return agent_ids, self.x, self.est_shocks[:, 0], self.rows
        obs = agent_ids % self.N
        sim = agent_ids // self.N
        return (
            agent_ids,
            self.x[obs],
            self.est_shocks[obs, sim],
            np.arange(len(agent_ids)),
        )

    def price_batch(self, theta: np.ndarray, local_ids: np.ndarray) -> cb.DemandBatch:
        ids, x, eps, rows = self.batch(local_ids)
        inside_values = (
            np.einsum("ijk,k->ij", x, theta[: self.C], optimize=True)
            + theta[self.C :]
            + eps[:, 1:]
        )
        outside_values = eps[:, 0]
        best_inside = np.argmax(inside_values, axis=1)
        best_inside_values = inside_values[rows, best_inside]
        choose_inside = best_inside_values > outside_values
        best = np.where(choose_inside, best_inside + 1, 0)
        payoffs = np.where(choose_inside, best_inside_values, outside_values)
        return cb.DemandBatch.exact(ids, best, payoffs)

    def features_batch(
        self,
        ids: np.ndarray,
        bundles: np.ndarray,
        weights: np.ndarray | None = None,
        aggregate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        ids, x, eps, rows = self.batch(ids)
        if bundles.ndim == 1:
            chosen = bundles
        else:
            chosen = np.argmax(bundles, axis=1)
        chosen_eps = eps[rows, chosen]
        inside_rows = np.flatnonzero(chosen > 0)
        items = chosen[inside_rows] - 1

        if aggregate:
            phi = np.zeros(self.K, dtype=np.float64)
            w_inside = weights[inside_rows]
            phi[: self.C] = w_inside @ x[inside_rows, items]
            phi[self.C :] = np.bincount(
                items,
                weights=w_inside,
                minlength=self.J,
            )
            return phi, np.dot(weights, chosen_eps)

        Phi = np.zeros((ids.size, self.K), dtype=np.float64)
        Phi[inside_rows, : self.C] = x[inside_rows, items]
        Phi[inside_rows, self.C + items] = 1.0
        return Phi, chosen_eps


@dataclass(frozen=True)
class MarketBlock:
    arrays: Mapping[str, np.ndarray]
    n_obs: int
    n_simulations: int
    n_markets: int
    n_inside_items: int
    n_per_t: int

    @property
    def t_values(self) -> np.ndarray:
        return self.arrays["t_values"]


def market_parameters(n_inside_items: int, bound: float = 4.0) -> cb.Parameters:
    return cb.Parameters(
        {
            "beta": (-3.0, 3.0, N_COVARIATES),
            "item": (-bound, bound, n_inside_items),
        }
    )


def _node_t_values(n_markets: int, node) -> np.ndarray:
    return np.arange(node.node_id, n_markets, node.n_nodes, dtype=np.int64)


def _market_seed(seed: int, t: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), int(t)])


def _empty_arrays(
    *, n_per_t: int, n_simulations: int, n_inside_items: int
) -> dict[str, np.ndarray]:
    m = n_inside_items + 1
    return {
        "t_values": np.empty(0, dtype=np.int64),
        "x": np.empty((0, n_per_t, n_inside_items, N_COVARIATES), dtype=np.float64),
        "instruments": np.empty((0, n_inside_items), dtype=np.float64),
        "prices": np.empty((0, n_inside_items), dtype=np.float64),
        "delta_true": np.empty((0, n_inside_items), dtype=np.float64),
        "est_shocks": np.empty((0, n_per_t, n_simulations, m), dtype=np.float64),
        "observed": np.empty((0, n_per_t, m), dtype=np.float64),
        "beta_true": np.array([0.65, -0.35, 0.2], dtype=np.float64),
    }


def _one_market_arrays(
    *,
    t: int,
    n_per_t: int,
    n_simulations: int,
    n_inside_items: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_market_seed(seed, t))
    m = n_inside_items + 1
    beta_true = np.array([0.65, -0.35, 0.2], dtype=np.float64)

    x = rng.normal(size=(n_per_t, n_inside_items, N_COVARIATES))
    instruments = rng.normal(size=n_inside_items)
    xi = XI_MEAN + rng.normal(size=n_inside_items)
    price_noise = rng.normal(size=n_inside_items)
    delta_noise = rng.normal(size=n_inside_items)
    prices = 0.7 * instruments + xi + 0.3 * price_noise
    delta_true = -ALPHA_TRUE * prices + XI_LOADING * xi + 0.5 * delta_noise

    dgp_shocks = rng.normal(scale=SHOCK_SCALE, size=(n_per_t, m))
    est_shocks = rng.normal(scale=SHOCK_SCALE, size=(n_per_t, n_simulations, m))
    dgp_utilities = dgp_shocks.copy()
    dgp_utilities[:, 1:] += (
        np.einsum("njc,c->nj", x, beta_true, optimize=True)
        + delta_true
    )
    choice_labels = np.argmax(dgp_utilities, axis=1)
    observed = np.eye(m, dtype=np.float64)[choice_labels]

    return {
        "x": x,
        "instruments": instruments,
        "prices": prices,
        "delta_true": delta_true,
        "est_shocks": est_shocks,
        "observed": observed,
        "beta_true": beta_true,
    }


def _build_node_arrays(
    *,
    t_values: np.ndarray,
    n_per_t: int,
    n_simulations: int,
    n_inside_items: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if t_values.size == 0:
        return _empty_arrays(
            n_per_t=n_per_t,
            n_simulations=n_simulations,
            n_inside_items=n_inside_items,
        )

    markets = [
        _one_market_arrays(
            t=int(t),
            n_per_t=n_per_t,
            n_simulations=n_simulations,
            n_inside_items=n_inside_items,
            seed=seed,
        )
        for t in t_values
    ]
    return {
        "t_values": np.asarray(t_values, dtype=np.int64),
        "x": np.stack([m["x"] for m in markets]),
        "instruments": np.stack([m["instruments"] for m in markets]),
        "prices": np.stack([m["prices"] for m in markets]),
        "delta_true": np.stack([m["delta_true"] for m in markets]),
        "est_shocks": np.stack([m["est_shocks"] for m in markets]),
        "observed": np.stack([m["observed"] for m in markets]),
        "beta_true": np.asarray(markets[0]["beta_true"], dtype=np.float64),
    }


def build_node_block(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    n_markets: int = DEFAULT_N_MARKETS,
    n_inside_items: int = DEFAULT_N_INSIDE_ITEMS,
    seed: int = 41,
    transport: cb.Transport | None = None,
) -> MarketBlock:
    transport = cb.SerialTransport() if transport is None else transport
    n_obs = int(n_obs)
    n_simulations = int(n_simulations)
    n_markets = int(n_markets)
    n_inside_items = int(n_inside_items)
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if n_markets <= 0:
        raise ValueError("n_markets must be positive")
    if n_inside_items < 2:
        raise ValueError("n_inside_items must be at least 2 for IV recovery")
    n_per_t = n_obs // n_markets
    if n_per_t <= 0:
        raise ValueError("n_obs must be at least n_markets")
    n_obs = n_per_t * n_markets
    t_values = _node_t_values(n_markets, transport.node)
    publish: dict[str, np.ndarray] = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            publish = _build_node_arrays(
                t_values=t_values,
                n_per_t=n_per_t,
                n_simulations=n_simulations,
                n_inside_items=n_inside_items,
                seed=int(seed),
            )
    arrays = dict(transport.node_shared(publish))
    return MarketBlock(
        arrays=arrays,
        n_obs=n_obs,
        n_simulations=n_simulations,
        n_markets=n_markets,
        n_inside_items=n_inside_items,
        n_per_t=n_per_t,
    )


def _local_block_indices(block: MarketBlock, transport: cb.Transport) -> np.ndarray:
    positions = np.arange(block.t_values.size, dtype=np.int64)
    return positions[positions % transport.node.node_size == transport.node.node_rank]


def fit_t(
    block: MarketBlock,
    block_index: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    progress: bool = False,
) -> dict[str, Any]:
    arrays = block.arrays
    t = int(arrays["t_values"][block_index])
    params = market_parameters(block.n_inside_items)
    oracle = MarketDemand(
        x=arrays["x"][block_index],
        est_shocks=arrays["est_shocks"][block_index],
    )
    data = cb.Data(
        observed_bundles=arrays["observed"][block_index],
        shocks=arrays["est_shocks"][block_index],
        observables=np.arange(block.n_per_t),
    )
    activity = (
        cb.ActivityConfig(label=f"t-{t:02d}", level="summary", stdout=True)
        if progress
        else None
    )
    # Serial by design: MPI shards markets above; each market fit is OneSlack.
    fit = cb.estimate(
        cb.Model(oracle, params, formulation=cb.OneSlack),
        data,
        transport=cb.SerialTransport(),
        master_backend="highs",
        master_params={"u_lower_bound": None},
        max_iterations=max_iterations,
        tolerance=tolerance,
        weights=np.full(
            block.n_per_t,
            1.0 / (block.n_per_t * block.n_simulations),
        ),
        activity=activity,
    )
    theta = fit.theta_named()
    beta_hat = theta["beta"]
    item_hat = theta["item"]
    beta_true = arrays["beta_true"]
    delta_true = arrays["delta_true"][block_index]
    item_error = item_hat - delta_true
    return {
        "t": t,
        "rank": None,
        "runtime_seconds": float(fit.runtime_seconds),
        "iterations": int(fit.metadata["iterations"]),
        "active_cuts": int(fit.n_active_cuts),
        "converged": bool(fit.metadata["converged"]),
        "beta_hat": beta_hat.tolist(),
        "beta_max_abs_error": float(np.max(np.abs(beta_hat - beta_true))),
        "item_hat": item_hat.tolist(),
        "item_true": delta_true.tolist(),
        "item_rmse": float(np.sqrt(np.mean(item_error**2))),
        "prices": arrays["prices"][block_index].tolist(),
        "instruments": arrays["instruments"][block_index].tolist(),
    }


def _gather_rows(
    local_rows: list[dict[str, Any]], transport: cb.Transport
) -> list[dict[str, Any]] | None:
    for row in local_rows:
        row["rank"] = int(transport.rank)
    gathered: list[dict[str, Any]] = []
    for source in range(transport.size):
        payload = transport.send_to_root(
            local_rows if transport.rank == source else None,
            source=source,
            root=0,
        )
        if transport.rank == 0 and payload is not None:
            gathered.extend(payload)
    if transport.rank != 0:
        return None
    return sorted(gathered, key=lambda row: row["t"])


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    n_obs: int,
    n_simulations: int,
    n_markets: int,
    n_inside_items: int,
    n_per_t: int,
    ranks: int,
    nodes: int,
) -> dict[str, Any]:
    if len(rows) != n_markets:
        raise ValueError(f"expected {n_markets} fitted markets; got {len(rows)}")
    beta_hat = np.asarray([row["beta_hat"] for row in rows], dtype=np.float64)
    item_hat = np.asarray([row["item_hat"] for row in rows], dtype=np.float64)
    item_true = np.asarray([row["item_true"] for row in rows], dtype=np.float64)
    prices = np.asarray([row["prices"] for row in rows], dtype=np.float64)
    instruments = np.asarray([row["instruments"] for row in rows], dtype=np.float64)
    beta_true = np.array([0.65, -0.35, 0.2], dtype=np.float64)
    ols_alpha, tsls_alpha = price_sensitivity_alpha(
        item_hat, prices, instruments
    )
    true_ols_alpha, true_tsls_alpha = price_sensitivity_alpha(
        item_true, prices, instruments
    )
    public_rows = [
        {
            "t": int(row["t"]),
            "rank": int(row["rank"]),
            "runtime_seconds": float(row["runtime_seconds"]),
            "iterations": int(row["iterations"]),
            "active_cuts": int(row["active_cuts"]),
            "converged": bool(row["converged"]),
            "beta_max_abs_error": float(row["beta_max_abs_error"]),
            "item_rmse": float(row["item_rmse"]),
        }
        for row in rows
    ]
    return {
        "n_obs": int(n_obs),
        "n_simulations": int(n_simulations),
        "n_markets": int(n_markets),
        "n_per_t": int(n_per_t),
        "n_inside_items": int(n_inside_items),
        "n_alternatives": int(n_inside_items) + 1,
        "n_covariates": N_COVARIATES,
        "n_local_parameters": int(N_COVARIATES + n_inside_items),
        "ranks": int(ranks),
        "nodes": int(nodes),
        "formulation": cb.OneSlack.__name__,
        "all_converged": all(row["converged"] for row in public_rows),
        "max_iterations_used": max(row["iterations"] for row in public_rows),
        "total_runtime_seconds": float(
            sum(row["runtime_seconds"] for row in public_rows)
        ),
        "mean_beta_hat": beta_hat.mean(axis=0).tolist(),
        "beta_true": beta_true.tolist(),
        "beta_market_rmse": float(np.sqrt(np.mean((beta_hat - beta_true) ** 2))),
        "market_item_rmse": float(np.sqrt(np.mean((item_hat - item_true) ** 2))),
        "ols_alpha": ols_alpha,
        "tsls_alpha": tsls_alpha,
        "ols_alpha_true_delta": true_ols_alpha,
        "tsls_alpha_true_delta": true_tsls_alpha,
        "markets": public_rows,
    }


def run_example(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    n_markets: int = DEFAULT_N_MARKETS,
    n_inside_items: int = DEFAULT_N_INSIDE_ITEMS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 41,
    progress: bool = False,
    transport: cb.Transport | None = None,
) -> dict[str, Any]:
    transport = cb.SerialTransport() if transport is None else transport
    block = build_node_block(
        n_obs=n_obs,
        n_simulations=n_simulations,
        n_markets=n_markets,
        n_inside_items=n_inside_items,
        seed=seed,
        transport=transport,
    )
    local_rows = [
        fit_t(
            block,
            i,
            max_iterations=max_iterations,
            tolerance=tolerance,
            progress=progress,
        )
        for i in _local_block_indices(block, transport)
    ]
    rows = _gather_rows(local_rows, transport)
    summary = None
    if transport.rank == 0 and rows is not None:
        summary = summarize_rows(
            rows,
            n_obs=block.n_obs,
            n_simulations=n_simulations,
            n_markets=n_markets,
            n_inside_items=n_inside_items,
            n_per_t=block.n_per_t,
            ranks=transport.size,
            nodes=transport.node.n_nodes,
        )
    return {"block": block, "local_rows": local_rows, "summary": summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--n-simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument("--n-markets", type=int, default=DEFAULT_N_MARKETS)
    parser.add_argument("--n-inside-items", type=int, default=DEFAULT_N_INSIDE_ITEMS)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--transport", choices=("auto", "serial", "mpi"), default="auto"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one start/final OneSlack line for each local t.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    transport = make_transport(args.transport)
    try:
        result = run_example(
            n_obs=args.n_obs,
            n_simulations=args.n_simulations,
            n_markets=args.n_markets,
            n_inside_items=args.n_inside_items,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            seed=args.seed,
            progress=bool(args.progress),
            transport=transport,
        )
        if transport.rank != 0:
            return
        summary = result["summary"]
        assert summary is not None
        print(
            "market-wise oneslack:",
            f"N={summary['n_obs']}",
            f"T={summary['n_markets']}",
            f"N_per_t={summary['n_per_t']}",
            f"K_local={summary['n_local_parameters']}",
            f"ranks={summary['ranks']}",
            f"nodes={summary['nodes']}",
        )
        print("formulation:", summary["formulation"])
        print("all converged:", summary["all_converged"])
        print("max iterations used:", summary["max_iterations_used"])
        print(
            "total local runtime seconds:",
            round(summary["total_runtime_seconds"], 3),
        )
        print("beta_true:", np.asarray(summary["beta_true"]).round(6).tolist())
        print("mean_beta_hat:", np.asarray(summary["mean_beta_hat"]).round(6).tolist())
        print("beta market rmse:", round(float(summary["beta_market_rmse"]), 6))
        print("market-item rmse:", round(float(summary["market_item_rmse"]), 6))
        print("target alpha:", round(ALPHA_TRUE, 6))
        print(
            "ols alpha (estimated delta):",
            round(float(summary["ols_alpha"]), 6),
        )
        print(
            "2sls alpha (estimated delta):",
            round(float(summary["tsls_alpha"]), 6),
        )
        print(
            "ols alpha (true delta):",
            round(float(summary["ols_alpha_true_delta"]), 6),
        )
        print(
            "2sls alpha (true delta):",
            round(float(summary["tsls_alpha_true_delta"]), 6),
        )
        for row in summary["markets"]:
            print(
                f"t={row['t']:02d}",
                f"rank={row['rank']}",
                f"iters={row['iterations']}",
                f"cuts={row['active_cuts']}",
                f"runtime={row['runtime_seconds']:.3f}s",
                f"converged={row['converged']}",
            )
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
