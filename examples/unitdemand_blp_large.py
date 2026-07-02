"""Market-wise OneSlack unit-demand BLP example."""

import argparse
import os

import numpy as np
from linearmodels.iv import IV2SLS

import combrum as cb

N_OBS = 5_000_000
N_SIMULATIONS = 1
N_MARKETS = 50
N_INSIDE_ITEMS = 10
MAX_ITERATIONS = 5_000
TOLERANCE = 1e-3
SEED = 41
N_COVARIATES = 3
SHOCK_SCALE = 1.0
ALPHA_TRUE = 1.0
XI_LOADING = 0.5
XI_MEAN = 0.5
BETA_TRUE = np.array([0.65, -0.35, 0.2], dtype=np.float64)


def make_transport(kind):
    if kind == "mpi" or (kind == "auto" and "OMPI_COMM_WORLD_SIZE" in os.environ):
        return cb.MpiTransport()
    return cb.SerialTransport()


def price_sensitivity_alpha(delta, prices, instruments):
    delta_flat = delta.ravel()
    price_flat = prices.ravel()
    instrument_flat = instruments.ravel()
    constant = np.ones(delta_flat.size)
    ols = IV2SLS(delta_flat, np.column_stack([constant, price_flat]), None, None).fit()
    tsls = IV2SLS(delta_flat, constant, price_flat, instrument_flat).fit()
    return -float(ols.params.iloc[-1]), -float(tsls.params.iloc[-1])


class MarketDemand(cb.Oracle, cb.FeatureMap):
    """One market: dense covariates plus local item fixed effects."""

    def __init__(self, *, x, est_shocks):
        self.x = x
        self.est_shocks = est_shocks
        self.N, self.J, self.C = self.x.shape
        self.K = self.C + self.J
        self.rows = np.arange(self.N)

    def batch(self, agent_ids):
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

    def price_batch(self, theta, local_ids):
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

    def features_batch(self, ids, bundles, weights=None, aggregate=False):
        ids, x, eps, rows = self.batch(ids)
        if bundles.ndim == 1:
            chosen = bundles
        else:
            chosen = np.argmax(bundles, axis=1)
        chosen_eps = eps[rows, chosen]
        inside_rows = np.flatnonzero(chosen > 0)
        items = chosen[inside_rows] - 1

        Phi = np.zeros((ids.size, self.K), dtype=np.float64)
        Phi[inside_rows, : self.C] = x[inside_rows, items]
        Phi[inside_rows, self.C + items] = 1.0
        if aggregate:
            return weights @ Phi, float(weights @ chosen_eps)
        return Phi, chosen_eps


def make_node_arrays(transport):
    n_per_t = N_OBS // N_MARKETS
    m = N_INSIDE_ITEMS + 1
    t_values = np.arange(transport.node.node_id, N_MARKETS, transport.node.n_nodes, dtype=np.int64)
    publish = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            xs, instruments, prices, deltas, shocks, observed = [], [], [], [], [], []
            for t in t_values:
                rng = np.random.default_rng(np.random.SeedSequence([SEED, int(t)]))
                x = rng.normal(size=(n_per_t, N_INSIDE_ITEMS, N_COVARIATES))
                z = rng.normal(size=N_INSIDE_ITEMS)
                xi = XI_MEAN + rng.normal(size=N_INSIDE_ITEMS)
                p = 0.7 * z + xi + 0.3 * rng.normal(size=N_INSIDE_ITEMS)
                delta = -ALPHA_TRUE * p + XI_LOADING * xi + 0.5 * rng.normal(size=N_INSIDE_ITEMS)

                dgp_shocks = rng.normal(scale=SHOCK_SCALE, size=(n_per_t, m))
                est_shocks = rng.normal(scale=SHOCK_SCALE, size=(n_per_t, N_SIMULATIONS, m))
                dgp_utilities = dgp_shocks.copy()
                dgp_utilities[:, 1:] += np.einsum("njc,c->nj", x, BETA_TRUE, optimize=True) + delta

                xs.append(x)
                instruments.append(z)
                prices.append(p)
                deltas.append(delta)
                shocks.append(est_shocks)
                observed.append(np.eye(m, dtype=np.float64)[np.argmax(dgp_utilities, axis=1)])

            if t_values.size:
                publish = {
                    "t_values": t_values,
                    "x": np.stack(xs),
                    "instruments": np.stack(instruments),
                    "prices": np.stack(prices),
                    "delta_true": np.stack(deltas),
                    "est_shocks": np.stack(shocks),
                    "observed": np.stack(observed),
                }
            else:
                publish = {
                    "t_values": t_values,
                    "x": np.empty((0, n_per_t, N_INSIDE_ITEMS, N_COVARIATES)),
                    "instruments": np.empty((0, N_INSIDE_ITEMS)),
                    "prices": np.empty((0, N_INSIDE_ITEMS)),
                    "delta_true": np.empty((0, N_INSIDE_ITEMS)),
                    "est_shocks": np.empty((0, n_per_t, N_SIMULATIONS, m)),
                    "observed": np.empty((0, n_per_t, m)),
                }
    return dict(transport.node_shared(publish)), n_per_t


def fit_market(
    arrays,
    block_index: int,
    *,
    progress: bool = False,
):
    n_per_t = N_OBS // N_MARKETS
    t = int(arrays["t_values"][block_index])
    params = cb.Parameters(
        {
            "beta": (-3.0, 3.0, N_COVARIATES),
            "item": (-4.0, 4.0, N_INSIDE_ITEMS),
        }
    )
    oracle = MarketDemand(
        x=arrays["x"][block_index],
        est_shocks=arrays["est_shocks"][block_index],
    )
    data = cb.Data(
        observed_bundles=arrays["observed"][block_index],
        shocks=arrays["est_shocks"][block_index],
        observables=np.arange(n_per_t),
    )
    activity = (
        cb.ActivityConfig(label=f"t-{t:02d}", level="summary", stdout=True)
        if progress
        else None
    )
    # MPI shards markets. Each market fit is a serial OneSlack problem.
    fit = cb.estimate(
        cb.Model(oracle, params, formulation=cb.OneSlack),
        data,
        transport=cb.SerialTransport(),
        master_backend="highs",
        master_params={"u_lower_bound": None},
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        weights=np.full(n_per_t, 1.0 / (n_per_t * N_SIMULATIONS)),
        activity=activity,
    )
    theta = fit.theta_named()
    beta_hat = theta["beta"]
    item_hat = theta["item"]
    delta_true = arrays["delta_true"][block_index]
    return {
        "t": t,
        "rank": None,
        "runtime_seconds": float(fit.runtime_seconds),
        "iterations": int(fit.metadata["iterations"]),
        "active_cuts": int(fit.n_active_cuts),
        "converged": bool(fit.metadata["converged"]),
        "beta_hat": beta_hat.tolist(),
        "item_hat": item_hat.tolist(),
        "item_true": delta_true.tolist(),
        "prices": arrays["prices"][block_index].tolist(),
        "instruments": arrays["instruments"][block_index].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="auto")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    transport = make_transport(args.transport)
    arrays, n_per_t = make_node_arrays(transport)
    positions = np.arange(arrays["t_values"].size, dtype=np.int64)
    local_positions = positions[positions % transport.node.node_size == transport.node.node_rank]
    local_rows = [fit_market(arrays, i, progress=bool(args.progress)) for i in local_positions]
    for row in local_rows:
        row["rank"] = int(transport.rank)

    rows = []
    for source in range(transport.size):
        payload = transport.send_to_root(
            local_rows if transport.rank == source else None,
            source=source,
            root=0,
        )
        if transport.rank == 0 and payload is not None:
            rows.extend(payload)
    if transport.rank != 0:
        return

    rows = sorted(rows, key=lambda row: row["t"])
    if len(rows) != N_MARKETS:
        raise RuntimeError(f"expected {N_MARKETS} fitted markets; got {len(rows)}")
    beta_hat = np.asarray([row["beta_hat"] for row in rows], dtype=np.float64)
    item_hat = np.asarray([row["item_hat"] for row in rows], dtype=np.float64)
    item_true = np.asarray([row["item_true"] for row in rows], dtype=np.float64)
    prices = np.asarray([row["prices"] for row in rows], dtype=np.float64)
    instruments = np.asarray([row["instruments"] for row in rows], dtype=np.float64)
    ols_alpha, tsls_alpha = price_sensitivity_alpha(item_hat, prices, instruments)
    true_ols_alpha, true_tsls_alpha = price_sensitivity_alpha(item_true, prices, instruments)

    print(
        "market-wise oneslack:",
        f"N={n_per_t * N_MARKETS}",
        f"T={N_MARKETS}",
        f"N_per_t={n_per_t}",
        f"K_local={N_COVARIATES + N_INSIDE_ITEMS}",
        f"ranks={transport.size}",
        f"nodes={transport.node.n_nodes}",
    )
    print("formulation:", cb.OneSlack.__name__)
    print("all converged:", all(row["converged"] for row in rows))
    print("max iterations used:", max(row["iterations"] for row in rows))
    print("total local runtime seconds:", round(sum(row["runtime_seconds"] for row in rows), 3))
    print("beta_true:", BETA_TRUE.round(6).tolist())
    print("mean_beta_hat:", beta_hat.mean(axis=0).round(6).tolist())
    print("beta market rmse:", round(float(np.sqrt(np.mean((beta_hat - BETA_TRUE) ** 2))), 6))
    print("market-item rmse:", round(float(np.sqrt(np.mean((item_hat - item_true) ** 2))), 6))
    print("target alpha:", round(ALPHA_TRUE, 6))
    print("ols alpha (estimated delta):", round(float(ols_alpha), 6))
    print("2sls alpha (estimated delta):", round(float(tsls_alpha), 6))
    print("ols alpha (true delta):", round(float(true_ols_alpha), 6))
    print("2sls alpha (true delta):", round(float(true_tsls_alpha), 6))
    for row in rows:
        print(
            f"t={row['t']:02d}",
            f"rank={row['rank']}",
            f"iters={row['iterations']}",
            f"cuts={row['active_cuts']}",
            f"runtime={row['runtime_seconds']:.3f}s",
            f"converged={row['converged']}",
        )


if __name__ == "__main__":
    main()
