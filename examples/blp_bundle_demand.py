"""BLP-style bundle-demand example with quadratic knapsack demand."""

import argparse
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from linearmodels.iv import IV2SLS

import combrum as cb

try:
    import gurobipy as gp
except Exception:
    gp = None

try:
    import highspy as hp
except Exception:
    hp = None

GUROBI_AVAILABLE = False
if gp is not None:
    try:
        _env = gp.Env(empty=True)
        _env.setParam("OutputFlag", 0)
        _env.start()
        _env.dispose()
        GUROBI_AVAILABLE = True
    except Exception:
        pass

K_BETA = 2
K_QUAD = 2
N_MARKETS = 15
N_ITEMS = 30
N_SIMULATIONS = 1
N_PER_MARKET = None
CAPACITY = 10
SEED = 0
TOLERANCE = 1e-3
MAX_ITERATIONS = 80
ALPHA_TRUE = 1.0
XI_MEAN = 0.5
NU_SCALE = 2.5
BETA_TRUE = np.array([0.6, -0.3], dtype=np.float64)
LAMBDA_TRUE = np.array([0.6, 0.3], dtype=np.float64)


def make_data(T, M, n_per_market, S, capacity, seed, nu_scale):
    n_per_market = 1000 // T if n_per_market is None else n_per_market
    N = T * n_per_market
    rng = np.random.default_rng(seed)

    market_idx = np.repeat(np.arange(T), n_per_market).astype(np.int64)
    X = rng.normal(size=(N, M, K_BETA))
    weights = np.ones(M, dtype=np.float64)
    capacities = np.full(N, float(capacity), dtype=np.float64)

    Qs = np.zeros((K_QUAD, M, M), dtype=np.float64)
    for k in range(K_QUAD):
        upper = np.triu((rng.random((M, M)) < 0.10).astype(np.float64), 1)
        Qs[k] = upper + upper.T

    xi = XI_MEAN + rng.normal(size=(T, M))
    instruments = rng.normal(size=(T, M))
    prices = 0.7 * instruments + xi + 0.3 * rng.normal(size=(T, M))
    delta_true = -ALPHA_TRUE * prices + xi + 0.5 * rng.normal(size=(T, M))

    return dict(
        X=X,
        market_idx=market_idx,
        weights=weights,
        capacities=capacities,
        Qs=Qs,
        instruments=instruments,
        prices=prices,
        delta_true=delta_true,
        dgp_shocks=nu_scale * rng.normal(size=(N, M)),
        est_shocks=nu_scale * rng.normal(size=(N, S, M)),
    )


def _default_oracle_workers():
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return max(1, min(4, (os.cpu_count() or 1) // world_size))


# Feature map: item characteristics, market-item fixed effects, and pairwise terms.
class BLPFeatures(cb.FeatureMap):
    def __init__(self, arrays):
        self.X = arrays["X"]
        self.market_idx = arrays["market_idx"]
        self.Qs = arrays["Qs"]
        self.shocks = arrays["est_shocks"]
        self.observed = arrays["observed"]
        self.N, self.M = self.X.shape[:2]
        self.T = arrays["prices"].shape[0]

    def observed_features_batch(self, observation_ids):
        Phi, _eps = self.features_batch(observation_ids, self.observed[observation_ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(self, ids, bundles, *, weights=None, aggregate=False):
        obs_ids = ids % self.N
        sim_ids = ids // self.N
        q_start = K_BETA + self.T * self.M
        feature_dim = q_start + K_QUAD
        quad_cols = slice(q_start, q_start + K_QUAD)
        delta_cols = self.market_idx[obs_ids, None] * self.M + np.arange(self.M)

        Phi = np.zeros((ids.size, feature_dim), dtype=np.float64)
        Phi[:, :K_BETA] = np.einsum("nm,nmk->nk", bundles, self.X[obs_ids], optimize=True)

        rows = np.arange(ids.size)[:, None]
        Phi[rows, K_BETA + delta_cols] = bundles

        Phi[:, quad_cols] = np.einsum("ni,kij,nj->nk", bundles, self.Qs, bundles, optimize=True)
        eps = np.einsum("nm,nm->n", self.shocks[obs_ids, sim_ids], bundles, optimize=True)
        if aggregate:
            return weights @ Phi, float(weights @ eps)
        return Phi, eps


class GurobiDemand:
    def __init__(self, weights):
        self._weights = weights
        self._env = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 0)
        self._env.start()
        self._model = gp.Model(env=self._env)
        self._model.Params.OutputFlag = 0
        self._x = self._model.addMVar(self._weights.size, vtype=gp.GRB.BINARY)
        self._capacity = self._model.addConstr(self._weights @ self._x <= 0.0)
        self._model.update()

    def apply_solver_settings(self, settings):
        if settings.time_limit_seconds is not None:
            self._model.Params.TimeLimit = settings.time_limit_seconds
        if settings.mip_focus is not None:
            self._model.Params.MIPFocus = settings.mip_focus

    def solve(self, linear, quadratic, capacity):
        self._capacity.RHS = capacity
        sense = gp.GRB.MAXIMIZE
        self._model.setMObjective(Q=quadratic, c=linear, constant=0.0, sense=sense)
        self._model.optimize()
        if self._model.Status != gp.GRB.OPTIMAL and self._model.SolCount == 0:
            raise RuntimeError("Gurobi found no feasible bundle")
        bundle = self._x.X > 0.5
        value = self._model.ObjVal
        raw_gap = self._model.MIPGap
        exact = self._model.Status == gp.GRB.OPTIMAL and math.isfinite(raw_gap)
        if exact and raw_gap <= 0.0:
            return cb.Demand.exact(bundle, value)
        return cb.Demand.uncertified(bundle, value, gap=raw_gap)

    def close(self):
        self._model.dispose()
        self._env.dispose()


class HighsDemand:
    def __init__(self, weights, Qs):
        self._weights = weights
        upper_nonzero = np.triu(np.any(Qs != 0.0, axis=0), 1)
        rows, cols = np.nonzero(upper_nonzero)
        self._edges = np.column_stack([rows, cols])
        self._edge_rows = self._edges[:, 0]
        self._edge_cols = self._edges[:, 1]
        self._costs = np.empty(self._weights.size + self._edges.shape[0])
        self._cost_indices = np.arange(self._costs.size, dtype=np.int32)
        self._model = self._build_model()

    def _build_model(self):
        M = self._weights.size
        n_edges = self._edges.shape[0]
        n_vars = M + n_edges
        item_cols = np.arange(M, dtype=np.int32)
        binary = np.full(M, hp.HighsVarType.kInteger.value, dtype=np.uint8)
        upper_link = np.array([1.0, -1.0], dtype=np.float64)
        product_link = np.array([1.0, 1.0, -1.0], dtype=np.float64)

        model = hp.Highs()
        model.setOptionValue("output_flag", False)
        model.setOptionValue("threads", 1)
        model.setOptionValue("parallel", "off")
        model.setOptionValue("presolve", "off")
        model.setOptionValue("mip_rel_gap", 0.0)
        model.addVars(n_vars, np.zeros(n_vars), np.ones(n_vars))
        model.changeColsIntegrality(M, item_cols, binary)
        model.addRow(-hp.kHighsInf, 0.0, M, item_cols, self._weights)

        for edge_id, (j, k) in enumerate(self._edges):
            y = M + edge_id
            yj = np.array([y, j], dtype=np.int32)
            yk = np.array([y, k], dtype=np.int32)
            jky = np.array([j, k, y], dtype=np.int32)
            model.addRow(-hp.kHighsInf, 0.0, 2, yj, upper_link)
            model.addRow(-hp.kHighsInf, 0.0, 2, yk, upper_link)
            model.addRow(-hp.kHighsInf, 1.0, 3, jky, product_link)
        model.setMaximize()
        return model

    def apply_solver_settings(self, settings):
        if settings.time_limit_seconds is not None:
            self._model.setOptionValue("time_limit", settings.time_limit_seconds)

    def solve(self, linear, quadratic, capacity):
        self._costs[: self._weights.size] = linear
        self._costs[self._weights.size :] = 2.0 * quadratic[self._edge_rows, self._edge_cols]
        self._model.changeRowBounds(0, -hp.kHighsInf, capacity)
        self._model.changeColsCost(self._costs.size, self._cost_indices, self._costs)
        self._model.run()
        model_status = self._model.getModelStatus()
        solution = self._model.getSolution()
        bundle = np.asarray(solution.col_value[: self._weights.size]) > 0.5
        value = self._model.getObjectiveValue()
        raw_gap = self._model.getInfo().mip_gap
        exact = model_status == hp.HighsModelStatus.kOptimal and math.isfinite(raw_gap)
        if exact and raw_gap <= 0.0:
            return cb.Demand.exact(bundle, value)
        return cb.Demand.uncertified(bundle, value, gap=raw_gap)

    def close(self):
        self._model.clear()


# Demand oracle: solve each simulated agent's quadratic knapsack.
class BLPDemandOracle(cb.Oracle):
    def __init__(self, arrays, parameters, shocks, backend="auto", oracle_workers=None):
        self.X = arrays["X"]
        self.market_idx = arrays["market_idx"]
        self.weights = arrays["weights"]
        self.capacities = arrays["capacities"]
        self.Qs = arrays["Qs"]
        self.parameters = parameters
        self.shocks = shocks
        self.N, self.M = self.X.shape[:2]
        self.T = arrays["prices"].shape[0]
        self.backend_name = backend
        self.settings = cb.SolverSettings()
        self._oracle_workers = (
            _default_oracle_workers()
            if oracle_workers is None
            else int(oracle_workers)
        )
        self._pool = ThreadPoolExecutor(max_workers=self._oracle_workers)
        self._thread_solver = threading.local()
        self._solvers = []
        self._solver_lock = threading.Lock()

    def _solve(self, linear, quadratic, capacity):
        solver = getattr(self._thread_solver, "solver", None)
        if solver is None:
            if self.backend_name == "gurobi":
                solver = GurobiDemand(self.weights)
            else:
                solver = HighsDemand(self.weights, self.Qs)
            solver.apply_solver_settings(self.settings)
            self._thread_solver.solver = solver
            with self._solver_lock:
                self._solvers.append(solver)
        return solver.solve(linear, quadratic, capacity)

    def apply_solver_settings(self, settings):
        if settings == self.settings:
            return
        self.settings = settings
        with self._solver_lock:
            solvers = tuple(self._solvers)
        for solver in solvers:
            solver.apply_solver_settings(settings)

    def price_batch(self, theta, local_ids):
        values = self.parameters.unpack(theta)
        beta = values["beta"]
        delta = values["delta"].reshape(self.T, self.M)
        quadratic = np.tensordot(values["lambda"], self.Qs, axes=([0], [0]))

        obs_ids = local_ids % self.N
        sim_ids = local_ids // self.N
        unique_obs, obs_pos = np.unique(obs_ids, return_inverse=True)
        base_linear = np.einsum("umk,k->um", self.X[unique_obs], beta, optimize=True)
        base_linear += delta[self.market_idx[unique_obs]]
        linears = base_linear[obs_pos] + self.shocks[obs_ids, sim_ids]
        quadratics = [quadratic] * local_ids.size
        demands = self._pool.map(self._solve, linears, quadratics, self.capacities[obs_ids])
        return dict(zip(local_ids, demands))

    def teardown(self):
        self._pool.shutdown(wait=True)
        with self._solver_lock:
            solvers = tuple(self._solvers)
            self._solvers.clear()
        for solver in solvers:
            solver.close()


def simulate_observed(arrays, parameters, theta_true, backend, oracle_workers):
    N, M = arrays["X"].shape[:2]
    shocks = arrays["dgp_shocks"].reshape(N, 1, M)
    oracle = BLPDemandOracle(arrays, parameters, shocks, backend, oracle_workers)
    ids = np.arange(N, dtype=np.int64)
    try:
        demands = oracle.price_batch(theta_true, ids)
        return np.vstack([demands[i].bundle for i in ids])
    finally:
        oracle.teardown()


def price_sensitivity_alpha(delta_tj, prices, instruments):
    delta = np.asarray(delta_tj, dtype=np.float64).ravel()
    price = np.asarray(prices, dtype=np.float64).ravel()
    instrument = np.asarray(instruments, dtype=np.float64).ravel()
    constant = np.ones(delta.size, dtype=np.float64)
    ols = IV2SLS(delta, np.column_stack([constant, price]), None, None).fit()
    tsls = IV2SLS(delta, constant, price, instrument).fit()
    return -float(ols.params.iloc[-1]), -float(tsls.params.iloc[-1])


def make_transport(mode):
    if mode == "mpi":
        return cb.MpiTransport()
    if mode == "auto" and "OMPI_COMM_WORLD_SIZE" in os.environ:
        return cb.MpiTransport()
    return cb.SerialTransport()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--master-backend", default="auto")
    parser.add_argument("--transport", default="auto")
    parser.add_argument("--oracle-workers", type=int)
    parser.add_argument("--activity-level", default="iterations")
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "gurobi" if GUROBI_AVAILABLE else "highs"
    parameters = cb.Parameters(
        {
            "beta": (-10.0, 10.0, K_BETA),
            "delta": (-10.0, 10.0, N_MARKETS * N_ITEMS),
            "lambda": (0.0, 5.0, K_QUAD),
        }
    )
    transport = make_transport(args.transport)

    publish = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            publish = make_data(
                N_MARKETS,
                N_ITEMS,
                N_PER_MARKET,
                N_SIMULATIONS,
                CAPACITY,
                SEED,
                NU_SCALE,
            )
            theta_true = parameters.pack(
                {
                    "beta": BETA_TRUE,
                    "delta": publish["delta_true"].ravel(),
                    "lambda": LAMBDA_TRUE,
                }
            )
            publish["observed"] = simulate_observed(
                publish, parameters, theta_true, backend, args.oracle_workers
            )

    arrays = dict(transport.node_shared(publish))
    N, M = arrays["X"].shape[:2]
    T = arrays["prices"].shape[0]
    S = arrays["est_shocks"].shape[1]

    features = BLPFeatures(arrays)
    demand = BLPDemandOracle(arrays, parameters, arrays["est_shocks"], backend, args.oracle_workers)
    model = cb.Model(
        demand,
        parameters,
        features=features,
        observed_features=features,
        formulation=cb.NSlack,
    )

    activity = None
    if args.activity_level.lower() != "off":
        activity = cb.ActivityConfig(label="blp", level=args.activity_level, stdout=True)

    fit = cb.estimate_distributed(
        model,
        n_observations=N,
        n_simulations=S,
        transport=transport,
        master_backend=args.master_backend,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        cut_policy=cb.AddAll(),
        iteration_callback=cb.point_timeout_callback(
            cb.Schedule((cb.Phase(timeout=1.0, iters=30), cb.Phase(timeout=20.0)))
        ),
        activity=activity,
    )
    if transport.rank != 0:
        return

    theta_hat = fit.theta_named()
    beta_hat = np.round(theta_hat["beta"], 6).tolist()
    lambda_hat = np.round(theta_hat["lambda"], 6).tolist()
    delta_hat = np.asarray(theta_hat["delta"], dtype=np.float64).reshape(T, M)
    prices = arrays["prices"]
    instruments = arrays["instruments"]
    ols_alpha, tsls_alpha = price_sensitivity_alpha(delta_hat, prices, instruments)
    true_ols_alpha, true_tsls_alpha = price_sensitivity_alpha(arrays["delta_true"], prices, instruments)
    delta_correlation = np.corrcoef(delta_hat.ravel(), arrays["delta_true"].ravel())[0, 1]
    iterations = fit.metadata["iterations"]
    converged = fit.metadata["converged"]

    print(f"blp: N={N} T={T} M={M} S={S} K={parameters.K} backend={backend}")
    print(f"mean bundle size: {arrays['observed'].sum(axis=1).mean():.2f}")
    print(f"iterations: {iterations} converged: {converged} wall: {fit.runtime_seconds:.1f}s")
    print(f"objective: {fit.objective:.6f}")
    print(f"beta_hat: {beta_hat}")
    print(f"lambda_hat: {lambda_hat}")
    print(f"delta correlation: {delta_correlation:.6f}")
    print(f"target alpha: {ALPHA_TRUE:.6f}")
    print(f"ols alpha (estimated delta): {ols_alpha:.6f}")
    print(f"2sls alpha (estimated delta): {tsls_alpha:.6f}")
    print(f"ols alpha (true delta): {true_ols_alpha:.6f}")
    print(f"2sls alpha (true delta): {true_tsls_alpha:.6f}")


if __name__ == "__main__":
    main()
