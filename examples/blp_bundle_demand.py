"""Standalone BLP-style bundle-demand estimation example.

The example has two application-specific combRUM surfaces:

* ``BLPDemandOracle`` prices quadratic-knapsack demand with a batched oracle
  and keeps one persistent MIP model per local agent id.
* ``BLPFeatures`` maps bundles to structural features and supports weighted
  aggregates plus the distributed observed-feature hook.

The demand oracle uses Gurobi's quadratic MIP objective when a licensed
installation is available. Otherwise it uses HiGHS with an exact binary-product
linearization of the quadratic knapsack.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from queue import Empty, Queue
from typing import Any

import numpy as np
from linearmodels.iv import IV2SLS

import combrum as cb

K_BETA = 2
K_QUAD = 2
ALPHA_TRUE = 1.0
XI_MEAN = 0.5
NU_SCALE = 2.5
BETA_TRUE = np.array([0.6, -0.3], dtype=np.float64)
LAMBDA_TRUE = np.array([0.6, 0.3], dtype=np.float64)

__all__ = [
    "ALPHA_TRUE",
    "BETA_TRUE",
    "BLPDemandOracle",
    "BLPDesign",
    "BLPFeatures",
    "GurobiDemand",
    "HighsDemand",
    "K_BETA",
    "K_QUAD",
    "LAMBDA_TRUE",
    "XI_MEAN",
    "build_design",
    "build_distributed_model",
    "choose_backend",
    "estimate_blp",
    "make_parameters",
    "simulate_observed",
    "summarize_fit",
]


@dataclass(frozen=True)
class BLPDesign:
    X: np.ndarray
    market_idx: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    Qs: np.ndarray
    xi: np.ndarray
    instruments: np.ndarray
    prices: np.ndarray
    delta_true: np.ndarray
    beta_true: np.ndarray
    lambda_true: np.ndarray
    theta_true: np.ndarray
    dgp_shocks: np.ndarray
    est_shocks: np.ndarray
    observed: np.ndarray
    parameters: cb.Parameters

    @property
    def N(self) -> int:
        return int(self.X.shape[0])

    @property
    def T(self) -> int:
        return int(self.prices.shape[0])

    @property
    def M(self) -> int:
        return int(self.X.shape[1])

    @property
    def S(self) -> int:
        return int(self.est_shocks.shape[1])

    @property
    def K(self) -> int:
        return int(self.parameters.K)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "X": self.X,
            "market_idx": self.market_idx,
            "weights": self.weights,
            "capacities": self.capacities,
            "Qs": self.Qs,
            "xi": self.xi,
            "instruments": self.instruments,
            "prices": self.prices,
            "delta_true": self.delta_true,
            "beta_true": self.beta_true,
            "lambda_true": self.lambda_true,
            "theta_true": self.theta_true,
            "dgp_shocks": self.dgp_shocks,
            "est_shocks": self.est_shocks,
            "observed": self.observed,
        }

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], parameters: cb.Parameters
    ) -> "BLPDesign":
        return cls(
            X=np.asarray(arrays["X"], dtype=np.float64),
            market_idx=np.asarray(arrays["market_idx"], dtype=np.int64),
            weights=np.asarray(arrays["weights"], dtype=np.float64),
            capacities=np.asarray(arrays["capacities"], dtype=np.float64),
            Qs=np.asarray(arrays["Qs"], dtype=np.float64),
            xi=np.asarray(arrays["xi"], dtype=np.float64),
            instruments=np.asarray(arrays["instruments"], dtype=np.float64),
            prices=np.asarray(arrays["prices"], dtype=np.float64),
            delta_true=np.asarray(arrays["delta_true"], dtype=np.float64),
            beta_true=np.asarray(arrays["beta_true"], dtype=np.float64),
            lambda_true=np.asarray(arrays["lambda_true"], dtype=np.float64),
            theta_true=np.asarray(arrays["theta_true"], dtype=np.float64),
            dgp_shocks=np.asarray(arrays["dgp_shocks"], dtype=np.float64),
            est_shocks=np.asarray(arrays["est_shocks"], dtype=np.float64),
            observed=np.asarray(arrays["observed"], dtype=bool),
            parameters=parameters,
        )


def make_parameters(T: int, M: int) -> cb.Parameters:
    return cb.Parameters(
        {
            "beta": (-10.0, 10.0, K_BETA),
            "delta": (-10.0, 10.0, int(T) * int(M)),
            "lambda": (0.0, 5.0, K_QUAD),
        }
    )


def build_design(
    *,
    T: int = 15,
    M: int = 30,
    n_per_market: int | None = None,
    n_simulations: int = 1,
    capacity: int = 10,
    seed: int = 0,
    nu_scale: float = NU_SCALE,
) -> BLPDesign:
    T = int(T)
    M = int(M)
    if n_per_market is None:
        n_per_market = 1000 // T
    n_per_market = int(n_per_market)
    N = T * n_per_market
    rng = np.random.default_rng(seed)
    parameters = make_parameters(T, M)

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
    theta_true = parameters.pack(
        {
            "beta": BETA_TRUE,
            "delta": delta_true.ravel(),
            "lambda": LAMBDA_TRUE,
        }
    )

    return BLPDesign(
        X=X,
        market_idx=market_idx,
        weights=weights,
        capacities=capacities,
        Qs=Qs,
        xi=xi,
        instruments=instruments,
        prices=prices,
        delta_true=delta_true,
        beta_true=BETA_TRUE,
        lambda_true=LAMBDA_TRUE,
        theta_true=theta_true,
        dgp_shocks=nu_scale * rng.normal(size=(N, M)),
        est_shocks=nu_scale * rng.normal(size=(N, int(n_simulations), M)),
        observed=np.zeros((N, M), dtype=bool),
        parameters=parameters,
    )


def _merge_settings(
    current: cb.SolverSettings, update: cb.SolverSettings
) -> cb.SolverSettings:
    return cb.SolverSettings(
        time_limit_seconds=(
            current.time_limit_seconds
            if update.time_limit_seconds is None
            else update.time_limit_seconds
        ),
        mip_focus=current.mip_focus if update.mip_focus is None else update.mip_focus,
    )


def _default_oracle_workers(backend: str) -> int:
    if backend != "highs":
        return 1
    world_size = 1
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        value = os.environ.get(name)
        if value is not None:
            try:
                world_size = max(1, int(value))
            except ValueError:
                pass
            break
    return max(1, min(4, (os.cpu_count() or 1) // world_size))


def _gurobi_available() -> bool:
    env = None
    try:
        import gurobipy

        env = gurobipy.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except Exception:
        return False
    finally:
        if env is not None:
            try:
                env.dispose()
            except Exception:
                pass
    return True


def _highs_available() -> bool:
    try:
        import highspy

        highspy.Highs()
    except Exception:
        return False
    return True


def choose_backend(requested: str = "auto") -> str:
    if requested not in {"auto", "gurobi", "highs"}:
        raise ValueError(
            f"backend must be one of auto, gurobi, highs; got {requested!r}"
        )
    if requested == "gurobi":
        if not _gurobi_available():
            raise RuntimeError(
                "gurobi was requested but no licensed Gurobi is available"
            )
        return "gurobi"
    if requested == "highs":
        if not _highs_available():
            raise RuntimeError("highs was requested but highspy is not installed")
        return "highs"
    if _gurobi_available():
        return "gurobi"
    if _highs_available():
        return "highs"
    raise RuntimeError("install gurobipy or highspy to price BLP demand")


class BLPFeatures(cb.FeatureMap):
    """Batched structural features for the BLP demand example."""

    def __init__(self, design: BLPDesign, shocks: np.ndarray) -> None:
        self.design = design
        self.shocks = np.asarray(shocks, dtype=np.float64)

    def setup_observed(
        self, transport: cb.Transport, observation_ids: np.ndarray
    ) -> None:
        self.observation_ids = np.asarray(observation_ids, dtype=np.int64)

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(observation_ids, dtype=np.int64)
        Phi, _eps = self.features_batch(ids, self.design.observed[ids])
        return np.ascontiguousarray(Phi, dtype=np.float64)

    def features_batch(
        self,
        ids: np.ndarray,
        bundles: np.ndarray,
        weights: np.ndarray | None = None,
        aggregate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        ids = np.asarray(ids, dtype=np.int64)
        bundles = np.asarray(bundles, dtype=np.float64)
        obs_ids = ids % self.design.N
        sim_ids = ids // self.design.N
        q_start = K_BETA + self.design.T * self.design.M
        feature_dim = q_start + K_QUAD
        delta_cols = (
            self.design.market_idx[obs_ids, None] * self.design.M
            + np.arange(self.design.M)
        )

        if aggregate:
            if weights is None:
                raise ValueError("weights are required when aggregate=True")
            w = np.asarray(weights, dtype=np.float64)
            phi = np.zeros(feature_dim, dtype=np.float64)
            phi[:K_BETA] = np.einsum(
                "n,nm,nmk->k",
                w,
                bundles,
                self.design.X[obs_ids],
                optimize=True,
            )
            phi[K_BETA:q_start] = np.bincount(
                delta_cols.ravel(),
                weights=(w[:, None] * bundles).ravel(),
                minlength=self.design.T * self.design.M,
            )
            phi[q_start : q_start + K_QUAD] = np.einsum(
                "n,ni,kij,nj->k",
                w,
                bundles,
                self.design.Qs,
                bundles,
                optimize=True,
            )
            eps = np.einsum(
                "n,nm,nm->",
                w,
                self.shocks[obs_ids, sim_ids],
                bundles,
                optimize=True,
            )
            return phi, float(eps)

        Phi = np.zeros((ids.size, feature_dim), dtype=np.float64)
        Phi[:, :K_BETA] = np.einsum(
            "nm,nmk->nk",
            bundles,
            self.design.X[obs_ids],
            optimize=True,
        )

        rows = np.arange(ids.size)[:, None]
        Phi[rows, K_BETA + delta_cols] = bundles

        Phi[:, q_start : q_start + K_QUAD] = np.einsum(
            "ni,kij,nj->nk",
            bundles,
            self.design.Qs,
            bundles,
            optimize=True,
        )
        eps = np.einsum(
            "nm,nm->n",
            self.shocks[obs_ids, sim_ids],
            bundles,
            optimize=True,
        )
        return Phi, eps


class GurobiDemand:
    """Quadratic-knapsack solver using Gurobi's native quadratic objective."""

    def __init__(self, weights: np.ndarray, *, quiet: bool = True) -> None:
        import gurobipy as gp

        self._gp = gp
        self._weights = np.asarray(weights, dtype=np.float64)
        self._env = gp.Env(empty=True)
        if quiet:
            self._env.setParam("OutputFlag", 0)
        self._env.start()
        self._model = gp.Model(env=self._env)
        if quiet:
            self._model.Params.OutputFlag = 0
        self._x = self._model.addMVar(self._weights.size, vtype=gp.GRB.BINARY)
        self._capacity = self._model.addConstr(self._weights @ self._x <= 0.0)
        self._model.update()
        self._settings = cb.SolverSettings()

    def apply_solver_settings(self, settings: cb.SolverSettings) -> None:
        self._settings = _merge_settings(self._settings, settings)
        self._push_settings()

    def _push_settings(self) -> None:
        if self._settings.time_limit_seconds is not None:
            self._model.Params.TimeLimit = self._settings.time_limit_seconds
        if self._settings.mip_focus is not None:
            self._model.Params.MIPFocus = self._settings.mip_focus

    def solve(
        self, linear: np.ndarray, quadratic: np.ndarray, capacity: float
    ) -> cb.Demand:
        self._capacity.RHS = float(capacity)
        self._model.setMObjective(
            Q=quadratic,
            c=linear,
            constant=0.0,
            sense=self._gp.GRB.MAXIMIZE,
        )
        self._push_settings()
        self._model.optimize()
        if self._model.Status != self._gp.GRB.OPTIMAL and self._model.SolCount == 0:
            raise RuntimeError("Gurobi found no feasible bundle")
        bundle = np.asarray(self._x.X, dtype=np.float64) > 0.5
        value = float(self._model.ObjVal)
        raw_gap = float(self._model.MIPGap)
        if (
            self._model.Status == self._gp.GRB.OPTIMAL
            and math.isfinite(raw_gap)
            and raw_gap <= 0.0
        ):
            return cb.Demand.exact(bundle, value)
        return cb.Demand.uncertified(bundle, value, gap=raw_gap)

    def close(self) -> None:
        self._model.dispose()
        self._env.dispose()


class HighsDemand:
    """Exact HiGHS MILP linearization for binary quadratic knapsack."""

    def __init__(
        self,
        weights: np.ndarray,
        Qs: np.ndarray,
        *,
        quiet: bool = True,
    ) -> None:
        import highspy

        self._highspy = highspy
        self._weights = np.asarray(weights, dtype=np.float64)
        upper_nonzero = np.triu(np.any(np.asarray(Qs) != 0.0, axis=0), 1)
        rows, cols = np.nonzero(upper_nonzero)
        self._edges = np.column_stack([rows, cols]).astype(np.int64)
        self._edge_rows = self._edges[:, 0]
        self._edge_cols = self._edges[:, 1]
        self._costs = np.empty(self._weights.size + self._edges.shape[0])
        self._cost_indices = np.arange(self._costs.size, dtype=np.int32)
        self._model = self._build_model(quiet=quiet)
        self._settings = cb.SolverSettings()

    def _build_model(self, *, quiet: bool) -> Any:
        hp = self._highspy
        M = self._weights.size
        n_edges = int(self._edges.shape[0])
        n_vars = M + n_edges
        model = hp.Highs()
        if quiet:
            model.setOptionValue("output_flag", False)
        model.setOptionValue("threads", 1)
        model.setOptionValue("parallel", "off")
        model.setOptionValue("presolve", "off")
        model.setOptionValue("mip_rel_gap", 0.0)
        model.addVars(n_vars, np.zeros(n_vars), np.ones(n_vars))
        model.changeColsIntegrality(
            M,
            np.arange(M, dtype=np.int32),
            np.full(M, hp.HighsVarType.kInteger.value, dtype=np.uint8),
        )
        model.addRow(
            -hp.kHighsInf,
            0.0,
            M,
            np.arange(M, dtype=np.int32),
            self._weights,
        )
        for edge_id, (j, k) in enumerate(self._edges):
            y = M + edge_id
            model.addRow(
                -hp.kHighsInf,
                0.0,
                2,
                np.array([y, j], dtype=np.int32),
                np.array([1.0, -1.0], dtype=np.float64),
            )
            model.addRow(
                -hp.kHighsInf,
                0.0,
                2,
                np.array([y, k], dtype=np.int32),
                np.array([1.0, -1.0], dtype=np.float64),
            )
            model.addRow(
                -hp.kHighsInf,
                1.0,
                3,
                np.array([j, k, y], dtype=np.int32),
                np.array([1.0, 1.0, -1.0], dtype=np.float64),
            )
        model.setMaximize()
        return model

    def apply_solver_settings(self, settings: cb.SolverSettings) -> None:
        self._settings = _merge_settings(self._settings, settings)
        self._push_settings()

    def _push_settings(self) -> None:
        if self._settings.time_limit_seconds is not None:
            self._model.setOptionValue(
                "time_limit", float(self._settings.time_limit_seconds)
            )

    def solve(
        self, linear: np.ndarray, quadratic: np.ndarray, capacity: float
    ) -> cb.Demand:
        self._costs[: self._weights.size] = linear
        self._costs[self._weights.size :] = (
            2.0 * quadratic[self._edge_rows, self._edge_cols]
        )
        hp = self._highspy
        self._model.changeRowBounds(0, -hp.kHighsInf, float(capacity))
        self._model.changeColsCost(
            self._costs.size,
            self._cost_indices,
            self._costs,
        )
        self._push_settings()
        run_status = self._model.run()
        model_status = self._model.getModelStatus()
        solution = self._model.getSolution()
        admissible = {
            hp.HighsModelStatus.kOptimal,
            hp.HighsModelStatus.kTimeLimit,
            hp.HighsModelStatus.kSolutionLimit,
        }
        admissible_run = {hp.HighsStatus.kOk, hp.HighsStatus.kWarning}
        if (
            run_status not in admissible_run
            or model_status not in admissible
            or not bool(getattr(solution, "value_valid", False))
        ):
            raise RuntimeError(
                "HiGHS found no feasible bundle:"
                f" run_status={run_status}, model_status={model_status}"
            )
        bundle = np.asarray(solution.col_value[: self._weights.size]) > 0.5
        value = float(self._model.getObjectiveValue())
        raw_gap = float(self._model.getInfo().mip_gap)
        if (
            model_status == hp.HighsModelStatus.kOptimal
            and math.isfinite(raw_gap)
            and raw_gap <= 0.0
        ):
            return cb.Demand.exact(bundle, value)
        return cb.Demand.uncertified(bundle, value, gap=raw_gap)

    def close(self) -> None:
        self._model.clear()


class BLPDemandOracle(cb.Oracle):
    """Batched demand oracle for market-item quadratic knapsacks."""

    def __init__(
        self,
        design: BLPDesign,
        shocks: np.ndarray,
        *,
        backend: str = "auto",
        quiet: bool = True,
        oracle_workers: int | None = None,
    ) -> None:
        self.design = design
        self.shocks = np.asarray(shocks, dtype=np.float64)
        self.backend_name = choose_backend(backend)
        self._quiet = bool(quiet)
        self.settings = cb.SolverSettings()
        workers = (
            _default_oracle_workers(self.backend_name)
            if oracle_workers is None
            else int(oracle_workers)
        )
        if workers < 1:
            raise ValueError("oracle_workers must be at least 1")
        self._oracle_workers = workers if self.backend_name == "highs" else 1
        self._solver_lock = threading.Lock()
        self._solver_queue: Queue[GurobiDemand | HighsDemand] = Queue()
        self._solvers: list[GurobiDemand | HighsDemand] = []

    def _new_solver(self) -> GurobiDemand | HighsDemand:
        if self.backend_name == "gurobi":
            solver = GurobiDemand(self.design.weights, quiet=self._quiet)
        else:
            solver = HighsDemand(
                self.design.weights, self.design.Qs, quiet=self._quiet
            )
        solver.apply_solver_settings(self.settings)
        return solver

    def _acquire_solver(self) -> GurobiDemand | HighsDemand:
        try:
            return self._solver_queue.get_nowait()
        except Empty:
            pass
        with self._solver_lock:
            if len(self._solvers) < self._oracle_workers:
                solver = self._new_solver()
                self._solvers.append(solver)
                return solver
        return self._solver_queue.get()

    def _release_solver(self, solver: GurobiDemand | HighsDemand) -> None:
        self._solver_queue.put(solver)

    def _solve(
        self, linear: np.ndarray, quadratic: np.ndarray, capacity: float
    ) -> cb.Demand:
        solver = self._acquire_solver()
        try:
            return solver.solve(linear, quadratic, capacity)
        finally:
            self._release_solver(solver)

    def apply_solver_settings(self, settings: cb.SolverSettings) -> None:
        updated = _merge_settings(self.settings, settings)
        if updated == self.settings:
            return
        self.settings = updated
        with self._solver_lock:
            solvers = tuple(self._solvers)
        for solver in solvers:
            solver.apply_solver_settings(updated)

    def price_batch(
        self, theta: np.ndarray, local_ids: np.ndarray
    ) -> dict[int, cb.Demand]:
        values = self.design.parameters.unpack(np.asarray(theta, dtype=np.float64))
        beta = values["beta"]
        delta = values["delta"].reshape(self.design.T, self.design.M)
        quadratic = np.tensordot(values["lambda"], self.design.Qs, axes=([0], [0]))

        ids = np.asarray(local_ids, dtype=np.int64)
        obs_ids = ids % self.design.N
        sim_ids = ids // self.design.N
        unique_obs, obs_pos = np.unique(obs_ids, return_inverse=True)
        base_linear = (
            np.einsum(
                "umk,k->um",
                self.design.X[unique_obs],
                beta,
                optimize=True,
            )
            + delta[self.design.market_idx[unique_obs]]
        )

        def solve_one(row: int) -> tuple[int, cb.Demand]:
            agent_id = int(ids[row])
            agent_key = int(agent_id)
            obs_id = int(obs_ids[row])
            linear = base_linear[obs_pos[row]] + self.shocks[obs_id, sim_ids[row]]
            demand = self._solve(
                linear, quadratic, float(self.design.capacities[obs_id])
            )
            return agent_key, demand

        if self._oracle_workers > 1 and ids.size > 1:
            n_workers = min(self._oracle_workers, int(ids.size))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                return dict(pool.map(solve_one, range(int(ids.size))))

        demands: dict[int, cb.Demand] = {}
        for row in range(int(ids.size)):
            agent_key, demand = solve_one(row)
            demands[agent_key] = demand
        return demands

    def teardown(self) -> None:
        with self._solver_lock:
            solvers = tuple(self._solvers)
            self._solvers.clear()
            self._solver_queue = Queue()
        for solver in solvers:
            solver.close()


def simulate_observed(
    design: BLPDesign,
    *,
    backend: str = "auto",
    quiet: bool = True,
    oracle_workers: int | None = None,
) -> BLPDesign:
    demand = BLPDemandOracle(
        design,
        design.dgp_shocks.reshape(design.N, 1, design.M),
        backend=backend,
        quiet=quiet,
        oracle_workers=oracle_workers,
    )
    ids = np.arange(design.N, dtype=np.int64)
    try:
        demands = demand.price_batch(design.theta_true, ids)
        observed = np.vstack([demands[int(i)].bundle for i in ids])
    finally:
        demand.teardown()
    return replace(design, observed=observed)


def build_distributed_model(
    design: BLPDesign,
    *,
    backend: str = "auto",
    quiet: bool = True,
    oracle_workers: int | None = None,
) -> cb.Model:
    features = BLPFeatures(design, design.est_shocks)
    demand = BLPDemandOracle(
        design,
        design.est_shocks,
        backend=backend,
        quiet=quiet,
        oracle_workers=oracle_workers,
    )
    return cb.Model(
        demand,
        design.parameters,
        features=features,
        observed_features=features,
        formulation=cb.NSlack,
    )


def estimate_blp(
    design: BLPDesign,
    *,
    transport: cb.Transport | None = None,
    backend: str = "auto",
    master_backend: str = "auto",
    tolerance: float = 1e-3,
    max_iterations: int = 80,
    activity_level: str = "iterations",
    oracle_workers: int | None = None,
) -> cb.FitResult:
    transport = cb.SerialTransport() if transport is None else transport
    model = build_distributed_model(
        design, backend=backend, oracle_workers=oracle_workers
    )
    time_limit_callback = cb.point_timeout_callback(
        cb.Schedule((cb.Phase(timeout=1.0, iters=30), cb.Phase(timeout=20.0)))
    )
    activity = None
    if activity_level.lower() != "off":
        activity = cb.ActivityConfig(
            label="blp",
            level=activity_level,
            stdout=True,
        )
    return cb.estimate_distributed(
        model,
        n_observations=design.N,
        n_simulations=design.S,
        transport=transport,
        master_backend=master_backend,
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        cut_policy=cb.AddAll(),
        iteration_callback=time_limit_callback,
        activity=activity,
    )


def price_sensitivity_alpha(
    delta_tj: np.ndarray, prices: np.ndarray, instruments: np.ndarray
) -> tuple[float, float]:
    delta = np.asarray(delta_tj, dtype=np.float64).ravel()
    price = np.asarray(prices, dtype=np.float64).ravel()
    instrument = np.asarray(instruments, dtype=np.float64).ravel()
    constant = np.ones(delta.size, dtype=np.float64)
    ols = IV2SLS(delta, np.column_stack([constant, price]), None, None).fit()
    tsls = IV2SLS(delta, constant, price, instrument).fit()
    return -float(ols.params.iloc[-1]), -float(tsls.params.iloc[-1])


def summarize_fit(design: BLPDesign, fit: cb.FitResult) -> dict[str, object]:
    named = fit.theta_named()
    beta_hat = np.asarray(named["beta"], dtype=np.float64)
    delta_hat = np.asarray(named["delta"], dtype=np.float64).reshape(
        design.T, design.M
    )
    lambda_hat = np.asarray(named["lambda"], dtype=np.float64)
    ols_alpha, tsls_alpha = price_sensitivity_alpha(
        delta_hat, design.prices, design.instruments
    )
    true_ols_alpha, true_tsls_alpha = price_sensitivity_alpha(
        design.delta_true, design.prices, design.instruments
    )
    return {
        "N": design.N,
        "T": design.T,
        "M": design.M,
        "S": design.S,
        "K": design.K,
        "iterations": int(fit.metadata["iterations"]),
        "converged": bool(fit.metadata["converged"]),
        "runtime_seconds": float(fit.runtime_seconds),
        "objective": float(fit.objective),
        "mean_bundle_size": float(design.observed.sum(axis=1).mean()),
        "beta_true": design.beta_true.tolist(),
        "beta_hat": beta_hat.tolist(),
        "lambda_true": design.lambda_true.tolist(),
        "lambda_hat": lambda_hat.tolist(),
        "delta_correlation": float(
            np.corrcoef(delta_hat.ravel(), design.delta_true.ravel())[0, 1]
        ),
        "ols_alpha": ols_alpha,
        "tsls_alpha": tsls_alpha,
        "ols_alpha_true_delta": true_ols_alpha,
        "tsls_alpha_true_delta": true_tsls_alpha,
    }


def make_transport(mode: str) -> cb.Transport:
    if mode == "serial":
        return cb.SerialTransport()
    if mode == "mpi":
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


def build_distributed_design(
    args: argparse.Namespace, backend: str, transport: cb.Transport
) -> BLPDesign:
    parameters = make_parameters(int(args.T), int(args.M))
    publish: dict[str, np.ndarray] = {}
    with transport.collective():
        if transport.node.node_rank == 0:
            design = build_design(
                T=int(args.T),
                M=int(args.M),
                n_per_market=args.n_per_market,
                n_simulations=int(args.S),
                capacity=int(args.capacity),
                seed=int(args.seed),
                nu_scale=float(args.nu_scale),
            )
            design = simulate_observed(
                design,
                backend=backend,
                quiet=not bool(args.solver_output),
                oracle_workers=args.oracle_workers,
            )
            publish = design.arrays()
    shared = transport.node_shared(publish)
    return BLPDesign.from_arrays(shared, parameters)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BLP-style bundle-demand estimation over a "
        "quadratic-knapsack demand oracle."
    )
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--n-per-market", type=int, default=None)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--capacity", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nu-scale", type=float, default=NU_SCALE)
    p.add_argument("--backend", choices=("auto", "gurobi", "highs"), default="auto")
    p.add_argument(
        "--master-backend", choices=("auto", "gurobi", "highs"), default="auto"
    )
    p.add_argument("--transport", choices=("auto", "serial", "mpi"), default="auto")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--max-iterations", type=int, default=80)
    p.add_argument("--oracle-workers", type=int, default=None)
    p.add_argument(
        "--activity-level",
        choices=("off", "summary", "iterations", "diagnostic"),
        default="iterations",
    )
    p.add_argument("--solver-output", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    backend = choose_backend(args.backend)
    transport = make_transport(args.transport)
    try:
        design = build_distributed_design(args, backend, transport)
        fit = estimate_blp(
            design,
            transport=transport,
            backend=backend,
            master_backend=args.master_backend,
            tolerance=float(args.tolerance),
            max_iterations=int(args.max_iterations),
            activity_level=str(args.activity_level),
            oracle_workers=args.oracle_workers,
        )
        if transport.rank == 0:
            summary = summarize_fit(design, fit)
            print(
                "blp:",
                f"N={summary['N']}",
                f"T={summary['T']}",
                f"M={summary['M']}",
                f"S={summary['S']}",
                f"K={summary['K']}",
                f"backend={backend}",
            )
            print(f"mean bundle size: {summary['mean_bundle_size']:.2f}")
            print(
                "iterations:",
                summary["iterations"],
                "converged:",
                summary["converged"],
                "wall:",
                f"{summary['runtime_seconds']:.1f}s",
            )
            print("objective:", round(float(summary["objective"]), 6))
            print(
                "beta_hat:",
                np.asarray(summary["beta_hat"], dtype=np.float64).round(6).tolist(),
            )
            print(
                "lambda_hat:",
                np.asarray(summary["lambda_hat"], dtype=np.float64).round(6).tolist(),
            )
            print("delta correlation:", round(float(summary["delta_correlation"]), 6))
            print(
                "target alpha:",
                round(ALPHA_TRUE, 6),
            )
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
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
