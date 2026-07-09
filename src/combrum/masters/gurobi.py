"""Gurobi-hosted master problem with native quadratic penalty support.

By default each instance owns an isolated ``gurobipy`` environment; the
shared process-global default is never used so masters don't couple
lifetimes and license checkouts. A caller building many sequential
masters may pass one started ``env`` instead — the caller then owns its
lifetime, and each ``Env.start()`` it saves is one license checkout.
Output is muted before start (also silencing the license banner); the
caller's ``params`` may re-enable it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from combrum.master import CutReadings, MasterBackend
from combrum.masters._common import validated_construction, validated_u_coefs
from combrum.transport.base import CutRow

_CUT_BATCH_SIZE = 4096


def available() -> bool:
    """True when gurobipy imports AND a licensed environment starts.

    Importability alone is not availability: gurobipy installs without a
    license, so backend auto-selection must check that an environment
    actually starts.
    """
    # Any import- or license-time failure means not available.
    try:
        import gurobipy
    except Exception:
        return False
    try:
        env = _started_env(gurobipy)
    except Exception:
        return False
    env.dispose()
    return True


def _started_env(gurobipy: object) -> object:
    # Mute output before start() so the license banner never prints.
    env = gurobipy.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _status_name(gurobipy: object, status: int) -> str:
    statuses = gurobipy.GRB.Status
    for name in dir(statuses):
        if not name.startswith("_") and getattr(statuses, name) == status:
            return name
    return f"status code {status}"


@dataclass(frozen=True)
class _Solution:
    """Last solve's results, copied out at solve time.

    Values are materialized while the solver's query surface is valid so
    accessors keep reporting the last solve after later cut installs or
    objective edits.
    """

    theta: np.ndarray
    objective: float


class GurobiMaster(MasterBackend):
    """Master relaxation hosted on a per-instance Gurobi model.

    Slack columns are created lazily at an agent's first cut (or all up
    front when ``n_agents`` is given). :meth:`set_penalty` installs the
    quadratic term natively and removes it entirely on revert.
    """

    def __init__(
        self,
        K: int,
        theta_bounds: tuple[np.ndarray, np.ndarray],
        c_theta: np.ndarray,
        u_coef: Callable[[int], float] | np.ndarray,
        params: Mapping[str, object] | None = None,
        n_agents: int | None = None,
        *,
        env: object | None = None,
    ) -> None:
        self._env = None
        self._env_owned = env is None
        self._model = None
        # n_agents given -> all u-columns declared up front (fixed column
        # structure); None -> lazy per-first-cut columns.
        self._n_agents = None if n_agents is None else int(n_agents)
        self._K, self._lower, self._upper, self._c = validated_construction(
            K, theta_bounds, c_theta
        )
        self._u_coef, self._u_coefs = validated_u_coefs(u_coef)
        if self._u_coefs is not None and self._n_agents is not None:
            if self._u_coefs.size < self._n_agents:
                raise ValueError(
                    f"u_coef array must cover all {self._n_agents} agents;"
                    f" got {self._u_coefs.size} coefficients"
                )
        self._params = dict(params) if params else {}
        u_lower = self._params.pop("u_lower_bound", 0.0)
        self._u_lower_bound = None if u_lower is None else float(u_lower)
        # Per-agent slack coefficients read once and kept for the master's
        # lifetime: penalty revert and reinstall must restore the exact
        # objective an agent entered with.
        self._u_obj: dict[int, float] = {}
        import gurobipy

        self._gp = gurobipy
        self._env = env if env is not None else _started_env(gurobipy)
        self._build()

    def _invalidate_solution(self) -> None:
        # The sorted-block cache survives invalidation: it is keyed on
        # _constrs_version, so it stays valid across solves that leave the
        # installed row set unchanged (e.g. after set_rhs).
        self._solution = None
        self._solved_keys = None
        self._cut_duals = None
        self._bound_duals = None

    def _build(self) -> None:
        model = self._gp.Model("master", env=self._env)
        for key, value in self._params.items():
            model.setParam(key, value)
        self._theta_mvar = model.addMVar(
            self._K, lb=self._lower, ub=self._upper, obj=self._c
        )
        self._theta_vars = self._theta_mvar.tolist()
        self._theta_eye = None
        self._linear_mvar = None
        self._linear_coeffs_base = None
        self._linear_coeffs_work = None
        self._model = model
        self._installed: dict[tuple[int, bytes], CutRow] = {}
        self._constrs: dict[tuple[int, bytes], object] = {}
        self._u_vars: dict[int, object] = {}
        self._u_mvar = None
        self._penalty: tuple[np.ndarray, float] | None = None
        self._solution: _Solution | None = None
        # Bumped on every installed-row membership change; solve() stamps the
        # key snapshot with it so readers can reuse the sorted MConstr block
        # for as long as the row set stands.
        self._constrs_version = 0
        self._solved_version = -1
        self._solved_keys: tuple[tuple[int, bytes], ...] | None = None
        self._solved_block_version = -1
        self._solved_block_keys: tuple[tuple[int, bytes], ...] | None = None
        self._solved_block = None
        self._cut_duals: dict[tuple[int, bytes], float] | None = None
        self._bound_duals: dict[int, float] | None = None
        if self._n_agents is not None:
            # Pre-declare ALL u-columns in agent order: a fixed column
            # structure keeps the warm re-solve vertex deterministic even at a
            # degenerate optimum. With the default lower bound, a cutless
            # agent's u sits at lb=0 -> 0, so the estimate is unchanged.
            self._u_mvar = model.addMVar(
                self._n_agents,
                lb=self._u_lb(),
                obj=self._slack_coef_vector(self._n_agents),
            )
            self._u_vars.update(enumerate(self._u_mvar.tolist()))

    def _slack_coef(self, agent_id: int) -> float:
        if self._u_coefs is not None:
            return float(self._u_coefs[agent_id])
        coef = self._u_obj.setdefault(agent_id, float(self._u_coef(agent_id)))
        if not np.isfinite(coef):
            raise ValueError(f"u_coef({agent_id}) must be finite; got {coef!r}")
        return coef

    def _slack_coef_vector(self, n: int) -> np.ndarray:
        if self._u_coefs is not None:
            return self._u_coefs[:n]
        out = np.empty(n, dtype=np.float64)
        for agent_id in range(n):
            out[agent_id] = self._slack_coef(agent_id)
        return out

    # -- cut management ----------------------------------------------------

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        fresh: list[CutRow] = []
        for row in rows:
            key = (row.agent_id, row.bundle_key)
            if key not in self._installed:
                # First occurrence wins; duplicates absorbed.
                self._installed[key] = row
                fresh.append(row)
        # Install in canonical key order so equal row sets build identical
        # models regardless of in-batch arrival permutation.
        ordered = sorted(fresh, key=lambda r: (r.agent_id, r.bundle_key))
        if ordered:
            self._install_batch(ordered)
            self._constrs_version += 1
        return len(fresh)

    def _install_batch(self, rows: Sequence[CutRow]) -> None:
        if self._u_mvar is None:
            # Canonical row order means new agents appear in ascending order,
            # so batch variable creation reproduces the creation order — and
            # with it the column indexing — of a one-row-at-a-time install.
            missing = [
                agent_id
                for agent_id in dict.fromkeys(row.agent_id for row in rows)
                if agent_id not in self._u_vars
            ]
            if missing:
                u_obj = np.fromiter(
                    (self._slack_coef(agent_id) for agent_id in missing),
                    dtype=np.float64,
                    count=len(missing),
                )
                new_vars = self._model.addMVar(
                    len(missing), lb=self._u_lb(), obj=u_obj
                ).tolist()
                self._u_vars.update(zip(missing, new_vars))
                self._invalidate_objective_cache()
        for start in range(0, len(rows), _CUT_BATCH_SIZE):
            chunk = rows[start : start + _CUT_BATCH_SIZE]
            agent_ids = np.fromiter(
                (row.agent_id for row in chunk),
                dtype=np.int64,
                count=len(chunk),
            )
            phi = np.vstack([row.phi for row in chunk])
            epsilon = np.fromiter(
                (row.epsilon for row in chunk),
                dtype=np.float64,
                count=len(chunk),
            )
            u_block = (
                self._u_mvar[agent_ids]
                if self._u_mvar is not None
                else self._gp.MVar.fromlist(
                    [self._u_vars[int(agent_id)] for agent_id in agent_ids]
                )
            )
            constrs = self._model.addConstr(
                u_block >= phi @ self._theta_mvar + epsilon
            ).tolist()
            for row, constr in zip(chunk, constrs):
                self._constrs[(row.agent_id, row.bundle_key)] = constr

    def _u_lb(self) -> float:
        if self._u_lower_bound is None:
            return -float(self._gp.GRB.INFINITY)
        return float(self._u_lower_bound)

    def extract_cuts(self) -> tuple[CutRow, ...]:
        return tuple(self._installed[key] for key in sorted(self._installed))

    @property
    def n_active_cuts(self) -> int:
        return len(self._installed)

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        # Full dispose + rebuild (the reinstall contract; surgical removal would
        # leave solver history). Rebuilding also drops the penalty, so this is a
        # pure LP.
        self._model.dispose()
        self._build()
        self.add_cuts(rows)

    def remove_cuts(self, keys: Iterable[tuple[int, bytes]]) -> int:
        # In-place retirement at O(retired): remove each retired constraint,
        # leaving other rows, the pre-declared u-columns, and the warm basis
        # untouched (no dispose/rebuild). Fixed column structure keeps the warm
        # re-solve vertex deterministic; the result equals reinstall(kept).
        removed = 0
        for key in set(keys):
            constr = self._constrs.pop(key, None)
            if constr is None:
                continue
            self._model.remove(constr)
            del self._installed[key]
            removed += 1
        if removed:
            self._constrs_version += 1
            self._invalidate_solution()
        return removed

    def set_rhs(self, updates: Mapping[tuple[int, bytes], float]) -> None:
        # Validate the whole key set first, so an unknown key makes the update
        # all-or-nothing and never leaves the relaxation partially changed.
        for key in updates:
            if key not in self._constrs:
                raise KeyError(key)
        # Invalidate before the writes so a mid-loop solver error cannot leave
        # accessors reporting the pre-edit solve.
        self._invalidate_solution()
        keys = list(updates)
        values = [float(updates[key]) for key in keys]
        self._model.setAttr("RHS", [self._constrs[key] for key in keys], values)
        for key, new_eps in zip(keys, values):
            # Mirror the new RHS into the installed row; the epsilon-only
            # _replace shares phi and the decoded-bundle memo with the old row.
            self._installed[key] = self._installed[key]._replace(epsilon=new_eps)

    # -- solving and reporting ----------------------------------------------

    def solve(self) -> None:
        self._model.optimize()
        status = self._model.Status
        if status != self._gp.GRB.OPTIMAL:
            # non-OPTIMAL is solver distress (see MasterBackend.solve), not a state.
            raise RuntimeError(
                "master solve terminated"
                f" {_status_name(self._gp, status)}; expected OPTIMAL"
            )
        theta = np.array(self._theta_mvar.X, dtype=np.float64)
        theta.setflags(write=False)
        self._solution = _Solution(
            theta=theta,
            objective=float(self._model.ObjVal),
        )
        # Snapshot the solved key set (not the dict): later installs only ever
        # ADD keys, and removals invalidate the solution, so readers can fetch
        # the live constraint objects by these keys.
        self._solved_keys = tuple(self._constrs)
        self._solved_version = self._constrs_version
        self._cut_duals = None
        self._bound_duals = None

    def _bound_duals_now(self, theta: np.ndarray) -> dict[int, float]:
        grb = self._gp.GRB
        out: dict[int, float] = {}
        if self._penalty is None:
            for k, var in enumerate(self._theta_vars):
                if var.VBasis in (grb.NONBASIC_LOWER, grb.NONBASIC_UPPER):
                    out[k] = float(var.RC)
            return out
        # An active penalty is not an LP certificate even when Gurobi solves it
        # by a simplex-capable path; report bound signals by primal proximity.
        tol = float(self._model.Params.FeasibilityTol)
        for k, var in enumerate(self._theta_vars):
            at_lower = float(theta[k]) - float(self._lower[k]) <= tol
            at_upper = float(self._upper[k]) - float(theta[k]) <= tol
            if at_lower or at_upper:
                out[k] = float(var.RC)
        return out

    def _last(self) -> _Solution:
        if self._solution is None:
            raise RuntimeError(
                "no solve to report: call solve() before reading results"
            )
        return self._solution

    def theta(self) -> np.ndarray:
        return np.array(self._last().theta)

    def objective(self) -> float:
        return self._last().objective

    def u_values(self) -> dict[int, float]:
        self._last()
        agent_ids = sorted({agent_id for agent_id, _key in self._installed})
        if not agent_ids:
            return {}
        vals = self._model.getAttr(
            "X", [self._u_vars[agent_id] for agent_id in agent_ids]
        )
        return {int(agent_id): float(value) for agent_id, value in zip(agent_ids, vals)}

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        self._last()
        if self._cut_duals is None:
            keys, block = self._solved_block_now()
            if not keys:
                self._cut_duals = {}
            else:
                assert block is not None
                pis = block.getAttr("Pi")
                self._cut_duals = {key: float(pi) for key, pi in zip(keys, pis)}
        return dict(self._cut_duals)

    def cut_readings(self, *, dual: bool = False, slack: bool = False) -> CutReadings:
        self._last()
        keys, block = self._solved_block_now()
        if not keys:
            return CutReadings(
                keys=keys,
                dual=np.empty(0, dtype=np.float64) if dual else None,
                slack=np.empty(0, dtype=np.float64) if slack else None,
            )
        assert block is not None
        dual_arr = np.asarray(block.getAttr("Pi"), dtype=np.float64) if dual else None
        slack_arr = None
        if slack:
            # Gurobi's row-sense Slack is negative for loose ``>=`` rows; negate
            # so binding ~= 0 and larger positive = looser.
            slack_arr = -np.asarray(block.getAttr("Slack"), dtype=np.float64)
        return CutReadings(keys=keys, dual=dual_arr, slack=slack_arr)

    def _solved_block_now(self) -> tuple[tuple[tuple[int, bytes], ...], object | None]:
        if not self._solved_keys:
            return (), None
        if self._solved_block_version != self._solved_version:
            keys = tuple(sorted(self._solved_keys))
            self._solved_block = self._gp.MConstr.fromlist(
                [self._constrs[key] for key in keys]
            )
            self._solved_block_keys = keys
            self._solved_block_version = self._solved_version
        return self._solved_block_keys, self._solved_block

    def solved_cut_keys(self) -> frozenset[tuple[int, bytes]]:
        self._last()
        return frozenset(self._solved_keys or ())

    def bound_duals(self) -> dict[int, float]:
        solution = self._last()
        if self._bound_duals is None:
            self._bound_duals = self._bound_duals_now(solution.theta)
        return dict(self._bound_duals)

    # -- objective shaping ---------------------------------------------------

    def _invalidate_objective_cache(self) -> None:
        self._linear_mvar = None
        self._linear_coeffs_base = None
        self._linear_coeffs_work = None

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        ref = np.array(ref, dtype=np.float64)
        if ref.shape != (self._K,):
            raise ValueError(f"ref must have shape ({self._K},); got {ref.shape}")
        if weight <= 0:
            if self._penalty is not None:
                self._penalty = None
                self._invalidate_solution()
                # Restore the linear objective so the terminating solve is a
                # true LP whose duals belong to the unpenalized relaxation.
                self._set_linear_objective()
            return
        self._set_quadratic_objective(ref, float(weight))
        self._invalidate_solution()
        ref.setflags(write=False)
        self._penalty = (ref, float(weight))

    def _linear_terms(self) -> tuple[object, np.ndarray]:
        if self._linear_mvar is not None and self._linear_coeffs_base is not None:
            return self._linear_mvar, self._linear_coeffs_base
        if self._u_mvar is not None:
            variables = self._gp.concatenate([self._theta_mvar, self._u_mvar])
            coeffs = np.empty(self._K + self._n_agents, dtype=np.float64)
            coeffs[: self._K] = self._c
            coeffs[self._K :] = self._slack_coef_vector(self._n_agents)
        elif self._u_vars:
            u_mvar = self._gp.MVar.fromlist(list(self._u_vars.values()))
            variables = self._gp.concatenate([self._theta_mvar, u_mvar])
            coeffs = np.empty(self._K + len(self._u_vars), dtype=np.float64)
            coeffs[: self._K] = self._c
            for offset, agent_id in enumerate(self._u_vars, start=self._K):
                coeffs[offset] = self._slack_coef(agent_id)
        else:
            variables = self._theta_mvar
            coeffs = np.array(self._c, dtype=np.float64, copy=True)
        coeffs.setflags(write=False)
        self._linear_mvar = variables
        self._linear_coeffs_base = coeffs
        return variables, coeffs

    def _linear_coefficients(
        self, theta_delta: np.ndarray | None = None
    ) -> tuple[object, np.ndarray]:
        variables, base = self._linear_terms()
        if theta_delta is None:
            return variables, base
        work = self._linear_coeffs_work
        if work is None or work.shape != base.shape:
            work = np.array(base, dtype=np.float64, copy=True)
            self._linear_coeffs_work = work
        else:
            np.copyto(work, base)
        work[: self._K] += theta_delta
        return variables, work

    def _set_linear_objective(self) -> None:
        variables, coeffs = self._linear_coefficients()
        self._model.setMObjective(
            None,
            coeffs,
            0.0,
            xc=variables,
            sense=self._gp.GRB.MINIMIZE,
        )

    def _set_quadratic_objective(self, ref: np.ndarray, weight: float) -> None:
        installed = self._penalty
        if installed is not None and weight == installed[1]:
            # Q = weight*I is already installed: rewrite only the theta
            # objective coefficients and the constant. setMObjective replaces
            # the whole objective and drops the simplex warm basis, so the
            # equal-weight re-anchor must never route through it; this path
            # is what keeps penalty re-solves warm across iterations.
            self._theta_mvar.Obj = self._c - 2.0 * weight * ref
            self._model.ObjCon = weight * float(ref @ ref)
            return
        # Weight change or first install: gurobipy has no in-place edit of
        # the quadratic term, so re-setting Q takes the full matrix-objective
        # push (and the next solve starts cold).
        theta_linear = -2.0 * weight * ref
        constant = weight * float(ref @ ref)
        variables, coeffs = self._linear_coefficients(theta_linear)
        self._model.setMObjective(
            self._quadratic_eye() * weight,
            coeffs,
            constant,
            xQ_L=self._theta_mvar,
            xQ_R=self._theta_mvar,
            xc=variables,
            sense=self._gp.GRB.MINIMIZE,
        )

    def _quadratic_eye(self) -> object:
        if self._theta_eye is None:
            from scipy import sparse

            self._theta_eye = sparse.eye(self._K, format="csr", dtype=np.float64)
        return self._theta_eye

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release the model, and the environment when this master owns it."""
        if self._model is not None:
            self._model.dispose()
            self._model = None
        if self._env is not None:
            if self._env_owned:
                self._env.dispose()
            self._env = None

    def __enter__(self) -> GurobiMaster:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
