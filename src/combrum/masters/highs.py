"""HiGHS-hosted master problem: the license-free LP backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from combrum.master import CutReadings, MasterBackend
from combrum.masters._common import validated_construction
from combrum.transport.base import CutRow


def available() -> bool:
    """True when highspy imports and a solver instance constructs."""
    # Broad excepts: any import-time or construction failure means not available.
    try:
        import highspy
    except Exception:
        return False
    try:
        highspy.Highs()
    except Exception:
        return False
    return True


@dataclass
class _Solution:
    """Last solve copied out at solve time, so accessors keep reporting it
    after later cut installs invalidate the solver's own query surface.
    """

    theta: np.ndarray
    objective: float
    row_keys: tuple[tuple[int, bytes], ...]
    sorted_row_keys: tuple[tuple[int, bytes], ...]
    sorted_row_order: np.ndarray
    u_values: dict[int, float]
    row_duals: np.ndarray | None = None
    row_slacks: np.ndarray | None = None
    bound_duals: dict[int, float] | None = None


class HighsMaster(MasterBackend):
    """One replication's relaxation hosted on a per-instance HiGHS model.

    Slack columns are created lazily at an agent's first installed cut, so
    master width tracks the active agent set unless ``n_agents`` is given.
    """

    def __init__(
        self,
        K: int,
        theta_bounds: tuple[np.ndarray, np.ndarray],
        c_theta: np.ndarray,
        u_coef: Callable[[int], float],
        params: Mapping[str, object] | None = None,
        n_agents: int | None = None,
    ) -> None:
        self._h = None
        # n_agents set: declare all u-columns up front; None: lazy per first cut.
        self._n_agents = None if n_agents is None else int(n_agents)
        self._K, self._lower, self._upper, self._c = validated_construction(
            K, theta_bounds, c_theta
        )
        if not callable(u_coef):
            raise ValueError(f"u_coef must be callable; got {type(u_coef).__name__}")
        self._u_coef = u_coef
        self._params = dict(params) if params else {}
        u_lower = self._params.pop("u_lower_bound", 0.0)
        self._u_lower_bound = None if u_lower is None else float(u_lower)
        # Slack coefficients read once and kept for the master's lifetime:
        # reinstall must restore the exact objective an agent entered with.
        self._u_obj: dict[int, float] = {}
        # Import here, not module scope: building a master commits to highspy.
        import highspy

        self._highspy = highspy
        self._build()

    def _invalidate_solution(self) -> None:
        self._solution = None

    def _build(self) -> None:
        solver = self._highspy.Highs()
        solver.setOptionValue("output_flag", False)
        if "solver" not in self._params:
            self._check_status(
                solver.setOptionValue("solver", "simplex"),
                "setOptionValue(solver)",
            )
        for key, value in self._params.items():
            self._check_status(
                solver.setOptionValue(key, value), f"setOptionValue({key})"
            )
        no_index = np.array([], dtype=np.int32)
        no_value = np.array([], dtype=np.float64)
        for k in range(self._K):
            self._check_status(
                solver.addCol(
                    float(self._c[k]),
                    float(self._lower[k]),
                    float(self._upper[k]),
                    0,
                    no_index,
                    no_value,
                ),
                f"addCol(theta[{k}])",
            )
        self._h = solver
        self._n_cols = self._K
        self._installed: dict[tuple[int, bytes], CutRow] = {}
        self._row_index: dict[tuple[int, bytes], int] = {}
        self._row_keys: list[tuple[int, bytes]] = []
        self._u_cols: dict[int, int] = {}
        self._u_upper: dict[int, float] = {}
        self._solution: _Solution | None = None
        if self._n_agents is not None:
            # Pre-declare all u-columns in agent order: a fixed column structure
            # makes a warm in-place re-solve deterministic even at a degenerate
            # optimum. With the default lower bound, a cutless agent's u sits at
            # lb=0 -> 0, leaving the estimate unchanged.
            for agent_id in range(self._n_agents):
                coef = self._u_obj.setdefault(agent_id, float(self._u_coef(agent_id)))
                if not np.isfinite(coef):
                    raise ValueError(f"u_coef({agent_id}) must be finite; got {coef!r}")
                self._check_status(
                    solver.addCol(
                        coef,
                        self._u_lb(),
                        self._highspy.kHighsInf,
                        0,
                        np.array([], dtype=np.int32),
                        np.array([], dtype=np.float64),
                    ),
                    f"addCol(u[{agent_id}])",
                )
                self._u_cols[agent_id] = self._n_cols
                self._u_upper[agent_id] = float(self._highspy.kHighsInf)
                self._n_cols += 1

    # -- cut management ----------------------------------------------------

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        fresh: dict[tuple[int, bytes], CutRow] = {}
        for row in rows:
            key = (row.agent_id, row.bundle_key)
            if key in self._installed or key in fresh:
                continue
            self._validate_row(row)
            # First occurrence wins; duplicates absorbed.
            fresh[key] = row
        # Invalidate before the installs so a mid-batch solver error cannot
        # leave accessors reporting pre-add results.
        if fresh:
            self._invalidate_solution()
        # Install in canonical key order so equal row sets build identical
        # models regardless of in-batch arrival permutation.
        for row in sorted(fresh.values(), key=lambda r: (r.agent_id, r.bundle_key)):
            self._install(row)
            self._installed[(row.agent_id, row.bundle_key)] = row
        return len(fresh)

    def _install(self, row: CutRow) -> None:
        u_col = self._u_cols.get(row.agent_id)
        created_u_col = False
        if u_col is None:
            coef = self._u_obj.setdefault(
                row.agent_id, float(self._u_coef(row.agent_id))
            )
            if not np.isfinite(coef):
                raise ValueError(f"u_coef({row.agent_id}) must be finite; got {coef!r}")
            self._add_u_columns(row.agent_id, coef, self._initial_u_upper(row, coef))
            u_col = self._u_cols[row.agent_id]
            created_u_col = True
        else:
            self._update_u_upper(row)
        # Pass only nonzero theta entries to the sparse interface.
        nonzero = np.flatnonzero(row.phi)
        indices = np.append(nonzero, u_col).astype(np.int32)
        values = np.append(-row.phi[nonzero], 1.0)
        status = self._h.addRow(
            float(row.epsilon), self._highspy.kHighsInf, indices.size, indices, values
        )
        try:
            self._check_status(status, f"addRow({(row.agent_id, row.bundle_key)!r})")
        except Exception:
            if created_u_col:
                self._remove_new_u_column(row.agent_id)
            raise
        key = (row.agent_id, row.bundle_key)
        self._row_index[key] = len(self._row_keys)
        self._row_keys.append(key)

    def _add_u_columns(self, agent_id: int, coef: float, upper: float) -> None:
        no_index = np.array([], dtype=np.int32)
        no_value = np.array([], dtype=np.float64)
        self._check_status(
            self._h.addCol(coef, self._u_lb(), upper, 0, no_index, no_value),
            f"addCol(u[{agent_id}])",
        )
        self._u_cols[agent_id] = self._n_cols
        self._u_upper[agent_id] = upper
        self._n_cols += 1

    def _remove_new_u_column(self, agent_id: int) -> None:
        col = self._u_cols[agent_id]
        status = self._h.deleteCols(1, np.array([col], dtype=np.int32))
        self._check_status(status, f"deleteCols(u[{agent_id}])")
        del self._u_cols[agent_id]
        del self._u_upper[agent_id]
        del self._u_obj[agent_id]
        self._n_cols -= 1

    def _u_lb(self) -> float:
        if self._u_lower_bound is None:
            return -float(self._highspy.kHighsInf)
        return float(self._u_lower_bound)

    def _validate_row(self, row: CutRow) -> None:
        if row.phi.shape != (self._K,):
            raise ValueError(
                f"cut phi must have shape ({self._K},); got {row.phi.shape}"
            )
        if not np.isfinite(row.phi).all():
            raise ValueError("cut phi must be finite everywhere")
        if not np.isfinite(row.epsilon):
            raise ValueError(f"cut epsilon must be finite; got {row.epsilon!r}")

    def extract_cuts(self) -> tuple[CutRow, ...]:
        return tuple(self._installed[key] for key in sorted(self._installed))

    @property
    def n_active_cuts(self) -> int:
        return len(self._installed)

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        # clear() + rebuild (the reinstall contract; in-place removal would leave
        # solver history).
        self._h.clear()
        self._build()
        self.add_cuts(rows)

    def remove_cuts(self, keys: Iterable[tuple[int, bytes]]) -> int:
        # In-place: delete retired rows, keeping other rows, u-columns, and the
        # warm basis. deleteRows shifts remaining rows down preserving order;
        # kept indices are recompacted below to match. Result equals
        # reinstall(kept).
        retired = [k for k in set(keys) if k in self._row_index]
        if not retired:
            return 0
        idx = np.array(sorted(self._row_index[k] for k in retired), dtype=np.int32)
        status = self._h.deleteRows(idx.size, idx)
        self._check_status(status, "deleteRows")
        for k in retired:
            del self._installed[k]
            del self._row_index[k]
        # Recompact tracked indices to contiguous 0..n-1, sorted by old index.
        for new_i, k in enumerate(sorted(self._row_index, key=self._row_index.get)):
            self._row_index[k] = new_i
        self._row_keys = sorted(self._row_index, key=self._row_index.get)
        self._invalidate_solution()
        return len(retired)

    def set_rhs(self, updates: Mapping[tuple[int, bytes], float]) -> None:
        # Validate the whole key set before mutating: makes the update
        # all-or-nothing, so a missing key cannot leave the relaxation
        # partially changed.
        for key in updates:
            if key not in self._row_index:
                raise KeyError(key)
        # Invalidate the cached solve before the writes so a mid-loop solver
        # error cannot leave accessors reporting pre-edit results.
        self._invalidate_solution()
        for key, new_eps in updates.items():
            idx = self._row_index[key]
            # Cut is `>=`: only the lower bound is the RHS; upper stays at
            # infinity to keep the row one-sided.
            status = self._h.changeRowBounds(
                idx, float(new_eps), self._highspy.kHighsInf
            )
            if status != self._highspy.HighsStatus.kOk:
                raise RuntimeError(
                    f"changeRowBounds for cut {key!r} returned"
                    f" {status.name}; expected kOk"
                )
            # Keep the installed-row mirror exact so extract_cuts/reinstall see
            # the new RHS; phi is untouched.
            row = replace(self._installed[key], epsilon=float(new_eps))
            self._installed[key] = row
            self._update_u_upper(row)

    def _initial_u_upper(self, row: CutRow, coef: float) -> float:
        if self._u_lower_bound is None or coef < 0.0:
            return float(self._highspy.kHighsInf)
        return self._finite_u_upper(row)

    def _finite_u_upper(self, row: CutRow) -> float:
        rhs_max = float(row.epsilon) + float(
            np.where(row.phi >= 0.0, row.phi * self._upper, row.phi * self._lower).sum()
        )
        return max(self._u_lb(), rhs_max)

    def _update_u_upper(self, row: CutRow) -> None:
        if self._u_lower_bound is None:
            return
        coef = self._u_obj[row.agent_id]
        if coef < 0.0:
            return
        current = self._u_upper.get(row.agent_id, float(self._highspy.kHighsInf))
        candidate = self._finite_u_upper(row)
        if current < self._highspy.kHighsInf and candidate <= current:
            return
        col = self._u_cols[row.agent_id]
        status = self._h.changeColBounds(col, self._u_lb(), candidate)
        self._check_status(status, f"changeColBounds(u[{row.agent_id}])")
        self._u_upper[row.agent_id] = candidate

    # -- solving and reporting ----------------------------------------------

    def solve(self) -> None:
        run_status = self._h.run()
        model_status = self._h.getModelStatus()
        optimal = self._highspy.HighsModelStatus.kOptimal
        if run_status != self._highspy.HighsStatus.kOk or (model_status != optimal):
            # anything but Optimal is solver distress here (see MasterBackend.solve).
            raise RuntimeError(
                "master solve terminated"
                f" {self._h.modelStatusToString(model_status)}"
                f" (run status {run_status.name}); expected Optimal"
            )
        col_values = np.asarray(self._h.allVariableValues(), dtype=np.float64)
        theta = np.array(col_values[: self._K], dtype=np.float64)
        theta.setflags(write=False)
        row_keys = tuple(self._row_keys)
        sorted_row_keys = tuple(sorted(row_keys))
        index = {key: i for i, key in enumerate(row_keys)}
        sorted_row_order = np.fromiter(
            (index[key] for key in sorted_row_keys),
            dtype=np.int64,
            count=len(sorted_row_keys),
        )
        sorted_row_order.setflags(write=False)
        self._solution = _Solution(
            theta=theta,
            objective=float(self._h.getObjectiveValue()),
            row_keys=row_keys,
            sorted_row_keys=sorted_row_keys,
            sorted_row_order=sorted_row_order,
            u_values=self._u_values_now(col_values),
        )

    def _u_values_now(self, col_values: np.ndarray) -> dict[int, float]:
        out: dict[int, float] = {}
        active_agents = {agent_id for agent_id, _key in self._installed}
        for agent_id in sorted(active_agents):
            col = self._u_cols[agent_id]
            out[int(agent_id)] = float(col_values[col])
        return out

    def _bound_duals_now(self, solution: object) -> dict[int, float]:
        at_bound = (
            self._highspy.HighsBasisStatus.kLower,
            self._highspy.HighsBasisStatus.kUpper,
        )
        statuses = self._h.getBasis().col_status
        return {
            k: float(solution.col_dual[k])
            for k in range(self._K)
            if statuses[k] in at_bound
        }

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
        return dict(self._last().u_values)

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        last = self._last()
        duals = self._row_duals(last)
        return {key: float(value) for key, value in zip(last.row_keys, duals)}

    def cut_readings(self, *, dual: bool = False, slack: bool = False) -> CutReadings:
        last = self._last()
        keys = last.sorted_row_keys
        order = last.sorted_row_order
        dual_arr = (
            np.asarray(self._row_duals(last)[order], dtype=np.float64) if dual else None
        )
        slack_arr = (
            np.asarray(self._row_slacks(last)[order], dtype=np.float64)
            if slack
            else None
        )
        return CutReadings(keys=keys, dual=dual_arr, slack=slack_arr)

    def solved_cut_keys(self) -> frozenset[tuple[int, bytes]]:
        return frozenset(self._last().row_keys)

    def bound_duals(self) -> dict[int, float]:
        last = self._last()
        if last.bound_duals is None:
            last.bound_duals = self._bound_duals_now(self._h.getSolution())
        return dict(last.bound_duals)

    def _row_duals(self, last: _Solution) -> np.ndarray:
        if last.row_duals is None:
            row_duals = np.asarray(
                self._h.getSolution().row_dual[: len(last.row_keys)],
                dtype=np.float64,
            )
            row_duals.setflags(write=False)
            last.row_duals = row_duals
        return last.row_duals

    def _row_slacks(self, last: _Solution) -> np.ndarray:
        if last.row_slacks is None:
            solution = self._h.getSolution()
            row_values = np.asarray(
                solution.row_value[: len(last.row_keys)],
                dtype=np.float64,
            )
            eps = np.fromiter(
                (self._installed[key].epsilon for key in last.row_keys),
                dtype=np.float64,
                count=len(last.row_keys),
            )
            row_slacks = row_values - eps
            row_slacks.setflags(write=False)
            last.row_slacks = row_slacks
        return last.row_slacks

    # -- objective shaping ---------------------------------------------------

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        ref = np.array(ref, dtype=np.float64)
        if ref.shape != (self._K,):
            raise ValueError(f"ref must have shape ({self._K},); got {ref.shape}")
        if weight <= 0:
            return
        raise NotImplementedError(
            "the highs backend does not expose quadratic penalties: native"
            " HiGHS QP solves stalled on full-size combRUM masters; use a"
            " quadratic-capable backend such as gurobi"
        )

    def _check_status(self, status: object, operation: str) -> None:
        # kWarning is success with a note (e.g. addRow drops |value| <= 1e-9
        # matrix entries); only kError means the edit did not take effect.
        if status == self._highspy.HighsStatus.kError:
            raise RuntimeError(f"{operation} returned kError")

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release the solver instance."""
        if self._h is not None:
            self._h.clear()
            self._h = None
        self._solution = None

    def __enter__(self) -> HighsMaster:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
