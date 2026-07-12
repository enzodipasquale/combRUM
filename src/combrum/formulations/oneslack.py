"""Single-aggregate-slack row generation: one epigraph variable in total.

The relaxation (built by the caller with a constant 1.0 slack coefficient;
agent weights live inside the aggregate rows) is

    minimize    c_theta . theta  +  u
    subject to  u >= phi_agg . theta + eps_agg     per installed aggregate
                lower <= theta <= upper, with u's lower bound supplied by
                the master backend (default 0; None means no lower bound)

where one iteration's aggregate row is the weighted sum of every agent's
priced optimum::

    phi_agg = sum_a w_a * phi_a(d*_a)      eps_agg = sum_a w_a * eps_a(d*_a)

Cut rows carry raw candidate features; the deterministic observed-bundle
part of the criterion lives in the caller-built ``c_theta``.

Requires pricing every agent every iteration: an aggregate row assembled
from a partial sweep would mix selections taken at different query points
into one constraint at no theta.

The feature map ``(agent_id, bundle) -> (phi, eps)`` is injected at
construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from combrum.context import FitContext
from combrum.demand import Demand
from combrum.formulation import (
    Evaluation,
    Formulation,
    FormulationResult,
    _require_owner_master,
    _staged_penalty,
)
from combrum.interface_resolution import (
    FeatureMap,
    Mode,
    Resolution,
    _aggregate_call,
    _aggregate_wants_k,
    check_agreement,
    feature_rows,
    resolve_features,
)
from combrum.master import MasterBackend
from combrum.rowgen import StepOutcome, SumContribution, SumReduced
from combrum.steprecord import TraceSink, _Pending, priced_features_from
from combrum.transport.base import CutRow

__all__ = [
    "FeatureMap",
    "OneSlack",
]

# the synthetic agent every aggregate row belongs to: the master keys cuts by
# (agent_id, bundle_key), so one fixed id pools all aggregate rows into a single
# epigraph column while the digest key keeps each row's identity
AGGREGATE_AGENT_ID = 0


def _aggregate_key(phi_agg: np.ndarray, eps_agg: float) -> bytes:
    # sum_reproducible's bitwise contract makes these float64 bytes (hence
    # the digest) identical on every rank and at every rank count.
    payload = np.ascontiguousarray(
        np.concatenate([phi_agg, [eps_agg]]), dtype=np.float64
    )
    return hashlib.sha256(payload.tobytes()).digest()


@dataclass(frozen=True)
class _MasterState:
    """Owner's view of the master after a solve, broadcast to every rank.

    Broadcasting the whole decision state once per update keeps every later
    accessor (``solve``, the violation, ``result``) rank-local and rank-invariant.
    """

    theta: np.ndarray
    u: float
    objective: float
    n_installed: int
    progressed: int


class OneSlack(Formulation):
    """Single-aggregate-slack row generation against the owner-rank master.

    Use ``OneSlack`` when one aggregate epigraph variable suffices: the master
    stays a single column regardless of the agent count, so it is the cheap
    choice for large ``N``. It prices every agent every iteration and ships one
    aggregate row, so it does not consult cut policies and does not expose
    per-agent duals or slack. Choose :class:`~combrum.formulations.NSlack`
    instead when you need per-agent slack, cut admission/retirement, or cut
    duals.

    The master lives only on the owner rank (``ctx.owner_rank``, default 0;
    ``ctx.master_backend``; ``None`` elsewhere); every touch happens
    owner-guarded inside a transport collective, followed by one broadcast of
    the full decision state.
    """

    def __init__(self, features: FeatureMap | Callable[..., Any]) -> None:
        # features is a bare per-agent callable or a FeatureMap subclass;
        # the mode is resolved at setup (it needs the transport for rank
        # agreement).
        self._features_arg = features
        self._trace_sink: TraceSink | None = None
        self._pending: _Pending | None = None
        self._iteration = 0

    def prepare_penalty_solve(self, ref: np.ndarray, weight: float) -> None:
        """Stage the next proximal objective change for ``apply_step``."""
        self._pending_penalty = _staged_penalty(ref, weight, self._ctx.K)

    def set_trace_sink(self, sink: TraceSink | None) -> None:
        """Attach (or detach) the capture sink; default is detached.

        With no sink the formulation builds no record. The sink receives one
        sealed :class:`StepRecord` per iteration, emitted at the end of
        ``apply_step``.
        """
        self._trace_sink = sink

    def setup(self, ctx: FitContext) -> None:
        self._ctx = ctx
        self._transport = ctx.transport
        # Every master touch + bcast uses owner_rank, so a non-root owner
        # hosts its master without a forked path.
        self._owner_rank = ctx.owner_rank
        self._is_owner = ctx.transport.rank == self._owner_rank
        self._master: MasterBackend | None = ctx.master_backend
        self._pending_penalty: tuple[np.ndarray, float] | None = None
        self._last_penalty_weight = 0.0
        # Resolve the active features path once, identically on every rank;
        # the rank-agreement token rides the setup broadcast below. The
        # aggregate-mode signature is resolved here too, so contribute never
        # re-inspects it per iteration.
        self._features_res: Resolution = resolve_features(self._features_arg)
        self._aggregate_wants_k = (
            _aggregate_wants_k(self._features_res.active)
            if self._features_res.mode is Mode.OPTIMIZED
            else None
        )
        packet: _MasterState | None = None
        # The owner-only master check and the features rank-agreement sit
        # inside one guard so a missing master or divergent build fails as an
        # agreed verdict on every rank instead of stranding peers in the bcast.
        with self._transport.collective():
            if self._is_owner:
                _require_owner_master(self._master, MasterBackend, "OneSlack")
                # theta_init is deliberately not a first query point: the
                # empty relaxation determines its own optimum, and a query
                # that is no master solution would report a violation
                # belonging to no iterate. The seed is the proximal/warm-start
                # anchor, which binds where a penalty is applied.
                self._master.solve()
                packet = self._state(progressed=0)
            # One broadcast carries the owner's master state and features
            # token; every rank checks its own token against the owner's here.
            owner_packet, owner_token = self._transport.bcast(
                (packet, self._features_res.token) if self._is_owner else None,
                root=self._owner_rank,
            )
            check_agreement(self._features_res.token, owner_token)
        self._adopt(owner_packet)

    def solve(self) -> np.ndarray:
        return self._theta.copy()

    def contribute(self, demands: Mapping[int, Demand]) -> SumContribution:
        # Rank-local half of evaluate without the reduce: the per-local-agent
        # weighted (phi | eps) rows and their global ids, for the engine to
        # SUM-reduce.
        K = self._ctx.K
        weights = self._ctx.agent_weights
        # OneSlack prices every local agent, so featurisation batches over all
        # local demand ids in demands order; the weighted aggregate (the row
        # key) is order-independent of batched vs per-agent path.
        batch_ids = getattr(demands, "ids", None)
        batch_bundles = getattr(demands, "bundles", None)
        if batch_ids is not None and batch_bundles is not None:
            ids = np.asarray(batch_ids, dtype=np.int64)
            bundles = np.asarray(batch_bundles)
        else:
            agent_ids = [int(a) for a in demands]
            ids = np.fromiter(agent_ids, dtype=np.int64, count=len(agent_ids))
            bundles = np.asarray([demand.bundle for demand in demands.values()])
        aggregate_fast_path = self._trace_sink is None and self._transport.size == 1
        if aggregate_fast_path and self._aggregate_wants_k is not None:
            phi_agg, eps_agg = _aggregate_call(
                self._features_res.active,
                ids,
                bundles,
                weights[ids],
                K,
                wants_K=self._aggregate_wants_k,
            )
            row = np.empty((1, K + 1), dtype=np.float64)
            row[0, :K] = phi_agg
            row[0, K] = eps_agg
            return SumContribution(
                terms=row,
                ids=np.asarray([AGGREGATE_AGENT_ID], dtype=np.int64),
            )
        phi_mat, eps_vec, featured = self._feature_block(ids, bundles)
        # Capture (sink only): open a fresh pending record and capture the
        # priced-demand/feature stream over all local agents from the same
        # `featured` rows the aggregate sums, not a recomputation.
        if self._trace_sink is not None:
            agent_ids = [int(a) for a in ids]
            if featured is None:
                featured = [
                    (np.ascontiguousarray(phi_mat[r]), float(eps_vec[r]))
                    for r in range(ids.size)
                ]
            pending = _Pending(iteration=self._iteration)
            pending.priced_features = list(
                priced_features_from(demands, agent_ids, featured)
            )
            self._pending = pending
        rows = np.empty((len(demands), K + 1), dtype=np.float64)
        row_weights = weights[ids]
        rows[:, :K] = row_weights[:, None] * phi_mat
        rows[:, K] = row_weights * eps_vec
        return SumContribution(terms=rows, ids=ids)

    def _feature_block(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, float]] | None]:
        if ids.size == 0:
            return (
                np.empty((0, self._ctx.K), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                [],
            )
        if self._features_res.mode is Mode.OPTIMIZED:
            phi_mat, eps_vec = self._features_res.active(
                ids,
                bundles,
            )
            phi_mat = np.asarray(phi_mat, dtype=np.float64)
            eps_vec = np.asarray(eps_vec, dtype=np.float64)
            if phi_mat.shape != (ids.size, self._ctx.K):
                raise ValueError(
                    "features_batch returned Phi with shape"
                    f" {phi_mat.shape}; expected ({ids.size},"
                    f" {self._ctx.K})"
                )
            if eps_vec.shape != (ids.size,):
                raise ValueError(
                    "features_batch returned Eps with shape"
                    f" {eps_vec.shape}; expected ({ids.size},)"
                )
            return phi_mat, eps_vec, None

        agent_ids = [int(a) for a in ids]
        featured = feature_rows(self._features_res, agent_ids, bundles)
        if featured:
            phi_mat = np.stack([phi for phi, _ in featured], axis=0)
            eps_vec = np.fromiter(
                (eps for _, eps in featured),
                dtype=np.float64,
                count=len(featured),
            )
        else:
            phi_mat = np.empty((0, self._ctx.K), dtype=np.float64)
            eps_vec = np.empty(0, dtype=np.float64)
        return phi_mat, eps_vec, featured

    def finalise(self, reduced: SumReduced) -> StepOutcome:
        # Every rank holds the identical summed aggregate, so the cut row and
        # the stop rule need no further agreement round.
        K = self._ctx.K
        agg = reduced.aggregate
        phi_agg = agg[:K]
        eps_agg = float(agg[K])
        raw = self._violation_raw(phi_agg, eps_agg)
        # Capture reads the same raw value and aggregate-key bytes apply_step's
        # install gate uses, so a <=1e-13 drift that would flip the row key is
        # witnessed here.
        if self._pending is not None:
            self._pending.aggregate_raw = raw
            self._pending.aggregate_bytes = _aggregate_key(phi_agg, eps_agg)
        # Floor at 0.0: at convergence the slack can sit a hair below zero from
        # float cancellation; the distance contract requires nonnegativity and
        # clamping changes no stop decision. With a free epigraph variable,
        # however, the first aggregate row is mandatory even when raw is
        # negative: before that row exists, there is no valid solver-owned
        # u-value to certify against.
        if self._needs_initial_free_u_cut():
            violation = (
                raw
                if raw > self._ctx.tolerance
                else np.nextafter(self._ctx.tolerance, np.inf)
            )
        else:
            violation = raw if raw > 0.0 else 0.0
        return StepOutcome(violation=violation, install_payload=(phi_agg, eps_agg))

    def _violation_raw(self, phi_agg: np.ndarray, eps_agg: float) -> float:
        # Unclamped aggregate slack at the current (pre-update) master
        # solution. self._theta/self._u are unchanged between finalise and
        # apply_step, so apply_step's install gate matches finalise's
        # violation > tolerance.
        theta_term = np.multiply(phi_agg, self._theta).sum(dtype=np.float64)
        return float(theta_term) + eps_agg - self._u

    def apply_step(self, install_payload: object) -> int:
        # Install half of update: at most one aggregate cut, owner-only, then
        # one master-state bcast. The install gate is recomputed from the
        # payload exactly as finalise computed the violation, so a converged
        # step installs nothing.
        pending_penalty = self._pending_penalty
        self._pending_penalty = None
        phi_agg, eps_agg = install_payload  # type: ignore[misc]
        raw = self._violation_raw(phi_agg, eps_agg)
        violation = raw if raw > 0.0 else 0.0
        install = violation > self._ctx.tolerance or self._needs_initial_free_u_cut()
        packet: _MasterState | None = None
        with self._transport.collective():
            if self._is_owner:
                progressed = 0
                if install:
                    row = CutRow(
                        rep_id=0,
                        agent_id=AGGREGATE_AGENT_ID,
                        phi=phi_agg,
                        epsilon=eps_agg,
                        bundle_key=_aggregate_key(phi_agg, eps_agg),
                    )
                    progressed = self._master.add_cuts((row,))
                must_solve = bool(progressed)
                if pending_penalty is not None:
                    ref, weight = pending_penalty
                    penalty_changed = weight > 0.0 or self._last_penalty_weight > 0.0
                    self._master.set_penalty(ref, weight)
                    self._last_penalty_weight = weight
                    must_solve = must_solve or penalty_changed
                if must_solve:
                    self._master.solve()
                packet = self._state(progressed=progressed)
        state = self._transport.bcast(packet, root=self._owner_rank)
        self._adopt(state)
        self._iteration += 1
        # The counter bumps on every apply_step (sink or not) so the captured
        # iteration index is correct whenever a sink is later attached. Emit
        # one sealed record per iteration, then clear for the next contribute.
        if self._pending is not None:
            sink = self._trace_sink
            if sink is not None:
                sink.emit(self._pending.seal())
            self._pending = None
        return state.progressed

    def evaluate(self, demands: Mapping[int, Demand]) -> Evaluation:
        c = self.contribute(demands)
        agg = np.asarray(
            self._transport.sum_reproducible(c.terms, c.ids),
            dtype=np.float64,
        )
        out = self.finalise(SumReduced(aggregate=agg))
        return Evaluation(violation=out.violation, payload=out.install_payload)

    def update(self, step: Evaluation) -> int:
        return self.apply_step(step.payload)

    def _state(self, progressed: int) -> _MasterState:
        theta = self._master.theta()
        u = self._aggregate_u()
        return _MasterState(
            theta=theta,
            u=u,
            objective=self._master.objective(),
            n_installed=self._master.n_active_cuts,
            progressed=int(progressed),
        )

    def _aggregate_u(self) -> float:
        values = self._master.u_values()
        extra = set(values) - {AGGREGATE_AGENT_ID}
        if extra:
            raise RuntimeError(
                "OneSlack master reported non-aggregate epigraph values:"
                f" {sorted(extra)}"
            )
        value = values.get(AGGREGATE_AGENT_ID)
        if value is not None:
            return float(value)
        if self._master.n_active_cuts == 0:
            return 0.0
        raise RuntimeError(
            "OneSlack master has aggregate cuts but reported no aggregate"
            " epigraph value"
        )

    def _needs_initial_free_u_cut(self) -> bool:
        return (
            self._ctx.master_params.get("u_lower_bound", 0.0) is None
            and self._n_installed == 0
        )

    def _adopt(self, state: _MasterState) -> None:
        self._theta = np.asarray(state.theta, dtype=np.float64)
        self._u = float(state.u)
        self._objective = float(state.objective)
        self._n_installed = int(state.n_installed)

    def result(self) -> FormulationResult:
        # All published values were mirrored at the last update, so the answer
        # is identical on every rank with no further communication.
        return FormulationResult(
            theta_hat=self._theta,
            objective=self._objective,
            n_active_cuts=self._n_installed,
        )
