"""Per-agent-slack row generation: one epigraph variable per agent.

The relaxation (built by the caller, never here) is

    minimize    c_theta . theta  +  sum_a u_coef(a) * u_a
    subject to  u_a >= phi_a(d) . theta + eps_a(d)   per installed cut (a, d)
                lower <= theta <= upper,   u_a >= 0

Convention: cut rows carry the RAW features of the candidate bundle. The
deterministic observed-bundle part of the criterion is a constant of the
fit, so it lives in caller-built ``c_theta``, never inside the rows. Under
that convention agent ``a``'s priced optimum ``d*`` violates its epigraph by

    rc_a = payoff_a(d*) - u_a = phi_a(d*) . theta + eps_a(d*) - u_a

the reduced cost this module measures, ships cuts by, and stops on.

The feature map ``(agent_id, bundle) -> (phi, eps)`` is injected at
construction so model-specific code stays on the caller's side.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from combrum._bundle_key import canonical_bundle_key as _canonical_bundle_key
from combrum.context import FitContext, ResultPublication
from combrum.demand import Demand, DemandBatch
from combrum.dual import DualSolution
from combrum.formulation import Evaluation, Formulation, FormulationResult
from combrum.interface_resolution import (
    FeatureMap,
    Resolution,
    check_agreement,
    feature_rows,
    resolve_features,
)
from combrum.master import MasterBackend
from combrum.policies import CutPolicyProfile, policy_profile
from combrum.rowgen import MaxContribution, MaxReduced, StepOutcome
from combrum.steprecord import (
    AdmitViolation,
    InstallSnapshot,
    PricedFeature,
    PricedReducedCost,
    PurgeInput,
    TraceSink,
    _Pending,
    priced_features_from,
)
from combrum.transport.base import CutRow, _pack_bundle, _unpack_bundle

# Feature-map injection accepts a bare ``(agent_id, bundle) -> (phi (K,), eps)``
# callable OR a FeatureMap subclass adding a batched ``features_batch``; both
# resolve to one active path at setup.
__all__ = ["FeatureMap", "NSlack"]

_RECEIVED_VIOLATION_BLOCK_ELEMENTS = 1_000_000
_CONTRIBUTE_FEATURE_BLOCK_ELEMENTS = 1_000_000


# Bundle keys share the package-level codec; these aliases keep local call sites
# compact.
bundle_key = _pack_bundle
_bundle_from_key = _unpack_bundle


@dataclass(frozen=True)
class _MasterState:
    """Owner's post-solve master view, mirrored to every rank.

    Broadcasts only scalar/global state; the owner scatters shard-local ``u``
    separately so workers never receive every agent's epigraph value.
    """

    theta: np.ndarray
    objective: float
    n_installed: int
    progressed: int
    u: dict[int, float] | None = None


@dataclass(frozen=True)
class _DualPacket:
    agent_ids: np.ndarray
    bundle_row_ids: np.ndarray
    pis: np.ndarray
    bundle_table: np.ndarray
    bound_duals: dict[int, float]


def _dual_solution(
    rows: tuple[CutRow, ...],
    duals: Mapping[tuple[int, bytes], float],
    bound_duals: Mapping[int, float],
    K: int,
) -> DualSolution:
    table_slots: dict[bytes, int] = {}
    bundles: list[np.ndarray] = []
    agent_ids: list[int] = []
    bundle_row_ids: list[int] = []
    pis: list[float] = []
    # rows arrive in canonical (agent_id, bundle_key) order: the table
    # (first-appearance) and parallel arrays are deterministic.
    for row in rows:
        slot = table_slots.get(row.bundle_key)
        if slot is None:
            slot = len(bundles)
            table_slots[row.bundle_key] = slot
            bundles.append(_bundle_from_key(row.bundle_key))
        agent_ids.append(row.agent_id)
        bundle_row_ids.append(slot)
        pis.append(duals[(row.agent_id, row.bundle_key)])
    table = (
        np.stack(bundles, axis=0)
        if bundles
        # Empty rows carry no bundle to infer width from; use the fit's K.
        else np.empty((0, K), dtype=np.float64)
    )
    return DualSolution(
        rep_id=0,
        agent_ids=np.asarray(agent_ids, dtype=np.int64),
        bundle_row_ids=np.asarray(bundle_row_ids, dtype=np.int64),
        pis=np.asarray(pis, dtype=np.float64),
        bundle_table=table,
        bound_duals=dict(bound_duals),
    )


def _deduplicate_cut_rows(rows: tuple[CutRow, ...]) -> tuple[CutRow, ...]:
    if len(rows) < 2:
        return _canonicalize_cut_rows(rows)
    seen: set[tuple[int, bytes]] = set()
    unique: list[CutRow] = []
    for row in _canonicalize_cut_rows(rows):
        key = (row.agent_id, row.bundle_key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return tuple(unique)


def _canonicalize_cut_row(row: CutRow) -> CutRow:
    key = _canonical_bundle_key(row.bundle_key)
    if key == row.bundle_key:
        return row
    return row._replace(bundle_key=key)


def _canonicalize_cut_rows(rows: tuple[CutRow, ...]) -> tuple[CutRow, ...]:
    if not rows:
        return rows
    return tuple(_canonicalize_cut_row(row) for row in rows)


def _feature_block_size(K: int) -> int:
    return max(1, _CONTRIBUTE_FEATURE_BLOCK_ELEMENTS // max(1, int(K)))


def _dual_packet(dual: DualSolution) -> _DualPacket:
    return _DualPacket(
        agent_ids=np.asarray(dual.agent_ids, dtype=np.int64),
        bundle_row_ids=np.asarray(dual.bundle_row_ids, dtype=np.int64),
        pis=np.asarray(dual.pis, dtype=np.float64),
        bundle_table=np.asarray(dual.bundle_table),
        bound_duals=dict(dual.bound_duals),
    )


def _dual_from_packet(packet: _DualPacket | None) -> DualSolution | None:
    if packet is None:
        return None
    return DualSolution(
        rep_id=0,
        agent_ids=packet.agent_ids,
        bundle_row_ids=packet.bundle_row_ids,
        pis=packet.pis,
        bundle_table=packet.bundle_table,
        bound_duals=packet.bound_duals,
    )


class NSlack(Formulation):
    """Per-agent-slack row generation against the owner-rank master.

    Use ``NSlack`` when you need per-agent structure: it carries one epigraph
    variable per agent, so cut policies can admit and retire rows, and the fit
    can return per-agent slack and cut duals. The master grows with the agent
    count, so for very large ``N`` where none of that is needed,
    :class:`~combrum.formulations.OneSlack` keeps the master to a single
    aggregate column. ``NSlack`` is also the only formulation supported by the
    distributed entry points.

    The master lives only on the owner rank (``ctx.owner_rank``, default 0;
    ``None`` elsewhere); every touch is owner-guarded inside a transport
    collective, followed by one broadcast of the decision state.
    Admission and retirement ride ``ctx.cut_policy`` when present; an unset
    policy admits everything and retires nothing.
    """

    def __init__(self, features: FeatureMap | Callable[..., Any]) -> None:
        # Validation runs at setup (needs the transport for rank agreement);
        # the arg is held until then.
        self._features_arg = features
        # Optional capture sink: when None no capture record is built and no
        # extra compute runs (every capture block guards on it).
        self._trace_sink: TraceSink | None = None
        self._pending: _Pending | None = None

    def set_trace_sink(self, sink: TraceSink | None) -> None:
        """Attach (or detach) the capture sink; default detached.

        Sink-gated: with no sink no record is built. The sink receives one
        sealed :class:`StepRecord` per iteration, emitted at the end of
        ``apply_step``.
        """
        self._trace_sink = sink

    def _prepare_setup(self, ctx: FitContext) -> None:
        self._ctx = ctx
        self._transport = ctx.transport
        if ctx.cut_policy is not None and ctx.weight_mode == "dense":
            validator = getattr(ctx.cut_policy, "validate_master_size", None)
            if callable(validator):
                validator(n_parameters=ctx.K, n_agents=ctx.n_agents)
        # Master hosted on owner_rank (default 0); this rank holds it iff it
        # is the owner. Every master touch + bcast below uses owner_rank.
        self._owner_rank = ctx.owner_rank
        self._owners = np.array([self._owner_rank], dtype=np.int64)
        self._owners.setflags(write=False)
        self._is_owner = ctx.transport.rank == self._owner_rank
        self._master: MasterBackend | None = ctx.master_backend
        self._iteration = 0
        # Resolve the features path once, identically on every rank, before
        # any data-dependent branch; agreement token rides the setup bcast.
        self._features_res: Resolution = resolve_features(self._features_arg)

    def _feature_token(self) -> object:
        return self._features_res.token

    def prepare_warm_cuts(self, rows: Sequence[CutRow]) -> tuple[CutRow, ...]:
        return _canonicalize_cut_rows(tuple(rows))

    def _distributed_feature_token(self) -> object:
        return self._feature_token()

    def _distributed_route_spec(self) -> tuple[np.ndarray, int]:
        return self._ctx.local_ids, self._ctx.n_agents

    def _distributed_owner_u(self) -> Mapping[int, float] | None:
        return self._u if self._is_owner else None

    def _distributed_set_owner_u(self, values: Mapping[int, float] | None) -> None:
        if self._is_owner and values is not None:
            self._u = dict(values)

    def _owner_initial_state(self) -> tuple[_MasterState | None, dict[int, float] | None]:
        packet: _MasterState | None = None
        full_u: dict[int, float] | None = None
        if self._is_owner:
            if self._master is None:
                raise ValueError(
                    "NSlack is master-based by definition:"
                    " ctx.master_backend must be set on the owner rank"
                )
            if not isinstance(self._master, MasterBackend):
                raise ValueError(
                    "ctx.master_backend must implement MasterBackend;"
                    f" got {type(self._master).__name__}"
                )
            self._master.solve()
            packet, full_u = self._state(progressed=0)
        return packet, full_u

    def _setup_owner_local(
        self, ctx: FitContext
    ) -> tuple[_MasterState | None, dict[int, float] | None]:
        """Initialise this rank; only the owner touches the master."""

        self._prepare_setup(ctx)
        return self._owner_initial_state()

    def _distributed_setup_owner_local(
        self, ctx: FitContext
    ) -> tuple[_MasterState | None, dict[int, float] | None]:
        return self._setup_owner_local(ctx)

    def setup(self, ctx: FitContext) -> None:
        self._prepare_setup(ctx)
        packet: _MasterState | None = None
        full_u: dict[int, float] | None = None
        # Master check and features rank-agreement share one guard so a
        # missing master or divergent build fails as an agreed verdict on
        # every rank rather than stranding peers in the broadcast.
        with self._transport.collective():
            packet, full_u = self._owner_initial_state()
            # One broadcast carries the owner's master state and features
            # token; each rank checks its token against the owner's so a per-rank
            # build divergence becomes the agreed transport verdict.
            owner_packet, owner_token = self._transport.bcast(
                (packet, self._features_res.token) if self._is_owner else None,
                root=self._owner_rank,
            )
            local_u = self._local_u(full_u)
            check_agreement(self._features_res.token, owner_token)
        self._adopt(owner_packet, local_u=local_u, full_u=full_u)

    def solve(self) -> np.ndarray:
        return self._theta.copy()

    def contribute(self, demands: Mapping[int, Demand]) -> MaxContribution:
        # Rank-local half of evaluate, minus the allreduce_max: the running
        # max reduced cost and the locally violated rows to exchange.
        worst = 0.0
        # Featurise the post-filter violated subset only: filter on
        # rc > tolerance first, then featurise survivors in iteration order.
        violated_ids: list[int] = []
        violated_bundles: list[np.ndarray] = []
        # Capture (sink-gated) records the full pre-filter rc per priced
        # agent: agents with rc <= tol emit no row but must still be recorded.
        capturing = self._trace_sink is not None
        pending = _Pending(iteration=self._iteration) if capturing else None
        if isinstance(demands, DemandBatch):
            ids = demands.ids
            bundles = demands.bundles
            payoffs = demands.payoffs
            u = np.fromiter(
                (self._u.get(int(agent_id), 0.0) for agent_id in ids),
                dtype=np.float64,
                count=ids.size,
            )
            rc = payoffs - u
            if rc.size:
                worst = max(0.0, float(rc.max()))
            if pending is not None:
                for agent_id, bundle, value in zip(ids, bundles, rc):
                    pending.priced_reduced_costs.append(
                        PricedReducedCost(
                            agent_id=int(agent_id),
                            bundle_key=bundle_key(bundle),
                            rc=float(value),
                        )
                    )
            keep_idx = np.flatnonzero(rc > self._ctx.tolerance)
            rows: list[CutRow] = []
            block_size = _feature_block_size(self._ctx.K)
            for start in range(0, keep_idx.size, block_size):
                block_idx = keep_idx[start : start + block_size]
                block_ids = ids[block_idx]
                block_bundles = bundles[block_idx]
                featured = feature_rows(
                    self._features_res,
                    block_ids,
                    block_bundles,
                )
                if pending is not None:
                    block_payoffs = payoffs[block_idx]
                    block_gaps = demands.gaps[block_idx]
                    for agent, bundle, payoff, gap, (phi, eps) in zip(
                        block_ids,
                        block_bundles,
                        block_payoffs,
                        block_gaps,
                        featured,
                    ):
                        phi_arr = np.array(phi, dtype=np.float64)
                        phi_arr.setflags(write=False)
                        pending.priced_features.append(
                            PricedFeature(
                                agent_id=int(agent),
                                bundle_key=bundle_key(bundle),
                                payoff=float(payoff),
                                gap=float(gap),
                                phi=phi_arr,
                                eps=float(eps),
                            )
                        )
                rows.extend(
                    CutRow(
                        rep_id=0,
                        agent_id=int(agent),
                        phi=phi,
                        epsilon=eps,
                        bundle_key=bundle_key(bundle),
                    )
                    for agent, bundle, (phi, eps) in zip(
                        block_ids, block_bundles, featured
                    )
                )
            if pending is not None:
                self._pending = pending
            return MaxContribution(worst=worst, local_rows=tuple(rows))

        for agent_id, demand in demands.items():
            agent = int(agent_id)
            payoff = float(demand.payoff)
            if not math.isfinite(payoff):
                raise ValueError("demand payoffs must be finite")
            rc = payoff - self._u.get(agent, 0.0)
            if rc > worst:
                worst = rc
            if pending is not None:
                pending.priced_reduced_costs.append(
                    PricedReducedCost(
                        agent_id=agent,
                        bundle_key=bundle_key(demand.bundle),
                        rc=rc,
                    )
                )
            # Rows ship at the same threshold that stops the walk, so a
            # converged step ships nothing.
            if rc > self._ctx.tolerance:
                violated_ids.append(agent)
                violated_bundles.append(demand.bundle)
        featured = feature_rows(self._features_res, violated_ids, violated_bundles)
        if pending is not None:
            # Read the same `featured` rows the cut build below consumes, so
            # the capture is the value, not a recomputation.
            pending.priced_features = list(
                priced_features_from(demands, violated_ids, featured)
            )
            self._pending = pending
        rows = [
            CutRow(
                rep_id=0,
                agent_id=agent,
                phi=phi,
                epsilon=eps,
                bundle_key=bundle_key(bundle),
            )
            for agent, bundle, (phi, eps) in zip(
                violated_ids, violated_bundles, featured
            )
        ]
        # worst floors at 0.0: a tiny negative residual from float
        # cancellation must not reach the stop rule; an empty shard gives 0.0.
        return MaxContribution(worst=worst, local_rows=tuple(rows))

    def finalise(self, reduced: MaxReduced) -> StepOutcome:
        # Every rank holds the identical reduction; no further agreement round.
        return StepOutcome(
            violation=reduced.global_worst,
            install_payload=reduced.received_rows,
        )

    def _owner_install_step(
        self, received: tuple[CutRow, ...]
    ) -> tuple[_MasterState | None, dict[int, float] | None]:
        # Pending record opened in contribute (None with no sink or no prior
        # contribute). Root fills admit/purge/install fields below.
        pending = self._pending
        packet: _MasterState | None = None
        full_u: dict[int, float] | None = None
        if self._is_owner:
            candidates = _deduplicate_cut_rows(received)
            policy = self._ctx.cut_policy
            profile = policy_profile(policy) if policy is not None else None
            if policy is not None:
                if profile.needs_admit_violations or pending is not None:
                    violations = self._received_violations(candidates)
                else:
                    violations = np.empty(0, dtype=np.float64)
                admitted = policy.admit(candidates, violations, self._iteration)
            elif pending is not None:
                violations = self._received_violations(candidates)
                admitted = candidates
            else:
                admitted = candidates
            installed_rows: tuple[CutRow, ...] = ()
            if pending is not None or (policy is not None and profile.retires_cuts):
                installed_rows = self._master.extract_cuts()
            if pending is not None:
                installed_before = frozenset(
                    (row.agent_id, row.bundle_key) for row in installed_rows
                )
                pending.admit_violations = [
                    AdmitViolation(
                        agent_id=row.agent_id,
                        bundle_key=row.bundle_key,
                        violation=float(v),
                    )
                    for row, v in zip(candidates, violations)
                ]
            retired_keys: set[tuple[int, bytes]] = set()
            if policy is not None and profile.retires_cuts:
                retired_keys = self._purge(policy, profile, installed_rows, pending)
            n_new = self._master.add_cuts(admitted)
            if pending is not None:
                pending.install = InstallSnapshot(
                    installed_before=installed_before,
                    admitted=frozenset(
                        (row.agent_id, row.bundle_key) for row in admitted
                    ),
                )
            if n_new or retired_keys:
                self._master.solve()
            packet, full_u = self._state(progressed=n_new)
        return packet, full_u

    def _apply_owner_step(
        self, install_payload: object
    ) -> tuple[_MasterState | None, dict[int, float] | None]:
        """Apply routed rows on the owner without transport calls."""

        received: tuple[CutRow, ...] = install_payload  # type: ignore[assignment]
        return self._owner_install_step(received)

    def _distributed_apply_owner_step(
        self, install_payload: object
    ) -> tuple[_MasterState | None, dict[int, float] | None]:
        return self._apply_owner_step(install_payload)

    def _adopt_owner_state(
        self,
        state: _MasterState,
        *,
        local_u: Mapping[int, float] | None = None,
        full_u: Mapping[int, float] | None = None,
        bump_iteration: bool = True,
    ) -> int:
        self._adopt(state, local_u=local_u, full_u=full_u)
        if bump_iteration:
            self._iteration += 1
            pending = self._pending
            if pending is not None:
                self._emit(pending)
                self._pending = None
        return int(state.progressed)

    def apply_step(self, install_payload: object) -> int:
        received: tuple[CutRow, ...] = install_payload  # type: ignore[assignment]
        packet: _MasterState | None = None
        full_u: dict[int, float] | None = None
        with self._transport.collective():
            packet, full_u = self._owner_install_step(received)
        state = self._transport.bcast(packet, root=self._owner_rank)
        local_u = self._local_u(full_u)
        return self._adopt_owner_state(state, local_u=local_u, full_u=full_u)

    def _received_violations(self, rows: tuple[CutRow, ...]) -> np.ndarray:
        if not rows:
            return np.empty(0, dtype=np.float64)
        theta = self._master.theta()
        out = np.empty(len(rows), dtype=np.float64)
        block_rows = max(
            1,
            _RECEIVED_VIOLATION_BLOCK_ELEMENTS // max(1, int(theta.size)),
        )
        for start in range(0, len(rows), block_rows):
            chunk = rows[start : start + block_rows]
            phi = np.vstack([row.phi for row in chunk])
            epsilon = np.fromiter(
                (row.epsilon for row in chunk),
                dtype=np.float64,
                count=len(chunk),
            )
            u = np.fromiter(
                (self._u.get(row.agent_id, 0.0) for row in chunk),
                dtype=np.float64,
                count=len(chunk),
            )
            out[start : start + len(chunk)] = phi @ theta + epsilon - u
        return out

    def _emit(self, pending: _Pending) -> None:
        sink = self._trace_sink
        if sink is not None:
            sink.emit(pending.seal())

    def evaluate(self, demands: Mapping[int, Demand]) -> Evaluation:
        # Bundled path: contribute, then this method's own max-reduce; the
        # exchange stays in update.
        c = self.contribute(demands)
        violation = self._transport.allreduce_max(c.worst)
        return Evaluation(violation=violation, payload=c.local_rows)

    def update(self, step: Evaluation) -> int:
        # Bundled path: the exchange this method owns, then the shared install.
        rows: tuple[CutRow, ...] = step.payload  # type: ignore[assignment]
        received = self._transport.exchange_cuts(rows, self._owners)
        return self.apply_step(received)

    def _purge(
        self,
        policy: object,
        profile: CutPolicyProfile,
        installed: tuple[CutRow, ...],
        pending: _Pending | None = None,
    ) -> set[tuple[int, bytes]]:
        if self._ctx.weight_mode == "distributed":
            validator = getattr(policy, "validate_lazy_master_size", None)
            if callable(validator):
                installed_agents = len({int(row.agent_id) for row in installed})
                validator(
                    n_parameters=self._ctx.K,
                    installed_agents=installed_agents,
                )
        if not installed:
            if pending is not None:
                pending.purge_inputs = []
            return set()
        readings = self._master.cut_readings(
            dual=profile.needs_purge_duals,
            slack=profile.needs_purge_slacks,
        )
        duals = readings.dual_map() if profile.needs_purge_duals else None
        # Row slack over the last solved relaxation. Newly admitted rows are
        # installed after purge, so rows without last-solve readings stay absent.
        slack = readings.slack_map() if profile.needs_purge_slacks else None
        if pending is not None:
            # Pre-retirement dual/slack the policy.purge reads next; None
            # where the last solve held no reading.
            pending.purge_inputs = [
                PurgeInput(
                    agent_id=row.agent_id,
                    bundle_key=row.bundle_key,
                    dual=(
                        float(duals[(row.agent_id, row.bundle_key)])
                        if duals is not None and (row.agent_id, row.bundle_key) in duals
                        else None
                    ),
                    slack=(
                        slack.get((row.agent_id, row.bundle_key))
                        if slack is not None
                        else None
                    ),
                )
                for row in installed
            ]
        retired = policy.purge(installed, duals, slack, self._iteration)
        retired_keys = {(row.agent_id, row.bundle_key) for row in retired}
        if retired_keys:
            # Retire in place (drop only retired rows, keep u-columns and warm
            # basis): the warm re-solve lands the same vertex as a rebuild from
            # the kept rows, but O(retired) not O(rows); no per-iteration cold
            # rebuild of a large master.
            self._master.remove_cuts(retired_keys)
        return retired_keys

    def _state(self, progressed: int) -> tuple[_MasterState, dict[int, float]]:
        theta = self._master.theta()
        u = self._master.u_values()
        return (
            _MasterState(
                theta=theta,
                objective=self._master.objective(),
                n_installed=self._master.n_active_cuts,
                progressed=int(progressed),
                u=(
                    None
                    if self._ctx.weight_mode == "distributed"
                    else (u if self._owner_rank != 0 else None)
                ),
            ),
            u,
        )

    def _dense_u(self, u: Mapping[int, float]) -> np.ndarray:
        values = np.zeros(self._ctx.n_agents, dtype=np.float64)
        for agent_id, value in u.items():
            values[int(agent_id)] = float(value)
        return values

    def _local_u(self, full_u: Mapping[int, float] | None) -> dict[int, float] | None:
        if self._ctx.weight_mode == "distributed":
            return self._transport.route_agent_values(
                full_u if self._is_owner else None,
                self._ctx.local_ids,
                source=self._owner_rank,
                n_agents=self._ctx.n_agents,
            )
        if self._owner_rank != 0:
            return None
        payload = {"u": self._dense_u(full_u or {})} if self._is_owner else None
        rows = self._transport.scatter_by_agent(payload, self._ctx.local_ids)["u"]
        return {
            int(agent_id): float(value)
            for agent_id, value in zip(self._ctx.local_ids, rows)
            if float(value) != 0.0
        }

    def _local_slack_values(self) -> np.ndarray:
        return np.asarray(
            [self._u.get(int(agent_id), 0.0) for agent_id in self._ctx.local_ids],
            dtype=np.float64,
        )

    def _adopt(
        self,
        state: _MasterState,
        *,
        local_u: Mapping[int, float] | None = None,
        full_u: Mapping[int, float] | None = None,
    ) -> None:
        self._theta = np.asarray(state.theta, dtype=np.float64)
        if full_u is not None and self._is_owner:
            self._u = dict(full_u)
        elif local_u is not None:
            self._u = dict(local_u)
        elif state.u is not None:
            self._u = dict(state.u)
        else:  # pragma: no cover - broken collective path
            raise RuntimeError("NSlack state adoption received no u values")
        self._objective = float(state.objective)
        self._n_installed = int(state.n_installed)

    def result(self) -> FormulationResult:
        publication = self._ctx.result_publication
        if publication & ResultPublication.BROADCAST:
            packet = None
            with self._transport.collective():
                if self._is_owner:
                    rows = self._master.extract_cuts()
                    dual_packet = _dual_packet(
                        _dual_solution(
                            rows,
                            self._master.dual_values(),
                            dict(self._master.bound_duals()),
                            self._ctx.K,
                        )
                    )
                    packet = (
                        rows,
                        dual_packet,
                        self._master.u_values(),
                    )
            # FULL mode: every rank receives every optional artifact.
            active, dual_packet, u_values = self._transport.bcast(
                packet, root=self._owner_rank
            )
            dual = _dual_from_packet(dual_packet)
            slack = np.zeros(self._ctx.n_agents, dtype=np.float64)
            for agent_id, value in u_values.items():
                slack[agent_id] = value
            return FormulationResult(
                theta_hat=self._theta,
                objective=self._objective,
                n_active_cuts=len(active),
                slack=slack,
                active_set=active,
                dual=dual,
            )

        slack = None
        if publication & ResultPublication.SLACK:
            slack = self._transport.gather_agent_values(
                self._local_slack_values(),
                self._ctx.local_ids,
                self._ctx.n_agents,
                root=0,
            )

        active = None
        dual_packet = None
        if publication & (ResultPublication.ACTIVE_SET | ResultPublication.DUAL):
            packet = None
            with self._transport.collective():
                if self._is_owner:
                    rows = self._master.extract_cuts()
                    active = (
                        rows if publication & ResultPublication.ACTIVE_SET else None
                    )
                    dual_packet = (
                        _dual_packet(
                            _dual_solution(
                                rows,
                                self._master.dual_values(),
                                dict(self._master.bound_duals()),
                                self._ctx.K,
                            )
                        )
                        if publication & ResultPublication.DUAL
                        else None
                    )
                    packet = (active, dual_packet)
            delivered = self._transport.send_to_root(
                packet, source=self._owner_rank, root=0
            )
            if delivered is not None:
                active, dual_packet = delivered
            else:
                active = dual_packet = None

        return FormulationResult(
            theta_hat=self._theta,
            objective=self._objective,
            n_active_cuts=self._n_installed,
            slack=slack,
            active_set=active,
            dual=_dual_from_packet(dual_packet),
        )
