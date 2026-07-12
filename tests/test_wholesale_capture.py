"""Wholesale-capture checks over the either-one ``StepRecord`` stream.

The formulation emits a typed :class:`StepRecord` for every filter-chain input
over its full pre-filter domain each iteration. These tests replay the same
walk along two independent axes and compare the captured records:

* **features**: per-agent bare-callable features versus a batch-only
  ``FeatureMap`` with the oracle, backend, transport, tolerance, and policy
  held fixed.
* **shards**: ``SerialTransport`` versus a ``LocalCluster`` interleave, so cut
  routing and reductions exercise rank-local ownership.

Discrete fields must match exactly (bundle keys, install/admit/purge key sets,
aggregate SHA-256 bytes, agent domains). Continuous fields must match within
``1e-13`` (reduced costs, admit violations, purge duals/slacks, payoff/gap,
aggregate raw, phi/eps). The fixtures also assert that every compared stream is
populated: NSlack drives admit, purge, and install events, while OneSlack drives
the aggregate fields.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import struct

import numpy as np
import pytest

from _family_oracles import (
    FamilyProblem,
    qkp_feature_map_batch_only,
    qkp_problem,
    toy_aggregate_bytes_perturbation,
    toy_eps_perturbation,
    toy_feature_map_batch_only,
    toy_install_gate_perturbation,
    toy_phi_support_perturbation,
    toy_phi_value_perturbation,
    toy_perturbation_price_oracle,
    toy_problem,
    toy_schedule_perturbation,
)
from _walk import WalkOutcome, run_walk
from combrum.cut_policies import Compose, PurgeInactive, SlackStrip, SlackThreshold
from combrum.demand import Demand
from combrum.formulations import NSlack, OneSlack
from combrum.formulations import nslack as _nslack_module
from combrum.oracle import Oracle
from _support.constants import TOLERANCE
from _support.families import load_qkp, load_toy
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum import informed_schedule as _informed_schedule_module
from combrum.informed_schedule import (
    _SUPPORT_ATOL,
    DualConcentration,
    DualInformed,
)
from combrum.steprecord import StepRecord
from combrum.transport import LocalCluster, SerialTransport

#: The continuous-field bar; discrete fields are compared byte-exact.
TOL = 1e-13

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

# Apple Accelerate raises spurious FP-status warnings on provably finite
# matmuls at these sizes; the feature maps are guarded regardless.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::RuntimeWarning:.*matmul.*"),
]


def _nslack_policy() -> Compose:
    """A non-trivial policy that fires admit, purge and install.

    ``SlackThreshold(epsilon=0.0)`` admits only strictly-violated candidates
    (the admit branch the admit-violation capture feeds), and
    ``PurgeInactive(max_age=1)`` retires a cut whose dual reads ~0 on the very
    next signalled solve (the purge branch the purge-input capture feeds). A
    fresh instance per call: ``PurgeInactive`` holds per-cut streak state, so
    it must never be shared across the threads of a ``LocalCluster`` run.
    """
    return Compose(
        admit_chain=[SlackThreshold(epsilon=0.0)],
        purge_chain=[PurgeInactive(max_age=1)],
    )


#: Admit bar for the strict-subset gate. Received pre-admit violations span
#: ~0.02 up past 40 (toy) / 100 (qkp), so a bar of 1.0 splits them: some
#: received rows clear it, some do not, and the admitted set is a strict
#: subset of the received candidates at most iterations.
_ADMIT_SUBSET_EPSILON = 1.0


def _nslack_admit_subset_policy() -> Compose:
    """A policy whose admit stage rejects the weakly-violated received rows.

    Under the ``epsilon=0.0`` default every received row clears the bar and
    admitted degenerates to the whole received set; ``epsilon=1.0`` makes the
    admit filter genuinely exclude rows. ``PurgeInactive(max_age=1)`` keeps
    the purge branch alive; fresh instance per call (no cross-thread streak
    state).
    """
    return Compose(
        admit_chain=[SlackThreshold(epsilon=_ADMIT_SUBSET_EPSILON)],
        purge_chain=[PurgeInactive(max_age=1)],
    )


def _nslack_slack_policy() -> Compose:
    """A policy whose retirement stage reads row SLACK, not the dual.

    ``PurgeInactive`` (the default above) has ``needs_purge_slacks=False``,
    so every captured ``PurgeInput.slack`` is ``None``; ``SlackStrip`` reads
    and captures the master's solver-native row slack instead.
    ``percentile=50.0`` keeps roughly half the rows each call, so cuts churn
    and the slack readings span both binding (~0) and loose (>0) rows. Fresh
    instance per call (no cross-thread state).
    """
    return Compose(
        admit_chain=[SlackThreshold(epsilon=0.0)],
        purge_chain=[SlackStrip(percentile=50.0)],
    )


def _with_features(problem: FamilyProblem, features: object) -> FamilyProblem:
    # Swap only the feature map; oracle + theta geometry untouched, so a
    # difference in the captured records is the features path alone.
    return type(problem)(
        oracle=problem.oracle,
        features=features,
        observed_features=problem.observed_features,
        K=problem.K,
        theta_bounds=problem.theta_bounds,
    )


# --- record-stream comparison helpers ----------------------------------------
#
# Two record streams must align iteration-for-iteration before a field-by-field
# comparison is meaningful.


def _assert_same_shape(a: Sequence[StepRecord], b: Sequence[StepRecord]) -> None:
    assert len(a) == len(b), (
        f"record streams differ in length: {len(a)} vs {len(b)} — the two"
        " walks took a different number of iterations"
    )
    for ra, rb in zip(a, b):
        assert ra.iteration == rb.iteration


def _close(x: float, y: float) -> bool:
    return abs(float(x) - float(y)) <= TOL


def _rc_map(rec: StepRecord) -> dict[tuple[int, bytes], float]:
    # Reduced costs keyed by (agent_id, bundle_key): the agent-id domain and
    # the bundle key are discrete (the keys must match as a set); rc is
    # continuous. Keying this way makes the comparison invariant to the order
    # the shard priced agents in — so the serial and interleaved walks compare.
    return {
        (prc.agent_id, prc.bundle_key): prc.rc
        for prc in rec.priced_reduced_costs
    }


def _admit_map(rec: StepRecord) -> dict[tuple[int, bytes], float]:
    return {
        (av.agent_id, av.bundle_key): av.violation
        for av in rec.admit_violations
    }


def _purge_map(
    rec: StepRecord,
) -> dict[tuple[int, bytes], tuple[float | None, float | None]]:
    return {
        (pi.agent_id, pi.bundle_key): (pi.dual, pi.slack)
        for pi in rec.purge_inputs
    }


def _feature_map(
    rec: StepRecord,
) -> dict[tuple[int, bytes], tuple[float, float, bytes, float]]:
    # (agent_id, bundle_key) -> (payoff, gap, phi-bytes, eps): payoff/gap/eps
    # continuous, phi both continuous (values) and discrete (zero-mask), so phi
    # rides both checks below.
    return {
        (pf.agent_id, pf.bundle_key): (
            pf.payoff,
            pf.gap,
            np.asarray(pf.phi, dtype=np.float64).tobytes(),
            pf.eps,
        )
        for pf in rec.priced_features
    }


def _decode_bundle_key(key: bytes) -> np.ndarray:
    # Independent inverse of the bundle-key codec: magic, dtype tag, shape,
    # raw bytes. Kept here rather than imported from the production helper so
    # the capture can re-featurise a cut key independently.
    magic, dtype_len, ndim = struct.unpack_from("!3sHH", key)
    assert magic == b"CB1"
    pos = struct.calcsize("!3sHH")
    dtype = np.dtype(key[pos : pos + dtype_len].decode("ascii"))
    pos += dtype_len
    shape = tuple(
        int(v) for v in np.frombuffer(key, dtype="<i8", count=ndim, offset=pos)
    )
    pos += ndim * 8
    return np.frombuffer(key[pos:], dtype=dtype).reshape(shape)


def _independent_eps(bundle_key: bytes, nu_row: np.ndarray) -> float:
    # eps = sum of the shock over the chosen items, accumulated by a plain
    # Python scalar loop — a structurally distinct reduction from the feature
    # map's vectorised dot. (Matches the captured float64 to <1e-15 here.)
    bundle = _decode_bundle_key(bundle_key)
    acc = 0.0
    for on, w in zip(bundle, nu_row):
        if on:
            acc += float(w)
    return acc


def _independent_phi(
    arrays: dict[str, np.ndarray], agent_id: int, bundle_key: bytes
) -> np.ndarray:
    # phi recomputed straight from the RAW family arrays, not via the
    # fixture's feature callable — that callable is the very function whose
    # output priced_features_from stores, so it cannot check itself. The
    # layout is the family's canonical phi (toy: b * r_a;
    # qkp: [x_a . b, -b, 0.5 b'Qb]).
    b = _decode_bundle_key(bundle_key).astype(np.float64)
    if "observables" in arrays:  # toy
        r = np.asarray(arrays["observables"], dtype=np.float64)
        return b * r[agent_id]
    # qkp
    x = np.asarray(arrays["x"], dtype=np.float64)
    q = np.asarray(arrays["Q"], dtype=np.float64)
    m = x.shape[1]
    phi = np.empty(m + 2, dtype=np.float64)
    phi[0] = float(x[agent_id] @ b)
    phi[1 : m + 1] = -b
    phi[m + 1] = 0.5 * float(b @ (q @ b))
    return phi


def _independent_payoff(
    arrays: dict[str, np.ndarray],
    nu: np.ndarray,
    theta: np.ndarray,
    agent_id: int,
    bundle_key: bytes,
) -> float:
    # The demand identity every family oracle honours: payoff(a, d) =
    # phi_a(d) . theta + eps_a(d), evaluated at the theta the master priced
    # under. phi and eps are recomputed from the raw arrays, and theta is an
    # independent per-iteration snapshot rather than a captured payoff readback.
    phi = _independent_phi(arrays, agent_id, bundle_key)
    eps = _independent_eps(bundle_key, nu[agent_id])
    return float(np.asarray(phi, dtype=np.float64) @ np.asarray(theta) + eps)


# A per-iteration snapshot of the pricing master's theta and per-agent u, taken
# at contribute time. self._master.theta() / self._u at apply_step equal these
# (no re-solve between price and install), so one snapshot feeds the payoff, rc
# and admit-violation witnesses below.
_ThetaSnapshot = tuple[np.ndarray, dict[int, float]]


def _assert_it0_feature_field_witness(
    records: Sequence[StepRecord],
    nu: np.ndarray,
    arrays: dict[str, np.ndarray],
    theta_snapshots: dict[int, _ThetaSnapshot],
) -> None:
    """Captured gap/eps/payoff/phi at iteration 0 hold their fixture values.

    Both paths share ``priced_features_from``, so comparing them to each
    other says nothing about the values themselves. Every family demand is
    ``Demand.exact``, so ``gap`` is exactly ``0.0``; eps is the shock summed
    over the chosen items; phi is the family's raw feature layout recomputed
    from the decoded bundle key; payoff is the demand identity
    ``phi . theta + eps`` at the iteration-0 theta snapshot.
    """
    assert records, "no StepRecord captured — nothing to witness"
    assert records[0].priced_features, "iteration 0 captured no priced feature"
    theta0, _ = theta_snapshots[records[0].iteration]
    for pf in records[0].priced_features:
        phi_expected = _independent_phi(arrays, pf.agent_id, pf.bundle_key)
        got_phi = np.asarray(pf.phi, dtype=np.float64)
        assert got_phi.shape == phi_expected.shape and np.all(
            np.abs(got_phi - phi_expected) <= TOL
        ), (
            f"captured phi {got_phi!r} differs from the fixture phi"
            f" {phi_expected!r} at agent {pf.agent_id} by more than {TOL!r}"
        )
        payoff_expected = _independent_payoff(
            arrays, nu, theta0, pf.agent_id, pf.bundle_key
        )
        assert _close(pf.payoff, payoff_expected), (
            f"captured payoff {pf.payoff!r} differs from the demand-identity"
            f" payoff {payoff_expected!r} at agent {pf.agent_id} by more than"
            f" {TOL!r} — the capture is not storing the priced payoff"
        )
    for pf in records[0].priced_features:
        assert float(pf.gap) == 0.0, (
            f"captured gap is not the exact-demand 0.0 at agent {pf.agent_id}:"
            f" gap={pf.gap!r} — the capture is not storing the demand's gap"
        )
        expected_eps = _independent_eps(pf.bundle_key, nu[pf.agent_id])
        assert _close(pf.eps, expected_eps), (
            f"captured eps {pf.eps!r} differs from the fixture eps"
            f" {expected_eps!r} at agent {pf.agent_id} by more than {TOL!r}"
        )


def _family_nu(arrays: dict[str, np.ndarray]) -> np.ndarray:
    # The per-agent eps weight vector both families share: eps(agent, bundle)
    # = bundle . shocks[agent, 0, :]. S == 1, so column 0 is the whole shock.
    return np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]


def arrays_K(arrays: dict[str, np.ndarray]) -> int:
    # The parameter width phi rides in, derived from the raw arrays: toy phi is
    # per-item (b * r_a), qkp phi is [alpha, delta_1..M, lambda] = M + 2.
    if "observables" in arrays:  # toy
        return int(np.asarray(arrays["observables"]).shape[1])
    return int(np.asarray(arrays["x"]).shape[1]) + 2  # qkp


def _assert_rc_field_witness(
    records: Sequence[StepRecord],
    nu: np.ndarray,
    arrays: dict[str, np.ndarray],
    theta_snapshots: dict[int, _ThetaSnapshot],
) -> None:
    """Each captured ``PricedReducedCost.rc`` equals ``payoff - u_a``.

    payoff is recomputed from the raw arrays at the per-iteration theta
    snapshot. At least one non-zero rc is required, so the check sees a real
    value.
    """
    checked = 0
    for r in records:
        snapshot = theta_snapshots.get(r.iteration)
        if snapshot is None or not r.priced_reduced_costs:
            continue
        theta, u = snapshot
        for prc in r.priced_reduced_costs:
            payoff = _independent_payoff(
                arrays, nu, theta, prc.agent_id, prc.bundle_key
            )
            expected = payoff - float(u.get(prc.agent_id, 0.0))
            assert _close(prc.rc, expected), (
                f"captured reduced cost {prc.rc!r} differs from payoff - u_a"
                f" {expected!r} at agent {prc.agent_id}, iteration"
                f" {r.iteration}, by more than {TOL!r}"
            )
            if abs(expected) > TOL:
                checked += 1
    assert checked > 0, (
        "no non-zero reduced cost was witnessed"
    )


def _assert_admit_violation_field_witness(
    records: Sequence[StepRecord],
    nu: np.ndarray,
    arrays: dict[str, np.ndarray],
    theta_snapshots: dict[int, _ThetaSnapshot],
    received_snapshots: dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]],
) -> None:
    """Each captured ``AdmitViolation.violation`` equals the pre-admit signal
    ``phi . theta + eps - u_a``, recomputed from the received cut row and the
    theta/u snapshot. At least one non-zero violation is required.
    """
    checked = 0
    for r in records:
        if not r.admit_violations:
            continue
        theta_u = theta_snapshots.get(r.iteration)
        rows = received_snapshots.get(r.iteration)
        assert theta_u is not None and rows is not None, (
            f"no independent admit snapshot for iteration {r.iteration}"
        )
        theta, u = theta_u
        for av in r.admit_violations:
            key = (av.agent_id, av.bundle_key)
            assert key in rows, (
                f"captured admit candidate {key} at iteration {r.iteration} is"
                " absent from the received rows — the capture is storing a"
                " violation for a row the exchange never delivered"
            )
            phi, eps = rows[key]
            expected = float(
                np.asarray(phi, dtype=np.float64) @ theta
                + eps
                - float(u.get(av.agent_id, 0.0))
            )
            assert _close(av.violation, expected), (
                f"captured admit violation {av.violation!r} differs from"
                f" phi . theta + eps - u_a {expected!r} at agent {av.agent_id},"
                f" iteration {r.iteration}, by more than {TOL!r}"
            )
            if abs(expected) > TOL:
                checked += 1
    assert checked > 0, (
        "no non-zero admit violation was witnessed"
    )


def _installed_key_set(
    rows: Sequence[object],
) -> frozenset[tuple[int, bytes]]:
    return frozenset((r.agent_id, r.bundle_key) for r in rows)


def _assert_install_before_field_witness(
    records: Sequence[StepRecord],
    installed_snapshots: Sequence[Sequence[object]],
) -> None:
    """Captured ``installed_before`` matches the driver's own snapshots.

    ``WalkOutcome.installed_snapshots`` is a separate capture site — the
    driver's per-iteration ``extract_cuts()``, taken in ``_walk`` after
    ``update`` — which the ``InstallSnapshot`` field never touches. The
    master's installed set at the start of iteration ``i``'s add is its state
    at the end of ``i-1``, so ``installed_before`` must equal
    ``installed_snapshots[i-1]`` (empty at iteration 0: no warm start).
    """
    assert records, "no StepRecord captured — nothing to witness"
    assert len(installed_snapshots) == len(records), (
        "driver installed-snapshot stream and record stream disagree in length"
    )
    nonempty = 0
    for i, r in enumerate(records):
        assert r.install is not None, (
            f"iteration {i} captured no InstallSnapshot — install not exercised"
        )
        got = r.install.installed_before
        expected = (
            frozenset()
            if i == 0
            else _installed_key_set(installed_snapshots[i - 1])
        )
        assert got == expected, (
            f"captured installed_before at iteration {i} differs from the"
            f" driver's prior-iteration installed snapshot: |captured|="
            f"{len(got)}, |expected|={len(expected)},"
            f" captured-only {sorted(got - expected)[:2]},"
            f" expected-only {sorted(expected - got)[:2]}"
        )
        if got:
            nonempty += 1
    assert nonempty > 0, (
        "every captured installed_before was empty — the pre-install key set is"
        " never populated"
    )


def _assert_admitted_field_witness(
    records: Sequence[StepRecord],
    theta_snapshots: dict[int, _ThetaSnapshot],
    received_snapshots: dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]],
    *,
    epsilon: float = 0.0,
    require_strict_subset: bool = False,
) -> None:
    """Captured ``InstallSnapshot.admitted`` matches the admit rule exactly.

    Reconstruct the whole admitted set from ``SlackThreshold(epsilon)``: a
    received candidate is admitted iff its pre-admit violation
    ``phi . theta + eps - u_a`` exceeds ``epsilon``, evaluated from the
    received row's own ``(phi, eps)`` and the theta/u snapshot — never read
    back from the record. The captured set must equal that reconstruction,
    be a subset of the received candidates, and be disjoint from
    ``installed_before`` under the fresh-admit policy.

    Under ``epsilon=0.0`` every received row clears the bar and the
    reconstruction degenerates to ``frozenset(received)``, which says nothing
    about the threshold. ``require_strict_subset`` (paired with a positive
    ``epsilon``) demands at least one iteration whose admitted set is a
    strict subset of the received candidates, so the threshold clause has to
    exclude a real row.
    """
    checked = 0
    strict_subset = 0
    for r in records:
        if r.install is None:
            continue
        theta_u = theta_snapshots.get(r.iteration)
        rows = received_snapshots.get(r.iteration)
        assert theta_u is not None and rows is not None, (
            f"no independent admit snapshot for iteration {r.iteration}"
        )
        theta, u = theta_u
        expected = frozenset(
            key
            for key, (phi, eps) in rows.items()
            if float(
                np.asarray(phi, dtype=np.float64) @ theta
                + eps
                - float(u.get(key[0], 0.0))
            )
            > epsilon
        )
        assert r.install.admitted <= frozenset(rows), (
            f"captured admitted at iteration {r.iteration} holds a key the"
            " exchange never delivered:"
            f" {sorted(r.install.admitted - frozenset(rows))[:2]}"
        )
        assert r.install.admitted.isdisjoint(r.install.installed_before), (
            f"captured admitted overlaps installed_before at iteration"
            f" {r.iteration}: {sorted(r.install.admitted & r.install.installed_before)[:2]}"
            " — the fresh-admit policy never re-admits an installed cut"
        )
        assert r.install.admitted == expected, (
            f"captured admitted at iteration {r.iteration} differs from the"
            f" independently reconstructed admit set (violation > {epsilon!r}):"
            f" captured-only {sorted(r.install.admitted - expected)[:2]},"
            f" expected-only {sorted(expected - r.install.admitted)[:2]}"
        )
        if r.install.admitted:
            checked += 1
        if r.install.admitted < frozenset(rows):
            strict_subset += 1
    assert checked > 0, (
        "no non-empty admitted set was witnessed"
    )
    if require_strict_subset:
        assert strict_subset > 0, (
            "no iteration admitted a strict subset of the received candidates —"
            " the admit threshold never excluded a row, so an admit-everything"
            " bug cannot be distinguished (raise the epsilon policy so some"
            " received violation falls below the bar)"
        )


def _assert_continuous_keyed(
    label: str,
    left: dict[tuple[int, bytes], float],
    right: dict[tuple[int, bytes], float],
) -> None:
    # Discrete: the key sets (agent-id + bundle-key domains) must be identical.
    assert set(left) == set(right), (
        f"{label}: the (agent_id, bundle_key) domain differs between the two"
        f" paths — left-only {sorted(set(left) - set(right))[:3]}, right-only"
        f" {sorted(set(right) - set(left))[:3]}"
    )
    # Continuous: each value within TOL.
    for key, lv in left.items():
        rv = right[key]
        assert _close(lv, rv), (
            f"{label}: continuous drift {abs(lv - rv)!r} > {TOL!r} at"
            f" agent {key[0]}"
        )


def _assert_records_equivalent(
    a: Sequence[StepRecord], b: Sequence[StepRecord]
) -> None:
    """Field-by-field law over two aligned single-rank record streams.

    Used for the features axis (both walks SerialTransport, one rank holds
    every field). Discrete fields ==, continuous within TOL, per iteration.
    """
    _assert_same_shape(a, b)
    for ra, rb in zip(a, b):
        # priced reduced costs: domain identical (incl. sub-threshold), rc TOL.
        _assert_continuous_keyed("priced_reduced_costs", _rc_map(ra), _rc_map(rb))
        # admit violations: candidate key set identical, violation TOL.
        _assert_continuous_keyed("admit_violations", _admit_map(ra), _admit_map(rb))
        # purge inputs: installed key set identical; dual/slack TOL (None==None).
        pa, pb = _purge_map(ra), _purge_map(rb)
        assert set(pa) == set(pb), "purge_inputs: installed key set differs"
        for key, (da, sa) in pa.items():
            db, sb = pb[key]
            assert (da is None) == (db is None), f"purge dual None-ness at {key}"
            assert (sa is None) == (sb is None), f"purge slack None-ness at {key}"
            if da is not None:
                assert _close(da, db), f"purge dual drift at {key}"
            if sa is not None:
                assert _close(sa, sb), f"purge slack drift at {key}"
        # install: the pre-call installed key set + admitted key set identical
        # (discrete; a one-bit key flip is a different install, never a drift).
        if ra.install is None or rb.install is None:
            assert ra.install is None and rb.install is None
        else:
            assert ra.install.installed_before == rb.install.installed_before
            assert ra.install.admitted == rb.install.admitted
        # aggregate: raw within TOL; bytes byte-identical (the aggregate row key).
        assert (ra.aggregate_raw is None) == (rb.aggregate_raw is None)
        if ra.aggregate_raw is not None:
            assert _close(ra.aggregate_raw, rb.aggregate_raw)
        assert ra.aggregate_bytes == rb.aggregate_bytes
        # priced features: domain identical; payoff/gap/eps TOL; phi via bytes below.
        fa, fb = _feature_map(ra), _feature_map(rb)
        assert set(fa) == set(fb), "priced_features: domain differs"
        for key, (pay_a, gap_a, phi_a, eps_a) in fa.items():
            pay_b, gap_b, phi_b, eps_b = fb[key]
            assert _close(pay_a, pay_b), f"payoff drift at {key}"
            assert _close(gap_a, gap_b), f"gap drift at {key}"
            assert _close(eps_a, eps_b), f"eps drift at {key}"
            # phi bytes byte-identical: this is the strongest check — it pins
            # the exact zero-mask and every coefficient at once.
            assert phi_a == phi_b, f"phi bytes differ at {key}"


def _union_shard_fields(
    records_per_rank: Sequence[Sequence[StepRecord]],
) -> list[StepRecord]:
    """Fold a per-rank record stream into one full-domain stream per iteration.

    The shard axis splits agents across ranks: a per-shard field
    (priced_reduced_costs / priced_features) lands on the owning rank only,
    while the root-only fields (admit/purge/install/aggregate) land whole on
    rank 0. This reconstructs the full-domain record for each iteration by
    unioning the per-shard fields across ranks and taking the root's root-only
    fields, so the result is directly comparable to a SerialTransport stream.
    """
    n_iters = len(records_per_rank[0])
    for rank_stream in records_per_rank:
        assert len(rank_stream) == n_iters, (
            "ranks disagree on iteration count — the walks diverged"
        )
    folded: list[StepRecord] = []
    for it in range(n_iters):
        rc: list = []
        pf: list = []
        for rank_stream in records_per_rank:
            rec = rank_stream[it]
            assert rec.iteration == it
            rc.extend(rec.priced_reduced_costs)
            pf.extend(rec.priced_features)
        root = records_per_rank[0][it]  # rank 0 holds the root-only fields
        folded.append(
            StepRecord(
                iteration=it,
                priced_reduced_costs=tuple(rc),
                admit_violations=root.admit_violations,
                purge_inputs=root.purge_inputs,
                install=root.install,
                aggregate_raw=root.aggregate_raw,
                aggregate_bytes=root.aggregate_bytes,
                priced_features=tuple(pf),
            )
        )
    return folded


# --- exercised-field guards --------------------------------------------------


def _assert_nslack_records_exercised(records: Sequence[StepRecord]) -> None:
    """Every NSlack field is actually populated, and the events fired."""
    assert records, "no StepRecord captured — the walk never iterated"
    # priced_reduced_costs must cover every priced agent, not only the rows
    # that cleared the emit threshold: require at least one captured
    # rc <= tolerance that shipped nothing.
    total_priced = sum(len(r.priced_reduced_costs) for r in records)
    sub_threshold = sum(
        1
        for r in records
        for prc in r.priced_reduced_costs
        if prc.rc <= TOLERANCE
    )
    assert total_priced > 0, "priced_reduced_costs was never populated"
    assert sub_threshold > 0, (
        "priced_reduced_costs never captured a sub-threshold agent — only"
        " rows that shipped were recorded"
    )
    # At least one admit candidate, one purge-input key, one install event.
    assert sum(len(r.admit_violations) for r in records) > 0, (
        "admit stream was never populated"
    )
    assert sum(len(r.purge_inputs) for r in records) > 0, (
        "purge stream was never populated"
    )
    fresh = sum(
        len(r.install.admitted - r.install.installed_before)
        for r in records
        if r.install is not None
    )
    assert fresh > 0, "no fresh cut was ever installed"
    # At least one purge-input must carry a real (non-None) dual reading.
    assert any(
        pi.dual is not None for r in records for pi in r.purge_inputs
    ), "every purge-input dual was None — the purge signal is never present"


def _assert_oneslack_records_exercised(records: Sequence[StepRecord]) -> None:
    assert records, "no StepRecord captured — the walk never iterated"
    assert all(r.aggregate_raw is not None for r in records), (
        "an OneSlack record is missing its aggregate_raw"
    )
    assert all(r.aggregate_bytes is not None for r in records), (
        "an OneSlack record is missing its aggregate_bytes"
    )
    assert sum(len(r.priced_features) for r in records) > 0, (
        "priced_features was never populated"
    )


# --- the gate: NSlack ---------------------------------------------------------


def _nslack_serial(
    arrays, problem, features, backend, *, capture_installed=False
) -> WalkOutcome:
    return run_walk(
        arrays,
        _with_features(problem, features),
        NSlack,
        SerialTransport(),
        backend=backend,
        cut_policy=_nslack_policy(),
        capture_steprecords=True,
        capture_installed=capture_installed,
    )


def _nslack_cluster(
    arrays, problem, features, backend, size
) -> list[StepRecord]:
    results = LocalCluster(size).run(
        lambda transport: run_walk(
            arrays,
            _with_features(problem, features),
            NSlack,
            transport,
            backend=backend,
            cut_policy=_nslack_policy(),
            capture_steprecords=True,
        )
    )
    return _union_shard_fields([out.step_records for out in results])


#: What one instrumented reference walk yields: per-iteration purge duals,
#: theta/u snapshots, and the exchanged rows' (phi, eps) — reads the records
#: never touch, enough to check the purge-dual, payoff, rc and
#: admit-violation fields.
_NSlackSnapshots = tuple[
    dict[int, dict[tuple[int, bytes], float]],  # purge duals at _purge time
    dict[int, _ThetaSnapshot],  # theta + u at price/install time
    dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]],  # received rows
]


def _nslack_serial_with_snapshots(
    arrays, problem, features, backend
) -> tuple[WalkOutcome, _NSlackSnapshots]:
    """A serial NSlack walk plus the per-iteration reference snapshots.

    At every ``_purge`` the master is still open, so we snapshot its
    published per-cut duals via ``master.dual_values()`` — a different
    accessor than the ``cut_readings().dual_map()`` the capture consumes. At
    every ``contribute`` we snapshot the pricing master's ``theta`` and
    per-agent ``u``, and at every ``apply_step`` the exchanged rows'
    ``(phi, eps)``. ``self._master.theta()`` / ``self._u`` at ``apply_step``
    equal the ``contribute`` snapshot (no re-solve between price and
    install), so one theta/u map serves the payoff, rc and admit checks. The
    master closes when the walk returns, hence the mid-walk snapshots.
    """
    duals: dict[int, dict[tuple[int, bytes], float]] = {}
    theta_u: dict[int, _ThetaSnapshot] = {}
    received: dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]] = {}
    original_purge = _nslack_module.NSlack._purge
    original_contribute = _nslack_module.NSlack.contribute
    original_apply = _nslack_module.NSlack.apply_step

    def _snapshotting_purge(self, policy, profile, installed, pending=None):
        duals.setdefault(self._iteration, dict(self._master.dual_values()))
        return original_purge(self, policy, profile, installed, pending)

    def _snapshotting_contribute(self, demands):
        theta_u.setdefault(
            self._iteration,
            (
                np.asarray(self._theta, dtype=np.float64).copy(),
                {int(a): float(v) for a, v in self._u.items()},
            ),
        )
        return original_contribute(self, demands)

    def _snapshotting_apply(self, install_payload):
        rows = install_payload
        received.setdefault(
            self._iteration,
            {
                (row.agent_id, row.bundle_key): (
                    np.asarray(row.phi, dtype=np.float64).copy(),
                    float(row.epsilon),
                )
                for row in rows
            },
        )
        return original_apply(self, install_payload)

    _nslack_module.NSlack._purge = _snapshotting_purge
    _nslack_module.NSlack.contribute = _snapshotting_contribute
    _nslack_module.NSlack.apply_step = _snapshotting_apply
    try:
        # capture_installed rides along so the same instrumented reference
        # also feeds the install-before check (which compares against the
        # driver's own per-iteration extract_cuts() snapshot).
        outcome = _nslack_serial(
            arrays, problem, features, backend, capture_installed=True
        )
    finally:
        _nslack_module.NSlack._purge = original_purge
        _nslack_module.NSlack.contribute = original_contribute
        _nslack_module.NSlack.apply_step = original_apply
    return outcome, (duals, theta_u, received)


def _assert_purge_dual_field_witness(
    records: Sequence[StepRecord],
    published_duals: dict[int, dict[tuple[int, bytes], float]],
) -> None:
    """Captured ``PurgeInput.dual`` matches the master's own duals.

    The reference is ``master.dual_values()`` read at purge time, not read
    back from the record. The loop runs over the master's key set, not the
    captured non-None entries: every cut the master published a dual for
    must be captured with a non-None matching value, so a real reading
    silently dropped to ``None`` also fails. A ``0.0`` reading proves
    nothing about the value path, so at least one non-zero dual is required.
    """
    checked = 0
    for r in records:
        if not r.purge_inputs:
            continue
        published = published_duals.get(r.iteration)
        assert published is not None, (
            f"no independent dual snapshot for purge iteration {r.iteration}"
        )
        captured = {(pi.agent_id, pi.bundle_key): pi.dual for pi in r.purge_inputs}
        for key, expected in published.items():
            assert key in captured, (
                f"master published a dual for cut {key} at iteration"
                f" {r.iteration} the capture never recorded — a purge input was"
                " dropped from the record"
            )
            assert captured[key] is not None, (
                f"cut {key} at iteration {r.iteration} was captured as a None"
                " purge dual, but the master's last solve published a real"
                f" dual {expected!r} for it — a reading was silently nulled"
            )
            assert _close(captured[key], expected), (
                f"captured purge dual {captured[key]!r} differs from the"
                f" master's published dual {expected!r} at agent {key[0]},"
                f" iteration {r.iteration}, by more than {TOL!r}"
            )
            if abs(expected) > TOL:
                checked += 1
    assert checked > 0, (
        "no non-zero purge dual was witnessed"
    )


# Theta, per-agent u, and each installed cut's (phi, eps) — everything the
# looseness recompute below needs, keyed by iteration.
_SlackSnapshot = tuple[
    np.ndarray,  # theta at purge time
    dict[int, float],  # u_values at purge time
    dict[tuple[int, bytes], tuple[np.ndarray, float]],  # cut -> (phi, eps)
]


def _nslack_serial_with_slack_snapshots(
    arrays, problem, features, backend
) -> tuple[WalkOutcome, dict[int, _SlackSnapshot]]:
    """A serial NSlack walk under a SLACK-reading policy, plus snapshots.

    Drives ``_nslack_slack_policy`` (a ``SlackStrip`` retirement stage) so
    the captured ``PurgeInput.slack`` is populated at all — under the default
    ``PurgeInactive`` policy every captured slack is ``None``. At each
    ``_purge`` the master is open, so we snapshot ``theta()``, ``u_values()``
    and the installed rows' ``(phi, eps)`` — accessors the slack capture
    (which reads ``cut_readings(slack=True)`` off
    ``getSolution().row_value``) never touches.
    """
    snapshots: dict[int, _SlackSnapshot] = {}
    original = _nslack_module.NSlack._purge

    def _snapshotting_purge(self, policy, profile, installed, pending=None):
        master = self._master
        theta = np.asarray(master.theta(), dtype=np.float64)
        u = {int(a): float(v) for a, v in master.u_values().items()}
        rows = {
            (row.agent_id, row.bundle_key): (
                np.asarray(row.phi, dtype=np.float64),
                float(row.epsilon),
            )
            for row in master.extract_cuts()
        }
        snapshots.setdefault(self._iteration, (theta, u, rows))
        return original(self, policy, profile, installed, pending)

    _nslack_module.NSlack._purge = _snapshotting_purge
    try:
        outcome = run_walk(
            arrays,
            _with_features(problem, features),
            NSlack,
            SerialTransport(),
            backend=backend,
            cut_policy=_nslack_slack_policy(),
            capture_steprecords=True,
        )
    finally:
        _nslack_module.NSlack._purge = original
    return outcome, snapshots


def _assert_purge_slack_field_witness(
    records: Sequence[StepRecord],
    slack_snapshots: dict[int, _SlackSnapshot],
) -> None:
    """Captured ``PurgeInput.slack`` equals the recomputed row looseness.

    ``slack(a, d) = u_a - (phi_a(d) . theta + eps_a(d))`` — binding ~0,
    looser rows larger — evaluated from ``theta()`` / ``u_values()`` / the
    cut ``(phi, eps)``, a structurally distinct path from the backend's
    ``getSolution().row_value - eps`` read the capture consumes. The loop
    runs over the snapshot's installed rows (the ``extract_cuts()`` set at
    purge time; every one holds a reading), not the captured non-None
    entries, so a real reading silently dropped to ``None`` also fails. At
    least one non-zero looseness is required, so the check sees a real value.
    """
    checked = 0
    for r in records:
        if not r.purge_inputs:
            continue
        snapshot = slack_snapshots.get(r.iteration)
        assert snapshot is not None, (
            f"no independent slack snapshot for purge iteration {r.iteration}"
        )
        theta, u, rows = snapshot
        captured = {
            (pi.agent_id, pi.bundle_key): pi.slack for pi in r.purge_inputs
        }
        for key, (phi, eps) in rows.items():
            assert key in captured, (
                f"master holds installed cut {key} at iteration {r.iteration}"
                " the capture never recorded — a purge input was dropped"
            )
            expected = float(u.get(key[0], 0.0)) - (float(phi @ theta) + eps)
            assert captured[key] is not None, (
                f"cut {key} at iteration {r.iteration} was captured as a None"
                " purge slack, but the master holds it in its last-solved"
                f" relaxation with looseness {expected!r} — a reading was nulled"
            )
            assert _close(captured[key], expected), (
                f"captured purge slack {captured[key]!r} differs from the"
                f" recomputed looseness {expected!r} at agent {key[0]},"
                f" iteration {r.iteration}, by more than {TOL!r}"
            )
            if abs(expected) > TOL:
                checked += 1
    assert checked > 0, (
        "no non-trivially-loose purge slack was witnessed"
    )


def _assert_nslack_slack_records_exercised(records: Sequence[StepRecord]) -> None:
    """The slack-reading walk actually populated ``PurgeInput.slack``.

    Under the default policy every reading is ``None``; the slack policy
    must produce real readings, including at least one non-zero looseness.
    """
    assert records, "no StepRecord captured — the walk never iterated"
    assert sum(len(r.purge_inputs) for r in records) > 0, (
        "purge stream was never populated"
    )
    non_none = sum(
        1 for r in records for pi in r.purge_inputs if pi.slack is not None
    )
    assert non_none > 0, (
        "every captured purge slack was None — the SlackStrip policy did not"
        " populate the slack field"
    )
    non_zero = sum(
        1
        for r in records
        for pi in r.purge_inputs
        if pi.slack is not None and abs(pi.slack) > TOL
    )
    assert non_zero > 0, (
        "every captured purge slack read ~0.0 — no loose cut was ever priced"
    )


def _nslack_slack_cluster(
    arrays, problem, features, backend, size
) -> list[StepRecord]:
    results = LocalCluster(size).run(
        lambda transport: run_walk(
            arrays,
            _with_features(problem, features),
            NSlack,
            transport,
            backend=backend,
            cut_policy=_nslack_slack_policy(),
            capture_steprecords=True,
        )
    )
    return _union_shard_fields([out.step_records for out in results])


def _gate_nslack(arrays, problem, batch_only_map, backend) -> None:
    # Reference walk: per-agent features, serial shard, instrumented so the
    # master's published duals, its theta/u, and the exchanged rows are
    # snapshot mid-walk. Both capture paths share the capture code, so the
    # field checks below compare against those snapshots, not the other path.
    per_agent, (published_duals, theta_snapshots, received_snapshots) = (
        _nslack_serial_with_snapshots(
            arrays, problem, problem.features, backend
        )
    )
    nu = _family_nu(arrays)
    _assert_nslack_records_exercised(per_agent.step_records)
    _assert_it0_feature_field_witness(
        per_agent.step_records, nu, arrays, theta_snapshots
    )
    _assert_rc_field_witness(
        per_agent.step_records, nu, arrays, theta_snapshots
    )
    _assert_admit_violation_field_witness(
        per_agent.step_records, nu, arrays, theta_snapshots, received_snapshots
    )
    _assert_purge_dual_field_witness(
        per_agent.step_records, published_duals
    )
    _assert_install_before_field_witness(
        per_agent.step_records, per_agent.installed_snapshots
    )
    _assert_admitted_field_witness(
        per_agent.step_records, theta_snapshots, received_snapshots
    )

    # Axis 1 — features: batched features, same serial shard. Must match the
    # per-agent reference field-by-field (discrete ==, continuous <= TOL).
    batched = _nslack_serial(arrays, problem, batch_only_map, backend)
    _assert_records_equivalent(per_agent.step_records, batched.step_records)

    # Axis 2 — shard: per-agent features, interleaved LocalCluster, folded
    # back to one full-domain stream.
    folded = _nslack_cluster(arrays, problem, problem.features, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded)

    # Both axes at once.
    folded_batched = _nslack_cluster(arrays, problem, batch_only_map, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded_batched)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_nslack_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    _gate_nslack(toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend)


@needs_gurobi
def test_nslack_wholesale_capture_qkp() -> None:
    # QKP enumeration is exact only under the gurobi-solved master the
    # references were captured against.
    qkp = load_qkp()
    _gate_nslack(qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi")


# --- the gate: NSlack strict-subset admit -------------------------------------
#
# Under the default epsilon=0.0 admit policy every received candidate is
# admitted (each already cleared the emit threshold), so admitted == received
# and the reconstruction has no threshold to check. This gate drives
# SlackThreshold(1.0) so the admit filter genuinely rejects the
# weakly-violated received rows and the admitted set is a strict subset.


def _nslack_admit_subset_serial(
    arrays, problem, features, backend
) -> tuple[WalkOutcome, dict[int, _ThetaSnapshot], dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]]]:
    """A serial NSlack walk under the positive-epsilon admit policy.

    Snapshots the pricing master's ``theta`` / per-agent ``u`` at
    ``contribute`` and the exchanged rows' ``(phi, eps)`` at ``apply_step`` —
    the two reads the admitted-set reconstruction needs, neither of which
    touches the captured ``InstallSnapshot.admitted``. No re-solve happens
    between price and install, so the one theta/u map feeds the whole
    reconstruction.
    """
    theta_u: dict[int, _ThetaSnapshot] = {}
    received: dict[int, dict[tuple[int, bytes], tuple[np.ndarray, float]]] = {}
    original_contribute = _nslack_module.NSlack.contribute
    original_apply = _nslack_module.NSlack.apply_step

    def _snapshotting_contribute(self, demands):
        theta_u.setdefault(
            self._iteration,
            (
                np.asarray(self._theta, dtype=np.float64).copy(),
                {int(a): float(v) for a, v in self._u.items()},
            ),
        )
        return original_contribute(self, demands)

    def _snapshotting_apply(self, install_payload):
        received.setdefault(
            self._iteration,
            {
                (row.agent_id, row.bundle_key): (
                    np.asarray(row.phi, dtype=np.float64).copy(),
                    float(row.epsilon),
                )
                for row in install_payload
            },
        )
        return original_apply(self, install_payload)

    _nslack_module.NSlack.contribute = _snapshotting_contribute
    _nslack_module.NSlack.apply_step = _snapshotting_apply
    try:
        outcome = run_walk(
            arrays,
            _with_features(problem, features),
            NSlack,
            SerialTransport(),
            backend=backend,
            cut_policy=_nslack_admit_subset_policy(),
            capture_steprecords=True,
        )
    finally:
        _nslack_module.NSlack.contribute = original_contribute
        _nslack_module.NSlack.apply_step = original_apply
    return outcome, theta_u, received


def _nslack_admit_subset_cluster(
    arrays, problem, features, backend, size
) -> list[StepRecord]:
    results = LocalCluster(size).run(
        lambda transport: run_walk(
            arrays,
            _with_features(problem, features),
            NSlack,
            transport,
            backend=backend,
            cut_policy=_nslack_admit_subset_policy(),
            capture_steprecords=True,
        )
    )
    return _union_shard_fields([out.step_records for out in results])


def _gate_nslack_admit_subset(arrays, problem, batch_only_map, backend) -> None:
    per_agent, theta_snapshots, received_snapshots = _nslack_admit_subset_serial(
        arrays, problem, problem.features, backend
    )
    _assert_nslack_records_exercised(per_agent.step_records)
    _assert_admitted_field_witness(
        per_agent.step_records,
        theta_snapshots,
        received_snapshots,
        epsilon=_ADMIT_SUBSET_EPSILON,
        require_strict_subset=True,
    )

    # Feature axis, shard axis, then both, as in _gate_nslack.
    batched = run_walk(
        arrays,
        _with_features(problem, batch_only_map),
        NSlack,
        SerialTransport(),
        backend=backend,
        cut_policy=_nslack_admit_subset_policy(),
        capture_steprecords=True,
    )
    _assert_records_equivalent(per_agent.step_records, batched.step_records)

    folded = _nslack_admit_subset_cluster(
        arrays, problem, problem.features, backend, 2
    )
    _assert_records_equivalent(per_agent.step_records, folded)

    folded_batched = _nslack_admit_subset_cluster(
        arrays, problem, batch_only_map, backend, 2
    )
    _assert_records_equivalent(per_agent.step_records, folded_batched)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_nslack_admit_subset_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    _gate_nslack_admit_subset(
        toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend
    )


@needs_gurobi
def test_nslack_admit_subset_wholesale_capture_qkp() -> None:
    qkp = load_qkp()
    _gate_nslack_admit_subset(
        qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi"
    )


# --- the gate: NSlack purge SLACK branch --------------------------------------
#
# The default NSlack policy (PurgeInactive) reads the dual, not the slack, so
# every captured PurgeInput.slack is None and the comparator's slack branch
# (_close(sa, sb)) never runs. This gate drives a SlackStrip retirement stage
# so the slack field is populated, then runs the full feature/shard comparator
# plus the row-looseness recompute.


def _gate_nslack_slack(arrays, problem, batch_only_map, backend) -> None:
    per_agent, slack_snapshots = _nslack_serial_with_slack_snapshots(
        arrays, problem, problem.features, backend
    )
    _assert_nslack_slack_records_exercised(per_agent.step_records)
    _assert_purge_slack_field_witness(per_agent.step_records, slack_snapshots)

    # Axis 1 — features: batched features, same serial shard. The whole record
    # (slack branch included) matches the per-agent reference field-by-field.
    batched = run_walk(
        arrays,
        _with_features(problem, batch_only_map),
        NSlack,
        SerialTransport(),
        backend=backend,
        cut_policy=_nslack_slack_policy(),
        capture_steprecords=True,
    )
    _assert_records_equivalent(per_agent.step_records, batched.step_records)

    folded = _nslack_slack_cluster(arrays, problem, problem.features, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded)

    folded_batched = _nslack_slack_cluster(
        arrays, problem, batch_only_map, backend, 2
    )
    _assert_records_equivalent(per_agent.step_records, folded_batched)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_nslack_purge_slack_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    _gate_nslack_slack(
        toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend
    )


@needs_gurobi
def test_nslack_purge_slack_wholesale_capture_qkp() -> None:
    qkp = load_qkp()
    _gate_nslack_slack(
        qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi"
    )


# --- the gate: OneSlack -------------------------------------------------------


def _oneslack_serial(arrays, problem, features, backend) -> WalkOutcome:
    return run_walk(
        arrays,
        _with_features(problem, features),
        OneSlack,
        SerialTransport(),
        backend=backend,
        capture_steprecords=True,
    )


def _oneslack_cluster(arrays, problem, features, backend, size) -> list[StepRecord]:
    results = LocalCluster(size).run(
        lambda transport: run_walk(
            arrays,
            _with_features(problem, features),
            OneSlack,
            transport,
            backend=backend,
            capture_steprecords=True,
        )
    )
    return _union_shard_fields([out.step_records for out in results])


def _oneslack_serial_with_theta_snapshots(
    arrays, problem, features, backend
) -> tuple[WalkOutcome, dict[int, _ThetaSnapshot]]:
    """A serial OneSlack walk plus a per-iteration theta/aggregate-u snapshot.

    ``finalise`` computes ``aggregate_raw`` from ``self._theta`` / ``self._u``
    at the current, pre-update master solution. Snapshot both there — a read
    the aggregate and priced-feature captures never touch. ``u`` rides the
    snapshot as a single float keyed by the synthetic aggregate agent 0.
    """
    snapshots: dict[int, _ThetaSnapshot] = {}
    from combrum.formulations import oneslack as _oneslack_module

    original = _oneslack_module.OneSlack.finalise

    def _snapshotting_finalise(self, reduced):
        snapshots.setdefault(
            self._iteration,
            (np.asarray(self._theta, dtype=np.float64).copy(), {0: float(self._u)}),
        )
        return original(self, reduced)

    _oneslack_module.OneSlack.finalise = _snapshotting_finalise
    try:
        outcome = _oneslack_serial(arrays, problem, features, backend)
    finally:
        _oneslack_module.OneSlack.finalise = original
    return outcome, snapshots


def _bitmatch_eps(bundle_key: bytes, nu_row: np.ndarray) -> float:
    # eps = bundle . nu via the same vectorised float64 dot the family feature
    # map uses. _independent_eps deliberately sums differently (its value is
    # only compared to TOL); the aggregate SHA-256 is byte-exact, so this
    # reconstruction must match combrum's arithmetic bit-for-bit — a 1-ULP
    # eps difference flips the digest.
    b = _decode_bundle_key(bundle_key).astype(np.float64)
    return float(b @ np.asarray(nu_row, dtype=np.float64))


def _reconstruct_aggregate(
    rec: StepRecord, nu: np.ndarray, arrays: dict[str, np.ndarray], K: int
) -> np.ndarray:
    # The aggregate row (phi_agg | eps_agg), rebuilt from the raw arrays
    # exactly as the reduction kernel forms it: one (phi | eps) row per
    # priced agent (weight 1.0, the walk's agent_weights), stacked in
    # ascending agent-id order and reduced with np.add.reduce — the contract
    # canonical_sum documents, reproduced here rather than called, so the
    # reconstruction is bitwise identical to reduced.aggregate and its
    # SHA-256 matches the true row key.
    pfs = sorted(rec.priced_features, key=lambda pf: pf.agent_id)
    rows = np.empty((len(pfs), K + 1), dtype=np.float64)
    for i, pf in enumerate(pfs):
        rows[i, :K] = _independent_phi(arrays, pf.agent_id, pf.bundle_key)
        rows[i, K] = _bitmatch_eps(pf.bundle_key, nu[pf.agent_id])
    return np.add.reduce(rows, axis=0)


def _assert_aggregate_field_witness(
    records: Sequence[StepRecord],
    nu: np.ndarray,
    arrays: dict[str, np.ndarray],
    theta_snapshots: dict[int, _ThetaSnapshot],
) -> None:
    """Captured OneSlack aggregate fields hold the reconstructed row.

    ``raw = phi_agg . theta + eps_agg - u`` at the theta/u snapshot, and
    ``bytes = sha256(concatenate([phi_agg, [eps_agg]]).tobytes())`` computed
    test-side over the reconstructed row — never read back from the record.
    At least one non-zero raw is required.
    """
    K = arrays_K(arrays)
    checked = 0
    for r in records:
        assert r.aggregate_raw is not None and r.aggregate_bytes is not None, (
            f"iteration {r.iteration} carried no aggregate — nothing to witness"
        )
        theta, u = theta_snapshots[r.iteration]
        agg = _reconstruct_aggregate(r, nu, arrays, K)
        phi_agg = agg[:K]
        eps_agg = float(agg[K])
        expected_raw = (
            float(np.multiply(phi_agg, theta).sum(dtype=np.float64))
            + eps_agg
            - float(u.get(0, 0.0))
        )
        assert _close(r.aggregate_raw, expected_raw), (
            f"captured aggregate_raw {r.aggregate_raw!r} differs from the"
            f" reconstructed phi_agg . theta + eps_agg - u {expected_raw!r} at"
            f" iteration {r.iteration} by more than {TOL!r}"
        )
        payload = np.ascontiguousarray(
            np.concatenate([phi_agg, [eps_agg]]), dtype=np.float64
        )
        expected_bytes = hashlib.sha256(payload.tobytes()).digest()
        assert r.aggregate_bytes == expected_bytes, (
            f"captured aggregate_bytes at iteration {r.iteration} differ from"
            " the test-side SHA-256 over the reconstructed [phi_agg, eps_agg] —"
            " the aggregate row key is not the true aggregate digest"
        )
        if abs(expected_raw) > TOL:
            checked += 1
    assert checked > 0, (
        "no non-zero aggregate_raw was witnessed"
    )


def _gate_oneslack(arrays, problem, batch_only_map, backend) -> None:
    per_agent, theta_snapshots = _oneslack_serial_with_theta_snapshots(
        arrays, problem, problem.features, backend
    )
    nu = _family_nu(arrays)
    _assert_oneslack_records_exercised(per_agent.step_records)
    _assert_it0_feature_field_witness(
        per_agent.step_records, nu, arrays, theta_snapshots
    )
    _assert_aggregate_field_witness(
        per_agent.step_records, nu, arrays, theta_snapshots
    )

    # Axis 1 — features: the aggregate raw within TOL and SHA-256 aggregate bytes
    # byte-identical across the per-agent and batched paths.
    batched = _oneslack_serial(arrays, problem, batch_only_map, backend)
    _assert_records_equivalent(per_agent.step_records, batched.step_records)

    # Axis 2 — shard: the interleaved sum reduction must land the byte-identical
    # aggregate (sum_reproducible's bitwise contract), so the folded stream
    # matches the serial reference.
    folded = _oneslack_cluster(arrays, problem, problem.features, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded)

    folded_batched = _oneslack_cluster(arrays, problem, batch_only_map, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded_batched)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_oneslack_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    _gate_oneslack(
        toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend
    )


@needs_gurobi
def test_oneslack_wholesale_capture_qkp() -> None:
    qkp = load_qkp()
    _gate_oneslack(
        qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi"
    )


# --- the gate: the priced-demand gap field ------------------------------------
#
# Every family oracle prices Demand.exact, so every captured
# PricedFeature.gap is exactly 0.0 — a constant that cannot distinguish
# "stores demand.gap" from "stores 0.0". This gate wraps the oracle to stamp
# each priced demand with a KNOWN nonzero certified gap (a pure function of
# the agent id, never a read of demand.gap). The bundle and payoff stay
# byte-identical — the gap is a certificate consumed only by the reporting
# tally, so the walk trajectory is unchanged and the feature/shard comparator
# stays valid.


def _injected_gap(agent_id: int) -> float:
    # A distinct positive gap per agent, injected by the wrapper below and
    # recomputed here as the expected value.
    return 0.5 + 0.25 * int(agent_id)


class _InexactGapOracle(Oracle):
    """Wraps a base oracle, re-stamping each priced demand's certified gap.

    The chosen bundle and achieved payoff are the base oracle's (byte-identical,
    so the walk prices the same rows and converges on the same shape); only the
    certified optimality ``gap`` is replaced by :func:`_injected_gap`, a known
    positive value per agent. Both the per-agent ``price`` and the batched
    ``price_batch`` paths are covered so the same gap flows whichever the
    formulation resolves.
    """

    def __init__(self, base: Oracle) -> None:
        self._base = base

    def setup(self, transport, local_ids) -> None:
        self._base.setup(transport, local_ids)

    def teardown(self) -> None:
        self._base.teardown()

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        demand = self._base.price(theta, int(agent_id))
        return Demand.inexact(
            bundle=demand.bundle,
            payoff=demand.payoff,
            gap=_injected_gap(agent_id),
        )

    def price_batch(self, theta: np.ndarray, local_ids):
        base = self._base.price_batch(theta, local_ids)
        out: dict[int, Demand] = {}
        for agent_id in np.asarray(local_ids, dtype=np.int64):
            a = int(agent_id)
            demand = base[a]
            out[a] = Demand.inexact(
                bundle=demand.bundle,
                payoff=demand.payoff,
                gap=_injected_gap(a),
            )
        return out


def _assert_gap_field_witness(records: Sequence[StepRecord]) -> None:
    """Every captured ``PricedFeature.gap`` equals the injected gap.

    :func:`_injected_gap` is recomputed test-side from the agent id — never
    read back from the record — and the whole gap stream is checked, with at
    least one non-zero value required.
    """
    checked = 0
    nontrivial = 0
    for r in records:
        for pf in r.priced_features:
            expected = _injected_gap(pf.agent_id)
            assert _close(pf.gap, expected), (
                f"captured gap {pf.gap!r} differs from the injected certified"
                f" gap {expected!r} at agent {pf.agent_id}, iteration"
                f" {r.iteration}, by more than {TOL!r} — the capture is not"
                " storing the demand's gap"
            )
            checked += 1
            if expected > TOL:
                nontrivial += 1
    assert checked > 0, (
        "no priced-feature gap was witnessed"
    )
    assert nontrivial > 0, (
        "every injected gap read ~0.0 — nothing distinguishes a gap=0.0"
        " hardcode"
    )


def _gate_nslack_gap(arrays, problem, batch_only_map, backend) -> None:
    inexact = _with_oracle(problem, _InexactGapOracle(problem.oracle))
    per_agent = _nslack_serial(arrays, inexact, inexact.features, backend)
    _assert_nslack_records_exercised(per_agent.step_records)
    _assert_gap_field_witness(per_agent.step_records)

    # Feature axis (same wrapped oracle, same gaps), shard axis, then both.
    batched = _nslack_serial(arrays, inexact, batch_only_map, backend)
    _assert_records_equivalent(per_agent.step_records, batched.step_records)

    folded = _nslack_cluster(arrays, inexact, inexact.features, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded)

    folded_batched = _nslack_cluster(arrays, inexact, batch_only_map, backend, 2)
    _assert_records_equivalent(per_agent.step_records, folded_batched)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_nslack_gap_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    _gate_nslack_gap(
        toy, toy_problem(toy), toy_feature_map_batch_only(toy), backend
    )


@needs_gurobi
def test_nslack_gap_wholesale_capture_qkp() -> None:
    qkp = load_qkp()
    _gate_nslack_gap(
        qkp, qkp_problem(qkp), qkp_feature_map_batch_only(qkp), "gurobi"
    )


# --- the gate: the driver-owned Schedule field --------------------------------
#
# The Schedule field is the per-iteration DualConcentration the driver's
# schedule branch reads (informed_schedule.from_cut_duals) — support agent_ids
# + per-support-agent max_weights. It is driver-owned (run_walk computes it on
# root and broadcasts it), not formulation-internal, so it rides
# WalkOutcome.schedule_concentrations rather than the formulation StepRecord.
# A features-path drift could change the duals, hence the concentration, hence
# the priced mask, so the gate compares it across the same feature + shard axes
# (discrete support set ==, max_weights <= 1e-13). Driven on NSlack: the
# schedule re-prices a per-agent subset, whereas OneSlack requires a full sweep.


def _conc_map(conc: DualConcentration) -> dict[int, float]:
    # support agent_id -> normalized max single-cut weight; the agent-id set is
    # discrete, the weight continuous.
    return {int(a): float(w) for a, w in zip(conc.agent_ids, conc.max_weights)}


def _assert_schedule_equivalent(
    a: Sequence[DualConcentration], b: Sequence[DualConcentration]
) -> None:
    assert len(a) == len(b), (
        f"schedule streams differ in length: {len(a)} vs {len(b)} — the two"
        " walks took a different number of iterations"
    )
    for ca, cb in zip(a, b):
        ma, mb = _conc_map(ca), _conc_map(cb)
        # discrete: the support-agent set is identical across paths (a
        # >1e-10 support flip is a different payload, never a 1e-13 drift).
        assert set(ma) == set(mb), (
            "schedule support domain differs: left-only"
            f" {sorted(set(ma) - set(mb))[:3]}, right-only"
            f" {sorted(set(mb) - set(ma))[:3]}"
        )
        # continuous: each support agent's max_weight within TOL.
        for agent, wa in ma.items():
            assert _close(wa, mb[agent]), (
                f"schedule max_weight drift {abs(wa - mb[agent])!r} > {TOL!r}"
                f" at support agent {agent}"
            )


def _assert_schedule_records_exercised(stream: Sequence[DualConcentration]) -> None:
    assert stream, "no schedule concentration captured — the schedule never ran"
    # The field is exercised only if the dual support actually formed at some
    # iteration; an all-empty stream would compare trivially equal.
    assert any(c.agent_ids.size > 0 for c in stream), (
        "every captured schedule concentration was empty — the dual support"
        " never formed, so the schedule field is not exercised"
    )


def _independent_max_weight(
    duals: dict[tuple[int, bytes], float],
) -> dict[int, float]:
    # max(support dual)/sum(support dual) per support agent, computed by a plain
    # Python max/sum scan — a structurally distinct reduction from
    # from_cut_duals's list comprehension (the code under test). Reads the same
    # ground-truth per-cut duals but applies the formula independently of the
    # captured payload.
    per_agent: dict[int, list[float]] = {}
    for (agent_id, _bundle_key), pi in duals.items():
        if pi > _SUPPORT_ATOL:
            per_agent.setdefault(int(agent_id), []).append(float(pi))
    out: dict[int, float] = {}
    for agent_id, vals in per_agent.items():
        largest = vals[0]
        total = 0.0
        for v in vals:
            if v > largest:
                largest = v
            total += v
        out[agent_id] = largest / total
    return out


def _nslack_schedule_walk_recording_duals(
    arrays, problem, features, backend, transport
) -> tuple[WalkOutcome, dict[int, dict[int, float]]]:
    """A dual-informed NSlack walk plus per-iteration expected max_weights.

    The driver builds the schedule payload by calling
    ``DualConcentration.from_cut_duals(master.dual_values())`` each
    iteration. Intercept that classmethod to record its ground-truth per-cut
    duals and recompute the per-agent max/sum share by a distinct reduction
    the captured ``max_weights`` never feed. Keyed by the payload index
    (schedule iterations are appended in loop order).
    """
    expected: dict[int, dict[int, float]] = {}
    index = [0]
    original = _informed_schedule_module.DualConcentration.from_cut_duals.__func__

    def _recording(cls, duals):
        expected[index[0]] = _independent_max_weight(dict(duals))
        index[0] += 1
        return original(cls, duals)

    _informed_schedule_module.DualConcentration.from_cut_duals = classmethod(
        _recording
    )
    try:
        outcome = _nslack_schedule_walk(
            arrays, problem, features, backend, transport
        )
    finally:
        _informed_schedule_module.DualConcentration.from_cut_duals = classmethod(
            original
        )
    return outcome, expected


def _assert_schedule_max_weight_witness(
    stream: Sequence[DualConcentration],
    expected_weights: dict[int, dict[int, float]],
    *,
    require_nontrivial: bool,
) -> None:
    """Each captured ``max_weight`` equals ``max(support)/sum(support)``.

    The expected values are recomputed from the ground-truth per-cut duals,
    never read back from the payload. Where every support agent holds a
    single cut the true weight is the constant ``1.0``;
    ``require_nontrivial`` demands at least one weight strictly below
    ``1.0``, which the multi-cut family must produce so the ``min`` vs
    ``max`` distinction is real.
    """
    assert stream, "no schedule concentration captured — nothing to check"
    assert len(stream) == len(expected_weights), (
        "schedule payload stream and expected-weight stream disagree in length"
    )
    checked = 0
    nontrivial = 0
    for i, conc in enumerate(stream):
        expected = expected_weights[i]
        captured = _conc_map(conc)
        assert set(captured) == set(expected), (
            f"schedule support at payload {i} differs from the recomputed"
            f" dual support: captured-only {sorted(set(captured) - set(expected))[:3]},"
            f" expected-only {sorted(set(expected) - set(captured))[:3]}"
        )
        for agent, weight in captured.items():
            assert _close(weight, expected[agent]), (
                f"captured schedule max_weight {weight!r} differs from"
                f" max(support)/sum(support) {expected[agent]!r} at support agent"
                f" {agent}, payload {i}, by more than {TOL!r}"
            )
            checked += 1
            if expected[agent] < 1.0 - 1e-9:
                nontrivial += 1
    assert checked > 0, (
        "no support agent was checked"
    )
    if require_nontrivial:
        assert nontrivial > 0, (
            "every recomputed max_weight was 1.0 (single-cut support) — this"
            " family must hold a support agent with unequal multi-cut dual"
            " mass"
        )


def _nslack_schedule_walk(
    arrays, problem, features, backend, transport
) -> WalkOutcome:
    return run_walk(
        arrays,
        _with_features(problem, features),
        NSlack,
        transport,
        backend=backend,
        # A dual-informed schedule so the driver builds + reads the
        # DualConcentration payload every iteration (the Schedule field).
        schedule=DualInformed(concentration_threshold=0.9, min_revisit_period=2),
        capture_steprecords=True,
    )


def _gate_schedule(
    arrays, problem, batch_only_map, backend, *, require_nontrivial_weight
) -> None:
    # Reference: per-agent features, serial shard, dual-informed schedule,
    # instrumented to record the ground-truth per-cut duals each schedule
    # build.
    per_agent, expected_weights = _nslack_schedule_walk_recording_duals(
        arrays, problem, problem.features, backend, SerialTransport()
    )
    _assert_schedule_records_exercised(per_agent.schedule_concentrations)
    _assert_schedule_max_weight_witness(
        per_agent.schedule_concentrations,
        expected_weights,
        require_nontrivial=require_nontrivial_weight,
    )

    # Axis 1 — features: batched features, same schedule. The driver-owned
    # concentration must match the per-agent reference per support agent.
    batched = _nslack_schedule_walk(
        arrays, problem, batch_only_map, backend, SerialTransport()
    )
    _assert_schedule_equivalent(
        per_agent.schedule_concentrations, batched.schedule_concentrations
    )

    # Axis 2 — shard: the payload is computed on root and broadcast, so a
    # LocalCluster interleave must produce the identical concentration stream
    # (read from any rank — they all hold the broadcast payload).
    cluster = LocalCluster(2).run(
        lambda t: _nslack_schedule_walk(
            arrays, problem, problem.features, backend, t
        )
    )
    _assert_schedule_equivalent(
        per_agent.schedule_concentrations, cluster[0].schedule_concentrations
    )

    # Combined — batched features and interleaved shard.
    cluster_batched = LocalCluster(2).run(
        lambda t: _nslack_schedule_walk(
            arrays, problem, batch_only_map, backend, t
        )
    )
    _assert_schedule_equivalent(
        per_agent.schedule_concentrations,
        cluster_batched[0].schedule_concentrations,
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_nslack_schedule_concentration_wholesale_capture_toy(backend) -> None:
    toy = load_toy()
    # Every toy support agent holds a single cut, so its true max_weight is
    # the constant 1.0 and a min-vs-max swap is indistinguishable here; the
    # qkp variant carries that obligation.
    _gate_schedule(
        toy,
        toy_problem(toy),
        toy_feature_map_batch_only(toy),
        backend,
        require_nontrivial_weight=False,
    )


@needs_gurobi
def test_nslack_schedule_concentration_wholesale_capture_qkp() -> None:
    qkp = load_qkp()
    # qkp forms support agents holding several cuts of unequal dual mass, so
    # at least one true max_weight sits strictly below 1.0.
    _gate_schedule(
        qkp,
        qkp_problem(qkp),
        qkp_feature_map_batch_only(qkp),
        "gurobi",
        require_nontrivial_weight=True,
    )


# --- wholesale-capture perturbation checks ------------------------------------
#
# One perturbation per filter stage, each proving the wholesale-capture
# comparator fails when the batched path diverges. Each perturbation is an
# optimized-only (``features_batch``-only) FeatureMap, so the divergence
# reaches ``feature_rows`` and the formulation capture without being
# intercepted by a both-supplied conformance check. Every perturbation sits
# above the 1e-13 comparison tolerance or crosses a discrete threshold:
# exact-zero support, aggregate key bytes, install threshold, emit threshold.
#
# Each test first asserts the clean ``*_batch_only`` map passes the same
# comparator (so it is not already failing for an unrelated reason), then
# asserts the perturbation trips it. The comparators are reused verbatim.


def _nslack_records(arrays, problem, features, backend):
    """The serial NSlack StepRecord stream under the policy axis."""
    return _nslack_serial(arrays, problem, features, backend).step_records


def _oneslack_records(arrays, problem, features, backend):
    return _oneslack_serial(arrays, problem, features, backend).step_records


def _it0_feature_phi(records, agent_id: int) -> bytes | None:
    """The phi bytes of one agent in iteration 0's priced_features, if present.

    Iteration 0 always aligns (it is the first record on both paths regardless
    of later convergence shape), so a field-level it-0 check holds even
    when the full comparator trips earlier on the stream length.
    """
    for pf in records[0].priced_features:
        if pf.agent_id == agent_id:
            return np.asarray(pf.phi, dtype=np.float64).tobytes()
    return None


# Stage 1 — feature phi value (continuous). A nonzero phi coefficient is
# lifted by 1e-6. The drift reaches the admit-side violation
# (phi.theta + eps - u) on the matched-shape backend and changes the
# convergence shape on the more sensitive one; iteration 0 always aligns, so
# its phi bytes are asserted divergent directly.
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_phi_value_nslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    clean = _nslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    reference = _nslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(reference, clean)
    # The perturbation must fail: an admit-violation drift on a matched
    # shape, or the stream-length check when the drift moves the shape.
    perturbed = _nslack_records(
        toy, problem, toy_phi_value_perturbation(toy), backend
    )
    with pytest.raises(AssertionError):
        _assert_records_equivalent(reference, perturbed)
    # The it-0 phi bytes for the perturbed agent must differ — the value
    # drift lands in the captured phi.
    assert _it0_feature_phi(perturbed, 0) != _it0_feature_phi(reference, 0)


# Stage 2 — feature support (discrete). An exact-zero phi entry becomes
# 1e-12: the support mask flips rather than moving within tolerance.
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_phi_support_nslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _nslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(
        reference, _nslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    )
    perturbed = _nslack_records(
        toy, problem, toy_phi_support_perturbation(toy), backend
    )
    # Where it trips is vertex/backend-dependent: with predeclared u-columns
    # the gurobi warm walk re-cycles the malformed cut to a different
    # iteration count (a length mismatch), while highs trips on the
    # admit-side violation the perturbed phi feeds.
    with pytest.raises(
        AssertionError, match="admit_violations|different number of iterations"
    ):
        _assert_records_equivalent(reference, perturbed)
    # The exact-zero phi entry became nonzero, so the it-0 phi bytes differ
    # (the zero-mask is part of those bytes).
    assert _it0_feature_phi(perturbed, 0) != _it0_feature_phi(reference, 0)


# Stage 3 — feature eps (continuous). One row's eps is lifted by 1e-6. eps
# enters the cut row and the admit-side violation, so the failure lands on
# admit_violations (stable on both backends — the lift does not move the
# convergence shape here).
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_eps_nslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _nslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(
        reference, _nslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    )
    perturbed = _nslack_records(toy, problem, toy_eps_perturbation(toy), backend)
    with pytest.raises(AssertionError, match="admit_violations"):
        _assert_records_equivalent(reference, perturbed)


def _with_oracle(problem: FamilyProblem, oracle: object) -> FamilyProblem:
    # Swap only the oracle (features + theta geometry untouched), so a
    # difference in the captured records is the price path alone — the dual of
    # _with_features used for the price-stage perturbation.
    return type(problem)(
        oracle=oracle,
        features=problem.features,
        observed_features=problem.observed_features,
        K=problem.K,
        theta_bounds=problem.theta_bounds,
    )


# Stage 4 — price payoff (continuous). Agent 0's payoff is lifted by 1e-6
# with the bundle byte-identical. On OneSlack the walk shape is preserved,
# so the failure lands precisely on the priced_features "payoff drift"
# check. (price_batch vs price itself is perturbed at its real call site by
# _DivergentBatchToy in test_either_one.py; this drives the same drift
# through the wholesale demand stream.)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_price_payoff_oneslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _oneslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(
        reference, _oneslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    )
    # The perturbation rides the oracle, not the features, but reaches the
    # same priced_features demand stream the gate compares.
    perturbed_problem = _with_oracle(problem, toy_perturbation_price_oracle(toy))
    perturbed = _oneslack_records(
        toy, perturbed_problem, perturbed_problem.features, backend
    )
    with pytest.raises(AssertionError, match="payoff drift"):
        _assert_records_equivalent(reference, perturbed)


# Stage 5 — NSlack emit rc threshold (straddle ctx.tolerance). rc =
# payoff - u is captured for every priced agent before the
# rc > ctx.tolerance emit threshold. The same price drift (agent 0's payoff
# +1e-6) drives rc apart: on gurobi the policy walk cycles to the same
# length on both paths, so the failure lands precisely on the
# priced_reduced_costs drift. (On highs the drift moves the convergence
# shape and the stream-length check fails instead, so this is gurobi-only.)
@needs_gurobi
def test_perturbation_emit_rc_nslack_rejects_divergence() -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _nslack_records(toy, problem, problem.features, "gurobi")
    _assert_records_equivalent(
        reference, _nslack_records(toy, problem, toy_feature_map_batch_only(toy), "gurobi")
    )
    perturbed_problem = _with_oracle(problem, toy_perturbation_price_oracle(toy))
    perturbed = _nslack_records(
        toy, perturbed_problem, perturbed_problem.features, "gurobi"
    )
    with pytest.raises(AssertionError, match="priced_reduced_costs"):
        _assert_records_equivalent(reference, perturbed)


# Stage 6 — OneSlack aggregate key (discrete). A shape-preserving 1e-11 phi lift
# drifts the summed aggregate by about 1e-10, flipping the SHA-256 over
# [phi_agg, eps_agg]. The comparator fails on the continuous aggregate_raw
# drift, and the captured aggregate_bytes are asserted divergent directly.
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_aggregate_bytes_oneslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _oneslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(
        reference, _oneslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    )
    perturbed = _oneslack_records(
        toy, problem, toy_aggregate_bytes_perturbation(toy), backend
    )
    # The aggregate fields trip (aggregate_raw drift, then the byte
    # identity); the assertion message there is positional, so no match.
    with pytest.raises(AssertionError):
        _assert_records_equivalent(reference, perturbed)
    # The it-0 aggregate SHA-256 flipped — the discrete row key moved —
    # while the drift that flipped it exceeds the comparison tolerance.
    assert perturbed[0].aggregate_bytes != reference[0].aggregate_bytes
    assert abs(perturbed[0].aggregate_raw - reference[0].aggregate_raw) > TOL


# Stage 7 — OneSlack install gate (discrete straddle of ctx.tolerance). A
# single-row 1e-6 phi lift the master cannot absorb keeps the aggregate
# slack above ctx.tolerance at the iteration the clean path converges on, so
# the install gate keeps firing on the perturbed path and the convergence
# shape diverges.
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_install_gate_oneslack_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _oneslack_records(toy, problem, problem.features, backend)
    _assert_records_equivalent(
        reference, _oneslack_records(toy, problem, toy_feature_map_batch_only(toy), backend)
    )
    perturbed = _oneslack_records(
        toy, problem, toy_install_gate_perturbation(toy), backend
    )
    # The install straddle diverges the iteration count, so the stream-length
    # (shape) check is the expected failure here.
    with pytest.raises(AssertionError, match="differ in length"):
        _assert_records_equivalent(reference, perturbed)
    assert len(perturbed) != len(reference)


# Stage 8 — Schedule DualConcentration. A 1e-6 phi lift moves the master
# duals the driver-owned DualConcentration is condensed from, diverging the
# dual-informed walk's convergence shape and hence the concentration stream
# length. (The support max_weights saturate at 1.0 on these families, so the
# continuous weight field cannot be moved off the ceiling.)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gurobi", marks=needs_gurobi),
        pytest.param("highs", marks=needs_highs),
    ],
)
def test_perturbation_schedule_concentration_rejects_divergence(backend) -> None:
    toy = load_toy()
    problem = toy_problem(toy)
    reference = _nslack_schedule_walk(
        toy, problem, problem.features, backend, SerialTransport()
    ).schedule_concentrations
    _assert_schedule_records_exercised(reference)
    clean = _nslack_schedule_walk(
        toy, problem, toy_feature_map_batch_only(toy), backend, SerialTransport()
    ).schedule_concentrations
    _assert_schedule_equivalent(reference, clean)
    perturbed = _nslack_schedule_walk(
        toy, problem, toy_schedule_perturbation(toy), backend, SerialTransport()
    ).schedule_concentrations
    with pytest.raises(AssertionError, match="schedule streams differ in length"):
        _assert_schedule_equivalent(reference, perturbed)
