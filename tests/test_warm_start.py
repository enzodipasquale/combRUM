"""Warm-start: the cut set a prior fit installed seeds the next fit.

A fit ends holding the published ``FormulationResult.active_set`` in
canonical ``(agent_id, bundle_key)`` order. Warm-start replays it onto a
fresh master before the refit prices anything: ``master.reinstall`` rebuilds
the same relaxation, and NSlack rebuilds its bookkeeping from
``extract_cuts()``/``theta()`` at setup, so the refit adopts the cuts
unchanged. The artifact is a deterministic, persistable tuple — the
checkpoint that chains bootstrap reps and sweep cells.
"""

from __future__ import annotations

import numpy as np
import pytest

from _family_oracles import qkp_problem, toy_problem
from _walk import run_walk
from combrum.formulations import NSlack
from _support.families import DEFAULT_SEED, qkp_family, toy_family
from _support.probes import measure
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.transport import LocalCluster, SerialTransport

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

# Both LP backends warm-start identically (reinstall is a pure-LP rebuild),
# so the cut-replay gates run on each available backend; the penalty
# composition gate is gurobi-only (HiGHS does not expose quadratic support),
# as is the wall-clock guard (the historical measurement convention).
needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
pytestmark = pytest.mark.slow
REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param(
        "highs",
        marks=pytest.mark.skipif(not HIGHS_AVAILABLE, reason="no highs"),
    ),
)

# Equal-objective claims are banded at 1e-9, never asserted bitwise: warm
# and cold can land on different vertices of a degenerate optimal face, so
# theta_hat is set-identified while the objective is the pinned quantity.
PARITY_BAND = 1e-9

# Fixture sizes for the iteration-win gates. Each cold fit needs visibly
# many row-generation iterations so "strictly fewer" is a real reduction,
# and the win holds at >= 2 sizes so it is not a single-size artifact.
SIZES = ((24, 8), (40, 6))
# The sweep-cell scale step. Large enough to move a real subset of observed
# bundles (a genuine perturbation, not the same problem), small enough that
# cell 0's cut set still covers most of cell 1 — where warm-start pays.
SWEEP_SCALE = 1.25

# QKP fixture sizes for the same-problem warm-start leg: known-good and
# brute-force-enumerable at n_items=6, same two-sizes rationale as SIZES.
QKP_SIZES = ((20, 6), (30, 6))

# A same-problem warm-start holds every cut already, so it only re-certifies:
# at most one full re-pricing sweep that finds no violation, plus one
# certifying sweep — a ceiling of 2 by construction (the walk is
# deterministic: SerialTransport, fixed seed, primal simplex). A reinstall
# that drops part of the seed must re-derive the missing cuts and pushes warm
# to 3+, so the ceiling has to sit exactly at 2; a looser 3 would wave the
# smallest partial drop through.
WARM_RECERT_CEILING = 2
assert WARM_RECERT_CEILING == 2, (
    "a looser ceiling stops separating a whole-seed replay from a 1-cut drop"
)

# Wall-clock is a soft guard: the reinstall pays an up-front solve and a
# single fit is milliseconds at these sizes, so the ceiling is loose and the
# cold baseline is floored so a sub-millisecond fit cannot explode the ratio.
WALL_SANITY_FACTOR = 8.0
WALL_FLOOR_SECONDS = 0.05

# A static prox anchored at the seed with qp_weight=1.0 makes the seed the
# minimizer of the penalty's theta block, so a penalized priced iterate sits
# on the seed to solve precision (observed ~1e-15); an anchor pointed
# anywhere else cannot bring any iterate within this band.
STATIC_ANCHOR_LANDING = 1e-6


def _toy(n_obs: int, n_items: int) -> dict[str, np.ndarray]:
    return toy_family(n_obs, n_items, DEFAULT_SEED)


def _qkp(n_obs: int, n_items: int) -> dict[str, np.ndarray]:
    return qkp_family(n_obs, n_items, DEFAULT_SEED)


def _fit(arrays: dict[str, np.ndarray], backend: str, **kw) -> object:
    return run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend=backend,
        **kw,
    )


def _qkp_fit(arrays: dict[str, np.ndarray], backend: str, **kw) -> object:
    # Same walk as _fit over the QKP problem; reinstall is a pure-LP cut
    # replay, independent of the pricing subproblem.
    return run_walk(
        arrays,
        qkp_problem(arrays),
        NSlack,
        SerialTransport(),
        backend=backend,
        **kw,
    )


def _seed_keys(result: object) -> set[tuple[int, bytes]]:
    """The ``(agent_id, bundle_key)`` identity of every published cut."""
    return {(row.agent_id, row.bundle_key) for row in result.active_set}


def _scaled_cell(
    base: dict[str, np.ndarray], factor: float
) -> dict[str, np.ndarray]:
    """A related toy cell: same agents/shocks, ``theta_true`` scaled.

    Keeping ``r`` and ``nu`` fixed while scaling ``theta_true`` is a sweep
    over the structural parameter: observed bundles change only where the
    rescaled score crosses zero, so the cells share most of their optimal cut
    structure — the regime where a prior cell's cuts warm-start the next.
    Regenerating ``observed`` under the scaled parameter keeps each cell
    rationalisable, so it is a genuine fit, not a perturbed-data artifact.
    """
    r = np.asarray(base["observables"], dtype=np.float64)
    nu = np.asarray(base["shocks"], dtype=np.float64)[:, 0, :]
    theta = np.asarray(base["theta_true"], dtype=np.float64) * factor
    observed = (r * theta[None, :] + nu) > 0.0
    cell = {
        "observables": r.copy(),
        "shocks": nu.reshape(nu.shape[0], 1, nu.shape[1]).copy(),
        "observed": observed,
        "theta_true": theta,
    }
    for arr in cell.values():
        arr.setflags(write=False)
    return cell


def _fit_capturing_priced_theta(
    arrays: dict[str, np.ndarray], **kw
) -> tuple[object, list[np.ndarray]]:
    """Run a gurobi fit, returning (outcome, priced_theta_per_iteration).

    ``NSlack.solve`` is called once at the top of each row-generation
    iteration and returns exactly the theta that iteration prices, so
    wrapping it yields one aligned iterate per iteration — the priced-theta
    stream a penalty is supposed to steer.
    """
    priced: list[np.ndarray] = []
    original = NSlack.solve

    def wrapped(self):  # type: ignore[no-untyped-def]
        theta = original(self)
        priced.append(np.asarray(theta, dtype=np.float64).copy())
        return theta

    NSlack.solve = wrapped
    try:
        outcome = _fit(arrays, "gurobi", **kw)
    finally:
        NSlack.solve = original
    return outcome, priced


# --------------------------------------------------------------------------
# (a): same-problem warm-start — fewer iterations at equal objective
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("n_obs,n_items", SIZES)
def test_same_problem_warm_start_converges_in_fewer_iterations(
    backend: str, n_obs: int, n_items: int
) -> None:
    # Every cut is already installed, so the warm refit only re-certifies —
    # a handful of iterations against the cold fit's many — at the same
    # optimum.
    arrays = _toy(n_obs, n_items)
    cold = _fit(arrays, backend)
    assert cold.converged and cold.result.active_set is not None
    warm = _fit(arrays, backend, warm_start=cold.result)
    assert warm.converged
    assert warm.iterations < cold.iterations, (
        f"{backend} {n_obs}x{n_items}: warm did not save iterations"
        f" (cold={cold.iterations}, warm={warm.iterations})"
    )
    # Stronger than "< cold": every cut is pre-installed, so the refit
    # re-certifies within the ceiling. A partial seed replay re-derives the
    # missing cuts and breaks it.
    assert warm.iterations <= WARM_RECERT_CEILING, (
        f"{backend} {n_obs}x{n_items}: warm did not re-certify within"
        f" {WARM_RECERT_CEILING} sweeps (warm={warm.iterations}) — the seed"
        f" cut set was not fully replayed"
    )
    # The ceiling misses a dropped-but-unviolated cut (warm still re-certifies
    # in one sweep), so also require cut identity: every cut the cold fit
    # published persists into the warm refit's final active set.
    seed_keys = _seed_keys(cold.result)
    warm_keys = _seed_keys(warm.result)
    missing = seed_keys - warm_keys
    assert not missing, (
        f"{backend} {n_obs}x{n_items}: warm dropped {len(missing)} of"
        f" {len(seed_keys)} seed cuts (never returned to the active set) — the"
        f" reinstall did not replay the whole same-problem seed"
    )
    assert abs(warm.objective - cold.objective) <= PARITY_BAND
    print(
        f"\nsame-problem {backend} {n_obs}x{n_items}:"
        f" cold_iters={cold.iterations} warm_iters={warm.iterations}"
        f" dobj={warm.objective - cold.objective:+.2e}"
    )


# --------------------------------------------------------------------------
# (b): sweep-cell warm-start — the realistic chained-fit win, >= 2 sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("n_obs,n_items", SIZES)
def test_sweep_cell_warm_start_beats_cold_at_equal_objective(
    backend: str, n_obs: int, n_items: int
) -> None:
    # The realistic case: fit cell 0 cold, then fit a related cell 1 cold vs
    # warm-started from cell 0's cuts. A real subset of observed bundles
    # moves under the scaled parameter, yet the shared cut structure lets the
    # warm fit reach the same objective in fewer iterations.
    base = _toy(n_obs, n_items)
    cell0 = _scaled_cell(base, 1.0)
    cell1 = _scaled_cell(base, SWEEP_SCALE)
    changed = int((cell0["observed"] != cell1["observed"]).sum())
    assert changed > 0, "scale step must move at least one observed bundle"

    fit0 = _fit(cell0, backend)
    assert fit0.converged and fit0.result.active_set is not None
    cold1 = _fit(cell1, backend)
    warm1 = _fit(cell1, backend, warm_start=fit0.result)
    assert cold1.converged and warm1.converged
    assert warm1.iterations < cold1.iterations, (
        f"{backend} {n_obs}x{n_items}: sweep warm did not save iterations"
        f" (cold={cold1.iterations}, warm={warm1.iterations},"
        f" changed_obs={changed})"
    )
    # The moved bundles genuinely need new cuts, so no re-certify ceiling
    # here. But the seed must be replayed whole: NSlack never retires cuts
    # without a policy, so every cut fit0 published persists into warm1's
    # final active_set.
    seed_keys = _seed_keys(fit0.result)
    warm_keys = _seed_keys(warm1.result)
    missing = seed_keys - warm_keys
    assert not missing, (
        f"{backend} {n_obs}x{n_items}: warm dropped {len(missing)} of"
        f" {len(seed_keys)} seed cuts (they never returned to the active set) —"
        f" the reinstall did not replay the whole seed"
    )
    assert abs(warm1.objective - cold1.objective) <= PARITY_BAND
    print(
        f"\nsweep-cell {backend} {n_obs}x{n_items} (scale={SWEEP_SCALE},"
        f" changed_obs={changed}): cold_iters={cold1.iterations}"
        f" warm_iters={warm1.iterations}"
        f" dobj={warm1.objective - cold1.objective:+.2e}"
    )


# --------------------------------------------------------------------------
# (c): the warm-start artifact is deterministic and persistable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_active_set_is_a_deterministic_artifact(backend: str) -> None:
    # The cut set is the warm-start checkpoint, so it must be reproducible to
    # the byte: two cold fits of the same problem yield active_set rows with
    # identical agent_ids, bundle_keys, phi bytes, and epsilon — hashable and
    # persistable, in canonical order.
    arrays = _toy(*SIZES[0])
    first = _fit(arrays, backend).result.active_set
    second = _fit(arrays, backend).result.active_set
    assert first is not None and len(first) > 0
    assert len(first) == len(second)
    for row_a, row_b in zip(first, second):
        assert row_a.agent_id == row_b.agent_id
        assert row_a.bundle_key == row_b.bundle_key
        assert row_a.phi.tobytes() == row_b.phi.tobytes()
        assert row_a.epsilon == row_b.epsilon
    # Canonical (agent_id, bundle_key) order, so a persisted artifact diffs
    # cleanly and rehydrates identically.
    keys = [(row.agent_id, row.bundle_key) for row in first]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------
# (d): penalty x warm-start composition (the penalty-interaction clause)
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("n_obs,n_items", SIZES)
def test_warm_start_composes_with_penalty_decay(
    n_obs: int, n_items: int
) -> None:
    # Warm-start seeds the cut set; the penalty steers within the optimal
    # face. Composed, the fit must still converge, terminate on a pure LP,
    # and reach the cold no-penalty optimum.
    qp_iterations = 3
    arrays = _toy(n_obs, n_items)
    cold = _fit(arrays, "gurobi")
    assert cold.converged
    # Both runs warm-start from the same seed, so their first priced iterate
    # (the setup solve, penalty-free) is identical.
    warm_alone, alone_theta = _fit_capturing_priced_theta(
        arrays, warm_start=cold.result
    )
    assert warm_alone.converged
    warm_pen, pen_theta = _fit_capturing_priced_theta(
        arrays,
        warm_start=cold.result,
        qp_weight=1.0,
        qp_iterations=qp_iterations,
        penalty_ref="static",
    )
    assert warm_pen.converged
    # The seed is the static anchor: with warm_start given and no explicit
    # theta_init, run_walk sets static_ref = theta_init = cold.result.theta_hat.
    # It is a nonzero LP vertex, so it is distinguishable from the origin.
    seed = np.asarray(cold.result.theta_hat, dtype=np.float64)
    assert np.max(np.abs(seed)) > 1e-3, (
        "static-anchor test needs a nonzero seed to tell the seed anchor apart"
        " from the origin"
    )
    # The penalty must actually move the iterates: at least one penalized
    # priced iterate has to differ from the corresponding warm-alone one.
    # Matching the pure-LP stream bit for bit would mean set_penalty never
    # installed the term.
    aligned = min(len(alone_theta), len(pen_theta))
    max_steer = max(
        (
            float(np.max(np.abs(alone_theta[i] - pen_theta[i])))
            for i in range(aligned)
        ),
        default=0.0,
    )
    assert max_steer > 1e-6, (
        f"warm+penalty {n_obs}x{n_items}: the penalty did not steer any"
        f" intermediate iterate off the pure-LP path (max|dtheta|={max_steer:.2e})"
        f" — set_penalty had no effect"
    )
    # And where it steers: with the prox centered on the seed at unit weight,
    # some penalized iterate must land on the seed to solve precision. The
    # target comes from a distinct penalty-free fit, never from the penalized
    # run; an anchor pointed anywhere else leaves every iterate out of the band.
    dist_to_seed = [float(np.max(np.abs(t - seed))) for t in pen_theta]
    min_dist = min(dist_to_seed, default=float("inf"))
    assert min_dist <= STATIC_ANCHOR_LANDING, (
        f"warm+penalty {n_obs}x{n_items}: no penalized iterate reached the"
        f" static seed anchor (min|theta-seed|={min_dist:.2e} >"
        f" {STATIC_ANCHOR_LANDING:.0e}) — the prox is not centered on the seed"
    )
    # Under a dynamic ref (re-centered each iteration) nothing pins theta to
    # the fixed seed, so no dynamic iterate lands on it — the static/dynamic
    # anchor choice must separate.
    warm_dyn, dyn_theta = _fit_capturing_priced_theta(
        arrays,
        warm_start=cold.result,
        qp_weight=1.0,
        qp_iterations=qp_iterations,
        penalty_ref="dynamic",
    )
    assert warm_dyn.converged
    dyn_min_dist = min(
        (float(np.max(np.abs(t - seed))) for t in dyn_theta),
        default=float("inf"),
    )
    assert dyn_min_dist > STATIC_ANCHOR_LANDING, (
        f"warm+penalty {n_obs}x{n_items}: a dynamic-ref iterate landed on the"
        f" seed (min|theta-seed|={dyn_min_dist:.2e}) — the static/dynamic anchor"
        f" choice is not being honored"
    )
    # Terminating solve is a pure LP: the published weight dropped to exactly 0.
    assert warm_pen.final_penalty_weight == 0.0
    assert abs(warm_pen.objective - cold.objective) <= PARITY_BAND
    print(
        f"\nwarm+penalty gurobi {n_obs}x{n_items}:"
        f" iters={warm_pen.iterations} final_w={warm_pen.final_penalty_weight}"
        f" max_steer={max_steer:.2e} min_dist_to_seed={min_dist:.2e}"
        f" dyn_min_dist={dyn_min_dist:.2e}"
        f" dobj={warm_pen.objective - cold.objective:+.2e}"
    )

# --------------------------------------------------------------------------
# (e): a warm-started fit is bitwise rank-invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("size", [2])
def test_warm_start_rank_invariant_bitwise(backend: str, size: int) -> None:
    # Warm-start is a root-only edit (reinstall touches the rank-0 master
    # alone) and adds no cross-rank reduction, so a warm-started fit must
    # come back bitwise identical whether the agents run serially or sharded
    # across a fake cluster: same theta_hat bytes, same objective.
    arrays = _toy(*SIZES[0])
    seed = _fit(arrays, backend).result
    serial = _fit(arrays, backend, warm_start=seed)
    # The run under test must be genuinely warm-started, else this compares
    # two cold fits. A warm refit takes far fewer sweeps than cold, so a
    # reinstall that failed to replay the seed trips here first.
    cold_iters = _fit(arrays, backend).iterations
    assert serial.iterations < cold_iters, (
        f"{backend}: warm-start not engaged (serial={serial.iterations},"
        f" cold={cold_iters}) — rank-invariance would compare two cold fits"
    )
    results = LocalCluster(size).run(
        lambda transport: run_walk(
            arrays,
            toy_problem(arrays),
            NSlack,
            transport,
            backend=backend,
            warm_start=seed,
        )
    )
    assert len(results) == size
    for outcome in results:
        assert (
            outcome.result.theta_hat.tobytes()
            == serial.result.theta_hat.tobytes()
        )
        assert outcome.objective == serial.objective
        assert outcome.iterations == serial.iterations


# --------------------------------------------------------------------------
# (f): same-problem warm-start on QKP — fewer iterations, >= 2 sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("n_obs,n_items", QKP_SIZES)
def test_qkp_same_problem_warm_start_converges_in_fewer_iterations(
    backend: str, n_obs: int, n_items: int
) -> None:
    # Same-problem warm-start carried onto the QKP family. The QKP
    # subproblem prices by brute force, but warm-start is a pure-LP cut
    # replay on the rank-0 master, so the same claim holds.
    arrays = _qkp(n_obs, n_items)
    cold = _qkp_fit(arrays, backend)
    assert cold.converged and cold.result.active_set is not None
    warm = _qkp_fit(arrays, backend, warm_start=cold.result)
    assert warm.converged
    assert warm.iterations < cold.iterations, (
        f"qkp {backend} {n_obs}x{n_items}: warm did not save iterations"
        f" (cold={cold.iterations}, warm={warm.iterations})"
    )
    # Same re-certify ceiling as the toy leg: the QKP seed set is fully
    # replayed, so the refit certifies within a couple of sweeps.
    assert warm.iterations <= WARM_RECERT_CEILING, (
        f"qkp {backend} {n_obs}x{n_items}: warm did not re-certify within"
        f" {WARM_RECERT_CEILING} sweeps (warm={warm.iterations})"
    )
    # The ceiling misses a dropped-but-unviolated cut, so also check cut
    # identity: every cut the cold fit published persists into the warm
    # refit's active set.
    seed_keys = _seed_keys(cold.result)
    warm_keys = _seed_keys(warm.result)
    missing = seed_keys - warm_keys
    assert not missing, (
        f"qkp {backend} {n_obs}x{n_items}: warm dropped {len(missing)} of"
        f" {len(seed_keys)} seed cuts (never returned to the active set) — the"
        f" reinstall did not replay the whole QKP seed"
    )
    assert abs(warm.objective - cold.objective) <= PARITY_BAND
    print(
        f"\nqkp same-problem {backend} {n_obs}x{n_items}:"
        f" cold_iters={cold.iterations} warm_iters={warm.iterations}"
        f" dobj={warm.objective - cold.objective:+.2e}"
    )


# --------------------------------------------------------------------------
# (g): warm-vs-cold wall-clock (soft), toy and qkp, gurobi
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize(
    "kind,n_obs,n_items",
    [("toy", *SIZES[0]), ("qkp", *QKP_SIZES[1])],
)
def test_warm_start_wall_clock_within_soft_sanity_ceiling(
    kind: str, n_obs: int, n_items: int
) -> None:
    # Soft guard: the warm fit stays within a generous multiple of the cold
    # fit's wall clock. The reinstall pays an up-front solve, so this never
    # claims warm is strictly faster — the iteration win above is the real
    # gate. Gurobi only, by the historical measurement convention.
    fit = _qkp_fit if kind == "qkp" else _fit
    arrays = (_qkp if kind == "qkp" else _toy)(n_obs, n_items)
    cold, cold_probe = measure(lambda: fit(arrays, "gurobi"))
    assert cold.converged and cold.result.active_set is not None
    warm, warm_probe = measure(
        lambda: fit(arrays, "gurobi", warm_start=cold.result)
    )
    # Fast and right: a warm fit that is quick because it bailed out early
    # (non-converged, or off the cold optimum) must fail here.
    assert warm.converged, f"warm {kind} {n_obs}x{n_items} did not converge"
    assert abs(warm.objective - cold.objective) <= PARITY_BAND, (
        f"warm {kind} {n_obs}x{n_items} ran fast but missed the cold optimum"
        f" (dobj={warm.objective - cold.objective:+.2e})"
    )
    ceiling = (
        max(cold_probe.wall_seconds, WALL_FLOOR_SECONDS) * WALL_SANITY_FACTOR
    )
    assert warm_probe.wall_seconds <= ceiling
    print(
        f"\nwarm-vs-cold wall {kind} {n_obs}x{n_items}:"
        f" cold_wall={cold_probe.wall_seconds:.4f}s"
        f" warm_wall={warm_probe.wall_seconds:.4f}s"
        f" ceiling={ceiling:.4f}s"
    )
