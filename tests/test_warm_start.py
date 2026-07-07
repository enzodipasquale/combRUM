"""Warm-start: the cut set a prior fit installed seeds the next fit.

A row-generation fit ends holding an installed cut set — the published
``FormulationResult.active_set``, in canonical ``(agent_id, bundle_key)``
order. Warm-start replays it onto a fresh master before the refit prices
anything: ``master.reinstall(prior.active_set)`` rebuilds the same
relaxation, and because NSlack rebuilds its bookkeeping from
``extract_cuts()``/``theta()`` at setup, the refit adopts the pre-installed
cuts unchanged. The artifact is a deterministic,
persistable tuple, so warm-start chains across bootstrap reps and sweep
cells — the cut set is the checkpoint.

The gates prove the win is real and survives composition:

* Same-problem (the strong case): a refit warm-started from the identical
  problem's cuts converges to the same objective in strictly fewer
  iterations — every cut is already present, so the refit only re-certifies.
* Sweep-cell (the realistic case): two related cells (the same agents and
  shocks under a scaled ``theta_true``, so a genuine but partial change of
  observed bundles) — the second cell warm-started from the first's cuts
  reaches the same objective in fewer iterations than cold, at >= 2 sizes.
* Determinism: two cold fits of one problem publish byte-identical
  ``active_set`` rows — the warm-start artifact is reproducible.
* Penalty x warm-start: a warm-started fit that also runs the decay penalty
  converges, terminates pure-LP, and matches the cold no-penalty objective.
* Rank-invariance: a warm-started fit is bitwise identical serial vs a fake
  cluster — warm-start is a root-only master edit, adding no cross-rank
  reduction.
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
# so the cut-replay gates run on each available backend; only the penalty
# composition gate is gurobi-only (HiGHS has no quadratic term).
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

# QKP fixture sizes for the same-problem warm-start leg. Reused from the
# stripping file's churning QKP sizes (known-good, brute-force-enumerable at
# n_items=6); same ">= 2 sizes / visibly many iterations" rationale as SIZES.
QKP_SIZES = ((20, 6), (30, 6))

# A same-problem warm-start holds every cut already, so it only re-certifies:
# at most one full re-pricing sweep to find no violation, plus one certifying
# sweep — an absolute ceiling of 2 sweeps by construction. The walk is
# deterministic here (SerialTransport, fixed seed, primal simplex), so the
# observed same-problem warm count is a hard 2 at every param/backend, not a
# jittery number needing a band. Derived from the "only re-certifies" property,
# not from `< cold`: a reinstall that silently drops part of the seed set has
# to re-derive the missing cuts and pushes warm to 3+ — one above this ceiling
# — even though it still beats cold. Pinning the ceiling at the honest max (not
# one above it) is what lets this separate a whole-seed replay from a 1-cut
# drop; a looser 3 would wave the smallest partial-drop through.
WARM_RECERT_CEILING = 2

# Guard the ceiling against a silent loosening: the recert argument only has
# the partial-seed signal while the ceiling sits exactly at the honest
# same-problem max (2). If a future edit bumps it to 3 the smallest
# partial-seed-drop regression (warm=3) slips through, so pin it here.
assert WARM_RECERT_CEILING == 2, (
    "WARM_RECERT_CEILING must stay at the honest same-problem max (2); a higher"
    " ceiling stops separating a whole-seed replay from a 1-cut drop"
)

# Wall-clock is a soft guard only. Warm-start does strictly less work, yet
# the reinstall pays an up-front solve and a single fit is milliseconds at
# these sizes — so the timing assertion is a loose ceiling, never a tight
# band, and the iteration-count win is the real claim.
# The cold baseline is floored before scaling so a sub-millisecond cold fit
# (where one scheduler hiccup on the warm side would explode the ratio)
# cannot turn this loose ceiling into a flake.
WALL_SANITY_FACTOR = 8.0
WALL_FLOOR_SECONDS = 0.05

# A static prox anchored at the seed with qp_weight=1.0 makes the seed the
# unconstrained minimizer of the penalty's theta block, so a penalized priced
# iterate sits on the seed to LP-solve precision (observed ~1e-15). This band
# is that solve precision, not a value read from combrum: an anchor pointed
# anywhere but the seed cannot bring any iterate within it.
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
    # Same walk as _fit, over the QKP oracle/features: the brute-force QKP
    # subproblem warm-starts identically (reinstall is a pure-LP cut replay,
    # oracle-agnostic), so the iteration-win gate carries straight across.
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
    # The stated claim is stronger than "< cold": every cut is pre-installed,
    # so the refit only re-certifies within WARM_RECERT_CEILING sweeps. A
    # reinstall that drops part of the seed still beats cold (it re-derives the
    # missing cuts) but breaks this ceiling.
    assert warm.iterations <= WARM_RECERT_CEILING, (
        f"{backend} {n_obs}x{n_items}: warm did not re-certify within"
        f" {WARM_RECERT_CEILING} sweeps (warm={warm.iterations}) — the seed"
        f" cut set was not fully replayed"
    )
    # The ceiling only bites when a dropped cut is still violated and forces an
    # extra sweep. A dropped-but-unviolated cut leaves warm re-certifying in one
    # sweep (ceiling passes) yet permanently absent from warm's active set, so
    # the identity oracle the sweep-cell leg uses is required here too: every cut
    # this problem's cold fit published must survive into the warm refit's final
    # active set, whole seed replayed, not merely "enough of it to re-certify".
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
    # The realistic case: fit cell 0 cold, then fit a related cell 1
    # cold vs warm-started from cell 0's cuts. The cells differ — a real
    # subset of observed bundles moves under the scaled parameter — yet the
    # shared cut structure lets the warm fit reach the same objective in
    # fewer iterations. Held at two sizes so the win survives scale.
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
    # The sweep cell can genuinely re-derive cuts for the moved bundles, so it
    # need not re-certify within WARM_RECERT_CEILING like the same-problem legs.
    # But the seed itself must have been replayed *whole*: NSlack installs cuts
    # and never retires without a policy, so every cut fit0 published has to
    # persist into warm1's final active_set. A reinstall that silently drops
    # part of the seed still beats cold (it re-derives the violated few) but
    # leaves the unviolated dropped cuts permanently absent from warm1's cut
    # set, so this subset check catches the partial-drop locally, not only via
    # the same-problem ceiling gates.
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
    # Warm-start and the decay penalty compose: a warm-started fit that also
    # runs the quadratic penalty (qp_weight>0, decay>0) still converges,
    # still terminates on a pure LP (final weight 0, so the published duals
    # are valid LP duals), and still reaches the cold no-penalty optimum.
    # The two seams are orthogonal — warm-start seeds the cut set, the
    # penalty steers within the optimal face — so their combination adds no
    # new failure mode. (gurobi only: HiGHS cannot host the quadratic term.)
    decay = 3
    arrays = _toy(n_obs, n_items)
    cold = _fit(arrays, "gurobi")
    assert cold.converged
    # Both runs warm-start from the same seed, so their first priced iterate
    # (the setup solve, penalty-free) is identical. Capture the priced-theta
    # stream from each: the penalty must actually move an intermediate iterate.
    warm_alone, alone_theta = _fit_capturing_priced_theta(
        arrays, warm_start=cold.result
    )
    assert warm_alone.converged
    warm_pen, pen_theta = _fit_capturing_priced_theta(
        arrays,
        warm_start=cold.result,
        qp_weight=1.0,
        decay=decay,
        penalty_ref="static",
    )
    assert warm_pen.converged
    # The seed is the static anchor: with warm_start given and no explicit
    # theta_init, run_walk sets static_ref = theta_init = cold.result.theta_hat.
    # It is nonzero here (a real LP vertex), so anchoring at it is observably
    # different from anchoring at the origin.
    seed = np.asarray(cold.result.theta_hat, dtype=np.float64)
    assert np.max(np.abs(seed)) > 1e-3, (
        "static-anchor test needs a nonzero seed to tell the seed anchor apart"
        " from the origin"
    )
    # Penalty EFFECT, not the tautological iteration count: the decay penalty
    # steers theta within the optimal face, so at least one intermediate priced
    # iterate under the penalty must differ from the corresponding pure-LP
    # warm-alone iterate. A no-op set_penalty leaves the two streams
    # bit-identical (both pure LP over the same seed), so this max drops to 0.
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
    # WHERE the static anchor points, not just that it steers: with the prox
    # centered on the seed at unit weight, the seed minimizes the penalty's
    # theta block, so a penalized priced iterate must land on the seed to solve
    # precision. This pins the anchor target independently — the expected point
    # is cold.result.theta_hat from a distinct penalty-free fit, never read back
    # from the penalized run. An anchor pointed at the origin (or any other
    # point) leaves every iterate ||seed - wrong_anchor|| away and cannot reach
    # this band, so a whole class of ref-corruption mutations dies here.
    dist_to_seed = [float(np.max(np.abs(t - seed))) for t in pen_theta]
    min_dist = min(dist_to_seed, default=float("inf"))
    assert min_dist <= STATIC_ANCHOR_LANDING, (
        f"warm+penalty {n_obs}x{n_items}: no penalized iterate reached the"
        f" static seed anchor (min|theta-seed|={min_dist:.2e} >"
        f" {STATIC_ANCHOR_LANDING:.0e}) — the prox is not centered on the seed"
    )
    # Contrast the anchor choice: with a dynamic ref (re-centered on the current
    # theta each iteration) nothing pins theta to the fixed seed, so no dynamic
    # iterate lands on it. If set_penalty ignored its ref both modes would behave
    # the same and this separation would vanish.
    warm_dyn, dyn_theta = _fit_capturing_priced_theta(
        arrays,
        warm_start=cold.result,
        qp_weight=1.0,
        decay=decay,
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
    # Terminating solve is a pure LP: the published weight decayed to exactly 0.
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
    # Anchor: the run under test must be genuinely warm-started, else this is
    # a rank-consistency check on two cold fits and says nothing about
    # warm-start. A warm refit re-certifies in far fewer sweeps than the cold
    # fit at the same size, so a reinstall that fails to replay the seed (warm
    # collapses to a cold fit) trips this before the rank comparison.
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
    # The strong case carried onto the QKP family: refit the identical QKP
    # problem warm-started from its own cut set. The QKP subproblem is a
    # different (brute-force) oracle than the toy, but warm-start is a
    # pure-LP cut replay on the rank-0 master, so the same claim holds — the
    # warm refit only re-certifies in a handful of iterations against the
    # cold fit's many, at the same optimum. Held at two QKP sizes (matching
    # the stripping file's churning sizes) so the win survives scale on this
    # family too.
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
    # And the whole QKP seed must be replayed, not just enough of it to
    # re-certify: the ceiling waves through a dropped-but-unviolated cut, so pin
    # cut identity directly — every cut the cold QKP fit published survives into
    # the warm refit's active set.
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
    # soft guard: a warm-started fit must not run away — it should stay
    # within a generous multiple of the cold fit's wall clock. The real
    # claim is the iteration win (gated deterministically above); warm-start
    # does strictly less row-generation work because the cut set is
    # pre-installed, but the reinstall pays an up-front solve, so this leg
    # asserts only the loose ceiling, never that warm is strictly faster.
    # Deliberately loose (the suite has a known wall flake under load, so a
    # tight millisecond-scale timing band is avoided here on purpose).
    # gurobi only, by the penalty/QP measurement convention.
    fit = _qkp_fit if kind == "qkp" else _fit
    arrays = (_qkp if kind == "qkp" else _toy)(n_obs, n_items)
    cold, cold_probe = measure(lambda: fit(arrays, "gurobi"))
    assert cold.converged and cold.result.active_set is not None
    warm, warm_probe = measure(
        lambda: fit(arrays, "gurobi", warm_start=cold.result)
    )
    # The warm leg must be fast *and* right: a warm fit that runs quickly by
    # bailing out early (non-converged, or landing off the cold optimum) is a
    # regression the wall ceiling alone would wave through.
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
