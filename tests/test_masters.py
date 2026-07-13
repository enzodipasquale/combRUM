"""Real-backend master semantics beyond the generic conformance battery.

Every asserted number below is hand-derived from a pinned LP whose
certificate sits next to its data, so a failure indicts the backend,
never the fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from combrum.master import MasterBackend
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master, master_environment, resolve_master_backend
from combrum.masters.gurobi import GurobiMaster
from combrum.masters.highs import HighsMaster
from combrum.transport import CutRow, LocalCluster

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param("highs", marks=needs_highs),
)

K = 2


def _rows_by_key(master: MasterBackend) -> dict[tuple[int, bytes], CutRow]:
    return {(r.agent_id, r.bundle_key): r for r in master.extract_cuts()}


def test_resolve_auto_preserves_public_preference(monkeypatch) -> None:
    monkeypatch.setattr(gurobi_backend, "available", lambda: True)
    monkeypatch.setattr(highs_backend, "available", lambda: True)

    assert resolve_master_backend("auto") == "gurobi"


def test_resolve_auto_falls_back_to_highs(monkeypatch) -> None:
    monkeypatch.setattr(gurobi_backend, "available", lambda: False)
    monkeypatch.setattr(highs_backend, "available", lambda: True)

    assert resolve_master_backend("auto") == "highs"


def test_resolve_explicit_highs_does_not_probe_solvers(monkeypatch) -> None:
    def fail_probe() -> bool:
        raise AssertionError("explicit highs should not probe solver availability")

    monkeypatch.setattr(gurobi_backend, "available", fail_probe)
    monkeypatch.setattr(highs_backend, "available", fail_probe)

    assert resolve_master_backend("highs") == "highs"


def test_resolve_explicit_highs_loads_no_solver_modules_in_fresh_process() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    code = """
import importlib
import json
import sys

masters = importlib.import_module("combrum.masters")
resolved = masters.resolve_master_backend("highs")
watched = (
    "combrum.masters.gurobi",
    "combrum.masters.highs",
    "gurobipy",
    "highspy",
    "scipy",
)
print(json.dumps({
    "resolved": resolved,
    "loaded": [name for name in watched if name in sys.modules],
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)

    assert payload == {"resolved": "highs", "loaded": []}


def test_resolve_quadratic_auto_requires_gurobi(monkeypatch) -> None:
    monkeypatch.setattr(gurobi_backend, "available", lambda: False)
    monkeypatch.setattr(highs_backend, "available", lambda: True)

    with pytest.raises(RuntimeError, match="quadratic-capable"):
        resolve_master_backend("auto", require_quadratic=True)


def test_resolve_with_transport_probes_owner_rank_only(monkeypatch) -> None:
    calls: list[str] = []

    def fake_available() -> bool:
        calls.append(threading.current_thread().name)
        return True

    monkeypatch.setattr(gurobi_backend, "available", lambda: False)
    monkeypatch.setattr(highs_backend, "available", fake_available)
    results = LocalCluster(3).run(
        lambda transport: resolve_master_backend(
            "auto", transport=transport, owner_rank=1
        )
    )

    assert results == ["highs", "highs", "highs"]
    # One probe, on the declared owner rank only.
    assert calls == ["local-rank-1"]


def test_resolve_with_transport_intersects_all_owner_ranks(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def gurobi_available() -> bool:
        name = threading.current_thread().name
        calls.append(("gurobi", name))
        return name != "local-rank-1"

    def highs_available() -> bool:
        name = threading.current_thread().name
        calls.append(("highs", name))
        return True

    monkeypatch.setattr(gurobi_backend, "available", gurobi_available)
    monkeypatch.setattr(highs_backend, "available", highs_available)
    results = LocalCluster(3).run(
        lambda transport: resolve_master_backend(
            "auto", transport=transport, owner_ranks=(0, 1)
        )
    )

    assert results == ["highs", "highs", "highs"]
    assert sorted(calls) == [
        ("gurobi", "local-rank-0"),
        ("gurobi", "local-rank-1"),
        ("highs", "local-rank-0"),
        ("highs", "local-rank-1"),
    ]


def test_resolve_with_transport_prefers_gurobi_when_owner_intersection_has_it(
    monkeypatch,
) -> None:
    # With gurobi available on every owner rank, the resolver's gurobi
    # preference decides, and all ranks must agree on it.
    monkeypatch.setattr(gurobi_backend, "available", lambda: True)
    monkeypatch.setattr(highs_backend, "available", lambda: True)
    results = LocalCluster(3).run(
        lambda transport: resolve_master_backend(
            "auto", transport=transport, owner_ranks=(0, 1)
        )
    )

    assert results == ["gurobi", "gurobi", "gurobi"]


def make_row(
    agent_id: int, key: bytes, phi: tuple[float, ...], epsilon: float
) -> CutRow:
    return CutRow(
        rep_id=0,
        agent_id=agent_id,
        phi=np.asarray(phi, dtype=np.float64),
        epsilon=epsilon,
        bundle_key=key,
    )


# Pinned non-degenerate LP (K=2, agents 1 and 2):
#
#   minimize  0.25*t0 - 1.25*t1 + 1.0*u1 + 1.5*u2,  t in [-5, 5]^2
#
# rows (agent, key, phi, eps):     value at the optimum:
#   (1, a)  u1 >=  t0      + 2     active
#   (1, b)  u1 >=  t1      + 1     active
#   (2, c)  u2 >=  t0 + t1 - 1     active
#   (2, d)  u2 >= -t0      + 3     active
#   (1, e)  u1 >=  t0 + t1 - 1     slack (2 < u1)
#   (2, f)  u2 >=      -t1 + 0.5   slack (-1.5 < u2)
#
# The four active rows pin (t0, t1, u1, u2) = (1, 2, 3, 2) uniquely:
# a,b give t1 = t0 + 1; c,d give 2*t0 + t1 = 4. Objective = 3.75.
# Duals are unique by column stationarity (theta interior, both slacks
# basic): y_a + y_b = 1, y_c + y_d = 1.5, y_a + y_c - y_d = -0.25,
# y_b + y_c = 1.25  =>  y = (0.25, 0.75, 0.5, 1.0), slack rows 0.
LP_BOUNDS = (np.full(K, -5.0), np.full(K, 5.0))
LP_C_THETA = np.array([0.25, -1.25])
LP_U_COEF = {1: 1.0, 2: 1.5}
LP_ROWS = (
    make_row(1, b"a", (1.0, 0.0), 2.0),
    make_row(1, b"b", (0.0, 1.0), 1.0),
    make_row(2, b"c", (1.0, 1.0), -1.0),
    make_row(2, b"d", (-1.0, 0.0), 3.0),
    make_row(1, b"e", (1.0, 1.0), -1.0),
    make_row(2, b"f", (0.0, -1.0), 0.5),
)
LP_THETA = np.array([1.0, 2.0])
LP_OBJECTIVE = 3.75
LP_DUALS = {
    (1, b"a"): 0.25,
    (1, b"b"): 0.75,
    (2, b"c"): 0.5,
    (2, b"d"): 1.0,
    (1, b"e"): 0.0,
    (2, b"f"): 0.0,
}
LP_SLACKS = {
    (1, b"a"): 0.0,
    (1, b"b"): 0.0,
    (2, b"c"): 0.0,
    (2, b"d"): 0.0,
    (1, b"e"): 1.0,
    (2, b"f"): 3.5,
}

# Pinned at-bound LP (K=2, box [-0.5, 0.5]^2):
#
#   minimize  -2*t0 - 0.5*t1 + 2*u1 + 0.25*u2
#   rows: (1, a) u1 >= t1 + 0.25;  (2, b) u2 >= 4*t0 - 0.5*t1 + 0.375
#
# c0 = -2 sends t0 to its 0.5 upper bound; the agent-2 row is active
# with u2 = 2.5 basic (y_b = 0.25), so t0's reduced cost is
# -2 + 4*0.25 = -1. t1 sits strictly inside at the agent-1 kink -0.25
# (u1 = 0, y_a = 0.5 + 0.5*0.25 = 0.625). Optimum: theta = (0.5, -0.25).
BOUND_BOUNDS = (np.full(K, -0.5), np.full(K, 0.5))
BOUND_C_THETA = np.array([-2.0, -0.5])
BOUND_U_COEF = {1: 2.0, 2: 0.25}
BOUND_ROWS = (
    make_row(1, b"a", (0.0, 1.0), 0.25),
    make_row(2, b"b", (4.0, -0.5), 0.375),
)
BOUND_THETA = np.array([0.5, -0.25])
BOUND_DUAL_T0 = -1.0


def lp_master(backend: str) -> MasterBackend:
    return make_master(
        K,
        LP_BOUNDS,
        LP_C_THETA,
        LP_U_COEF.__getitem__,
        backend=backend,
    )


def bound_master(backend: str) -> MasterBackend:
    return make_master(
        K,
        BOUND_BOUNDS,
        BOUND_C_THETA,
        BOUND_U_COEF.__getitem__,
        backend=backend,
    )


def negative_epigraph_master(
    backend: str, *, u_lower_bound: float | None = 0.0
) -> MasterBackend:
    return make_master(
        K,
        (np.zeros(K), np.zeros(K)),
        np.zeros(K),
        lambda _agent_id: 1.0,
        backend=backend,
        params={"u_lower_bound": u_lower_bound},
    )


@pytest.fixture(params=REAL_BACKENDS)
def solved_lp(request: pytest.FixtureRequest) -> Iterator[MasterBackend]:
    with lp_master(request.param) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        yield master


def test_known_lp_optimum(solved_lp: MasterBackend) -> None:
    np.testing.assert_allclose(
        solved_lp.theta(), LP_THETA, rtol=0, atol=1e-9
    )
    assert solved_lp.objective() == pytest.approx(LP_OBJECTIVE, abs=1e-9)


@needs_gurobi
@needs_highs
def test_backends_agree_on_known_lp() -> None:
    results = {}
    for backend in ("gurobi", "highs"):
        with lp_master(backend) as master:
            master.add_cuts(LP_ROWS)
            master.solve()
            results[backend] = (master.theta(), master.objective())
    theta_g, objective_g = results["gurobi"]
    theta_h, objective_h = results["highs"]
    np.testing.assert_allclose(theta_g, theta_h, rtol=0, atol=1e-9)
    assert objective_g == pytest.approx(objective_h, abs=1e-9)


def test_known_lp_duals(solved_lp: MasterBackend) -> None:
    duals = solved_lp.dual_values()
    assert duals == pytest.approx(LP_DUALS, abs=1e-9)
    theta = solved_lp.theta()

    # u_a is internal to the backend, but its positive objective
    # coefficient pins it to its lower envelope at any optimum, so it is
    # recoverable from theta and the installed rows.
    u_hat = {
        agent: max(
            0.0,
            max(
                float(row.phi @ theta) + row.epsilon
                for row in LP_ROWS
                if row.agent_id == agent
            ),
        )
        for agent in LP_U_COEF
    }
    for row in LP_ROWS:
        dual = duals[(row.agent_id, row.bundle_key)]
        assert dual >= -1e-9
        slack = u_hat[row.agent_id] - (float(row.phi @ theta) + row.epsilon)
        assert dual * slack == pytest.approx(0.0, abs=1e-9)

    # LP column optimality: every agent's slack is strictly positive
    # here, so its column is basic and its rows' duals sum to u_coef(a).
    for agent, coef in LP_U_COEF.items():
        assert u_hat[agent] > 0
        total = sum(
            duals[(row.agent_id, row.bundle_key)]
            for row in LP_ROWS
            if row.agent_id == agent
        )
        assert total == pytest.approx(coef, abs=1e-9)


def test_cut_readings_are_row_aligned_and_normalized(
    solved_lp: MasterBackend,
) -> None:
    readings = solved_lp.cut_readings(dual=True, slack=True)

    assert readings.keys == tuple(sorted(LP_DUALS))
    assert readings.dual_map() == pytest.approx(LP_DUALS, abs=1e-9)
    assert readings.slack_map() == pytest.approx(LP_SLACKS, abs=1e-9)
    assert readings.dual is not None and not readings.dual.flags.writeable
    assert readings.slack is not None and not readings.slack.flags.writeable

    slack_only = solved_lp.cut_readings(slack=True)
    assert slack_only.keys == readings.keys
    assert slack_only.dual is None
    assert slack_only.slack_map() == pytest.approx(LP_SLACKS, abs=1e-9)


@needs_highs
def test_cut_readings_realign_when_install_order_is_unsorted() -> None:
    # cut_readings labels its arrays by sorted key, but the highs backend
    # caches per-row solver signals in install order and must gather them back
    # into sorted order. Adding agent 2's rows before agent 1's makes install
    # order != sorted order. (Gurobi rebuilds its readings from sorted keys
    # directly, so this path is highs-specific.)
    agent2 = tuple(row for row in LP_ROWS if row.agent_id == 2)
    agent1 = tuple(row for row in LP_ROWS if row.agent_id == 1)
    with lp_master("highs") as master:
        master.add_cuts(agent2)
        master.add_cuts(agent1)
        master.solve()

        # highs keys dual_values() by install order; confirm the split really
        # left it unsorted.
        install_order = tuple(master.dual_values())
        assert install_order != tuple(sorted(install_order))

        readings = master.cut_readings(dual=True, slack=True)
        assert readings.keys == tuple(sorted(LP_DUALS))
        assert readings.dual_map() == pytest.approx(LP_DUALS, abs=1e-9)
        assert readings.slack_map() == pytest.approx(LP_SLACKS, abs=1e-9)
        # dual_values bypasses the sorted-order gather; the two views must
        # agree key for key.
        assert master.dual_values() == pytest.approx(readings.dual_map(), abs=1e-9)


def test_bound_duals_empty_on_interior_lp(solved_lp: MasterBackend) -> None:
    assert solved_lp.bound_duals() == {}


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_bound_duals_at_bound(backend: str) -> None:
    with bound_master(backend) as master:
        master.add_cuts(BOUND_ROWS)
        master.solve()
        np.testing.assert_allclose(
            master.theta(), BOUND_THETA, rtol=0, atol=1e-9
        )
        duals = master.bound_duals()
        assert set(duals) == {0}
        # Coordinate 0 sits on its upper bound of a minimization, so the
        # correct reduced-cost sign is nonpositive.
        assert duals[0] < 0
        assert duals[0] == pytest.approx(BOUND_DUAL_T0, abs=1e-9)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_u_lower_bound_none_makes_epigraph_variable_free(
    backend: str,
) -> None:
    row = make_row(0, b"negative", (0.0, 0.0), -2.0)

    with negative_epigraph_master(backend) as bounded:
        bounded.add_cuts((row,))
        bounded.solve()
        assert bounded.u_values()[0] == pytest.approx(0.0, abs=1e-9)
        assert bounded.objective() == pytest.approx(0.0, abs=1e-9)

    with negative_epigraph_master(backend, u_lower_bound=None) as free:
        free.add_cuts((row,))
        free.solve()
        assert free.u_values()[0] == pytest.approx(-2.0, abs=1e-9)
        assert free.objective() == pytest.approx(-2.0, abs=1e-9)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_u_values_publish_cut_bearing_agents_with_predeclared_columns(
    backend: str,
) -> None:
    row = make_row(2, b"only-agent-two", (0.0, 0.0), 1.0)
    with make_master(
        K,
        LP_BOUNDS,
        np.zeros(K, dtype=np.float64),
        lambda _agent_id: 1.0,
        backend=backend,
        n_agents=5,
    ) as master:
        master.add_cuts((row,))
        master.solve()
        assert set(master.u_values()) == {2}
        assert master.u_values()[2] == pytest.approx(1.0, abs=1e-9)


def _rows_with_epsilon(
    key: tuple[int, bytes], new_eps: float
) -> tuple[CutRow, ...]:
    # LP_ROWS with one epsilon overwritten, for fresh-build comparisons
    # against an in-place set_rhs.
    return tuple(
        replace(row, epsilon=new_eps)
        if (row.agent_id, row.bundle_key) == key
        else row
        for row in LP_ROWS
    )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_matches_fresh_build_without_rebuild(backend: str) -> None:
    # Overwriting one cut's RHS in place must land the same relaxation a
    # fresh master gets from the modified row set. Row (1, a): eps 2.0 -> 3.5.
    key = (1, b"a")
    new_eps = 3.5
    with lp_master(backend) as persistent:
        persistent.add_cuts(LP_ROWS)
        persistent.solve()
        model_before = getattr(persistent, "_model", None) or getattr(
            persistent, "_h"
        )

        persistent.set_rhs({key: new_eps})
        persistent.solve()

        # No rebuild: the live solver handle is the same instance.
        model_after = getattr(persistent, "_model", None) or getattr(
            persistent, "_h"
        )
        assert model_after is model_before

        with lp_master(backend) as fresh:
            fresh.add_cuts(_rows_with_epsilon(key, new_eps))
            fresh.solve()
            np.testing.assert_allclose(
                persistent.theta(), fresh.theta(), rtol=0, atol=1e-9
            )
            assert persistent.objective() == pytest.approx(
                fresh.objective(), abs=1e-9
            )
            assert persistent.dual_values() == pytest.approx(
                fresh.dual_values(), abs=1e-9
            )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_updates_extracted_epsilon_phi_unchanged(
    backend: str,
) -> None:
    key = (1, b"a")
    new_eps = -4.25
    with lp_master(backend) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        phi_before = {
            (row.agent_id, row.bundle_key): row.phi.tobytes()
            for row in master.extract_cuts()
        }

        master.set_rhs({key: new_eps})

        extracted = _rows_by_key(master)
        assert extracted[key].epsilon == new_eps
        for row in LP_ROWS:
            rkey = (row.agent_id, row.bundle_key)
            if rkey != key:
                assert extracted[rkey].epsilon == row.epsilon
            assert extracted[rkey].phi.tobytes() == phi_before[rkey]
        # The internal installed mirror agrees with the extracted view.
        assert master._installed[key].epsilon == new_eps

        # The checks above only read the Python-side mirror, so a set_rhs
        # that skipped the solver-side RHS write would still pass them.
        # Dropping (1, a) to eps -4.25 slackens it out of the active set and
        # moves the optimum from (1, 2)/3.75 to (2, 0)/3.0 (hand-derived);
        # re-solve and check the solver actually moved.
        master.solve()
        moved_theta = np.array([2.0, 0.0])
        moved_objective = 3.0
        np.testing.assert_allclose(
            master.theta(), moved_theta, rtol=0, atol=1e-9
        )
        assert master.objective() == pytest.approx(moved_objective, abs=1e-9)
        with lp_master(backend) as fresh:
            fresh.add_cuts(_rows_with_epsilon(key, new_eps))
            fresh.solve()
            np.testing.assert_allclose(
                master.theta(), fresh.theta(), rtol=0, atol=1e-9
            )
            assert master.objective() == pytest.approx(
                fresh.objective(), abs=1e-9
            )
            assert master.dual_values() == pytest.approx(
                fresh.dual_values(), abs=1e-9
            )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_invalidates_solution_until_resolve(backend: str) -> None:
    # Between set_rhs and the next solve the cached solution is stale;
    # every accessor must refuse rather than report the pre-edit problem.
    with lp_master(backend) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        master.set_rhs({(1, b"a"): 0.0})
        for accessor in (
            master.theta,
            master.objective,
            master.dual_values,
            master.bound_duals,
            master.u_values,
            master.cut_readings,
        ):
            with pytest.raises(RuntimeError, match="no solve"):
                accessor()
        master.solve()
        assert master.theta().shape == (K,)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_unknown_key_raises_key_error(backend: str) -> None:
    with lp_master(backend) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        with pytest.raises(KeyError):
            master.set_rhs({(99, b"missing"): 1.0})


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_missing_key_after_valid_does_not_partially_mutate(
    backend: str,
) -> None:
    # set_rhs validates the whole key set before writing, so a missing key
    # aborts the update with nothing rewritten — even for valid keys ordered
    # ahead of it.
    valid = (1, b"a")
    with lp_master(backend) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        theta_before = master.theta().tobytes()
        eps_before = master._installed[valid].epsilon

        with pytest.raises(KeyError):
            master.set_rhs({valid: 3.5, (99, b"missing"): 1.0})

        # The valid key's epsilon is untouched in mirror and extract.
        assert master._installed[valid].epsilon == eps_before
        extracted = _rows_by_key(master)
        assert extracted[valid].epsilon == eps_before
        # The relaxation never moved, so the cached solve stays valid.
        assert master.theta().tobytes() == theta_before


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_rhs_leaves_add_extract_reinstall_unchanged(backend: str) -> None:
    # set_rhs must not perturb the other cut primitives: extract still
    # round-trips through reinstall bitwise, and add_cuts still dedups.
    with lp_master(backend) as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        master.set_rhs({(1, b"a"): 3.5})
        master.solve()
        artifact = master.extract_cuts()
        assert master.add_cuts(LP_ROWS) == 0

        with lp_master(backend) as fresh:
            fresh.reinstall(artifact)
            fresh.solve()
            master.solve()
            assert master.theta().tobytes() == fresh.theta().tobytes()
            assert master.objective() == fresh.objective()
            assert master.dual_values() == fresh.dual_values()


@needs_gurobi
def test_penalty_pulls_toward_ref_then_reverts_exactly() -> None:
    with lp_master("gurobi") as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        theta_lp = master.theta()
        objective_lp = master.objective()
        duals_lp = master.dual_values()
        assert master.bound_duals() == {}

        ref = np.zeros(K)
        master.set_penalty(ref, weight=10.0)
        master.solve()
        theta_pen = master.theta()
        # Strictly toward ref, and strictly interior — which also walks
        # the no-basis proximity path of bound_duals under the penalty.
        assert np.linalg.norm(theta_pen - ref) < np.linalg.norm(
            theta_lp - ref
        )
        assert master.bound_duals() == {}
        # Exact QP optimum: near theta = 0 the active epigraph rows are
        # (1,a) u1 = t0+2 and (2,d) u2 = -t0+3, so the objective reduces to
        # -0.25*t0 - 1.25*t1 + 6.5 + 10*(t0^2 + t1^2), stationary at
        # t0 = 0.25/20, t1 = 1.25/20 (cross-checked with a scipy QP solve).
        np.testing.assert_allclose(
            theta_pen, np.array([0.0125, 0.0625]), rtol=0, atol=1e-9
        )

        master.set_penalty(ref, weight=0.0)
        master.solve()
        # weight=0.0 removes the penalty outright rather than leaving a
        # zero-weight QP hanging: the model reverts to a pure LP and the
        # reported objective is strictly linear.
        assert master._penalty is None
        np.testing.assert_allclose(
            master.theta(), theta_lp, rtol=0, atol=1e-12
        )
        assert master.objective() == pytest.approx(objective_lp, abs=1e-12)
        assert master.dual_values() == pytest.approx(duals_lp, abs=1e-9)
        assert master.bound_duals() == {}
        reverted_theta = master.theta()
        reverted_u = master.u_values()
        linear_only = float(LP_C_THETA @ reverted_theta) + sum(
            LP_U_COEF[agent_id] * reverted_u[agent_id] for agent_id in LP_U_COEF
        )
        assert master.objective() == pytest.approx(linear_only, abs=1e-9)


@needs_gurobi
def test_penalty_objective_reports_linear_plus_exact_quadratic() -> None:
    with lp_master("gurobi") as master:
        master.add_cuts(LP_ROWS)
        ref = np.array([-1.5, 4.0], dtype=np.float64)
        weight = 2.25

        master.set_penalty(ref, weight=weight)
        master.solve()

        theta = master.theta()
        u_values = master.u_values()
        linear = float(LP_C_THETA @ theta) + sum(
            LP_U_COEF[agent_id] * u_values[agent_id]
            for agent_id in LP_U_COEF
        )
        quadratic = weight * float((theta - ref) @ (theta - ref))
        assert master.objective() == pytest.approx(
            linear + quadratic, abs=1e-9
        )


@needs_gurobi
def test_bound_duals_under_active_penalty_then_revert() -> None:
    with bound_master("gurobi") as master:
        master.add_cuts(BOUND_ROWS)
        master.solve()
        duals_lp = master.bound_duals()
        assert set(duals_lp) == {0}

        # A penalty centered on the LP optimum keeps that optimum, so
        # theta_0 stays ON its bound while the solve is a QP — the case
        # the proximity fallback exists for.
        master.set_penalty(BOUND_THETA, weight=1.0)
        master.solve()
        duals_pen = master.bound_duals()
        assert set(duals_pen) == {0}
        assert duals_pen[0] < 0
        # The proximity path must report the same -1.0 reduced cost the
        # LP/VBasis path does — the value, not just the sign.
        assert duals_pen[0] == pytest.approx(BOUND_DUAL_T0, abs=1e-9)

        master.set_penalty(BOUND_THETA, weight=0.0)
        master.solve()
        # After weight=0.0 the terminating solve is a true LP whose bound dual
        # comes off the simplex VBasis, not a residual zero-weight QP on the
        # proximity path. Both report -1.0 here, so pin the teardown itself.
        assert master._penalty is None
        assert master.bound_duals() == pytest.approx(duals_lp, abs=1e-9)


@needs_highs
def test_highs_slack_upper_bound_is_exact_theta_box_envelope() -> None:
    with make_master(
        K,
        (np.array([-1.0, -2.0]), np.array([3.0, 4.0])),
        np.zeros(K, dtype=np.float64),
        lambda _agent_id: 1.0,
        backend="highs",
        n_agents=1,
    ) as master:
        assert isinstance(master, HighsMaster)
        master.add_cuts((make_row(0, b"a", (2.0, -1.0), 5.0),))
        lp = master._h.getLp()
        assert lp.col_upper_[K] == pytest.approx(13.0, abs=1e-12)

        master.set_rhs({(0, b"a"): 7.0})
        lp = master._h.getLp()
        assert lp.col_upper_[K] == pytest.approx(15.0, abs=1e-12)


@needs_highs
def test_highs_free_slack_keeps_solver_native_infinite_upper_bound() -> None:
    with make_master(
        K,
        LP_BOUNDS,
        np.zeros(K, dtype=np.float64),
        lambda _agent_id: 1.0,
        backend="highs",
        params={"u_lower_bound": None},
    ) as master:
        assert isinstance(master, HighsMaster)
        master.add_cuts((make_row(0, b"a", (1.0, 0.0), 1.0),))

        lp = master._h.getLp()
        assert lp.col_upper_[K] == pytest.approx(master._highspy.kHighsInf)


@needs_highs
def test_highs_rejects_nonfinite_cut_without_solver_mirror_drift() -> None:
    with make_master(
        K,
        LP_BOUNDS,
        np.zeros(K, dtype=np.float64),
        lambda _agent_id: 1.0,
        backend="highs",
        params={"u_lower_bound": None},
    ) as master:
        assert isinstance(master, HighsMaster)
        bad = CutRow(0, 7, np.array([np.nan, 0.0]), 1.0, b"bad")

        with pytest.raises(ValueError, match="phi must be finite"):
            master.add_cuts((bad,))

        assert master.extract_cuts() == ()
        assert master.n_active_cuts == 0
        assert master._h.getNumRow() == 0
        assert master._h.getNumCol() == K


@needs_highs
def test_highs_rejects_nonfinite_cut_without_lazy_column_leak() -> None:
    with lp_master("highs") as master:
        assert isinstance(master, HighsMaster)
        bad = CutRow(0, 7, np.array([1.0, 0.0]), np.inf, b"bad")

        with pytest.raises(ValueError, match="epsilon must be finite"):
            master.add_cuts((bad,))

        assert master.extract_cuts() == ()
        assert master._h.getNumRow() == 0
        assert master._h.getNumCol() == K


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_set_penalty_validates_ref_shape(backend: str) -> None:
    with lp_master(backend) as master:
        with pytest.raises(ValueError, match="shape"):
            master.set_penalty(np.zeros(K + 1), weight=1.0)
        # Shape validity is not conditional on the weight branch.
        with pytest.raises(ValueError, match="shape"):
            master.set_penalty(np.zeros(K + 1), weight=0.0)


@needs_gurobi
@needs_highs
def test_extracted_cuts_warm_start_across_backends() -> None:
    with lp_master("gurobi") as producer:
        producer.add_cuts(LP_ROWS)
        producer.solve()
        artifact = producer.extract_cuts()
        theta_g = producer.theta()
        objective_g = producer.objective()

    with lp_master("highs") as consumer:
        consumer.reinstall(artifact)
        consumer.solve()
        np.testing.assert_allclose(
            consumer.theta(), theta_g, rtol=0, atol=1e-9
        )
        assert consumer.objective() == pytest.approx(objective_g, abs=1e-9)


@pytest.mark.parametrize(
    ("backend", "expected_cls"),
    (
        pytest.param("gurobi", GurobiMaster, marks=needs_gurobi, id="gurobi"),
        pytest.param("highs", HighsMaster, marks=needs_highs, id="highs"),
    ),
)
def test_factory_explicit_backend(
    backend: str, expected_cls: type[MasterBackend]
) -> None:
    with lp_master(backend) as master:
        assert isinstance(master, expected_cls)


@needs_gurobi
def test_master_environment_shares_one_gurobi_env() -> None:
    with master_environment("gurobi") as env:
        assert env is not None
        # Sequential masters on the shared environment: each close() releases
        # only the model, so the next build needs no fresh license checkout.
        for _ in range(2):
            with make_master(
                K,
                LP_BOUNDS,
                LP_C_THETA,
                LP_U_COEF.__getitem__,
                backend="gurobi",
                env=env,
            ) as master:
                assert master._env is env
                master.add_cuts(LP_ROWS)
                master.solve()
                np.testing.assert_allclose(
                    master.theta(), LP_THETA, rtol=0, atol=1e-9
                )


def test_master_environment_yields_none_for_highs() -> None:
    with master_environment("highs") as env:
        assert env is None


@pytest.mark.skipif(
    not (GUROBI_AVAILABLE or HIGHS_AVAILABLE),
    reason="no real backend available",
)
def test_u_coef_array_matches_callable() -> None:
    # Index 0 is a cutless agent whose coefficient is never read.
    coefs = np.array([0.0, LP_U_COEF[1], LP_U_COEF[2]], dtype=np.float64)
    with lp_master("auto") as by_callable:
        by_callable.add_cuts(LP_ROWS)
        by_callable.solve()
        with make_master(
            K,
            LP_BOUNDS,
            LP_C_THETA,
            coefs,
            backend="auto",
        ) as by_array:
            by_array.add_cuts(LP_ROWS)
            by_array.solve()
            assert by_array.theta().tobytes() == by_callable.theta().tobytes()
            assert by_array.objective() == by_callable.objective()

    with pytest.raises(ValueError, match="must cover all"):
        make_master(
            K,
            LP_BOUNDS,
            LP_C_THETA,
            coefs[:1],
            backend="auto",
            n_agents=coefs.size,
        )


@needs_highs
def test_make_master_rejects_env_for_highs() -> None:
    with pytest.raises(ValueError, match="only meaningful for the gurobi backend"):
        make_master(
            K,
            LP_BOUNDS,
            LP_C_THETA,
            LP_U_COEF.__getitem__,
            backend="highs",
            env=object(),
        )


@needs_highs
def test_highs_defaults_to_simplex_solver_unless_overridden() -> None:
    with lp_master("highs") as master:
        assert isinstance(master, HighsMaster)
        assert master._h is not None
        status, value = master._h.getOptionValue("solver")
        assert status == master._highspy.HighsStatus.kOk
        assert value == "simplex"

    with make_master(
        K,
        LP_BOUNDS,
        LP_C_THETA,
        LP_U_COEF.__getitem__,
        backend="highs",
        params={"solver": "choose"},
    ) as master:
        assert isinstance(master, HighsMaster)
        assert master._h is not None
        status, value = master._h.getOptionValue("solver")
        assert status == master._highspy.HighsStatus.kOk
        assert value == "choose"


@pytest.mark.skipif(
    not (GUROBI_AVAILABLE or HIGHS_AVAILABLE),
    reason="no real backend available",
)
def test_factory_auto_returns_a_solving_master() -> None:
    with lp_master("auto") as master:
        master.add_cuts(LP_ROWS)
        master.solve()
        np.testing.assert_allclose(
            master.theta(), LP_THETA, rtol=0, atol=1e-9
        )


def test_factory_unknown_backend_names_the_valid_set() -> None:
    with pytest.raises(ValueError, match=r"'auto', 'gurobi', 'highs'"):
        lp_master("cplex")


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_construction_validation(backend: str) -> None:
    # The two backends validate independently; pinning both here keeps
    # their construction errors from drifting apart.
    with pytest.raises(ValueError, match="c_theta"):
        make_master(
            K, LP_BOUNDS, np.zeros(K + 1), LP_U_COEF.__getitem__,
            backend=backend,
        )
    with pytest.raises(ValueError, match="lower"):
        make_master(
            K,
            (np.full(K, 1.0), np.full(K, -1.0)),
            LP_C_THETA,
            LP_U_COEF.__getitem__,
            backend=backend,
        )
    with pytest.raises(ValueError, match="K"):
        make_master(
            0, LP_BOUNDS, LP_C_THETA, LP_U_COEF.__getitem__, backend=backend
        )


def _run_import_probe(probe: str, tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=str(src))
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )


def test_import_masters_pulls_no_solver(tmp_path: Path) -> None:
    # Importing the package must not import any solver. Probe in a fresh
    # subprocess (this pytest process has both loaded) from a neutral cwd
    # so resolution can't lean on the repo root.
    _run_import_probe(
        "import sys\n"
        "import combrum.masters\n"
        "assert 'gurobipy' not in sys.modules\n"
        "assert 'highspy' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n",
        tmp_path,
    )


def test_import_gurobi_backend_pulls_no_scipy(tmp_path: Path) -> None:
    _run_import_probe(
        "import sys\n"
        "import combrum.masters.gurobi\n"
        "assert 'scipy' not in sys.modules\n",
        tmp_path,
    )


def test_scipy_dependency_is_gurobi_extra_only() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    assert pyproject["project"]["dependencies"] == ["numpy>=1.24"]
    assert "scipy>=1.10" in pyproject["project"]["optional-dependencies"][
        "gurobi"
    ]
