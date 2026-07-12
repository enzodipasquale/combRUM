from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest

from combrum.master import MasterBackend
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.transport import CutRow

K = 2

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

# The conformance battery runs every contract property over the
# reference double and both real backends. The double ignores the
# objective data (its solve is the clipped phi mean), so each test picks
# a construction whose real optimum makes the shared assertions true on
# all three; the certificates are spelled out where the rows live.
ALL_BACKENDS = (
    "fake",
    pytest.param(
        "gurobi",
        marks=pytest.mark.skipif(
            not GUROBI_AVAILABLE,
            reason="gurobipy missing or no environment starts",
        ),
    ),
    pytest.param(
        "highs",
        marks=pytest.mark.skipif(
            not HIGHS_AVAILABLE, reason="highspy missing or broken"
        ),
    ),
)
# Only quadratic-capable backends install/revert the penalty; the highs
# hard error has its own test below.
PENALTY_CAPABLE = ("fake", ALL_BACKENDS[1])


def make_row(
    agent_id: int, key: bytes, phi: Sequence[float], epsilon: float = 0.0
) -> CutRow:
    return CutRow(
        rep_id=0, agent_id=agent_id, phi=np.asarray(phi), epsilon=epsilon,
        bundle_key=key,
    )


class FakeMaster(MasterBackend):
    """Test double proving the contract is implementable.

    The 'relaxation' is deterministic: theta is the mean of the installed
    phi rows (pulled toward the penalty reference when one is set),
    clipped into the theta box. This makes every contract property
    observable: dedup counts, canonical extraction, the pure-LP flag, and
    bound duals when the clip engages.
    """

    def __init__(
        self, K: int, lower: Sequence[float], upper: Sequence[float]
    ) -> None:
        self._K = K
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)
        self._installed: dict[tuple[int, bytes], CutRow] = {}
        self._theta = np.zeros(K, dtype=np.float64)
        self._objective = 0.0
        self._bound_duals: dict[int, float] = {}
        self._penalty: tuple[np.ndarray, float] | None = None

    @property
    def pure_lp(self) -> bool:
        return self._penalty is None

    def _rows(self) -> tuple[CutRow, ...]:
        return tuple(
            self._installed[key] for key in sorted(self._installed)
        )

    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        new = 0
        for row in rows:
            key = (row.agent_id, row.bundle_key)
            if key not in self._installed:
                self._installed[key] = row
                new += 1
        return new

    def solve(self) -> None:
        rows = self._rows()
        if rows:
            raw = np.mean(np.stack([row.phi for row in rows]), axis=0)
        else:
            raw = np.zeros(self._K, dtype=np.float64)
        if self._penalty is not None:
            ref, weight = self._penalty
            # Proximal pull, so an installed penalty is visible in theta.
            raw = (raw + weight * ref) / (1.0 + weight)
        theta = np.clip(raw, self._lower, self._upper)
        self._bound_duals = {
            int(k): float(raw[k] - theta[k])
            for k in np.flatnonzero(theta != raw)
        }
        self._theta = theta
        self._objective = float(theta @ theta) + sum(
            row.epsilon for row in rows
        )

    def theta(self) -> np.ndarray:
        return self._theta

    def objective(self) -> float:
        return self._objective

    def u_values(self) -> dict[int, float]:
        u: dict[int, float] = {}
        for row in self._rows():
            value = float(row.phi @ self._theta + row.epsilon)
            held = u.get(row.agent_id)
            if held is None or value > held:
                u[row.agent_id] = value
        return {a: max(0.0, value) for a, value in u.items()}

    def dual_values(self) -> dict[tuple[int, bytes], float]:
        return {
            (row.agent_id, row.bundle_key): float(row.epsilon)
            for row in self._rows()
        }

    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        if weight <= 0:
            self._penalty = None  # revert to a pure LP
        else:
            self._penalty = (np.asarray(ref, dtype=np.float64), float(weight))

    def extract_cuts(self) -> tuple[CutRow, ...]:
        return self._rows()

    def reinstall(self, rows: Sequence[CutRow]) -> None:
        self._installed = {}
        self.add_cuts(rows)

    def bound_duals(self) -> dict[int, float]:
        return dict(self._bound_duals)


def _default_u(agent_id: int) -> float:
    return 0.5


# Interior LP (real backends): minimize 0.25*t0 - 1.25*t1 + u1 + 1.5*u2
# over the default box with the four rows below. All four rows are
# active and pin the unique nondegenerate optimum (t0, t1, u1, u2) =
# (1, 2, 3, 2) — theta strictly interior. The double lands interior
# too: its clipped phi mean is (0.25, 0.5).
INTERIOR_C = (0.25, -1.25)


def _interior_u(agent_id: int) -> float:
    return {1: 1.0, 2: 1.5}[agent_id]


INTERIOR_ROWS = (
    make_row(1, b"a", [1.0, 0.0], epsilon=2.0),
    make_row(1, b"b", [0.0, 1.0], epsilon=1.0),
    make_row(2, b"c", [1.0, 1.0], epsilon=-1.0),
    make_row(2, b"d", [-1.0, 0.0], epsilon=3.0),
)

# All four INTERIOR_ROWS bind at the optimum, so the row gradients A (in
# `u_agent - phi.theta >= epsilon`) form a square full-rank system and the
# cut duals are the unique solution of A^T pi = c with c = (0.25, -1.25,
# u1=1.0, u2=1.5): pi = (0.25, 0.75, 0.5, 1.0), each strictly positive.
# These are the true LP multipliers, deliberately distinct from the rows'
# own epsilons (2, 1, -1, 3) and from the slack coefficients u_coef, so a
# dual accessor that reports epsilon (or u_coef) instead of Pi is caught.
INTERIOR_CUT_DUALS = {
    (1, b"a"): 0.25,
    (1, b"b"): 0.75,
    (2, b"c"): 0.5,
    (2, b"d"): 1.0,
}
# The double reports each row's epsilon by design (its solve ignores the
# objective data), so its cut-dual map is the epsilon map, not Pi.
INTERIOR_FAKE_DUALS = {
    (row.agent_id, row.bundle_key): row.epsilon for row in INTERIOR_ROWS
}


def _at_bound_u(agent_id: int) -> float:
    return {1: 2.0, 2: 0.25}[agent_id]


def _make_real(
    backend: str,
    lower: float,
    upper: float,
    c_theta: Sequence[float],
    u_coef: Callable[[int], float],
) -> MasterBackend:
    return make_master(
        K,
        (
            np.full(K, lower, dtype=np.float64),
            np.full(K, upper, dtype=np.float64),
        ),
        np.asarray(c_theta, dtype=np.float64),
        u_coef,
        backend=backend,
    )


Builder = Callable[..., MasterBackend]


def _closing_builder(backend: str) -> Iterator[Builder]:
    made: list[MasterBackend] = []

    def build(
        lower: float = -10.0,
        upper: float = 10.0,
        c_theta: Sequence[float] = (-1.0, -1.5),
        u_coef: Callable[[int], float] = _default_u,
    ) -> MasterBackend:
        if backend == "fake":
            master: MasterBackend = FakeMaster(K, [lower] * K, [upper] * K)
        else:
            master = _make_real(backend, lower, upper, c_theta, u_coef)
        made.append(master)
        return master

    yield build
    # close() is implementation-owned, not part of the ABC: real
    # backends hold solver handles, the double holds nothing.
    for master in made:
        close = getattr(master, "close", None)
        if close is not None:
            close()


@pytest.fixture(params=ALL_BACKENDS)
def build(request: pytest.FixtureRequest) -> Iterator[Builder]:
    yield from _closing_builder(request.param)


@pytest.fixture(params=PENALTY_CAPABLE)
def build_penalty_capable(
    request: pytest.FixtureRequest,
) -> Iterator[Builder]:
    yield from _closing_builder(request.param)


def test_abc_not_instantiable() -> None:
    with pytest.raises(TypeError):
        MasterBackend()  # type: ignore[abstract]


def test_master_implements_contract(build: Builder) -> None:
    master = build()
    assert isinstance(master, MasterBackend)
    master.add_cuts([make_row(1, b"a", [1.0, 2.0], epsilon=0.5)])
    master.solve()
    assert master.theta().shape == (K,)
    assert isinstance(master.objective(), float)
    # 0.5 on every backend: the double reports epsilon; on the real LP the
    # default c = (-1, -1.5) sends theta to its upper bounds, the row stays
    # active with the slack basic, so the row dual is u_coef(1) = 0.5.
    assert master.dual_values() == {(1, b"a"): 0.5}


def test_add_cuts_returns_new_count_under_duplicates(build: Builder) -> None:
    master = build()
    r1 = make_row(1, b"x", [1.0, 0.0])
    r1_dup = make_row(1, b"x", [1.0, 0.0])  # same (agent_id, bundle_key)
    r2 = make_row(2, b"y", [0.0, 1.0])
    # Within-batch duplicate: counted once.
    assert master.add_cuts([r1, r1_dup, r2]) == 2
    # Versus-installed duplicate: counted zero.
    assert master.add_cuts([r1]) == 0
    # Mixed batch: only the genuinely new key counts.
    assert master.add_cuts([r1_dup, make_row(2, b"z", [1.0, 1.0])]) == 1


def test_extract_cuts_canonical_order(build: Builder) -> None:
    master = build()
    scrambled = [
        make_row(2, b"b", [0.0, 1.0]),
        make_row(1, b"z", [1.0, 0.0]),
        make_row(1, b"a", [0.5, 0.5]),
        make_row(2, b"a", [1.0, 1.0]),
    ]
    master.add_cuts(scrambled)
    keys = [(row.agent_id, row.bundle_key) for row in master.extract_cuts()]
    assert keys == sorted(keys)
    assert keys == [(1, b"a"), (1, b"z"), (2, b"a"), (2, b"b")]


def test_extract_reinstall_round_trip(build: Builder) -> None:
    original = build()
    original.add_cuts(
        [
            make_row(3, b"k", [2.0, -1.0], epsilon=0.25),
            make_row(1, b"m", [0.0, 4.0], epsilon=-0.5),
            make_row(2, b"k", [1.0, 1.0], epsilon=0.0),
        ]
    )
    extracted = original.extract_cuts()

    fresh = build()
    fresh.reinstall(extracted)
    round_tripped = fresh.extract_cuts()

    assert [
        (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
        for row in round_tripped
    ] == [
        (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
        for row in extracted
    ]
    # Same relaxation: both masters solve to bitwise-identical state.
    original.solve()
    fresh.solve()
    assert original.theta().tobytes() == fresh.theta().tobytes()
    assert original.objective() == fresh.objective()
    assert original.dual_values() == fresh.dual_values()


def test_reinstall_replaces_installed_set_on_used_master(
    build: Builder,
) -> None:
    row_a = make_row(1, b"a", [1.0, 0.0], epsilon=1.0)
    row_b = make_row(2, b"b", [0.0, 1.0], epsilon=1.0)
    row_c = make_row(3, b"c", [1.0, 1.0], epsilon=2.0)

    used = build()
    used.add_cuts([row_a, row_b])
    used.solve()
    # replace on a used master: the prior installed set must vanish
    # entirely, not merge with the new rows.
    used.reinstall([row_c])
    assert used.extract_cuts() == (row_c,)
    used.solve()

    fresh = build()
    fresh.add_cuts([row_c])
    fresh.solve()
    assert set(used.dual_values()) == {(3, b"c")}
    assert used.theta().tobytes() == fresh.theta().tobytes()
    assert used.objective() == fresh.objective()
    assert used.dual_values() == fresh.dual_values()


def test_fake_double_pure_lp_flag_tracks_penalty_revert() -> None:
    # pure_lp exists on the double alone — not on the ABC, not on any real
    # backend — so this pins FakeMaster's own revert semantics. The real
    # backends are covered by test_set_penalty_installs_then_reverts_exactly
    # and test_set_penalty_unsupported_backend_is_a_hard_error.
    for real in (gurobi_backend.GurobiMaster, highs_backend.HighsMaster):
        assert not hasattr(real, "pure_lp")
    assert not hasattr(MasterBackend, "pure_lp")

    master = FakeMaster(K, [-10.0] * K, [10.0] * K)
    master.add_cuts([make_row(1, b"a", [4.0, 4.0])])
    assert master.pure_lp

    master.set_penalty(np.zeros(K), weight=1.0)
    assert not master.pure_lp
    master.solve()
    penalized = master.theta().copy()

    # weight <= 0 removes the penalty entirely, not approximately.
    master.set_penalty(np.zeros(K), weight=0.0)
    assert master.pure_lp
    master.solve()
    assert not np.array_equal(master.theta(), penalized)

    master.set_penalty(np.zeros(K), weight=-2.0)
    assert master.pure_lp


def test_set_penalty_installs_then_reverts_exactly(
    build_penalty_capable: Builder,
) -> None:
    master = build_penalty_capable(c_theta=INTERIOR_C, u_coef=_interior_u)
    master.add_cuts(list(INTERIOR_ROWS))
    master.solve()
    theta_lp = master.theta()
    objective_lp = master.objective()
    duals_lp = master.dual_values()

    ref = np.full(K, 5.0)
    master.set_penalty(ref, weight=4.0)
    master.solve()
    # Where the penalty lands, in closed form. The real backends minimize
    # 0.25*t0 - 1.25*t1 + u1 + 1.5*u2 + 4*||theta - 5||^2 over the four
    # INTERIOR_ROWS. Rows a (u1 >= t0+2) and c (u2 >= t0+t1-1) bind;
    # substituting them, the theta objective is
    # 2.75*t0 + 0.25*t1 + 0.5 + 4*[(t0-5)^2 + (t1-5)^2], stationary at
    # t0 = 5 - 2.75/8 = 149/32, t1 = 5 - 0.25/8 = 159/32, with the other
    # two rows slack. The double's pull is (mean + w*ref)/(1+w) = (4.05, 4.1).
    if isinstance(master, FakeMaster):
        np.testing.assert_allclose(
            master.theta(), [4.05, 4.1], rtol=0, atol=1e-12
        )
    else:
        np.testing.assert_allclose(
            master.theta(), [149.0 / 32, 159.0 / 32], rtol=0, atol=1e-6
        )
        assert master.objective() == pytest.approx(1923.0 / 128, abs=1e-6)
    assert not np.array_equal(master.theta(), theta_lp)

    master.set_penalty(ref, weight=0.0)
    master.solve()
    np.testing.assert_allclose(master.theta(), theta_lp, rtol=0, atol=1e-12)
    assert master.objective() == pytest.approx(objective_lp, abs=1e-12)
    assert master.dual_values() == pytest.approx(duals_lp, abs=1e-9)


@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="highspy missing or broken")
def test_set_penalty_unsupported_backend_is_a_hard_error() -> None:
    # A backend without native quadratic support must refuse weight > 0
    # loudly and treat weight <= 0 as the no-op it is; never approximate.
    with _make_real("highs", -10.0, 10.0, (-1.0, -1.5), _default_u) as master:
        master.add_cuts([make_row(1, b"a", [1.0, 2.0], epsilon=0.5)])
        master.solve()
        theta_before = master.theta()
        with pytest.raises(NotImplementedError, match="does not expose quadratic"):
            master.set_penalty(np.zeros(K), weight=1.0)
        master.set_penalty(np.zeros(K), weight=0.0)
        master.set_penalty(np.zeros(K), weight=-2.0)
        master.solve()
        assert np.array_equal(master.theta(), theta_before)


def test_set_rhs_default_raises_for_non_overriding_subclass() -> None:
    # set_rhs defaults to raising rather than being abstract: a subclass
    # that omits it still instantiates, but calling it fails loudly instead
    # of silently skipping the RHS rewrite. FakeMaster does not override it.
    master = FakeMaster(K, [-10.0] * K, [10.0] * K)
    master.add_cuts([make_row(1, b"a", [1.0, 0.0], epsilon=1.0)])
    with pytest.raises(NotImplementedError, match="set_rhs"):
        master.set_rhs({(1, b"a"): 2.0})


def test_bound_duals_empty_when_interior(build: Builder) -> None:
    master = build(c_theta=INTERIOR_C, u_coef=_interior_u)
    master.add_cuts(list(INTERIOR_ROWS))
    master.solve()
    assert master.bound_duals() == {}
    # The real optimum is (t0,t1,u1,u2) = (1,2,3,2), so with c=(0.25,-1.25)
    # and u_coef {1:1,2:1.5} the objective is 0.25 - 2.5 + 3 + 3 = 3.75.
    # The double reports theta@theta + sum(epsilon) at its clipped mean
    # (0.25,0.5): 0.0625 + 0.25 + 5 = 5.3125.
    if isinstance(master, FakeMaster):
        assert master.objective() == pytest.approx(5.3125, abs=1e-12)
    else:
        assert master.objective() == pytest.approx(3.75, abs=1e-9)
    # At this nondegenerate point the hand-solved multipliers (Pi) differ
    # per key from the rows' epsilons, so dual_values must be reporting Pi.
    # The double reports epsilon by design.
    if isinstance(master, FakeMaster):
        assert master.dual_values() == pytest.approx(INTERIOR_FAKE_DUALS)
    else:
        assert master.dual_values() == pytest.approx(
            INTERIOR_CUT_DUALS, abs=1e-9
        )


def test_bound_duals_nonempty_at_bound(build: Builder) -> None:
    # Real backends: c0 = -2 pushes theta_0 onto the 0.5 upper bound
    # (the agent-2 row prices it at reduced cost -2 + 4*0.25 = -1) while
    # the agent-1 row pins theta_1 strictly inside at -0.25. The double
    # lands the same shape: clipped phi mean (2, 0.25) -> (0.5, 0.25).
    master = build(
        lower=-0.5, upper=0.5, c_theta=(-2.0, -0.5), u_coef=_at_bound_u
    )
    master.add_cuts(
        [
            make_row(1, b"a", [0.0, 1.0], epsilon=0.25),
            make_row(2, b"b", [4.0, -0.5], epsilon=0.375),
        ]
    )
    master.solve()
    duals = master.bound_duals()
    assert set(duals) == {0}
    assert isinstance(duals[0], float)
    assert master.theta()[0] == 0.5
    # Value with sign: on a real solver theta_0 sits on its upper bound
    # priced at c0 + 4*u_coef(2) = -2 + 4*0.25 = -1. The double reports the
    # clip residual raw - theta = 2 - 0.5 = 1.5.
    if isinstance(master, FakeMaster):
        assert duals[0] == pytest.approx(1.5, abs=1e-12)
    else:
        assert duals[0] == pytest.approx(-1.0, abs=1e-9)
