"""Deterministic problem-family inputs for the parity gates.

One canonical array schema per family, generated bitwise-deterministically
from a seed. Reference implementations adapt to these arrays; the arrays
never vary per reference.

Canonical TOY (modular binary choice) schema, all agent-major:

- ``observables`` ``(n_obs, n_items) float64`` in ``{-1.0, +1.0}`` —
  the per-agent sign vector ``r``;
- ``shocks`` ``(n_obs, 1, n_items) float64`` — the per-item shock
  ``nu`` (one simulation draw shared by generation and estimation);
- ``observed`` ``(n_obs, n_items) bool`` — demand at ``theta_true``:
  item ``k`` is chosen iff ``r[i, k] * theta_true[k] + nu[i, k] > 0``,
  so the data is rationalised by ``theta_true`` (min regret 0);
- ``theta_true`` ``(n_items,) float64`` with ``|theta| >= 0.5`` —
  kept away from 0 so the demand argmax is strict.

Canonical QKP (quadratic knapsack) schema, in the
``theta = [alpha, delta_1..delta_M, lambda]`` layout (K = M + 2):

- ``x`` ``(n_obs, n_items) float64`` — per-agent per-item modular
  values;
- ``Q`` ``(n_items, n_items) float64`` — symmetric, zero-diagonal,
  sparse off-diagonal pairwise interaction matrix;
- ``weights`` ``(n_items,) float64 > 0`` — knapsack item weights;
- ``capacities`` ``(n_obs,) float64`` — per-agent knapsack capacity;
- ``shocks`` ``(n_obs, 1, n_items) float64`` — the per-item shock
  ``nu`` (one simulation draw shared by generation and estimation);
- ``observed`` ``(n_obs, n_items) bool`` — the exact argmax of
  ``alpha * x[i]·b - delta·b + 0.5 * lambda * b'Qb + nu[i]·b`` over
  ``b in {0,1}^M`` with ``weights·b <= capacities[i]``, so the data is
  rationalised by ``theta_true``;
- ``theta_true`` ``(n_items + 2,) float64`` — ``[alpha, delta, lambda]``
  with ``alpha > 0`` and ``lambda > 0``.

Both generators enforce a decision margin (:data:`MIN_MARGIN`) so no
reference can flip an observed bundle through float-order arithmetic.
Defaults: toy ``n_obs=12, n_items=5``; QKP ``n_obs=10, n_items=6``;
shared seed :data:`DEFAULT_SEED`.
"""

from __future__ import annotations

import hashlib
import json
import operator
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

#: Family tags hashed into the seed so the two families draw disjoint
#: streams from the same user seed.
_TOY_TAG = 1
_QKP_TAG = 2

#: Smallest tolerated decision margin; far above cross-implementation
#: float noise (~1e-15 relative) so observed bundles cannot flip.
MIN_MARGIN = 1e-6

TOY_DEFAULT_N_OBS = 12
TOY_DEFAULT_N_ITEMS = 5
QKP_DEFAULT_N_OBS = 10
QKP_DEFAULT_N_ITEMS = 6
DEFAULT_SEED = 20260612

# Cap on brute-force enumeration over {0,1}^M.
_MAX_ENUMERATED_ITEMS = 16


def _validated_count(name: str, value: object, minimum: int) -> int:
    count = operator.index(value)
    if count < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value!r}")
    return count


def _family_rng(seed: int, tag: int) -> np.random.Generator:
    base = operator.index(seed)
    if base < 0:
        raise ValueError(f"seed must be >= 0; got {seed!r}")
    # SeedSequence((seed, tag)) gives bitwise-equal streams across machines
    # for equal inputs.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((base, tag))))


def _frozen(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Read-only so a caller cannot mutate a shared fixture array.
    for arr in arrays.values():
        arr.setflags(write=False)
    return arrays


def toy_family(n_obs: int, n_items: int, seed: int) -> dict[str, np.ndarray]:
    """Build the canonical toy (modular binary choice) family arrays.

    Each item ``k`` is chosen iff ``r[i,k] * theta_k + nu[i,k] > 0``;
    ``observed`` is demand at ``theta_true``, so the data is rationalised
    by ``theta_true``. See the module docstring for shapes and dtypes.
    """
    n_obs = _validated_count("n_obs", n_obs, 1)
    n_items = _validated_count("n_items", n_items, 1)
    rng = _family_rng(seed, _TOY_TAG)
    r = rng.choice([-1.0, 1.0], size=(n_obs, n_items))
    nu = rng.standard_normal((n_obs, n_items))
    # sign and magnitude drawn separately so |theta| >= 0.5, keeping
    # every coordinate away from 0 where the argmax would go weak.
    signs = rng.choice([-1.0, 1.0], size=n_items)
    theta_true = signs * rng.uniform(0.5, 2.0, size=n_items)
    scores = r * theta_true[None, :] + nu
    margin = float(np.min(np.abs(scores)))
    if margin <= MIN_MARGIN:
        raise ValueError(
            f"toy_family(seed={seed}): decision margin {margin!r} is"
            f" within float-flip range (<= {MIN_MARGIN}); choose"
            " another seed — a knife-edge choice is not a parity anchor"
        )
    observed = scores > 0.0
    return _frozen(
        {
            "observables": r,
            "shocks": nu.reshape(n_obs, 1, n_items),
            "observed": observed,
            "theta_true": theta_true,
        }
    )


def _enumerate_bundles(n_items: int) -> np.ndarray:
    """All ``2**n_items`` bundles as a ``(2**M, M) float64`` 0/1 matrix."""
    count = np.arange(2**n_items)
    return ((count[:, None] >> np.arange(n_items)[None, :]) & 1).astype(
        np.float64
    )


def qkp_family(n_obs: int, n_items: int, seed: int) -> dict[str, np.ndarray]:
    """Build the canonical QKP family arrays.

    ``observed`` is the exact brute-force argmax of
    ``alpha * x[i]·b - delta·b + 0.5 * lambda * b'Qb + nu[i]·b`` subject
    to ``weights·b <= capacities[i]`` at ``theta_true``, so the data is
    rationalised by ``theta_true``; the runner-up is verified to trail
    by more than :data:`MIN_MARGIN`. See the module docstring for shapes
    and dtypes.
    """
    n_obs = _validated_count("n_obs", n_obs, 1)
    n_items = _validated_count("n_items", n_items, 2)
    if n_items > _MAX_ENUMERATED_ITEMS:
        raise ValueError(
            "qkp_family generates observed bundles by exact enumeration"
            f" and supports n_items <= {_MAX_ENUMERATED_ITEMS};"
            f" got {n_items}"
        )
    rng = _family_rng(seed, _QKP_TAG)
    x = rng.uniform(0.0, 2.0, size=(n_obs, n_items))
    weights = rng.uniform(0.5, 1.5, size=n_items)
    # Capacities bind: a strict subset fits, never all or none.
    capacities = rng.uniform(0.35, 0.65, size=n_obs) * float(weights.sum())
    # Sparse symmetric zero-diagonal Q: complementarities on a subset of
    # unordered pairs.
    all_pairs = np.array(
        [(j, k) for j in range(n_items) for k in range(j + 1, n_items)],
        dtype=np.int64,
    )
    n_pairs = min(2 * n_items, len(all_pairs))
    chosen = all_pairs[rng.choice(len(all_pairs), size=n_pairs, replace=False)]
    coupling = rng.uniform(0.2, 1.0, size=n_pairs)
    q = np.zeros((n_items, n_items), dtype=np.float64)
    q[chosen[:, 0], chosen[:, 1]] = coupling
    q[chosen[:, 1], chosen[:, 0]] = coupling
    alpha = float(rng.uniform(0.3, 0.7))
    delta = rng.uniform(-0.5, 0.5, size=n_items)
    lam = float(rng.uniform(0.2, 0.5))
    theta_true = np.concatenate([[alpha], delta, [lam]])
    nu = rng.normal(0.0, 0.5, size=(n_obs, n_items))

    bundles = _enumerate_bundles(n_items)
    loads = bundles @ weights
    quad = 0.5 * lam * np.einsum("bj,jk,bk->b", bundles, q, bundles)
    observed = np.empty((n_obs, n_items), dtype=bool)
    for i in range(n_obs):
        utility = bundles @ (alpha * x[i] - delta + nu[i]) + quad
        utility = np.where(loads <= capacities[i], utility, -np.inf)
        best = int(np.argmax(utility))
        runner_up = float(np.partition(utility, -2)[-2])
        # Empty bundle is always feasible, so an argmax exists; a -inf
        # runner-up (only one feasible bundle) also fails the margin check.
        if utility[best] - runner_up <= MIN_MARGIN:
            raise ValueError(
                f"qkp_family(seed={seed}): agent {i} optimum leads its"
                f" runner-up by {utility[best] - runner_up!r}"
                f" (<= {MIN_MARGIN}); choose another seed — a near-tie"
                " is not a parity anchor"
            )
        observed[i] = bundles[best] > 0.5
    return _frozen(
        {
            "x": x,
            "Q": q,
            "weights": weights,
            "capacities": capacities,
            "shocks": nu.reshape(n_obs, 1, n_items),
            "observed": observed,
            "theta_true": theta_true,
        }
    )


def persist_family(
    name: str, arrays: Mapping[str, np.ndarray], directory: Path
) -> Path:
    """Write ``arrays`` as ``<name>.npz``; :func:`load_family` reloads it
    bitwise-exact (npz preserves dtype, shape, and raw bytes).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    np.savez(path, **dict(arrays))
    return path


def load_family(name: str, directory: Path) -> dict[str, np.ndarray]:
    """Reload a family written by :func:`persist_family`, read-only."""
    with np.load(directory / f"{name}.npz") as npz:
        arrays = {key: npz[key] for key in sorted(npz.files)}
    return _frozen(arrays)


#: On-disk home of the persisted default family fixtures.
FAMILY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "families"


def load_toy() -> dict[str, np.ndarray]:
    return load_family("toy", FAMILY_DIR)


def load_qkp() -> dict[str, np.ndarray]:
    return load_family("qkp", FAMILY_DIR)


def family_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-256 hex over a canonical byte rendering of the family arrays.

    Each array contributes (name, dtype, shape, raw C-order bytes) in
    sorted name order, every field length-prefixed so concatenations
    cannot alias. Any value, dtype or shape change changes the digest.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = arrays[name]
        for piece in (
            name.encode(),
            str(arr.dtype).encode(),
            json.dumps(list(arr.shape)).encode(),
            np.ascontiguousarray(arr).tobytes(),
        ):
            h.update(len(piece).to_bytes(8, "big"))
            h.update(piece)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Golden-drift guard. The persisted ``tests/fixtures/families/{toy,qkp}.npz``
# files feed a dozen downstream parity tests via :func:`load_family` and
# nothing else re-derives them, so a generator change would silently leave
# those tests on stale bytes. Pin the on-disk fixtures to what the generators
# produce at their documented defaults.
# --------------------------------------------------------------------------


def _assert_arrays_bitwise_equal(
    disk: Mapping[str, np.ndarray], live: Mapping[str, np.ndarray]
) -> None:
    assert sorted(disk) == sorted(live)
    for key in sorted(live):
        want = live[key]
        got = disk[key]
        assert got.dtype == want.dtype, key
        assert got.shape == want.shape, key
        assert np.array_equal(got, want), key


def test_toy_fixture_matches_generator_at_defaults() -> None:
    live = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    disk = load_family("toy", FAMILY_DIR)
    _assert_arrays_bitwise_equal(disk, live)


def test_qkp_fixture_matches_generator_at_defaults() -> None:
    live = qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)
    disk = load_family("qkp", FAMILY_DIR)
    _assert_arrays_bitwise_equal(disk, live)


def test_persist_load_round_trips_bitwise_exact(tmp_path: Path) -> None:
    live = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    path = persist_family("toy", live, tmp_path)
    assert path == tmp_path / "toy.npz"
    reloaded = load_family("toy", tmp_path)
    _assert_arrays_bitwise_equal(reloaded, live)


def test_persist_load_honor_the_name_argument(tmp_path: Path) -> None:
    # Two distinct families under distinct names: an implementation that
    # ignores its name argument (say a hardcoded "toy.npz") would collide the
    # files on disk and cross-contaminate the reloads.
    toy = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    qkp = qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)

    toy_path = persist_family("toy", toy, tmp_path)
    qkp_path = persist_family("qkp_probe", qkp, tmp_path)
    assert toy_path == tmp_path / "toy.npz"
    assert qkp_path == tmp_path / "qkp_probe.npz"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "qkp_probe.npz",
        "toy.npz",
    ]

    # Each name reloads its own payload on the read side too.
    _assert_arrays_bitwise_equal(load_family("toy", tmp_path), toy)
    _assert_arrays_bitwise_equal(load_family("qkp_probe", tmp_path), qkp)


def _assert_all_read_only(arrays: Mapping[str, np.ndarray]) -> None:
    # Every returned array must be read-only so a caller cannot mutate a
    # shared fixture (the _frozen/load_family contract).
    for key, arr in arrays.items():
        assert arr.flags.writeable is False, key


def test_family_arrays_are_read_only() -> None:
    _assert_all_read_only(
        toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    )
    _assert_all_read_only(
        qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)
    )
    _assert_all_read_only(load_family("toy", FAMILY_DIR))


def test_loaded_family_is_read_only(tmp_path: Path) -> None:
    live = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    persist_family("toy", live, tmp_path)
    _assert_all_read_only(load_family("toy", tmp_path))


def _toy_min_margin(fam: Mapping[str, np.ndarray]) -> float:
    # min |r*theta + nu| in plain Python, structurally distinct from the
    # vectorised generator.
    r = fam["observables"]
    theta = fam["theta_true"]
    nu = fam["shocks"].reshape(r.shape)
    n_obs, n_items = r.shape
    return min(
        abs(float(r[i, k] * theta[k] + nu[i, k]))
        for i in range(n_obs)
        for k in range(n_items)
    )


def _qkp_min_gap(fam: Mapping[str, np.ndarray]) -> float:
    # Brute-force best-minus-runner-up across agents in plain Python; a
    # separate implementation of the objective the generator vectorises.
    x = fam["x"]
    q = fam["Q"]
    weights = fam["weights"]
    capacities = fam["capacities"]
    theta = fam["theta_true"]
    n_obs, n_items = x.shape
    nu = fam["shocks"].reshape(n_obs, n_items)
    alpha = float(theta[0])
    delta = theta[1 : 1 + n_items]
    lam = float(theta[-1])
    worst = float("inf")
    for i in range(n_obs):
        utils = []
        for mask in range(2**n_items):
            bundle = [(mask >> k) & 1 for k in range(n_items)]
            if sum(bundle[k] * weights[k] for k in range(n_items)) > capacities[i]:
                utils.append(float("-inf"))
                continue
            modular = sum(
                bundle[k] * (alpha * x[i, k] - delta[k] + nu[i, k])
                for k in range(n_items)
            )
            quad = sum(
                0.5 * lam * bundle[j] * q[j, k] * bundle[k]
                for j in range(n_items)
                for k in range(n_items)
            )
            utils.append(modular + quad)
        ordered = sorted(utils, reverse=True)
        worst = min(worst, ordered[0] - ordered[1])
    return worst


#: Documented cross-implementation float noise floor (~1e-15 relative), taken
#: from the module docstring, not from MIN_MARGIN. The guard must sit well
#: above this so a knife-edge family cannot flip through float-order arithmetic.
_FLOAT_NOISE = 1e-15


def test_min_margin_is_a_small_positive_threshold() -> None:
    # Both ends pinned: small enough to sit below any real decision gap, yet
    # at least 1e3x above float noise so it cannot decay to a level where a
    # float-flippable family passes as a parity anchor.
    assert MIN_MARGIN >= 1e3 * _FLOAT_NOISE
    assert 0.0 < MIN_MARGIN <= 1e-4


def test_default_families_clear_the_margin() -> None:
    toy = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    assert _toy_min_margin(toy) > MIN_MARGIN
    qkp = qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)
    assert _qkp_min_gap(qkp) > MIN_MARGIN


def _indep_toy_min_margin(fam: Mapping[str, np.ndarray]) -> float:
    # min|r*theta + nu| a third way: broadcast the whole score grid and min
    # the flattened absolute value in one shot.
    r = fam["observables"]
    theta = fam["theta_true"]
    nu = fam["shocks"].reshape(r.shape)
    scores = r * theta[None, :] + nu
    return float(np.min(np.abs(scores.ravel())))


def _indep_qkp_min_gap(fam: Mapping[str, np.ndarray]) -> float:
    # Best-minus-second a third way: itertools.product enumeration with numpy
    # dot products, unlike the generator's einsum or _qkp_min_gap's index
    # loops.
    import itertools

    x = fam["x"]
    q = fam["Q"]
    weights = fam["weights"]
    capacities = fam["capacities"]
    theta = fam["theta_true"]
    n_obs, n_items = x.shape
    nu = fam["shocks"].reshape(n_obs, n_items)
    alpha = float(theta[0])
    delta = theta[1 : 1 + n_items]
    lam = float(theta[-1])
    all_bundles = np.array(
        list(itertools.product((0.0, 1.0), repeat=n_items))
    )
    worst = float("inf")
    for i in range(n_obs):
        utilities = []
        for bundle in all_bundles:
            if float(bundle @ weights) > capacities[i]:
                utilities.append(-np.inf)
                continue
            modular = float(bundle @ (alpha * x[i] - delta + nu[i]))
            quad = float(0.5 * lam * (bundle @ q @ bundle))
            utilities.append(modular + quad)
        utilities.sort(reverse=True)
        worst = min(worst, utilities[0] - utilities[1])
    return worst


def test_margin_recomputations_agree() -> None:
    # The margin tests lean on _toy_min_margin / _qkp_min_gap; a stub that
    # returns a large constant would clear every `> MIN_MARGIN` check while
    # waving through a genuinely float-flippable family. Each helper must
    # therefore match a separately-written recomputation and the known values
    # of the frozen defaults.
    toy = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    qkp = qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)

    assert _toy_min_margin(toy) == pytest.approx(_indep_toy_min_margin(toy))
    assert _qkp_min_gap(qkp) == pytest.approx(_indep_qkp_min_gap(qkp))

    assert _toy_min_margin(toy) == pytest.approx(0.2206431761206875)
    assert _qkp_min_gap(qkp) == pytest.approx(0.0316466606532404)


def test_margin_helpers_flag_knife_edge_families() -> None:
    # The other direction: on a deliberately knife-edge family the helpers
    # must report a value BELOW MIN_MARGIN, which a large-constant stub
    # cannot.
    r, nu, signs, magnitude = _scripted_toy_knife_edge()
    knife_toy = {
        "observables": r,
        "shocks": nu.reshape(r.shape[0], 1, r.shape[1]),
        "theta_true": signs * magnitude,
    }
    assert _toy_min_margin(knife_toy) <= MIN_MARGIN

    # Rebuild the scripted QKP agent's arrays for the QKP helper.
    (
        x,
        weights,
        capacity_fractions,
        _chosen,
        _coupling,
        alpha,
        delta,
        lam,
        nu_q,
    ) = _scripted_qkp_knife_edge()
    n_obs, n_items = x.shape
    knife_qkp = {
        "x": x,
        "Q": np.zeros((n_items, n_items)),
        "weights": weights,
        "capacities": capacity_fractions * float(weights.sum()),
        "shocks": nu_q.reshape(n_obs, 1, n_items),
        "theta_true": np.concatenate([[alpha], delta, [lam]]),
    }
    assert _qkp_min_gap(knife_qkp) <= MIN_MARGIN


def test_toy_family_rejects_knife_edge_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bump MIN_MARGIN one ulp above the default seed's real min|scores|.
    # Under a looser bump (say edge + 1.0) a guard measuring the wrong
    # quantity — min|nu| ~= 0.015, for one — would raise as well, leaving
    # |r*theta + nu| unpinned.
    edge = _toy_min_margin(
        toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    )
    monkeypatch.setattr(
        sys.modules[__name__], "MIN_MARGIN", np.nextafter(edge, np.inf)
    )
    with pytest.raises(ValueError, match="parity anchor"):
        toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)


class _ScriptedRng:
    """Hands back a fixed sequence of arrays, one per draw call, so a test can
    drive ``toy_family``/``qkp_family`` through a hand-built family with exact
    decision margins.
    """

    def __init__(self, draws: list[object]) -> None:
        self._draws = list(draws)
        self._i = 0

    def _next(self) -> object:
        value = self._draws[self._i]
        self._i += 1
        return value

    def choice(self, *args: object, **kwargs: object) -> object:
        return self._next()

    def standard_normal(self, *args: object, **kwargs: object) -> object:
        return self._next()

    def uniform(self, *args: object, **kwargs: object) -> object:
        return self._next()

    def normal(self, *args: object, **kwargs: object) -> object:
        return self._next()


def _scripted_toy_knife_edge() -> list[object]:
    # r, nu, signs, magnitude in toy_family's draw order. |scores| at [0,0] is
    # ~1e-9 (knife-edge, far below MIN_MARGIN) while every |nu| is >= ~0.5, so
    # the correct min|scores| guard fires but a min|nu| guard would not.
    r = np.ones((2, 2))
    nu = np.full((2, 2), 3.0)
    nu[0, 0] = -0.5 + 1e-9  # r*theta = +0.5 here, so score ~= 1e-9
    signs = np.ones(2)
    magnitude = np.full(2, 0.5)  # theta_true = +0.5 per item
    return [r, nu, signs, magnitude]


def test_toy_guard_measures_decision_score_not_shock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This hand-built family is knife-edge in |scores| (~1e-9) but has large
    # |nu| (~0.5), so it is rejected only by a guard reading the decision
    # score; a guard on min|nu| would accept a family whose observed choice
    # can float-flip.
    monkeypatch.setattr(
        sys.modules[__name__],
        "_family_rng",
        lambda seed, tag: _ScriptedRng(_scripted_toy_knife_edge()),
    )
    with pytest.raises(ValueError, match="parity anchor"):
        toy_family(2, 2, 0)

    # Both directions, from the same scripted family: min|scores| trips the
    # guard, min|nu| would not.
    r, nu, signs, magnitude = _scripted_toy_knife_edge()
    scores = r * (signs * magnitude)[None, :] + nu
    assert float(np.min(np.abs(scores))) <= MIN_MARGIN
    assert float(np.min(np.abs(nu))) > MIN_MARGIN


def test_qkp_family_rejects_near_tie_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As with the toy guard: bump MIN_MARGIN to just above the true
    # best-minus-second gap. The 1e-6 relative slack clears the einsum-vs-
    # Python float noise between the generator's gap and the helper's while
    # staying ~3.9x below best-minus-third, so only a guard measuring the
    # SECOND-best fires.
    edge = _qkp_min_gap(
        qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)
    )
    monkeypatch.setattr(
        sys.modules[__name__], "MIN_MARGIN", edge * (1.0 + 1e-6)
    )
    with pytest.raises(ValueError, match="parity anchor"):
        qkp_family(QKP_DEFAULT_N_OBS, QKP_DEFAULT_N_ITEMS, DEFAULT_SEED)


def _scripted_qkp_knife_edge() -> list[object]:
    # qkp_family draw order: x, weights, capacity fractions, chosen pair
    # indices, coupling, alpha, delta, lambda, nu. One agent, two items, zero
    # coupling, roomy capacity so all four bundles are feasible. Per-item
    # scores c0 = 1.0 and c1 = 1e-9 give bundle utilities
    # {0.0, 1.0, 1e-9, 1.0 + 1e-9}: best-minus-second ~= 1e-9 (knife-edge)
    # while best-minus-third ~= 1.0.
    x = np.array([[2.0, 2.0]])
    weights = np.array([1.0, 1.0])
    capacity_fractions = np.array([100.0])  # * weights.sum() -> all feasible
    chosen_indices = np.array([0])  # the single (0, 1) pair
    coupling = np.array([0.0])  # zero Q -> quad term vanishes
    alpha = 0.5
    delta = np.array([0.0, 0.0])
    lam = 0.4
    nu = np.array([[0.0, 1e-9 - 1.0]])  # c0 = 1.0, c1 = 1e-9
    return [
        x,
        weights,
        capacity_fractions,
        chosen_indices,
        coupling,
        alpha,
        delta,
        lam,
        nu,
    ]


def test_qkp_guard_measures_second_best_not_third(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scripted agent's best-minus-second is ~1e-9 but best-minus-third is
    # ~1.0, so it is rejected only by a guard that measures the runner-up; a
    # guard on best-minus-third would accept a genuine near-tie between the
    # optimum and the runner-up.
    monkeypatch.setattr(
        sys.modules[__name__],
        "_family_rng",
        lambda seed, tag: _ScriptedRng(_scripted_qkp_knife_edge()),
    )
    with pytest.raises(ValueError, match="parity anchor"):
        qkp_family(1, 2, 0)

    # Both directions, from the bundle utilities directly.
    utilities = sorted([0.0, 1.0, 1e-9, 1.0 + 1e-9], reverse=True)
    assert utilities[0] - utilities[1] <= MIN_MARGIN
    assert utilities[0] - utilities[2] > MIN_MARGIN


def test_family_digest_detects_any_perturbation() -> None:
    live = toy_family(TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS, DEFAULT_SEED)
    baseline = family_digest(live)

    # A one-ulp value change flips the digest.
    perturbed = {key: arr.copy() for key, arr in live.items()}
    perturbed["theta_true"][0] = np.nextafter(
        perturbed["theta_true"][0], np.inf
    )
    assert family_digest(perturbed) != baseline

    # A dtype change (same values) flips the digest.
    retyped = {key: arr.copy() for key, arr in live.items()}
    retyped["observed"] = retyped["observed"].astype(np.int8)
    assert family_digest(retyped) != baseline

    # A shape change flips the digest.
    reshaped = {key: arr.copy() for key, arr in live.items()}
    reshaped["shocks"] = reshaped["shocks"].reshape(
        TOY_DEFAULT_N_OBS, TOY_DEFAULT_N_ITEMS
    )
    assert family_digest(reshaped) != baseline

    # A key rename flips the digest even when the bytes are identical,
    # because the name is folded into the hash.
    renamed = {key: arr.copy() for key, arr in live.items()}
    renamed["observables_renamed"] = renamed.pop("observables")
    assert family_digest(renamed) != baseline

    # The length prefix: two DISTINCT families whose unprefixed fields
    # concatenate to the same byte stream must still hash apart. The "u"
    # migrates between array name and dtype string — {'xu': int8} ->
    # b'xu'+b'int8' and {'x': uint8} -> b'x'+b'uint8' both run together as
    # b'xuint8...' — so without per-field prefixes these collide.
    aliased_a = {"xu": np.array([7], dtype=np.int8)}
    aliased_b = {"x": np.array([7], dtype=np.uint8)}
    assert family_digest(aliased_a) != family_digest(aliased_b)
