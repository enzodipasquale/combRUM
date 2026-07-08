"""Deterministic synthetic fixtures for certification, stripping, and IO gates.

- :func:`inexact_certification_fixture`: per-agent optimality gaps with a known
  inexact subset.
- :func:`stripping_snapshot_fixture`: a frozen cut-set snapshot plus the keep-set
  its stripping rule selects.
- :func:`large_m_generator`: directed-pair bundles whose column count
  ``M = T*(T-1)`` grows quadratically, for disk-bytes/IO measurements.

All fixtures derive from ``SeedSequence((seed, tag))`` for bitwise-reproducible
output, and return frozen, read-only arrays.
"""

from __future__ import annotations

import operator

import numpy as np

from combrum.transport.base import _pack_bundle

#: Per-fixture seed tags; must stay disjoint so no two fixture kinds share a
#: stream at the same user seed.
_CERT_TAG = 3
_STRIP_TAG = 4
_LARGE_M_TAG = 5

#: Stripping rule mirrored by :func:`stripping_snapshot_fixture`: keep rows up to
#: the looseness percentile, then cap at this many most-binding rows.
STRIP_PERCENTILE = 95.0
STRIP_HARD_THRESHOLD = 150

_STRIP_N_ITEMS = 12

#: Lower bound of the certification injection band, kept well above float
#: noise so an injected gap can never read as an exact (zero) gap.
_CERT_GAP_FLOOR = 1e-4

#: Band ends restated as literals the import-time self-check owns. Neither is
#: derived from ``_CERT_GAP_FLOOR`` or the ``uniform`` draw: an edit to the
#: band constants must be measured against bounds that stay put.
_MIN_MEANINGFUL_GAP = 1e-4
_MAX_MEANINGFUL_GAP = 1e-2

#: Sweep budget: 40 seeds x n_inexact=100 at n_agents=300 = 4000 injected
#: draws, enough that the empirical minimum hugs the band's low end to within
#: ~0.3% and the empirical maximum hugs the top to within ~0.04%.
_CERT_SWEEP_SEEDS = 40
_CERT_SWEEP_N_AGENTS = 300

#: Two-sided windows for the sweep's empirical extremes, as fractions of the
#: band ends. The honest fixture lands the sampled minimum at ~1.003x the low
#: end and the sampled maximum at ~0.9996x the top, so these clear it while a
#: band end moved by >=1% in either direction — a lowered or raised floor, a
#: shrunken or widened top — pushes an extreme out of its window.
_CERT_OBS_CEILING = 1.008
_CERT_HIGH_OBS_FLOOR = 0.995
_CERT_HIGH_OBS_CEILING = 1.005


def _validated_count(name: str, value: object, minimum: int) -> int:
    count = operator.index(value)
    if count < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value!r}")
    return count


def _fixture_rng(seed: int, tag: int) -> np.random.Generator:
    base = operator.index(seed)
    if base < 0:
        raise ValueError(f"seed must be >= 0; got {seed!r}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((base, tag))))


def _frozen(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for arr in arrays.values():
        arr.setflags(write=False)
    return arrays


def inexact_certification_fixture(n_agents: int, seed: int) -> dict[str, object]:
    """Per-agent gaps with a known inexact subset, plus the expected triple.

    Args:
        n_agents: agent count, ``>= 2``.
        seed: nonnegative reproducibility seed.

    Returns:
        ``{"gaps", "inexact_ids", "expected"}``: ``gaps`` ``(n_agents,) float64``
        (``0.0`` for exact agents, strictly positive for the inexact subset),
        ``inexact_ids`` (sorted ``int64``, a nonempty subset of about a
        third of agents), and ``expected`` the
        ``(n_agents, n_inexact, worst_gap)`` triple an exactness report must
        produce.
    """
    n_agents = _validated_count("n_agents", n_agents, 2)
    rng = _fixture_rng(seed, _CERT_TAG)
    n_inexact = max(1, n_agents // 3)
    inexact_ids = np.sort(
        rng.choice(n_agents, size=n_inexact, replace=False)
    ).astype(np.int64)
    gaps = np.zeros(n_agents, dtype=np.float64)
    # Gaps kept well above float noise so "inexact" is unambiguous.
    gaps[inexact_ids] = rng.uniform(_CERT_GAP_FLOOR, 1e-2, size=n_inexact)
    expected = (n_agents, int(n_inexact), float(gaps.max()))
    return {
        **_frozen({"gaps": gaps, "inexact_ids": inexact_ids}),
        "expected": expected,
    }


def _bundle_keys(bundles: np.ndarray) -> np.ndarray:
    """Per-row dedup keys, using combrum's canonical cut-identity codec.

    Each key is exactly what a cut store dedups on: the byte string
    :func:`combrum.transport.base._pack_bundle` produces for the row
    (``dtype.str`` tag + ``b":"`` + raw bytes). Stored in an ``object`` array
    of ``bytes`` rather than a fixed-width ``S`` array: NumPy strips trailing
    NUL bytes from ``S`` elements on readback, which would drop the tail of a
    bundle key ending in ``False`` and no longer be byte-identical to the
    store's key.
    """
    return np.array([_pack_bundle(row) for row in bundles], dtype=object)


def stripping_snapshot_fixture(n_cuts: int, seed: int) -> dict[str, object]:
    """A frozen cut-set snapshot plus the keep-set its rule selects.

    Args:
        n_cuts: number of cuts, ``>= 2``.
        seed: nonnegative reproducibility seed.

    Returns:
        ``{"agent_ids", "bundle_keys", "slacks", "expected_keep", "threshold",
        "hard_cap_active"}``: per-cut owning agent (``int64``), bundle key
        (fixed-width bytes), nonnegative slack (``float64``; 0 = binding,
        larger = looser), the expected keep mask (``bool``), the
        95th-percentile slack
        cutoff ``threshold`` (``float``), and ``hard_cap_active`` (``bool``,
        whether the hard cap overrode the percentile keep).

    The keep rule keeps row ``i`` iff ``slacks[i] <= percentile(slacks,
    95.0)``; if that keeps more than the hard threshold, it instead keeps only
    the ``hard_threshold`` smallest slacks.

    Slacks are bimodal (tight body in ``[0, 1]``, loose outliers in ``[3, 8]``)
    so the snapshot always keeps at least one cut and strips at least one.
    """
    n_cuts = _validated_count("n_cuts", n_cuts, 2)
    rng = _fixture_rng(seed, _STRIP_TAG)
    agent_ids = rng.integers(0, max(2, n_cuts // 2), size=n_cuts).astype(
        np.int64
    )
    bundles = rng.random((n_cuts, _STRIP_N_ITEMS)) < 0.5
    n_loose = max(1, n_cuts // 5)
    body = rng.uniform(0.0, 1.0, size=n_cuts - n_loose)
    loose = rng.uniform(3.0, 8.0, size=n_loose)
    slacks = np.concatenate([body, loose])[rng.permutation(n_cuts)]
    threshold = float(np.percentile(slacks, STRIP_PERCENTILE))
    expected_keep = slacks <= threshold
    hard_cap_active = int(expected_keep.sum()) > STRIP_HARD_THRESHOLD
    if hard_cap_active:
        expected_keep = np.zeros(n_cuts, dtype=bool)
        order = np.argsort(slacks, kind="stable")
        expected_keep[order[:STRIP_HARD_THRESHOLD]] = True
    return {
        **_frozen(
            {
                "agent_ids": agent_ids,
                "bundle_keys": _bundle_keys(bundles),
                "slacks": slacks,
                "expected_keep": expected_keep,
            }
        ),
        "threshold": threshold,
        "hard_cap_active": hard_cap_active,
    }


def large_m_generator(T: int, n_rows: int, seed: int) -> dict[str, object]:
    """Directed-pair bundles with ``M = T*(T-1)`` columns, plus phi features.

    Args:
        T: node count, ``>= 2``.
        n_rows: number of payload rows, ``>= 1``.
        seed: nonnegative reproducibility seed.

    Returns:
        ``{"senders", "receivers", "bundles", "phi", "T", "M"}``: one column per
        ordered pair ``(sender, receiver)`` with ``sender != receiver``,
        enumerated row-major (``senders``/``receivers`` both ``(M,) int64``);
        ``bundles`` ``(n_rows, M) bool``; ``phi`` ``(n_rows, 2*T + 1) float64``;
        and the scalars ``T`` and ``M``.
    """
    T = _validated_count("T", T, 2)
    n_rows = _validated_count("n_rows", n_rows, 1)
    rng = _fixture_rng(seed, _LARGE_M_TAG)
    grid_sender, grid_receiver = np.indices((T, T))
    off_diagonal = grid_sender != grid_receiver
    senders = grid_sender[off_diagonal].astype(np.int64)
    receivers = grid_receiver[off_diagonal].astype(np.int64)
    m_columns = T * (T - 1)
    # Guard the ordered sequence, not just membership: a sender/receiver swap
    # or reordering leaves the *set* of pairs intact.
    expected_pairs = [(i, j) for i in range(T) for j in range(T) if i != j]
    if list(zip(senders.tolist(), receivers.tolist())) != expected_pairs:
        raise AssertionError(
            "senders/receivers must enumerate off-diagonal pairs row-major"
        )
    bundles = rng.random((n_rows, m_columns)) < 0.5
    phi = rng.standard_normal((n_rows, 2 * T + 1))
    return {
        **_frozen(
            {
                "senders": senders,
                "receivers": receivers,
                "bundles": bundles,
                "phi": phi,
            }
        ),
        "T": T,
        "M": m_columns,
    }


def _assert_cert_gap_floor_holds() -> None:
    """Import-time self-check: the injected band spans ``(1e-4, 1e-2]``.

    ``_CERT_GAP_FLOOR`` must sit at or above ``_MIN_MEANINGFUL_GAP`` by value,
    and over the seed sweep the empirical minimum and maximum injected gaps
    must land in the two-sided windows around the band ends. Every bound is a
    literal the fixture never draws from, so any edit to the ``uniform``
    constants — a lowered floor, a low end hardcoded away from the constant,
    a shrunken or widened top — moves an extreme out of its window even
    though the draws still respect their own (edited) band.
    """
    if _CERT_GAP_FLOOR < _MIN_MEANINGFUL_GAP:
        raise AssertionError(
            "certification band floor dropped below the meaningful gap floor:"
            f" _CERT_GAP_FLOOR {_CERT_GAP_FLOOR!r} < {_MIN_MEANINGFUL_GAP!r}"
        )
    empirical_min = float("inf")
    empirical_max = 0.0
    for seed in range(_CERT_SWEEP_SEEDS):
        fix = inexact_certification_fixture(_CERT_SWEEP_N_AGENTS, seed)
        injected = fix["gaps"][fix["inexact_ids"]]
        empirical_min = min(empirical_min, float(injected.min()))
        empirical_max = max(empirical_max, float(injected.max()))
    ceiling = _MIN_MEANINGFUL_GAP * _CERT_OBS_CEILING
    if not (_MIN_MEANINGFUL_GAP <= empirical_min <= ceiling):
        raise AssertionError(
            "certification injection band decoupled from the meaningful floor:"
            f" min injected gap {empirical_min!r} outside"
            f" [{_MIN_MEANINGFUL_GAP!r}, {ceiling!r}]"
        )
    high_floor = _MAX_MEANINGFUL_GAP * _CERT_HIGH_OBS_FLOOR
    high_ceiling = _MAX_MEANINGFUL_GAP * _CERT_HIGH_OBS_CEILING
    if not (high_floor <= empirical_max <= high_ceiling):
        raise AssertionError(
            "certification injection band decoupled from the meaningful high"
            f" bound: max injected gap {empirical_max!r} outside"
            f" [{high_floor!r}, {high_ceiling!r}]"
        )


_assert_cert_gap_floor_holds()
