"""Replication draw derivation and replayed-weight injection.

:func:`rep_seed` is placement-invariant: the stream of replication ``b``
is a pure function of ``(base_seed, b)``. ``SeedSequence.spawn`` is
rejected because it is stateful (each call advances the parent's spawn
key), which would make the same replication draw differently depending
on slot or order.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np

MASK64 = 0xFFFFFFFFFFFFFFFF
BOOTSTRAP_NAMESPACE = 0xC0B2202606250001


def rep_seed(base_seed: int, rep_id: int) -> np.random.SeedSequence:
    """Placement-invariant seed of replication ``rep_id``.

    Hashing ``(base_seed, rep_id)`` into the entropy pool gives
    bitwise-equal streams for equal inputs anywhere.
    """
    base = operator.index(base_seed)
    rep = operator.index(rep_id)
    if base < 0:
        raise ValueError(f"base_seed must be >= 0; got {base_seed!r}")
    if rep < 0:
        raise ValueError(f"rep_id must be >= 0; got {rep_id!r}")
    return np.random.SeedSequence((base, rep))


def rep_rng(base_seed: int, rep_id: int) -> np.random.Generator:
    """Fresh PCG64 generator over :func:`rep_seed` of the replication."""
    return np.random.Generator(np.random.PCG64(rep_seed(base_seed, rep_id)))


def _normalize_to_sum(raw: np.ndarray, n: int) -> np.ndarray:
    """Scale ``raw`` in place to sum to ``n`` (uniform if it sums to zero)."""
    total = float(raw.sum())
    if total <= 0.0:  # pragma: no cover - exponential is a.s. positive
        return np.ones(n, dtype=np.float64)
    raw *= n / total
    return raw


def multiplier_weights(n_units: int, base_seed: int, rep_id: int) -> np.ndarray:
    """Exponential multiplier weights for replication ``rep_id``.

    Normalized to sum to ``n_units`` so the bootstrap criterion keeps the
    same scale as the unit-weight point estimate.
    """
    n = operator.index(n_units)
    if n < 1:
        raise ValueError(f"n_units must be >= 1; got {n_units!r}")
    rng = rep_rng(base_seed, rep_id)
    raw = rng.standard_exponential(n)
    return _normalize_to_sum(raw, n)


def _splitmix64_step(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _splitmix64_steps(x: np.ndarray) -> np.ndarray:
    # uint64 wraparound reproduces _splitmix64_step elementwise, bit for bit.
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _bootstrap_word(base_seed: int, rep_id: int, obs_id: int) -> int:
    x = BOOTSTRAP_NAMESPACE
    for name, field in (
        ("base_seed", base_seed),
        ("rep_id", rep_id),
        ("obs_id", obs_id),
    ):
        value = operator.index(field)
        if value < 0:
            raise ValueError(
                f"bootstrap RNG keys must be nonnegative; {name}={field!r}"
            )
        x ^= value & MASK64
        x = _splitmix64_step(x)
    return x


def bootstrap_multiplier(base_seed: int, rep_id: int, obs_id: int) -> float:
    """Counter-based exponential multiplier for one observed unit.

    The draw is a pure function of ``(base_seed, rep_id, obs_id)``. Distributed
    callers can compute any observation's raw multiplier without carrying a
    generator stream or materializing the whole observation axis.
    """
    word = _bootstrap_word(base_seed, rep_id, obs_id)
    # Half-open [0, 1): never feed log1p the rounded endpoint u == 1.0.
    u = ((word >> 11) & ((1 << 53) - 1)) * 2.0**-53
    return -math.log1p(-u)


def bootstrap_multipliers(
    base_seed: int, rep_id: int, obs_ids: np.ndarray
) -> np.ndarray:
    """Vector of :func:`bootstrap_multiplier` draws over ``obs_ids``.

    Bitwise-equal to the scalar draw at every observation, so batched and
    per-observation callers stay interchangeable.
    """
    obs = np.asarray(obs_ids)
    if obs.ndim != 1 or not np.issubdtype(obs.dtype, np.integer):
        raise ValueError(
            "obs_ids must be a 1-D integer array;"
            f" got shape {obs.shape}, dtype {obs.dtype}"
        )
    if obs.size and int(obs.min()) < 0:
        raise ValueError(
            f"bootstrap RNG keys must be nonnegative; obs_id={int(obs.min())!r}"
        )
    # The (base_seed, rep_id) prefix of the mix is observation-invariant.
    x = BOOTSTRAP_NAMESPACE
    for name, field in (("base_seed", base_seed), ("rep_id", rep_id)):
        value = operator.index(field)
        if value < 0:
            raise ValueError(
                f"bootstrap RNG keys must be nonnegative; {name}={field!r}"
            )
        x ^= value & MASK64
        x = _splitmix64_step(x)
    words = _splitmix64_steps(np.uint64(x) ^ obs.astype(np.uint64))
    u = (words >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return -np.log1p(-u)


def bootstrap_observation_weights(
    n_observations: int, base_seed: int, rep_id: int
) -> np.ndarray:
    """Observation-axis multiplier weights, normalized to sum to ``N``."""
    n = operator.index(n_observations)
    if n < 1:
        raise ValueError(f"n_observations must be >= 1; got {n_observations!r}")
    raw = bootstrap_multipliers(base_seed, rep_id, np.arange(n, dtype=np.int64))
    return _normalize_to_sum(raw, n)


@dataclass(frozen=True)
class ReplayedWeights:
    """Captured ``(B, N)`` weight matrix, replayed as an input source.

    Row ``b`` is the weight vector of replication ``b``. The matrix is
    copied and frozen read-only at construction so a caller's writable
    alias cannot mutate the source.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D (B, N); got shape {matrix.shape}")
        if matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(
                "matrix must hold at least one replication of at least"
                f" one weight (B >= 1, N >= 1); got shape {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("matrix must be finite; got NaN or inf entries")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    def weights_for(self, rep_id: int) -> np.ndarray:
        """Read-only ``(N,)`` weight row of replication ``rep_id``."""
        n_reps = self.matrix.shape[0]
        rep = operator.index(rep_id)
        if not 0 <= rep < n_reps:
            raise IndexError(
                f"rep_id {rep_id} out of range; this ReplayedWeights"
                f" holds replications [0, {n_reps})"
            )
        # View of the frozen matrix is read-only; no copy needed.
        return self.matrix[rep]
