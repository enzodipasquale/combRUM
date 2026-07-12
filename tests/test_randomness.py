from __future__ import annotations

import numpy as np
import pytest

import combrum.randomness as randomness
from combrum.randomness import (
    ReplayedWeights,
    bootstrap_multiplier,
    bootstrap_observation_weights,
    multiplier_weights,
    rep_rng,
    rep_seed,
)


def _draws_by_rep(base_seed: int, order: list[int]) -> dict[int, np.ndarray]:
    return {rep: rep_rng(base_seed, rep).standard_normal(16) for rep in order}


def _assert_reproducible_normalized(
    first: np.ndarray, second: np.ndarray, n: int
) -> None:
    assert first.tobytes() == second.tobytes()
    assert first.shape == (n,)
    assert first.dtype == np.float64
    np.testing.assert_allclose(first.sum(), float(n))


def test_rep_seed_is_placement_invariant() -> None:
    # Seeds depend only on (base_seed, rep_id): the same reps drawn in two
    # different orders yield bitwise-identical streams. The entropy pool is
    # the (base_seed, rep_id) tuple itself, not a spawned child sequence.
    first = _draws_by_rep(123, [5, 3, 9])
    second = _draws_by_rep(123, [9, 5, 3])
    for rep in (3, 5, 9):
        assert first[rep].tobytes() == second[rep].tobytes()

    seed = randomness.rep_seed(123, 3)
    assert seed.entropy == (123, 3)
    reference = np.random.SeedSequence((123, 3))
    assert np.array_equal(seed.generate_state(4), reference.generate_state(4))
    # Golden values pin the exact stream.
    stream = randomness.rep_rng(123, 3).standard_normal(4)
    golden = np.array(
        [
            float.fromhex(h)
            for h in (
                "-0x1.2354c95316e9cp+0",
                "0x1.274d860974642p-1",
                "0x1.59a92fba8f846p-2",
                "-0x1.527b7d6c66d96p+0",
            )
        ]
    )
    np.testing.assert_array_equal(stream, golden)


def test_fresh_generators_replay_identical_streams() -> None:
    a = rep_rng(7, 11).random(32)
    b = rep_rng(7, 11).random(32)
    assert a.tobytes() == b.tobytes()
    seed = rep_seed(7, 11)
    assert isinstance(seed, np.random.SeedSequence)


def test_distinct_reps_and_bases_give_distinct_streams() -> None:
    streams = {rep_rng(123, rep).random(16).tobytes() for rep in range(10)}
    assert len(streams) == 10
    assert rep_rng(1, 0).random(16).tobytes() != rep_rng(2, 0).random(16).tobytes()


def test_rep_seed_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="base_seed must be >= 0"):
        rep_seed(-1, 0)
    with pytest.raises(ValueError, match="rep_id must be >= 0"):
        rep_seed(0, -1)


def test_multiplier_weights_are_reproducible_and_normalized() -> None:
    first = multiplier_weights(8, 123, 4)
    second = multiplier_weights(8, 123, 4)

    _assert_reproducible_normalized(first, second, 8)

    # n_units < 1 is rejected up front, not normalized into an empty array.
    with pytest.raises(ValueError, match="n_units must be >= 1"):
        multiplier_weights(0, 123, 4)
    with pytest.raises(ValueError, match="n_units must be >= 1"):
        multiplier_weights(-1, 123, 4)

    # The draw is rep_rng(...).standard_exponential, normalized to sum n.
    raw = rep_rng(123, 4).standard_exponential(8)
    expected = raw * (8.0 / raw.sum())
    np.testing.assert_array_equal(first, expected)

    # Literal values guard against a change in the seeding path itself.
    golden = np.array(
        [
            float.fromhex(h)
            for h in (
                "0x1.113a6d294985cp-2",
                "0x1.0c1463dbf28cap+0",
                "0x1.369d8465cedc1p-1",
                "0x1.7439d9b2b51dfp-1",
                "0x1.04618161c424ap-3",
                "0x1.0e521fd568310p+1",
                "0x1.e066dfc46cbfdp+0",
                "0x1.3c9a0232036e8p+0",
            )
        ]
    )
    np.testing.assert_array_equal(first, golden)

    # A second size checks that the normalization target tracks n.
    raw5 = rep_rng(123, 4).standard_exponential(5)
    expected5 = raw5 * (5.0 / raw5.sum())
    got5 = multiplier_weights(5, 123, 4)
    assert got5.shape == (5,)
    np.testing.assert_array_equal(got5, expected5)


def test_bootstrap_multiplier_is_counter_based() -> None:
    first = bootstrap_multiplier(123, 4, 7)
    second = bootstrap_multiplier(123, 4, 7)

    assert first == second
    assert first > 0.0
    assert first != bootstrap_multiplier(123, 4, 8)
    assert first != bootstrap_multiplier(123, 5, 7)
    with pytest.raises(ValueError, match="base_seed"):
        bootstrap_multiplier(-1, 0, 0)
    with pytest.raises(ValueError, match="rep_id"):
        bootstrap_multiplier(0, -1, 0)
    with pytest.raises(ValueError, match="obs_id"):
        bootstrap_multiplier(0, 0, -1)


def test_bootstrap_multiplier_golden_literals() -> None:
    cases = {
        (0, 0, 0): "0x1.484c44f02554ap-2",
        (123, 4, 7): "0x1.2a22b0a88d7abp+1",
        (12345, 7, 251): "0x1.488fdbb8cee13p-6",
        (99, 3, 5): "0x1.ca6276cfbe97cp+0",
    }

    assert {
        key: bootstrap_multiplier(*key).hex() for key in cases
    } == cases


def test_bootstrap_observation_weights_are_reproducible_and_normalized() -> None:
    first = bootstrap_observation_weights(8, 123, 4)
    second = bootstrap_observation_weights(8, 123, 4)

    _assert_reproducible_normalized(first, second, 8)
    raw = np.array([bootstrap_multiplier(123, 4, i) for i in range(8)])
    np.testing.assert_allclose(first, raw * (8.0 / raw.sum()))

    # Same check as multiplier_weights: the normalization target tracks n.
    raw6 = np.array([bootstrap_multiplier(123, 4, i) for i in range(6)])
    expected6 = raw6 * (6.0 / raw6.sum())
    got6 = bootstrap_observation_weights(6, 123, 4)
    assert got6.shape == (6,)
    np.testing.assert_array_equal(got6, expected6)

    with pytest.raises(ValueError, match="n_observations"):
        bootstrap_observation_weights(0, 123, 4)


def make_replayed_weights(**overrides: object) -> ReplayedWeights:
    kwargs: dict[str, object] = dict(
        matrix=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    kwargs.update(overrides)
    return ReplayedWeights(**kwargs)  # type: ignore[arg-type]


def test_replayed_weights_validation() -> None:
    # Both sides of the `ndim != 2` guard: 1-D and 3-D.
    with pytest.raises(ValueError, match=r"must be 2-D \(B, N\)"):
        make_replayed_weights(matrix=np.ones(3))
    with pytest.raises(ValueError, match=r"must be 2-D \(B, N\)"):
        make_replayed_weights(matrix=np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="B >= 1, N >= 1"):
        make_replayed_weights(matrix=np.empty((0, 3)))
    with pytest.raises(ValueError, match="B >= 1, N >= 1"):
        make_replayed_weights(matrix=np.empty((3, 0)))
    with pytest.raises(ValueError, match="NaN or inf"):
        make_replayed_weights(matrix=np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="NaN or inf"):
        make_replayed_weights(matrix=np.array([[1.0, np.inf], [0.0, 1.0]]))


def test_replayed_weights_owned_and_read_only() -> None:
    source = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = make_replayed_weights(matrix=source)
    source[0, 0] = 99.0  # mutating the caller's array must not affect the copy
    assert weights.matrix[0, 0] == 1.0
    assert not weights.matrix.flags.writeable
    with pytest.raises(ValueError):
        weights.matrix[0, 0] = 99.0


def test_weights_for_returns_read_only_row() -> None:
    weights = make_replayed_weights()
    row = weights.weights_for(1)
    assert row.shape == (3,)
    assert np.array_equal(row, np.array([4.0, 5.0, 6.0]))
    assert not row.flags.writeable
    with pytest.raises(ValueError):
        row[0] = 99.0


def test_weights_for_out_of_range_names_valid_range() -> None:
    weights = make_replayed_weights()
    for bad in (2, -1):
        with pytest.raises(IndexError, match=r"\[0, 2\)"):
            weights.weights_for(bad)
