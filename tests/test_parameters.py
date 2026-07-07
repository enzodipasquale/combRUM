from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from combrum.parameters import Parameters


def make_params() -> Parameters:
    return Parameters({"beta": (-5.0, 5.0, 2), "gamma": (0.0, 1.0, 1)})


def test_layout_basics() -> None:
    params = make_params()
    assert params.K == 3
    assert params.names == ("beta", "gamma")
    assert params.block("beta") == slice(0, 2)
    assert params.block("gamma") == slice(2, 3)


def test_bounds_vectors() -> None:
    lb, ub = make_params().bounds()
    assert lb.dtype == np.float64 and ub.dtype == np.float64
    np.testing.assert_array_equal(lb, [-5.0, -5.0, 0.0])
    np.testing.assert_array_equal(ub, [5.0, 5.0, 1.0])


def test_pack_unpack_identity() -> None:
    params = make_params()
    # Fractional, signed values: an int/truncating pack buffer would round these
    # to [1, -2, 0] and fail the round-trip. Values chosen inside the bounds so
    # the round-trip exercises real magnitudes, not just integer coordinates.
    theta = np.array([1.5, -2.25, 0.75])
    named = params.unpack(theta)
    np.testing.assert_array_equal(named["beta"], [1.5, -2.25])
    np.testing.assert_array_equal(named["gamma"], [0.75])
    packed = params.pack(named)
    # Pin pack's float64 contract directly: a narrowed/int dtype loses precision.
    assert packed.dtype == np.float64
    np.testing.assert_array_equal(packed, theta)


def test_unpack_validates_length() -> None:
    # Cross the length boundary in both directions. Oversized (4 > K=3) and
    # undersized (2 < K=3) 1-D thetas must both raise; a guard that only checks
    # `shape[0] > self.K` accepts the undersized case and returns a silently
    # truncated last block. K=3 comes from the fixture (beta 2 + gamma 1).
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        make_params().unpack(np.zeros(4))
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        make_params().unpack(np.zeros(2))


def test_unpack_rejects_wrong_dimensionality() -> None:
    # Right total size (3 = K), wrong shape: a column/row vector must be
    # rejected so unpack compares the full shape tuple, not just a length.
    # A length-only guard would slice (3,1) into blocks of shape (2,1)/(1,1).
    params = make_params()
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        params.unpack(np.zeros((3, 1)))
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        params.unpack(np.zeros((1, 3)))


def test_pack_validates_names_and_lengths() -> None:
    params = make_params()
    with pytest.raises(ValueError, match="missing block"):
        params.pack({"beta": np.zeros(2)})
    with pytest.raises(ValueError, match="unknown block name"):
        params.pack(
            {"beta": np.zeros(2), "gamma": np.zeros(1), "delta": np.zeros(1)}
        )
    with pytest.raises(ValueError, match="must have length 2"):
        params.pack({"beta": np.zeros(3), "gamma": np.zeros(1)})
    # Non-first block: a guard that only validates the first iteration would
    # mis-fill gamma's slice and never raise. Pin the block NAME so the guard
    # is proven to fire on gamma (k=1 from the fixture), not just on beta.
    with pytest.raises(ValueError, match="block 'gamma' must have length 1"):
        params.pack({"beta": np.zeros(2), "gamma": np.zeros(5)})
    # Undersized/scalar direction: a scalar or length-1 value would silently
    # broadcast to fill the size-2 block if the guard only rejected oversized
    # arrays. Expected length 2 comes from the fixture (beta k=2), not pack.
    with pytest.raises(ValueError, match="must have length 2"):
        params.pack({"beta": 5.0, "gamma": np.zeros(1)})
    with pytest.raises(ValueError, match="must have length 2"):
        params.pack({"beta": np.zeros(1), "gamma": np.zeros(1)})


def test_rejects_duplicate_names() -> None:
    pairs = [("a", (0.0, 1.0, 1)), ("a", (0.0, 1.0, 2))]
    with pytest.raises(ValueError, match="duplicate parameter block name"):
        Parameters(pairs)


def test_rejects_empty_specification() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        Parameters({})


def test_rejects_invalid_blocks() -> None:
    with pytest.raises(ValueError, match="k must be an integer >= 1"):
        Parameters({"a": (0.0, 1.0, 0)})
    with pytest.raises(ValueError, match="k must be an integer >= 1"):
        Parameters({"a": (0.0, 1.0, 2.0)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="lb <= ub required"):
        Parameters({"a": (2.0, 1.0, 1)})
    # NaN bounds compare False both ways, so a plain `lb > ub` check would admit
    # them; the `not lb <= ub` guard must reject NaN in either position.
    with pytest.raises(ValueError, match="lb <= ub required"):
        Parameters({"a": (float("nan"), 1.0, 1)})
    with pytest.raises(ValueError, match="lb <= ub required"):
        Parameters({"a": (0.0, float("nan"), 1)})
    with pytest.raises(ValueError, match=r"spec must be \(lb, ub, k\)"):
        Parameters({"a": (0.0, 1.0)})  # type: ignore[dict-item]


def test_insertion_order_preserved() -> None:
    pairs = [("z", (0.0, 1.0, 1)), ("a", (0.0, 1.0, 2)), ("m", (0.0, 1.0, 1))]
    params = Parameters(pairs)
    assert params.names == ("z", "a", "m")
    assert params.block("z") == slice(0, 1)
    assert params.block("a") == slice(1, 3)
    assert params.block("m") == slice(3, 4)


def test_unknown_block_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown parameter block"):
        make_params().block("delta")


def test_immutable_after_construction() -> None:
    params = make_params()
    with pytest.raises(AttributeError, match="immutable"):
        params.K = 9  # type: ignore[misc]


def test_layout_equality() -> None:
    # Equality gates result merging, so every field of the layout spec
    # (name, lb, ub, size, order) must participate. Each inequality below
    # differs from the baseline in exactly one dimension.
    baseline = Parameters({"beta": (-5.0, 5.0, 2), "gamma": (0.0, 1.0, 1)})
    assert make_params() == baseline
    assert hash(make_params()) == hash(baseline)

    # Different block set / K.
    assert baseline != Parameters({"beta": (-5.0, 5.0, 2)})

    # Same names, same sizes, only a lower bound differs.
    diff_lb = Parameters({"beta": (-4.0, 5.0, 2), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_lb
    # Same names, same sizes, only an upper bound differs.
    diff_ub = Parameters({"beta": (-5.0, 4.0, 2), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_ub

    # Same names, same bounds, only a block size differs (misaligns theta).
    diff_size = Parameters({"beta": (-5.0, 5.0, 3), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_size

    # Same names, same bounds, same sizes, only block order differs.
    reordered = Parameters({"gamma": (0.0, 1.0, 1), "beta": (-5.0, 5.0, 2)})
    assert baseline != reordered

    # Hash tracks the full spec: every field that participates in __eq__ must
    # also change the hash, or the two de-sync. (Distinct hashes for unequal
    # objects is not required by the contract, but a hash that ignores ub or
    # order silently drops fields __eq__ still checks.)
    assert hash(baseline) != hash(diff_lb)
    assert hash(baseline) != hash(diff_ub)
    assert hash(baseline) != hash(diff_size)
    assert hash(baseline) != hash(reordered)

    # Cross-type comparison drives the isinstance/NotImplemented branch: an
    # unrecognized type must compare unequal, never error. Dropping the guard
    # so __eq__ dereferences other._spec() turns each of these into an
    # AttributeError. Cover scalars, containers, None and str so a guard that
    # only special-cases one type still fails.
    for other in (5, object(), None, "beta", (("beta", -5.0, 5.0, 2),)):
        assert (baseline == other) is False
        assert (baseline != other) is True
        # Return the NotImplemented singleton (not a bare False) so the
        # reflected operand gets to answer. A `return False` regression keeps
        # the truthiness assertions above green but silently breaks reflected
        # equality against a wildcard whose __eq__ answers True.
        assert baseline.__eq__(other) is NotImplemented
        assert baseline.__ne__(other) is NotImplemented


def test_pickle_and_deepcopy_round_trip() -> None:
    params = make_params()

    for restored in (
        copy.deepcopy(params),
        pickle.loads(pickle.dumps(params)),
    ):
        assert restored == params
        assert restored is not params
        assert restored.names == ("beta", "gamma")
        # Fractional values so a truncating/int pack buffer on the restored
        # object is caught, not just an integer-coordinate round-trip.
        theta = np.array([1.5, -2.25, 0.75])
        packed = restored.pack(params.unpack(theta))
        assert packed.dtype == np.float64
        np.testing.assert_array_equal(packed, theta)
        with pytest.raises(AttributeError, match="immutable"):
            restored.extra = 1  # type: ignore[attr-defined]
