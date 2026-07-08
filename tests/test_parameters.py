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
    # fractional, signed values inside the bounds
    theta = np.array([1.5, -2.25, 0.75])
    named = params.unpack(theta)
    np.testing.assert_array_equal(named["beta"], [1.5, -2.25])
    np.testing.assert_array_equal(named["gamma"], [0.75])
    packed = params.pack(named)
    assert packed.dtype == np.float64
    np.testing.assert_array_equal(packed, theta)


def test_unpack_validates_length() -> None:
    # oversized and undersized both raise (K=3 from the fixture)
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        make_params().unpack(np.zeros(4))
    with pytest.raises(ValueError, match=r"theta must have shape \(K,\)"):
        make_params().unpack(np.zeros(2))


def test_unpack_rejects_wrong_dimensionality() -> None:
    # right total size but 2-D -- the full shape must be checked, not just size
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
    # wrong length on a non-first block
    with pytest.raises(ValueError, match="block 'gamma' must have length 1"):
        params.pack({"beta": np.zeros(2), "gamma": np.zeros(5)})
    # scalar / undersized values must not broadcast into the size-2 block
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
    # NaN compares False both ways, so a plain `lb > ub` check would admit it
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
    # equality gates result merging, so every spec field must participate:
    # block set, bounds, sizes, and order
    baseline = Parameters({"beta": (-5.0, 5.0, 2), "gamma": (0.0, 1.0, 1)})
    assert make_params() == baseline
    assert hash(make_params()) == hash(baseline)

    assert baseline != Parameters({"beta": (-5.0, 5.0, 2)})

    diff_lb = Parameters({"beta": (-4.0, 5.0, 2), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_lb
    diff_ub = Parameters({"beta": (-5.0, 4.0, 2), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_ub

    diff_size = Parameters({"beta": (-5.0, 5.0, 3), "gamma": (0.0, 1.0, 1)})
    assert baseline != diff_size

    reordered = Parameters({"gamma": (0.0, 1.0, 1), "beta": (-5.0, 5.0, 2)})
    assert baseline != reordered

    # distinct hashes are not required by the contract, but the hash should
    # cover the same fields as __eq__
    assert hash(baseline) != hash(diff_lb)
    assert hash(baseline) != hash(diff_ub)
    assert hash(baseline) != hash(diff_size)
    assert hash(baseline) != hash(reordered)

    # unrecognized types compare unequal, never raise
    for other in (5, object(), None, "beta", (("beta", -5.0, 5.0, 2),)):
        assert (baseline == other) is False
        assert (baseline != other) is True
        # NotImplemented (not False) so the reflected operand gets to answer
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
        theta = np.array([1.5, -2.25, 0.75])
        packed = restored.pack(params.unpack(theta))
        assert packed.dtype == np.float64
        np.testing.assert_array_equal(packed, theta)
        with pytest.raises(AttributeError, match="immutable"):
            restored.extra = 1  # type: ignore[attr-defined]
