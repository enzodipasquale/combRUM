from __future__ import annotations

import numpy as np
import pytest

from combrum.context import FitContext
from combrum.transport import SerialTransport

K, N, S = 3, 4, 2


def base_kwargs() -> dict[str, object]:
    # Asymmetric, non-uniform values throughout, so a swapped, reversed, or
    # scrambled field shows up on read-back.
    return dict(
        K=K,
        N=N,
        S=S,
        theta_bounds=(np.array([-1.0, -2.0, -3.0]), np.array([4.0, 5.0, 6.0])),
        theta_coef=np.arange(N * S, dtype=np.float64),
        agent_weights=_ramp_weights(N * S),
        local_ids=np.arange(N * S, dtype=np.int64),
        transport=SerialTransport(),
        tolerance=1e-6,
    )


def _ramp_weights(n: int) -> np.ndarray:
    # Strictly increasing and normalized to sum 1; w[::-1] != w.
    w = np.arange(1, n + 1, dtype=np.float64)
    return w / w.sum()


def make_context(**overrides: object) -> FitContext:
    kwargs = base_kwargs()
    kwargs.update(overrides)
    return FitContext(**kwargs)  # type: ignore[arg-type]


def test_valid_context_accepted() -> None:
    ctx = make_context()
    assert ctx.n_agents == N * S
    assert ctx.theta_init is None

    # Expected values are rebuilt here, not read back from ctx.
    exp_lower = np.array([-1.0, -2.0, -3.0])
    exp_upper = np.array([4.0, 5.0, 6.0])
    lower, upper = ctx.theta_bounds
    np.testing.assert_array_equal(lower, exp_lower)
    np.testing.assert_array_equal(upper, exp_upper)
    assert lower.dtype == np.float64
    assert upper.dtype == np.float64

    np.testing.assert_array_equal(ctx.theta_coef, np.arange(N * S, dtype=np.float64))
    assert ctx.theta_coef.dtype == np.float64

    exp_weights = _ramp_weights(N * S)
    np.testing.assert_allclose(ctx.agent_weights, exp_weights)
    assert ctx.agent_weights.dtype == np.float64

    np.testing.assert_array_equal(ctx.local_ids, np.arange(N * S, dtype=np.int64))
    assert np.issubdtype(ctx.local_ids.dtype, np.integer)

    # Non-zero, non-palindromic seed: a reversed, scrambled, or zeroed
    # warm-start vector cannot match it.
    seed = np.array([7.0, -3.0, 5.0])
    assert seed.shape == (K,)
    ctx = make_context(theta_init=seed)
    assert ctx.theta_init is not None
    assert ctx.theta_init.shape == (K,)
    np.testing.assert_array_equal(ctx.theta_init, np.array([7.0, -3.0, 5.0]))
    assert ctx.theta_init.dtype == np.float64


def test_int_array_inputs_coerced_to_float64() -> None:
    # Integer inputs are the case the float64 coercion exists for; the float64
    # arrays in base_kwargs make it a no-op.
    ctx = make_context(
        theta_bounds=(np.array([-1, -2, -3]), np.array([4, 5, 6])),
        theta_coef=np.arange(N * S, dtype=np.int64),
        agent_weights=np.arange(1, N * S + 1, dtype=np.int64),
        theta_init=np.zeros(K, dtype=np.int64),
    )
    lower, upper = ctx.theta_bounds
    assert lower.dtype == np.float64
    assert upper.dtype == np.float64
    assert ctx.theta_coef.dtype == np.float64
    assert ctx.agent_weights.dtype == np.float64
    assert ctx.theta_init is not None
    assert ctx.theta_init.dtype == np.float64
    # Values still preserved through the coercion.
    np.testing.assert_array_equal(ctx.theta_coef, np.arange(N * S))
    np.testing.assert_array_equal(lower, np.array([-1.0, -2.0, -3.0]))


def test_rejects_wrong_theta_coef_length() -> None:
    # Both directions: a one-sided size check would accept one of these.
    with pytest.raises(ValueError, match=r"theta_coef must have shape \(n_agents,\)"):
        make_context(theta_coef=np.zeros(N * S + 1))
    with pytest.raises(ValueError, match=r"theta_coef must have shape \(n_agents,\)"):
        make_context(theta_coef=np.zeros(N * S - 1))


def test_rejects_wrong_agent_weights_length() -> None:
    with pytest.raises(ValueError, match=r"agent_weights must have shape"):
        make_context(agent_weights=np.zeros(N * S - 1))
    with pytest.raises(ValueError, match=r"agent_weights must have shape"):
        make_context(agent_weights=np.zeros(N * S + 1))


def test_rejects_duplicate_local_ids() -> None:
    with pytest.raises(ValueError, match="local_ids must be unique"):
        make_context(local_ids=np.array([0, 1, 1, 2], dtype=np.int64))


def test_rejects_out_of_range_local_ids() -> None:
    with pytest.raises(ValueError, match=r"local_ids must lie in \[0, n_agents\)"):
        make_context(local_ids=np.array([0, N * S], dtype=np.int64))
    with pytest.raises(ValueError, match=r"local_ids must lie in \[0, n_agents\)"):
        make_context(local_ids=np.array([-1, 0], dtype=np.int64))


def test_rejects_float_local_ids() -> None:
    with pytest.raises(ValueError, match="local_ids must have an integer dtype"):
        make_context(local_ids=np.array([0.0, 1.0]))


def test_rejects_multidimensional_local_ids() -> None:
    # A 2-D array passes the uniqueness check (np.unique flattens) and the
    # range check; only the ndim guard rejects it.
    two_d = np.arange(N * S, dtype=np.int64).reshape(2, 4)
    with pytest.raises(ValueError, match="local_ids must be one-dimensional"):
        make_context(local_ids=two_d)


def test_rejects_lb_above_ub() -> None:
    lower = np.array([0.0, 2.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="lower <= upper"):
        make_context(theta_bounds=(lower, upper))


def test_accepts_equal_bounds() -> None:
    # lower == upper fixes a parameter and must be accepted.
    pinned = np.full(K, 0.5)
    ctx = make_context(theta_bounds=(pinned, pinned.copy()))
    lower, upper = ctx.theta_bounds
    assert np.array_equal(lower, np.full(K, 0.5))
    assert np.array_equal(upper, np.full(K, 0.5))


def test_rejects_wrong_bounds_length() -> None:
    with pytest.raises(ValueError, match=r"theta_bounds lower must have shape \(K,\)"):
        make_context(theta_bounds=(np.zeros(K + 1), np.ones(K + 1)))
    # A size-1 upper broadcasts against (K,) in the lower<=upper comparison,
    # so only the shape guard can catch it; lower stays correct-length here.
    with pytest.raises(ValueError, match=r"theta_bounds upper must have shape \(K,\)"):
        make_context(theta_bounds=(np.zeros(K), np.ones(1)))
    with pytest.raises(ValueError, match=r"theta_bounds upper must have shape \(K,\)"):
        make_context(theta_bounds=(np.zeros(K), np.ones(K + 1)))


def test_rejects_malformed_theta_bounds_container() -> None:
    # Non-tuple, too-long tuple, too-short tuple: the container guard's cases.
    lower = np.zeros(K)
    upper = np.ones(K)
    with pytest.raises(
        ValueError, match=r"theta_bounds must be a \(lower, upper\) 2-tuple"
    ):
        make_context(theta_bounds=[lower, upper])
    with pytest.raises(
        ValueError, match=r"theta_bounds must be a \(lower, upper\) 2-tuple"
    ):
        make_context(theta_bounds=(lower, upper, np.zeros(K)))
    with pytest.raises(
        ValueError, match=r"theta_bounds must be a \(lower, upper\) 2-tuple"
    ):
        make_context(theta_bounds=(lower,))


def test_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        make_context(tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        make_context(tolerance=-1.0)


def test_rejects_wrong_theta_init_length() -> None:
    with pytest.raises(ValueError, match=r"theta_init must have shape \(K,\)"):
        make_context(theta_init=np.zeros(K + 1))
    with pytest.raises(ValueError, match=r"theta_init must have shape \(K,\)"):
        make_context(theta_init=np.zeros(K - 1))


def test_rejects_nonfinite_theta_init() -> None:
    bad = np.zeros(K)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="theta_init must be finite"):
        make_context(theta_init=bad)
    bad = np.zeros(K)
    bad[1] = np.inf
    with pytest.raises(ValueError, match="theta_init must be finite"):
        make_context(theta_init=bad)


def test_rejects_nonpositive_K() -> None:
    with pytest.raises(
        ValueError, match=r"K \(parameter dimension\) must be >= 1; got 0"
    ):
        make_context(K=0, theta_bounds=(np.zeros(0), np.zeros(0)))


def test_rejects_nonpositive_N() -> None:
    with pytest.raises(ValueError, match=r"N \(observations\) must be >= 1; got 0"):
        make_context(
            N=0,
            theta_coef=np.zeros(0),
            agent_weights=np.zeros(0),
            local_ids=np.zeros(0, dtype=np.int64),
        )


def test_rejects_nonpositive_S() -> None:
    with pytest.raises(ValueError, match=r"S \(simulations\) must be >= 1; got 0"):
        make_context(
            S=0,
            theta_coef=np.zeros(0),
            agent_weights=np.zeros(0),
            local_ids=np.zeros(0, dtype=np.int64),
        )


def test_accepts_minimal_dimensions() -> None:
    # K=N=S=1 is the smallest legal geometry and must construct.
    ctx = make_context(
        K=1,
        N=1,
        S=1,
        theta_bounds=(np.zeros(1), np.ones(1)),
        theta_coef=np.zeros(1),
        agent_weights=np.ones(1),
        local_ids=np.zeros(1, dtype=np.int64),
    )
    assert ctx.n_agents == 1
    assert ctx.K == 1 and ctx.N == 1 and ctx.S == 1


def test_transport_required() -> None:
    # Transport has no default, so omitting it raises TypeError rather than
    # silently running unwired — even the serial path goes through it.
    kwargs = base_kwargs()
    del kwargs["transport"]
    with pytest.raises(TypeError):
        FitContext(**kwargs)  # type: ignore[arg-type]


def test_transport_type_validated() -> None:
    with pytest.raises(ValueError, match="transport must implement"):
        make_context(transport=object())


def test_master_fields_default_to_none() -> None:
    # A master-free method needs no master backend, cut policy, or schedule.
    ctx = make_context()
    assert ctx.master_backend is None
    assert ctx.cut_policy is None
    assert ctx.schedule is None
    assert ctx.master_params == {}


def test_master_params_stored_unchanged() -> None:
    ctx = make_context(master_params={"method": 2, "crossover": 0})
    assert ctx.master_params == {"method": 2, "crossover": 0}


def test_master_params_must_be_dict() -> None:
    with pytest.raises(ValueError, match="master_params must be a dict"):
        make_context(master_params=[("method", 2)])


def test_stored_arrays_are_read_only() -> None:
    ctx = make_context(theta_init=np.zeros(K))
    lower, upper = ctx.theta_bounds
    arrays = (
        ctx.theta_coef,
        ctx.agent_weights,
        ctx.local_ids,
        ctx.theta_init,
        lower,
        upper,
    )
    for arr in arrays:
        assert arr is not None
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 99
