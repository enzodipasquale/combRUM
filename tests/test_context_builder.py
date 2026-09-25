from __future__ import annotations

import numpy as np
import pytest

from combrum.engine.context_builder import build_fit_context
from combrum.formulations import NSlack, OneSlack
from combrum.parameters import Parameters
from combrum.transport import SerialTransport


def test_build_fit_context_keeps_gurobi_warm_start_defaults(monkeypatch) -> None:
    import combrum.engine.context_builder as cb

    captured: list[dict[str, object] | None] = []

    class _Master:
        def reinstall(self, rows) -> None:
            raise AssertionError("warm cuts are not part of this test")

    def fake_make_master(
        K,
        bounds,
        c_theta,
        u_coef,
        *,
        backend,
        params,
        n_agents,
        env=None,
    ):
        captured.append(None if params is None else dict(params))
        return _Master()

    def features(_agent_id: int, bundle: np.ndarray):
        return np.asarray(bundle, dtype=np.float64), 0.0

    def observed_features(_agent_id: int, bundle: np.ndarray) -> np.ndarray:
        return np.asarray(bundle, dtype=np.float64)

    user_params = {"TimeLimit": 3.0, "LPWarmStart": 1}
    monkeypatch.setattr(cb, "make_master", fake_make_master)
    build_fit_context(
        Parameters({"theta": (-1.0, 1.0, 2)}),
        observables=[0, 1],
        observed_bundles=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        shocks=np.zeros((2, 1, 2), dtype=np.float64),
        formulation=NSlack(features),
        features=features,
        observed_features=observed_features,
        transport=SerialTransport(),
        master_backend="gurobi",
        master_params=user_params,
    )

    assert user_params == {"TimeLimit": 3.0, "LPWarmStart": 1}
    assert captured == [{"Method": 0, "LPWarmStart": 1, "TimeLimit": 3.0}]


def _features(_agent_id: int, bundle: np.ndarray):
    return np.asarray(bundle, dtype=np.float64), 0.0


def _build_oneslack(**kwargs):
    return build_fit_context(
        Parameters({"theta": (-1.0, 1.0, 2)}),
        observables=[0, 1],
        observed_bundles=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        shocks=np.zeros((2, 1, 2), dtype=np.float64),
        formulation=OneSlack(_features),
        features=_features,
        observed_features=None,
        transport=SerialTransport(),
        master_backend="highs",
        **kwargs,
    )


def test_build_fit_context_rejects_warm_cuts_for_oneslack() -> None:
    with pytest.raises(ValueError, match="warm_cuts are not supported for OneSlack"):
        _build_oneslack(warm_cuts=())


def test_build_fit_context_scales_a_fresh_oneslack_master(monkeypatch) -> None:
    import combrum.engine.context_builder as cb

    captured: dict[str, object] = {}

    def fake_make_master(
        K, bounds, c_theta, u_coef, *, backend, params, n_agents, env=None
    ):
        captured.update(c_theta=c_theta, params=params)

    monkeypatch.setattr(cb, "make_master", fake_make_master)
    built = _build_oneslack(
        weights=np.array([1.0, 3.0]), master_params={"u_lower_bound": -8.0}
    )

    # The master works per unit of total weight: the objective, and with it
    # u's floor, are divided by 4 while the context reports the unit.
    assert built.ctx.master_scale == 4.0
    np.testing.assert_array_equal(captured["c_theta"], built.c_theta / 4.0)
    assert captured["params"] == {"u_lower_bound": -2.0}
