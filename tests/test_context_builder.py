from __future__ import annotations

import numpy as np

from combrum.engine.context_builder import build_fit_context
from combrum.formulations import NSlack
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
