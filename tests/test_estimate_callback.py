"""Point-estimation callback contract at the public ``estimate`` boundary."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from combrum.engine import LoopConfig, LoopDiagnostics, LoopOutcome, run_fit
from combrum.formulation import FormulationResult
from combrum.model import Data, Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport, TransportError


class _Oracle(Oracle):
    pass


class _Formulation:
    def __init__(self, features) -> None:  # type: ignore[no-untyped-def]
        self.features = features


@pytest.fixture
def parameters() -> Parameters:
    return Parameters({"theta": (-1.0, 1.0, 1)})


@pytest.fixture
def model(parameters) -> Model:  # type: ignore[no-untyped-def]
    return Model(
        _Oracle(),
        parameters,
        features=lambda agent_id, bundle: (np.zeros(1), 0.0),
        formulation=_Formulation,
    )


@pytest.fixture
def data() -> Data:
    return Data(
        observed_bundles=np.zeros((1, 1), dtype=bool),
        shocks=np.zeros((1, 1, 1)),
        observables=np.zeros(1),
    )


def test_estimate_wires_iteration_callback_into_loopconfig(
    monkeypatch, parameters, model, data
) -> None:
    estimate_mod = importlib.import_module("combrum.engine.estimate")
    oracle = model.oracle
    captured: dict[str, object] = {}

    def fake_build_fit_context(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            ctx=SimpleNamespace(),
            empirical_moment=np.zeros(parameters.K, dtype=np.float64),
        )

    def fake_run_fit(  # type: ignore[no-untyped-def]
        ctx, oracle_arg, formulation, config, *, demand_sink=None
    ):
        captured["floor"] = config.iteration_callback(0, oracle_arg)
        captured["oracle"] = oracle_arg
        return LoopOutcome(
            result=FormulationResult(
                theta_hat=np.zeros(parameters.K, dtype=np.float64),
                objective=0.0,
                n_active_cuts=0,
            ),
            diagnostics=LoopDiagnostics(
                converged=True,
                iterations=1,
                cuts_admitted=0,
            ),
        )

    monkeypatch.setattr(estimate_mod, "build_fit_context", fake_build_fit_context)
    monkeypatch.setattr(estimate_mod, "run_fit", fake_run_fit)

    def callback(iteration: int, oracle_arg: Oracle) -> int:
        assert iteration == 0
        assert oracle_arg is oracle
        return 7

    estimate_mod.estimate(
        model,
        data,
        transport=SerialTransport(),
        iteration_callback=callback,
    )

    assert captured == {"floor": 7, "oracle": oracle}


def _convergence_floor(callback, *, base_floor, iteration=3, oracle=None):
    from combrum.engine.agreement import callback_convergence_floor

    return callback_convergence_floor(
        name="iteration_callback floor",
        callback=callback,
        iteration=iteration,
        oracle=oracle,
        base_floor=base_floor,
        transport=SerialTransport(),
    )


def test_callback_convergence_floor_takes_max_of_base_and_callback() -> None:
    # The callback can raise the floor above the base (decay/min-derived)
    # value but never lower it.
    assert _convergence_floor(lambda it, oracle: 2, base_floor=5) == 5
    assert _convergence_floor(lambda it, oracle: 9, base_floor=5) == 9
    assert _convergence_floor(lambda it, oracle: 6, base_floor=6) == 6


def test_callback_convergence_floor_none_falls_back_to_base() -> None:
    # Returning None opts out of raising the floor this iteration.
    assert _convergence_floor(lambda it, oracle: None, base_floor=4) == 4
    assert _convergence_floor(None, base_floor=6) == 6


def test_callback_convergence_floor_forwards_iteration_and_oracle() -> None:
    seen: dict[str, object] = {}
    marker = object()

    def callback(iteration: int, oracle_arg: object) -> int:
        seen["iteration"] = iteration
        seen["oracle"] = oracle_arg
        return 0

    _convergence_floor(callback, base_floor=1, iteration=42, oracle=marker)
    assert seen == {"iteration": 42, "oracle": marker}


def test_callback_convergence_floor_rejects_negative_return() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        _convergence_floor(lambda it, oracle: -1, base_floor=3)


def test_callback_convergence_floor_rejects_non_integer_return() -> None:
    with pytest.raises(TypeError, match="must be an integer; got bool"):
        _convergence_floor(lambda it, oracle: True, base_floor=3)
    with pytest.raises(TypeError, match="must be an integer; got float"):
        _convergence_floor(lambda it, oracle: 1.5, base_floor=3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iterations": 0}, "max_iterations must be >= 1"),
        ({"min_iterations": -1}, "min_iterations must be >= 0"),
        ({"qp_weight": 1.0, "decay": 0}, "qp_weight>0 needs decay>=1"),
        ({"penalty_ref": "moving"}, "penalty_ref must be"),
    ],
)
def test_estimate_validates_loop_controls_before_context(
    monkeypatch, kwargs, match, model, data
) -> None:
    estimate_mod = importlib.import_module("combrum.engine.estimate")

    def fail_build_fit_context(*args, **build_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("build_fit_context should not run")

    monkeypatch.setattr(
        estimate_mod, "build_fit_context", fail_build_fit_context
    )

    with pytest.raises(ValueError, match=match):
        estimate_mod.estimate(
            model,
            data,
            transport=SerialTransport(),
            **kwargs,
        )


def test_estimate_rejects_multirank_dense_transport(model, data) -> None:
    estimate_mod = importlib.import_module("combrum.engine.estimate")

    def probe_transport(transport):
        try:
            estimate_mod.estimate(model, data, transport=transport)
        except ValueError as exc:
            return "does not support non-serial transport" in str(exc)
        return False

    assert LocalCluster(2).run(probe_transport) == [True, True]


def test_run_fit_propagates_rank_local_oracle_setup_failure() -> None:
    class _SetupFailingOracle(Oracle):
        def setup(self, transport, local_ids):  # type: ignore[no-untyped-def]
            if transport.rank == 1:
                raise RuntimeError("setup boom")

    class _NeverSetupFormulation:
        def setup(self, ctx):  # type: ignore[no-untyped-def]
            raise AssertionError("formulation setup should not run")

        def dispose(self) -> None:
            pass

    def run_setup(transport):
        ctx = SimpleNamespace(
            transport=transport,
            owner_rank=0,
            n_agents=2,
            local_ids=np.array([transport.rank], dtype=np.int64),
            tolerance=1e-6,
            K=1,
            theta_init=None,
            master_backend=None,
            N=1,
            S=2,
            cut_policy=None,
        )
        try:
            run_fit(
                ctx,
                _SetupFailingOracle(),
                _NeverSetupFormulation(),
                LoopConfig(max_iterations=1),
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "setup boom" in message for message in LocalCluster(2).run(run_setup)
    )


# (qp_weight, decay) pairs accepted by _validate_loop_controls. (0.0, 1) is
# the legal config -- penalty weight off, decay set -- where the two factors
# of the AND disagree.
_REQUIRE_QUADRATIC_CASES = [
    (0.0, 0),
    (0.0, 1),
    (1.0, 1),
    (0.5, 2),
]


def _expected_require_quadratic(qp_weight: float, decay: int) -> bool:
    # A quadratic-capable backend is needed exactly when the proximal penalty
    # is live: a positive weight that decays over at least one iteration.
    penalty_live = {
        (0.0, 0): False,
        (0.0, 1): False,
        (1.0, 1): True,
        (0.5, 2): True,
    }
    return penalty_live[(qp_weight, decay)]


@pytest.mark.parametrize(("qp_weight", "decay"), _REQUIRE_QUADRATIC_CASES)
def test_estimate_require_quadratic_tracks_penalty_regime(
    monkeypatch, model, data, qp_weight, decay
) -> None:
    estimate_mod = importlib.import_module("combrum.engine.estimate")
    captured_require_quadratic: list[bool] = []

    # Sentinel so estimate halts right after backend resolution on both the
    # penalty-on and penalty-off paths.
    class _StopAfterResolve(Exception):
        pass

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        captured_require_quadratic.append(kwargs["require_quadratic"])
        raise _StopAfterResolve

    def fail_build_fit_context(*args, **build_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("build_fit_context should not run")

    monkeypatch.setattr(estimate_mod, "resolve_master_backend", fake_resolve)
    monkeypatch.setattr(
        estimate_mod, "build_fit_context", fail_build_fit_context
    )

    with pytest.raises(_StopAfterResolve):
        estimate_mod.estimate(
            model,
            data,
            transport=SerialTransport(),
            master_backend="highs",
            qp_weight=qp_weight,
            decay=decay,
        )

    assert captured_require_quadratic == [
        _expected_require_quadratic(qp_weight, decay)
    ]


def test_estimate_require_quadratic_truth_table_matches_and_semantics() -> None:
    # The penalty_live table must encode exactly `qp_weight > 0 and decay > 0`.
    for qp_weight, decay in _REQUIRE_QUADRATIC_CASES:
        both_factors_hold = qp_weight > 0.0 and decay > 0
        assert _expected_require_quadratic(qp_weight, decay) is both_factors_hold
