from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np

from combrum.engine.estimate import estimate_distributed
from combrum.engine.driver import LoopDiagnostics, LoopOutcome
from combrum.formulation import FormulationResult
import pytest

from combrum.context import ResultPublication
from combrum.cut_policies import AddAll, PurgeInactive
from combrum.formulations import NSlack, OneSlack
from combrum.model import Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport, TransportError
from combrum.transport.base import CutRow


class _Oracle(Oracle):
    pass


def _fail_serial_builder(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise AssertionError("estimate_distributed must not call build_fit_context")


def _converged_outcome(K: int) -> LoopOutcome:
    return LoopOutcome(
        result=FormulationResult(
            theta_hat=np.zeros(K, dtype=np.float64),
            objective=0.0,
            n_active_cuts=0,
        ),
        diagnostics=LoopDiagnostics(
            converged=True,
            iterations=0,
            cuts_admitted=0,
        ),
    )


class _ObservedSurface:
    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        self.setup_ids = tuple(map(int, observation_ids))

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(observation_ids, dtype=np.float64)
        return np.ascontiguousarray(
            np.column_stack([ids + 1.0, 2.0 * ids + 1.0]),
            dtype=np.float64,
        )

    def __call__(self, agent_id: int, bundle: np.ndarray):
        b = np.asarray(bundle, dtype=np.float64)
        return np.array([b[0], b[0]], dtype=np.float64), 0.0


def test_estimate_distributed_builds_split_axis_context(monkeypatch) -> None:
    estimate_module = importlib.import_module("combrum.engine.estimate")
    surface = _ObservedSurface()
    seen = {}

    def fake_run_fit(ctx, oracle, formulation, config, *, demand_sink=None):  # type: ignore[no-untyped-def]
        seen["ctx"] = ctx
        seen["config"] = config
        return _converged_outcome(ctx.K)

    monkeypatch.setattr(estimate_module, "build_fit_context", _fail_serial_builder)
    monkeypatch.setattr(estimate_module, "run_fit", fake_run_fit)
    # qp_weight>0 makes require_quadratic true; fix the resolved backend so
    # loop-control propagation is under test, not backend selection.
    monkeypatch.setattr(
        estimate_module, "resolve_master_backend", lambda *args, **kwargs: "highs"
    )
    warm_start = SimpleNamespace(
        theta_hat=np.array([0.25, -0.25], dtype=np.float64)
    )

    def sentinel_callback(iteration, oracle):  # type: ignore[no-untyped-def]
        return None

    result = estimate_distributed(
        Model(
            _Oracle(),
            Parameters({"theta": (-2.0, 2.0, 2)}),
            features=surface,
            observed_features=surface,
            formulation=NSlack,
        ),
        n_observations=5,
        n_simulations=4,
        transport=SerialTransport(),
        master_backend="highs",
        min_iterations=3,
        max_iterations=7,
        tolerance=7e-3,
        qp_weight=0.5,
        qp_iterations=3,
        penalty_ref="dynamic",
        iteration_callback=sentinel_callback,
        warm_start=warm_start,
    )

    ctx = seen["ctx"]
    assert ctx.N == 5
    assert ctx.S == 4
    assert ctx.n_agents == 20
    assert ctx.weight_mode == "distributed"
    assert ctx.theta_coef is None
    assert ctx.agent_weights is None
    # Serial owns everything, so local_ids is arange(20) under any blocking
    # order; the layout itself is covered by the agent-axis shard test below.
    np.testing.assert_array_equal(ctx.local_ids, np.arange(20, dtype=np.int64))
    np.testing.assert_array_equal(ctx.theta_init, warm_start.theta_hat)
    assert ctx.theta_init.flags.writeable is False
    assert ctx.tolerance == 7e-3
    # Every loop control must reach the driver's config; each value here is
    # non-default so a hardcoded constant would not reproduce it.
    config = seen["config"]
    assert config.max_iterations == 7
    assert config.min_iterations == 3
    assert config.qp_weight == 0.5
    assert config.qp_iterations == 3
    assert config.penalty_ref == "dynamic"
    assert config.iteration_callback is sentinel_callback
    np.testing.assert_allclose(
        result.empirical_moment,
        np.array([3.0, 5.0], dtype=np.float64),
    )
    # setup_observed receives the owned observation shard; serial owns all five.
    assert surface.setup_ids == tuple(range(5))


def test_estimate_distributed_defaults_to_summary_publication(monkeypatch) -> None:
    estimate_module = importlib.import_module("combrum.engine.estimate")
    surface = _ObservedSurface()
    seen = {}

    def fake_run_fit(ctx, oracle, formulation, config, *, demand_sink=None):  # type: ignore[no-untyped-def]
        seen["ctx"] = ctx
        return _converged_outcome(ctx.K)

    monkeypatch.setattr(estimate_module, "build_fit_context", _fail_serial_builder)
    monkeypatch.setattr(estimate_module, "run_fit", fake_run_fit)

    result = estimate_distributed(
        Model(
            _Oracle(),
            Parameters({"theta": (-2.0, 2.0, 2)}),
            features=surface,
            observed_features=surface,
            formulation=NSlack,
        ),
        n_observations=5,
        n_simulations=4,
        transport=SerialTransport(),
        master_backend="highs",
        max_iterations=1,
    )

    assert seen["ctx"].result_publication is ResultPublication.SUMMARY
    assert result.slack is None
    assert result.cuts is None
    assert result.cut_duals is None


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"return_slack": True}, ResultPublication.SLACK),
        ({"return_cuts": True}, ResultPublication.ACTIVE_SET),
        ({"return_cut_duals": True}, ResultPublication.DUAL),
        (
            {"return_slack": True, "return_cuts": True, "return_cut_duals": True},
            ResultPublication.SLACK
            | ResultPublication.ACTIVE_SET
            | ResultPublication.DUAL,
        ),
    ],
)
def test_estimate_distributed_publication_flags_reach_result(
    monkeypatch, flags: dict[str, bool], expected: ResultPublication
) -> None:
    estimate_module = importlib.import_module("combrum.engine.estimate")
    surface = _ObservedSurface()
    seen = {}
    # The formulation owns artifact gating, so whatever it publishes must
    # reach the FitResult untouched.
    slack = np.arange(20, dtype=np.float64)
    active_set = object()
    dual = object()

    def fake_run_fit(ctx, oracle, formulation, config, *, demand_sink=None):  # type: ignore[no-untyped-def]
        seen["ctx"] = ctx
        return LoopOutcome(
            result=FormulationResult(
                theta_hat=np.zeros(ctx.K, dtype=np.float64),
                objective=0.0,
                n_active_cuts=1,
                slack=slack,
                active_set=active_set,
                dual=dual,
            ),
            diagnostics=LoopDiagnostics(
                converged=True,
                iterations=0,
                cuts_admitted=0,
            ),
        )

    monkeypatch.setattr(estimate_module, "build_fit_context", _fail_serial_builder)
    monkeypatch.setattr(estimate_module, "run_fit", fake_run_fit)

    result = estimate_distributed(
        Model(
            _Oracle(),
            Parameters({"theta": (-2.0, 2.0, 2)}),
            features=surface,
            observed_features=surface,
            formulation=NSlack,
        ),
        n_observations=5,
        n_simulations=4,
        transport=SerialTransport(),
        master_backend="highs",
        max_iterations=1,
        **flags,
    )

    assert seen["ctx"].result_publication == expected
    np.testing.assert_array_equal(result.slack, slack)
    assert result.cuts is active_set
    assert result.cut_duals is dual


def test_estimate_distributed_empirical_moment_divides_by_global_n(
    monkeypatch,
) -> None:
    # N=5 over two ranks gives shards [0,1,2] and [3,4]: no rank owns all N
    # rows, so dividing by N and dividing by shard size disagree — serial
    # (owned_obs.size == N) cannot tell them apart.
    estimate_module = importlib.import_module("combrum.engine.estimate")

    def fake_run_fit(ctx, oracle, formulation, config, *, demand_sink=None):  # type: ignore[no-untyped-def]
        return _converged_outcome(ctx.K)

    monkeypatch.setattr(estimate_module, "build_fit_context", _fail_serial_builder)
    monkeypatch.setattr(estimate_module, "run_fit", fake_run_fit)

    def run(transport):
        surface = _ObservedSurface()
        result = estimate_distributed(
            Model(
                _Oracle(),
                Parameters({"theta": (-2.0, 2.0, 2)}),
                features=surface,
                observed_features=surface,
                formulation=NSlack,
            ),
            n_observations=5,
            n_simulations=4,
            transport=transport,
            master_backend="highs",
            max_iterations=1,
        )
        return np.asarray(result.empirical_moment, dtype=np.float64).copy()

    # Rows i -> [i+1, 2i+1] for i in 0..4: sum(phi) = [15, 25], /N = [3, 5].
    for moment in LocalCluster(2).run(run):
        np.testing.assert_allclose(
            moment, np.array([3.0, 5.0], dtype=np.float64)
        )


def test_estimate_distributed_local_ids_use_agent_axis_shards(
    monkeypatch,
) -> None:
    # Observed moments shard over N, but pricing must shard over N*S so every
    # simulated agent is assigned once even when ranks exceed N.
    estimate_module = importlib.import_module("combrum.engine.estimate")

    captured: dict[int, np.ndarray] = {}

    def fake_run_fit(ctx, oracle, formulation, config, *, demand_sink=None):  # type: ignore[no-untyped-def]
        captured[ctx.transport.rank] = np.asarray(ctx.local_ids).copy()
        return _converged_outcome(ctx.K)

    monkeypatch.setattr(estimate_module, "build_fit_context", _fail_serial_builder)
    monkeypatch.setattr(estimate_module, "run_fit", fake_run_fit)

    def run(transport):
        surface = _ObservedSurface()
        estimate_distributed(
            Model(
                _Oracle(),
                Parameters({"theta": (-2.0, 2.0, 2)}),
                features=surface,
                observed_features=surface,
                formulation=NSlack,
            ),
            n_observations=5,
            n_simulations=4,
            transport=transport,
            master_backend="highs",
            max_iterations=1,
        )
        return transport.rank

    LocalCluster(2).run(run)

    np.testing.assert_array_equal(
        captured[0],
        np.arange(0, 10, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        captured[1],
        np.arange(10, 20, dtype=np.int64),
    )


def test_estimate_distributed_requires_rank_uniform_loop_controls() -> None:
    def run(transport):
        surface = _ObservedSurface()
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=NSlack,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                max_iterations=1 if transport.rank == 0 else 2,
                master_backend="highs",
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "max_iterations must match" in message
        for message in LocalCluster(2).run(run)
    )


@pytest.mark.parametrize("bad", [True, np.bool_(True)], ids=["bool", "np.bool_"])
def test_estimate_distributed_rejects_bool_tolerance(bad: object) -> None:
    # float(np.bool_(True)) is 1.0, so only the explicit bool guard in the
    # float agreement path can reject it; builtin True must fail identically.
    surface = _ObservedSurface()
    with pytest.raises(
        TypeError, match="tolerance must be a finite float, got bool"
    ):
        estimate_distributed(
            Model(
                _Oracle(),
                Parameters({"theta": (-2.0, 2.0, 2)}),
                features=surface,
                observed_features=surface,
                formulation=NSlack,
            ),
            n_observations=5,
            n_simulations=4,
            transport=SerialTransport(),
            master_backend="highs",
            tolerance=bad,  # type: ignore[arg-type]
        )


def test_estimate_distributed_requires_rank_uniform_formulation_support() -> None:
    def run(transport):
        surface = _ObservedSurface()
        formulation = NSlack if transport.rank == 0 else OneSlack
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=formulation,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                master_backend="highs",
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "model.formulation is NSlack must match" in message
        for message in LocalCluster(2).run(run)
    )


def test_estimate_distributed_requires_rank_uniform_master_backend() -> None:
    def run(transport):
        surface = _ObservedSurface()
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=NSlack,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                # Both are valid backends, so only cross-rank disagreement
                # can raise here.
                master_backend="highs" if transport.rank == 0 else "gurobi",
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "master_backend must match" in message
        for message in LocalCluster(2).run(run)
    )


@pytest.mark.parametrize(
    "flag", ["return_slack", "return_cuts", "return_cut_duals"]
)
def test_estimate_distributed_requires_rank_uniform_publication_flags(
    flag: str,
) -> None:
    def run(transport):
        surface = _ObservedSurface()
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=NSlack,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                master_backend="highs",
                max_iterations=1,
                **{flag: transport.rank == 0},
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        f"{flag} must match" in message for message in LocalCluster(2).run(run)
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("presence", "warm_start must match"),
        ("shape", "warm_start must match"),
        ("value", "warm_start must match"),
        ("access", "warm_start must match"),
        # Both ranks fail identically, so cross-rank disagreement cannot fire;
        # the message must come from the access guard itself.
        ("access_both", "warm_start.theta_hat could not be read"),
        ("nonfinite_head", "warm_start.theta_hat must be finite"),
        ("nonfinite_tail", "warm_start.theta_hat must be finite"),
        ("inf_tail", "warm_start.theta_hat must be finite"),
    ],
)
def test_estimate_distributed_requires_rank_uniform_warm_start(
    monkeypatch, case: str, match: str
) -> None:
    estimate_module = importlib.import_module("combrum.engine.estimate")
    monkeypatch.setattr(
        estimate_module, "resolve_master_backend", lambda *args, **kwargs: "highs"
    )

    def warm(theta):
        return SimpleNamespace(theta_hat=np.asarray(theta, dtype=np.float64))

    class _BrokenWarmStart:
        @property
        def theta_hat(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("theta boom")

    def run(transport):
        if case == "presence":
            warm_start = warm([0.0, 0.0]) if transport.rank == 0 else None
        elif case == "shape":
            warm_start = warm([0.0, 0.0] if transport.rank == 0 else [0.0, 0.0, 0.0])
        elif case == "value":
            warm_start = warm([0.0, 0.0] if transport.rank == 0 else [0.0, 0.25])
        elif case == "access":
            warm_start = warm([0.0, 0.0]) if transport.rank == 0 else _BrokenWarmStart()
        elif case == "access_both":
            warm_start = _BrokenWarmStart()
        elif case == "nonfinite_head":
            warm_start = warm([np.nan, 0.0])
        elif case == "nonfinite_tail":
            # NaN away from index 0, in case only the first entry is inspected.
            warm_start = warm([0.0, np.nan])
        else:
            warm_start = warm([0.0, np.inf])
        surface = _ObservedSurface()
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=NSlack,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                master_backend="highs",
                warm_start=warm_start,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(match in message for message in LocalCluster(2).run(run))


def test_estimate_distributed_serial_rejects_unreadable_warm_start(monkeypatch) -> None:
    # size==1 has no cross-rank axis, so the access guard alone must reject an
    # unreadable theta_hat rather than silently coercing it to None.
    estimate_module = importlib.import_module("combrum.engine.estimate")
    monkeypatch.setattr(
        estimate_module, "resolve_master_backend", lambda *args, **kwargs: "highs"
    )

    class _BrokenWarmStart:
        @property
        def theta_hat(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("theta boom")

    surface = _ObservedSurface()
    with pytest.raises(ValueError, match="warm_start.theta_hat could not be read"):
        estimate_distributed(
            Model(
                _Oracle(),
                Parameters({"theta": (-2.0, 2.0, 2)}),
                features=surface,
                observed_features=surface,
                formulation=NSlack,
            ),
            n_observations=5,
            n_simulations=4,
            transport=SerialTransport(),
            master_backend="highs",
            warm_start=_BrokenWarmStart(),
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("master_params", "master_params must match"),
        ("cut_policy", "cut_policy must match"),
        ("warm_cuts", "warm_cuts must match"),
    ],
)
def test_estimate_distributed_requires_rank_uniform_object_templates(
    case: str, match: str
) -> None:
    def run(transport):
        kwargs = {
            "master_params": None,
            "cut_policy": None,
            "warm_cuts": None,
        }
        if case == "master_params":
            kwargs["master_params"] = (
                {"presolve": "on"} if transport.rank == 0 else {"presolve": "off"}
            )
        elif case == "cut_policy":
            kwargs["cut_policy"] = (
                AddAll() if transport.rank == 0 else PurgeInactive(max_age=1)
            )
        else:
            kwargs["warm_cuts"] = (
                ()
                if transport.rank == 0
                else (
                    CutRow(
                        rep_id=0,
                        agent_id=0,
                        phi=np.array([1.0, 0.0], dtype=np.float64),
                        epsilon=0.0,
                        bundle_key=b"b",
                    ),
                )
            )
        surface = _ObservedSurface()
        try:
            estimate_distributed(
                Model(
                    _Oracle(),
                    Parameters({"theta": (-2.0, 2.0, 2)}),
                    features=surface,
                    observed_features=surface,
                    formulation=NSlack,
                ),
                n_observations=5,
                n_simulations=4,
                transport=transport,
                master_backend="highs",
                max_iterations=1,
                **kwargs,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(match in message for message in LocalCluster(2).run(run))


def test_estimate_distributed_rejects_unsupported_formulation_before_observed_setup() -> None:
    class _ExplodingObserved(_ObservedSurface):
        def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
            raise AssertionError("observed setup should not run")

    surface = _ExplodingObserved()
    with pytest.raises(NotImplementedError, match="NSlack"):
        estimate_distributed(
            Model(
                _Oracle(),
                Parameters({"theta": (-2.0, 2.0, 2)}),
                features=surface,
                observed_features=surface,
                formulation=OneSlack,
            ),
            n_observations=5,
            n_simulations=4,
            transport=SerialTransport(),
            master_backend="highs",
        )
