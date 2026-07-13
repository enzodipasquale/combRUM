from __future__ import annotations

import inspect
import importlib
from pathlib import Path
import re
import typing

import combrum
import combrum.engine as engine_module
from combrum import (
    ActivityConfig,
    AddAll,
    BootstrapResult,
    CutRow,
    Data,
    Demand,
    DemandBatch,
    DualInformed,
    FeatureMap,
    FitResult,
    LoopConfig,
    LocalCluster,
    Model,
    MpiTransport,
    ExponentialDraws,
    NSlack,
    OneSlack,
    Oracle,
    Parameters,
    Phase,
    PersistentMasterFit,
    PurgeInactive,
    ReplayedWeights,
    RepricingSchedule,
    ResolveAll,
    RoundRobin,
    RunInfoLevel,
    TimeoutSchedule,
    SerialTransport,
    SlackStrip,
    SolverSettings,
    Transport,
    WeightSource,
    bootstrap,
    bootstrap_distributed,
    bootstrap_timeout_callback,
    estimate,
    estimate_distributed,
    point_timeout_callback,
)

def test_published_import_root_exports_public_surface() -> None:
    expected = [
        "ActivityConfig",
        "AddAll",
        "BootstrapResult",
        "CutRow",
        "Data",
        "Demand",
        "DemandBatch",
        "DualInformed",
        "ExponentialDraws",
        "FeatureMap",
        "FitResult",
        "LoopConfig",
        "LocalCluster",
        "Model",
        "MpiTransport",
        "NSlack",
        "OneSlack",
        "Oracle",
        "Parameters",
        "PersistentMasterFit",
        "Phase",
        "PurgeInactive",
        "ReplayedWeights",
        "RepricingSchedule",
        "ResolveAll",
        "RoundRobin",
        "RunInfoLevel",
        "SerialTransport",
        "SlackStrip",
        "SolverSettings",
        "TimeoutSchedule",
        "Transport",
        "WeightSource",
        "bootstrap",
        "bootstrap_distributed",
        "bootstrap_timeout_callback",
        "estimate",
        "estimate_distributed",
        "point_timeout_callback",
    ]

    assert combrum.ActivityConfig is ActivityConfig
    assert combrum.AddAll is AddAll
    assert combrum.BootstrapResult is BootstrapResult
    assert combrum.CutRow is CutRow
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)?", combrum.__version__)
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib

    metadata = tomllib.loads(pyproject.read_text())
    assert metadata["tool"]["hatch"]["version"]["path"] == "src/combrum/_version.py"
    assert combrum.__all__ == expected
    assert combrum.Data is Data
    assert combrum.Demand is Demand
    assert combrum.DemandBatch is DemandBatch
    assert combrum.DualInformed is DualInformed
    assert combrum.FeatureMap is FeatureMap
    assert combrum.FitResult is FitResult
    assert combrum.LoopConfig is LoopConfig
    assert combrum.LocalCluster is LocalCluster
    assert combrum.Model is Model
    assert combrum.MpiTransport is MpiTransport
    assert combrum.NSlack is NSlack
    assert combrum.ExponentialDraws is ExponentialDraws
    assert combrum.OneSlack is OneSlack
    assert combrum.Oracle is Oracle
    assert combrum.Parameters is Parameters
    assert combrum.PersistentMasterFit is PersistentMasterFit
    assert combrum.Phase is Phase
    assert combrum.PurgeInactive is PurgeInactive
    assert combrum.RepricingSchedule is RepricingSchedule
    assert combrum.ResolveAll is ResolveAll
    assert combrum.RoundRobin is RoundRobin
    assert combrum.TimeoutSchedule is TimeoutSchedule
    assert combrum.SerialTransport is SerialTransport
    assert combrum.SlackStrip is SlackStrip
    assert combrum.SolverSettings is SolverSettings
    assert combrum.Transport is Transport
    assert combrum.WeightSource is WeightSource
    assert combrum.bootstrap_timeout_callback is bootstrap_timeout_callback
    assert combrum.bootstrap is bootstrap
    assert combrum.bootstrap_distributed is bootstrap_distributed
    assert combrum.ReplayedWeights is ReplayedWeights
    assert combrum.RunInfoLevel is RunInfoLevel
    assert combrum.estimate is estimate
    assert combrum.estimate_distributed is estimate_distributed
    assert combrum.point_timeout_callback is point_timeout_callback


def test_distributed_entry_point_identities() -> None:
    estimate_module = importlib.import_module("combrum.engine.estimate")

    assert combrum.estimate_distributed is estimate_module.estimate_distributed
    assert engine_module.estimate_distributed is estimate_module.estimate_distributed


def test_distributed_public_signatures_drop_single_process_params() -> None:
    boot_params = inspect.signature(combrum.bootstrap_distributed).parameters
    est_params = inspect.signature(combrum.estimate_distributed).parameters

    # single-process-only params the distributed entry points must not accept
    for removed in (
        "data",
        "observed_bundles",
        "weight_source",
        "collect_payload",
        "only_converged",
        "checkpoint_dir",
        "load_dir",
    ):
        assert removed not in boot_params, removed
    for name in ("n_observations", "n_simulations", "n_bootstrap", "base_seed", "transport"):
        assert boot_params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert boot_params[name].default is inspect.Parameter.empty
    assert boot_params["max_live_reps"].kind is inspect.Parameter.KEYWORD_ONLY
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in boot_params.values())

    for removed in (
        "data",
        "weights",
        "schedule",
        "result_publication",
    ):
        assert removed not in est_params
    for name in ("n_observations", "n_simulations", "transport"):
        assert est_params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert est_params[name].default is inspect.Parameter.empty
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in est_params.values())

    # distributed entry point defaults
    RIL = combrum.RunInfoLevel
    expected_boot_defaults = {
        "max_live_reps": 64,
        "master_backend": "auto",
        "master_params": None,
        "tolerance": 1e-06,
        "max_iterations": 1000,
        "min_iterations": 0,
        "iteration_callback": None,
        "warm_start": None,
        "warm_cuts": None,
        "cut_policy": None,
        "dual_store_dir": None,
        "activity": None,
    }
    expected_est_defaults = {
        "master_backend": "auto",
        "master_params": None,
        "tolerance": 1e-06,
        "max_iterations": 1000,
        "min_iterations": 0,
        "qp_weight": 0.0,
        "qp_iterations": 0,
        "penalty_ref": "static",
        "iteration_callback": None,
        "warm_start": None,
        "warm_cuts": None,
        "cut_policy": None,
        "return_slack": False,
        "return_cuts": False,
        "return_cut_duals": False,
        "activity": None,
        "run_info_level": RIL.DEFAULT,
    }
    default_failures: list[str] = []
    for params, expected in (
        (boot_params, expected_boot_defaults),
        (est_params, expected_est_defaults),
    ):
        for name, want in expected.items():
            got = params[name].default
            if got != want or type(got) is not type(want):
                default_failures.append(f"{name}: {got!r} != {want!r}")
    assert default_failures == [], default_failures

    # every optional param must appear in the expected-defaults dicts above
    boot_optional = {
        n for n, p in boot_params.items() if p.default is not inspect.Parameter.empty
    }
    est_optional = {
        n for n, p in est_params.items() if p.default is not inspect.Parameter.empty
    }
    assert boot_optional == set(expected_boot_defaults), (
        boot_optional ^ set(expected_boot_defaults)
    )
    assert est_optional == set(expected_est_defaults), (
        est_optional ^ set(expected_est_defaults)
    )

    # single-process defaults; annotation checks do not catch default drift
    sp_est_params = inspect.signature(combrum.estimate).parameters
    sp_boot_params = inspect.signature(combrum.bootstrap).parameters

    expected_estimate_defaults = {
        "transport": None,
        "master_backend": "auto",
        "master_params": None,
        "tolerance": 1e-06,
        "max_iterations": 1000,
        "min_iterations": 0,
        "qp_weight": 0.0,
        "qp_iterations": 0,
        "penalty_ref": "static",
        "schedule": None,
        "iteration_callback": None,
        "weights": None,
        "warm_start": None,
        "warm_cuts": None,
        "cut_policy": None,
        "return_slack": False,
        "return_cuts": False,
        "return_cut_duals": False,
        "activity": None,
        "run_info_level": RIL.DEFAULT,
    }
    expected_bootstrap_defaults = {
        "transport": None,
        "master_backend": "auto",
        "master_params": None,
        "tolerance": 1e-06,
        "max_iterations": 1000,
        "min_iterations": 0,
        "warm_start": None,
        "warm_cuts": None,
        "dual_store_dir": None,
        "activity": None,
    }
    sp_default_failures: list[str] = []
    for params, expected in (
        (sp_est_params, expected_estimate_defaults),
        (sp_boot_params, expected_bootstrap_defaults),
    ):
        for name, want in expected.items():
            got = params[name].default
            # bool is an int subclass (1 == True), so compare types too:
            # False vs 1 and 1e-6 vs 1 must both fail
            if got != want or type(got) is not type(want):
                sp_default_failures.append(f"{name}: {got!r} != {want!r}")
    assert sp_default_failures == [], sp_default_failures

    # all optional params live after the ``*`` -- keyword-only
    sp_kind_failures: list[str] = []
    for params, expected in (
        (sp_est_params, expected_estimate_defaults),
        (sp_boot_params, expected_bootstrap_defaults),
    ):
        for name in expected:
            if params[name].kind is not inspect.Parameter.KEYWORD_ONLY:
                sp_kind_failures.append(f"{name}: {params[name].kind.name}")
    assert sp_kind_failures == [], sp_kind_failures

    # same exhaustiveness check for estimate()/bootstrap()
    sp_est_optional = {
        n for n, p in sp_est_params.items() if p.default is not inspect.Parameter.empty
    }
    sp_boot_optional = {
        n for n, p in sp_boot_params.items() if p.default is not inspect.Parameter.empty
    }
    assert sp_est_optional == set(expected_estimate_defaults), (
        sp_est_optional ^ set(expected_estimate_defaults)
    )
    assert sp_boot_optional == set(expected_bootstrap_defaults), (
        sp_boot_optional ^ set(expected_bootstrap_defaults)
    )


def test_public_type_hints_resolve() -> None:
    failures: list[str] = []
    for name in combrum.__all__:
        obj = getattr(combrum, name)
        targets = []
        if inspect.isclass(obj):
            targets.extend([(f"{name}", obj), (f"{name}.__init__", obj.__init__)])
        elif inspect.isfunction(obj):
            targets.append((name, obj))
        for label, target in targets:
            try:
                typing.get_type_hints(target)
            except Exception as exc:  # pragma: no cover - failures collected, asserted below
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
    for label, target in (
        ("RepricingSchedule.select", combrum.RepricingSchedule.select),
        ("ResolveAll.select", combrum.ResolveAll.select),
        ("RoundRobin.select", combrum.RoundRobin.select),
        ("DualInformed.select", combrum.DualInformed.select),
    ):
        try:
            typing.get_type_hints(target)
        except Exception as exc:  # pragma: no cover - assertion formats all failures
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    assert failures == []

    # get_type_hints returns {} for a target with no annotations at all, so
    # also check the resolved hints of a few anchors
    import numpy as np

    estimate_hints = typing.get_type_hints(combrum.estimate)
    assert estimate_hints["return"] is combrum.FitResult
    assert estimate_hints["model"] is combrum.Model
    assert estimate_hints["data"] is combrum.Data

    estimate_distributed_hints = typing.get_type_hints(combrum.estimate_distributed)
    assert estimate_distributed_hints["return"] is combrum.FitResult
    assert estimate_distributed_hints["n_observations"] is int
    assert estimate_distributed_hints["n_simulations"] is int
    # estimate_distributed drops the single-process ``data`` parameter
    assert "data" not in estimate_distributed_hints

    assert typing.get_type_hints(combrum.bootstrap)["return"] is combrum.BootstrapResult
    assert (
        typing.get_type_hints(combrum.bootstrap_distributed)["return"]
        is combrum.BootstrapResult
    )

    # both timeout-callback factories take a TimeoutSchedule and return the
    # engine's per-iteration hook shape
    from collections.abc import Callable

    hook_return = Callable[[int, combrum.Oracle], int | None]
    for callback in (combrum.point_timeout_callback, combrum.bootstrap_timeout_callback):
        callback_hints = typing.get_type_hints(callback)
        assert callback_hints["schedule"] is combrum.TimeoutSchedule
        assert callback_hints["return"] == hook_return

    # every concrete RepricingSchedule.select keeps the ABC contract:
    # (iteration: int, n_agents: int, dual, last_resolved) -> np.ndarray
    for select in (
        combrum.RepricingSchedule.select,
        combrum.ResolveAll.select,
        combrum.RoundRobin.select,
        combrum.DualInformed.select,
    ):
        select_hints = typing.get_type_hints(select)
        assert select_hints["iteration"] is int
        assert select_hints["n_agents"] is int
        assert select_hints["return"] is np.ndarray
        assert select_hints["dual"] == (object | None)
        assert select_hints["last_resolved"] == (np.ndarray | None)

    # expected hints for every annotated constructor param on the public classes
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    # internal types named in public constructor annotations
    from combrum.activity import ActivityLevel, ActivityRun
    from combrum.dual import DualSolution
    from combrum.formulation import Formulation
    from combrum.runinfo import RunMetadata
    from combrum.transport.base import CutRow

    block_spec = tuple[float, float, int]
    features_hint = combrum.FeatureMap | Callable[..., Any]
    callback_hint = Callable[[int, combrum.Oracle], int | None]

    expected_ctor_hints: dict[str, dict[str, object]] = {
        "ActivityConfig": {
            "label": str,
            "run_id": (str | None),
            "level": (ActivityLevel | str),
            "stdout": bool,
        },
        "BootstrapResult": {
            "thetas": np.ndarray,
            "converged": np.ndarray,
            "parameters": combrum.Parameters,
            "point_estimate": (combrum.FitResult | None),
            "slack_samples": (np.ndarray | None),
            "duals": (tuple[DualSolution, ...] | None),
            "metadata": dict[str, object],
            "iterations": (int | None),
            "dual_store_dir": (Path | None),
            "n_duals_stored": int,
            "run_info": (RunMetadata | None),
        },
        "CutRow": {
            "rep_id": int,
            "agent_id": int,
            "phi": np.ndarray,
            "epsilon": float,
            "bundle_key": bytes,
        },
        "Data": {
            "observed_bundles": np.ndarray,
            "shocks": np.ndarray,
            "observables": Sequence[Any],
        },
        "Demand": {"bundle": np.ndarray, "payoff": float, "gap": float},
        "DemandBatch": {
            "ids": np.ndarray,
            "bundles": np.ndarray,
            "payoffs": np.ndarray,
            "gaps": np.ndarray,
        },
        "DualInformed": {
            "concentration_threshold": float,
            "max_staleness": int,
        },
        "FitResult": {
            "theta_hat": np.ndarray,
            "objective": float,
            "empirical_moment": np.ndarray,
            "runtime_seconds": float,
            "n_active_cuts": int,
            "parameters": combrum.Parameters,
            "slack": (np.ndarray | None),
            "metadata": dict[str, object],
            "run_info": (RunMetadata | None),
            "cuts": (Sequence[CutRow] | None),
            "cut_duals": (DualSolution | None),
        },
        "LoopConfig": {
            "max_iterations": int,
            "schedule": (combrum.RepricingSchedule | None),
            "qp_weight": float,
            "qp_iterations": int,
            "penalty_ref": str,
            "min_iterations": int,
            "iteration_callback": (callback_hint | None),
            "activity": (ActivityRun | None),
        },
        "LocalCluster": {
            "size": int,
            "ranks_per_node": (int | None),
            "rendezvous_timeout": float,
        },
        "Model": {
            "oracle": combrum.Oracle,
            "parameters": combrum.Parameters,
            "features": (features_hint | None),
            "observed_features": (object | None),
            "formulation": type[Formulation],
        },
        "MpiTransport": {"comm": (Any | None), "scatter_chunk_bytes": int},
        "NSlack": {"features": features_hint},
        "ExponentialDraws": {"n_observations": int, "base_seed": int},
        "OneSlack": {"features": features_hint},
        "Parameters": {
            "blocks": (
                Mapping[str, block_spec] | Iterable[tuple[str, block_spec]]
            )
        },
        "PersistentMasterFit": {
            "parameters": Any,
            "observables": Sequence[Any],
            "observed_bundles": np.ndarray,
            "transport": combrum.Transport,
            "config": combrum.LoopConfig,
            "rhs_transform": Callable[[CutRow, Any], float],
            "geometry_signature": (Callable[[Any], Any] | None),
            "master_backend": str,
            "master_params": (dict[str, object] | None),
            "tolerance": float,
            "weights": (np.ndarray | None),
        },
        "Phase": {"timeout": float, "iters": (int | None), "retire": bool},
        "PurgeInactive": {"max_age": int},
        "ReplayedWeights": {"matrix": np.ndarray},
        "RoundRobin": {"chunks": int},
        "TimeoutSchedule": {"phases": Sequence[combrum.Phase]},
        "SlackStrip": {"percentile": float, "max_live_cuts": float},
        "SolverSettings": {
            "time_limit_seconds": (float | None),
            "mip_focus": (int | None),
        },
    }

    # a dropped annotation removes the key; a retyped one changes the value
    hint_failures: list[str] = []
    for cls_name, params in expected_ctor_hints.items():
        resolved = typing.get_type_hints(getattr(combrum, cls_name).__init__)
        for param, expected in params.items():
            if param not in resolved:
                hint_failures.append(f"{cls_name}.{param}: annotation dropped")
            elif resolved[param] != expected:
                hint_failures.append(
                    f"{cls_name}.{param}: {resolved[param]!r} != {expected!r}"
                )
    assert hint_failures == [], hint_failures

    # every annotated ctor param of every public class must appear in the
    # registry above (per-param, so a new class or new param cannot be missed)
    _no_annotated_ctor = {"Oracle", "AddAll", "Transport", "SerialTransport"}
    unpinned_ctor_params: list[str] = []
    for name in combrum.__all__:
        obj = getattr(combrum, name)
        if not inspect.isclass(obj):
            continue
        try:
            hints = typing.get_type_hints(obj.__init__)
        except Exception:
            continue
        annotated_params = {k for k in hints if k != "return"}
        pinned = set(expected_ctor_hints.get(name, {}))
        missing = annotated_params - pinned
        if missing and name not in _no_annotated_ctor:
            unpinned_ctor_params.extend(f"{name}.{p}" for p in sorted(missing))
        # a class listed as having no annotated ctor must actually have none
        if name in _no_annotated_ctor and annotated_params:
            unpinned_ctor_params.extend(
                f"{name}.{p}" for p in sorted(annotated_params)
            )
    assert unpinned_ctor_params == [], (
        "public constructor params with no hint pin: " f"{unpinned_ctor_params}"
    )

    # same treatment for the public functions: expected hints for every
    # annotated param
    from combrum.policies import CutPolicy

    expected_func_hints: dict[str, dict[str, object]] = {
        "estimate": {
            "model": combrum.Model,
            "data": combrum.Data,
            "transport": (combrum.Transport | None),
            "master_backend": str,
            "master_params": (dict[str, object] | None),
            "tolerance": float,
            "max_iterations": int,
            "min_iterations": int,
            "qp_weight": float,
            "qp_iterations": int,
            "penalty_ref": str,
            "schedule": (combrum.RepricingSchedule | None),
            "iteration_callback": (callback_hint | None),
            "weights": (np.ndarray | None),
            "warm_start": (combrum.FitResult | None),
            "warm_cuts": (Sequence[CutRow] | None),
            "cut_policy": (CutPolicy | None),
            "return_slack": bool,
            "return_cuts": bool,
            "return_cut_duals": bool,
            "activity": (combrum.ActivityConfig | None),
            "run_info_level": combrum.RunInfoLevel,
            "return": combrum.FitResult,
        },
        "estimate_distributed": {
            "model": combrum.Model,
            "n_observations": int,
            "n_simulations": int,
            "transport": combrum.Transport,
            "master_backend": str,
            "master_params": (dict[str, object] | None),
            "tolerance": float,
            "max_iterations": int,
            "min_iterations": int,
            "qp_weight": float,
            "qp_iterations": int,
            "penalty_ref": str,
            "iteration_callback": (callback_hint | None),
            "warm_start": (combrum.FitResult | None),
            "warm_cuts": (Sequence[CutRow] | None),
            "cut_policy": (CutPolicy | None),
            "return_slack": bool,
            "return_cuts": bool,
            "return_cut_duals": bool,
            "activity": (combrum.ActivityConfig | None),
            "run_info_level": combrum.RunInfoLevel,
            "return": combrum.FitResult,
        },
        "bootstrap": {
            "model": combrum.Model,
            "data": combrum.Data,
            "n_bootstrap": int,
            "weight_source": combrum.WeightSource,
            "transport": (combrum.Transport | None),
            "master_backend": str,
            "master_params": (dict[str, object] | None),
            "tolerance": float,
            "max_iterations": int,
            "min_iterations": int,
            "warm_start": (object | None),
            "warm_cuts": (Sequence[CutRow] | None),
            "dual_store_dir": (Path | str | None),
            "activity": (combrum.ActivityConfig | None),
            "return": combrum.BootstrapResult,
        },
        "bootstrap_distributed": {
            "model": combrum.Model,
            "n_observations": int,
            "n_simulations": int,
            "n_bootstrap": int,
            "base_seed": int,
            "transport": combrum.Transport,
            "max_live_reps": int,
            "master_backend": str,
            "master_params": (dict[str, object] | None),
            "tolerance": float,
            "max_iterations": int,
            "min_iterations": int,
            "iteration_callback": (callback_hint | None),
            "warm_start": (object | None),
            "warm_cuts": (Sequence[CutRow] | None),
            "cut_policy": (CutPolicy | None),
            "dual_store_dir": (Path | str | None),
            "activity": (combrum.ActivityConfig | None),
            "return": combrum.BootstrapResult,
        },
        "point_timeout_callback": {
            "schedule": combrum.TimeoutSchedule,
            "return": callback_hint,
        },
        "bootstrap_timeout_callback": {
            "schedule": combrum.TimeoutSchedule,
            "return": callback_hint,
        },
    }

    func_hint_failures: list[str] = []
    for fn_name, params in expected_func_hints.items():
        resolved = typing.get_type_hints(getattr(combrum, fn_name))
        for param, expected in params.items():
            if param not in resolved:
                func_hint_failures.append(f"{fn_name}.{param}: annotation dropped")
            elif resolved[param] != expected:
                func_hint_failures.append(
                    f"{fn_name}.{param}: {resolved[param]!r} != {expected!r}"
                )
    assert func_hint_failures == [], func_hint_failures

    # every annotated param (including ``return``) must appear in the registry
    unpinned_func_params: list[str] = []
    for name in combrum.__all__:
        obj = getattr(combrum, name)
        if not inspect.isfunction(obj):
            continue
        resolved = typing.get_type_hints(obj)
        pinned = set(expected_func_hints.get(name, {}))
        missing = set(resolved) - pinned
        if missing:
            unpinned_func_params.extend(f"{name}.{p}" for p in sorted(missing))
    assert unpinned_func_params == [], (
        "public function params with no hint pin: " f"{unpinned_func_params}"
    )
