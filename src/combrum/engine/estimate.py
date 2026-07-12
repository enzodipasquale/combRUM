"""Point estimation through the public ``Model`` and ``Data`` objects."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from combrum.activity import ActivityConfig, build_activity_run
from combrum.context import ResultPublication
from combrum.engine.agreement import (
    agree_public_bool,
    agree_public_choice,
    agree_public_float,
    agree_public_int,
    agree_public_optional_theta,
    reject_multirank_dense_transport,
    require_public_object_agreement,
)
from combrum.engine.certify import GapTally, certification_metadata
from combrum.engine.context_builder import (
    BuiltContext,
    build_fit_context,
    resolve_master_backend,
)
from combrum.engine.distributed_context import (
    build_distributed_fit_context,
    distributed_c_theta,
    prepare_distributed_observed,
)
from combrum.engine.driver import LoopConfig, _validate_loop_controls, run_fit
from combrum.formulations import NSlack
from combrum.model import Data, Model
from combrum.oracle import Oracle
from combrum.policies import CutPolicy
from combrum.result import FitResult
from combrum.runinfo import (
    RunInfoLevel,
    RunMetadata,
    collect_provenance,
    peak_rss_bytes,
)
from combrum.schedule import RepricingSchedule
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow, Transport

if TYPE_CHECKING:
    from combrum.parameters import Parameters
    from combrum.rowgen import RowGenStep


def _requested_publication(
    return_slack: bool, return_cuts: bool, return_cut_duals: bool
) -> ResultPublication:
    publication = ResultPublication.SUMMARY
    if return_slack:
        publication |= ResultPublication.SLACK
    if return_cuts:
        publication |= ResultPublication.ACTIVE_SET
    if return_cut_duals:
        publication |= ResultPublication.DUAL
    return publication


def _run_and_package(
    *,
    built: BuiltContext,
    oracle: Oracle,
    formulation: RowGenStep,
    parameters: Parameters,
    transport: Transport,
    activity: ActivityConfig | None,
    level: RunInfoLevel,
    master_backend: str,
    resolved_master_backend: str,
    max_iterations: int,
    min_iterations: int,
    qp_weight: float,
    decay: int,
    penalty_ref: str,
    iteration_callback: Callable[[int, Oracle], int | None] | None,
    schedule: RepricingSchedule | None = None,
) -> FitResult:
    """Shared fit-run tail of :func:`estimate` and :func:`estimate_distributed`.

    Every collective here (timing bcast, gap certify, FULL-level peak
    reduction) runs inside the activity scope, before its closing summary.
    """
    with build_activity_run(activity, is_root=transport.rank == 0) as activity_run:
        config = LoopConfig(
            max_iterations=max_iterations,
            schedule=schedule,
            qp_weight=qp_weight,
            decay=decay,
            penalty_ref=penalty_ref,
            min_iterations=min_iterations,
            iteration_callback=iteration_callback,
            activity=activity_run,
        )

        tally = GapTally()

        start = time.perf_counter()
        outcome = run_fit(
            built.ctx, oracle, formulation, config, demand_sink=tally.observe
        )
        local_wall = time.perf_counter() - start
        runtime_seconds = float(
            transport.bcast(local_wall if transport.rank == 0 else None, root=0)
        )

        certification = tally.certify(transport)

        _meta = level >= RunInfoLevel.META
        wall_max_seconds = rss_max_bytes = None
        if level >= RunInfoLevel.FULL:
            peaks = transport.batched_max(
                np.array([local_wall, float(peak_rss_bytes())], dtype=np.float64)
            )
            wall_max_seconds = float(peaks[0])
            rss_max_bytes = int(peaks[1])
        run_info = (
            RunMetadata(
                level=level,
                rank=transport.rank,
                size=transport.size,
                node=transport.node,
                runtime_seconds=runtime_seconds,
                diagnostics=outcome.diagnostics,
                certification=certification,
                provenance=(
                    collect_provenance(
                        master_backend,
                        resolved_backend=resolved_master_backend,
                    )
                    if _meta
                    else None
                ),
                peak_rss_bytes=(
                    peak_rss_bytes() if _meta and transport.rank == 0 else None
                ),
                wall_max_seconds=wall_max_seconds,
                rss_max_bytes=rss_max_bytes,
            )
            if level >= RunInfoLevel.DEFAULT
            else None
        )

    result = outcome.result
    metadata = {
        "certification": certification_metadata(certification),
        "converged": bool(outcome.diagnostics.converged),
        "iterations": int(outcome.diagnostics.iterations),
    }
    return FitResult(
        theta_hat=result.theta_hat,
        objective=result.objective,
        empirical_moment=built.empirical_moment,
        runtime_seconds=runtime_seconds,
        n_active_cuts=result.n_active_cuts,
        parameters=parameters,
        slack=result.slack,
        metadata=metadata,
        run_info=run_info,
        cuts=result.active_set,
        cut_duals=result.dual,
    )


def estimate(
    model: Model,
    data: Data,
    *,
    transport: Transport | None = None,
    master_backend: str = "auto",
    master_params: dict[str, object] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    min_iterations: int = 0,
    qp_weight: float = 0.0,
    decay: int = 0,
    penalty_ref: str = "static",
    schedule: RepricingSchedule | None = None,
    iteration_callback: Callable[[int, Oracle], int | None] | None = None,
    weights: np.ndarray | None = None,
    warm_start: FitResult | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    cut_policy: CutPolicy | None = None,
    return_slack: bool = False,
    return_cuts: bool = False,
    return_cut_duals: bool = False,
    activity: ActivityConfig | None = None,
    level: RunInfoLevel = RunInfoLevel.DEFAULT,
) -> FitResult:
    """Fit ``theta`` by row generation.

    Args:
        transport: Defaults to the serial reference.
        master_backend: ``"auto"``, ``"gurobi"``, or ``"highs"``.
        master_params: Backend-owned solver knobs.
        qp_weight, decay, penalty_ref: Proximal penalty (off by default):
            weight ``qp_weight`` for the first ``decay`` iterations, then
            exactly zero so the terminating solve is a pure LP.
        iteration_callback: Per-iteration hook; may update oracle-owned
            settings and return an additional convergence floor.
        cut_policy: Bounds an NSlack master by retiring non-binding cuts;
            ``None`` keeps every admitted cut.
        return_cut_duals: NSlack formulations only.
        activity: Root-only row-generation progress sink; off when omitted.

    Returns:
        A :class:`~combrum.result.FitResult`; ``metadata["certification"]``
        records pricing exactness.

    Example::

        model = Model(
            MyOracle(app_data),
            params,
            features=features,
            observed_features=observed_features,
        )
        data  = Data(observed_bundles=Y, shocks=draws, observables=X)
        fit   = estimate(model, data)
    """
    oracle = model.oracle
    parameters = model.parameters
    features = model.features
    observed_features = model.observed_features
    formulation = model.formulation(model.features)
    observed_bundles = data.observed_bundles
    shocks = data.shocks
    observables = data.observables
    transport = transport if transport is not None else SerialTransport()
    reject_multirank_dense_transport("estimate", transport)

    if return_cut_duals:
        cls = type(formulation)
        if (
            cls.__module__ != "combrum.formulations.nslack"
            or cls.__qualname__ != "NSlack"
        ):
            raise ValueError("return_cut_duals is only supported for NSlack")

    # Fail fast on loop controls before any context/master setup (pinned by
    # test_estimate_validates_loop_controls_before_context); LoopConfig also
    # re-validates at construction.
    _validate_loop_controls(
        max_iterations, qp_weight, decay, penalty_ref, min_iterations
    )
    resolved_master_backend = resolve_master_backend(
        master_backend,
        require_quadratic=qp_weight > 0.0 and decay > 0,
        transport=transport,
    )
    built = build_fit_context(
        parameters,
        observables=observables,
        observed_bundles=observed_bundles,
        shocks=shocks,
        formulation=formulation,
        features=features,
        observed_features=observed_features,
        transport=transport,
        master_backend=master_backend,
        resolved_master_backend=resolved_master_backend,
        master_params=master_params,
        tolerance=tolerance,
        schedule=schedule,
        weights=weights,
        warm_start=warm_start,
        warm_cuts=warm_cuts,
        cut_policy=cut_policy,
        result_publication=_requested_publication(
            return_slack, return_cuts, return_cut_duals
        ),
    )

    return _run_and_package(
        built=built,
        oracle=oracle,
        formulation=formulation,
        parameters=parameters,
        transport=transport,
        activity=activity,
        level=level,
        master_backend=master_backend,
        resolved_master_backend=resolved_master_backend,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        qp_weight=qp_weight,
        decay=decay,
        penalty_ref=penalty_ref,
        iteration_callback=iteration_callback,
        schedule=schedule,
    )


def estimate_distributed(
    model: Model,
    *,
    n_observations: int,
    n_simulations: int,
    transport: Transport,
    master_backend: str = "auto",
    master_params: dict[str, object] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    min_iterations: int = 0,
    qp_weight: float = 0.0,
    decay: int = 0,
    penalty_ref: str = "static",
    iteration_callback: Callable[[int, Oracle], int | None] | None = None,
    warm_start: FitResult | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    cut_policy: CutPolicy | None = None,
    return_slack: bool = False,
    return_cuts: bool = False,
    return_cut_duals: bool = False,
    activity: ActivityConfig | None = None,
    level: RunInfoLevel = RunInfoLevel.DEFAULT,
) -> FitResult:
    """Fit an NSlack model from rank-owned observed shards.

    This is the distributed entry point. ``n_observations`` is the
    observed row count ``N``; ``n_simulations`` is the number of simulated
    pricing agents per observation, so the global pricing ids are
    ``0, ..., N*S-1`` and agent ``gid`` belongs to observation ``gid % N``.

    No ``Data`` object is accepted here. The model must use ``NSlack`` (other
    formulations raise :class:`NotImplementedError`) and must expose observed
    features through ``observed_features_batch(observation_ids)`` on
    ``model.observed_features`` or ``model.features``. If observed rows need
    setup before that call, the same surface may also define
    ``setup_observed(transport, observation_ids)``. If priced feature rows need
    setup, ``model.features`` may define
    ``setup_pricing_agents(transport, agent_ids)``. Observed moments are
    reduced over observation-owned shards. Pricing work is sharded over the
    global agent axis, so a rank prices its contiguous global-agent ids; for
    ``S > 1``, their design rows are selected by ``gid % N``.

    The remaining keywords match :func:`estimate`, except that requested
    slack/cut/cut-dual artifacts are root-gathered: only rank 0's result
    carries them; other ranks read ``None``. ``metadata["certification"]``
    aggregates pricing exactness across ranks.
    """
    oracle = model.oracle
    parameters = model.parameters
    supported_formulation = agree_public_bool(
        "model.formulation is NSlack", model.formulation is NSlack, transport
    )
    if not supported_formulation:
        raise NotImplementedError(
            "estimate_distributed currently supports model.formulation=NSlack"
            " only; serial estimate remains available for other formulations"
        )
    formulation = model.formulation(model.features)
    master_backend = str(
        agree_public_choice(
            "master_backend",
            master_backend,
            transport,
            choices=("auto", "gurobi", "highs"),
        )
    )

    max_iterations = agree_public_int(
        "max_iterations", max_iterations, transport, lower=1
    )
    min_iterations = agree_public_int(
        "min_iterations", min_iterations, transport, lower=0
    )
    qp_weight = agree_public_float("qp_weight", qp_weight, transport, lower=0.0)
    decay = agree_public_int("decay", decay, transport, lower=0)
    penalty_ref = str(
        agree_public_choice(
            "penalty_ref",
            penalty_ref,
            transport,
            choices=("dynamic", "static"),
        )
    )
    tolerance = agree_public_float(
        "tolerance", tolerance, transport, lower=0.0, strict_lower=True
    )
    level = RunInfoLevel(agree_public_int("level", level, transport, lower=0))
    master_params = require_public_object_agreement(
        "master_params", master_params, transport
    )
    warm_cuts = require_public_object_agreement("warm_cuts", warm_cuts, transport)
    cut_policy = require_public_object_agreement("cut_policy", cut_policy, transport)
    return_slack = agree_public_bool("return_slack", return_slack, transport)
    return_cuts = agree_public_bool("return_cuts", return_cuts, transport)
    return_cut_duals = agree_public_bool(
        "return_cut_duals", return_cut_duals, transport
    )

    _validate_loop_controls(
        max_iterations, qp_weight, decay, penalty_ref, min_iterations
    )
    resolved_master_backend = resolve_master_backend(
        master_backend,
        require_quadratic=qp_weight > 0.0 and decay > 0,
        transport=transport,
        owner_ranks=(0,),
    )
    prep = prepare_distributed_observed(
        model,
        n_observations=n_observations,
        n_simulations=n_simulations,
        transport=transport,
    )
    theta_init = agree_public_optional_theta(
        "warm_start", warm_start, transport, K=prep.K
    )
    c_theta = distributed_c_theta(prep, transport=transport)
    built = build_distributed_fit_context(
        prep,
        model=model,
        formulation=formulation,
        c_theta=c_theta,
        slack_coef=lambda agent_id: 1.0,
        transport=transport,
        owner_rank=0,
        master_backend=resolved_master_backend,
        master_params=master_params,
        tolerance=tolerance,
        theta_init=theta_init,
        warm_cuts=warm_cuts,
        cut_policy=cut_policy,
        result_publication=_requested_publication(
            return_slack, return_cuts, return_cut_duals
        ),
    )

    return _run_and_package(
        built=built,
        oracle=oracle,
        formulation=formulation,
        parameters=parameters,
        transport=transport,
        activity=activity,
        level=level,
        master_backend=master_backend,
        resolved_master_backend=resolved_master_backend,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        qp_weight=qp_weight,
        decay=decay,
        penalty_ref=penalty_ref,
        iteration_callback=iteration_callback,
    )
