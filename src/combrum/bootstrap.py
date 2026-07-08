"""Serial multiplier bootstrap over a ``Model`` and ``Data`` pair."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from combrum.activity import (
    ActivityConfig,
    BootstrapFinal,
    BootstrapRepFinal,
    BootstrapStart,
    _activity_details,
    _object_name,
    build_activity_run,
)
from combrum.dual import DualSolution
from combrum.dualstore import DualStoreWriter
from combrum.engine import (
    build_fit_context,
    master_environment,
    resolve_master_backend,
    run_fit,
)
from combrum.engine.agreement import reject_multirank_dense_transport
from combrum.engine.driver import LoopConfig
from combrum.engine.observed import observed_objective_cache
from combrum.model import Data, Model
from combrum.randomness import multiplier_weights
from combrum.result import BootstrapResult
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow, Transport


class WeightSource(Protocol):
    """Bootstrap weight source indexed by replication id."""

    def weights_for(self, rep_id: int) -> np.ndarray:
        """Return the length-``N`` multiplier row for one replication."""
        ...


@dataclass(frozen=True)
class NativeDraws:
    """Fresh multiplier-bootstrap weights.

    Each replication's weights are exponential(1) multipliers normalised so the
    row sums to ``n_obs``. ``base_seed`` makes the stream reproducible.

    Example::

        weights = NativeDraws(n_obs=N, base_seed=42)
        boot = bootstrap(model, data, n_bootstrap=500, weights=weights)
    """

    n_obs: int
    base_seed: int

    def __post_init__(self) -> None:
        n_obs = operator.index(self.n_obs)
        if n_obs < 1:
            raise ValueError(f"n_obs must be >= 1; got {self.n_obs!r}")
        base = operator.index(self.base_seed)
        if base < 0:
            raise ValueError(f"base_seed must be >= 0; got {self.base_seed!r}")
        object.__setattr__(self, "n_obs", n_obs)
        object.__setattr__(self, "base_seed", base)

    def weights_for(self, rep_id: int) -> np.ndarray:
        """Replication ``rep_id``'s ``(n_obs,)`` multiplier weights."""
        return multiplier_weights(self.n_obs, self.base_seed, rep_id)


def _restamp(dual: DualSolution, rep_id: int) -> DualSolution:
    """Re-key an already validated dual payload without copying arrays."""
    return dual.with_rep_id(rep_id)


def _weight_source_seed(weights: object) -> int | None:
    seed = getattr(weights, "base_seed", None)
    return None if seed is None else int(seed)


def _validate_n_bootstrap(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("n_bootstrap must be an integer >= 1; got bool")
    try:
        out = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(
            f"n_bootstrap must be an integer >= 1; got {type(value).__name__}"
        ) from exc
    if out < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {value!r}")
    return out


def bootstrap(
    model: Model,
    data: Data,
    *,
    n_bootstrap: int,
    weights: WeightSource,
    transport: Transport | None = None,
    master_backend: str = "auto",
    master_params: dict[str, object] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    min_iterations: int = 0,
    warm_start: object | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    dual_store_dir: Path | str | None = None,
    activity: ActivityConfig | None = None,
) -> BootstrapResult:
    """Run the serial multiplier bootstrap over a ``Model``/``Data`` pair.

    ``weights`` is any object with ``weights_for(rep_id) -> (N,)``:
    :class:`NativeDraws` for fresh multiplier weights, or
    :class:`~combrum.randomness.ReplayedWeights` to replay a stored matrix.
    ``transport`` defaults to the serial reference. Each replication reweights
    the criterion and runs one fit; by default that fit is cold.

    ``warm_start`` and ``warm_cuts`` mirror
    :func:`~combrum.bootstrap_distributed.bootstrap_distributed`: an object
    whose ``theta_hat`` seeds each replication's proximal anchor, and cut rows
    reinstalled onto each replication's fresh master before its first solve
    (typically the point estimate and its active set). Warm replications
    usually converge in far fewer iterations but walk a different cut path
    than cold ones; the defaults keep every replication cold.

    :class:`NativeDraws` and the distributed
    :func:`~combrum.bootstrap_distributed.bootstrap_distributed` use different
    RNG streams, so the same ``base_seed`` does not reproduce identical draws
    across the serial and distributed paths.

    When ``dual_store_dir`` is given, duals stream to that directory one at a
    time and ``BootstrapResult.duals`` stays ``None``. ``activity`` surfaces
    root-only progress; omitting it (default ``None``) builds no sinks.

    Returns a :class:`~combrum.result.BootstrapResult` with ``thetas`` of shape
    ``(n_bootstrap, K)`` and a convergence mask.

    Example::

        model = Model(
            MyOracle(app_data),
            params,
            features=features,
            observed_features=observed_features,
        )
        data  = Data(observed_bundles=Y, shocks=draws, observables=X)
        boot  = bootstrap(model, data, n_bootstrap=500,
                          weights=NativeDraws(n_obs=N, base_seed=7))
    """
    oracle = model.oracle
    parameters = model.parameters
    features = model.features
    observed_features = model.observed_features

    def formulation_factory():
        return model.formulation(model.features)

    observed_bundles = data.observed_bundles
    shocks = data.shocks
    observables = data.observables

    transport = transport if transport is not None else SerialTransport()
    reject_multirank_dense_transport("bootstrap", transport)
    n_bootstrap = _validate_n_bootstrap(n_bootstrap)
    store_per_rep = dual_store_dir is not None
    K = parameters.K
    n_obs = len(observables)
    observed_bundles = np.asarray(observed_bundles)
    shocks = np.asarray(shocks)
    if observed_bundles.ndim != 2 or observed_bundles.shape[0] != n_obs:
        raise ValueError(
            "observed_bundles must be 2-D (N, M) with N = len(observables) ="
            f" {n_obs}; got shape {observed_bundles.shape}"
        )
    if shocks.ndim < 2 or shocks.shape[0] != n_obs:
        raise ValueError(
            "shocks must have shape (N, S, ...) with N ="
            f" {n_obs}; got shape {shocks.shape}"
        )
    n_simulations = int(shocks.shape[1])
    n_agents = n_obs * n_simulations
    result_publication = "dual" if store_per_rep else "summary"
    config = LoopConfig(max_iterations=max_iterations, min_iterations=min_iterations)
    resolved_master_backend = resolve_master_backend(
        master_backend, transport=transport
    )
    local_ids = np.arange(transport.rank, n_agents, transport.size, dtype=np.int64)
    observed_cache = observed_objective_cache(
        K=K,
        N=n_obs,
        observed_bundles=observed_bundles,
        local_ids=local_ids,
        transport=transport,
        features=features,
        observed_features=observed_features,
    )

    writer = (
        DualStoreWriter(dual_store_dir)
        if store_per_rep and transport.rank == 0
        else None
    )
    stored = 0

    with build_activity_run(activity, is_root=transport.rank == 0) as log:
        run_t0 = perf_counter() if log.enabled else None
        log_details = log.enabled and _activity_details(log.config.level)
        log.emit(
            BootstrapStart(
                run_id=log.config.run_id,
                label=log.config.label,
                n_bootstrap=n_bootstrap,
                base_seed=_weight_source_seed(weights),
                resampling=type(weights).__name__,
                tolerance=tolerance,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
                n_obs=n_obs,
                n_simulations=n_simulations,
                n_parameters=K,
                n_agents=n_agents,
                master_backend=master_backend,
                formulation=_object_name(model.formulation),
                result_publication=result_publication,
                transport=type(transport).__name__,
                rank=transport.rank,
                world_size=transport.size,
                activity_level=log.config.level,
                dual_store_dir=(
                    str(dual_store_dir) if dual_store_dir is not None else None
                ),
            )
        )

        total_iterations = 0
        thetas = np.zeros((n_bootstrap, K), dtype=np.float64)
        converged = np.zeros(n_bootstrap, dtype=bool)
        # One shared solver environment for the whole run: each replication's
        # master is still built fresh, but the license checkout happens once.
        with master_environment(resolved_master_backend) as master_env:
            for b in range(n_bootstrap):
                rep_t0 = perf_counter() if log_details else None
                weights_b = np.asarray(weights.weights_for(b), dtype=np.float64)
                formulation = formulation_factory()
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
                    weights=weights_b,
                    warm_start=warm_start,
                    warm_cuts=warm_cuts,
                    result_publication=result_publication,
                    observed_cache=observed_cache,
                    master_env=master_env,
                )
                outcome = run_fit(built.ctx, oracle, formulation, config)
                thetas[b] = outcome.result.theta_hat
                converged[b] = outcome.diagnostics.converged
                total_iterations += int(outcome.diagnostics.iterations)

                if writer is not None:
                    dual = outcome.result.dual
                    if dual is not None:
                        restamped = _restamp(dual, b)
                        writer.write(restamped)
                        stored += 1
                        del restamped

                if log_details:
                    log.emit(
                        BootstrapRepFinal(
                            run_id=log.config.run_id,
                            label=log.config.label,
                            rep_id=b,
                            state=(
                                "computed"
                                if outcome.diagnostics.converged
                                else "nonconverged"
                            ),
                            converged=bool(outcome.diagnostics.converged),
                            iterations=int(outcome.diagnostics.iterations),
                            objective=float(outcome.result.objective),
                            active_cuts=int(outcome.result.n_active_cuts),
                            wall_seconds=(
                                perf_counter() - rep_t0 if rep_t0 is not None else None
                            ),
                        )
                    )

        log.emit(
            BootstrapFinal(
                run_id=log.config.run_id,
                label=log.config.label,
                n_requested=n_bootstrap,
                n_persisted=0,
                n_computed=n_bootstrap,
                n_nonconverged=n_bootstrap - int(np.count_nonzero(converged)),
                n_converged=int(np.count_nonzero(converged)),
                total_super_steps=total_iterations,
                wall_seconds=(perf_counter() - run_t0 if run_t0 is not None else None),
                n_duals_stored=int(stored),
            )
        )
    return BootstrapResult(
        thetas=thetas,
        converged=converged,
        parameters=parameters,
        duals=None,
        iterations=total_iterations,
        dual_store_dir=Path(dual_store_dir) if store_per_rep else None,
        n_duals_stored=int(stored),
    )
