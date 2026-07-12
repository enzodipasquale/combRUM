"""Observed-bundle feature materialization and objective reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from combrum.interface_resolution import (
    Mode,
    feature_batch_aggregate,
    resolve_features,
    supports_feature_batch_aggregate,
)
from combrum.transport.base import Transport


@dataclass(frozen=True)
class ObservedObjectiveCache:
    """Weight-independent observed rows for repeated objective builds."""

    phi_local: np.ndarray
    empirical_moment: np.ndarray


def _checked_phi_matrix(Phi: object, n_rows: int, K: int) -> np.ndarray:
    phi = np.asarray(Phi, dtype=np.float64)
    expected = (n_rows, K)
    if phi.shape != expected:
        raise ValueError(
            f"expected observed feature rows of shape {expected}, got {phi.shape}"
        )
    return phi


def _checked_eps_vector(Eps: object, n_rows: int) -> np.ndarray:
    eps = np.asarray(Eps, dtype=np.float64)
    expected = (n_rows,)
    if eps.shape != expected:
        raise ValueError(
            f"expected Eps of shape {expected} from features_batch, got {eps.shape}"
        )
    return eps


def _checked_observed_objective(
    aggregate: object,
    K: int,
    N: int,
    theta_coef: np.ndarray,
    observed_bundles: np.ndarray,
    local_ids: np.ndarray,
    transport: Transport,
) -> tuple[np.ndarray, np.ndarray]:
    c_theta, empirical_moment = aggregate(
        K,
        N,
        theta_coef,
        observed_bundles,
        local_ids,
        transport,
    )
    c_theta = np.asarray(c_theta, dtype=np.float64)
    empirical_moment = np.asarray(empirical_moment, dtype=np.float64)
    if c_theta.shape != (K,):
        raise ValueError(
            f"expected c_theta of shape ({K},) from observed_objective,"
            f" got {c_theta.shape}"
        )
    if empirical_moment.shape != (K,):
        raise ValueError(
            f"expected empirical_moment of shape ({K},) from observed_objective,"
            f" got {empirical_moment.shape}"
        )
    return c_theta, empirical_moment


def observed_phi_rows(
    *,
    K: int,
    observed_bundles: np.ndarray,
    local_ids: np.ndarray,
    features: object,
    observed_features: object | None,
) -> np.ndarray:
    """Materialize this shard's observed-bundle ``Phi`` rows.

    ``observed_features`` is an explicit phi-only surface. When it is omitted,
    observed rows are inferred from the priced feature map by evaluating
    ``features`` / ``features_batch`` at ``observed_bundles[a % N]`` and
    discarding the priced-error term. Resolution is comm-free and uses the
    active batched member directly when available; the priced row-generation
    surface remains the conformance backstop when both feature paths are present.
    """
    ids = np.asarray(local_ids, dtype=np.int64)
    n_rows = int(ids.size)
    if n_rows == 0:
        return np.empty((0, K), dtype=np.float64)

    observed_bundles = np.asarray(observed_bundles)
    bundles = observed_bundles[ids % observed_bundles.shape[0]]
    if observed_features is not None:
        rows = [
            np.asarray(observed_features(int(a), b), dtype=np.float64)
            for a, b in zip(ids, bundles)
        ]
        return _checked_phi_matrix(np.stack(rows, axis=0), n_rows, K)

    resolution = resolve_features(features)
    if resolution.runs_optimized:
        Phi, Eps = resolution.active(ids, bundles)
        _checked_eps_vector(Eps, n_rows)
        return _checked_phi_matrix(Phi, n_rows, K)

    rows = [
        np.asarray(phi, dtype=np.float64)
        for phi, _eps in (resolution.active(int(a), b) for a, b in zip(ids, bundles))
    ]
    return _checked_phi_matrix(np.stack(rows, axis=0), n_rows, K)


def _objective_from_rows(
    *,
    K: int,
    theta_coef: np.ndarray,
    local_ids: np.ndarray,
    transport: Transport,
    phi_local: np.ndarray,
    empirical_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = _checked_phi_matrix(phi_local, local_ids.size, K)
    local_c_rows = -theta_coef[local_ids, None] * phi
    c_theta = np.asarray(
        transport.sum_reproducible(local_c_rows, local_ids), dtype=np.float64
    )
    return c_theta, empirical_moment


def _empirical_moment_from_rows(
    *,
    N: int,
    local_ids: np.ndarray,
    transport: Transport,
    phi_local: np.ndarray,
) -> np.ndarray:
    obs_mask = local_ids < N
    obs_phi_sum = np.asarray(
        transport.sum_reproducible(phi_local[obs_mask], local_ids[obs_mask]),
        dtype=np.float64,
    )
    return obs_phi_sum / float(N)


def observed_objective_cache(
    *,
    K: int,
    N: int,
    observed_bundles: np.ndarray,
    local_ids: np.ndarray,
    transport: Transport,
    features: object,
    observed_features: object | None,
) -> ObservedObjectiveCache | None:
    """Build a reusable observed-row cache when it preserves the old path.

    Hooks and serial aggregate feature maps remain uncached so their existing
    per-replication contracts and reduction order stay unchanged.
    """
    if observed_features is not None:
        if callable(getattr(observed_features, "observed_objective", None)):
            return None
    else:
        if callable(getattr(features, "observed_objective", None)):
            return None
        resolution = resolve_features(features)
        if (
            transport.size == 1
            and resolution.mode is Mode.OPTIMIZED
            and supports_feature_batch_aggregate(resolution.active)
        ):
            return None

    phi_local = observed_phi_rows(
        K=K,
        observed_bundles=observed_bundles,
        local_ids=local_ids,
        features=features,
        observed_features=observed_features,
    )
    empirical_moment = _empirical_moment_from_rows(
        N=N,
        local_ids=local_ids,
        transport=transport,
        phi_local=phi_local,
    )
    phi_cache = np.array(phi_local, dtype=np.float64, copy=True)
    phi_cache.setflags(write=False)
    empirical_moment.setflags(write=False)
    return ObservedObjectiveCache(
        phi_local=phi_cache,
        empirical_moment=empirical_moment,
    )


def observed_objective(
    *,
    K: int,
    N: int,
    theta_coef: np.ndarray,
    observed_bundles: np.ndarray,
    local_ids: np.ndarray,
    transport: Transport,
    features: object,
    observed_features: object | None,
    cache: ObservedObjectiveCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced observed objective vector and empirical moment."""
    local_ids = np.asarray(local_ids, dtype=np.int64)
    theta_coef = np.asarray(theta_coef, dtype=np.float64)
    if cache is not None:
        return _objective_from_rows(
            K=K,
            theta_coef=theta_coef,
            local_ids=local_ids,
            transport=transport,
            phi_local=cache.phi_local,
            empirical_moment=cache.empirical_moment,
        )

    if observed_features is not None:
        aggregate = getattr(observed_features, "observed_objective", None)
        if callable(aggregate):
            return _checked_observed_objective(
                aggregate,
                K,
                N,
                theta_coef,
                observed_bundles,
                local_ids,
                transport,
            )
    else:
        aggregate = getattr(features, "observed_objective", None)
        if callable(aggregate):
            return _checked_observed_objective(
                aggregate,
                K,
                N,
                theta_coef,
                observed_bundles,
                local_ids,
                transport,
            )
        resolution = resolve_features(features)
        if transport.size == 1 and resolution.mode is Mode.OPTIMIZED:
            observed = np.asarray(observed_bundles)
            bundles = observed[local_ids % N]
            aggregated = feature_batch_aggregate(
                resolution.active,
                local_ids,
                bundles,
                theta_coef[local_ids],
                K,
            )
            if aggregated is not None:
                local_c = -aggregated[0]
                c_theta = np.asarray(
                    transport.sum_reproducible(
                        local_c[None, :],
                        np.asarray([0], dtype=np.int64),
                    ),
                    dtype=np.float64,
                )

                obs_mask = local_ids < N
                empirical_local = np.zeros(K, dtype=np.float64)
                if np.any(obs_mask):
                    obs_ids = local_ids[obs_mask]
                    empirical_agg = feature_batch_aggregate(
                        resolution.active,
                        obs_ids,
                        observed[obs_ids],
                        np.ones(obs_ids.size, dtype=np.float64),
                        K,
                    )
                    empirical_local = empirical_agg[0]
                empirical_sum = np.asarray(
                    transport.sum_reproducible(
                        empirical_local[None, :],
                        np.asarray([0], dtype=np.int64),
                    ),
                    dtype=np.float64,
                )
                return c_theta, empirical_sum / float(N)

    Phi = observed_phi_rows(
        K=K,
        observed_bundles=observed_bundles,
        local_ids=local_ids,
        features=features,
        observed_features=observed_features,
    )
    local_c_rows = -theta_coef[local_ids, None] * Phi
    c_theta = np.asarray(
        transport.sum_reproducible(local_c_rows, local_ids), dtype=np.float64
    )
    empirical_moment = _empirical_moment_from_rows(
        N=N,
        local_ids=local_ids,
        transport=transport,
        phi_local=Phi,
    )
    return c_theta, empirical_moment
