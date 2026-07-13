"""Hold one master across an outer psi search.

Reuse is valid only when psi changes the additive RHS terms while leaving the
cut geometry, objective linear term, weights, and theta bounds fixed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from combrum.engine.context_builder import build_fit_context, resolve_master_backend
from combrum.engine.driver import LoopConfig, run_fit
from combrum.transport.base import CutRow, Transport


@dataclass(frozen=True)
class PersistentFitResult:
    """One psi evaluation's published answer for driving an outer search."""

    theta_hat: np.ndarray
    objective: float
    empirical_moment: np.ndarray
    dual: object | None
    converged: bool
    iterations: int
    n_active_cuts: int


class PersistentMasterFit:
    """Hold one master across an outer psi search."""

    def __init__(
        self,
        parameters: Any,
        *,
        observables: Sequence[Any],
        observed_bundles: np.ndarray,
        transport: Transport,
        config: LoopConfig,
        rhs_transform: Callable[[CutRow, Any], float],
        geometry_signature: Callable[[Any], Any] | None = None,
        master_backend: str = "auto",
        master_params: dict[str, object] | None = None,
        tolerance: float = 1e-6,
        weights: np.ndarray | None = None,
    ) -> None:
        self._parameters = parameters
        self._observables = observables
        self._observed_bundles = np.asarray(observed_bundles)
        self._transport = transport
        self._config = config
        self._rhs_transform = rhs_transform
        self._geometry_signature = geometry_signature
        self._master_backend = master_backend
        self._master_params = master_params
        self._tolerance = tolerance
        self._weights = weights

        self._master: Any = None
        self._resolved_master_backend: str | None = None
        self._c_theta0: np.ndarray | None = None
        self._agent_weights0: np.ndarray | None = None
        self._theta_bounds0: tuple[np.ndarray, np.ndarray] | None = None
        self._geometry0: Any = None

    @staticmethod
    def _require_nslack(formulation: Any) -> None:
        cls = type(formulation)
        if (
            cls.__module__ != "combrum.formulations.nslack"
            or cls.__qualname__ != "NSlack"
        ):
            raise TypeError(
                "PersistentMasterFit is NSlack-only: got"
                f" {cls.__module__}.{cls.__qualname__}. Only the real"
                " combrum.formulations.nslack.NSlack is admitted; a per-cut"
                " set_rhs RHS rewrite is undefined for OneSlack's aggregate cut,"
                " whose RHS depends on the priced joint selection."
            )

    def _fit_surfaces(
        self,
        oracle: Any,
        formulation: Any | None,
        features: object | None,
    ) -> tuple[Any, object]:
        active_features = oracle if features is None else features
        if formulation is None:
            from combrum.formulations.nslack import NSlack

            formulation = NSlack(active_features)
        self._require_nslack(formulation)
        return formulation, active_features

    def fit(
        self,
        psi: Any,
        *,
        oracle: Any,
        shocks: np.ndarray,
        formulation: Any | None = None,
        features: object | None = None,
        observed_features: Any = None,
    ) -> PersistentFitResult:
        """Cold fit at psi0: build the live master and stash its signature."""
        formulation, features = self._fit_surfaces(oracle, formulation, features)
        if self._resolved_master_backend is None:
            self._resolved_master_backend = resolve_master_backend(
                self._master_backend,
                require_quadratic=(
                    self._config.qp_weight > 0.0 and self._config.qp_iterations > 0
                ),
                transport=self._transport,
            )
        built = build_fit_context(
            self._parameters,
            observables=self._observables,
            observed_bundles=self._observed_bundles,
            shocks=np.asarray(shocks),
            formulation=formulation,
            features=features,
            observed_features=observed_features,
            transport=self._transport,
            master_backend=self._master_backend,
            resolved_master_backend=self._resolved_master_backend,
            master_params=self._master_params,
            tolerance=self._tolerance,
            weights=self._weights,
            master=None,
            result_publication="dual",
        )
        self._master = built.ctx.master_backend
        try:
            outcome = run_fit(
                built.ctx,
                oracle,
                formulation,
                self._config,
                suppress_close=True,
            )
        except BaseException:
            self.close()
            raise

        if self._transport.rank == built.ctx.owner_rank:
            self._c_theta0 = np.array(built.c_theta, dtype=np.float64)
            self._agent_weights0 = np.array(built.ctx.agent_weights, dtype=np.float64)
            lower, upper = built.ctx.theta_bounds
            self._theta_bounds0 = (
                np.array(lower, dtype=np.float64),
                np.array(upper, dtype=np.float64),
            )
        self._geometry0 = (
            None if self._geometry_signature is None else self._geometry_signature(psi)
        )

        return self._publish(built, outcome)

    def reevaluate(
        self,
        psi: Any,
        *,
        oracle: Any,
        shocks: np.ndarray,
        formulation: Any | None = None,
        features: object | None = None,
        observed_features: Any = None,
    ) -> PersistentFitResult:
        """Reuse the live master at psi."""
        self._require_live_master()
        formulation, features = self._fit_surfaces(oracle, formulation, features)
        built = build_fit_context(
            self._parameters,
            observables=self._observables,
            observed_bundles=self._observed_bundles,
            shocks=np.asarray(shocks),
            formulation=formulation,
            features=features,
            observed_features=observed_features,
            transport=self._transport,
            master_backend=self._master_backend,
            resolved_master_backend=self._resolved_master_backend,
            master_params=self._master_params,
            tolerance=self._tolerance,
            weights=self._weights,
            master=self._master,
            result_publication="dual",
        )

        self._rewrite_live_rhs(psi, built)

        try:
            outcome = run_fit(
                built.ctx,
                oracle,
                formulation,
                self._config,
                suppress_close=True,
            )
        except BaseException:
            self.close()
            raise

        return self._publish(built, outcome)

    def _require_live_master(self) -> None:
        message = (
            "reevaluate() called with no live master: either fit(psi0) has"
            " not run yet or the master was already closed."
        )
        if self._transport.size == 1:
            if self._master is None:
                raise RuntimeError(message)
            return
        with self._transport.collective():
            if self._transport.rank == 0 and self._master is None:
                raise RuntimeError(message)

    def _rewrite_live_rhs(self, psi: Any, built: Any) -> None:
        if self._transport.size == 1:
            self._rewrite_live_rhs_owner(psi, built)
            return
        with self._transport.collective():
            if self._transport.rank == built.ctx.owner_rank:
                self._rewrite_live_rhs_owner(psi, built)

    def _rewrite_live_rhs_owner(self, psi: Any, built: Any) -> None:
        try:
            self._assert_reuse_valid(psi, built)
            cuts = self._master.extract_cuts()
            self._master.set_rhs(
                {
                    (row.agent_id, row.bundle_key): float(self._rhs_transform(row, psi))
                    for row in cuts
                }
            )
        except BaseException:
            self.close()
            raise

    def _assert_reuse_valid(self, psi: Any, built: Any) -> None:
        if (
            np.asarray(built.c_theta, dtype=np.float64).tobytes()
            != np.asarray(self._c_theta0, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G1: c_theta at psi does not"
                " match psi0; master reuse needs a psi-invariant objective"
                " linear term."
            )
        if (
            np.asarray(built.ctx.agent_weights, dtype=np.float64).tobytes()
            != np.asarray(self._agent_weights0, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G1: agent_weights at psi and"
                " psi0 disagree over the full (N*S,) vector; the master's"
                " u_coef closure is frozen at psi0."
            )
        if self._geometry_signature is not None and not _signature_equal(
            self._geometry_signature(psi), self._geometry0
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G2: geometry_signature(psi)"
                " no longer matches psi0; the cut geometry is not"
                " psi-invariant."
            )
        lower, upper = built.ctx.theta_bounds
        ref_lower, ref_upper = self._theta_bounds0
        if (
            np.asarray(lower, dtype=np.float64).tobytes()
            != np.asarray(ref_lower, dtype=np.float64).tobytes()
            or np.asarray(upper, dtype=np.float64).tobytes()
            != np.asarray(ref_upper, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G3: theta_bounds moved"
                " between psi0 and psi; the theta box must stay fixed across"
                " the search."
            )

    def _publish(self, built: Any, outcome: Any) -> PersistentFitResult:
        result = outcome.result
        diagnostics = outcome.diagnostics
        return PersistentFitResult(
            theta_hat=result.theta_hat,
            objective=result.objective,
            empirical_moment=built.empirical_moment,
            dual=result.dual,
            converged=bool(diagnostics.converged),
            iterations=int(diagnostics.iterations),
            n_active_cuts=int(result.n_active_cuts),
        )

    def close(self) -> None:
        """Close the live master; idempotent, and a no-op before ``fit``."""
        master = self._master
        self._master = None
        if master is not None:
            master.close()

    def __enter__(self) -> PersistentMasterFit:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _signature_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (bytes, bytearray)) or isinstance(b, (bytes, bytearray)):
        return bytes(a) == bytes(b)
    a_t = tuple(np.asarray(x) for x in a)
    b_t = tuple(np.asarray(x) for x in b)
    if len(a_t) != len(b_t):
        return False
    return all(
        x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        for x, y in zip(a_t, b_t)
    )
