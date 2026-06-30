"""Hold one master across an outer psi search.

``psi`` is the outer-search parameter the same fit is re-evaluated over (for
example a peer-effects ``sigma`` swept on a grid). An outer search over an
opaque psi may re-evaluate the same fit many times.
:class:`PersistentMasterFit` keeps one live master: the cold fit at psi0 builds
and converges it, and each later psi rewrites the RHS of the carried cuts via
:meth:`MasterBackend.set_rhs` before warm-solving. Reuse is valid only when psi
changes the additive RHS terms while leaving the cut geometry, objective linear
term, weights, and theta bounds fixed.

The caller supplies the reuse contract through:

* ``rhs_transform(row, psi) -> float``: maps each live ``CutRow`` to its new RHS.
  A live row carries only the current RHS and an opaque ``bundle_key``, so a
  transform that depends on a baseline RHS must keep that baseline separately.
* ``geometry_signature(psi) -> bytes | tuple[np.ndarray, ...]``: optional
  fingerprint for priced-bundle geometry not covered by the observed
  ``c_theta`` guard. When omitted, the caller is asserting that this geometry is
  fixed across the search.
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
    """One psi evaluation's published answer for driving an outer search.

    ``objective`` is the row-generation master objective, on the same scale as
    :func:`~combrum.engine.estimate.estimate`, so a parity test can compare a
    persistent evaluation against a cold rebuild. ``dual`` is the root-rank
    NSlack dual payload for outer moments. ``n_active_cuts`` is monotone across
    a warm search (the carried superset only grows).
    """

    theta_hat: np.ndarray
    objective: float
    empirical_moment: np.ndarray
    dual: object | None
    converged: bool
    iterations: int
    n_active_cuts: int


class PersistentMasterFit:
    """Hold one master across an outer psi search.

    Construct it with the psi-invariant inputs plus ``rhs_transform`` and,
    optionally, ``geometry_signature``. Then :meth:`fit` once at psi0 (cold fit,
    builds and converges the live master and stashes the reuse signature), and
    :meth:`reevaluate` for each later psi. If omitted, ``features``
    defaults to the oracle and ``formulation`` defaults to ``NSlack(features)``.

    The driver owns the live master's lifecycle: :meth:`close` is idempotent
    and is called after any guard or solve exception, and on teardown or
    context-manager exit (``run_fit`` runs with ``suppress_close=True``).

    NSlack-only: OneSlack installs one aggregate cut whose RHS depends on the
    priced joint selection, so a per-cut ``set_rhs`` rewrite and per-cut guard is
    undefined for it.
    """

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
        """Hold the psi-invariant fit inputs and callbacks.

        ``rhs_transform`` is required. ``geometry_signature`` is optional; when
        omitted, the priced-bundle geometry check is skipped and the caller
        asserts that this geometry is psi-invariant. See the module docstring.
        """
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

        # None until fit() and after close().
        self._master: Any = None
        self._resolved_master_backend: str | None = None
        # psi0 reuse signature, byte-compared by every reevaluate's guard.
        self._c_theta0: np.ndarray | None = None
        self._agent_weights0: np.ndarray | None = None
        self._theta_bounds0: tuple[np.ndarray, np.ndarray] | None = None
        self._geometry0: Any = None

    @staticmethod
    def _require_nslack(formulation: Any) -> None:
        # Match by exact module + qualname, not isinstance (would add an
        # engine->formulations import edge) and not a bare name check.
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
        """Cold fit at psi0: build the live master and stash its signature.

        Builds a fresh master, runs the fit with ``suppress_close=True`` so the
        master survives, and stashes the psi0 reuse signature (``c_theta0``, the
        full ``agent_weights`` vector, the theta bounds, and optional
        ``geometry_signature(psi0)``) that the later reevaluate guard
        byte-compares against.
        """
        formulation, features = self._fit_surfaces(oracle, formulation, features)
        if self._resolved_master_backend is None:
            self._resolved_master_backend = resolve_master_backend(
                self._master_backend,
                require_quadratic=(
                    self._config.qp_weight > 0.0 and self._config.decay > 0
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
        # Own the master before the solve, so a solve exception leaves it
        # closeable by this driver rather than leaked.
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
            # run_fit was told not to close; close here before propagating.
            self.close()
            raise

        # Stash the psi0 reuse signature on the owner rank (the only rank that
        # holds the master). c_theta0 is the same reduced vector the builder
        # bakes into the master, so the guard compares apples to apples.
        if self._transport.rank == built.ctx.owner_rank:
            self._c_theta0 = np.array(built.c_theta, dtype=np.float64)
            self._agent_weights0 = np.array(
                built.ctx.agent_weights, dtype=np.float64
            )
            lower, upper = built.ctx.theta_bounds
            self._theta_bounds0 = (
                np.array(lower, dtype=np.float64),
                np.array(upper, dtype=np.float64),
            )
        # psi-derived, not master-bound: stash on every rank so the guard is
        # rank-uniform. None means the caller opts out of G2.
        self._geometry0 = (
            None
            if self._geometry_signature is None
            else self._geometry_signature(psi)
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
        """Reuse the live master at psi.

        Builds the context against the live master (recomputes
        ``c_theta``/``empirical_moment`` at psi but skips ``make_master``).
        The reuse guard checks G1 ``c_theta`` and the full ``agent_weights``
        vector, optional G2 ``geometry_signature(psi)``, and G3
        ``theta_bounds`` against the psi0 signature. The method then rewrites
        every carried cut's RHS via ``set_rhs`` over the full ``extract_cuts()``
        key set and warm-solves with ``suppress_close=True``.
        """
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
        if self._transport.size == 1:
            if self._transport.rank == 0 and self._master is None:
                raise RuntimeError(
                    "reevaluate() before a successful fit(): no live master"
                    "; call fit(psi0) first (or the master was closed)."
                )
            return
        with self._transport.collective():
            if self._transport.rank == 0 and self._master is None:
                raise RuntimeError(
                    "reevaluate() before a successful fit(): no live master"
                    "; call fit(psi0) first (or the master was closed)."
                )

    def _rewrite_live_rhs(self, psi: Any, built: Any) -> None:
        if self._transport.size == 1:
            if self._transport.rank == built.ctx.owner_rank:
                self._rewrite_live_rhs_owner(psi, built)
            return
        with self._transport.collective():
            if self._transport.rank == built.ctx.owner_rank:
                self._rewrite_live_rhs_owner(psi, built)

    def _rewrite_live_rhs_owner(self, psi: Any, built: Any) -> None:
        try:
            self._assert_reuse_valid(psi, built)
            # Map over the full extract_cuts() key set so no carried cut keeps
            # a stale psi0 RHS; set_rhs raises on an unknown key.
            cuts = self._master.extract_cuts()
            self._master.set_rhs(
                {
                    (row.agent_id, row.bundle_key): float(
                        self._rhs_transform(row, psi)
                    )
                    for row in cuts
                }
            )
        except BaseException:
            self.close()
            raise

    def _assert_reuse_valid(self, psi: Any, built: Any) -> None:
        # Byte-exact .tobytes() identity, not a tolerance compare: any drift
        # means the carried cuts no longer describe this psi.
        # G1: c_theta plus the full (N*S,) agent_weights vector. Full vector,
        # not a subset: the master's u_coef closure is frozen at psi0 and serves
        # psi0's weight to any newly-priced-agent cut.
        if (
            np.asarray(built.c_theta, dtype=np.float64).tobytes()
            != np.asarray(self._c_theta0, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G1: c_theta at psi differs"
                " from psi0; the objective linear term is not psi-invariant."
            )
        if (
            np.asarray(built.ctx.agent_weights, dtype=np.float64).tobytes()
            != np.asarray(self._agent_weights0, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G1: agent_weights at psi"
                " differs from psi0 over the full (N*S,) vector; the master's"
                " u_coef closure is frozen at psi0."
            )
        # G2 geometry: optional; covers the all-bundle phi that the observed-only
        # c_theta guard cannot reach. If omitted, the caller asserts that
        # priced-bundle geometry is psi-invariant.
        if self._geometry_signature is not None and not _signature_equal(
            self._geometry_signature(psi), self._geometry0
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G2: geometry_signature(psi)"
                " differs from psi0; the cut geometry is not psi-invariant."
            )
        # G3 bounds: the theta box.
        lower, upper = built.ctx.theta_bounds
        ref_lower, ref_upper = self._theta_bounds0
        if (
            np.asarray(lower, dtype=np.float64).tobytes()
            != np.asarray(ref_lower, dtype=np.float64).tobytes()
            or np.asarray(upper, dtype=np.float64).tobytes()
            != np.asarray(ref_upper, dtype=np.float64).tobytes()
        ):
            raise ValueError(
                "PersistentMasterFit reuse guard G3: theta_bounds at psi"
                " differ from psi0; the theta bounds are not psi-invariant."
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
        """Close the live master (idempotent).

        The driver owns the master's lifecycle since ``run_fit`` ran with
        ``suppress_close=True``. Also called automatically on any guard
        or solve exception and on context-manager exit. A second
        call (or a call before ``fit``) is a no-op.
        """
        master = self._master
        self._master = None
        if master is not None:
            master.close()

    def __enter__(self) -> PersistentMasterFit:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _signature_equal(a: Any, b: Any) -> bool:
    """Byte-exact equality for a geometry signature (bytes or array tuple)."""
    if isinstance(a, (bytes, bytearray)) or isinstance(b, (bytes, bytearray)):
        return bytes(a) == bytes(b)
    a_t = tuple(np.asarray(x) for x in a)
    b_t = tuple(np.asarray(x) for x in b)
    if len(a_t) != len(b_t):
        return False
    return all(
        x.shape == y.shape
        and x.dtype == y.dtype
        and x.tobytes() == y.tobytes()
        for x, y in zip(a_t, b_t)
    )
