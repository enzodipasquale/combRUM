"""Result objects returned by point estimates and bootstraps."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from combrum.certification import Certification, certification_metadata
from combrum.dual import DualSolution
from combrum.parameters import Parameters
from combrum.runinfo import RunMetadata
from combrum.transport.base import CutRow


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _restore_readonly(obj: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = obj.__dict__.get(name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def _certification_from_metadata(value: object) -> Certification:
    if not isinstance(value, dict):
        raise ValueError(
            f"certification metadata must be a dict (got {type(value).__name__})"
        )
    unknown = bool(value.get("worst_gap_unknown", False))
    worst = np.inf if unknown else float(value.get("worst_gap", 0.0))
    return Certification(
        n_priced=int(value.get("n_priced", 0)),
        n_inexact=int(value.get("n_inexact", 0)),
        worst_gap=worst,
    )


def _merge_certifications(
    results: Sequence["BootstrapResult"],
) -> dict[str, object] | None:
    present = ["certification" in result.metadata for result in results]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "concat requires certification metadata on all shards or on none;"
            " got a mixture"
        )
    certifications = [
        _certification_from_metadata(result.metadata["certification"])
        for result in results
    ]
    n_priced = sum(cert.n_priced for cert in certifications)
    n_inexact = sum(cert.n_inexact for cert in certifications)
    worst_gap = max(cert.worst_gap for cert in certifications)
    return certification_metadata(
        Certification(
            n_priced=n_priced,
            n_inexact=n_inexact,
            worst_gap=worst_gap,
        )
    )


@dataclass(frozen=True)
class FitResult:
    """Outcome of one fit: estimated ``theta_hat`` with optional diagnostics.

    ``theta_hat`` and ``empirical_moment`` are length-``K`` vectors aligned with
    ``parameters``; ``empirical_moment`` is the observed-data moment. ``slack``
    is the per-agent slack vector, ``None`` unless the fit requested it.
    ``run_info``, ``cuts``, and ``cut_duals`` carry optional provenance;
    ``cuts`` can seed a warm start.
    """

    theta_hat: np.ndarray
    objective: float
    empirical_moment: np.ndarray
    runtime_seconds: float
    n_active_cuts: int
    parameters: Parameters
    slack: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    run_info: RunMetadata | None = None
    cuts: Sequence[CutRow] | None = None
    cut_duals: DualSolution | None = None

    def __post_init__(self) -> None:
        K = self.parameters.K
        theta_hat = np.asarray(self.theta_hat, dtype=np.float64)
        if theta_hat.shape != (K,):
            raise ValueError(
                f"expected theta_hat of shape (K,) = ({K},), got {theta_hat.shape}"
            )
        if np.any(~np.isfinite(theta_hat)):
            raise ValueError("theta_hat must be finite")
        object.__setattr__(self, "theta_hat", _readonly(theta_hat))

        empirical_moment = np.asarray(self.empirical_moment, dtype=np.float64)
        if empirical_moment.shape != (K,):
            raise ValueError(
                f"expected empirical_moment of shape (K,) = ({K},),"
                f" got {empirical_moment.shape}"
            )
        if np.any(~np.isfinite(empirical_moment)):
            raise ValueError("empirical_moment must be finite")
        object.__setattr__(self, "empirical_moment", _readonly(empirical_moment))

        if self.n_active_cuts < 0:
            raise ValueError(f"n_active_cuts must be >= 0; got {self.n_active_cuts}")

        if self.slack is not None:
            object.__setattr__(
                self, "slack", _readonly(np.asarray(self.slack, dtype=np.float64))
            )

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        _restore_readonly(self, ("theta_hat", "empirical_moment", "slack"))

    def theta_named(self) -> dict[str, np.ndarray]:
        """``theta_hat`` split into ``{block name: values}`` by the layout."""
        return self.parameters.unpack(self.theta_hat)

    def empirical_moment_named(self) -> dict[str, np.ndarray]:
        """``empirical_moment`` split into ``{block name: values}``."""
        return self.parameters.unpack(self.empirical_moment)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready estimate fields.

        Excludes parameters, run_info, cuts, and cut_duals.
        """
        return {
            "theta_hat": self.theta_hat.tolist(),
            "objective": float(self.objective),
            "empirical_moment": self.empirical_moment.tolist(),
            "runtime_seconds": float(self.runtime_seconds),
            "n_active_cuts": int(self.n_active_cuts),
            "slack": None if self.slack is None else self.slack.tolist(),
            "metadata": dict(self.metadata),
        }

    def slack_summary(self) -> dict[str, float]:
        """Summary stats of the per-agent slack vector.

        Returns ``total_slack``, ``mean_slack``, ``max_slack``, and
        ``n_binding`` (the count of agents with exactly zero slack). Raises
        ``ValueError`` if ``slack`` is unset or empty.
        """
        if self.slack is None:
            raise ValueError(
                "slack_summary requires the slack field;"
                " this FitResult was built with slack=None"
            )
        if self.slack.size == 0:
            raise ValueError("slack_summary requires a nonempty slack vector")
        return {
            "total_slack": float(self.slack.sum()),
            "mean_slack": float(self.slack.mean()),
            "max_slack": float(self.slack.max()),
            "n_binding": int(np.count_nonzero(self.slack == 0.0)),
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap estimates with optional per-replication payloads.

    ``thetas`` has one row per replication and ``converged`` flags which rows
    converged; ``mean()``, ``se()``, ``cov()``, and ``ci()`` summarize over the
    converged rows by default. ``point_estimate`` is the full-sample fit when
    requested, and ``slack_samples`` holds the per-replication agent slacks shaped
    ``(B, n_agents)``. ``duals`` carries one dual payload per replication, or is
    ``None`` when the duals were streamed to ``dual_store_dir`` instead (then
    ``n_duals_stored`` counts the files written). ``iterations`` is the
    row-generation count; ``run_info`` and ``metadata`` carry run provenance.
    When present, ``metadata["certification"]`` summarizes pricing exactness
    only; replication convergence is still reported by ``converged``.
    """

    thetas: np.ndarray
    converged: np.ndarray
    parameters: Parameters
    point_estimate: FitResult | None = None
    slack_samples: np.ndarray | None = None
    duals: tuple[DualSolution, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    iterations: int | None = None
    dual_store_dir: Path | None = None
    n_duals_stored: int = 0
    run_info: RunMetadata | None = None

    def __post_init__(self) -> None:
        K = self.parameters.K
        thetas = np.asarray(self.thetas, dtype=np.float64)
        if thetas.ndim != 2 or thetas.shape[1] != K:
            raise ValueError(
                f"expected thetas of shape (B, K) = (B, {K}), got {thetas.shape}"
            )
        B = thetas.shape[0]
        if B < 1:
            raise ValueError(
                f"B must be >= 1 replication (got thetas of shape {thetas.shape})"
            )
        if np.any(~np.isfinite(thetas)):
            raise ValueError("thetas must be finite")
        object.__setattr__(self, "thetas", _readonly(thetas))

        converged = np.asarray(self.converged, dtype=bool)
        if converged.shape != (B,):
            raise ValueError(
                f"expected converged of shape (B,) = ({B},), got {converged.shape}"
            )
        object.__setattr__(self, "converged", _readonly(converged))

        if self.slack_samples is not None:
            slack_samples = np.asarray(self.slack_samples)
            if slack_samples.ndim < 1 or slack_samples.shape[0] != B:
                raise ValueError(
                    f"expected slack_samples with leading dimension B = {B},"
                    f" got shape {slack_samples.shape}"
                )
            object.__setattr__(self, "slack_samples", _readonly(slack_samples))

        if self.duals is not None:
            duals = tuple(self.duals)
            if len(duals) != B:
                raise ValueError(
                    f"duals must hold one payload per replication (B = {B});"
                    f" got {len(duals)}"
                )
            object.__setattr__(self, "duals", duals)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        _restore_readonly(self, ("thetas", "converged", "slack_samples"))

    @property
    def n_converged(self) -> int:
        """Number of replications flagged converged."""
        return int(np.count_nonzero(self.converged))

    def _selected(self, only_converged: bool) -> np.ndarray:
        if not only_converged:
            return self.thetas
        if self.n_converged == 0:
            raise ValueError(
                "only_converged=True requires at least one converged"
                f" replication; got 0 of {self.converged.size}"
                " (cannot summarize an empty converged subset;"
                " use only_converged=False to summarize all replications)"
            )
        excluded = self.converged.size - self.n_converged
        if excluded:
            warnings.warn(
                f"summaries exclude {excluded} non-converged"
                f" replication(s) of {self.converged.size};"
                " pass only_converged=False to include every replication",
                UserWarning,
                stacklevel=3,
            )
        return self.thetas[self.converged]

    def mean(self, only_converged: bool = True) -> np.ndarray:
        """Length-``K`` mean of the bootstrap estimates.

        With ``only_converged=True`` (default) it averages the converged rows
        and warns when it drops any; pass ``False`` to use every replication.
        """
        return self._selected(only_converged).mean(axis=0)

    def _selected_for_ddof1(self, only_converged: bool, summary: str) -> np.ndarray:
        selected = self._selected(only_converged)
        if selected.shape[0] < 2:
            raise ValueError(
                f"{summary} requires at least two"
                f" {'converged ' if only_converged else ''}bootstrap replications;"
                f" got {selected.shape[0]}"
            )
        return selected

    def se(self, only_converged: bool = True) -> np.ndarray:
        """Length-``K`` bootstrap standard error (``ddof=1``).

        Requires at least two selected replications; see :meth:`mean` for the
        ``only_converged`` selection.
        """
        return self._selected_for_ddof1(only_converged, "se").std(axis=0, ddof=1)

    def cov(self, only_converged: bool = True) -> np.ndarray:
        """``(K, K)`` bootstrap covariance (``ddof=1``).

        Requires at least two selected replications; see :meth:`mean` for the
        ``only_converged`` selection.
        """
        selected = self._selected_for_ddof1(only_converged, "cov")
        return np.atleast_2d(np.cov(selected, rowvar=False, ddof=1))

    def ci(
        self, level: float = 0.95, only_converged: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Percentile confidence band ``(lo, hi)``, each length ``K``.

        ``level`` is the central mass (default 0.95, i.e. the 2.5/97.5
        percentiles); see :meth:`mean` for the ``only_converged`` selection.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1); got {level}")
        selected = self._selected(only_converged)
        tail = 100.0 * (1.0 - level) / 2.0
        lo = np.percentile(selected, tail, axis=0)
        hi = np.percentile(selected, 100.0 - tail, axis=0)
        return lo, hi

    def se_named(self, only_converged: bool = True) -> dict[str, np.ndarray]:
        """:meth:`se` split into ``{block name: values}`` by the layout."""
        return self.parameters.unpack(self.se(only_converged))

    def ci_named(
        self, level: float = 0.95, only_converged: bool = True
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """:meth:`ci` split into ``{block name: (lo, hi)}`` by the layout."""
        lo, hi = self.ci(level, only_converged=only_converged)
        lo_named = self.parameters.unpack(lo)
        hi_named = self.parameters.unpack(hi)
        return {
            name: (lo_named[name], hi_named[name]) for name in self.parameters.names
        }

    @classmethod
    def concat(cls, results: Sequence[BootstrapResult]) -> BootstrapResult:
        """Merge bootstrap shards along the replication axis.

        All shards must use the same parameter layout and the same optional
        payload shape. If shards carry a point estimate, every shard's
        ``theta_hat`` must match and the first shard's point estimate is kept.
        Metadata merges in sequence order, with later values winning on
        duplicate keys, except ``"certification"`` which aggregates pricing
        counts and worst gaps.
        """
        results = tuple(results)
        if not results:
            raise ValueError("concat requires at least one BootstrapResult")
        first = results[0]
        for other in results[1:]:
            if other.parameters != first.parameters:
                raise ValueError(
                    "concat requires identical parameter layouts;"
                    f" got {first.parameters!r} and {other.parameters!r}"
                )

        estimates = [r.point_estimate for r in results]
        if any(pe is not None for pe in estimates):
            if any(pe is None for pe in estimates):
                raise ValueError(
                    "concat requires point_estimate provenance on all"
                    " shards or on none; got a mixture"
                )
            anchor = estimates[0]
            for pe in estimates[1:]:
                if not np.array_equal(pe.theta_hat, anchor.theta_hat):
                    raise ValueError(
                        "concat requires identical point_estimate"
                        " provenance across shards; got differing theta_hat"
                    )

        has_slack = [r.slack_samples is not None for r in results]
        slack_samples = None
        if any(has_slack):
            if not all(has_slack):
                raise ValueError(
                    "concat requires slack_samples on all shards or on none;"
                    " got a mixture that cannot define a common"
                    " replication alignment"
                )
            trailing = {r.slack_samples.shape[1:] for r in results}
            if len(trailing) > 1:
                raise ValueError(
                    "concat requires matching slack_samples payload shapes;"
                    f" got trailing shapes {sorted(trailing)}"
                )
            slack_samples = np.concatenate([r.slack_samples for r in results], axis=0)

        has_duals = [r.duals is not None for r in results]
        duals = None
        if any(has_duals):
            if not all(has_duals):
                raise ValueError(
                    "concat requires duals on all shards or on none;"
                    " got a mixture that cannot define a common"
                    " replication alignment"
                )
            duals = tuple(payload for r in results for payload in r.duals)

        metadata: dict[str, object] = {}
        for r in results:
            metadata.update(
                {
                    key: value
                    for key, value in r.metadata.items()
                    if key != "certification"
                }
            )
        merged_certification = _merge_certifications(results)
        if merged_certification is not None:
            metadata["certification"] = merged_certification
        return cls(
            thetas=np.concatenate([r.thetas for r in results], axis=0),
            converged=np.concatenate([r.converged for r in results], axis=0),
            parameters=first.parameters,
            point_estimate=first.point_estimate,
            slack_samples=slack_samples,
            duals=duals,
            metadata=metadata,
        )
