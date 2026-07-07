"""Model and data containers for the public fit functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from combrum.formulation import Formulation
from combrum.formulations import NSlack
from combrum.interface_resolution import FeatureMap
from combrum.oracle import Oracle
from combrum.parameters import Parameters


@dataclass(frozen=True)
class Model:
    """Oracle, parameter layout, feature map, and formulation class.

    Args:
        oracle: Oracle instance.
        parameters: Parameter layout.
        features: Priced-row callable ``(agent_id, bundle) -> (phi, eps)`` or
            :class:`FeatureMap` instance; defaults to ``oracle`` when omitted.
        observed_features: Optional observed-row surface. Serial fits call it as
            ``(agent_id, bundle) -> phi``; when omitted, serial fits infer
            observed rows from ``features`` on the observed bundles.
            Distributed fits require this surface, or ``features`` itself, to
            provide ``observed_features_batch(observation_ids)``. If observed
            rows need setup before that batch call, the surface may also define
            ``setup_observed(transport, observation_ids)``. If priced feature
            rows need setup on the agent axis, ``features`` may define
            ``setup_pricing_agents(transport, agent_ids)``.
        formulation: Formulation class (not an instance), e.g. ``NSlack``.
    """

    oracle: Oracle
    parameters: Parameters
    features: FeatureMap | Callable[..., Any] | None = field(default=None, kw_only=True)
    observed_features: object | None = field(default=None, kw_only=True)
    formulation: type[Formulation] = field(default=NSlack, kw_only=True)

    def __post_init__(self) -> None:
        features = self.oracle if self.features is None else self.features
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class Data:
    """Observed choices, shocks, and per-row labels passed to the fit functions.

    ``observed_bundles`` is the ``(N, M)`` observed-choice array. ``shocks`` is
    ``(N, S, ...)`` with ``S`` simulation draws per observation. ``observables``
    sets ``N``: combRUM's core loops use only its length. Per-observation
    covariates should be captured by the feature/oracle objects or another
    user-owned surface.
    """

    observed_bundles: np.ndarray
    shocks: np.ndarray
    observables: Sequence[Any]
