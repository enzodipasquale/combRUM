"""Model and data containers for the public fit functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from combrum.formulation import Formulation
from combrum.formulations import NSlack
from combrum.oracle import Oracle
from combrum.parameters import Parameters


@dataclass(frozen=True)
class Model:
    """Oracle, parameter layout, feature map, and formulation class.

    Args:
        oracle: Oracle instance.
        parameters: Parameter layout.
        features: Priced-row callable ``(agent_id, bundle) -> (phi, eps)``;
            defaults to ``oracle`` when omitted.
        observed_features: Phi-only callable for the observed-data linear
            term. When omitted, the serial engine infers the observed rows from
            ``features`` on the observed bundles. Distributed fits require this
            surface, or ``features`` itself, to provide
            ``observed_features_batch(observation_ids)``. If the observed rows
            need setup before that call, the surface may also define
            ``setup_observed(transport, observation_ids)``.
        formulation: Formulation class (not an instance), e.g. ``NSlack``.
    """

    oracle: Oracle
    parameters: Parameters
    features: Callable[..., Any] | None = field(default=None, kw_only=True)
    observed_features: Callable[..., Any] | None = field(default=None, kw_only=True)
    formulation: type[Formulation] = field(default=NSlack, kw_only=True)

    def __post_init__(self) -> None:
        features = self.oracle if self.features is None else self.features
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class Data:
    """Observed choices, shocks, and the row count passed to ``estimate`` and ``bootstrap``.

    ``observed_bundles`` is the ``(N, M)`` observed-choice array. ``shocks`` is
    ``(N, S, ...)`` with ``S`` simulation draws per observation. ``observables``
    sets ``N``: only its length is read, so any length-``N`` sequence works
    (e.g. ``np.arange(N)``). Per-observation covariates belong in the feature
    callables, not here.
    """

    observed_bundles: np.ndarray
    shocks: np.ndarray
    observables: Sequence[Any]
