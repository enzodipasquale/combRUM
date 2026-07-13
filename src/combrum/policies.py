"""Cut admission and retirement policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from combrum.transport.base import CutRow


@dataclass(frozen=True)
class CutPolicyProfile:
    """Signals a policy needs from the formulation hot path."""

    needs_admit_violations: bool = True
    retires_cuts: bool = True
    needs_purge_duals: bool = True
    needs_purge_slacks: bool = True


DEFAULT_CUT_POLICY_PROFILE = CutPolicyProfile()


def policy_profile(policy: object) -> CutPolicyProfile:
    """Return a policy's profile, or the full-signal default if absent."""

    profile = getattr(policy, "profile", None)
    if not isinstance(profile, CutPolicyProfile):
        return DEFAULT_CUT_POLICY_PROFILE
    return profile


class CutPolicy(ABC):
    """Pluggable cut admission and retirement."""

    profile = DEFAULT_CUT_POLICY_PROFILE

    @abstractmethod
    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        """Which violated candidate cuts enter the master this iteration.

        ``candidates`` are the violated rows from this iteration's pricing;
        the returned tuple is the subset to install. ``violations`` is a
        float64 array parallel to ``candidates``: ``violations[i]`` is the
        violation of ``candidates[i]`` (``phi @ theta + epsilon - u_a``,
        ``>= 0``) at the current master solution.
        """

    @abstractmethod
    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        """Which installed cuts to RETIRE this iteration.

        ``dual`` and ``slack`` are per-cut signals keyed by
        ``(agent_id, bundle_key)``. ``None`` means the caller cannot
        supply that signal this iteration; a policy must degrade to a
        signal-free decision rather than fail.
        """
