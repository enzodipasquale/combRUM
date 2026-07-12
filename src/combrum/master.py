"""The backend-agnostic master-problem contract.

:class:`MasterBackend` decouples the row-generation engine from the
optimizer hosting the relaxation, so the optimizer binding is swappable
without touching estimation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from combrum.transport.base import CutRow


@dataclass(frozen=True)
class CutReadings:
    """Solver readings over the rows present in the last solved relaxation.

    ``keys`` is the row domain, always ``(agent_id, bundle_key)``; ``dual``
    and ``slack`` are optional arrays parallel to it. ``slack`` convention:
    binding is near zero and looser rows are larger (backends normalize their
    solver's row-slack sign before populating it).
    """

    keys: tuple[tuple[int, bytes], ...]
    dual: np.ndarray | None = None
    slack: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name, values in (("dual", self.dual), ("slack", self.slack)):
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != (len(self.keys),):
                raise ValueError(
                    f"expected {name} readings of shape ({len(self.keys)},),"
                    f" got {arr.shape}"
                )
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    def dual_map(self) -> dict[tuple[int, bytes], float]:
        if self.dual is None:
            raise ValueError("CutReadings carries no dual readings")
        return {key: float(value) for key, value in zip(self.keys, self.dual)}

    def slack_map(self) -> dict[tuple[int, bytes], float]:
        if self.slack is None:
            raise ValueError("CutReadings carries no slack readings")
        return {key: float(value) for key, value in zip(self.keys, self.slack)}


class MasterBackend(ABC):
    """One replication's master problem behind a backend-neutral interface.

    A backend without native quadratic support must raise on
    ``set_penalty`` with ``weight > 0`` rather than approximate the penalty:
    a solve against an approximated objective reports duals of the wrong
    problem.

    One backend instance hosts one replication, so cut identity is rep-less:
    the dedup and canonical-order key is ``(agent_id, bundle_key)`` (``rep_id``
    is constant across a backend's rows and could order nothing).
    """

    @abstractmethod
    def add_cuts(self, rows: Sequence[CutRow]) -> int:
        """Install cut rows; return how many were NEW.

        New means not already installed under the ``(agent_id, bundle_key)``
        key, deduped against the installed set and within the batch. Duplicate
        input is contract-permitted (ranks may deliver equal-key rows) and must
        be absorbed, never rejected.
        """

    @abstractmethod
    def solve(self) -> None:
        """Re-optimize the current relaxation.

        The master is feasible and bounded by construction (finite theta box,
        slacks bounded below), so a non-optimal terminating status is solver
        distress, not a reachable state; backends should raise on it.

        The accessors below (:meth:`theta`, :meth:`objective`,
        :meth:`u_values`, :meth:`dual_values`, :meth:`bound_duals`) report
        the state of the LAST solve.
        """

    @abstractmethod
    def theta(self) -> np.ndarray:
        """Current theta_hat of the master, shape ``(K,)``."""

    @abstractmethod
    def objective(self) -> float:
        """Objective value of the last solve."""

    @abstractmethod
    def u_values(self) -> dict[int, float]:
        """Current epigraph/slack values keyed by agent id.

        Must not be reconstructed from installed row algebra: that duplicates
        solver state and can silently drift from the actual variable value
        under backend-specific bounds.
        """

    @abstractmethod
    def dual_values(self) -> dict[tuple[int, bytes], float]:
        """Dual value per installed cut, keyed by ``(agent_id, bundle_key)``.

        Valid LP duals only when the last solve was a pure LP; see
        :meth:`set_penalty` for why the terminating solve must be one.
        """

    def solved_cut_keys(self) -> frozenset[tuple[int, bytes]]:
        """Cut keys with per-row readings in the last solved relaxation.

        Slack-only retirement policies need this domain to avoid treating
        just-admitted rows as priced by the last solve. The default derives it
        from duals; backends may override with a cheaper solved-row snapshot.
        """
        return frozenset(self.dual_values())

    def cut_readings(self, *, dual: bool = False, slack: bool = False) -> CutReadings:
        """Read solver-owned row signals for the last solved relaxation.

        Per-cut signals keyed by ``(agent_id, bundle_key)``, sourced from the
        solver rather than recomputed from copied row algebra. The default
        serves dual-only policies via :meth:`dual_values`; slack raises unless
        a backend supplies a solver-native implementation.
        """
        if slack:
            raise NotImplementedError(
                "MasterBackend.cut_readings(slack=True) is not overridden;"
                " this backend exposes no solver-native row slack"
            )
        duals = self.dual_values() if dual else {}
        keys = tuple(sorted(duals if dual else self.solved_cut_keys()))
        dual_arr = (
            np.asarray([duals[key] for key in keys], dtype=np.float64) if dual else None
        )
        return CutReadings(keys=keys, dual=dual_arr)

    @abstractmethod
    def set_penalty(self, ref: np.ndarray, weight: float) -> None:
        """Layer ``weight * ||theta - ref||^2`` onto the master objective.

        ``weight <= 0`` restores the master to a pure LP: any installed penalty
        term is removed entirely, not zeroed approximately. The terminating solve
        must be a pure LP for its duals to be valid LP duals; a residual
        quadratic term would report multipliers for a different problem.
        """

    @abstractmethod
    def extract_cuts(self) -> tuple[CutRow, ...]:
        """The installed cut rows in canonical ``(agent_id, bundle_key)`` order.

        Half of the warm-start/checkpoint primitive: extract on one master,
        :meth:`reinstall` on a fresh one, and the two hold the same relaxation.
        The canonical order makes the extracted tuple a deterministic artifact,
        safe to hash, persist, and diff.
        """

    @abstractmethod
    def reinstall(self, rows: Sequence[CutRow]) -> None:
        """Rebuild the installed set from extracted rows.

        The other half of the warm-start/checkpoint primitive in
        :meth:`extract_cuts`. This is the seed path, not the retirement path;
        a cut policy sheds rows through :meth:`remove_cuts` (in place), never
        this rebuild.
        """

    def remove_cuts(self, keys: Iterable[tuple[int, bytes]]) -> int:
        """Retire the named installed cuts, returning the count removed.

        The retirement path a cut policy drives. The default rebuilds the
        relaxation from the kept rows via :meth:`reinstall`; correct, but
        O(all rows) and it drops the warm basis. A backend that can delete rows
        should override this with warm-basis-preserving in-place removal. The
        kept installed set is identical either way; with pre-declared columns
        and a warm start the in-place re-solve lands the same vertex, so the
        published estimate is unchanged.
        """
        keyset = set(keys)
        if not keyset:
            return 0
        rows = self.extract_cuts()
        kept = [r for r in rows if (r.agent_id, r.bundle_key) not in keyset]
        self.reinstall(kept)
        return len(rows) - len(kept)

    def set_rhs(self, updates: Mapping[tuple[int, bytes], float]) -> None:
        """Rewrite the RHS (epsilon) of already-installed cuts in place.

        The cut is ``u - phi.theta >= epsilon``, so epsilon is the constraint
        RHS; ``updates`` maps each ``(agent_id, bundle_key)`` to its new epsilon
        and only the RHS moves: phi, the slack columns, and every other row are
        left exactly as installed. Lets a persistent master be reused across an
        outer search by overwriting each cut's RHS rather than rebuilding the
        relaxation.
        """
        raise NotImplementedError(
            "MasterBackend.set_rhs is not overridden;"
            " this backend has no in-place cut-RHS rewrite"
        )

    @property
    def n_active_cuts(self) -> int:
        """The number of cuts installed behind the current relaxation.

        The live master size, which a cut-retirement policy (e.g.
        :class:`~combrum.cut_policies.SlackStrip`) bounds even as the cumulative
        admitted count grows. The default is O(n_cuts); a backend that tracks
        its installed set should override it with an O(1) read, since the
        heartbeat reads it on the hot path.
        """
        return len(self.extract_cuts())

    @abstractmethod
    def bound_duals(self) -> dict[int, float]:
        """Bound reduced-costs of the theta box, keyed by coordinate index.

        Contains exactly the coordinates at a bound in the last solve; an empty
        dict when theta_hat is interior. The persisted dual snapshot is
        incomplete without these whenever theta_hat sits on a box bound.
        """
