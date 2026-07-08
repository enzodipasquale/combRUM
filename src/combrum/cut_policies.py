"""Built-in cut policies for candidate admission and cut retirement.

The top-level package exposes :class:`AddAll`, :class:`PurgeInactive`, and
:class:`SlackStrip`: the identity policy, dual-staleness retirement, and
slack-based retirement. The other policies in this module are lower-level
composition/admission tools for callers that import from ``combrum.cut_policies``
directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from combrum.policies import CutPolicy, CutPolicyProfile, policy_profile
from combrum.transport.base import CutRow

# dual mass at or below this is solver noise, not support; must match the
# dual-payload support cutoff so "inactive" here and "outside the support"
# elsewhere name one set of cuts
_DUAL_ATOL = 1e-10


def _key(row: CutRow) -> tuple[int, bytes]:
    # Per-master cut identity used as the dedup key for all purge signal maps.
    return (row.agent_id, row.bundle_key)


# Sentinel distinguishing "no reading for this cut" from any float value.
_NO_READING = object()


def _parallel_violations(
    candidates: Sequence[CutRow], violations: np.ndarray
) -> tuple[tuple[CutRow, ...], np.ndarray]:
    rows = tuple(candidates)
    viol = np.asarray(violations, dtype=np.float64)
    if viol.shape != (len(rows),):
        raise ValueError(
            "violations must be a 1-D array parallel to candidates;"
            f" got shape {viol.shape} for {len(rows)} candidates"
        )
    return rows, viol


class AddAll(CutPolicy):
    """Admit every candidate, retire nothing: the identity policy.

    Equivalent to running with no policy; fills a policy slot explicitly
    without changing behaviour.
    """

    profile = CutPolicyProfile(
        needs_admit_violations=False,
        retires_cuts=False,
        needs_purge_duals=False,
        needs_purge_slacks=False,
    )

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return tuple(candidates)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return ()

    def __repr__(self) -> str:
        return "AddAll()"


class Compose(CutPolicy):
    """Sequential admission pipeline, independent retirement votes.

    ``admit`` pipes candidates through each admission stage in order
    (stage ``i + 1`` sees only what stage ``i`` passed), so admission
    order is meaningful.

    ``purge`` gives every retirement stage the SAME installed set and
    retires the union of their votes. Voting on the original set rather
    than on survivors keeps retirement order-free (union of independent
    votes is commutative). Returned rows keep installed order, each
    retired cut appearing once regardless of how many stages voted.
    """

    def __init__(
        self,
        admit_chain: Sequence[CutPolicy],
        purge_chain: Sequence[CutPolicy],
    ) -> None:
        for name, chain in (
            ("admit_chain", admit_chain),
            ("purge_chain", purge_chain),
        ):
            for stage in chain:
                if not isinstance(stage, CutPolicy):
                    raise ValueError(
                        f"{name} stages must be CutPolicy instances;"
                        f" got {type(stage).__name__}"
                    )
        self._admit_chain = tuple(admit_chain)
        self._purge_chain = tuple(purge_chain)
        self._admit_profiles = tuple(
            policy_profile(stage) for stage in self._admit_chain
        )
        self._purge_profiles = tuple(
            policy_profile(stage) for stage in self._purge_chain
        )
        self._profile = CutPolicyProfile(
            needs_admit_violations=any(
                p.needs_admit_violations for p in self._admit_profiles
            ),
            retires_cuts=any(p.retires_cuts for p in self._purge_profiles),
            needs_purge_duals=any(p.needs_purge_duals for p in self._purge_profiles),
            needs_purge_slacks=any(p.needs_purge_slacks for p in self._purge_profiles),
        )

    @property
    def profile(self) -> CutPolicyProfile:
        return self._profile

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        original = tuple(candidates)
        rows = original
        viol = (
            np.array(violations, dtype=np.float64)
            if self._profile.needs_admit_violations
            else None
        )
        # Violations are re-mapped by object identity, but only after a stage
        # has actually thinned the rows; until then ``viol`` stays parallel.
        viol_by_id: dict[int, float] | None = None
        aligned = True
        empty = np.empty(0, dtype=np.float64)
        for stage, profile in zip(self._admit_chain, self._admit_profiles):
            if profile.needs_admit_violations:
                if not aligned:
                    if viol_by_id is None:
                        viol_by_id = {
                            id(row): float(v) for row, v in zip(original, violations)
                        }
                    viol = np.array(
                        [viol_by_id[id(row)] for row in rows], dtype=np.float64
                    )
                    aligned = True
                stage_viol = viol
            else:
                stage_viol = empty
            out = tuple(stage.admit(rows, stage_viol, iteration))
            aligned = (
                aligned
                and len(out) == len(rows)
                and all(a is b for a, b in zip(out, rows))
            )
            rows = out
        return rows

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        retired_keys: set[tuple[int, bytes]] = set()
        for stage in self._purge_chain:
            retired_keys.update(
                _key(row) for row in stage.purge(installed, dual, slack, iteration)
            )
        if not retired_keys:
            return ()
        return tuple(row for row in installed if _key(row) in retired_keys)

    def validate_master_size(self, *, n_parameters: int, n_agents: int) -> None:
        for stage in self._admit_chain + self._purge_chain:
            validator = getattr(stage, "validate_master_size", None)
            if callable(validator):
                validator(n_parameters=n_parameters, n_agents=n_agents)

    def validate_lazy_master_size(
        self, *, n_parameters: int, installed_agents: int
    ) -> None:
        for stage in self._admit_chain + self._purge_chain:
            validator = getattr(stage, "validate_lazy_master_size", None)
            if callable(validator):
                validator(
                    n_parameters=n_parameters,
                    installed_agents=installed_agents,
                )

    def __repr__(self) -> str:
        return (
            f"Compose(admit_chain={list(self._admit_chain)!r},"
            f" purge_chain={list(self._purge_chain)!r})"
        )


class PurgeInactive(CutPolicy):
    """Retire cuts whose dual has stayed (near-)zero for ``max_age`` calls.

    One counter per installed cut counts CONSECUTIVE signalled purge calls
    whose dual reading was within :data:`_DUAL_ATOL` of zero. A nonzero
    reading resets the counter; a cut with no reading holds its counter
    unchanged. ``dual=None`` retires nothing and moves no counter, so a
    signal-free call cannot create a zero streak. Counters of cuts
    absent from ``installed`` are pruned each call, so a re-entering cut
    starts a fresh streak.
    """

    def __init__(self, max_age: int) -> None:
        if not isinstance(max_age, (int, np.integer)) or max_age < 1:
            raise ValueError(f"max_age must be an integer >= 1; got {max_age!r}")
        self._max_age = int(max_age)
        self._zero_streak: dict[tuple[int, bytes], int] = {}

    profile = CutPolicyProfile(
        needs_admit_violations=False,
        retires_cuts=True,
        needs_purge_duals=True,
        needs_purge_slacks=False,
    )

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return tuple(candidates)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        keys = [_key(row) for row in installed]
        live = set(keys)
        for key in [k for k in self._zero_streak if k not in live]:
            del self._zero_streak[key]
        if dual is None:
            return ()
        retired: list[CutRow] = []
        for row, key in zip(installed, keys):
            pi = dual.get(key)
            if pi is None:
                continue
            if abs(pi) <= _DUAL_ATOL:
                streak = self._zero_streak.get(key, 0) + 1
                self._zero_streak[key] = streak
                if streak >= self._max_age:
                    retired.append(row)
            else:
                self._zero_streak.pop(key, None)
        return tuple(retired)

    def __repr__(self) -> str:
        return f"PurgeInactive(max_age={self._max_age})"


class SlackStrip(CutPolicy):
    """Strip loose cuts by slack percentile, capped by a live-count limit.

    Slack here follows the convention larger = looser (``u - (phi . theta
    + eps) >= 0``). A cut is stripped iff its looseness is strictly above
    the ``percentile``-th percentile; ties at the cutoff are kept.

    ``hard_threshold`` is not a slack magnitude: it is a max-live-constraint
    cap. If the percentile leg would keep more than ``hard_threshold`` rows,
    keep only the ``hard_threshold`` most-binding rows (smallest looseness).

    The cutoff uses :func:`numpy.percentile` in its default
    (linear-interpolation) form.

    Only cuts present in ``slack`` are strippable and only they form the
    percentile population; a cut without a reading is neither evidence nor a
    candidate. ``slack=None`` retires nothing that call.
    """

    def __init__(
        self,
        percentile: float = 100.0,
        hard_threshold: float = math.inf,
    ) -> None:
        percentile = float(percentile)
        if not 0.0 < percentile <= 100.0 or math.isnan(percentile):
            raise ValueError(f"percentile must lie in (0, 100]; got {percentile!r}")
        hard_threshold = float(hard_threshold)
        if math.isinf(hard_threshold):
            max_live_cuts = math.inf
        else:
            if not hard_threshold.is_integer():
                raise ValueError(
                    "hard_threshold is a max-live constraint count and"
                    f" must be integer-valued or inf; got {hard_threshold!r}"
                )
            max_live_cuts = int(hard_threshold)
            if max_live_cuts < 1:
                raise ValueError(
                    "hard_threshold is a max-live constraint count and"
                    f" must be >= 1 or inf; got {hard_threshold!r}"
                )
        self._percentile = percentile
        self._hard_threshold = max_live_cuts

    profile = CutPolicyProfile(
        needs_admit_violations=False,
        retires_cuts=True,
        needs_purge_duals=False,
        needs_purge_slacks=True,
    )

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return tuple(candidates)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        if slack is None:
            return ()
        signalled = []
        for row in installed:
            value = slack.get(_key(row), _NO_READING)
            if value is not _NO_READING:
                signalled.append((row, value))
        if not signalled:
            return ()
        slacks = np.asarray([value for _, value in signalled])
        keep = slacks <= float(np.percentile(slacks, self._percentile))
        if not math.isinf(self._hard_threshold) and int(keep.sum()) > int(
            self._hard_threshold
        ):
            keep = np.zeros(slacks.size, dtype=bool)
            # Most-binding rows are the smallest looseness values; stable sort
            # makes equal-looseness ties deterministic in installed-row order.
            order = np.argsort(slacks, kind="stable")
            keep[order[: int(self._hard_threshold)]] = True
        return tuple(
            row for (row, _value), keep_row in zip(signalled, keep) if not keep_row
        )

    def validate_master_size(self, *, n_parameters: int, n_agents: int) -> None:
        minimum = int(n_parameters) + int(n_agents)
        cap = self._hard_threshold
        if math.isinf(cap) or int(cap) >= minimum:
            return
        raise ValueError(
            "SlackStrip hard_threshold is a max-live constraint count"
            " and must be at least the NSlack master variable count"
            " K + n_agents; got hard_threshold="
            f"{int(cap)}, K={int(n_parameters)},"
            f" n_agents={int(n_agents)}, K + n_agents={minimum}"
        )

    def validate_lazy_master_size(
        self, *, n_parameters: int, installed_agents: int
    ) -> None:
        minimum = int(n_parameters) + int(installed_agents)
        cap = self._hard_threshold
        if math.isinf(cap) or int(cap) >= minimum:
            return
        raise ValueError(
            "SlackStrip hard_threshold is a max-live constraint count"
            " and must be at least the current lazy NSlack master variable"
            " count K + installed_agents; got hard_threshold="
            f"{int(cap)}, K={int(n_parameters)},"
            f" installed_agents={int(installed_agents)},"
            f" K + installed_agents={minimum}"
        )

    def __repr__(self) -> str:
        return (
            f"SlackStrip(percentile={self._percentile},"
            f" hard_threshold={self._hard_threshold})"
        )


class MostViolated(CutPolicy):
    """Admit only the ``k`` (or ``fraction``) most-violated candidates.

    Candidates with positive violation are ranked by it and the largest
    kept: ``k`` absolute, or ``max(1, int(fraction * n_positive))`` for a
    fraction. Exactly one of ``k``/``fraction`` is set. Ties at the cutoff
    break toward the earlier candidate, and admitted rows are returned in
    input order, so the result is a deterministic function of the
    candidates. Retires nothing.
    """

    def __init__(self, k: int | None = None, fraction: float | None = None) -> None:
        if (k is None) == (fraction is None):
            raise ValueError("specify exactly one of k or fraction")
        if k is not None:
            if not isinstance(k, (int, np.integer)) or k < 1:
                raise ValueError(f"k must be an integer >= 1; got {k!r}")
            k = int(k)
        if fraction is not None:
            fraction = float(fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"fraction must lie in (0, 1]; got {fraction!r}")
        self._k = k
        self._fraction = fraction

    profile = CutPolicyProfile(
        needs_admit_violations=True,
        retires_cuts=False,
        needs_purge_duals=False,
        needs_purge_slacks=False,
    )

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        rows, viol = _parallel_violations(candidates, violations)
        positive = np.flatnonzero(viol > 0.0)
        if positive.size == 0:
            return ()
        n = (
            self._k
            if self._k is not None
            else max(1, int(self._fraction * int(positive.size)))
        )
        if n >= positive.size:
            # Nothing to thin; positive is already in input order.
            return tuple(rows[i] for i in positive)
        ranked = positive[np.argsort(-viol[positive], kind="stable")]
        keep = np.zeros(len(rows), dtype=bool)
        keep[ranked[:n]] = True
        return tuple(row for row, keep_row in zip(rows, keep) if keep_row)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return ()

    def __repr__(self) -> str:
        if self._k is not None:
            return f"MostViolated(k={self._k})"
        return f"MostViolated(fraction={self._fraction})"


class SlackThreshold(CutPolicy):
    """Admit only candidates whose violation exceeds ``epsilon``.

    ``epsilon`` is a violation magnitude (the quantity ``violations``
    carries), so the policy needs no theta or u. Retires nothing.
    """

    def __init__(self, epsilon: float) -> None:
        epsilon = float(epsilon)
        if not epsilon >= 0.0:
            raise ValueError(f"epsilon must be >= 0; got {epsilon!r}")
        self._epsilon = epsilon

    profile = CutPolicyProfile(
        needs_admit_violations=True,
        retires_cuts=False,
        needs_purge_duals=False,
        needs_purge_slacks=False,
    )

    def admit(
        self,
        candidates: Sequence[CutRow],
        violations: np.ndarray,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        rows, viol = _parallel_violations(candidates, violations)
        keep = viol > self._epsilon
        return tuple(row for row, keep_row in zip(rows, keep) if keep_row)

    def purge(
        self,
        installed: Sequence[CutRow],
        dual: Mapping[tuple[int, bytes], float] | None,
        slack: Mapping[tuple[int, bytes], float] | None,
        iteration: int,
    ) -> tuple[CutRow, ...]:
        return ()

    def __repr__(self) -> str:
        return f"SlackThreshold(epsilon={self._epsilon!r})"
