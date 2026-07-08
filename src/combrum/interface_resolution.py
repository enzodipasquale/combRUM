"""Resolve a per-agent / batched method pair to the member a provider overrode.

Method pairs like ``price``/``price_batch`` and ``features``/``features_batch``
both raise by default; a provider overrides at least one. We resolve the pair
once into a :class:`Resolution` the caller dispatches on. Override detection
walks the whole MRO and binds the class method, so an instance-level
monkeypatch is ignored. When both members are overridden the batched one runs
and :func:`assert_conforms` checks it against the per-agent member; under MPI
the resolution must agree across ranks (see :func:`check_agreement`).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from combrum.transport.base import Transport

# Max drift allowed between the batched and per-agent outputs on continuous
# fields; discrete fields are compared byte-exact and never use this.
CONTINUOUS_TOL: float = 1e-13


def _continuous_drift(opt: np.ndarray, ref: np.ndarray) -> float:
    if not opt.size:
        return 0.0
    same_inf = np.isinf(opt) & np.isinf(ref) & (np.signbit(opt) == np.signbit(ref))
    diff = np.zeros_like(opt, dtype=np.float64)
    np.subtract(opt, ref, out=diff, where=~same_inf)
    return float(np.max(np.abs(diff)))


class Mode(Enum):
    """Which member of a method pair a provider resolved to.

    Dispatched exhaustively by :func:`dispatch`.
    """

    DEFAULT = "default"  # only the per-agent member is overridden
    OPTIMIZED = "optimized"  # only the batched member is overridden
    BOTH = "both"  # both overridden; the batched path runs, gated by assert_conforms


@dataclass(frozen=True)
class Resolution:
    """The once-resolved choice for one method pair.

    ``surface`` is the pair name ("price", "features"); ``active`` is the bound
    class function the caller invokes (instance monkeypatch cannot redirect
    it); ``reference`` is the bound per-agent member, present only in
    :attr:`Mode.BOTH` as the conformance reference and divergence fallback,
    else ``None``.
    """

    surface: str
    mode: Mode
    active: Callable[..., Any]
    reference: Callable[..., Any] | None
    # provider class identity, folded into the rank-agreement token
    _module: str
    _qualname: str

    @property
    def runs_optimized(self) -> bool:
        """True iff the active member is the batched/optimized one."""
        return self.mode in (Mode.OPTIMIZED, Mode.BOTH)

    @property
    def token(self) -> tuple[str, str, str, str]:
        """Rank-agreement fingerprint ``(surface, module, qualname, mode)``.

        Identical on every rank by construction; fold into an existing
        setup broadcast and call :func:`check_agreement`.
        """
        return (self.surface, self._module, self._qualname, self.mode.value)


def needs_conformance_guard(*resolutions: object) -> bool:
    """Whether any resolved method pair can run a strict BOTH check."""
    return any(
        getattr(resolution, "mode", None) is Mode.BOTH for resolution in resolutions
    )


def _is_overridden(instance: object, name: str, base_default: Any) -> bool:
    # getattr over the whole MRO (not __dict__, which misses intermediate-base
    # overrides), comparing unbound function objects (bound methods never
    # compare identical).
    reached = getattr(type(instance), name, None)
    return reached is not None and reached is not base_default


def resolve_local(
    instance: object,
    *,
    surface: str,
    default_name: str,
    optimized_name: str,
    default_func: Any,
    optimized_func: Any,
) -> Resolution:
    """Resolve one surface once, without any cross-rank round.

    ``default_name`` / ``optimized_name`` are the member names on
    ``instance``'s class; ``default_func`` / ``optimized_func`` are the base
    default function objects override detection compares against. The active
    member is returned bound via its class function (instance monkeypatch
    ignored). Comm-free; the caller folds :attr:`Resolution.token` into its
    own setup broadcast (or uses :func:`resolve`, which adds the round).

    Raises:
        TypeError: if neither member is overridden.
    """
    has_default = _is_overridden(instance, default_name, default_func)
    has_optimized = _is_overridden(instance, optimized_name, optimized_func)

    if not has_default and not has_optimized:
        raise TypeError(
            f"{type(instance).__qualname__} must override"
            f" {default_name!r} or {optimized_name!r} for {surface!r}"
        )

    cls = type(instance)
    bound_default = _bind(cls, default_name, instance)
    bound_optimized = _bind(cls, optimized_name, instance)

    if has_default and has_optimized:
        mode = Mode.BOTH
        active: Callable[..., Any] = bound_optimized
        reference: Callable[..., Any] | None = bound_default
    elif has_optimized:
        mode = Mode.OPTIMIZED
        active = bound_optimized
        reference = None
    else:
        mode = Mode.DEFAULT
        active = bound_default
        reference = None

    return Resolution(
        surface=surface,
        mode=mode,
        active=active,
        reference=reference,
        _module=cls.__module__,
        _qualname=cls.__qualname__,
    )


def resolve(
    instance: object,
    *,
    surface: str,
    default_name: str,
    optimized_name: str,
    default_func: Any,
    optimized_func: Any,
    transport: Transport,
) -> Resolution:
    """:func:`resolve_local` plus a standalone cross-rank agreement round.

    For a caller that owns no setup-time broadcast to piggyback on. Raises
    from inside ``transport.collective()`` if ranks resolve differently.
    """
    resolution = resolve_local(
        instance,
        surface=surface,
        default_name=default_name,
        optimized_name=optimized_name,
        default_func=default_func,
        optimized_func=optimized_func,
    )
    local = resolution.token
    with transport.collective():
        check_agreement(local, transport.bcast(local, root=0))
    return resolution


def check_agreement(
    local: tuple[str, str, str, str], root: tuple[str, str, str, str]
) -> None:
    """Raise unless this rank's resolution token equals root's.

    Must run inside ``transport.collective()`` so a disagreeing rank fails
    as one agreed verdict. ``root`` is ``local`` broadcast from rank 0; a
    mismatch is a divergent build (different class wired per rank).
    """
    if local != root:
        raise ValueError(
            "method-pair resolution disagrees across ranks for"
            f" {local[0]!r}: this rank resolved {local!r}, root resolved"
            f" {root!r}; a divergent build wired a different class or"
            " override set per rank"
        )


def _bind(cls: type, name: str, instance: object) -> Callable[..., Any]:
    # __get__ off the class, not the instance, so an instance-level attribute
    # of the same name cannot shadow the bound class method.
    func = getattr(cls, name)
    return func.__get__(instance, cls)


def dispatch(
    resolution: Resolution,
    *,
    on_default: Callable[[Callable[..., Any]], Any],
    on_optimized: Callable[[Callable[..., Any]], Any],
    on_both: Callable[[Callable[..., Any], Callable[..., Any]], Any],
) -> Any:
    """Exhaustive dispatch over a resolved mode.

    ``on_both`` receives ``(active_optimized, reference_default)``; the
    single-member callbacks receive just the active member. An unhandled
    mode raises.
    """
    mode = resolution.mode
    if mode is Mode.DEFAULT:
        return on_default(resolution.active)
    if mode is Mode.OPTIMIZED:
        return on_optimized(resolution.active)
    if mode is Mode.BOTH:
        assert resolution.reference is not None  # Mode.BOTH always carries it
        return on_both(resolution.active, resolution.reference)
    raise AssertionError(f"unhandled resolution mode: {mode!r}")


def assert_conforms(
    surface: str,
    *,
    optimized: Sequence[tuple[np.ndarray, ...]],
    reference: Sequence[tuple[np.ndarray, ...]],
    discrete: Sequence[int],
    continuous: Sequence[int],
    tol: float = CONTINUOUS_TOL,
) -> None:
    """Hard-fail unless the optimized output matches the per-agent output.

    Each element of ``optimized`` / ``reference`` is one item's tuple of
    fields. ``discrete`` indexes fields compared byte-identical;
    ``continuous`` indexes fields compared within ``tol``.

    Raises:
        AssertionError: on any mismatch.
    """
    if len(optimized) != len(reference):
        raise AssertionError(
            f"method-pair conformance for {surface!r}: optimized produced"
            f" {len(optimized)} items, per-agent produced {len(reference)}"
        )
    for pos, (opt_fields, ref_fields) in enumerate(zip(optimized, reference)):
        for idx in discrete:
            opt = np.asarray(opt_fields[idx])
            ref = np.asarray(ref_fields[idx])
            if (
                opt.shape != ref.shape
                or opt.dtype != ref.dtype
                or not (opt.tobytes() == ref.tobytes())
            ):
                raise AssertionError(
                    f"method-pair conformance for {surface!r}: discrete field"
                    f" {idx} of item {pos} differs between the optimized and"
                    " per-agent paths (byte-exact equality required)"
                )
        for idx in continuous:
            opt = np.asarray(opt_fields[idx], dtype=np.float64)
            ref = np.asarray(ref_fields[idx], dtype=np.float64)
            if opt.shape != ref.shape:
                raise AssertionError(
                    f"method-pair conformance for {surface!r}: continuous"
                    f" field {idx} of item {pos} has shape {opt.shape} on the"
                    f" optimized path, {ref.shape} on the per-agent path"
                )
            drift = _continuous_drift(opt, ref)
            if not drift <= tol:
                raise AssertionError(
                    f"method-pair conformance for {surface!r}: continuous"
                    f" field {idx} of item {pos} drifts by {drift!r} between"
                    f" the optimized and per-agent paths (> tol {tol!r})"
                )


class FeatureMap:
    """Base class for the features surface.

    Subclass and override at least one of :meth:`features` /
    :meth:`features_batch` (both raise by default); :func:`resolve_features`
    picks the active member. A bare ``(agent_id, bundle) -> (phi, eps)``
    callable may be passed instead of an instance.

    ``features_batch`` is the preferred surface for large designs: it maps
    chosen bundles to rows in one array-oriented call. This is separate from
    :meth:`combrum.oracle.Oracle.price_batch`, which solves the demand problem.

    When both are overridden, ``features_batch`` must match per-agent
    ``features`` applied row-by-row within the conformance contract; the
    per-agent member is the divergence fallback.
    """

    def features(self, agent_id: int, bundle: np.ndarray) -> tuple[np.ndarray, float]:
        """Per-agent feature row ``(phi (K,), eps)``; default raises."""
        raise NotImplementedError(
            "FeatureMap.features is not overridden; override features or features_batch"
        )

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched feature rows ``(Phi (n, K), Eps (n,))``; default raises.

        ``ids`` are global agent ids; ``bundles`` are the matching chosen
        bundles in ``ids`` order; the return is in that same order.

        Implementations may also accept explicit keyword parameters
        ``weights`` and ``aggregate``. With ``aggregate=True`` they return the
        weighted aggregate ``(phi (K,), eps)`` directly, avoiding dense row
        materialization in serial aggregate fast paths. ``K`` may also be
        accepted, but is not required when the feature map already knows its
        own dimension.
        """
        raise NotImplementedError(
            "FeatureMap.features_batch is not overridden;"
            " override features or features_batch"
        )


def resolve_features(features: object, surface: str = "features") -> Resolution:
    """Resolve a ``features`` argument: bare callable or :class:`FeatureMap`.

    A bare callable resolves to :class:`Mode.DEFAULT` directly; a
    :class:`FeatureMap` instance is resolved by override detection. Comm-free.

    Raises:
        TypeError: for a non-callable, non-:class:`FeatureMap` argument, or
            a :class:`FeatureMap` overriding neither member.
    """
    if isinstance(features, FeatureMap):
        return resolve_local(
            features,
            surface=surface,
            default_name="features",
            optimized_name="features_batch",
            default_func=FeatureMap.features,
            optimized_func=FeatureMap.features_batch,
        )
    if callable(features):
        # No members to detect; the token keys on the callable's identity.
        module = getattr(features, "__module__", type(features).__module__)
        if module is None:
            module = type(features).__module__
        qualname = getattr(features, "__qualname__", type(features).__name__)
        return Resolution(
            surface=surface,
            mode=Mode.DEFAULT,
            active=features,
            reference=None,
            _module=module,
            _qualname=qualname,
        )
    raise TypeError(
        f"features must be a callable (agent_id, bundle) -> (phi, eps) or a"
        f" FeatureMap subclass instance; got {type(features).__name__}"
    )


def supports_feature_batch_aggregate(member: Callable[..., Any]) -> bool:
    """Whether ``features_batch`` advertises weighted aggregate mode."""

    try:
        params = inspect.signature(member).parameters
    except (TypeError, ValueError):
        return False
    return _aggregate_capable(params)


def _aggregate_capable(params: Mapping[str, inspect.Parameter]) -> bool:
    required = {"weights", "aggregate"}
    if not required.issubset(params):
        return False
    keyword_capable = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if not all(params[name].kind in keyword_capable for name in required):
        return False
    return "K" not in params or params["K"].kind in keyword_capable


def _aggregate_wants_k(member: Callable[..., Any]) -> bool | None:
    """``None`` when ``member`` lacks aggregate mode, else whether it takes K.

    One signature inspection per fit: callers resolve at setup and pass the
    verdict to :func:`_aggregate_call` every iteration.
    """
    try:
        params = inspect.signature(member).parameters
    except (TypeError, ValueError):
        return None
    if not _aggregate_capable(params):
        return None
    return "K" in params


def _aggregate_call(
    member: Callable[..., Any],
    ids: np.ndarray,
    bundles: np.ndarray,
    weights: np.ndarray,
    K: int,
    *,
    wants_K: bool,
) -> tuple[np.ndarray, float]:
    weights = np.asarray(weights, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if weights.shape != (ids.size,):
        raise ValueError(
            "features_batch aggregate weights must have shape"
            f" ({ids.size},); got {weights.shape}"
        )
    kwargs: dict[str, Any] = {"weights": weights, "aggregate": True}
    if wants_K:
        kwargs["K"] = int(K)
    phi, eps = member(ids, bundles, **kwargs)
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (int(K),):
        raise ValueError(
            "features_batch aggregate returned phi with shape"
            f" {phi.shape}; expected ({int(K)},)"
        )
    return phi, float(eps)


def feature_batch_aggregate(
    member: Callable[..., Any],
    ids: np.ndarray,
    bundles: np.ndarray,
    weights: np.ndarray,
    K: int,
) -> tuple[np.ndarray, float] | None:
    """Call optional aggregate mode on a batched feature map if available."""

    wants_k = _aggregate_wants_k(member)
    if wants_k is None:
        return None
    return _aggregate_call(member, ids, bundles, weights, K, wants_K=wants_k)


def _per_agent_rows(
    member: Callable[..., Any],
    ids: Sequence[int],
    bundles: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    # Per-agent member once per id, normalized to (float64 phi, float eps).
    out: list[tuple[np.ndarray, float]] = []
    for agent_id, bundle in zip(ids, bundles):
        phi, eps = member(int(agent_id), bundle)
        out.append((np.asarray(phi, dtype=np.float64), float(eps)))
    return out


def _batched_rows(
    member: Callable[..., Any],
    ids: Sequence[int],
    bundles: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    # Batched member once over the subset, split into the same per-row
    # (float64 phi, float eps) pairs the per-agent path produces.
    id_arr = np.asarray(ids, dtype=np.int64)
    bundle_arr = np.asarray(bundles)
    phi_mat, eps_vec = member(id_arr, bundle_arr)
    phi_mat = np.asarray(phi_mat, dtype=np.float64)
    eps_vec = np.asarray(eps_vec, dtype=np.float64)
    if phi_mat.shape[0] != len(ids) or eps_vec.shape[0] != len(ids):
        raise ValueError(
            "features_batch returned"
            f" Phi/Eps of length {phi_mat.shape[0]}/{eps_vec.shape[0]} for"
            f" {len(ids)} ids; the batch return must be in ids order"
        )
    return [
        (np.ascontiguousarray(phi_mat[r]), float(eps_vec[r])) for r in range(len(ids))
    ]


def feature_rows(
    resolution: Resolution,
    ids: Sequence[int],
    bundles: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    """Resolve ``(ids, bundles)`` to ``(phi (K,), eps)`` rows by the mode.

    * ``DEFAULT``: the per-agent member, one call per id.
    * ``OPTIMIZED``: the batched member, one call over the subset.
    * ``BOTH``: the batched member, then :func:`assert_conforms` against the
      per-agent member; raises on divergence.

    ``ids`` are global agent ids; ``bundles`` are the matching chosen bundles.
    """
    if len(ids) == 0:
        # Empty shard: return before any batched-member call, whose shape
        # contract may not survive a zero-length axis.
        return []
    return dispatch(
        resolution,
        on_default=lambda member: _per_agent_rows(member, ids, bundles),
        on_optimized=lambda member: _batched_rows(member, ids, bundles),
        on_both=lambda batched, per_agent: _conform_rows(
            resolution.surface, batched, per_agent, ids, bundles
        ),
    )


def _conform_rows(
    surface: str,
    batched: Callable[..., Any],
    per_agent: Callable[..., Any],
    ids: Sequence[int],
    bundles: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    # Run the optimized batch, gate against the per-agent reference. phi is
    # checked on both axes: its sign/zero-mask discretely (support
    # preservation), its values continuously; eps is continuous.
    opt = _batched_rows(batched, ids, bundles)
    ref = _per_agent_rows(per_agent, ids, bundles)
    assert_conforms(
        surface,
        optimized=[(phi, np.sign(phi), eps) for phi, eps in opt],
        reference=[(phi, np.sign(phi), eps) for phi, eps in ref],
        discrete=(1,),  # the sign/zero-mask of phi (support-preserving)
        continuous=(0, 2),  # phi values and eps
    )
    return opt


# Price surface, symmetric to the features surface: the per-agent member prices
# one (theta, agent_id) -> Demand; the batched member prices a shard
# (theta, local_ids) -> {id: Demand}. Demand is duck-typed (.bundle/.payoff/.gap).


def _per_agent_demands(
    member: Callable[..., Any],
    theta: np.ndarray,
    ids: Sequence[int] | np.ndarray,
) -> dict[int, Any]:
    return {int(a): member(theta, int(a)) for a in ids}


def conform_demands(
    surface: str,
    *,
    optimized: Mapping[int, Any],
    reference: Mapping[int, Any],
    ids: Sequence[int],
    tol: float = CONTINUOUS_TOL,
) -> None:
    """Hard-fail unless the batch's Demands match the per-agent Demands.

    The price analog of :func:`assert_conforms`: per id the chosen
    ``bundle`` is compared byte-identical, ``payoff`` and ``gap`` within
    ``tol``.

    Raises:
        AssertionError: on a mismatch or an id the batch failed to price.
    """
    opt_fields: list[tuple[np.ndarray, ...]] = []
    ref_fields: list[tuple[np.ndarray, ...]] = []
    for i in ids:
        a = int(i)
        if a not in optimized:
            raise AssertionError(
                f"method-pair conformance for {surface!r}: the batch did not"
                f" price agent {a} that the per-agent path covers"
            )
        opt_d, ref_d = optimized[a], reference[a]
        opt_fields.append(
            (
                np.asarray(opt_d.bundle),
                np.array([opt_d.payoff], dtype=np.float64),
                np.array([opt_d.gap], dtype=np.float64),
            )
        )
        ref_fields.append(
            (
                np.asarray(ref_d.bundle),
                np.array([ref_d.payoff], dtype=np.float64),
                np.array([ref_d.gap], dtype=np.float64),
            )
        )
    assert_conforms(
        surface,
        optimized=opt_fields,
        reference=ref_fields,
        discrete=(0,),  # the chosen bundle (bytes + dtype exact)
        continuous=(1, 2),  # payoff and the certified gap
        tol=tol,
    )


def _batched_demands(
    member: Callable[..., Any], theta: np.ndarray, ids: Sequence[int] | np.ndarray
) -> Mapping[int, Any]:
    # Optimized member once over the shard. Contract is shard-exact: the
    # mapping must key exactly local_ids (extra/missing ids break the per-rank
    # O(shard) bound), validated here rather than handed to the engine.
    requested = np.asarray(ids, dtype=np.int64)
    raw = member(theta, requested)
    raw_ids = getattr(raw, "ids", None)
    if raw_ids is not None:
        got_arr = np.asarray(raw_ids, dtype=np.int64)
        if got_arr.shape == requested.shape and np.array_equal(got_arr, requested):
            return raw
        out = raw
        got = set(got_arr.tolist())
    else:
        out = {int(k): v for k, v in dict(raw).items()}
        got = set(out)
    want = set(requested.tolist())
    if got != want:
        raise ValueError(
            "price_batch returned a mapping outside its requested domain"
            f" (extra ids {sorted(got - want)}, missing ids"
            f" {sorted(want - got)}): price_batch(theta, local_ids) must key"
            " exactly local_ids; this keeps the result shard-local"
        )
    return out


def price_demands(
    resolution: Resolution, theta: np.ndarray, ids: Sequence[int]
) -> Mapping[int, Any]:
    """Resolve ``(theta, ids) -> {id: Demand}`` by the mode.

    Symmetric to :func:`feature_rows`:

    * ``DEFAULT``: the per-agent member, one call per id.
    * ``OPTIMIZED``: the batched member, one call over the shard.
    * ``BOTH``: the batched member, then :func:`conform_demands` against the
      per-agent member; raises on divergence.

    ``ids`` are global agent ids; an empty shard prices nothing.
    """
    id_arr = np.asarray(ids, dtype=np.int64)
    if id_arr.size == 0:
        return {}
    return dispatch(
        resolution,
        on_default=lambda member: _per_agent_demands(member, theta, id_arr),
        on_optimized=lambda member: _batched_demands(member, theta, id_arr),
        on_both=lambda batched, per_agent: _conform_demands(
            resolution.surface, batched, per_agent, theta, id_arr
        ),
    )


def _conform_demands(
    surface: str,
    batched: Callable[..., Any],
    per_agent: Callable[..., Any],
    theta: np.ndarray,
    ids: Sequence[int],
) -> dict[int, Any]:
    # Run the optimized batch, gate against the per-agent reference, return
    # the batched mapping on a pass.
    opt = _batched_demands(batched, theta, ids)
    ref = _per_agent_demands(per_agent, theta, ids)
    conform_demands(surface, optimized=opt, reference=ref, ids=ids)
    return opt
