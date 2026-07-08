from __future__ import annotations

import numpy as np
import pytest

from combrum.engine.driver import _validate_schedule_formulation
from combrum.formulations import NSlack, OneSlack
from combrum.informed_schedule import DualInformed
from combrum.schedule import RepricingSchedule, ResolveAll, RoundRobin
from combrum.transport import LocalCluster, Transport


def test_abc_not_instantiable() -> None:
    with pytest.raises(TypeError):
        RepricingSchedule()  # type: ignore[abstract]


def test_resolve_all_mask() -> None:
    schedule = ResolveAll()
    assert isinstance(schedule, RepricingSchedule)
    for iteration in (0, 7):
        mask = schedule.select(iteration, 5)
        assert mask.dtype == bool and mask.shape == (5,)
        assert mask.all()


def test_round_robin_mask_form() -> None:
    mask = RoundRobin(3).select(0, 8)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (8,)
    # Chunk 0 of 8 agents over 3 slices is the first ceil(8/3)=3 agents.
    assert np.flatnonzero(mask).tolist() == [0, 1, 2]


def test_round_robin_cycle_is_exact_partition() -> None:
    # Over one full cycle of `chunks` iterations each agent is selected
    # exactly once (partition: union covers all, pairwise disjoint).
    for chunks, n_agents in [(1, 5), (3, 8), (4, 10), (5, 3)]:
        schedule = RoundRobin(chunks)
        masks = [schedule.select(i, n_agents) for i in range(chunks)]
        counts = np.sum(masks, axis=0)
        np.testing.assert_array_equal(
            counts, np.ones(n_agents, dtype=np.int64)
        )


def test_round_robin_slices_balanced_and_contiguous() -> None:
    schedule = RoundRobin(4)
    sizes = []
    for i in range(4):
        mask = schedule.select(i, 10)
        selected = np.flatnonzero(mask)
        sizes.append(selected.size)
        # Contiguous slice: consecutive indices throughout.
        assert np.array_equal(
            selected, np.arange(selected[0], selected[0] + selected.size)
        )
    assert max(sizes) - min(sizes) <= 1  # balanced
    assert sizes == [3, 3, 2, 2]


def test_round_robin_pure_function_of_iteration_mod_chunks() -> None:
    schedule = RoundRobin(3)
    for i in range(3):
        a = schedule.select(i, 8)
        b = schedule.select(i, 8)
        c = RoundRobin(3).select(i, 8)  # independent instance
        d = schedule.select(i + 3, 8)  # next cycle
        assert a.tobytes() == b.tobytes() == c.tobytes() == d.tobytes()


class _RecordingTransport:
    """Delegate to a real transport, recording every attribute it hands out.

    Any collective call or property read a schedule performs shows up in
    ``touched``; a zero-communication schedule leaves it empty.
    """

    def __init__(self, inner: Transport) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name: str) -> object:
        self.touched.append(name)
        return getattr(self._inner, name)


def test_round_robin_rank_local_zero_communication() -> None:
    # Each rank builds its schedule from (iteration, n_agents) alone; any
    # transport access would show up in the proxy's `touched`.
    def per_rank(transport: Transport) -> tuple[list[bytes], list[str]]:
        spy = _RecordingTransport(transport)
        schedule = RoundRobin(3)
        # RoundRobin must ignore the informed-signal arguments entirely: a
        # spy-passing mask equals the bare (dual=None) mask byte-for-byte.
        masks = []
        for i in range(6):
            informed = schedule.select(
                i, 8, dual=spy, last_resolved=np.arange(8)
            ).tobytes()
            bare = schedule.select(i, 8).tobytes()
            assert informed == bare, (
                f"iter {i}: passing dual/last_resolved changed the mask"
            )
            masks.append(informed)
        return masks, list(spy.touched)

    per_rank_out = LocalCluster(2).run(per_rank)
    masks0, touched0 = per_rank_out[0]
    masks1, touched1 = per_rank_out[1]
    assert touched0 == [], f"rank 0 touched the transport: {touched0}"
    assert touched1 == [], f"rank 1 touched the transport: {touched1}"
    assert masks0 == masks1


def test_round_robin_chunks_validation() -> None:
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError, match="chunks must be an integer >= 1"):
            RoundRobin(bad)  # type: ignore[arg-type]
    assert RoundRobin(np.int64(2)).select(0, 4).sum() == 2


def test_oneslack_accepts_only_full_schedules() -> None:
    formulation = OneSlack(lambda agent_id, bundle: (np.zeros(1), 0.0))

    _validate_schedule_formulation(formulation, None)
    _validate_schedule_formulation(formulation, ResolveAll())
    with pytest.raises(ValueError, match="OneSlack requires full re-pricing"):
        _validate_schedule_formulation(formulation, RoundRobin(2))
    with pytest.raises(ValueError, match="OneSlack requires full re-pricing"):
        _validate_schedule_formulation(formulation, DualInformed())


def test_oneslack_guard_pins_defining_module_not_just_name() -> None:
    # The restriction keys on the defining module AND qualname: a user's own
    # class that happens to be named "OneSlack" keeps full access to partial
    # schedules. Same-name-fake setup as in test_persistent_master.py.
    fake = type("OneSlack", (), {})()
    assert type(fake).__qualname__ == "OneSlack"
    assert type(fake).__module__ != "combrum.formulations.oneslack"
    for schedule in (None, ResolveAll(), RoundRobin(2), DualInformed()):
        _validate_schedule_formulation(fake, schedule)


def test_partial_schedules_accepted_for_non_oneslack() -> None:
    # Only OneSlack forbids a partial schedule. Cover a bare object and a real
    # NSlack instance so the guard cannot over-fire on other formulations.
    nslack = NSlack(lambda agent_id, bundle: (np.zeros(1), 0.0))
    for formulation in (object(), nslack):
        for schedule in (None, ResolveAll(), RoundRobin(2), DualInformed()):
            _validate_schedule_formulation(formulation, schedule)


def test_schedule_validation_rejects_timeout_schedule_shape() -> None:
    with pytest.raises(TypeError, match="RepricingSchedule"):
        _validate_schedule_formulation(object(), object())


def test_resolve_all_repr() -> None:
    assert repr(ResolveAll()) == "ResolveAll()"


def test_round_robin_repr_shows_chunks() -> None:
    assert repr(RoundRobin(3)) == "RoundRobin(chunks=3)"


def test_dual_informed_repr_shows_params() -> None:
    assert (
        repr(DualInformed(concentration_threshold=0.75, min_revisit_period=8))
        == "DualInformed(concentration_threshold=0.75, min_revisit_period=8)"
    )
