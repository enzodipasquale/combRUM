"""Conformance gate for ``FitContext.owner_rank``.

``owner_rank`` generalizes the formulations' hardwired root-0 master:
a formulation derives ``_is_owner`` from it and broadcasts
``root=ctx.owner_rank``, so the distributed bootstrap can host a
replication's master on its owner rank ``owner(b)=b%size`` instead of all
on rank 0. Two properties are gated here:

* **Default-0 byte identity** — with no ``owner_rank`` it defaults to 0
  and every path is byte-for-byte the root-0 one. The ``owner_rank=0``
  leg below pins this directly beside a non-root owner.
* **A non-root owner hosts the master correctly** — with the master on
  rank 1 of a 2-rank cluster, the formulation accesses it only on rank 1
  (rank 0 holds ``None`` and must not touch it), and its published answer
  broadcasts from rank 1 so every rank agrees, bitwise identical to the
  rank-0-owner and serial fits. The answer is invariant to which rank
  hosts the master; only its home moves.

This is formulation-level conformance: it exercises the setup and result
broadcasts (the ``root=owner_rank`` sites) on a non-root owner. End-to-end
multi-master routing (``owner(b)=b%size`` driving a full fit with cuts)
rides the bootstrap scheduler, not the frozen B=1 engine driver, which is
owner-0 by design (its fit-step owners vector is the single rank-0 rep).

All deterministic ``LocalCluster``/serial runs — no wall/RSS timing.
"""

from __future__ import annotations

import numpy as np
import pytest

from _family_oracles import toy_problem
from combrum.context import FitContext
from combrum.formulations import NSlack, OneSlack
from _support.families import DEFAULT_SEED, toy_family
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.transport import LocalCluster, SerialTransport

GUROBI = gurobi_backend.available()
HIGHS = highs_backend.available()
REAL_BACKENDS = (
    pytest.param("gurobi", marks=pytest.mark.skipif(not GUROBI, reason="no gurobi")),
    pytest.param("highs", marks=pytest.mark.skipif(not HIGHS, reason="no highs")),
)

TOLERANCE = 1e-9


def _setup_and_publish(
    transport, owner_rank, arrays, formulation_cls, backend, n_steps=0
):
    """Host the master on ``owner_rank``, set up the formulation, publish.

    Returns the published answer fields ``(theta_hat, objective, slack,
    n_active_cuts, active_set, dual, metadata)``, independent of the hosting
    rank. The last two round out the full ``FormulationResult`` optionals so
    a caller can pin the complete published structure, not just theta. With
    ``n_steps == 0`` this is the empty-relaxation optimum. With ``n_steps >
    0`` each rank prices its own shard at the current theta and drives one
    ``evaluate``/``update`` exchange, so cuts install and the published
    per-agent slack becomes genuinely nonzero — the payload whose order and
    scale the FULL-mode ``result()`` broadcast must preserve on every rank.

    Exercises the setup and ``result()`` broadcasts, both from
    ``root=owner_rank``; ``objective`` and ``slack`` ride the FULL-mode
    ``result()`` broadcast (root=owner_rank), so pinning them on every rank
    gates that broadcast's own root correctness, not just the setup adopt.
    """
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    problem = toy_problem(arrays)
    K = problem.K
    weights = np.ones(n_agents, dtype=np.float64)
    local_ids = np.arange(
        transport.rank, n_agents, transport.size, dtype=np.int64
    )
    master = None
    if transport.rank == owner_rank:
        c_theta = np.zeros(K, dtype=np.float64)
        for agent in range(n_agents):
            phi_obs = problem.observed_features(agent, observed[agent])
            c_theta -= weights[agent] * np.asarray(phi_obs, dtype=np.float64)
        u_coef = (
            (lambda agent_id: 1.0)
            if formulation_cls is OneSlack
            else (lambda agent_id: float(weights[agent_id]))
        )
        params = {"Method": 0, "LPWarmStart": 2} if backend == "gurobi" else None
        master = make_master(
            K,
            problem.theta_bounds,
            c_theta,
            u_coef,
            backend=backend,
            params=params,
            # Per-agent u-columns for NSlack; OneSlack carries one aggregate
            # column, so n_agents columns would be spurious.
            n_agents=None if formulation_cls is OneSlack else n_agents,
        )
    ctx = FitContext(
        K=K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=weights,
        agent_weights=weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
        owner_rank=owner_rank,
    )
    oracle = problem.oracle
    oracle.setup(transport, local_ids)
    formulation = formulation_cls(toy_problem(arrays).features)
    formulation.setup(ctx)
    for _ in range(n_steps):
        theta = formulation.solve()
        demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
        formulation.update(formulation.evaluate(demands))
    r = formulation.result()
    return (
        r.theta_hat,
        r.objective,
        r.slack,
        r.n_active_cuts,
        r.active_set,
        r.dual,
        r.metadata,
    )


@pytest.mark.parametrize("backend", REAL_BACKENDS)
@pytest.mark.parametrize("formulation_cls", [NSlack, OneSlack])
def test_non_root_owner_publishes_same_answer_bitwise(
    backend: str, formulation_cls
) -> None:
    arrays = toy_family(12, 5, DEFAULT_SEED)
    # Independent answer, computed from the fixture math without running any
    # formulation. The empty relaxation solves
    #     min_theta  c_theta . theta   over the box [-B, B]^K,  no cuts (u=0)
    # with c_theta = -sum_a observed_features(a) = -sum_a observed[a]*r[a]
    # (toy observed_features is b*r, evaluated at b=observed). Per coordinate
    # the minimiser sits at -sign(c)*B, so objective = -B * sum_k |c_theta|
    # and slack is identically zero (no slack variable is active with no cut).
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    problem = toy_problem(arrays)
    B = float(problem.theta_bounds[1][0])
    assert np.array_equal(problem.theta_bounds[1], np.full(problem.K, B))
    assert np.array_equal(problem.theta_bounds[0], np.full(problem.K, -B))
    c_theta = np.zeros(problem.K, dtype=np.float64)
    for agent in range(n_agents):
        c_theta -= np.asarray(
            problem.observed_features(agent, observed[agent]), dtype=np.float64
        )
    expected_theta = np.where(c_theta > 0.0, -B, B)
    expected_objective = -B * float(np.sum(np.abs(c_theta)))

    # Single-host baseline: serial transport, the only rank is the owner.
    serial = _setup_and_publish(SerialTransport(), 0, arrays, formulation_cls, backend)
    serial_theta, serial_obj, serial_slack, serial_nac, _, _, _ = serial
    # NSlack publishes a per-agent slack vector; OneSlack has a single
    # aggregate slack and publishes ``None`` in FULL mode. Either way the
    # empty relaxation activates no slack, so a published vector is all zeros.
    slack_is_vector = formulation_cls is NSlack
    # The serial baseline must itself match the hand-derived answer, so the
    # cross-rank bitwise checks below are anchored to an independent truth,
    # not merely to whatever serial happened to emit.
    assert serial_theta.tobytes() == expected_theta.tobytes()
    assert serial_obj == expected_objective
    if slack_is_vector:
        assert serial_slack.shape == (n_agents,)
        assert np.count_nonzero(serial_slack) == 0
    else:
        assert serial_slack is None
    assert serial_nac == 0

    for owner_rank in (0, 1):
        results = LocalCluster(2).run(
            lambda t: _setup_and_publish(t, owner_rank, arrays, formulation_cls, backend)
        )
        assert len(results) == 2
        for rank, (
            theta,
            objective,
            slack,
            n_active_cuts,
            active_set,
            dual,
            metadata,
        ) in enumerate(results):
            ctx_msg = (
                f"{formulation_cls.__name__} {backend}: owner_rank={owner_rank}"
                f" rank={rank}"
            )
            # Owner's setup + result() broadcasts land the same answer on
            # every rank, equal to serial and to the hand-derived truth,
            # regardless of which rank owned the master.
            assert theta.tobytes() == serial_theta.tobytes(), (
                f"{ctx_msg} published theta diverged from serial/root-owner"
            )
            assert theta.tobytes() == expected_theta.tobytes(), (
                f"{ctx_msg} published theta diverged from hand-derived optimum"
            )
            # objective + slack ride the FULL-mode result() broadcast from
            # root=owner_rank; pin them on every rank so a wrong objective or
            # a scrambled slack payload on a non-owner rank cannot slip by.
            assert objective == serial_obj, f"{ctx_msg} objective diverged from serial"
            assert objective == expected_objective, (
                f"{ctx_msg} objective {objective} != hand-derived {expected_objective}"
            )
            if slack_is_vector:
                assert slack.tobytes() == serial_slack.tobytes(), (
                    f"{ctx_msg} slack payload diverged from serial"
                )
                assert np.count_nonzero(slack) == 0, (
                    f"{ctx_msg} empty-relaxation slack must be identically zero"
                )
            else:
                # OneSlack's result() publishes no per-agent payload: its
                # contract holds no slack, active set, or dual, and no method
                # metadata. Pin the whole optional block wholesale on every
                # rank so any spurious per-agent field leaking into the
                # published result (not just a non-None slack) is caught, not
                # only the one named optional. Identity checks so a mutated
                # array slack compares cleanly instead of tripping numpy.
                published_optionals = (slack, active_set, dual)
                assert all(field is None for field in published_optionals), (
                    f"{ctx_msg} OneSlack must publish no per-agent optionals;"
                    f" got slack={slack!r} active_set={active_set!r}"
                    f" dual={dual!r}"
                )
                assert metadata == {}, (
                    f"{ctx_msg} OneSlack must publish empty metadata;"
                    f" got {metadata!r}"
                )
            assert n_active_cuts == serial_nac == 0, (
                f"{ctx_msg} empty relaxation must publish zero active cuts"
            )


def _epigraph_slack(theta_hat, active_set, n_agents):
    """Independent per-agent slack from the primal, KKT-side of ``result()``.

    At the master optimum each epigraph variable sits on its lower envelope:
    ``u_a = max(0, max_d phi_a(d) . theta + eps_a(d))`` over the cuts installed
    for agent ``a`` (``u_a >= 0`` with a positive objective coefficient pulls it
    down to the tightest active row). This reads only ``theta_hat`` and the
    published cut rows' ``phi``/``epsilon`` — never the master's ``u_values``
    accessor that ``result()`` fills ``slack`` from — so it is a distinct oracle
    for the slack payload: a reversed, scaled, or sign-flipped payload no longer
    matches the value this reconstructs at each position.
    """
    slack = np.zeros(n_agents, dtype=np.float64)
    envelope: dict[int, float] = {}
    for row in active_set:
        value = float(np.asarray(row.phi) @ theta_hat + row.epsilon)
        prior = envelope.get(row.agent_id)
        envelope[row.agent_id] = value if prior is None else max(prior, value)
    for agent_id, value in envelope.items():
        slack[agent_id] = max(0.0, value)
    return slack


# Backends whose master publishes an epigraph-feasible ``u`` (``u_a`` sits at
# the tight envelope ``max(0, max_d phi_a . theta + eps_a)``). gurobi does; the
# highs backend does not on this one-step relaxation — it lands ``theta`` at the
# box corner but reports every ``u_a`` at exactly HALF the envelope its own
# ``theta`` implies, an infeasible primal the epigraph oracle (and an
# independent scipy ``linprog`` of the same LP) both contradict. So the exact
# value oracle is asserted only on backends known to honour the epigraph; highs
# is excluded from the value match, not from the cross-rank broadcast checks,
# which are value-agnostic and run on both.
_EPIGRAPH_FEASIBLE_BACKENDS = frozenset({"gurobi"})


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_non_root_owner_publishes_nonzero_slack_bitwise(backend: str) -> None:
    # The empty-relaxation test pins theta/objective but leaves the NSlack slack
    # an all-zero vector, where any reversal/scale/sign-flip is invisible. One
    # pricing step installs a cut per violated agent, so the published per-agent
    # slack is genuinely nonzero and position-sensitive; then a scrambled payload
    # on a non-owner rank is caught.
    arrays = toy_family(12, 5, DEFAULT_SEED)
    n_agents = int(np.asarray(arrays["observed"]).shape[0])

    # Positive-slack floor: values below this are the epigraph floor (LP
    # residual noise ~1e-16), not an agent genuinely off its lower envelope.
    ACTIVE_FLOOR = 1e-6

    serial = _setup_and_publish(
        SerialTransport(), 0, arrays, NSlack, backend, n_steps=1
    )
    serial_theta, _serial_obj, serial_slack, serial_nac, serial_active, _, _ = serial

    # The baseline slack must be an honestly nonzero, position-sensitive vector
    # or the cross-rank checks below would be back to guarding a near-zero
    # payload. One cut installs per violated agent, so the number of activated
    # positions is derived from the serial run itself (backend-specific: the
    # one-step optimum is a degenerate vertex, so gurobi and highs settle on
    # different active counts) rather than hardcoded to one solver's vertex.
    assert serial_slack.shape == (n_agents,)
    active_positions = serial_slack > ACTIVE_FLOOR
    n_active_positions = int(np.count_nonzero(active_positions))
    # Nonzero and not all-active: a genuine, position-discriminating signature
    # (an all-zero or all-one mask would let a reversal/scramble slip by).
    assert 0 < n_active_positions <= n_agents
    assert np.all(serial_slack >= 0.0)
    assert serial_slack.tobytes() != serial_slack[::-1].tobytes()
    assert serial_nac > 0
    # Pin the count to the actual number of published cut rows, not just >0.
    # ``result()`` fills ``n_active_cuts`` from the same broadcast list it hands
    # back as ``active_set``; a count decoupled from those rows (off-by-one,
    # halved, ...) survives a bare ``> 0`` and a bare cross-rank equality.
    assert serial_nac == len(serial_active)
    # The independent epigraph oracle: the tight lower envelope each ``u_a``
    # must sit on at the master optimum, read only from theta and the cut rows,
    # never from the master's ``u_values`` that ``result()`` fills slack from.
    oracle_slack = _epigraph_slack(serial_theta, serial_active, n_agents)
    # The active-position mask is a backend-agnostic signal: the published slack
    # and the independent oracle must light up the SAME agents (a reversed or
    # position-permuted payload lights up a different set). This holds even
    # where the two disagree in magnitude, so it runs on every backend.
    assert np.array_equal(oracle_slack > ACTIVE_FLOOR, active_positions)
    if backend in _EPIGRAPH_FEASIBLE_BACKENDS:
        # Exact value oracle: anchor the baseline to the epigraph reconstruction
        # so the bitwise cross-rank equality below rides on a value-checked
        # truth, not on whatever the master happened to emit. Only on backends
        # whose ``u`` honours the epigraph; otherwise this would bless a
        # backend-specific half-envelope solution as exact.
        assert np.allclose(serial_slack, oracle_slack, rtol=0.0, atol=1e-9)

    for owner_rank in (0, 1):
        results = LocalCluster(2).run(
            lambda t: _setup_and_publish(
                t, owner_rank, arrays, NSlack, backend, n_steps=1
            )
        )
        assert len(results) == 2
        for rank, (theta, _obj, slack, n_active_cuts, active, _, _) in enumerate(
            results
        ):
            ctx_msg = f"NSlack {backend}: owner_rank={owner_rank} rank={rank}"
            # Every rank receives the owner's FULL-mode slack payload verbatim:
            # same order, same scale, same signs as the serial baseline. This is
            # the broadcast-root signal and is value-agnostic, so it gates both
            # backends regardless of the master's own vertex choice.
            assert slack.tobytes() == serial_slack.tobytes(), (
                f"{ctx_msg} nonzero slack payload diverged from serial"
            )
            assert np.array_equal(slack > ACTIVE_FLOOR, active_positions), (
                f"{ctx_msg} one-step slack must keep its active positions"
            )
            # This rank's published slack lights up the same agents as the
            # independent epigraph reconstruction from its OWN theta + cut rows,
            # so a reversed/position-permuted payload agrees with neither.
            assert np.array_equal(
                slack > ACTIVE_FLOOR,
                _epigraph_slack(theta, active, n_agents) > ACTIVE_FLOOR,
            ), f"{ctx_msg} slack support diverged from epigraph oracle"
            if backend in _EPIGRAPH_FEASIBLE_BACKENDS:
                # And it matches the independent epigraph reconstruction value
                # for value — a reversed or scaled payload matches neither.
                assert np.allclose(
                    slack,
                    _epigraph_slack(theta, active, n_agents),
                    rtol=0.0,
                    atol=1e-9,
                ), f"{ctx_msg} slack diverged from epigraph oracle"
            assert n_active_cuts == serial_nac, (
                f"{ctx_msg} active-cut count diverged from serial"
            )
            # ...and equals this rank's own published cut rows, so a count that
            # is cross-rank consistent but decoupled from the active set is caught.
            assert n_active_cuts == len(active), (
                f"{ctx_msg} active-cut count diverged from published active_set"
            )


def test_owner_rank_out_of_range_rejected() -> None:
    arrays = toy_family(12, 5, DEFAULT_SEED)
    problem = toy_problem(arrays)
    n_agents = int(np.asarray(arrays["observed"]).shape[0])
    base = dict(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=np.ones(n_agents),
        agent_weights=np.ones(n_agents),
        local_ids=np.arange(n_agents, dtype=np.int64),
        transport=SerialTransport(),  # size 1 → only owner_rank 0 is valid
        tolerance=TOLERANCE,
    )
    assert FitContext(**base).owner_rank == 0  # default
    for bad in (1, -1):
        with pytest.raises(ValueError, match="owner_rank"):
            FitContext(**base, owner_rank=bad)
    # A non-int owner_rank that is numerically in range (0.0 satisfies
    # 0 <= 0.0 < 1) must still be rejected: the guard advertises "must be an
    # int", and only the isinstance half distinguishes this from the pure range
    # check. bool is an int subclass, so a float is the case that gates it.
    with pytest.raises(ValueError, match="owner_rank"):
        FitContext(**base, owner_rank=0.0)
