from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import combrum
from _family_oracles import toy_problem
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.families import load_family
from combrum.dual import DualSolution
from combrum.engine import estimate
from combrum.formulations import NSlack, OneSlack
from combrum.masters import highs as highs_backend
from combrum.model import Data, Model
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import Transport


FAMILY_DIR = Path(__file__).resolve().parent / "fixtures" / "families"

needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)

# The JSON-ready key set of FitResult.to_dict(), hand-transcribed from the
# result.py literal. The exact-set check rejects accidental cut payloads under
# any key, both when duals are unset and when they are populated.
_EXPECTED_TO_DICT_KEYS = frozenset(
    {
        "theta_hat",
        "objective",
        "empirical_moment",
        "runtime_seconds",
        "n_active_cuts",
        "slack",
        "metadata",
    }
)


def _toy_fit(
    transport: Transport,
    *,
    formulation: type = NSlack,
    **kwargs: object,
):
    arrays = load_family("toy", FAMILY_DIR)
    problem = toy_problem(arrays)
    observed = np.asarray(arrays["observed"])
    n_obs = observed.shape[0]
    model = Model(
        problem.oracle,
        Parameters({"theta": (-THETA_BOUND, THETA_BOUND, problem.K)}),
        features=problem.features,
        observed_features=problem.observed_features,
        formulation=formulation,
    )
    data = Data(
        observed_bundles=observed,
        shocks=np.asarray(arrays["shocks"]),
        observables=list(range(n_obs)),
    )
    return estimate(
        model,
        data,
        transport=transport,
        master_backend="highs",
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        **kwargs,
    )


@needs_highs
def test_default_fit_keeps_cut_duals_unset() -> None:
    fit = _toy_fit(SerialTransport())

    assert fit.cut_duals is None
    assert fit.cuts is None
    assert set(fit.to_dict()) == _EXPECTED_TO_DICT_KEYS


@needs_highs
def test_return_cut_duals_publishes_compact_payload_without_cuts() -> None:
    fit = _toy_fit(SerialTransport(), return_cut_duals=True)

    dual = fit.cut_duals
    assert fit.cuts is None
    assert set(fit.to_dict()) == _EXPECTED_TO_DICT_KEYS
    assert isinstance(dual, DualSolution)
    assert dual.rep_id == 0
    assert dual.agent_ids.shape == (fit.n_active_cuts,)
    assert dual.bundle_row_ids.shape == (fit.n_active_cuts,)
    assert dual.pis.shape == (fit.n_active_cuts,)
    assert dual.bundle_table.ndim == 2
    assert np.isfinite(dual.pis).all()
    for arr in (dual.agent_ids, dual.bundle_row_ids, dual.pis, dual.bundle_table):
        assert not arr.flags.writeable

    # Pin the published dual VALUES against a fixture-derived oracle, not just
    # their shape/finiteness. The toy family has one observed choice per agent
    # (unit weight), so each agent's epigraph variable carries objective
    # coefficient 1.0; LP duality forces the active epigraph cuts of a given
    # agent to split exactly that unit of dual mass. Hence: total mass n_obs,
    # every agent sums to 1.0, and (this fixture is degenerate to a single
    # binding cut per agent) exactly n_obs rows carry the full unit each. The
    # per-agent aggregation below is a plain-Python loop keyed off agent_ids,
    # so it catches mis-assigned mass (e.g. reversed pis) that a bare sum would
    # not.
    observed = np.asarray(load_family("toy", FAMILY_DIR)["observed"])
    n_obs = observed.shape[0]
    assert dual.pis.sum() == pytest.approx(float(n_obs))
    per_agent: dict[int, float] = defaultdict(float)
    for agent_id, pi in zip(dual.agent_ids.tolist(), dual.pis.tolist()):
        per_agent[int(agent_id)] += float(pi)
    assert len(per_agent) == n_obs
    for total in per_agent.values():
        assert total == pytest.approx(1.0)
    unit_rows = sum(1 for pi in dual.pis.tolist() if abs(pi - 1.0) < 1e-9)
    assert unit_rows == n_obs

    # The aggregate checks above are invariant to any within-agent reshuffle
    # of pis, so they never pin which bundle_row each pi belongs to. Tie every
    # unit of mass to its bundle: the data is rationalised by theta_true, so at
    # convergence each agent's priced optimum is its observed choice, and the
    # single binding epigraph cut (pi==1.0) must therefore index that observed
    # bundle. `observed` is read from the fixture, not from combrum, so this is
    # an independent oracle; a pi<->bundle_row swap (e.g. reversed pis within an
    # agent) moves the unit row to a non-observed bundle and fails here.
    rows_by_agent: dict[int, list[int]] = defaultdict(list)
    for r, agent_id in enumerate(dual.agent_ids.tolist()):
        rows_by_agent[int(agent_id)].append(r)
    for agent_id, row_ids in rows_by_agent.items():
        unit = [r for r in row_ids if abs(float(dual.pis[r]) - 1.0) < 1e-9]
        assert len(unit) == 1
        binding = unit[0]
        np.testing.assert_array_equal(
            dual.bundle_table[int(dual.bundle_row_ids[binding])],
            observed[agent_id].astype(np.float64),
        )

    rows = list(dual.rows(n_obs=n_obs))
    assert len(rows) == fit.n_active_cuts
    assert rows
    # Pin the rows() accessor row-for-row against the payload arrays it decodes
    # from -- read directly, never back through rows(). The toy payload spans
    # many distinct bundle_row_ids (most != 0), so a per-row decode bug (e.g.
    # every row returning bundle_table[0], an off-by-one, or a reversed index)
    # cannot survive: it would move some row's generated_bundle off the table
    # row its bundle_row_id names.
    assert [r.agent_id for r in rows] == dual.agent_ids.tolist()
    assert [r.pi for r in rows] == dual.pis.tolist()
    for r, row in enumerate(rows):
        np.testing.assert_array_equal(
            row.generated_bundle,
            dual.bundle_table[int(dual.bundle_row_ids[r])],
        )
    # The (observation_id, simulation_id) split of every agent id must follow
    # the documented convention a = simulation_id*n_obs + observation_id.
    # Expected values are the modular decomposition, not the fields' own output.
    for row in rows:
        assert row.observation_id == row.agent_id % n_obs
        assert row.simulation_id == row.agent_id // n_obs


def test_dual_rows_decompose_agent_id_by_n_obs() -> None:
    # rows(n_obs) is a pure function of agent_ids and n_obs. Pin the split with
    # agent ids that exceed n_obs (n_sims > 1) so a swapped %/// or a wrong
    # divisor is caught -- the toy fit above has n_sims == 1, where the split is
    # degenerate. Expected pairs are hand-derived for n_obs=3.
    dual = DualSolution(
        rep_id=0,
        agent_ids=np.array([0, 2, 3, 5, 7], dtype=np.int64),
        bundle_row_ids=np.zeros(5, dtype=np.int64),
        pis=np.ones(5, dtype=np.float64),
        bundle_table=np.array([[1, 0]], dtype=np.int8),
        bound_duals={},
    )
    expected = {0: (0, 0), 2: (2, 0), 3: (0, 1), 5: (2, 1), 7: (1, 2)}
    got = {
        r.agent_id: (r.observation_id, r.simulation_id)
        for r in dual.rows(n_obs=3)
    }
    assert got == expected


@needs_highs
def test_return_slack_and_cut_duals_leaves_cuts_unset() -> None:
    fit = _toy_fit(
        SerialTransport(),
        return_slack=True,
        return_cut_duals=True,
        return_cuts=False,
    )

    assert fit.slack is not None
    assert fit.cuts is None
    dual = fit.cut_duals
    assert isinstance(dual, DualSolution)

    # `slack is not None` alone lets an all-zeros (or otherwise corrupted)
    # vector through. Pin the published slack against a fixture-feature oracle
    # that never touches combrum's slack path: agent a's epigraph value u_a is
    # the payoff of its binding generated bundle d*, phi_a(d*).theta + eps_a(d*)
    # (NSlack docstring: u_a >= phi_a(d).theta + eps_a(d), tight at the optimum
    # under the unit epigraph coefficient). d* is the pi==1 row in cut_duals;
    # phi/eps come from the fixture oracle and theta from the LP estimate, so the
    # expected values are recomputed on a distinct code path, not read back off
    # fit.slack. A zero-slack or reordered vector fails the elementwise compare.
    arrays = load_family("toy", FAMILY_DIR)
    problem = toy_problem(arrays)
    observed = np.asarray(arrays["observed"])
    n_obs = observed.shape[0]
    assert fit.slack.shape == (n_obs,)

    binding_row: dict[int, int] = {}
    for r, agent_id in enumerate(dual.agent_ids.tolist()):
        if abs(float(dual.pis[r]) - 1.0) < 1e-9:
            assert int(agent_id) not in binding_row
            binding_row[int(agent_id)] = int(dual.bundle_row_ids[r])
    assert set(binding_row) == set(range(n_obs))

    theta = np.asarray(fit.theta_hat, dtype=np.float64)
    expected_slack = np.empty(n_obs, dtype=np.float64)
    for agent_id in range(n_obs):
        bundle = dual.bundle_table[binding_row[agent_id]].astype(np.float64)
        phi, eps = problem.features(agent_id, bundle)
        expected_slack[agent_id] = float(np.asarray(phi) @ theta + float(eps))

    # Epigraph values are >= 0 by construction; the oracle here is strictly
    # positive on the toy data, so the all-zeros regression cannot match.
    assert (expected_slack > 1e-9).any()
    assert (np.asarray(fit.slack) >= -1e-9).all()
    np.testing.assert_allclose(np.asarray(fit.slack), expected_slack, atol=1e-6)


@needs_highs
def test_return_cuts_and_cut_duals_rows_align() -> None:
    fit = _toy_fit(
        SerialTransport(),
        return_cuts=True,
        return_cut_duals=True,
    )

    dual = fit.cut_duals
    assert fit.cuts is not None
    assert isinstance(dual, DualSolution)
    assert len(fit.cuts) == dual.pis.size == fit.n_active_cuts
    for cut, agent_id, bundle_row_id in zip(
        fit.cuts,
        dual.agent_ids,
        dual.bundle_row_ids,
    ):
        assert cut.agent_id == int(agent_id)
        np.testing.assert_array_equal(
            cut.bundle,
            dual.bundle_table[int(bundle_row_id)],
        )


# Expected message is read straight from the f-string in
# reject_multirank_dense_transport (agreement.py) with name="estimate"; it is
# NOT captured from a live estimate() call. The publication kwargs are
# deliberately varied to pin that the serial-only guard fires *before* any
# cut-dual / slack / cut publication and is unaffected by those flags: the
# top-level guard runs at estimate.py:120, ahead of the return_cut_duals check
# and all artifact assembly.
_EXPECTED_MULTIRANK_MESSAGE = (
    f"estimate does not support non-serial transport in combRUM {combrum.__version__};"
    " use estimate_distributed for distributed runs"
)


@needs_highs
@pytest.mark.parametrize(
    "publication_kwargs",
    [
        pytest.param({}, id="no-publication-flags"),
        pytest.param({"return_cut_duals": True}, id="cut-duals"),
        pytest.param({"return_slack": True}, id="slack"),
        pytest.param({"return_cuts": True}, id="cuts"),
        pytest.param(
            {
                "return_slack": True,
                "return_cuts": True,
                "return_cut_duals": True,
            },
            id="all-artifacts",
        ),
    ],
)
def test_dense_estimate_rejects_multirank_regardless_of_publication(
    publication_kwargs: dict[str, bool],
) -> None:
    # The no-flags case is the control: it proves the raise is driven by the
    # blanket serial-only guard, not by any publication path. The remaining
    # cases pin that adding cut-dual / slack / cut flags neither bypasses the
    # guard nor changes the message, so no publication branch can leak a
    # multirank run past the serial gate.
    with pytest.raises(ValueError) as excinfo:
        LocalCluster(2).run(
            lambda transport: _toy_fit(transport, **publication_kwargs)
        )
    assert str(excinfo.value) == _EXPECTED_MULTIRANK_MESSAGE


def test_oneslack_return_cut_duals_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="return_cut_duals is only supported for NSlack"
    ):
        _toy_fit(
            SerialTransport(),
            formulation=OneSlack,
            return_cut_duals=True,
        )
