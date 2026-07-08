from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

import combrum
from _family_oracles import toy_problem
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.families import FAMILY_DIR, load_family
from combrum.dual import DualSolution
from combrum.engine import estimate
from combrum.formulations import NSlack, OneSlack
from combrum.masters import highs as highs_backend
from combrum.model import Data, Model
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import Transport

needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)

# Key set of FitResult.to_dict(); the exact-set checks below reject cut
# payloads leaking in under any key.
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

    # One observed choice per agent (unit epigraph coefficient), so LP duality
    # splits exactly one unit of dual mass across each agent's active cuts; on
    # this fixture a single cut binds per agent, so exactly n_obs rows carry
    # the full unit.
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

    # The data is rationalised by theta_true, so at convergence each agent's
    # binding cut (pi == 1.0) must index its observed bundle.
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
    # rows() decodes the payload arrays; compare row-for-row against the
    # arrays read directly.
    assert [r.agent_id for r in rows] == dual.agent_ids.tolist()
    assert [r.pi for r in rows] == dual.pis.tolist()
    for r, row in enumerate(rows):
        np.testing.assert_array_equal(
            row.generated_bundle,
            dual.bundle_table[int(dual.bundle_row_ids[r])],
        )
    # Agent id convention: a = simulation_id*n_obs + observation_id.
    for row in rows:
        assert row.observation_id == row.agent_id % n_obs
        assert row.simulation_id == row.agent_id // n_obs


def test_dual_rows_decompose_agent_id_by_n_obs() -> None:
    # Use agent ids above n_obs (n_sims > 1) so a swapped % / // shows up; the
    # toy fit has n_sims == 1, where the split is degenerate.
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

    # NSlack: u_a >= phi_a(d).theta + eps_a(d), tight at the optimum under the
    # unit epigraph coefficient, so slack[a] equals the payoff of the binding
    # bundle d* -- the pi == 1 row in cut_duals. Recompute it from the fixture
    # features and theta_hat.
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

    # Epigraph values are >= 0 by construction and strictly positive on the
    # toy data.
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


# Matches the f-string in reject_multirank_dense_transport (agreement.py) with
# name="estimate". The guard runs before any cut-dual / slack / cut
# publication, so the message does not depend on those flags.
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
