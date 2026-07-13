from __future__ import annotations

import importlib
import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from _support.commprobe import CountingTransport
from combrum.activity import ActivityConfig, ActivityRun, BootstrapStart
from combrum.bootstrap_distributed import (
    _Replica,
    _bootstrap_local_rows,
    _bootstrap_slack_coef,
    _bootstrap_wave_c_theta_and_normalizers,
    _cut_exchange_block_size,
    _finish_bootstrap_reduction,
    _observed_cut_row_nbytes,
    _owner_vector,
    _pack_master_state,
    _run_replica_wave,
    _store_dual,
    _unpack_master_state,
    batched_reduce,
    bootstrap_distributed,
)
from combrum.context import ResultPublication
from combrum.cut_policies import AddAll, PurgeInactive
from combrum.demand import Demand
from combrum.dual import DualSolution
from combrum.dualstore import DualStoreReader, DualStoreWriter
from combrum.engine.distributed_context import prepare_distributed_observed
from combrum.formulation import FormulationResult
from combrum.formulations import NSlack, OneSlack
from combrum.interface_resolution import resolve
from combrum.model import Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.randomness import bootstrap_multiplier
from combrum.result import BootstrapResult
from combrum.rowgen import (
    MaxContribution,
    MaxReduced,
    StepOutcome,
)
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import CutRow, Transport, TransportError, _cut_row_nbytes


class _CapturingTransport(CountingTransport):
    def __init__(self, inner) -> None:
        super().__init__(inner)
        self.sum_ids: list[np.ndarray] = []
        self.sum_shapes: list[tuple[int, ...]] = []

    def sum_reproducible(self, values: np.ndarray, global_ids: np.ndarray):
        self.sum_shapes.append(tuple(np.asarray(values).shape))
        self.sum_ids.append(np.asarray(global_ids, dtype=np.int64).copy())
        return super().sum_reproducible(values, global_ids)


class _ReduceShapeTransport(SerialTransport):
    def __init__(self) -> None:
        self.max_shapes: list[tuple[int, ...]] = []
        self.exchange_owner_shapes: list[tuple[int, ...]] = []

    def batched_max(self, values: np.ndarray) -> np.ndarray:
        self.max_shapes.append(tuple(np.asarray(values).shape))
        return super().batched_max(values)

    def exchange_cuts(self, rows, owners: np.ndarray):
        self.exchange_owner_shapes.append(tuple(np.asarray(owners).shape))
        return super().exchange_cuts(rows, owners)


class _ObservedSurface:
    def __init__(self, K: int) -> None:
        self.K = K
        self.setup_ids: tuple[int, ...] = ()

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        self.setup_ids = tuple(map(int, observation_ids))

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(observation_ids, dtype=np.float64)
        rows = np.column_stack([ids + j + 1.0 for j in range(self.K)])
        return np.ascontiguousarray(rows, dtype=np.float64)

    def __call__(self, agent_id: int, bundle: np.ndarray):
        b = np.asarray(bundle, dtype=np.float64)
        return np.array([float(b[0])], dtype=np.float64), 0.0


class _RecordingOracle(Oracle):
    def __init__(self) -> None:
        self.setup_ids: tuple[int, ...] = ()

    def setup(self, transport: Transport, agent_ids: np.ndarray) -> None:
        self.setup_ids = tuple(map(int, agent_ids))

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        return Demand.exact(
            bundle=np.array([1.0], dtype=np.float64),
            payoff=float(theta[0]),
        )

    def price_batch(self, theta: np.ndarray, agent_ids: np.ndarray):
        return {int(agent_id): self.price(theta, int(agent_id)) for agent_id in agent_ids}


def _model(
    *,
    K: int = 1,
    oracle: Oracle | None = None,
    observed: _ObservedSurface | None = None,
    formulation=NSlack,
) -> Model:
    surface = observed if observed is not None else _ObservedSurface(K)
    return Model(
        oracle if oracle is not None else _RecordingOracle(),
        Parameters({"theta": (-2.0, 2.0, K)}),
        features=surface,
        observed_features=surface,
        formulation=formulation,
    )


def _replica_wave_result(replicas, **kwargs) -> BootstrapResult:
    B = len(replicas)
    K = kwargs["K"]
    return BootstrapResult(
        thetas=np.zeros((B, K), dtype=np.float64),
        converged=np.ones(B, dtype=bool),
        parameters=kwargs["parameters"],
        iterations=1,
    )


class _FakeDistributedFormulation(NSlack):
    def __init__(self, prep) -> None:
        self._prep = prep
        self.owner_u: dict[int, float] | None = None

    def _distributed_feature_token(self) -> tuple[str, int]:
        return ("fake-features", int(self._prep.K))

    def _distributed_route_spec(self) -> tuple[np.ndarray, int]:
        return self._prep.local_ids, int(self._prep.n_agents)

    def _distributed_set_owner_u(self, values) -> None:
        self.owner_u = None if values is None else dict(values)

    def _distributed_owner_u(self) -> None:
        return None

    def _adopt_owner_state(
        self,
        state,
        *,
        local_u,
        full_u,
        bump_iteration: bool,
    ) -> None:
        self.last_state = state
        self.last_local_u = dict(local_u)
        self.last_bump_iteration = bool(bump_iteration)


def _fake_replica(rep_id: int, prep) -> SimpleNamespace:
    return SimpleNamespace(
        rep_id=rep_id,
        formulation=_FakeDistributedFormulation(prep),
        initial_state=SimpleNamespace(
            theta=np.zeros(prep.K, dtype=np.float64),
            objective=0.0,
            n_installed=0,
            progressed=0,
        ),
        initial_full_u={},
    )


def _price_resolution(oracle, transport):
    return resolve(
        oracle,
        surface="price",
        default_name="price",
        optimized_name="price_batch",
        default_func=Oracle.price,
        optimized_func=Oracle.price_batch,
        transport=transport,
    )


def test_public_signature_is_split_axis_only() -> None:
    sig = inspect.signature(bootstrap_distributed)
    params = sig.parameters

    assert list(params) == [
        "model",
        "n_observations",
        "n_simulations",
        "n_bootstrap",
        "base_seed",
        "transport",
        "max_live_reps",
        "master_backend",
        "master_params",
        "tolerance",
        "max_iterations",
        "min_iterations",
        "iteration_callback",
        "warm_start",
        "warm_cuts",
        "cut_policy",
        "dual_store_dir",
        "activity",
    ]
    assert params["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in list(params)[1:]:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["max_live_reps"].default == 64
    for removed in (
        "observed_bundles",
        "weights",
        "collect_payload",
        "only_converged",
        "checkpoint_dir",
        "load_dir",
    ):
        assert removed not in params
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_bootstrap_wave_reduction_is_keyed_by_observations() -> None:
    prep = SimpleNamespace(
        K=2,
        N=4,
        S=3,
        owned_obs=np.arange(4, dtype=np.int64),
        phi_obs_local=np.ascontiguousarray(
            np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0],
                ],
                dtype=np.float64,
            )
        ),
    )
    transport = _CapturingTransport(SerialTransport())
    c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
        prep,
        [0, 1, 2],
        base_seed=11,
        transport=transport,
    )

    assert c_thetas.shape == (3, 2)
    assert normalizers.shape == (3,)
    # Per rep: raw = bootstrap_multiplier(11, rep_id, obs), normalizer =
    # sum(raw), c_theta = sum_obs(-S * raw * phi_obs) * N / normalizer.
    raw = np.array(
        [
            [bootstrap_multiplier(11, rep_id, int(obs)) for obs in prep.owned_obs]
            for rep_id in (0, 1, 2)
        ],
        dtype=np.float64,
    )
    expected_normalizers = raw.sum(axis=1)
    unscaled = np.einsum("ro,ok->rk", -float(prep.S) * raw, prep.phi_obs_local)
    expected_c_thetas = unscaled * (float(prep.N) / expected_normalizers)[:, None]
    np.testing.assert_allclose(
        normalizers, expected_normalizers, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(c_thetas, expected_c_thetas, rtol=0.0, atol=1e-12)
    assert transport.counts() == {"sum_reproducible": 1}
    assert transport.sum_shapes == [(4, 9)]
    np.testing.assert_array_equal(transport.sum_ids[0], np.arange(4, dtype=np.int64))
    assert transport.bytes_moved()["sum_reproducible"] == 4 * 9 * 8 + 4 * 8
    assert "sum_vectors_reproducible" not in transport.counts()


def test_bootstrap_wave_reduction_chunks_large_observed_blocks(monkeypatch) -> None:
    prep = SimpleNamespace(
        K=1,
        N=3,
        S=2,
        owned_obs=np.arange(3, dtype=np.int64),
        phi_obs_local=np.ascontiguousarray(
            np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        ),
    )
    boot_module = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(
        boot_module,
        "_BOOTSTRAP_OBS_BLOCK_ELEMENTS",
        prep.owned_obs.size * (prep.K + 1),
    )
    transport = _CapturingTransport(SerialTransport())

    c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
        prep,
        [0, 1, 2, 3],
        base_seed=13,
        transport=transport,
    )

    assert c_thetas.shape == (4, 1)
    assert normalizers.shape == (4,)
    assert transport.counts() == {"sum_reproducible": 4}
    assert transport.sum_shapes == [(3, 2)] * 4
    for ids in transport.sum_ids:
        np.testing.assert_array_equal(ids, np.arange(3, dtype=np.int64))


def test_bootstrap_wave_reduction_chunks_observation_axis(monkeypatch) -> None:
    prep = SimpleNamespace(
        K=1,
        N=3,
        S=2,
        owned_obs=np.arange(3, dtype=np.int64),
        phi_obs_local=np.ascontiguousarray(
            np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        ),
    )
    boot_module = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(boot_module, "_BOOTSTRAP_OBS_BLOCK_ELEMENTS", 4)
    transport = _CapturingTransport(SerialTransport())

    c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
        prep,
        [0, 1],
        base_seed=17,
        transport=transport,
    )

    assert c_thetas.shape == (2, 1)
    assert normalizers.shape == (2,)
    assert transport.counts() == {"sum_reproducible": 4}
    assert transport.sum_shapes == [(2, 2), (1, 2), (2, 2), (1, 2)]
    np.testing.assert_array_equal(transport.sum_ids[0], np.array([0, 1]))
    np.testing.assert_array_equal(transport.sum_ids[1], np.array([2]))
    np.testing.assert_array_equal(transport.sum_ids[2], np.array([0, 1]))
    np.testing.assert_array_equal(transport.sum_ids[3], np.array([2]))


def test_finish_bootstrap_reduction_checks_expected_rep_count() -> None:
    prep = SimpleNamespace(K=2, N=4)

    with pytest.raises(ValueError, match=r"expected \(3, 3\)"):
        _finish_bootstrap_reduction(
            prep,
            np.ones((2, 3), dtype=np.float64),
            n_reps=3,
        )


def test_bootstrap_wave_workspace_scales_with_observations_not_agents(
    monkeypatch,
) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")

    prep = SimpleNamespace(
        K=3,
        N=252,
        S=20,
        owned_obs=np.arange(51, dtype=np.int64),
        phi_obs_local=np.ascontiguousarray(
            np.arange(51 * 3, dtype=np.float64).reshape(51, 3)
        ),
    )
    rep_ids = list(range(7))
    shapes: list[object] = []
    real_empty = np.empty

    def guard_no_grid_alloc(shape, *args, **kwargs):  # type: ignore[no-untyped-def]
        shapes.append(shape)
        if tuple(shape) == (len(rep_ids), prep.N * prep.S):
            raise AssertionError("bootstrap must not allocate a (B, N*S) grid")
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(bd.np, "empty", guard_no_grid_alloc)

    rows = _bootstrap_local_rows(prep, base_seed=3, rep_ids=rep_ids)

    assert rows.shape == (prep.owned_obs.size, len(rep_ids) * (prep.K + 1))
    assert shapes == [rows.shape]


def test_bootstrap_normalizers_are_bitwise_rank_invariant() -> None:
    N, S, K = 252, 20, 2

    def run(transport):
        prep = prepare_distributed_observed(
            _model(K=K),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
            prep,
            [7, 8],
            base_seed=12345,
            transport=transport,
        )
        return c_thetas.tobytes(), normalizers.tobytes()

    expected = run(SerialTransport())
    for size in (2, 4, 5):
        for got in LocalCluster(size).run(run):
            assert got == expected


def test_bootstrap_reduction_chunks_are_global_rank_invariant(monkeypatch) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(bd, "_BOOTSTRAP_OBS_BLOCK_ELEMENTS", 12)
    N, S, K = 17, 3, 2

    def run(transport):
        prep = prepare_distributed_observed(
            _model(K=K),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
            prep,
            [1, 4, 9],
            base_seed=123,
            transport=transport,
        )
        return c_thetas.tobytes(), normalizers.tobytes()

    expected = run(SerialTransport())
    for size in (2, 3, 5, 19):
        for got in LocalCluster(size).run(run):
            assert got == expected


def test_bootstrap_slack_coef_reuses_observation_weight_across_simulations() -> None:
    N = 252
    raw_sum = sum(bootstrap_multiplier(7, 3, i) for i in range(N))
    coef = _bootstrap_slack_coef(
        n_observations=N,
        base_seed=7,
        rep_id=3,
        normalizer=raw_sum,
    )

    assert coef(17) == coef(N + 17)
    assert coef(17) == coef(19 * N + 17)
    assert coef(17) != coef(18)
    expected_17 = float.fromhex("0x1.4084e8ee37b0bp+1")
    expected_18 = float.fromhex("0x1.82146633c47f8p-3")
    assert coef(17) == pytest.approx(
        expected_17, rel=0.0, abs=2.0 * np.spacing(expected_17)
    )
    assert coef(18) == pytest.approx(
        expected_18, rel=0.0, abs=2.0 * np.spacing(expected_18)
    )


def test_bootstrap_distributed_wires_split_axis_without_full_data(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    def fake_build(rep_id, **kwargs):
        prep = kwargs["prep"]
        records.append(
            {
                "rep_id": rep_id,
                "N": prep.N,
                "S": prep.S,
                "local_ids": np.asarray(prep.local_ids).copy(),
                "c_theta": np.asarray(kwargs["c_theta"]).copy(),
                "normalizer": float(kwargs["normalizer"]),
                "owner_rank": int(kwargs["owner_rank"]),
            }
        )
        return _fake_replica(rep_id, kwargs["prep"])

    # Per-slot-distinct thetas/converged make the slot->rep_id scatter visible:
    # row = [1000*rep_id + 1, 1000*rep_id + 2], and rep 1 alone non-converged.
    def distinct_replica_wave(replicas, **kwargs):
        K = kwargs["K"]
        rep_ids = [replica.rep_id for replica in replicas]
        thetas = np.array(
            [[1000.0 * rid + (j + 1) for j in range(K)] for rid in rep_ids],
            dtype=np.float64,
        )
        converged = np.array([rid != 1 for rid in rep_ids], dtype=bool)
        return BootstrapResult(
            thetas=thetas,
            converged=converged,
            parameters=kwargs["parameters"],
            iterations=1,
        )

    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(bd, "_build_distributed_replica", fake_build)
    monkeypatch.setattr(bd, "_nslack", lambda replica: replica.formulation)
    monkeypatch.setattr(bd, "_run_replica_wave", distinct_replica_wave)

    oracle = _RecordingOracle()
    observed = _ObservedSurface(K=2)
    transport = CountingTransport(SerialTransport())
    result = bootstrap_distributed(
        _model(K=2, oracle=oracle, observed=observed),
        n_observations=252,
        n_simulations=20,
        n_bootstrap=3,
        base_seed=5,
        transport=transport,
        master_backend="highs",
        # Two waves ((0,1) and (2,)) so slot != rep_id in the second wave.
        max_live_reps=2,
    )

    assert result.thetas.shape == (3, 2)
    # Each stamped row must land at its global rep index, not its wave slot.
    np.testing.assert_array_equal(
        result.thetas,
        np.array([[1.0, 2.0], [1001.0, 1002.0], [2001.0, 2002.0]]),
    )
    assert result.converged.tolist() == [True, False, True]
    assert observed.setup_ids == tuple(range(252))
    assert len(oracle.setup_ids) == 252 * 20
    assert oracle.setup_ids[:252] == tuple(range(252))
    assert oracle.setup_ids[-1] == 19 * 252 + 251
    assert [r["rep_id"] for r in records] == [0, 1, 2]
    assert all(r["N"] == 252 and r["S"] == 20 for r in records)
    assert all(len(r["local_ids"]) == 252 * 20 for r in records)
    assert [r["owner_rank"] for r in records] == [0, 0, 0]

    # Each replica must receive its own rep's bootstrap weights, not slot 0's.
    # Expected values recomputed per rep_id: phi_obs row = [obs+1, obs+2]
    # (_ObservedSurface(K=2)), raw = bootstrap_multiplier(5, rep_id, obs),
    # normalizer = sum(raw), c_theta = sum_obs(-S * raw * phi) * N / normalizer.
    N, S = 252, 20
    obs = np.arange(N, dtype=np.float64)
    phi_obs = np.column_stack([obs + 1.0, obs + 2.0])
    for record in records:
        rep_id = int(record["rep_id"])
        raw = np.array(
            [bootstrap_multiplier(5, rep_id, int(o)) for o in obs],
            dtype=np.float64,
        )
        expected_normalizer = raw.sum()
        expected_c_theta = np.einsum("o,ok->k", -float(S) * raw, phi_obs) * (
            float(N) / expected_normalizer
        )
        assert record["normalizer"] == pytest.approx(expected_normalizer, rel=0, abs=1e-9)
        np.testing.assert_allclose(
            record["c_theta"], expected_c_theta, rtol=0.0, atol=1e-6
        )
    # Distinct reps draw distinct weights, so sharing slot 0's could not pass.
    assert records[0]["normalizer"] != pytest.approx(records[1]["normalizer"])
    assert not np.allclose(records[0]["c_theta"], records[1]["c_theta"])
    # Empirical observed moments (1), the two live waves' c_theta/normalizers
    # (2), and final pricing-gap certification (1) all use row-keyed
    # reproducible sums. The owner map gathers and broadcasts three topology
    # columns once.
    assert transport.counts()["sum_reproducible"] == 4
    assert transport.counts()["gather_agent_values"] == 3
    assert transport.counts()["bcast"] >= 3
    assert transport.counts()["allreduce_max"] == 1


def test_bootstrap_distributed_routes_wave_masters_by_rep_id(monkeypatch) -> None:
    # Multi-rank + multi-wave: size=3 with max_live_reps=1 runs five single-rep
    # waves, so slot is always 0 while rep_id sweeps 0..4. The owner-routing
    # boundary (owners[rep_id] vs owners[slot]) is only crossed here; a single
    # rank or single wave makes the two indices coincide.
    bd = importlib.import_module("combrum.bootstrap_distributed")
    lock = threading.Lock()
    seen: dict[tuple[int, int], int] = {}

    def fake_build(rep_id, **kwargs):
        rank = int(kwargs["transport"].rank)
        with lock:
            seen[(rank, int(rep_id))] = int(kwargs["owner_rank"])
        return _fake_replica(rep_id, kwargs["prep"])

    # Set once (not per-rank in run) so concurrent rank threads don't race the
    # patched attribute; the recorder keys by the transport's own rank.
    monkeypatch.setattr(bd, "_build_distributed_replica", fake_build)
    monkeypatch.setattr(bd, "_run_replica_wave", _replica_wave_result)
    monkeypatch.setattr(bd, "_publish_nslack_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(bd, "_nslack", lambda replica: replica.formulation)

    def run(transport):
        bootstrap_distributed(
            _model(),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=5,
            base_seed=1,
            transport=transport,
            master_backend="highs",
            max_live_reps=1,
        )
        return int(transport.rank)

    LocalCluster(3).run(run)

    rank0_seen = {rep_id: owner for (rank, rep_id), owner in seen.items() if rank == 0}
    # On one node, node-interleaved placement reduces to rep_id % size =
    # [0, 1, 2, 0, 1]; owners[slot] (slot is always 0) would give all zeros.
    assert rank0_seen == {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}
    for rank in range(3):
        rank_seen = {rid: owner for (rk, rid), owner in seen.items() if rk == rank}
        assert rank_seen == {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}


def test_bootstrap_start_event_reports_resolved_split_axis_metadata(
    monkeypatch,
) -> None:
    events: list[object] = []

    class _Sink:
        def emit(self, event: object) -> None:
            events.append(event)

    def fake_activity(config, *, is_root, stream=None):  # type: ignore[no-untyped-def]
        return ActivityRun(config=config, sink=_Sink() if is_root else None)

    def fake_build(rep_id, **kwargs):
        return _fake_replica(rep_id, kwargs["prep"])

    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(bd, "build_activity_run", fake_activity)
    monkeypatch.setattr(bd, "resolve_master_backend", lambda *args, **kwargs: "highs")
    monkeypatch.setattr(bd, "_build_distributed_replica", fake_build)
    monkeypatch.setattr(bd, "_nslack", lambda replica: replica.formulation)
    monkeypatch.setattr(bd, "_run_replica_wave", _replica_wave_result)

    bootstrap_distributed(
        _model(K=2),
        n_observations=11,
        n_simulations=5,
        n_bootstrap=2,
        base_seed=99,
        transport=SerialTransport(),
        master_backend="auto",
        tolerance=3e-4,
        max_iterations=17,
        min_iterations=3,
        activity=ActivityConfig(level="summary", stdout=True),
    )

    starts = [event for event in events if isinstance(event, BootstrapStart)]
    assert len(starts) == 1
    start = starts[0]
    assert start.n_obs == 11
    assert start.n_simulations == 5
    assert start.n_agents == 55
    assert start.min_iterations == 3
    assert start.master_backend == "highs"
    # Each field echoes the distinct literal supplied in the call.
    assert start.n_bootstrap == 2
    assert start.base_seed == 99
    assert start.tolerance == pytest.approx(3e-4)
    assert start.max_iterations == 17
    assert start.n_parameters == 2
    assert start.resampling == "multiplier"


def test_bootstrap_observed_setup_reduces_once_per_wave(monkeypatch) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(
        bd,
        "_build_distributed_replica",
        lambda rep_id, **kwargs: _fake_replica(rep_id, kwargs["prep"]),
    )
    waves: list[tuple[int, ...]] = []

    def fake_run(replicas, **kwargs):
        waves.append(tuple(replica.rep_id for replica in replicas))
        return _replica_wave_result(replicas, **kwargs)

    monkeypatch.setattr(bd, "_run_replica_wave", fake_run)
    monkeypatch.setattr(bd, "_nslack", lambda replica: replica.formulation)
    transport = CountingTransport(SerialTransport())
    bootstrap_distributed(
        _model(K=2),
        n_observations=8,
        n_simulations=3,
        n_bootstrap=5,
        base_seed=4,
        transport=transport,
        master_backend="highs",
        max_live_reps=2,
    )

    assert waves == [(0, 1), (2, 3), (4,)]
    # One sum for empirical observed moments, three for the live waves
    # (2, 2, 1 reps), and one more for final pricing-gap certification
    # (which also drives the single max reduction) = 5 sums, 1 max. Owner
    # placement gathers and broadcasts three topology columns once.
    assert transport.counts()["sum_reproducible"] == 5
    assert transport.counts()["gather_agent_values"] == 3
    assert transport.counts()["bcast"] >= 3
    assert transport.counts()["allreduce_max"] == 1


def test_bootstrap_distributed_validates_max_live_reps() -> None:
    for bad in (0, -1, 1.5, "2", True, np.bool_(True)):
        with pytest.raises((TypeError, ValueError, TransportError), match="max_live_reps"):
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=SerialTransport(),
                max_live_reps=bad,  # type: ignore[arg-type]
            )


def test_bootstrap_distributed_requires_rank_uniform_max_live_reps() -> None:
    def run(transport):
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
                max_live_reps=1 if transport.rank == 0 else 2,
            )
        except TransportError as exc:
            return (exc.rank, exc.message)
        return (-1, "no error")

    results = LocalCluster(2).run(run)

    assert all(rank >= 0 for rank, _ in results)
    assert all("max_live_reps must match" in message for _, message in results)


@pytest.mark.parametrize(
    ("name", "rank0", "rank1", "match"),
    [
        ("n_bootstrap", 1, 2, "n_bootstrap must match"),
        ("base_seed", 1, 2, "base_seed must match"),
        ("max_iterations", 1, 2, "max_iterations must match"),
        ("min_iterations", 0, 1, "min_iterations must match"),
        ("tolerance", 1e-6, 1e-5, "tolerance must match"),
        ("master_backend", "highs", "bad", "master_backend must match"),
    ],
)
def test_bootstrap_distributed_requires_rank_uniform_public_controls(
    name, rank0, rank1, match
) -> None:
    def run(transport):
        kwargs = {
            "n_observations": 3,
            "n_simulations": 2,
            "n_bootstrap": 1,
            "base_seed": 1,
            "transport": transport,
            "max_iterations": 1,
            "min_iterations": 0,
            "tolerance": 1e-6,
        }
        kwargs[name] = rank0 if transport.rank == 0 else rank1
        try:
            bootstrap_distributed(_model(), **kwargs)
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(match in message for message in LocalCluster(2).run(run))


def test_bootstrap_distributed_rejects_min_iterations_above_max() -> None:
    with pytest.raises(ValueError, match="min_iterations must be <= max_iterations"):
        bootstrap_distributed(
            _model(),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=1,
            base_seed=1,
            transport=SerialTransport(),
            max_iterations=2,
            min_iterations=3,
            tolerance=1e-6,
        )

    # min_iterations == max_iterations is a valid boundary: the guard is
    # strictly-greater, and the run does exactly that many iterations.
    result = bootstrap_distributed(
        _model(),
        n_observations=3,
        n_simulations=2,
        n_bootstrap=1,
        base_seed=1,
        transport=SerialTransport(),
        master_backend="highs",
        max_iterations=2,
        min_iterations=2,
        tolerance=1e-6,
    )
    assert result.iterations == 2

    # The default model certifies in 2 iterations (min=0, max=5 -> 2), so
    # iterations == 4 can only come from the public min_iterations floor being
    # threaded through to _run_replica_wave's convergence check.
    floored = bootstrap_distributed(
        _model(),
        n_observations=3,
        n_simulations=2,
        n_bootstrap=1,
        base_seed=1,
        transport=SerialTransport(),
        master_backend="highs",
        max_iterations=5,
        min_iterations=4,
        tolerance=1e-6,
    )
    assert floored.iterations == 4


def test_bootstrap_distributed_batches_master_state_and_sparse_u_routes() -> None:
    transport = CountingTransport(SerialTransport())
    result = bootstrap_distributed(
        _model(),
        n_observations=3,
        n_simulations=2,
        n_bootstrap=2,
        base_seed=1,
        transport=transport,
        master_backend="highs",
        max_live_reps=2,
        max_iterations=1,
        tolerance=1e9,
    )

    counts = transport.counts()
    assert result.iterations == 1
    assert counts.get("route_agent_values", 0) == 0
    assert counts["route_agent_values_batched"] == 2
    assert counts["owner_broadcast"] == 2


class _NoBootstrapHooksTransport(SerialTransport):
    route_agent_values_batched = Transport.route_agent_values_batched
    owner_broadcast = Transport.owner_broadcast


def test_bootstrap_distributed_requires_batched_transport_hooks() -> None:
    with pytest.raises(NotImplementedError, match="route_agent_values_batched"):
        bootstrap_distributed(
            _model(),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=1,
            base_seed=1,
            transport=_NoBootstrapHooksTransport(),
            master_backend="highs",
            max_iterations=1,
        )


def test_bootstrap_distributed_handles_empty_observation_and_agent_shards() -> None:
    def run(transport):
        result = bootstrap_distributed(
            _model(),
            n_observations=2,
            n_simulations=2,
            n_bootstrap=2,
            base_seed=1,
            transport=transport,
            master_backend="highs",
            max_live_reps=2,
            max_iterations=2,
            tolerance=1e9,
        )
        return result.thetas.shape, result.converged.tolist(), int(result.iterations)

    results = LocalCluster(5).run(run)

    for shape, converged, iterations in results:
        assert shape == (2, 1)
        assert converged == [True, True]
        assert iterations == 1


def test_bootstrap_distributed_requires_rank_uniform_formulation_support() -> None:
    def run(transport):
        try:
            bootstrap_distributed(
                _model(formulation=NSlack if transport.rank == 0 else OneSlack),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "model.formulation is NSlack must match" in message
        for message in LocalCluster(2).run(run)
    )


def test_bootstrap_distributed_requires_rank_uniform_dual_store_presence(
    monkeypatch, tmp_path
) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(bd, "resolve_master_backend", lambda *args, **kwargs: "highs")

    def run(transport):
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
                dual_store_dir=tmp_path if transport.rank == 0 else None,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "dual_store_dir is not None must match" in message
        for message in LocalCluster(2).run(run)
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("presence", "warm_start must match"),
        ("shape", "warm_start must match"),
        ("value", "warm_start must match"),
        ("access", "warm_start must match"),
        ("nonfinite", "warm_start.theta_hat must be finite"),
    ],
)
def test_bootstrap_distributed_requires_rank_uniform_warm_start(
    case: str, match: str
) -> None:
    def warm(theta):
        return SimpleNamespace(theta_hat=np.asarray(theta, dtype=np.float64))

    class _BrokenWarmStart:
        @property
        def theta_hat(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("theta boom")

    def run(transport):
        if case == "presence":
            warm_start = warm([0.0]) if transport.rank == 0 else None
        elif case == "shape":
            warm_start = warm([0.0] if transport.rank == 0 else [0.0, 0.0])
        elif case == "value":
            warm_start = warm([0.0] if transport.rank == 0 else [0.25])
        elif case == "access":
            warm_start = warm([0.0]) if transport.rank == 0 else _BrokenWarmStart()
        else:
            warm_start = warm([np.nan])
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
                warm_start=warm_start,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(match in message for message in LocalCluster(2).run(run))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("master_params", "master_params must match"),
        ("cut_policy", "cut_policy must match"),
        ("warm_cuts", "warm_cuts must match"),
    ],
)
def test_bootstrap_distributed_requires_rank_uniform_object_templates(
    case: str, match: str
) -> None:
    def run(transport):
        kwargs = {
            "master_params": None,
            "cut_policy": None,
            "warm_cuts": None,
        }
        if case == "master_params":
            kwargs["master_params"] = (
                {"presolve": "on"} if transport.rank == 0 else {"presolve": "off"}
            )
        elif case == "cut_policy":
            kwargs["cut_policy"] = (
                AddAll() if transport.rank == 0 else PurgeInactive(max_age=1)
            )
        else:
            kwargs["warm_cuts"] = (
                ()
                if transport.rank == 0
                else (
                    CutRow(
                        rep_id=0,
                        agent_id=0,
                        phi=np.array([1.0], dtype=np.float64),
                        epsilon=0.0,
                        bundle_key=b"b",
                    ),
                )
            )
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
                max_iterations=1,
                **kwargs,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(match in message for message in LocalCluster(2).run(run))


def test_bootstrap_distributed_rejects_unaudited_formulations() -> None:
    with pytest.raises(NotImplementedError, match="NSlack"):
        bootstrap_distributed(
            _model(formulation=OneSlack),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=1,
            base_seed=1,
            transport=SerialTransport(),
        )


def test_bootstrap_replicas_receive_warm_start_and_isolated_policy(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    # A real CutPolicy (e.g. PurgeInactive) carries per-(agent_id, bundle_key)
    # aging state in a nested mutable dict. Model that with `ages` so the deep
    # copy is observable: a shallow copy would share this dict across reps.
    class _Policy:
        def __init__(self) -> None:
            self.ages: dict[tuple[int, bytes], int] = {}

    warm_cuts = ()
    warm_start = SimpleNamespace(theta_hat=np.array([0.25], dtype=np.float64))
    policy = _Policy()

    def fake_build(rep_id, **kwargs):
        records.append(
            {
                "rep_id": rep_id,
                "theta_writeable": kwargs["theta_init"].flags.writeable,
                "theta_init": np.asarray(kwargs["theta_init"]).copy(),
                "warm_cuts": kwargs["warm_cuts"],
                "policy": kwargs["cut_policy"],
            }
        )
        return _fake_replica(rep_id, kwargs["prep"])

    bd = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(bd, "_build_distributed_replica", fake_build)
    monkeypatch.setattr(bd, "_nslack", lambda replica: replica.formulation)
    monkeypatch.setattr(bd, "_run_replica_wave", _replica_wave_result)

    bootstrap_distributed(
        _model(),
        n_observations=4,
        n_simulations=2,
        n_bootstrap=2,
        base_seed=7,
        transport=SerialTransport(),
        master_backend="highs",
        warm_start=warm_start,
        warm_cuts=warm_cuts,
        cut_policy=policy,
    )

    assert [r["rep_id"] for r in records] == [0, 1]
    for record in records:
        np.testing.assert_array_equal(record["theta_init"], warm_start.theta_hat)
        assert record["theta_writeable"] is False
        assert record["warm_cuts"] is warm_cuts
        assert record["policy"] is not policy
    p0, p1 = records[0]["policy"], records[1]["policy"]
    assert p0 is not p1
    assert p0.ages is not policy.ages
    assert p1.ages is not policy.ages
    assert p0.ages is not p1.ages
    # Mutation through one copy must not reach the other rep or the template.
    p0.ages[(0, b"k")] = 1
    assert p1.ages == {}
    assert policy.ages == {}


def test_bootstrap_distributed_callback_index_resets_by_live_wave() -> None:
    # max_live_reps=1 with n_bootstrap=2 runs two single-rep waves, each
    # certifying in two iterations. A per-wave index reports [0, 1] twice; a
    # running global counter would report [2, 3] in the second wave.
    calls: list[int] = []

    bootstrap_distributed(
        _model(),
        n_observations=3,
        n_simulations=2,
        n_bootstrap=2,
        base_seed=1,
        transport=SerialTransport(),
        max_live_reps=1,
        master_backend="highs",
        max_iterations=5,
        iteration_callback=lambda it, _oracle: calls.append(it) or None,
    )

    assert calls == [0, 1, 0, 1]


def test_store_dual_restamps_and_skips_none(tmp_path) -> None:
    writer = DualStoreWriter(tmp_path)
    dual = DualSolution(
        rep_id=0,
        agent_ids=np.array([2], dtype=np.int64),
        bundle_row_ids=np.array([0], dtype=np.int64),
        pis=np.array([1.5], dtype=np.float64),
        bundle_table=np.array([[1.0, 0.0]], dtype=np.float64),
        bound_duals={},
    )

    assert _store_dual(writer, 7, None) == 0
    assert _store_dual(writer, 7, dual) == 1

    reader = DualStoreReader(tmp_path)
    assert reader.rep_ids() == (7,)
    loaded = reader.load(7)
    assert loaded.rep_id == 7
    np.testing.assert_array_equal(loaded.agent_ids, dual.agent_ids)
    np.testing.assert_array_equal(loaded.bundle_table, dual.bundle_table)


def test_packed_master_state_round_trips_and_rejects_missing_sentinel() -> None:
    packet = SimpleNamespace(
        theta=np.array([1.0, -2.0], dtype=np.float64),
        objective=3.5,
        n_installed=7,
        progressed=2,
    )

    row = _pack_master_state(packet, K=2)
    state = _unpack_master_state(row, K=2)

    np.testing.assert_array_equal(state.theta, packet.theta)
    assert state.objective == packet.objective
    assert state.n_installed == packet.n_installed
    assert state.progressed == packet.progressed

    missing = row.copy()
    missing[-1] = 0.0
    with pytest.raises(RuntimeError, match="missing owner master-state"):
        _unpack_master_state(missing, K=2)
    with pytest.raises(ValueError, match="master-state row has shape"):
        _unpack_master_state(row[:-1], K=2)


def test_bootstrap_distributed_dual_store_failure_is_rank_agreed(
    monkeypatch, tmp_path
) -> None:
    def fail_write(self, dual):  # type: ignore[no-untyped-def]
        raise RuntimeError("store boom")

    monkeypatch.setattr(DualStoreWriter, "write", fail_write)

    def run(transport):
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=1,
                base_seed=1,
                transport=transport,
                master_backend="highs",
                max_iterations=1,
                tolerance=1e9,
                dual_store_dir=tmp_path,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all("store boom" in message for message in LocalCluster(2).run(run))


def test_bootstrap_distributed_dual_store_receives_nonroot_owner_dual(
    tmp_path,
) -> None:
    def run(transport):
        owners = tuple(_owner_vector(2, transport).tolist())
        result = bootstrap_distributed(
            _model(),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=2,
            base_seed=1,
            transport=transport,
            master_backend="highs",
            max_live_reps=2,
            max_iterations=1,
            tolerance=1e9,
            dual_store_dir=tmp_path,
        )
        return owners, result.n_duals_stored

    outcomes = LocalCluster(2).run(run)

    assert outcomes == [((0, 1), 2), ((0, 1), 2)]
    reader = DualStoreReader(tmp_path)
    assert reader.rep_ids() == (0, 1)
    for rep_id in (0, 1):
        assert reader.load(rep_id).rep_id == rep_id


def test_bootstrap_distributed_owner_master_step_failure_is_rank_agreed(
    monkeypatch,
) -> None:
    original = NSlack._owner_install_step

    def fail_one_owner(self, received):  # type: ignore[no-untyped-def]
        if self._is_owner and self._owner_rank == 1:
            raise RuntimeError("owner boom")
        return original(self, received)

    monkeypatch.setattr(NSlack, "_owner_install_step", fail_one_owner)

    def run(transport):
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=2,
                base_seed=1,
                transport=transport,
                master_backend="highs",
                max_live_reps=2,
                max_iterations=1,
                tolerance=1e9,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all("owner boom" in message for message in LocalCluster(2).run(run))


class _CloseFailingMaster:
    def close(self) -> None:
        raise RuntimeError("close boom")


def test_replica_wave_teardown_close_failure_is_rank_agreed_after_dual_count_bcast(
    tmp_path,
) -> None:
    def run(transport):
        oracle = _CallbackOracle()
        resolution = _price_resolution(oracle, transport)
        replicas = [
            _Replica(
                rep_id=0,
                formulation=_ConvergingNslackFormulation((0.0,)),
                price_resolution=resolution,
                scheduled_local_ids=np.array([transport.rank], dtype=np.int64),
                master_backend=(
                    _CloseFailingMaster() if transport.rank == 0 else None
                ),
            )
        ]
        try:
            _run_replica_wave(
                replicas,
                oracle=oracle,
                transport=transport,
                owners=np.array([0], dtype=np.int64),
                K=1,
                parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
                tolerance=1e9,
                max_iterations=1,
                dual_store_dir=tmp_path,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all("close boom" in message for message in LocalCluster(2).run(run))


class _CallbackOracle(Oracle):
    def __init__(self) -> None:
        self.calls: list[int] = []

    def price(self, theta, agent_id):  # type: ignore[no-untyped-def]
        self.calls.append(int(agent_id))
        return Demand.exact(
            bundle=np.array([int(agent_id)], dtype=np.float64),
            payoff=0.0,
        )


class _RankFailingPriceOracle(Oracle):
    def price(self, theta, agent_id):  # type: ignore[no-untyped-def]
        if int(agent_id) == 1:
            raise RuntimeError("price boom")
        return Demand.exact(
            bundle=np.array([int(agent_id)], dtype=np.float64),
            payoff=0.0,
        )


class _ConvergingNslackFormulation(NSlack):
    def __init__(self, violations: tuple[float, ...]) -> None:
        self._violations = tuple(float(v) for v in violations)
        self.finalise_calls = 0
        self.apply_calls = 0
        self._owner_u: dict[int, float] = {}

    def solve(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)

    def contribute(self, demands):  # type: ignore[no-untyped-def]
        return MaxContribution(worst=0.0, local_rows=())

    def finalise(self, reduced):  # type: ignore[no-untyped-def]
        idx = min(self.finalise_calls, len(self._violations) - 1)
        self.finalise_calls += 1
        return StepOutcome(
            violation=self._violations[idx],
            install_payload=None,
        )

    def _distributed_route_spec(self) -> tuple[np.ndarray, int]:
        return np.empty(0, dtype=np.int64), 1

    def _distributed_set_owner_u(self, values) -> None:  # type: ignore[no-untyped-def]
        self._owner_u = {} if values is None else dict(values)

    def _distributed_owner_u(self) -> dict[int, float]:
        return self._owner_u

    def _distributed_apply_owner_step(self, install_payload):  # type: ignore[no-untyped-def]
        self.apply_calls += 1
        return (
            SimpleNamespace(
                theta=np.zeros(1, dtype=np.float64),
                objective=0.0,
                n_installed=0,
                progressed=0,
            ),
            {},
        )

    def _adopt_owner_state(
        self,
        state,
        *,
        local_u,
        full_u,
        bump_iteration: bool,
    ) -> int:
        return int(state.progressed)

    def result(self) -> FormulationResult:
        return FormulationResult(
            theta_hat=np.array([float(self.finalise_calls)], dtype=np.float64),
            objective=0.0,
            n_active_cuts=0,
        )

    def dispose(self) -> None:
        pass


class _PlainFormulation:
    def dispose(self) -> None:
        pass


def test_replica_wave_rejects_non_nslack_replicas() -> None:
    oracle = _CallbackOracle()
    resolution = _price_resolution(oracle, SerialTransport())
    replicas = [
        _Replica(
            rep_id=0,
            formulation=_PlainFormulation(),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0], dtype=np.int64),
        )
    ]

    with pytest.raises(TransportError, match="NSlack replicas only"):
        _run_replica_wave(
            replicas,
            oracle=oracle,
            transport=SerialTransport(),
            owners=np.array([0], dtype=np.int64),
            K=1,
            parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
            tolerance=1e-9,
            max_iterations=1,
        )


def test_replica_wave_iteration_callback_is_once_per_wave_iteration() -> None:
    oracle = _CallbackOracle()
    resolution = _price_resolution(oracle, SerialTransport())
    calls: list[int] = []
    replicas = [
        _Replica(
            rep_id=0,
            formulation=_ConvergingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
        _Replica(
            rep_id=1,
            formulation=_ConvergingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
    ]

    result = _run_replica_wave(
        replicas,
        oracle=oracle,
        transport=SerialTransport(),
        owners=np.array([0, 0], dtype=np.int64),
        K=1,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        tolerance=1e-9,
        max_iterations=5,
        min_iterations=0,
        iteration_callback=lambda it, _oracle: calls.append(it) or 2,
    )

    assert calls == [0, 1]
    assert result.converged.tolist() == [True, True]
    assert result.iterations == 2


def test_replica_wave_chunks_cut_exchange_by_live_rep_blocks(monkeypatch) -> None:
    boot_module = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(boot_module, "_CUT_EXCHANGE_BLOCK_ELEMENTS", 2)
    oracle = _CallbackOracle()
    transport = _ReduceShapeTransport()
    resolution = _price_resolution(oracle, transport)

    class _RowContributingNslackFormulation(_ConvergingNslackFormulation):
        def contribute(self, demands):  # type: ignore[no-untyped-def]
            return MaxContribution(
                worst=1.0,
                local_rows=(
                    CutRow(
                        rep_id=0,
                        agent_id=0,
                        phi=np.ones(1, dtype=np.float64),
                        epsilon=0.0,
                        bundle_key=b"k",
                    ),
                ),
            )

    replicas = [
        _Replica(
            rep_id=0,
            formulation=_RowContributingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
        _Replica(
            rep_id=1,
            formulation=_RowContributingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
    ]

    result = _run_replica_wave(
        replicas,
        oracle=oracle,
        transport=transport,
        owners=np.array([0, 0], dtype=np.int64),
        K=1,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        tolerance=1e-9,
        max_iterations=1,
    )

    assert result.converged.tolist() == [True, True]
    # One batched max per single-slot block, plus the row-width lane that
    # rides it in place of a separate allreduce.
    assert transport.max_shapes == [(2,), (2,)]
    assert transport.exchange_owner_shapes == [(1,), (1,)]


def test_replica_wave_disposes_retired_replicas_before_slow_reps_finish() -> None:
    events: list[tuple[str, int, int]] = []

    class _TrackedFormulation(_ConvergingNslackFormulation):
        def __init__(self, slot: int, violations: tuple[float, ...]) -> None:
            super().__init__(violations)
            self.slot = int(slot)

        def finalise(self, reduced):  # type: ignore[no-untyped-def]
            outcome = super().finalise(reduced)
            events.append(("finalise", self.slot, self.finalise_calls))
            return outcome

        def result(self) -> FormulationResult:
            events.append(("result", self.slot, self.finalise_calls))
            return super().result()

        def dispose(self) -> None:
            events.append(("dispose", self.slot, self.finalise_calls))

    oracle = _CallbackOracle()
    transport = SerialTransport()
    resolution = _price_resolution(oracle, transport)
    replicas = [
        _Replica(
            rep_id=0,
            formulation=_TrackedFormulation(0, (0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0], dtype=np.int64),
        ),
        _Replica(
            rep_id=1,
            formulation=_TrackedFormulation(1, (1.0, 1.0, 0.0)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0], dtype=np.int64),
        ),
    ]

    result = _run_replica_wave(
        replicas,
        oracle=oracle,
        transport=transport,
        owners=np.array([0, 0], dtype=np.int64),
        K=1,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        tolerance=1e-9,
        max_iterations=3,
    )

    assert result.converged.tolist() == [True, True]
    assert result.thetas[:, 0].tolist() == [1.0, 3.0]
    assert events.index(("dispose", 0, 1)) < events.index(("finalise", 1, 2))


def test_replica_wave_batches_no_row_wave_without_slot_probing() -> None:
    oracle = _CallbackOracle()
    transport = _ReduceShapeTransport()
    resolution = _price_resolution(oracle, transport)
    replicas = [
        _Replica(
            rep_id=rep,
            formulation=_ConvergingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        )
        for rep in range(3)
    ]

    result = _run_replica_wave(
        replicas,
        oracle=oracle,
        transport=transport,
        owners=np.array([0, 0, 0], dtype=np.int64),
        K=1,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        tolerance=1e-9,
        max_iterations=1,
    )

    assert result.converged.tolist() == [True, True, True]
    assert [replica.formulation.finalise_calls for replica in replicas] == [1, 1, 1]
    # All three live slots reduce in one round; the extra lane is the
    # row-width bookkeeping riding the same batched max.
    assert transport.max_shapes == [(4,)]
    assert transport.exchange_owner_shapes == [(3,)]


def test_replica_wave_does_not_reprocess_no_row_prefix(monkeypatch) -> None:
    boot_module = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(boot_module, "_CUT_EXCHANGE_BLOCK_ELEMENTS", 6)
    oracle = _CallbackOracle()
    transport = _ReduceShapeTransport()
    resolution = _price_resolution(oracle, transport)

    class _WideRowFormulation(_ConvergingNslackFormulation):
        def contribute(self, demands):  # type: ignore[no-untyped-def]
            return MaxContribution(
                worst=1.0,
                local_rows=(
                    CutRow(
                        rep_id=0,
                        agent_id=0,
                        phi=np.ones(1, dtype=np.float64),
                        epsilon=0.0,
                        bundle_key=b"x" * 256,
                    ),
                ),
            )

    replicas = [
        _Replica(
            rep_id=0,
            formulation=_ConvergingNslackFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
        _Replica(
            rep_id=1,
            formulation=_WideRowFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
        _Replica(
            rep_id=2,
            formulation=_WideRowFormulation((0.0,)),
            price_resolution=resolution,
            scheduled_local_ids=np.array([0, 1], dtype=np.int64),
        ),
    ]

    result = _run_replica_wave(
        replicas,
        oracle=oracle,
        transport=transport,
        owners=np.array([0, 0, 0], dtype=np.int64),
        K=1,
        parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
        tolerance=1e-9,
        max_iterations=1,
    )

    assert result.converged.tolist() == [True, True, True]
    assert [replica.formulation.finalise_calls for replica in replicas] == [1, 1, 1]


def test_cut_exchange_block_size_uses_cut_row_bytes(monkeypatch) -> None:
    boot_module = importlib.import_module("combrum.bootstrap_distributed")
    monkeypatch.setattr(boot_module, "_CUT_EXCHANGE_BLOCK_ELEMENTS", 4)

    small = _cut_exchange_block_size(3, row_nbytes=8, max_scheduled_ids=2)
    wide = _cut_exchange_block_size(3, row_nbytes=128, max_scheduled_ids=2)

    assert small == 2
    assert wide == 1


def test_observed_cut_row_nbytes_includes_bundle_key_bytes() -> None:
    short = CutRow(
        rep_id=0,
        agent_id=0,
        phi=np.ones(2, dtype=np.float64),
        epsilon=0.0,
        bundle_key=b"k",
    )
    wide = CutRow(
        rep_id=0,
        agent_id=1,
        phi=np.ones(2, dtype=np.float64),
        epsilon=0.0,
        bundle_key=b"x" * 257,
    )

    assert _observed_cut_row_nbytes(
        [short, wide], transport=SerialTransport()
    ) == _cut_row_nbytes(wide)
    assert _observed_cut_row_nbytes([], transport=SerialTransport()) is None


def test_replica_wave_uses_max_of_min_iterations_and_callback_floor() -> None:
    # floor = max(min_iterations, callback_floor). The formulation converges on
    # iteration 0 (violation 0.0 <= tol), so iterations equals the floor and
    # the callback sees 0..floor-1.
    def run(*, min_iterations: int, callback_floor: int) -> tuple[list[int], int, list[bool]]:
        oracle = _CallbackOracle()
        resolution = _price_resolution(oracle, SerialTransport())
        calls: list[int] = []
        replicas = [
            _Replica(
                rep_id=0,
                formulation=_ConvergingNslackFormulation((0.0,)),
                price_resolution=resolution,
                scheduled_local_ids=np.array([0], dtype=np.int64),
            )
        ]
        result = _run_replica_wave(
            replicas,
            oracle=oracle,
            transport=SerialTransport(),
            owners=np.array([0], dtype=np.int64),
            K=1,
            parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
            tolerance=1e-9,
            max_iterations=5,
            min_iterations=min_iterations,
            iteration_callback=lambda it, _oracle: calls.append(it) or callback_floor,
        )
        return calls, int(result.iterations), result.converged.tolist()

    # Callback floor dominates: max(1, 3) = 3.
    calls, iterations, converged = run(min_iterations=1, callback_floor=3)
    assert calls == [0, 1, 2]
    assert iterations == 3
    assert converged == [True]

    # min_iterations dominates: max(3, 1) = 3.
    calls, iterations, converged = run(min_iterations=3, callback_floor=1)
    assert calls == [0, 1, 2]
    assert iterations == 3
    assert converged == [True]


def test_replica_wave_uses_rank_zero_callback_floor_for_live_mask() -> None:
    def run(transport):
        oracle = _CallbackOracle()
        resolution = _price_resolution(oracle, transport)
        result = _run_replica_wave(
            [
                _Replica(
                    rep_id=0,
                    formulation=_ConvergingNslackFormulation((0.0, 0.0, 0.0)),
                    price_resolution=resolution,
                    scheduled_local_ids=np.array([transport.rank], dtype=np.int64),
                )
            ],
            oracle=oracle,
            transport=transport,
            owners=np.array([0], dtype=np.int64),
            K=1,
            parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
            tolerance=1e-9,
            max_iterations=3,
            min_iterations=0,
            iteration_callback=(
                lambda _it, _oracle: 0 if transport.rank == 0 else 2
            ),
        )
        return int(result.iterations)

    assert LocalCluster(2).run(run) == [1, 1]


def test_replica_wave_propagates_rank_local_callback_failure() -> None:
    def run(transport):
        oracle = _CallbackOracle()
        resolution = _price_resolution(oracle, transport)

        def callback(_it, _oracle):  # type: ignore[no-untyped-def]
            if transport.rank == 1:
                raise RuntimeError("callback boom")
            return None

        try:
            _run_replica_wave(
                [
                    _Replica(
                        rep_id=0,
                        formulation=_ConvergingNslackFormulation((0.0,)),
                        price_resolution=resolution,
                        scheduled_local_ids=np.array(
                            [transport.rank], dtype=np.int64
                        ),
                    )
                ],
                oracle=oracle,
                transport=transport,
                owners=np.array([0], dtype=np.int64),
                K=1,
                parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
                tolerance=1e-9,
                max_iterations=1,
                min_iterations=0,
                iteration_callback=callback,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "callback boom" in message for message in LocalCluster(2).run(run)
    )


def test_replica_wave_propagates_rank_local_pricing_failure_before_reduce() -> None:
    def run(transport):
        oracle = _RankFailingPriceOracle()
        resolution = _price_resolution(oracle, transport)
        try:
            _run_replica_wave(
                [
                    _Replica(
                        rep_id=0,
                        formulation=_ConvergingNslackFormulation((0.0,)),
                        price_resolution=resolution,
                        scheduled_local_ids=np.array(
                            [transport.rank], dtype=np.int64
                        ),
                    )
                ],
                oracle=oracle,
                transport=transport,
                owners=np.array([0], dtype=np.int64),
                K=1,
                parameters=Parameters({"theta": (-1.0, 1.0, 1)}),
                tolerance=1e-9,
                max_iterations=1,
                min_iterations=0,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all("price boom" in message for message in LocalCluster(2).run(run))


def test_batched_reduce_max_kind_routes_rows_by_rep_to_owner() -> None:
    # owners=[0,1]: slot 0's rows must land only on rank 0, slot 1's only on
    # rank 1, restamped from the caller's slot (contribute builds them under
    # rep_id=0). Each slot carries one distinct row; phi[0] tags the source rank
    # so both ranks' copies are observable, agent_id tags the slot.
    owners = np.array([0, 1], dtype=np.int64)

    def per_rank(transport: Transport):
        r = float(transport.rank)
        contribution = {
            0: MaxContribution(
                worst=r,
                local_rows=(
                    CutRow(
                        rep_id=0,
                        agent_id=10,
                        phi=np.array([r], dtype=np.float64),
                        epsilon=0.5,
                        bundle_key=b"s0",
                    ),
                ),
            ),
            1: MaxContribution(
                worst=3.0,
                local_rows=(
                    CutRow(
                        rep_id=0,
                        agent_id=20,
                        phi=np.array([r + 100.0], dtype=np.float64),
                        epsilon=0.7,
                        bundle_key=b"s1",
                    ),
                ),
            ),
        }
        reduced = batched_reduce(transport, contribution, owners=owners)
        per_slot = []
        for red in reduced:
            assert isinstance(red, MaxReduced)
            per_slot.append(
                (
                    red.global_worst,
                    tuple(
                        (row.rep_id, row.agent_id, float(row.phi[0]), row.epsilon)
                        for row in red.received_rows
                    ),
                )
            )
        return per_slot

    got = LocalCluster(2).run(per_rank)

    # Slot 0 lands on rank 0 with both ranks' rows restamped rep_id=0
    # (canonical (rep_id, agent_id, bundle_key) order puts rank-0's phi=0.0
    # before rank-1's phi=1.0); slot 1 lands on rank 1 restamped rep_id=1; the
    # non-owning rank sees an empty tuple. worsts: slot 0 = max(0.0, 1.0),
    # slot 1 = max(3.0, 3.0).
    slot0_rows = ((0, 10, 0.0, 0.5), (0, 10, 1.0, 0.5))
    slot1_rows = ((1, 20, 100.0, 0.7), (1, 20, 101.0, 0.7))
    rank0 = [(1.0, slot0_rows), (3.0, ())]
    rank1 = [(1.0, ()), (3.0, slot1_rows)]
    assert got == [rank0, rank1]


def test_batched_reduce_compacts_retired_slots_before_communication() -> None:
    owners = np.array([0, 0, 0], dtype=np.int64)
    transport = _ReduceShapeTransport()
    contribution = {
        2: MaxContribution(
            worst=7.0,
            local_rows=(
                CutRow(
                    rep_id=0,
                    agent_id=4,
                    phi=np.array([1.0], dtype=np.float64),
                    epsilon=0.5,
                    bundle_key=b"s2",
                ),
            ),
        )
    }

    reduced = batched_reduce(transport, contribution, owners=owners)

    assert transport.max_shapes == [(1,)]
    assert transport.exchange_owner_shapes == [(1,)]
    assert reduced[0] is None
    assert reduced[1] is None
    assert isinstance(reduced[2], MaxReduced)
    assert reduced[2].global_worst == 7.0
    assert tuple(row.rep_id for row in reduced[2].received_rows) == (2,)


def test_owner_vector_distributes_reps_across_nodes_first() -> None:
    owners_by_rank = LocalCluster(6, ranks_per_node=2).run(
        lambda transport: _owner_vector(8, transport)
    )
    expected = np.array([0, 2, 4, 1, 3, 5, 0, 2], dtype=np.int64)
    for owners in owners_by_rank:
        np.testing.assert_array_equal(owners, expected)


class _SetupRaisingNSlack(NSlack):
    def setup(self, ctx):  # type: ignore[no-untyped-def]
        raise RuntimeError("setup boom")


class _CloseCountingMaster:
    def __init__(self) -> None:
        self.closed = 0

    def reinstall(self, cuts):  # type: ignore[no-untyped-def]
        pass

    def close(self) -> None:
        self.closed += 1


def test_build_distributed_replica_closes_master_when_setup_raises(
    monkeypatch,
) -> None:
    fake_master = _CloseCountingMaster()

    bd = importlib.import_module("combrum.bootstrap_distributed")
    dc = importlib.import_module("combrum.engine.distributed_context")
    monkeypatch.setattr(dc, "make_master", lambda *args, **kwargs: fake_master)

    oracle = _RecordingOracle()
    model = _model(oracle=oracle, formulation=_SetupRaisingNSlack)
    transport = SerialTransport()
    prep = prepare_distributed_observed(
        model,
        n_observations=3,
        n_simulations=2,
        transport=transport,
    )
    resolution = _price_resolution(oracle, transport)

    with pytest.raises(RuntimeError, match="setup boom"):
        bd._build_distributed_replica(
            0,
            prep=prep,
            model=model,
            c_theta=np.zeros(1, dtype=np.float64),
            normalizer=float(prep.N),
            price_resolution=resolution,
            transport=transport,
            owner_rank=0,
            backend="highs",
            master_params=None,
            tolerance=1e-6,
            cut_policy=None,
            result_publication=ResultPublication.SUMMARY,
            theta_init=None,
            warm_cuts=None,
            base_seed=1,
        )

    # The master is created before formulation.setup, so a setup failure must
    # close it before re-raising to avoid a leak.
    assert fake_master.closed == 1


def test_bootstrap_distributed_disposes_replicas_when_initial_publish_raises(
    monkeypatch,
) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")
    real_dispose = bd._dispose_replicas
    disposed: list[int] = []

    def record_dispose(replicas):  # type: ignore[no-untyped-def]
        disposed.append(len(replicas))
        real_dispose(replicas)

    def fail_publish(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("publish boom")

    monkeypatch.setattr(bd, "_dispose_replicas", record_dispose)
    monkeypatch.setattr(bd, "_publish_nslack_states", fail_publish)

    with pytest.raises(RuntimeError, match="publish boom"):
        bootstrap_distributed(
            _model(),
            n_observations=3,
            n_simulations=2,
            n_bootstrap=2,
            base_seed=1,
            transport=SerialTransport(),
            master_backend="highs",
            max_live_reps=2,
            max_iterations=1,
        )

    assert disposed == [2]


def test_bootstrap_distributed_initial_publish_failure_is_rank_agreed_and_disposes(
    monkeypatch,
) -> None:
    bd = importlib.import_module("combrum.bootstrap_distributed")
    real_dispose = bd._dispose_replicas
    real_pack = bd._pack_master_state
    thread_rank = threading.local()
    lock = threading.Lock()
    disposed: list[tuple[int, int]] = []

    def record_dispose(replicas):  # type: ignore[no-untyped-def]
        with lock:
            disposed.append((int(thread_rank.rank), len(replicas)))
        real_dispose(replicas)

    def fail_on_rank_one(state, K):  # type: ignore[no-untyped-def]
        if int(thread_rank.rank) == 1:
            raise RuntimeError("publish boom")
        return real_pack(state, K)

    monkeypatch.setattr(bd, "_dispose_replicas", record_dispose)
    monkeypatch.setattr(bd, "_pack_master_state", fail_on_rank_one)

    def run(transport):
        thread_rank.rank = transport.rank
        try:
            bootstrap_distributed(
                _model(),
                n_observations=3,
                n_simulations=2,
                n_bootstrap=2,
                base_seed=1,
                transport=transport,
                master_backend="highs",
                max_live_reps=2,
                max_iterations=1,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all("publish boom" in message for message in LocalCluster(2).run(run))
    assert sorted(disposed) == [(0, 2), (1, 2)]
