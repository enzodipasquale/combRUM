"""mpirun-driven conformance of MpiTransport against the references.

Each parametrized launch shells out ``mpirun -n N`` on the rank program
(``tests/_mpi/conformance_core.py``), which prints one JSON record per
rank; replaying the identical battery on the in-process references must
reproduce those records exactly. All payloads are digests or bit-hex
strings, so record equality is bitwise equality of results. The launch
carries a hard timeout so a guard regression that hangs fails loudly instead of
wedging the suite.

The single-host launch puts every rank in one shared-memory domain,
which is exactly the references' default topology (one node holding all
ranks), so topology records compare directly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from _support.families import load_family
from _support.skeleton import run_skeleton
from combrum.masters import highs as highs_backend
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.mpi import MpiTransport

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_PATH = _TESTS_DIR / "_mpi" / "conformance_core.py"
_SRC_DIR = _TESTS_DIR.parent / "src"
_LAUNCH_TIMEOUT_S = 120.0
_SIZES = (1, 2, 4, 8)
_MIB = 2**20

needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)


def _expected_counts(rank: int, size: int) -> dict[str, dict[str, int]]:
    """The pinned round shape of every counted scenario, per rank."""
    expected: dict[str, dict[str, int]] = {
        # Windowed gather-to-root + bcast: one small typed allgather agrees the
        # per-rank layout/id span (O(size)); each bounded id window gathers row
        # counts, values, and ids, then broadcasts a root-side validation status;
        # one final Bcast returns the small result. The fixture is one window.
        "sum_reproducible": {"allgather": 1, "gather": 3, "bcast": 2},
        # One shape agreement and one rank-vector payload allgather. Payload
        # is O(size * M), independent of any row count collapsed before entry.
        "sum_vectors_reproducible": {"allgather": 2},
        # One shape agreement, one fixed-size owner-vector fingerprint, and one
        # owner-row allgather. Only owner rows are sent.
        "owner_broadcast": {"allreduce": 2, "allgather": 1},
        # One fixed preflight agreement, one sparse destination exchange, and
        # one fixed receive-side membership agreement. No dense agent vector
        # is ever broadcast or gathered.
        "route_agent_values": {
            "allreduce": 2,
            "alltoall": 1,
            "alltoallv": 1,
        },
        # Same round shape as route_agent_values, but the sparse payload is
        # keyed by live replication id as well as global agent id, with one
        # extra fixed-size owner-vector fingerprint.
        "route_agent_values_batched": {
            "allreduce": 3,
            "alltoall": 1,
            "alltoallv": 1,
        },
        # One shape agreement, one fixed-size owner-vector fingerprint, plus the
        # two-round super-step: byte counts (Alltoall), then packed payload (Alltoallv) —
        # constant no matter how many replications or rows are live.
        "exchange_cuts": {"allreduce": 2, "alltoall": 1, "alltoallv": 1},
        # The guard's steady-state cost: one word-sized allreduce.
        "guard_success": {"allreduce": 1},
        # Failure adds only the verdict broadcast.
        "guard_failure": {"allreduce": 1, "bcast": 1},
    }
    # Chunked scatter: one ids gather, one verdict broadcast, then one
    # point-to-point chunk per (destination, key) — the tiny payload is
    # the single-window case, and the root sends nothing to itself.
    scatter: dict[str, int] = {"gather": 1, "bcast": 1}
    if rank == 0:
        if size > 1:
            scatter["p2p_send"] = 2 * (size - 1)
    else:
        scatter["p2p_recv"] = 2
    expected["scatter_small"] = scatter
    # node_shared: one global verdict allgather, one node-local layout
    # broadcast, then exactly one window — allocate, lifetime lock, and
    # the sync-barrier-sync publication episode.
    expected["node_shared_small"] = {
        "allgather": 1,
        "bcast": 1,
        "win_allocate_shared": 1,
        "win_lock_all": 1,
        "win_sync": 2,
        "barrier": 1,
    }
    return expected


def _load_core() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "combrum_mpi_conformance_core", _CORE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_core()


def _normalize(obj: Any) -> Any:
    """JSON round-trip: tuples become lists, exactly as on the wire."""
    return json.loads(json.dumps(obj))


def _launch_records(
    mpirun_path: str, n: int, *args: str
) -> list[dict[str, Any]]:
    """Run the rank program under ``mpirun -n n`` and parse rank records."""
    cmd = [
        mpirun_path,
        "-n",
        str(n),
        "--oversubscribe",
        sys.executable,
        str(_CORE_PATH),
        *args,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=_LAUNCH_TIMEOUT_S
    )
    assert proc.returncode == 0, (
        f"mpirun -n {n} {' '.join(args)} failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    records: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue  # launcher noise is not a conformance record
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "rank" in obj:
            records.append(obj)
    assert sorted(rec["rank"] for rec in records) == list(range(n)), (
        f"expected one record per rank under mpirun -n {n};"
        f" got {len(records)}:\n{proc.stdout}"
    )
    records.sort(key=lambda rec: rec["rank"])
    assert all(rec["size"] == n for rec in records)
    return records


@pytest.fixture(scope="module", params=_SIZES, ids=lambda n: f"n{n}")
def launch(
    request: pytest.FixtureRequest, mpirun_path: str
) -> tuple[int, list[dict[str, Any]]]:
    """One mpirun launch per rank count, shared by every assertion on it."""
    n = request.param
    return n, _launch_records(mpirun_path, n)


@pytest.mark.requires_mpi
def test_every_rank_matches_the_reference_battery(
    launch: tuple[int, list[dict[str, Any]]],
) -> None:
    n, records = launch
    expected = [
        _normalize(result) for result in LocalCluster(n).run(_core.battery)
    ]
    got = [rec["results"] for rec in records]
    assert got == expected


@pytest.mark.requires_mpi
def test_rank_invariant_results_identical_on_every_rank(
    launch: tuple[int, list[dict[str, Any]]],
) -> None:
    n, records = launch
    # A one-element set is length-1 for any value the single rank produces,
    # so "identical across ranks" is unverifiable below two ranks. The n1
    # launch's invariant paths are pinned by the reference battery instead
    # (test_every_rank_matches_the_reference_battery[n1]); skip here rather
    # than pass vacuously.
    if n < 2:
        pytest.skip("cross-rank identity needs at least two ranks")
    for key in _core.INVARIANT_KEYS:
        rendered = {
            json.dumps(rec["results"][key], sort_keys=True) for rec in records
        }
        assert len(rendered) == 1, f"{key} differs across ranks: {rendered}"


@pytest.mark.requires_mpi
def test_pooled_sums_match_the_serial_reference(
    launch: tuple[int, list[dict[str, Any]]],
) -> None:
    # The serial transport computes each sum from the undivided tables; any
    # rank count and any sharding must reproduce those exact bytes.
    _, records = launch
    serial = _normalize(_core.battery(SerialTransport()))
    for rec in records:
        for key in _core.POOLED_SUM_KEYS:
            assert rec["results"][key] == serial[key], key


@pytest.mark.requires_mpi
def test_internal_counters_pin_the_round_shape(
    launch: tuple[int, list[dict[str, Any]]],
) -> None:
    n, records = launch
    for rec in records:
        assert rec["counters"] == _expected_counts(rec["rank"], n)


def test_mpi_module_import_is_lazy(tmp_path: Path) -> None:
    # The package walk imports every module wherever the suite runs, so
    # this module must import without mpi4py. A fresh interpreter asserts
    # the import pulled in no mpi4py; a neutral cwd keeps resolution from
    # leaning on the repository root.
    probe = (
        "import sys\n"
        "import combrum.transport.mpi\n"
        "polluted = [m for m in sys.modules\n"
        "            if m == 'mpi4py' or m.startswith('mpi4py.')]\n"
        "assert not polluted, polluted\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_SRC_DIR))
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )


def test_instantiation_without_mpi4py_raises_actionable_error(
    tmp_path: Path,
) -> None:
    # A None entry in sys.modules makes 'from mpi4py import MPI' raise
    # ImportError — the absent-extra situation, simulated without
    # uninstalling anything.
    probe = (
        "import sys\n"
        "sys.modules['mpi4py'] = None\n"
        "from combrum.transport.mpi import MpiTransport\n"
        "try:\n"
        "    MpiTransport()\n"
        "except ModuleNotFoundError as exc:\n"
        "    text = str(exc)\n"
        "    assert 'mpi4py' in text and 'combrum[mpi]' in text, text\n"
        "else:\n"
        "    raise AssertionError('MpiTransport() must fail without mpi4py')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_SRC_DIR))
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )


def test_mpi_transport_overrides_every_abstract_member() -> None:
    assert MpiTransport.__abstractmethods__ == frozenset()


@pytest.mark.requires_mpi
def test_exchange_cuts_bad_row_validation_is_rank_agreed(
    mpirun_path: str,
) -> None:
    records = _launch_records(mpirun_path, 3, "exchange-bad-row")
    outcomes = [rec["outcome"] for rec in records]
    assert [outcome[0] for outcome in outcomes] == ["caught"] * 3
    assert {outcome[1] for outcome in outcomes} == {1}
    assert all("exchange_cuts" in outcome[2] for outcome in outcomes)
    for rec in records:
        assert rec["counters"] == {"allreduce": 1, "bcast": 1}


@pytest.mark.requires_mpi
def test_route_agent_values_bad_value_validation_is_rank_agreed(
    mpirun_path: str,
) -> None:
    records = _launch_records(mpirun_path, 3, "route-bad-value")
    outcomes = [rec["outcome"] for rec in records]
    assert [outcome[0] for outcome in outcomes] == ["caught"] * 3
    assert all("route_agent_values" in outcome[1] for outcome in outcomes)
    assert all("float64" in outcome[1] for outcome in outcomes)
    for rec in records:
        assert rec["counters"] == {"allreduce": 1, "bcast": 1}


@pytest.mark.requires_mpi
def test_batched_max_shape_validation_precedes_native_reduce(
    mpirun_path: str,
) -> None:
    records = _launch_records(mpirun_path, 3, "batched-max-bad-shape")
    outcomes = [rec["outcome"] for rec in records]
    assert [outcome[0] for outcome in outcomes] == ["caught"] * 3
    assert all("same (B,)" in outcome[1] for outcome in outcomes)
    for rec in records:
        assert rec["counters"] == {"allreduce": 1}


@pytest.mark.requires_mpi
def test_sum_and_gather_reject_bad_shapes_before_collectives(
    mpirun_path: str,
) -> None:
    records = _launch_records(mpirun_path, 2, "sum-gather-bad-inputs")
    for rec in records:
        outcomes = rec["outcomes"]
        assert "global_ids must be a 1-D integer array" in outcomes["sum_float_ids"]
        assert "values must have shape (n,) or (n, M)" in outcomes["sum_3d_values"]
        assert (
            "global_ids must be a 1-D integer array"
            in outcomes["gather_float_ids"]
        )
        assert rec["counters"] == {}


# --- the walking-skeleton end-to-end over real MPI ---------------------------


@pytest.mark.requires_mpi
@pytest.mark.parametrize("family", ("toy", "qkp"))
@pytest.mark.parametrize("n", (1, 2, 4), ids=lambda n: f"n{n}")
def test_skeleton_end_to_end_matches_serial_bitwise(
    family: str, n: int, mpirun_path: str
) -> None:
    # The skeleton solved over MpiTransport at N ranks must reach exactly
    # the theta_hat a single serial rank reaches — the distributed walk
    # (chunked scatter + canonical reductions + one-super-step exchange +
    # guard agreement) composing to the same bytes is the end-to-end
    # determinism claim the skeleton exists to make over real MPI.
    fixtures = _TESTS_DIR / "fixtures" / "families"
    serial = run_skeleton(
        load_family(family, fixtures),
        SerialTransport(),
        family=family,
    )
    serial_hex = serial.theta_hat.tobytes().hex()
    records = _launch_records(mpirun_path, n, "skeleton-e2e", family)
    for rec in records:
        assert rec["family"] == family
        assert rec["theta_hat_hex"] == serial_hex, rec["rank"]
        assert rec["objective"] == repr(serial.objective)
        assert rec["n_active_cuts"] == serial.n_active_cuts
        assert rec["n_iterations"] == serial.metadata["n_iterations"]
        assert rec["best_total_regret"] == repr(0.0)


# --- the split-axis smoke's independent fixture-arithmetic oracle -----------

# The smoke fixture, restated here so the expected bytes derive from the
# fixture math rather than from any combrum reduction. Observed table row
# i is [i+1, 10+3i, 100+5i] over N observations; per-obs weights are the
# linspace the smoke passes to distributed_c_theta; S multiplies the c_theta
# rows; the bootstrap wave draws replications 2 and 3 at base seed 77.
_SPLIT_N, _SPLIT_S, _SPLIT_K = 11, 4, 3
_SPLIT_REP_IDS = (2, 3)
_SPLIT_BASE_SEED = 77
# splitmix64 constants of the counter-based multiplier RNG, reimplemented so
# the bootstrap oracle owes nothing to combrum.randomness.
_MASK64 = 0xFFFFFFFFFFFFFFFF
_BOOTSTRAP_NAMESPACE = 0xC0B2202606250001


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _multiplier(base_seed: int, rep_id: int, obs_id: int) -> float:
    """Independent redraw of one bootstrap multiplier from the RNG spec."""
    x = _BOOTSTRAP_NAMESPACE
    for field in (base_seed, rep_id, obs_id):
        x ^= field & _MASK64
        x = _splitmix64(x)
    u = ((x >> 11) & ((1 << 53) - 1)) * 2.0**-53
    return -math.log1p(-u)


def _split_axis_oracles() -> dict[str, str]:
    """Byte-exact expectations for every pooled key of split_axis_smoke.

    Each value is recomputed from the fixture arithmetic and (for the
    bootstrap) an independent reimplementation of the multiplier draw, then
    reduced in canonical ascending-observation-id order. The smoke's shards
    are contiguous and cover obs 0..N-1, so the pooled reduction is the
    plain ``add.reduce`` over rows in id order — no combrum reduction kernel
    or accessor is consulted, so a shared scaling regression on either
    MpiTransport or the LocalCluster reference is caught here even though it
    would leave the record equality and cross-rank identity intact.
    """
    obs = np.arange(_SPLIT_N)
    table = np.column_stack(
        [obs + 1.0, 10.0 + 3.0 * obs, 100.0 + 5.0 * obs]
    ).astype(np.float64)
    weights = np.linspace(0.5, 2.0, _SPLIT_N, dtype=np.float64)

    empirical = np.add.reduce(table, axis=0) / float(_SPLIT_N)
    c_theta = np.add.reduce(-float(_SPLIT_S) * (weights[:, None] * table), axis=0)

    boot = np.empty((len(_SPLIT_REP_IDS), _SPLIT_K), dtype=np.float64)
    norms = np.empty(len(_SPLIT_REP_IDS), dtype=np.float64)
    for slot, rep_id in enumerate(_SPLIT_REP_IDS):
        raw = np.array(
            [_multiplier(_SPLIT_BASE_SEED, rep_id, o) for o in range(_SPLIT_N)],
            dtype=np.float64,
        )
        norms[slot] = np.add.reduce(raw, axis=0)
        reduced_c = np.add.reduce(
            -float(_SPLIT_S) * (raw[:, None] * table), axis=0
        )
        boot[slot] = reduced_c * (float(_SPLIT_N) / norms[slot])

    return {
        "empirical_hex": empirical.tobytes().hex(),
        "c_theta_hex": c_theta.tobytes().hex(),
        "bootstrap_c_theta_hex": boot.reshape(-1).tobytes().hex(),
        "normalizers_hex": norms.tobytes().hex(),
    }


@pytest.mark.requires_mpi
def test_split_axis_observed_and_bootstrap_smoke_matches_localcluster(
    mpirun_path: str,
) -> None:
    n = 4
    records = _launch_records(mpirun_path, n, "split-axis-smoke")
    expected = [
        _normalize(result) for result in LocalCluster(n).run(_core.split_axis_smoke)
    ]

    assert [rec["results"] for rec in records] == expected
    observed_ids = sorted(
        obs_id
        for rec in records
        for obs_id in rec["results"]["owned_obs"]
    )
    local_ids = sorted(
        agent_id
        for rec in records
        for agent_id in rec["results"]["local_ids"]
    )
    assert observed_ids == list(range(11))
    assert local_ids == list(range(44))

    # Independent byte oracle for every pooled derived key. Each expected
    # hex is recomputed from the fixture arithmetic (and, for the bootstrap,
    # a from-spec redraw of the multiplier), reduced in canonical id order —
    # never from a combrum reduction or accessor. This pins the bytes to the
    # fixture, so a scaling regression that both MpiTransport and the
    # LocalCluster reference share — which leaves the record equality above
    # and cross-rank identity below both satisfied — is still caught. The
    # empirical column means are the sanity anchors: mean(1..11)=6,
    # mean(10+3i)=25, mean(100+5i)=125.
    oracles = _split_axis_oracles()
    assert oracles["empirical_hex"] == np.array(
        [6.0, 25.0, 125.0], dtype=np.float64
    ).tobytes().hex()
    for key, expected_hex in oracles.items():
        assert records[0]["results"][key] == expected_hex, key

    # Cross-rank identity for the derived hex keys corroborates the record
    # equality above; the oracle already pins each one against the fixture.
    for key in oracles:
        assert {rec["results"][key] for rec in records} == {
            records[0]["results"][key]
        }


@pytest.mark.requires_mpi
@needs_highs
@pytest.mark.parametrize("n", (2, 3), ids=lambda n: f"n{n}")
def test_nslack_bootstrap_distributed_matches_serial_bitwise(
    n: int, mpirun_path: str
) -> None:
    expected = _core.nslack_bootstrap_serial_oracle()
    records = _launch_records(mpirun_path, n, "nslack-bootstrap")
    for rec in records:
        assert rec["theta_hex"] == expected["theta_hex"], rec["rank"]
        assert rec["converged"] == expected["converged"]
        assert rec["n_bootstrap"] == expected["n_bootstrap"]
        assert rec["n_agents"] == expected["n_agents"]
        assert rec["iterations"] > 0
        assert rec["counters"].get("alltoallv", 0) > 0


# --- the chunked-scatter gate ------------------------------------------------

# Test-reduced chunk: small enough that kilobyte fixtures span several
# windows, exercised through the constructor override that exists for
# exactly this gate (the shipping default stays the module constant).
_CHUNK_TEST_BYTES = 256 * 1024


def _n_windows(n_rows: int, row_nbytes: int, chunk_bytes: int) -> int:
    """Mirror of the transport's chunk schedule length."""
    if n_rows == 0 or row_nbytes == 0:
        return 0
    rows_per = max(1, chunk_bytes // row_nbytes)
    return -(-n_rows // rows_per)


def _expected_scatter_counters(
    rows: list[int], n: int, root: int
) -> dict[int, dict[str, int]]:
    """The full per-rank counter dict of a chunked scatter, from fixture math.

    The scatter is one ids gather, one verdict broadcast, then one
    point-to-point window per (destination, key). The root sends every
    non-self destination's windows and receives none; each peer receives
    exactly its own windows and sends none. Both keys and values are
    derived here from the row partition and row widths — no combrum
    reduction or accessor is consulted — so the returned dict is an
    independent oracle for the whole counter shape, not just the p2p tally.
    Both the root-0 and nonzero-root scatter tests share it, differing
    only in which rank is root.
    """
    row_nbytes = _core._CHUNK_ROW_NBYTES
    expected: dict[int, dict[str, int]] = {}
    for r in range(n):
        counters = {"gather": 1, "bcast": 1}
        if r == root:
            sends = sum(
                _n_windows(rows[d], rb, _CHUNK_TEST_BYTES)
                for d in range(n)
                if d != root
                for rb in row_nbytes.values()
            )
            if sends:
                counters["p2p_send"] = sends
        else:
            counters["p2p_recv"] = sum(
                _n_windows(rows[r], rb, _CHUNK_TEST_BYTES)
                for rb in row_nbytes.values()
            )
        expected[r] = counters
    return expected


def _assert_scatter_send_recv_counts(
    records: list[dict[str, Any]], rows: list[int], n: int, root: int
) -> None:
    """Pin every rank's whole counter dict against the fixture-derived shape.

    The comparison is the entire ``rec["counters"]`` mapping against
    ``_expected_scatter_counters`` — exact keys and exact values — so it
    rejects not only an over- or under-fragmented p2p tally but any
    counter-shape drift: a peer that erroneously sends, a root that
    self-receives, a dropped ids gather or verdict broadcast, or a
    spurious extra collective.
    """
    expected = _expected_scatter_counters(rows, n, root)
    for rec in records:
        assert rec["counters"] == expected[rec["rank"]], rec["rank"]


@pytest.mark.requires_mpi
@pytest.mark.parametrize("n", (2, 4), ids=lambda n: f"n{n}")
def test_chunked_scatter_spans_windows_bitwise_with_bounded_sends(
    n: int, mpirun_path: str
) -> None:
    records = _launch_records(
        mpirun_path, n, "chunked", str(_CHUNK_TEST_BYTES)
    )
    rows: list[int] = records[0]["n_local_by_rank"]
    row_nbytes = _core._CHUNK_ROW_NBYTES
    _assert_scatter_send_recv_counts(records, rows, n, root=0)
    for rec in records:
        r = rec["rank"]
        # The mapping survives chunking bit for bit, read-only included.
        assert rec["bitwise"] == {"m": True, "v": True}
        assert rec["read_only"] == [True, True]
        # The fixture genuinely crosses chunks: every shard's wide key
        # needs at least four windows (~3.5 chunks of selected bytes).
        assert _n_windows(rows[r], row_nbytes["m"], _CHUNK_TEST_BYTES) >= 4


@pytest.mark.requires_mpi
def test_chunked_scatter_nonzero_root_counts_rank_zero_windows(
    mpirun_path: str,
) -> None:
    n = 4
    root = 2
    records = _launch_records(
        mpirun_path, n, "chunked", str(_CHUNK_TEST_BYTES), str(root)
    )
    rows: list[int] = records[0]["n_local_by_rank"]
    row_nbytes = _core._CHUNK_ROW_NBYTES
    rank_zero_windows = sum(
        _n_windows(rows[0], rb, _CHUNK_TEST_BYTES)
        for rb in row_nbytes.values()
    )
    competing_windows = max(
        sum(
            _n_windows(rows[r], rb, _CHUNK_TEST_BYTES)
            for rb in row_nbytes.values()
        )
        for r in range(n)
        if r not in (0, root)
    )
    assert rank_zero_windows > competing_windows
    for rec in records:
        assert rec["root"] == root
        assert rec["bitwise"] == {"m": True, "v": True}
        assert rec["read_only"] == [True, True]
    _assert_scatter_send_recv_counts(records, rows, n, root=root)


# --- the shared-window smoke -------------------------------------------------


@pytest.mark.requires_mpi
def test_shared_window_smoke_single_copy_readback_and_clean_close(
    mpirun_path: str,
) -> None:
    # np.shares_memory is meaningless across processes, so single-copy
    # placement is proved structurally: MPI reports zero bytes allocated
    # on the non-publishing rank, and both ranks read back the publisher's
    # exact bytes from the mapped pages. The launch exiting 0 (asserted in
    # _launch_records) is the close() proof: the window freed node-wide
    # without a leak-shaped hang.
    pub, peer = _launch_records(mpirun_path, 2, "window-smoke")
    assert [pub["node_rank"], peer["node_rank"]] == [0, 1]
    for rec in (pub, peer):
        assert rec["content"] == {"a": True, "b": True}
        assert rec["dtype"] == {"a": "float64", "b": "int16"}
        assert rec["shape"] == {"a": [4096, 8], "b": [2048]}
        assert rec["read_only"] == [True, True]
        # Read-only is un-flippable (the backing memoryview itself is
        # read-only) and the mapping rejects key insertion.
        assert rec["flip_raises"] is True
        assert rec["mapping_rejects_set"] is True
    # The fixture's arrays are 64-byte aligned end to end, so the
    # publisher's segment is exactly the payload — and the peer's is 0.
    assert pub["own_segment_bytes"] == pub["payload_nbytes"]
    assert peer["own_segment_bytes"] == 0


# --- the per-rank RSS ladder -------------------------------------------------

# Geometric ladder, ×4 per rung: (N, 16) float64 totals of 1, 4, 16 and
# 64 MiB scattered over 4 ranks (per-rank shards 0.25 → 16 MiB). Each
# rung is its own mpirun launch because ru_maxrss is a process-lifetime
# high-water mark — one process per measurement, or every rung after the
# first would inherit the previous rung's peak.
_LADDER_ROWS = (8192, 32768, 131072, 524288)
_LADDER_CHUNK_BYTES = _MIB  # gate sizing: 16 windows per shard at the top


@pytest.mark.requires_mpi
def test_per_rank_peak_rss_tracks_shard_not_total_including_root(
    mpirun_path: str,
) -> None:
    ladder: dict[int, list[dict[str, Any]]] = {}
    for n_rows in _LADDER_ROWS:
        records = _launch_records(
            mpirun_path,
            4,
            "rss-ladder",
            str(n_rows),
            str(_LADDER_CHUNK_BYTES),
        )
        assert all(rec["sample_ok"] for rec in records)
        ladder[n_rows] = records
    bottom, top = _LADDER_ROWS[0], _LADDER_ROWS[-1]
    # Receivers: peak growth across a ×64 total growth tracks ONE shard,
    # not two. Measured growth-minus-shard is ~5-6 MiB (interpreter, numpy
    # and MPI pools jitter by single-digit MiB), so the band is one shard
    # plus a 12 MiB fixed slack. A receiver that kept a redundant second
    # copy of its shard would add ~d_shard again (measured growth-minus-
    # shard ~21 MiB at the top rung), overshooting this band — a 2.0×
    # coefficient could never separate one held shard from two.
    for r in range(1, 4):
        d_shard = (
            ladder[top][r]["shard_nbytes"] - ladder[bottom][r]["shard_nbytes"]
        )
        growth = (
            ladder[top][r]["peak_rss_bytes"]
            - ladder[bottom][r]["peak_rss_bytes"]
        )
        assert growth <= d_shard + 12 * _MIB, (
            f"rank {r} peak grew {growth / _MIB:.1f} MiB for a shard"
            f" growth of {d_shard / _MIB:.1f} MiB"
        )

    # Root: subtract the two holdings the fixture mandates — the
    # memmap-backed source (macOS counts its faulted-in file pages in RSS;
    # the memmap keeps generation out of the peak, not the read pass) and
    # the root's own contract-held shard. What remains is the scatter's
    # anonymous transient, which the chunk bound caps: it must stay flat
    # while the total grows ×64. A root that materialized every
    # destination's rows at once would add ~2× total here and blow this
    # band by an order of magnitude.
    def root_excess(n_rows: int) -> int:
        rec = ladder[n_rows][0]
        return (
            rec["peak_rss_bytes"] - rec["source_nbytes"] - rec["shard_nbytes"]
        )

    d_shard_root = (
        ladder[top][0]["shard_nbytes"] - ladder[bottom][0]["shard_nbytes"]
    )
    excess_growth = root_excess(top) - root_excess(bottom)
    assert excess_growth <= d_shard_root + 32 * _MIB, (
        f"root anonymous excess grew {excess_growth / _MIB:.1f} MiB across"
        f" the ladder — the chunked stream must keep it near-flat"
    )
    # The headline claim at the heaviest rung: root's peak stays within
    # a documented factor of the non-root peaks once the source's
    # resident pages are credited (measured ~150-158 vs ~64+64 MiB, with
    # occasional ~1.7x page-fault spikes). This whole-process ratio is
    # loose corroboration only — RSS is noisy enough that the tight signal
    # live in the in-flight bound below; the 2.0 factor stops noise from
    # false-failing while still rejecting any total-scaled root transient.
    top_root = ladder[top][0]["peak_rss_bytes"]
    top_nonroot = max(ladder[top][r]["peak_rss_bytes"] for r in (1, 2, 3))
    assert top_root <= 2.0 * top_nonroot + ladder[top][0]["source_nbytes"]

    # The sharp structural bound behind the loose RSS bands: the meter in
    # the rank program records the root's peak concurrently-outstanding
    # scatter-chunk bytes. The one-chunk-per-destination window cap holds
    # it at chunk_bytes * (size - 1) no matter how many windows the shards
    # span, so it must stay flat across the ×64 ladder while the total
    # grows. A last window is often partial, so the peak may sit below the
    # cap; it can never sit above it. A root that let its in-flight buffer
    # grow past one window per destination (e.g. a Waitall lifted out of
    # the window loop) breaks this at once, independent of the RSS noise.
    n_ranks = 4
    in_flight_cap = _LADDER_CHUNK_BYTES * (n_ranks - 1)
    for n_rows in _LADDER_ROWS:
        root_in_flight = ladder[n_rows][0]["peak_in_flight_bytes"]
        assert 0 < root_in_flight <= in_flight_cap + 64 * 1024, (
            f"root held {root_in_flight / _MIB:.2f} MiB in flight at"
            f" n_rows={n_rows}; the window cap is"
            f" {in_flight_cap / _MIB:.2f} MiB (chunk × (size-1))"
        )
        # Peers never Isend, so their in-flight footprint is zero.
        assert all(
            ladder[n_rows][r]["peak_in_flight_bytes"] == 0 for r in (1, 2, 3)
        )


# --- the one-copy-per-node accounting ---------------------------------------


@pytest.mark.requires_mpi
def test_node_shared_rss_accounting_one_copy_per_node(
    mpirun_path: str,
) -> None:
    records = _launch_records(mpirun_path, 4, "node-rss")
    payload = records[0]["payload_nbytes"]
    assert payload == 16 * _MIB
    assert all(rec["content_ok"] for rec in records)
    pub, peers = records[0], records[1:]
    assert pub["node_rank"] == 0
    # The sharp structural check: the publisher's window segment is the
    # whole payload, every peer's is zero bytes.
    assert pub["own_segment_bytes"] == payload
    assert all(rec["own_segment_bytes"] == 0 for rec in peers)
    # The corroborating measurement, with generous bands (RSS is noisy):
    # publishing costs the publisher about one copy — the single
    # contract-mandated copy into the window — while a peer's resident
    # set barely moves because it maps pages instead of copying them
    # (measured ~16.2 MiB vs ~0.2 MiB for a 16 MiB payload).
    assert 0.6 * payload <= pub["publish_delta_bytes"] <= 1.8 * payload
    for rec in peers:
        assert rec["publish_delta_bytes"] <= payload // 4
