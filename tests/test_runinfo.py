"""Gate: the opt-in run-metadata side-channel costs the default path nothing.

``estimate`` takes a ``level: RunInfoLevel`` and attaches a
``RunMetadata`` to ``run_info``. Two invariants hold at every level:

* **No collective delta.** Metadata is built from already-computed data plus
  cached rank-local reads (rank/size/node, provenance, a rank-0 getrusage), so
  OFF/DEFAULT/META issue identical collectives; FULL adds exactly one
  ``batched_max``. Asserted through the comm-probe's per-kind counts.
* **No pinned byte moves.** The parity-pinned fields (theta_hat, objective,
  empirical_moment, n_active_cuts, slack, metadata) are byte-identical across
  OFF/DEFAULT/META/FULL, and ``run_info`` is never a ``to_dict`` key — it rides
  alongside the result, not inside the captured bytes. FULL is pinned too, so
  the guard stays permanent.

Tiers stack: DEFAULT fills diagnostics/certification/runtime + layout; META
adds provenance + rank-0 peak RSS; FULL adds the cross-rank peaks.

Deterministic serial runs; HiGHS.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from _family_oracles import qkp_problem, toy_problem
from combrum.engine import estimate
from combrum.engine.certify import certification_metadata
from combrum.formulations import NSlack
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.commprobe import CountingTransport
from _support.families import load_family
from combrum.masters import highs as highs_backend
from combrum.model import Data, Model
from combrum.parameters import Parameters
import combrum.runinfo as runinfo
from combrum.runinfo import (
    Provenance,
    RunInfoLevel,
    RunMetadata,
    normalize_maxrss,
)
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import NodeTopology, Transport

TESTS = Path(__file__).resolve().parent
FAMILY_DIR = TESTS / "fixtures" / "families"

needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)

_PROBLEMS = {"toy": toy_problem, "qkp": qkp_problem}

# The pinned parity surface: everything a parity gate captures off the result.
# runtime_seconds is the wall clock — performance, not a parity pin — so it
# jitters run-to-run and is excluded from the byte-identical comparison.
_PINNED = ("theta_hat", "objective", "empirical_moment", "n_active_cuts", "slack", "metadata")


def test_provenance_positional_constructor_keeps_old_optional_order() -> None:
    # Distinguishable sentinel per slot so any reorder of the positional fields
    # binds a value to the wrong attribute and fails. Pinning all eight slots
    # (not just the optional tail) makes the whole positional order the oracle:
    # a head swap (python_version/numpy_version/platform) or a
    # platform<->solver_backend swap diverges here.
    provenance = Provenance(
        "py",
        "np",
        "plat",
        "auto",
        "mpi",
        "blas",
        "12.0.1",
    )

    assert provenance.python_version == "py"
    assert provenance.numpy_version == "np"
    assert provenance.platform == "plat"
    assert provenance.solver_backend == "auto"
    assert provenance.mpi_lib == "mpi"
    assert provenance.blas == "blas"
    assert provenance.gurobi_version == "12.0.1"
    assert provenance.resolved_backend is None


def test_provenance_resolved_backend_is_keyword_extension() -> None:
    provenance = Provenance(
        "3.11",
        "2.0",
        "platform",
        "auto",
        "mpi",
        "blas",
        "12.0.1",
        resolved_backend="gurobi",
    )

    assert provenance.resolved_backend == "gurobi"


def _parameters(family: str, n_items: int) -> Parameters:
    if family == "toy":
        return Parameters({"theta": (-THETA_BOUND, THETA_BOUND, n_items)})
    return Parameters(
        {
            "alpha": (0.0, THETA_BOUND, 1),
            "delta": (-THETA_BOUND, THETA_BOUND, n_items - 2),
            "lam": (0.0, THETA_BOUND, 1),
        }
    )


def _estimate(
    family: str,
    transport: Transport,
    *,
    level: RunInfoLevel,
    master_backend: str = "highs",
):
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    observed = np.asarray(arrays["observed"])
    n_obs = observed.shape[0]
    model = Model(
        problem.oracle,
        _parameters(family, problem.K),
        features=problem.features,
        observed_features=problem.observed_features,
        formulation=NSlack,
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
        master_backend=master_backend,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        level=level,
    )


def _assert_same_collectives(baseline: RunInfoLevel, candidate: RunInfoLevel) -> None:
    """Run qkp at two levels and assert byte-for-byte equal comm profiles."""
    base = CountingTransport(SerialTransport())
    cand = CountingTransport(SerialTransport())
    _estimate("qkp", base, level=baseline)
    _estimate("qkp", cand, level=candidate)
    assert cand.counts() == base.counts(), (
        f"{candidate.name} changed the collective profile vs {baseline.name}:"
        f" {baseline.name}={base.counts()}, {candidate.name}={cand.counts()}"
    )


@needs_highs
def test_off_vs_default_zero_collective_delta() -> None:
    """A DEFAULT run issues exactly the collectives an OFF run does.

    Tier-0 metadata reads rank/size/node (cached rank-local properties the
    comm-probe does not count) + already-computed diagnostics/certification, so
    surfacing it adds no collective. The per-kind counts must match exactly.
    """
    _assert_same_collectives(RunInfoLevel.OFF, RunInfoLevel.DEFAULT)


@needs_highs
@pytest.mark.parametrize("family", ["toy", "qkp"])
def test_off_vs_default_pinned_bytes_identical(family: str) -> None:
    """Every parity-pinned field is byte-identical; run_info never in to_dict."""
    off = _estimate(family, SerialTransport(), level=RunInfoLevel.OFF)
    default = _estimate(family, SerialTransport(), level=RunInfoLevel.DEFAULT)

    assert off.run_info is None, "OFF must attach no metadata"
    assert isinstance(default.run_info, RunMetadata)

    off_d, default_d = off.to_dict(), default.to_dict()
    assert "run_info" not in off_d and "run_info" not in default_d, (
        "run_info leaked into the result's plain rendering (the parity surface)"
    )
    # Pin the exact metadata key set, not just the absence of "run_info": the
    # dict is a closed 3-key literal, so `"run_info" not in metadata` is always
    # true no matter how the run-info wiring breaks. The exact set catches any
    # stray key — run_info or otherwise — entering the pinned parity metadata.
    assert set(default_d["metadata"]) == {"certification", "converged", "iterations"}, (
        "the pinned metadata gained/lost a key — parity surface moved"
    )
    for key in _PINNED:
        assert off_d[key] == default_d[key], (
            f"{family}: pinned field {key!r} differs OFF vs DEFAULT —"
            " run_info perturbed the parity surface"
        )
    assert off.theta_hat.tobytes() == default.theta_hat.tobytes()


@needs_highs
def test_default_surfaces_tier0_only() -> None:
    """DEFAULT populates the already-computed Tier-0 data; Tier-1 stays None."""
    fit = _estimate("qkp", SerialTransport(), level=RunInfoLevel.DEFAULT)
    ri = fit.run_info
    assert ri.level == RunInfoLevel.DEFAULT
    assert ri.rank == 0 and ri.size == 1
    # Pin the whole node layout, not just its type. isinstance is redundant with
    # RunMetadata.__post_init__ (it already rejects a non-NodeTopology), so it
    # cannot catch a fabricated-but-valid topology. Hand-construct the serial
    # layout from the topology contract (rank 0 of 1, one node) as an oracle that
    # never routes through transport.node: a wrong node_id/node_rank/node_size/
    # n_nodes diverges here. Also equals the independent SerialTransport().node.
    assert isinstance(ri.node, NodeTopology)
    assert ri.node == NodeTopology(node_id=0, node_rank=0, node_size=1, n_nodes=1)
    assert (ri.node.node_id, ri.node.node_rank, ri.node.node_size, ri.node.n_nodes) == (
        0,
        0,
        1,
        1,
    )
    assert ri.node == SerialTransport().node
    # estimate.py bcasts one runtime_seconds and wires it into BOTH
    # FitResult.runtime_seconds and RunMetadata.runtime_seconds, so they are the
    # same value bit for bit. fit.runtime_seconds is the independent oracle: it
    # is set from the real run outside the run_info wiring, so pointing run_info's
    # runtime at a wrong constant (0.0) or scaling it diverges here, where the
    # bare `>= 0.0` never noticed.
    assert ri.runtime_seconds == fit.runtime_seconds
    assert ri.runtime_seconds >= 0.0
    assert ri.diagnostics is not None
    assert ri.diagnostics.iterations >= 1
    assert ri.certification is not None
    # The surfaced Tier-0 objects must describe this run. metadata is built
    # independently from the real outcome
    # (estimate.py: metadata["iterations"] = int(outcome.diagnostics.iterations),
    # metadata["certification"] = certification_metadata(certification)), so it
    # is the distinct oracle: swapping a fabricated diagnostics/certification
    # into RunMetadata leaves metadata pointed at the real run and diverges here.
    assert ri.diagnostics.iterations == fit.metadata["iterations"]
    assert ri.diagnostics.converged == fit.metadata["converged"]
    assert certification_metadata(ri.certification) == fit.metadata["certification"]
    # Tier 1 (META) is opt-in — absent at DEFAULT.
    assert ri.provenance is None
    assert ri.peak_rss_bytes is None


@needs_highs
def test_off_attaches_nothing() -> None:
    """level=OFF leaves run_info None (the explicit no-metadata path)."""
    fit = _estimate("qkp", SerialTransport(), level=RunInfoLevel.OFF)
    assert fit.run_info is None


# --- Tier 1 (META) -----------------------------------------------------------


def _mpi_lib_oracle() -> str | None:
    """The MPI library banner, read straight from mpi4py (not collect_provenance)."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.Get_library_version()


def _gurobi_version_oracle() -> str | None:
    """Dotted gurobi version, composed independently from the raw tuple."""
    try:
        import gurobipy as grb
    except Exception:
        return None
    return ".".join(str(x) for x in grb.gurobi.version())


def _blas_oracle() -> str | None:
    """BLAS name/version, recomposed from numpy's config by the documented rule.

    Distinct from collect_provenance: this reads np.show_config directly and
    applies the contract ("name/version when both present and version is known,
    else name-or-None") so a swap or separator change inside collect_provenance
    diverges from this value.
    """
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
    except Exception:
        return None
    name = str(blas.get("name", ""))
    version = str(blas.get("version", ""))
    if name and version and version != "unknown":
        return f"{name}/{version}"
    return name or None


@needs_highs
def test_meta_populates_provenance_and_peak_rss() -> None:
    """level=META fills the Tier-1 provenance + rank-0 peak RSS."""
    fit = _estimate("qkp", SerialTransport(), level=RunInfoLevel.META)
    ri = fit.run_info
    assert ri.level == RunInfoLevel.META
    assert isinstance(ri.provenance, Provenance)
    # The three "always populated" fields must carry the right content, not just
    # be non-empty: a swapped assignment (python_version=np.__version__, etc.)
    # keeps all three truthy but wrong. Oracles come from stdlib/numpy directly.
    assert ri.provenance.python_version == platform.python_version()
    assert ri.provenance.numpy_version == np.__version__
    assert ri.provenance.platform == platform.platform()
    assert ri.provenance.solver_backend == "highs"
    assert ri.provenance.resolved_backend == "highs"
    assert isinstance(ri.peak_rss_bytes, int) and ri.peak_rss_bytes > 0
    # Optional provenance fields: pin the exact composed value against an
    # independently-derived oracle (raw library reads) where the library is
    # present, else None. isinstance alone let a derivation regression inside
    # collect_provenance slip through (a ',' gurobi join, a swapped blas
    # name/version) while the field stayed a non-None str. The oracles below
    # never route through collect_provenance. Guarded with `is None or` so the
    # test stays portable to hosts missing any of these libraries.
    assert ri.provenance.mpi_lib is None or ri.provenance.mpi_lib == _mpi_lib_oracle()
    assert ri.provenance.blas is None or ri.provenance.blas == _blas_oracle()
    assert (
        ri.provenance.gurobi_version is None
        or ri.provenance.gurobi_version == _gurobi_version_oracle()
    )


@needs_highs
def test_meta_records_requested_and_resolved_backend(monkeypatch) -> None:
    estimate_mod = importlib.import_module("combrum.engine.estimate")
    monkeypatch.setattr(
        estimate_mod,
        "resolve_master_backend",
        lambda requested, **kwargs: "highs",
    )

    fit = _estimate(
        "qkp",
        SerialTransport(),
        level=RunInfoLevel.META,
        master_backend="auto",
    )

    assert fit.run_info.provenance.solver_backend == "auto"
    assert fit.run_info.provenance.resolved_backend == "highs"


def _install_provenance_fakes(
    monkeypatch, *, mpi_ver, gurobi_tuple, blas_name, blas_version
) -> None:
    """Fake the three probed libraries so collect_provenance's composition is
    exercised on known inputs, host-independent.

    collect_provenance imports mpi4py/gurobipy lazily and calls np.show_config,
    so replacing sys.modules entries and runinfo.np.show_config feeds it exact
    values without touching whatever is really installed. monkeypatch restores
    everything on teardown.
    """
    import types

    fake_mpi = types.ModuleType("mpi4py")
    fake_mpi.MPI = types.SimpleNamespace(Get_library_version=lambda: mpi_ver)
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake_mpi.MPI)

    fake_grb = types.ModuleType("gurobipy")
    fake_grb.gurobi = types.SimpleNamespace(version=lambda: gurobi_tuple)
    monkeypatch.setitem(sys.modules, "gurobipy", fake_grb)

    monkeypatch.setattr(
        runinfo.np,
        "show_config",
        lambda mode="dicts": {
            "Build Dependencies": {"blas": {"name": blas_name, "version": blas_version}}
        },
    )


def test_collect_provenance_composes_optional_fields_exactly(monkeypatch) -> None:
    """collect_provenance pins the EXACT composed strings for the optional fields.

    Feeds known library readings through fakes and checks the whole composed
    output against hand-derived expected values (dotted gurobi version, BLAS
    name/version joined by '/', MPI banner passed through verbatim). This is the
    only place collect_provenance's derivation is exercised, and pinning the
    exact strings kills the composition class wholesale: a ',' gurobi join, a
    swapped blas name/version, a wrong separator, or a mangled MPI read all
    diverge from these oracles. The always-populated head is pinned too.
    """
    _install_provenance_fakes(
        monkeypatch,
        mpi_ver="Open MPI v9.9.9, faked banner",
        gurobi_tuple=(11, 2, 3),
        blas_name="openblas",
        blas_version="0.3.21",
    )
    prov = runinfo.collect_provenance("auto", resolved_backend="highs")

    # Hand-derived expected, never read off collect_provenance:
    assert prov.mpi_lib == "Open MPI v9.9.9, faked banner"
    assert prov.gurobi_version == "11.2.3"  # ".".join(map(str, (11, 2, 3)))
    assert prov.blas == "openblas/0.3.21"   # name/version, both present & known
    # The always-populated head + backend passthrough round out the full object.
    assert prov.python_version == platform.python_version()
    assert prov.numpy_version == np.__version__
    assert prov.platform == platform.platform()
    assert prov.solver_backend == "auto"
    assert prov.resolved_backend == "highs"


def test_collect_provenance_blas_unknown_version_falls_back_to_name(monkeypatch) -> None:
    """When BLAS version is 'unknown', blas is the bare name (no '/unknown')."""
    _install_provenance_fakes(
        monkeypatch,
        mpi_ver="banner",
        gurobi_tuple=(12,),
        blas_name="accelerate",
        blas_version="unknown",
    )
    prov = runinfo.collect_provenance("highs")
    assert prov.blas == "accelerate"
    assert prov.gurobi_version == "12"  # single-element tuple -> no separator


def test_collect_provenance_none_when_libraries_absent(monkeypatch) -> None:
    """Missing mpi4py/gurobipy and a config-less numpy yield the three Nones."""
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name in {"mpi4py", "gurobipy"} or name.startswith("mpi4py."):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mpi4py", raising=False)
    monkeypatch.delitem(sys.modules, "gurobipy", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)
    monkeypatch.setattr(
        runinfo.np, "show_config", lambda mode="dicts": {"Build Dependencies": {}}
    )

    prov = runinfo.collect_provenance("highs")
    assert prov.mpi_lib is None
    assert prov.blas is None
    assert prov.gurobi_version is None
    # The mandatory head is still populated on the fallback path.
    assert prov.python_version == platform.python_version()
    assert prov.solver_backend == "highs"


@needs_highs
def test_meta_adds_zero_collectives_vs_default() -> None:
    """META's extra reads (provenance + one getrusage) add NO collective.

    The Tier-1 reads are rank-local — interpreter/library versions + a single
    getrusage on rank 0 — so a META run issues exactly DEFAULT's collectives.
    """
    _assert_same_collectives(RunInfoLevel.DEFAULT, RunInfoLevel.META)


@needs_highs
def test_meta_keeps_pinned_bytes_identical() -> None:
    """META still never touches the parity surface (run_info out of to_dict)."""
    default = _estimate("qkp", SerialTransport(), level=RunInfoLevel.DEFAULT)
    meta = _estimate("qkp", SerialTransport(), level=RunInfoLevel.META)
    assert "run_info" not in meta.to_dict()
    d_d, m_d = default.to_dict(), meta.to_dict()
    for key in _PINNED:
        assert d_d[key] == m_d[key], f"META perturbed pinned field {key!r}"


# --- Capture metadata --------------------------------------------------------

# Metadata fields expected to stay independent of run_info serialization.
_CAPTURE_METADATA_KEYS = (
    "certification",
    "converged",
    "iterations",
)


@needs_highs
def test_run_info_quarantined_from_captured_bytes() -> None:
    """A META run captures byte-identical to an OFF run.

    Every field the capture _doc/manifest consume — objective, theta_hat,
    empirical_moment, and the pinned metadata subset — matches, and run_info
    is not a key in the serialised surface, so the capture is blind to it.
    """
    off = _estimate("qkp", SerialTransport(), level=RunInfoLevel.OFF)
    meta = _estimate("qkp", SerialTransport(), level=RunInfoLevel.META)
    assert off.run_info is None and meta.run_info is not None

    off_d, meta_d = off.to_dict(), meta.to_dict()
    assert "run_info" not in off_d and "run_info" not in meta_d
    for key in ("objective", "theta_hat", "empirical_moment"):
        assert off_d[key] == meta_d[key], f"run_info moved captured field {key!r}"
    for key in _CAPTURE_METADATA_KEYS:
        assert off.metadata[key] == meta.metadata[key], (
            f"run_info moved metadata[{key!r}] the capture _doc/manifest reads"
        )
    # The capture reads a fixed metadata key set; pin it exactly rather than the
    # always-true `"run_info" not in meta.metadata` (the dict is a closed literal
    # that no run-info code writes into). This catches any stray metadata key.
    assert set(meta.metadata) == set(_CAPTURE_METADATA_KEYS)


@needs_highs
@pytest.mark.parametrize(
    "level",
    [RunInfoLevel.DEFAULT, RunInfoLevel.META, RunInfoLevel.FULL],
    ids=["DEFAULT", "META", "FULL"],
)
def test_pins_identical_to_off_at_every_level(level: RunInfoLevel) -> None:
    """Parity bytes are byte-identical to OFF at every level, FULL included.

    Pinning FULL (not just OFF/DEFAULT/META) makes the guard permanent: future
    work that lights up at FULL fails the instant it moves a captured byte.
    Every pinned to_dict field and every capture-read metadata field must match
    the OFF baseline, and run_info must stay out of the serialised surface.
    """
    off = _estimate("qkp", SerialTransport(), level=RunInfoLevel.OFF)
    fit = _estimate("qkp", SerialTransport(), level=level)
    off_d, fit_d = off.to_dict(), fit.to_dict()
    assert "run_info" not in fit_d, f"level={level.name}: run_info entered to_dict"
    for key in _PINNED:
        assert off_d[key] == fit_d[key], (
            f"level={level.name}: pinned field {key!r} differs from OFF"
        )
    for key in _CAPTURE_METADATA_KEYS:
        assert off.metadata[key] == fit.metadata[key], (
            f"level={level.name}: metadata[{key!r}] the capture reads differs from OFF"
        )


# --- Tier 2 (FULL) -----------------------------------------------------------


@needs_highs
def test_full_populates_tier2_cross_rank_peaks() -> None:
    """level=FULL fills the Tier-2 cross-rank peak wall + RSS (and keeps 0/1)."""
    fit = _estimate("qkp", SerialTransport(), level=RunInfoLevel.FULL)
    ri = fit.run_info
    assert ri.level == RunInfoLevel.FULL
    assert isinstance(ri.wall_max_seconds, float)
    # On a serial transport batched_max over [local_wall, rss] is the identity,
    # so peaks[0] is exactly the local wall — the same value bcast into
    # runtime_seconds. wall_max_seconds must therefore equal runtime_seconds bit
    # for bit; a bug that reduced the wrong element into it (e.g. RSS-as-seconds)
    # would break this while still passing a loose >= 0.0 check.
    assert ri.wall_max_seconds == ri.runtime_seconds
    # RSS is read twice: batched_max reads peak_rss_bytes() at reduce time
    # (estimate.py:199), then rank 0 reads it again for peak_rss_bytes
    # (estimate.py:221). ru_maxrss is a monotone non-decreasing high-water mark
    # and the reduce read happens first, so rss_max_bytes <= peak_rss_bytes is an
    # invariant independent of the FULL-tier reduce arithmetic. A scaling/unit
    # bug in that reduce (2x inflation, kib-vs-byte drift, +constant) breaks the
    # ordering while still passing a bare positive-int shape check.
    assert isinstance(ri.rss_max_bytes, int) and ri.rss_max_bytes > 0
    assert ri.rss_max_bytes <= ri.peak_rss_bytes
    # The upper bound alone catches upward mis-scale but is blind to a downward
    # one (bytes reported as kibibytes, ~1024x too small, e.g. `// 1024`). Both
    # values come from the same monotone ru_maxrss high-water mark: the reduce
    # read (estimate.py:199) then the rank-0 read (estimate.py:221). Real growth
    # between them is bounded, so a floor at half the later read brackets
    # rss_max_bytes to the correct magnitude and kills the whole scaling class
    # (1024x down, 1024x up, +constant, 2x) that a bare positive-int check misses.
    assert ri.rss_max_bytes >= ri.peak_rss_bytes // 2
    # Each level adds to the prior: Tier-0/1 are still present at FULL.
    assert ri.provenance is not None and ri.peak_rss_bytes is not None


@needs_highs
@pytest.mark.parametrize(
    "level", [RunInfoLevel.DEFAULT, RunInfoLevel.META], ids=["DEFAULT", "META"]
)
def test_below_full_leaves_tier2_none(level: RunInfoLevel) -> None:
    """DEFAULT and META leave the Tier-2 peaks None — FULL-only."""
    fit = _estimate("qkp", SerialTransport(), level=level)
    assert fit.run_info.wall_max_seconds is None
    assert fit.run_info.rss_max_bytes is None


@needs_highs
def test_full_adds_exactly_one_max_collective_vs_meta() -> None:
    """FULL adds exactly one collective over META: the end-of-fit batched_max.

    Tier 2 reduces ``[wall, rss]`` in one vector max-collective (batched_max,
    since scalar allreduce_max cannot reduce both at once). FULL's comm profile
    is META's plus that one batched_max — no new primitive, no per-iteration cost.
    """
    meta = CountingTransport(SerialTransport())
    full = CountingTransport(SerialTransport())
    _estimate("qkp", meta, level=RunInfoLevel.META)
    _estimate("qkp", full, level=RunInfoLevel.FULL)
    meta_c, full_c = meta.counts(), full.counts()
    delta = {
        k: full_c.get(k, 0) - meta_c.get(k, 0) for k in set(full_c) | set(meta_c)
    }
    delta = {k: v for k, v in delta.items() if v != 0}
    assert delta == {"batched_max": 1}, (
        f"FULL must add exactly one batched_max over META; got delta {delta}"
    )


@needs_highs
@pytest.mark.parametrize(
    "level",
    [RunInfoLevel.OFF, RunInfoLevel.DEFAULT, RunInfoLevel.META, RunInfoLevel.FULL],
    ids=["OFF", "DEFAULT", "META", "FULL"],
)
@pytest.mark.parametrize("size", [2, 4])
def test_dense_estimate_rejects_multirank_transport_at_every_level(
    size: int, level: RunInfoLevel
) -> None:
    """The dense estimator refuses a non-serial transport regardless of level.

    ``estimate`` guards ``reject_multirank_dense_transport`` at the very top,
    before it inspects ``level`` or reaches any run_info / batched_max code. The
    rejection is therefore level-agnostic: OFF through FULL all raise on a
    multirank transport, and none of them ever construct a RunMetadata or run the
    Tier-2 collective. Parametrizing over level documents that the guard, not the
    metadata path, is what fires — and pins that no level slips past it.
    """
    with pytest.raises(ValueError, match="does not support non-serial"):
        LocalCluster(size).run(lambda t: _estimate("qkp", t, level=level))


# --- Direct unit tests for the RSS unit logic --------------------------------
#
# The estimate-path tests above only pin peak_rss_bytes as a positive int, which
# cannot catch a wrong unit convention (darwin and linux disagree by 1024x).
# normalize_maxrss is the sole place that conversion lives, so it gets tested
# directly by faking the platform. Expected values are hand-derived from the
# kibibyte convention (1 KiB = 1024 bytes), never read off combrum output.


@pytest.mark.parametrize(
    "fake_platform, raw, expected",
    [
        ("darwin", 1000, 1000),          # darwin ru_maxrss is already bytes
        ("linux", 1000, 1000 * 1024),    # linux ru_maxrss is kibibytes -> ×1024
        ("linux2", 7, 7 * 1024),         # startswith("linux") also converts
    ],
)
def test_normalize_maxrss_unit_convention(
    monkeypatch, fake_platform, raw, expected
) -> None:
    monkeypatch.setattr(runinfo.sys, "platform", fake_platform)
    assert normalize_maxrss(raw) == expected


@pytest.mark.parametrize("fake_platform", ["win32", "cygwin", "freebsd12"])
def test_normalize_maxrss_rejects_unknown_platform(monkeypatch, fake_platform) -> None:
    """An unrecognised platform must raise rather than silently mis-scale."""
    monkeypatch.setattr(runinfo.sys, "platform", fake_platform)
    with pytest.raises(RuntimeError, match="unit convention unknown"):
        normalize_maxrss(1000)


def test_peak_rss_bytes_rises_after_large_allocation(tmp_path: Path) -> None:
    """peak_rss_bytes must report a monotone, unit-correct peak.

    ru_maxrss is a per-process high-water mark, so this can only be checked in a
    FRESH subprocess: within the shared pytest process an earlier test may have
    already pushed the mark far above current usage, and a new 64 MiB allocation
    would then not raise it at all. The child samples the peak, faults in a large
    buffer page by page, and requires the reported peak to rise by well over half
    the allocation. A unit bug (returning kibibytes on darwin, i.e. dropping the
    ×1024 scaling) would report ~1024x too small and fall under the floor; a
    zeroed-out implementation fails outright.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    probe = textwrap.dedent(
        """
        import resource
        from combrum.runinfo import peak_rss_bytes, normalize_maxrss

        before = normalize_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        n_bytes = 64 * 1024 * 1024
        buf = bytearray(n_bytes)
        for i in range(0, n_bytes, 4096):
            buf[i] = 1
        reported = peak_rss_bytes()
        # Rose by well over half of what we faulted in (headroom for the OS not
        # backing every page), and ~1024x above a mis-scaled darwin path.
        floor = before + 32 * 1024 * 1024
        assert reported > floor, (before, reported, floor)
        # Bytes, not kibibytes: the peak must be at least the buffer we allocated.
        assert reported >= n_bytes, (reported, n_bytes)
        print("OK")
        """
    )
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")
