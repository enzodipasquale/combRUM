"""The opt-in run-metadata side-channel costs the default path nothing.

``estimate`` takes a ``run_info_level: RunInfoLevel`` and attaches
``RunMetadata`` to ``run_info``. Two invariants hold at every level:

* Metadata is built from already-computed data plus cached rank-local reads,
  so OFF/DEFAULT/META issue identical collectives; FULL adds exactly one
  ``batched_max``.
* The ``_PINNED`` result fields are byte-identical across levels, and
  ``run_info`` is never a ``to_dict`` key — it rides alongside the result,
  not inside the serialised surface.

Tiers stack: DEFAULT fills diagnostics/certification/runtime + layout; META
adds provenance + rank-0 peak RSS; FULL adds the cross-rank peaks.

Deterministic serial runs; HiGHS.
"""

from __future__ import annotations

import importlib
import platform
import sys

import numpy as np
import pytest

from _family_oracles import qkp_problem, toy_problem
from combrum.engine import estimate
from combrum.engine.certify import certification_metadata
from combrum.formulations import NSlack
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.commprobe import CountingTransport
from _support.families import FAMILY_DIR, load_family
from combrum.masters import highs as highs_backend
from combrum.model import Data, Model
from combrum.parameters import Parameters
import combrum.runinfo as runinfo
from combrum.runinfo import Provenance, RunInfoLevel, RunMetadata
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import NodeTopology, Transport

needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)

_PROBLEMS = {"toy": toy_problem, "qkp": qkp_problem}

# Result fields that must be byte-identical across run-info levels.
# runtime_seconds is wall clock and jitters run-to-run, so it is excluded.
_PINNED = ("theta_hat", "objective", "empirical_moment", "n_active_cuts", "slack", "metadata")


def test_provenance_positional_constructor_keeps_old_optional_order() -> None:
    # One distinguishable sentinel per positional slot: any reorder binds a
    # value to the wrong attribute.
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
        run_info_level=level,
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

    Tier-0 metadata comes from cached rank-local properties plus
    already-computed diagnostics/certification, so surfacing it adds no
    collective.
    """
    _assert_same_collectives(RunInfoLevel.OFF, RunInfoLevel.DEFAULT)


@needs_highs
@pytest.mark.parametrize("family", ["toy", "qkp"])
def test_off_vs_default_pinned_bytes_identical(family: str) -> None:
    """Every _PINNED field is byte-identical OFF vs DEFAULT; run_info stays out of to_dict."""
    off = _estimate(family, SerialTransport(), level=RunInfoLevel.OFF)
    default = _estimate(family, SerialTransport(), level=RunInfoLevel.DEFAULT)

    assert off.run_info is None, "OFF must attach no metadata"
    assert isinstance(default.run_info, RunMetadata)

    off_d, default_d = off.to_dict(), default.to_dict()
    assert "run_info" not in off_d and "run_info" not in default_d, (
        "run_info leaked into the result's plain rendering (the parity surface)"
    )
    # The metadata dict is a closed 3-key literal; comparing the exact key set
    # notices any stray key, run_info or otherwise.
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
    # Serial layout is rank 0 of 1 on a single node; compare against a
    # hand-built topology and against SerialTransport().node.
    assert isinstance(ri.node, NodeTopology)
    assert ri.node == NodeTopology(node_id=0, node_rank=0, node_size=1, n_nodes=1)
    assert ri.node == SerialTransport().node
    # estimate.py bcasts one runtime_seconds into both FitResult and
    # RunMetadata, so the two are the same value bit for bit.
    assert ri.runtime_seconds == fit.runtime_seconds
    assert ri.runtime_seconds >= 0.0
    assert ri.diagnostics is not None
    assert ri.diagnostics.iterations >= 1
    assert ri.certification is not None
    # The surfaced Tier-0 objects must describe this run: fit.metadata is
    # filled from the real outcome by a separate path in estimate.py.
    assert ri.diagnostics.iterations == fit.metadata["iterations"]
    assert ri.diagnostics.converged == fit.metadata["converged"]
    assert certification_metadata(ri.certification) == fit.metadata["certification"]
    # Tier 1 (META) is opt-in — absent at DEFAULT.
    assert ri.provenance is None
    assert ri.peak_rss_bytes is None


# --- Tier 1 (META) -----------------------------------------------------------


def _expected_mpi_lib() -> str | None:
    """MPI library banner, read straight from mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.Get_library_version()


def _expected_gurobi_version() -> str | None:
    """Dotted gurobi version, composed from the raw tuple."""
    try:
        import gurobipy as grb
    except Exception:
        return None
    return ".".join(str(x) for x in grb.gurobi.version())


def _expected_blas() -> str | None:
    """BLAS field recomputed from np.show_config by the documented rule:
    name/version when both are present and the version is known, else the
    bare name or None.
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
    # Content, not just non-emptiness: compare against stdlib/numpy directly.
    assert ri.provenance.python_version == platform.python_version()
    assert ri.provenance.numpy_version == np.__version__
    assert ri.provenance.platform == platform.platform()
    assert ri.provenance.solver_backend == "highs"
    assert ri.provenance.resolved_backend == "highs"
    assert isinstance(ri.peak_rss_bytes, int) and ri.peak_rss_bytes > 0
    # Optional fields must match a recomputation from raw library reads when
    # the library is present; `is None or` keeps this portable to hosts
    # missing any of them.
    assert ri.provenance.mpi_lib is None or ri.provenance.mpi_lib == _expected_mpi_lib()
    assert ri.provenance.blas is None or ri.provenance.blas == _expected_blas()
    assert (
        ri.provenance.gurobi_version is None
        or ri.provenance.gurobi_version == _expected_gurobi_version()
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
    """Fake the three probed libraries so collect_provenance sees known inputs.

    collect_provenance imports mpi4py/gurobipy lazily and calls np.show_config,
    so sys.modules entries and runinfo.np.show_config are enough to feed it
    exact values regardless of what is really installed.
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
    """The exact composed strings for the optional provenance fields.

    Dotted gurobi version, BLAS name/version joined by '/', MPI banner passed
    through verbatim — checked against hand-derived values on faked library
    readings.
    """
    _install_provenance_fakes(
        monkeypatch,
        mpi_ver="Open MPI v9.9.9, faked banner",
        gurobi_tuple=(11, 2, 3),
        blas_name="openblas",
        blas_version="0.3.21",
    )
    prov = runinfo.collect_provenance("auto", resolved_backend="highs")

    assert prov.mpi_lib == "Open MPI v9.9.9, faked banner"
    assert prov.gurobi_version == "11.2.3"
    assert prov.blas == "openblas/0.3.21"  # name/version, both present & known
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
    """Tier-1 reads are rank-local (interpreter/library versions plus a single
    getrusage on rank 0), so a META run issues exactly DEFAULT's collectives.
    """
    _assert_same_collectives(RunInfoLevel.DEFAULT, RunInfoLevel.META)


# --- Capture metadata --------------------------------------------------------

# Metadata fields expected to stay independent of run_info serialization.
_CAPTURE_METADATA_KEYS = (
    "certification",
    "converged",
    "iterations",
)


@needs_highs
def test_run_info_quarantined_from_captured_bytes() -> None:
    """A META run serialises byte-identical to an OFF run.

    Every field the capture _doc/manifest consume — objective, theta_hat,
    empirical_moment, the metadata subset — matches, and run_info is not a
    key in the serialised surface.
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
    # The metadata key set is closed; compare it exactly so any stray key shows.
    assert set(meta.metadata) == set(_CAPTURE_METADATA_KEYS)


@needs_highs
@pytest.mark.parametrize(
    "level",
    [RunInfoLevel.DEFAULT, RunInfoLevel.META, RunInfoLevel.FULL],
    ids=["DEFAULT", "META", "FULL"],
)
def test_pins_identical_to_off_at_every_level(level: RunInfoLevel) -> None:
    """Every level, FULL included, matches the OFF baseline byte for byte on
    the ``_PINNED`` to_dict fields and the capture-read metadata, with
    run_info out of the serialised surface.
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
    # On a serial transport batched_max is the identity, so wall_max_seconds
    # is exactly the local wall bcast into runtime_seconds — equal bit for bit.
    assert ri.wall_max_seconds == ri.runtime_seconds
    # RSS is read twice: once at reduce time, then again on rank 0. ru_maxrss
    # is a monotone high-water mark and the reduce read happens first, so
    # rss_max_bytes <= peak_rss_bytes regardless of the reduce arithmetic.
    assert isinstance(ri.rss_max_bytes, int) and ri.rss_max_bytes > 0
    assert ri.rss_max_bytes <= ri.peak_rss_bytes
    # And a floor at half the later read: real RSS growth between the two reads
    # is small, so this brackets rss_max_bytes to the right magnitude and rules
    # out a kibibyte-vs-byte mis-scale in either direction.
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

    Tier 2 reduces ``[wall, rss]`` in a single vector max-collective — no new
    primitive, no per-iteration cost.
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
    """The dense estimator refuses a non-serial transport at every level.

    ``estimate`` guards ``reject_multirank_dense_transport`` before it inspects
    ``run_info_level``, so OFF through FULL all raise without constructing
    RunMetadata or reaching the Tier-2 collective.
    """
    with pytest.raises(ValueError, match="does not support non-serial"):
        LocalCluster(size).run(lambda t: _estimate("qkp", t, level=level))


# RSS unit conventions (normalize_maxrss / peak_rss_bytes) are covered in
# tests/test_probes.py.
