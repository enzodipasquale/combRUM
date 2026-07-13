"""Run metadata surfaced on a fit result."""

from __future__ import annotations

import enum
import platform
import resource
import sys
from dataclasses import dataclass

import numpy as np

from combrum.certification import Certification
from combrum.engine.driver import LoopDiagnostics
from combrum.transport.base import NodeTopology


class RunInfoLevel(enum.IntEnum):
    """How much run metadata to attach.

    Ordered (``OFF < DEFAULT < META < FULL``) so a producer can gate with
    ``level >= RunInfoLevel.META``; each level adds to the prior.
    """

    OFF = 0
    DEFAULT = 1
    META = 2
    FULL = 3


@dataclass(frozen=True)
class Provenance:
    """The run's environment fingerprint.

    ``resolved_backend`` is the concrete backend selected after ``"auto"``
    resolution when the caller provides it.
    """

    python_version: str
    numpy_version: str
    platform: str
    solver_backend: str
    mpi_lib: str | None = None
    blas: str | None = None
    gurobi_version: str | None = None
    resolved_backend: str | None = None


@dataclass(frozen=True)
class RunMetadata:
    """Provenance and diagnostics surfaced on a result.

    ``DEFAULT`` carries ``diagnostics``, ``certification``, ``runtime_seconds``,
    and the ``rank`` / ``size`` / ``node`` layout. ``provenance`` and
    ``peak_rss_bytes`` are added at ``META`` (``None`` otherwise).
    ``wall_max_seconds`` (slowest rank's wall) and ``rss_max_bytes`` (highest
    rank's RSS) are the cross-rank peaks added at ``FULL`` (``None`` otherwise).
    """

    level: RunInfoLevel
    rank: int
    size: int
    node: NodeTopology
    runtime_seconds: float
    diagnostics: LoopDiagnostics
    certification: Certification | None = None
    provenance: Provenance | None = None
    peak_rss_bytes: int | None = None
    wall_max_seconds: float | None = None
    rss_max_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodeTopology):
            raise TypeError(
                "expected a transport NodeTopology for RunMetadata.node,"
                f" got {type(self.node).__name__}"
            )


def normalize_maxrss(ru_maxrss: int) -> int:
    """Normalise a ``getrusage`` ``ru_maxrss`` reading to bytes.

    darwin reports bytes; linux reports kibibytes (×1024). An unknown platform
    raises rather than risk a 1024x error.
    """
    if sys.platform == "darwin":
        return int(ru_maxrss)
    if sys.platform.startswith("linux"):
        return int(ru_maxrss) * 1024
    raise RuntimeError(
        f"ru_maxrss unit convention unknown for platform {sys.platform!r}"
    )


def collect_provenance(
    solver_backend: str,
    *,
    resolved_backend: str | None = None,
) -> Provenance:
    """Probe the run's environment fingerprint."""
    try:
        from mpi4py import MPI  # noqa: PLC0415

        mpi_lib: str | None = MPI.Get_library_version()
    except ImportError:
        mpi_lib = None

    try:
        cfg = np.show_config(mode="dicts")
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        name = str(b.get("name", ""))
        ver = str(b.get("version", ""))
        blas: str | None = (
            f"{name}/{ver}"
            if name and ver and ver != "unknown"
            else (name or None)
        )
    except Exception:
        blas = None

    try:
        import gurobipy as grb  # noqa: PLC0415

        v = grb.gurobi.version()
        gurobi_version: str | None = ".".join(str(x) for x in v)
    except Exception:
        gurobi_version = None

    return Provenance(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        platform=platform.platform(),
        solver_backend=solver_backend,
        resolved_backend=resolved_backend,
        mpi_lib=mpi_lib,
        blas=blas,
        gurobi_version=gurobi_version,
    )


def peak_rss_bytes() -> int:
    """This process's peak resident set size in bytes (one local ``getrusage``)."""
    return normalize_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
