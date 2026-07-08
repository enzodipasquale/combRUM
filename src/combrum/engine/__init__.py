"""Single-replication fitting engine.

The public entry points are :func:`estimate`, :func:`estimate_distributed`,
the :class:`PersistentMasterFit` driver, and :class:`LoopConfig`. Sibling
modules reach the lower-level
composition pieces (context assembly, the row-generation loop, certification,
and the step/diagnostics types) by their fully qualified submodule paths; the
names re-exported below are used internally and are not public API.
"""

from __future__ import annotations

from combrum.engine.context_builder import (
    build_fit_context as build_fit_context,
)
from combrum.engine.context_builder import (
    master_environment as master_environment,
)
from combrum.engine.context_builder import (
    resolve_master_backend as resolve_master_backend,
)
from combrum.engine.driver import (
    LoopConfig,
)
from combrum.engine.driver import (
    LoopDiagnostics as LoopDiagnostics,
)
from combrum.engine.driver import (
    LoopOutcome as LoopOutcome,
)
from combrum.engine.driver import (
    run_fit as run_fit,
)
from combrum.engine.estimate import estimate, estimate_distributed
from combrum.engine.persistent import (
    PersistentFitResult as PersistentFitResult,
)
from combrum.engine.persistent import (
    PersistentMasterFit,
)

__all__ = [
    "LoopConfig",
    "PersistentMasterFit",
    "estimate",
    "estimate_distributed",
]
