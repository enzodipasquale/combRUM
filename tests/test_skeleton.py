from __future__ import annotations

import importlib
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import combrum

# Count of importable modules under the combrum.* namespace, derived
# independently from the source tree (every .py file, with packages counted by
# their __init__.py). walk_packages must discover and import exactly these.
_EXPECTED_SUBMODULES = 50


def _count_source_submodules() -> int:
    root = Path(combrum.__file__).resolve().parent
    modules: set[str] = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(root).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            if not parts:  # the combrum root package itself
                continue
        else:
            parts[-1] = parts[-1][: -len(".py")]
        modules.add("combrum." + ".".join(parts))
    return len(modules)


def test_version() -> None:
    version = combrum.__version__
    assert isinstance(version, str)
    assert version
    assert re.fullmatch(r"\d+\.\d+\.\d+(a|b|rc)?\d*", version)


def test_package_walk_imports_clean() -> None:
    # Walk (not just import the root) so every subpackage is imported; any
    # import-time error in any module raises here.
    walked = [
        info.name
        for info in pkgutil.walk_packages(combrum.__path__, prefix="combrum.")
    ]
    for name in walked:
        importlib.import_module(name)
    # The walk must discover exactly the modules present in the source tree.
    # Pinning the count catches a module that silently fails to import (and is
    # skipped) as well as accidental additions/removals; the expected value is
    # cross-checked against the source files rather than against the walk.
    assert _count_source_submodules() == _EXPECTED_SUBMODULES
    assert len(walked) == _EXPECTED_SUBMODULES
    # The lazy-solver-import property (importing combrum.masters.gurobi must not
    # force-load gurobipy) cannot be checked here: this shared pytest process may
    # already have gurobipy loaded by an earlier solver test. It is asserted
    # hermetically in a fresh subprocess by test_core_import_pulls_no_solver.


def test_core_import_pulls_no_solver(tmp_path: Path) -> None:
    # A bare `import combrum` must not eagerly pull in either optional solver
    # backend: gurobipy is imported lazily inside combrum.masters.gurobi and
    # highspy inside combrum.masters.highs, so the package stays importable
    # without a Gurobi or HiGHS install/license. A fresh subprocess is the only
    # reliable probe (this pytest process may already have those modules
    # loaded); a neutral cwd keeps import resolution off the repo root. The
    # probe also walk-imports every submodule -- which forces both master
    # modules to load -- and re-checks that neither solver library leaks into
    # sys.modules, so the laziness lives inside functions, not the module body.
    src = Path(__file__).resolve().parents[1] / "src"
    probe = (
        "import sys, importlib, pkgutil\n"
        "import combrum\n"
        "assert 'combrum.masters.gurobi' not in sys.modules, "
        "'import combrum eagerly loaded the gurobi master'\n"
        "assert 'gurobipy' not in sys.modules, "
        "'import combrum eagerly loaded gurobipy'\n"
        "assert 'combrum.masters.highs' not in sys.modules, "
        "'import combrum eagerly loaded the highs master'\n"
        "assert 'highspy' not in sys.modules, "
        "'import combrum eagerly loaded highspy'\n"
        "for info in pkgutil.walk_packages(combrum.__path__, prefix='combrum.'):\n"
        "    importlib.import_module(info.name)\n"
        "assert 'combrum.masters.gurobi' in sys.modules, "
        "'walk failed to import the gurobi master -- probe would be unexercised'\n"
        "assert 'gurobipy' not in sys.modules, "
        "'importing the gurobi master eagerly loaded gurobipy'\n"
        "assert 'combrum.masters.highs' in sys.modules, "
        "'walk failed to import the highs master -- probe would be unexercised'\n"
        "assert 'highspy' not in sys.modules, "
        "'importing the highs master eagerly loaded highspy'\n"
    )
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )
    # Guard against the probe silently doing nothing (e.g. combrum failing to
    # import in the child would surface here, not as an unexercised pass).
    assert result.returncode == 0
