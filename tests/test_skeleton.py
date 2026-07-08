from __future__ import annotations

import importlib
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import combrum

# Modules under combrum.*, counted from the source tree (packages by their
# __init__.py). walk_packages must discover and import exactly this many.
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
    # Walk the whole tree so an import-time error in any submodule raises.
    walked = [
        info.name
        for info in pkgutil.walk_packages(combrum.__path__, prefix="combrum.")
    ]
    for name in walked:
        importlib.import_module(name)
    # Both counts must match the pin: a module the walk skips, or an unnoticed
    # addition/removal, moves one of them.
    assert _count_source_submodules() == _EXPECTED_SUBMODULES
    assert len(walked) == _EXPECTED_SUBMODULES
    # Solver-import laziness is checked in test_core_import_pulls_no_solver;
    # this shared process may already have gurobipy loaded by an earlier test.


def test_core_import_pulls_no_solver(tmp_path: Path) -> None:
    # `import combrum` must not pull in gurobipy or highspy: the master modules
    # import them lazily so the package works without a solver install/license.
    # Only a fresh subprocess can show this (this process may already have them
    # loaded); the neutral cwd keeps import resolution off the repo root. The
    # probe then walk-imports every submodule -- loading both master modules --
    # and re-checks sys.modules, so the laziness must live inside functions,
    # not the module body.
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
    assert result.returncode == 0
