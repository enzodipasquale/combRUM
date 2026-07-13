"""Suite-local pytest configuration."""

from __future__ import annotations

import importlib.util
import os
import shutil

import pytest


def _mpirun_path() -> str | None:
    return os.environ.get("COMBRUM_MPIRUN") or shutil.which("mpirun")


def _mpi_available() -> bool:
    return (
        _mpirun_path() is not None
        and importlib.util.find_spec("mpi4py") is not None
    )


# Markers are declared in pyproject.toml [tool.pytest.ini_options] so
# --strict-markers can reject typos.


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _mpi_available():
        skip = pytest.mark.skip(
            reason="requires mpirun (PATH or COMBRUM_MPIRUN) and mpi4py"
        )
        for item in items:
            if "requires_mpi" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def mpirun_path() -> str:
    path = _mpirun_path()
    if path is None:
        pytest.skip("mpirun is not discoverable on this host")
    return path
