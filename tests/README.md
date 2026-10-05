# Tests

Install the package with its test dependencies and run pytest from the repository root:

```bash
pip install -e ".[test,highs]"
pytest tests -m "not slow and not requires_mpi"
```

This is the selection the main continuous-integration job runs.

## Tiers

Tests are grouped by markers so heavier tiers can be selected or skipped:

- default — unit, contract, and small end-to-end tests; runs in seconds.
- `slow` — repeated real solves, timing/RSS probes, and larger synthetic sweeps.
- `requires_mpi` — shells out to `mpirun`; auto-skipped unless `mpirun` and `mpi4py` are available. CI runs its `mpi_smoke` subset on every push.

## Solver backends

Tests parametrized over the master backends run against whichever solver is
installed. The HiGHS backend is pulled in by the `highs` extra. Gurobi tests
are skipped unless `gurobipy` is importable and a license is present; the
quadratic penalty path requires Gurobi, since the HiGHS master does not
implement it.
