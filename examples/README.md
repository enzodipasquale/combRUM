# Examples

Install the example dependencies from the repository root:

```bash
python -m pip install ".[examples]"
```

Run the smallest example:

```bash
python examples/tiny_oracle.py
```

The same script uses MPI automatically when launched with `mpiexec`:

```bash
mpiexec -n 2 python examples/tiny_oracle.py
```

MPI runs need the `mpi` extra: `python -m pip install ".[examples,mpi]"`.

The larger scripts mirror the worked notebooks:

- `unitdemand_blp_large.py`: BLP inversion with many agents per market, an
  outside option, and item fixed effects.
- `blp_bundle_demand.py`: bundle demand with endogenous prices and quadratic
  knapsack choice problems.
- `network_formation.py`: directed network formation with reciprocal links and
  a min-cut demand oracle.
- `peer_effects_large_network.py`: peer effects on a large undirected network,
  with estimation of a nonlinear shock-correlation parameter.

For the peer-effects MPI example:

```bash
mpiexec -n 4 python examples/peer_effects_large_network.py --transport mpi
```

The distributed APIs are `cb.estimate_distributed(...)` and
`cb.bootstrap_distributed(...)`. Use them when observed rows are split across
MPI ranks. Each rank prices its own simulated agents and provides observed
features with `setup_observed(...)` and `observed_features_batch(...)`.

The larger MPI examples use `transport.node_shared(...)` for large arrays that
can be shared by ranks on the same compute node. Each rank then indexes those
arrays by observation id or `local_ids`.

Distributed bootstrap also accepts `max_live_reps` to control how many
replications run in one wave.

Run `python examples/<script>.py --help` to inspect the command-line options on
the larger examples.
