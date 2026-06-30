# Examples

Install the example dependencies from the repository root:

```bash
python -m pip install ".[examples]"
```

Run the smallest public-API example:

```bash
python examples/tiny_oracle.py
```

The same script uses MPI automatically when launched with `mpiexec`:

```bash
mpiexec -n 2 python examples/tiny_oracle.py
```

MPI runs need the `mpi` extra: `python -m pip install ".[examples,mpi]"`.

The larger scripts mirror the worked notebooks:

- `unitdemand_blp_large.py`: market-wise OneSlack estimation with an outside
  option, item fixed effects, and many agents per market.
- `blp_bundle_demand.py`: multi-market bundle demand with endogenous prices and
  quadratic-knapsack pricing.
- `network_formation.py`: directed network formation with reciprocal links and
  a min-cut pricing oracle.
- `peer_effects_large_network.py`: large undirected peer-effects game with a
  persistent NSlack master over a sigma grid.

For the peer-effects MPI example:

```bash
mpiexec -n 4 python examples/peer_effects_large_network.py --transport mpi
```

The public distributed APIs are `cb.estimate_distributed(...)` and
`cb.bootstrap_distributed(...)`. They are for sharded NSlack jobs where each
rank owns observed rows and prices the corresponding simulated agents. They do
not accept dense `Data` or `observed_bundles`; the model must expose
`setup_observed(...)`, `observed_features_batch(...)`, and a batched pricing
oracle. Distributed bootstrap also accepts `max_live_reps` to bound how many
replications are live in one wave.

Run `python examples/<script>.py --help` to inspect the command-line options on
the larger examples.
