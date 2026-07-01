# combRUM

`combRUM` estimates random-utility models where each choice is a bundle.
Users provide an `Oracle` that solves the combinatorial problem, returning the
optimal bundle at a given parameter vector. combRUM runs row generation on the
implied master LP and returns the estimated structural parameters.

The distribution is named `combRUM`. The import root is lowercase:

```python
import combrum as cb
```

## Install

```bash
git clone https://github.com/enzodipasquale/combRUM
cd combRUM
python -m pip install ".[examples]"
```

Useful extras:

```bash
python -m pip install ".[highs]"      # HiGHS backend only
python -m pip install ".[notebooks]"  # notebook runtime dependencies
python -m pip install ".[mpi]"        # MPI transport for cluster runs
python -m pip install ".[gurobi]"     # Gurobi backend, if licensed
```

HiGHS is the license-free master-LP backend. With `master_backend="auto"`,
combRUM uses Gurobi when it is installed and licensed, otherwise it falls
back to HiGHS.

## Quickstart

Run the examples:

```bash
python examples/tiny_oracle.py
python examples/unitdemand_blp_large.py
python examples/blp_bundle_demand.py
python examples/network_formation.py
python examples/peer_effects_large_network.py
```

The tiny example builds a small `Model` and `Oracle`, estimates it with
`cb.estimate_distributed(...)`, then runs a small bootstrap with
`cb.bootstrap_distributed(...)`.
The unit-demand example builds a market-level demand oracle and feature map for
a large outside-option run with market-item fixed effects kept as integer
parameter columns rather than dense agent covariates. It reports
row-generation logging, parameter recovery, and an IV price-sensitivity
diagnostic from the estimated fixed effects.
The BLP demand example and notebook both show the batched quadratic-knapsack
setting as standalone estimation code, with a Gurobi demand backend and a
license-free HiGHS fallback. The script can also run under MPI.
For MPI runs, launch the standalone scripts with `mpiexec` and pass
`--transport mpi`.

## Distributed Fits

`cb.estimate(...)` and `cb.bootstrap(...)` are the dense public path: pass a
`Data` object with `observed_bundles` on the observation axis `N` and shocks on
the simulation axis `S`.

For data that is already sharded across ranks, use `cb.estimate_distributed(...)`
or `cb.bootstrap_distributed(...)`. These functions do not take `Data` or
`observed_bundles`. Instead:

- pass `n_observations=N` and `n_simulations=S`;
- price global agent ids `0, ..., N*S-1`;
- treat `agent_id % N` as the observed row for the default simulation geometry;
- implement `Oracle.price_batch(theta, local_ids)` for the rank's pricing shard;
- provide an observed-feature surface with
  `setup_observed(transport, observation_ids)` and
  `observed_features_batch(observation_ids)`.

Distributed bootstrap multipliers are drawn on the `N` observed rows, then reused
across the `S` simulated agents belonging to the same observation. The public
distributed entry points currently support `NSlack`; other formulations should
use the serial path until their distributed contracts are audited.

`bootstrap_distributed(max_live_reps=...)` controls how many bootstrap
replications are live in one wave. Higher values use more memory and fewer
waves; lower values use less memory and more wave setup. Callback iteration
indices are wave-local.

## Notebooks

- `notebooks/01_quickstart.ipynb`: runnable public-API quickstart covering
  `Oracle`, `FeatureMap`, `Parameters`, `Model`, `Data`, `estimate`,
  `bootstrap`, and warm-started follow-up fits.
- `notebooks/02_blp_bundle_demand.ipynb`: multi-market bundle demand at the applied
  scale with a batched quadratic-knapsack MIP oracle and BLP-style 2SLS.
- `notebooks/03_network_formation.ipynb`: directed network formation at the
  applied scale with the min-cut pricing oracle.
- `notebooks/04_unitdemand_blp_large.ipynb`: market-wise OneSlack large-`N`
  example with an outside option, three agent-specific covariates, and local
  item fixed effects generated from BLP-style prices and instruments. It runs
  one independent OneSlack estimate per market `t`, keeps each master small,
  and closes with the IV price-sensitivity diagnostic.
- `notebooks/05_peer_effects_large_network.ipynb`: large-network peer-effects
  example with MPI row generation and a persistent NSlack master over a sigma
  grid.
- `notebooks/06_combinatorial_auction.ipynb`: combinatorial auction with
  assignment valuations. This is not an estimation notebook. It uses the
  row-generation algorithm in combRUM to compute Walrasian prices and allocation.

The BLP and network-formation notebooks are substantive applied examples and
can take longer than the quickstart. The OneSlack notebook is the larger
market-wise example for batched oracles, array-backed demand batches, and
market-level decomposition.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
