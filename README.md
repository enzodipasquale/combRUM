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

The tiny example builds a small bundle-choice model, estimates it, and runs a
small bootstrap. It can run either serially or under MPI.

The unit-demand example estimates a BLP-style model with many agents per market,
an outside option, and market-item fixed effects. It reports parameter recovery
and a simple IV price-sensitivity diagnostic.

The BLP bundle-demand example estimates price sensitivity in a market where
agents choose bundles under a quadratic knapsack problem. It uses Gurobi when
available and otherwise falls back to HiGHS. The script can also run under MPI.

For MPI runs, launch the standalone scripts with `mpiexec` and pass
`--transport mpi`.

## Distributed Fits

Use `cb.estimate(...)` and `cb.bootstrap(...)` when the observed bundles and
simulation draws fit in one Python process. Pass a `Data` object with
`observed_bundles` indexed by observation and shocks indexed by observation and
simulation draw.

Use `cb.estimate_distributed(...)` or `cb.bootstrap_distributed(...)` when the
data is already split across MPI ranks. The distributed entry points do not take
a `Data` object. Instead:

- Pass `n_observations=N` and `n_simulations=S`.
- Price global agent ids `0, ..., N*S-1`.
- Use `agent_id % N` to recover the observation index.
- Implement `Oracle.price_batch(theta, local_ids)` for the agent ids owned by
  the rank.
- Provide observed features with `setup_observed(transport, observation_ids)`
  and `observed_features_batch(observation_ids)`.

Distributed bootstrap weights are drawn for the `N` observed rows and reused
across the `S` simulation draws for the same observation. The distributed entry
points currently support `NSlack`. Use the serial path for other formulations.

`bootstrap_distributed(max_live_reps=...)` controls how many bootstrap
replications run in one wave. Higher values use more memory and fewer waves.
Lower values use less memory and more wave setup.

## Notebooks

- `notebooks/01_quickstart.ipynb`: the smallest complete combRUM example, with
  estimation, bootstrap, and a warm-started follow-up fit.
- `notebooks/02_blp_bundle_demand.ipynb`: bundle demand with endogenous prices,
  quadratic knapsack choice problems, and a BLP-style 2SLS second stage.
- `notebooks/03_network_formation.ipynb`: directed network formation with
  reciprocity and a min-cut demand oracle.
- `notebooks/04_unitdemand_blp_large.ipynb`: BLP inversion with many agents per
  market, an outside option, and market-item fixed effects.
- `notebooks/05_peer_effects_large_network.ipynb`: peer effects on a large
  network, with estimation of non-linear shocks parameters.
- `notebooks/06_combinatorial_auction.ipynb`: a combinatorial auction example
  that uses combRUM to find equilibrium prices and an allocation.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
