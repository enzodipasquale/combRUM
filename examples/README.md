# Examples

The scripts in this directory are command-line versions of the worked examples
in `notebooks/`.

Install the dependencies for all examples:

```bash
python -m pip install ".[examples]"
```

## Scripts

- `quickstart.py`: small bundle-choice model with estimation and bootstrap.
- `unitdemand_blp_large.py`: BLP inversion with many agents per market, an
  outside option, and item fixed effects.
- `blp_bundle_demand.py`: bundle demand with endogenous prices and quadratic
  knapsack choice problems.
- `network_formation.py`: directed network formation with a min-cut demand
  oracle.
- `peer_effects_large_network.py`: peer effects on a large undirected network,
  with estimation of a nonlinear shock-correlation parameter.

## MPI

The BLP, network-formation, peer-effects, and unit-demand scripts can be
launched with MPI:

```bash
python -m pip install ".[examples,mpi]"
mpiexec -n 4 python examples/blp_bundle_demand.py --transport mpi
```
