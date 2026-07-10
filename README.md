# combRUM

`combRUM` estimates random-utility models where each observed choice solves

```math
d_i^* \in \arg\max_{d \in \mathcal C_i \subset \{0,1\}^M}
\phi_i(d)^\top \theta + \varepsilon_i(d).
```

The researcher supplies the model-specific parts: how to compute features
$`\phi_i(d)`$ for a candidate choice $`d`$, and how to solve the
combinatorial optimization at a candidate parameter vector. combRUM estimates
$`\theta`$ by row generation, provides bootstrap inference, and can distribute
large runs across multiple processes or compute nodes.

combRUM is designed for models where $`\mathcal C_i`$ is too large to enumerate
(already with $`M=50`$ items and $`\mathcal C_i = \{0,1\}^M`$, there are
$`2^{50} \approx 1.1 \times 10^{15}`$ possible choices!). Estimation proceeds
by row generation: combRUM iteratively queries the researcher's custom oracle
and uses the returned choices to build the linear program.

## Install

```bash
git clone https://github.com/enzodipasquale/combRUM
cd combRUM
python -m pip install ".[highs]"
```

```python
import combrum as cb
```

## First Run

```bash
python examples/quickstart.py
```

This script builds a small bundle-choice model, estimates the coefficients, and
runs a multiplier bootstrap. The same example is explained step by step in
`notebooks/01_quickstart.ipynb`.

## Using combRUM

To use combRUM, specify two model-specific pieces:

- an `Oracle` that solves the combinatorial optimization for a given parameter
  vector
- a `FeatureMap` that computes the priced-row pair
  $`(\phi_i(d), \varepsilon_i(d))`$ for a choice $`d \in \{0,1\}^M`$

Serial runs pass observed choices and simulation draws through `cb.Data`, then
call `cb.estimate(...)`. For the serial bootstrap, keep the same `Model` and
`Data` and call `cb.bootstrap(...)`.

On a high-performance computing (HPC) cluster, the distributed entry points run
row generation across MPI ranks. Pass `cb.MpiTransport` to
`cb.estimate_distributed(...)` and `cb.bootstrap_distributed(...)` to communicate
through MPI. The worked examples and notebooks show both paths.

## Distributed Runs

combRUM supports distributed execution with MPI (Message Passing Interface),
through `mpi4py`. In distributed row generation, ranks work in parallel on the
expensive part of the computation: solving simulated agents' choice problems.
Each rank calls the oracle for its assigned simulated agents, and combRUM
combines the returned choices to update the linear program.

```bash
python -m pip install ".[examples,mpi]"
mpiexec -n 4 python examples/blp_bundle_demand.py --transport mpi
```

For large runs, `Transport` also provides data-movement helpers. Use
`transport.node_shared(...)` for arrays shared by ranks on the same compute
node, and `transport.scatter_by_agent(...)` for arrays indexed by simulated
agent.

## Worked Examples

The notebooks are the best place to continue after the first run. Install the
notebook dependencies with:

```bash
python -m pip install ".[notebooks]"
```

- `notebooks/01_quickstart.ipynb`: a small combRUM example, with estimation,
  bootstrap, and a warm-started follow-up fit.
- `notebooks/02_blp_bundle_demand.ipynb`: bundle demand with endogenous prices
  and quadratic knapsack choice problems.
- `notebooks/03_network_formation.ipynb`: directed network formation with
  reciprocity and a min-cut demand oracle.
- `notebooks/04_unitdemand_blp_large.ipynb`: BLP inversion with many agents per
  market, an outside option, and market-item fixed effects.
- `notebooks/05_peer_effects_large_network.ipynb`: peer effects on a large
  network, with estimation of a nonlinear shock-correlation parameter.
- `notebooks/06_combinatorial_auction.ipynb`: a combinatorial auction example
  that uses combRUM to find equilibrium prices and an allocation.

The `examples/` scripts provide command-line versions of the worked examples.
See `examples/README.md`.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
