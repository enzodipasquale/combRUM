# Changelog

## 0.1.0

Initial public release of combRUM for combinatorial random-utility estimation:

- `Model`, `Data`, and `Oracle` objects for defining an estimation problem.
- `estimate`, `bootstrap`, `estimate_distributed`, and `bootstrap_distributed`
  entry points.
- `max_live_reps` for limiting the number of distributed bootstrap replications
  running in one wave.
- Cut policies `AddAll`, `PurgeInactive`, and `SlackStrip`.
- Progress reporting through `ActivityConfig`.
- Runnable notebooks and scripts for the applied examples.
