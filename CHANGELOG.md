# Changelog

## 0.1.0

Initial public release of combRUM for combinatorial random-utility estimation:

- The `Model` / `Data` / `Oracle` API with `estimate`, `bootstrap`, and
  split-axis `estimate_distributed`.
- Serial bootstrap and split-axis distributed bootstrap entry points.
- Configurable `max_live_reps` for distributed bootstrap wave memory control.
- Public cut policies `AddAll`, `PurgeInactive`, and `SlackStrip`.
- Root-local stdout activity reporting through `ActivityConfig`.
- Runnable notebooks and scripts for the applied examples.
