# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-21

Initial tagged release.

### Added

- `fit` package: bandwidth-bound decode prediction
  (`tok/s = bandwidth × efficiency / weight_bytes`), 8 llama.cpp
  quantization presets (bits-per-weight including format overhead), 18
  hardware presets (Apple M1–M4 tiers, RTX 3090/4090/5090, DDR4/DDR5
  dual-channel), 1.15× fit-overhead margin, `Advise`/`Best`, and
  parameter-count parsing (`"8b"`, `"350m"`).
- `fit.AdviseMoE(totalParams, activeParams, hw, efficiency)` and the CLI's
  `--active-params` flag: memory footprint and `Fits` are computed from
  total params (every expert stays resident), decode speed from active
  params (only the routed subset is read per token). `Advise` is now
  `AdviseMoE(params, params, ...)` — byte-identical output for dense
  models. Validated against Mixtral-8x7B's public 46.7B/12.9B shape.
- CLI: `--hw`, `--bandwidth`, `--mem`, `--model`, `--efficiency`,
  `--active-params`, `--json`, `--list-hw`.
- `METHODOLOGY.md`: formulas, the 0.7 default efficiency and its
  observed range, and an explicit list of what's not modeled (prefill,
  batching, CPU/GPU split).
- Table-driven tests with hand-computed expectations; CI (gofmt, vet,
  test, build, CLI smoke including an expected-failure path).
