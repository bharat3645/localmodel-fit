# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Real asciinema demo (`demo/localmodel-fit-demo.cast` + driver script
  `demo/run_demo.sh`): dense-model quant/decode table, the MoE-aware
  decode correction (~3.6x on Mixtral-8x7B), and prefill/TTFT/end-to-end
  latency, all from the compiled binary. README gained a "Demo" section.
- `main_test.go`: direct unit tests of the CLI entrypoint (flag parsing,
  error paths, table/JSON output formatting) alongside the existing `fit/`
  and `bench/` package tests. `run` now takes `(args []string, stdout
  io.Writer)` instead of parsing the process-wide `flag.CommandLine` and
  writing straight to `os.Stdout`, so tests can invoke it directly and
  repeatedly with an isolated flag set — behavior for real invocations
  (`main`) is unchanged.
- README: an Installation section (`go install`, build-from-source, and
  prebuilt release binaries) ahead of Usage.
- **Prefill (prompt-processing) model.** `fit.PrefillTokS`,
  `fit.TTFTSeconds`, `fit.EndToEndSeconds`, and CLI flags `--flops`, `--mfu`,
  `--prompt-tokens`, `--gen-tokens`. Prefill is compute-bound —
  `tok/s = flops · mfu / (2 · params)` from the ~`2·N·P` forward-pass FLOP
  count (Kaplan et al. 2020) — so the CLI now also reports prompt-processing
  throughput, time-to-first-token, and end-to-end latency. Prefill is
  quant-independent and, for MoE, scales with `--active-params` like decode.
- **FP16 compute figures on every GPU preset** (`Hardware.FP16TFLOPS`,
  `FP16FLOPS()`), sourced from primary specs and marked honestly in
  METHODOLOGY.md: NVIDIA dense FP32-accumulate Tensor TFLOPS (Blackwell/Ada/
  Ampere whitepapers); Apple M1–M4 with FP16 = FP32 rate (no 2× multiplier —
  verified via metal-benchmarks + KTH arXiv:2502.05317). DDR/CPU presets carry
  no figure and omit prefill unless `--flops` is given. `--list-hw` shows the
  new column.
- **`bench/` benchmark harness.** Measures real prefill/decode throughput via
  ollama and compares against the predictions, reporting the machine's
  achieved bandwidth-efficiency and compute-MFU. Live and offline
  (`-response`) modes; pure analysis logic unit-tested against a recorded real
  response so CI validates it with no ollama and no model download.
  [bench/RESULTS.md](bench/RESULTS.md) records measured-vs-predicted on an M4:
  prefill scales as `1/params` to within ~2%, achieved MFU is size-independent
  (~0.83–0.86), and 1.5B decode is within ~10–15% of the default.

### Changed

- METHODOLOGY.md documents the prefill model, the compute/bandwidth regime
  split, FP16 preset sourcing (with primary-source citations), and updated
  limitations (attention N² and short-prompt caveats replace the old
  "prefill not modeled" note). Default MFU is 0.5 (conservative screening
  midpoint; see RESULTS).

## [0.1.0] - 2026-07-21

Initial tagged release. (Prebuilt binaries for darwin/linux/windows were
built from this tag and attached to the GitHub release after the fact —
the release originally shipped source-only.)

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
