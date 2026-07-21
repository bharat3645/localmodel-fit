# Benchmark harness

Validates `localmodel-fit`'s predictions against **real measured runs**. It
drives [ollama](https://ollama.com), reads ollama's own timing counters, and
prints measured-vs-predicted throughput plus the machine's *achieved*
bandwidth-efficiency and compute-MFU.

This is the empirical anchor for the methodology: the tool does not just assert
`tok/s = bandwidth·eff/bytes` and `tok/s = flops·mfu/(2·params)` — the harness
measures the real numbers so the residuals are visible and honest. Captured
results for this project's hardware are in [RESULTS.md](RESULTS.md).

## Live run

Needs ollama running and the model pulled (`ollama pull qwen2.5:1.5b`):

```
go run ./bench -model qwen2.5:1.5b -hw m4 -params 1.54b
```

The harness (`main.go` + `harness.go`):

1. **warms up** with a throwaway prompt (loads weights; a per-run nonce stops
   the measured call reusing a cached KV prefix);
2. measures **prefill** with a long input prompt (compute-bound regime);
3. measures **decode** with a long-output prompt capped at 128 tokens (so the
   per-token rate isn't dominated by first-token noise);
4. reads `prompt_eval_*` / `eval_*` counters (excludes model-load time);
5. predicts from the `fit` package and prints the comparison.

Flags: `-params` (real parameter count — **not** the rounded name), optional
`-active-params` (MoE), `-quant` (default `Q4_K_M`, ollama's default), `-flops`
(FP16 TFLOPS override), `-efficiency`, `-mfu`, `-url`.

## Offline replay (no ollama, no download)

```
go run ./bench -response bench/testdata/qwen2.5-0.5b-response.json -hw m4 -params 0.494b -flops 3.5
```

## CI

CI does **not** download multi-GB weights on every push — that would be slow
and flaky. Instead:

- `go test ./bench/` validates the harness math against a **recorded real
  response** (`testdata/qwen2.5-0.5b-response.json`, captured on an M4) — fully
  deterministic, no ollama required.
- The live measurement is run locally (or on a self-hosted runner with a GPU),
  and its output is committed to [RESULTS.md](RESULTS.md).

This keeps CI fast and honest: it proves the analysis logic is correct against
real data, without pretending the runner benchmarks a GPU.
