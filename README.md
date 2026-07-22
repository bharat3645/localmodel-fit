# localmodel-fit

[![CI](https://github.com/bharat3645/localmodel-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/bharat3645/localmodel-fit/actions/workflows/ci.yml)

Local-LLM performance advisor: given your hardware and a model size, predicts
**decode** tokens/sec (memory-bandwidth-bound) and **prefill** tokens/sec plus
time-to-first-token (compute-bound), shows which quantizations actually fit,
and picks the best one. MoE-aware. Single binary, zero dependencies, published
[methodology](METHODOLOGY.md) and a [benchmark harness](bench/) that checks the
predictions against real measured runs.

```
$ localmodel-fit --hw m4-pro --model 8b
Apple M4 Pro: 273.0 GB/s peak, 48 GB memory, efficiency 0.70
model: 8.0B parameters

QUANT        WEIGHTS  FITS       DECODE
F16         16.00 GB   yes      11.9 t/s
Q8_0         8.50 GB   yes      22.5 t/s
Q6_K         6.59 GB   yes      29.0 t/s
Q5_K_M       5.69 GB   yes      33.6 t/s
Q4_K_M       4.85 GB   yes      39.4 t/s
...

best fit: F16 (16.00 GB), predicted 11.9 tok/s decode
```

("best" = highest-quality quant that fits; speed/quality trade-off is
yours — the table shows the whole frontier.)

## Usage

    localmodel-fit --hw <preset> --model <8b|70b|350m> [--efficiency 0.7] [--json]
    localmodel-fit --bandwidth 273 --mem 24 --model 8b   # custom hardware
    localmodel-fit --list-hw

Presets cover Apple M1-M4 tiers, RTX 3090/4090/5090, and dual-channel
DDR4/DDR5. `--mem` overrides a preset's default capacity (Apple configs
vary).

### Mixture-of-Experts models

Add `--active-params` to get correct decode predictions for MoE models:
every expert stays resident in memory (so the fit/footprint check still
uses `--model`'s total count), but only the routed subset is read per
token, which is what actually bounds decode speed.

```
$ localmodel-fit --hw m2-ultra --model 46.7b --active-params 12.9b   # Mixtral-8x7B
Apple M2 Ultra: 800.0 GB/s peak, 192 GB memory, efficiency 0.70
model: 46.7B parameters total, 12.9B active per token (MoE)

QUANT        WEIGHTS  FITS       DECODE
F16         93.40 GB   yes      21.7 t/s
...
```

Using the 46.7B total for decode speed too (as if it were dense) would
underestimate it by ~3.6x here - the gap the roadmap's "MoE-aware
predictions" item used to leave open.

### Prefill (prompt processing) and latency

Decode is memory-bandwidth-bound; **prefill is compute-bound** — the whole
prompt runs through one batched forward pass, so throughput is set by FP16
compute, not bandwidth. With an FP16-compute figure for the hardware (built
into the Apple/NVIDIA presets, or `--flops`), the tool adds a prefill rate
and, given prompt/generation lengths, time-to-first-token and end-to-end
latency:

```
$ localmodel-fit --hw m4-pro --model 8b --prompt-tokens 2048 --gen-tokens 256
...
best fit: F16 (16.00 GB), predicted 11.9 tok/s decode

prefill: 266 tok/s prompt processing (compute-bound, MFU 0.50, quant-independent)
  TTFT for a 2048-token prompt: 7.71 s
  end-to-end for +256 generated tokens (F16 decode): 29.14 s
```

Prefill throughput is quant-independent (the matmul FLOP count is set by the
model dimensions, not the on-disk format) and, for MoE, scales with
`--active-params` just like decode. Formula: `prefill tok/s =
flops · mfu / (2 · params)` (forward pass ≈ `2·N·P` FLOPs, Kaplan et al. 2020).

## Does it actually predict? (benchmark harness)

[`bench/`](bench/) is a real harness that measures prefill and decode
throughput via [ollama](https://ollama.com) and compares them to these
predictions, reporting the machine's *achieved* efficiency and MFU. On this
project's own M4, measured prefill scales as `1/params` to within ~2% and the
1.5B decode lands within ~10–15% of the default — see
[bench/RESULTS.md](bench/RESULTS.md) for the numbers and honest caveats.

```
go run ./bench -model qwen2.5:1.5b -hw m4 -params 1543714304
```

## Why bandwidth (for decode)?

Decode reads ~all weights per generated token, so memory bandwidth — not
FLOPS — is the binding constraint for single-stream local generation. The
formulas, the 0.7 efficiency / 0.5 MFU defaults, the bits-per-weight table,
the FP16-compute presets, and the 1.15x fit overhead are all documented with
their limitations and primary sources in [METHODOLOGY.md](METHODOLOGY.md) —
including what this still does not model (attention's N² term at long context,
batching, CPU/GPU splits).

## Honest framing

These are screening estimates for hardware decisions, not benchmarks.
Real-world numbers vary with runtime, kernel, context length, and thermals.
Measure before you buy.

## Roadmap

- Done: MoE-aware decode, prefill (compute-bound) model, and a benchmark
  harness that checks predictions against real runs (see [bench/](bench/)).
- Attention's O(N^2) term for accurate long-context prefill
- Batched-serving regime (batch > 1 amortizes weight reads)
- CPU/GPU offload split

## License

MIT
