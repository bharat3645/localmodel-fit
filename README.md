# localmodel-fit

Memory-bandwidth-aware local LLM advisor: given your hardware and a model
size, predicts decode tokens/sec, shows which quantizations actually fit,
and picks the best one. Single binary, zero dependencies, published
[methodology](METHODOLOGY.md).

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

## Why bandwidth?

Decode reads ~all weights per generated token, so memory bandwidth — not
FLOPS — is the binding constraint for single-stream local inference. The
formula, the 0.7 efficiency default, the bits-per-weight table, and the
1.15x fit overhead are all documented with their limitations in
[METHODOLOGY.md](METHODOLOGY.md) — including what this deliberately does
not model (prefill, MoE, batching, CPU/GPU splits).

## Honest framing

These are screening estimates for hardware decisions, not benchmarks.
Real-world numbers vary with runtime, kernel, context length, and thermals.
Measure before you buy.

## Roadmap

- MoE-aware predictions (active-parameter counts)
- Prefill (compute-bound) model
- CI benchmark harness: measured vs predicted on known runners

## License

MIT
