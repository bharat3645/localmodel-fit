# Methodology

## The model

Single-stream LLM decode is **memory-bandwidth-bound**: producing one token
requires streaming approximately every weight through the memory system once
(matrix-vector products dominate; cache reuse across a single token is
minimal for models much larger than L2/SLC).

    predicted decode tok/s = (peak_bandwidth_GBs * 1e9 * efficiency) / weight_bytes

    weight_bytes = params * bits_per_weight / 8

## Efficiency default: 0.7

Real decode loops do not reach spec-sheet bandwidth. Across published
llama.cpp benchmarks on Apple silicon and discrete GPUs, achieved fraction
of peak typically lands between 0.5 and 0.85 depending on kernel, context
length, and thermal state. We default to 0.7 and expose `--efficiency`.

## Bits-per-weight table

Approximate llama.cpp GGUF costs including scale/zero-point overhead,
derived from quantize output sizes (they vary slightly per architecture):
F16 16.0 / Q8_0 8.5 / Q6_K 6.59 / Q5_K_M 5.69 / Q4_K_M 4.85 / Q4_0 4.55 /
Q3_K_M 3.91 / Q2_K 3.35.

## Fit overhead: 1.15x

Weights are not the whole footprint: KV cache at moderate context (4-8k),
compute buffers, and allocator slack add roughly 10-20% for common configs.
We require `weights * 1.15 <= memory` to call something a fit. Long contexts
or large batch sizes need more; this is a screening threshold, not a
guarantee.

## Speculative decoding hint

For models >= 7B, pairing a same-family draft model 10-20x smaller
typically yields 1.3-2.0x decode speedups (published speculative-decoding
results cluster in that band; acceptance-rate dependent). The hint is a
heuristic pointer, not a prediction.

## Known limitations (deliberate v0.1 scope)

- **Prefill is not modeled.** Prompt processing is compute-bound and follows
  different math; figures here are steady-state generation only.
- **MoE is not modeled.** Mixture-of-experts models read only active experts
  per token; using total params underestimates their speed. Treat MoE
  predictions as lower bounds or use active-param counts.
- **Batch size 1.** Batched serving amortizes weight reads and changes the
  regime entirely.
- **No CPU/GPU split.** Partial offload blends two bandwidth domains; not
  modeled.
- Predictions are estimates for sanity-checking hardware choices, not
  benchmarks. Measure before you buy.
