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

## Prefill (prompt processing): compute-bound

Decode streams the weights once per generated token, so it is
bandwidth-bound. Prefill is different: the whole prompt is processed in a
single forward pass, which is a batched matrix-matrix multiply over `N`
prompt tokens. Each weight is reused `N` times, so arithmetic intensity is
high and the binding constraint is **compute throughput (FLOP/s), not
bandwidth**.

A forward pass over `N` tokens through a model with `P` active parameters
costs approximately `2·N·P` FLOPs — one multiply and one add per parameter
per token. This is the standard estimate from Kaplan et al. 2020 (*Scaling
Laws for Neural Language Models*), whose training cost `C ≈ 6ND` decomposes
as `2ND` forward + `4ND` backward. Per-token prefill cost is therefore ~`2·P`
FLOPs, independent of prompt length in this regime, so:

    prefill tok/s = flops · mfu / (2 · P_active)

- **MFU** (model FLOPs utilization) is the achieved fraction of peak,
  analogous to the `efficiency` used for decode. We default to **0.5**, a
  deliberately conservative screening midpoint — measured prefill MFU is
  hardware-, kernel-, and model-dependent, so absolute prefill numbers are
  softer than the decode ones. Expose it via `--mfu` (and `--flops`). The
  `bench/` harness measures the achieved value on real hardware (see below),
  so this default is checked, not just asserted: on this project's own M4,
  measured prefill MFU on small models was ~0.83 (tight across two model
  sizes), i.e. the 0.5 default under-predicts M4 prefill — the safe direction
  for a screening tool.
- **Quantization does not change prefill speed** to first order: llama.cpp
  dequantizes to run the matmul, so the FLOP count is set by the model
  dimensions, not the on-disk format. Prefill throughput is reported once,
  not per-quant.
- **MoE uses active params.** Only the routed experts run each forward pass,
  exactly as for decode, so prefill scales with `--active-params`.

### What this omits
- **Attention's O(N²) term.** The `2ND` count is the dense matmul work; the
  attention score/output cost grows with `N²` and becomes non-negligible at
  long context. Short-to-moderate prompts are dominated by the linear term.
- **Short prompts.** Below the machine's roofline ridge point (a handful of
  tokens) prefill falls back to memory-bound and the compute ceiling
  over-predicts. The harness prompts are ~450 tokens, well into the
  compute-bound regime.

### TTFT and end-to-end latency
Time-to-first-token is `prompt_tokens / prefill_tok_s`; end-to-end latency
for a generation is `TTFT + gen_tokens / decode_tok_s`. Both ignore
sampling/scheduling overhead and KV-cache growth during decode.

## FP16 compute presets

Prefill prediction needs a peak FP16 throughput per machine. These are
sourced differently per vendor and marked honestly:

- **NVIDIA GPUs:** published FP16 Tensor Core TFLOPS, **dense** (non-sparse),
  **FP32 accumulate** — the realistic figure for inference (consumer GeForce
  cards run FP32-accumulate at half the FP16-accumulate rate, and the 2×
  "sparsity" / FP4 "AI TOPS" marketing numbers do not apply to dense LLM
  matmuls). RTX 3090 / 4090 / 5090 = 71.2 / 165.2 / 209.5 TFLOPS, from the
  NVIDIA RTX Blackwell GPU Architecture Whitepaper, Appendix A (which lists
  all three), cross-checked against the Ada and Ampere GA102 whitepapers.
- **Apple Silicon:** **FP16 runs at the same rate as FP32** on M1–M4 (128
  ALUs/core, 256 FLOP/core/cycle for both — there is *no* 2× FP16 multiplier;
  some LLM trackers wrongly apply one, which is pre-M1 A-series behavior).
  Verified via Philip Turner's `metal-benchmarks` and the peer-reviewed KTH
  "Apple vs. Oranges" measurements (arXiv:2502.05317). Base chips are
  published/measured; Pro/Max/Ultra are derived by scaling GPU-core count at
  the tier clock, so treat those as ±10–15%. The `m4` preset (4.26 TFLOPS) is
  the KTH-measured figure and was cross-checked against measured prefill on
  this project's own M4: the ~3.5 TFLOP/s *achieved* rate the harness
  recovered is consistent with the 4.26 peak (MFU ~0.83) and rules out the
  conservative 3.76-TFLOPS base-clock reading, which would require an
  implausible >0.9 MFU. See `bench/RESULTS.md`.
- Presets with no usable FP16 figure (the DDR/CPU entries) carry `0` and
  simply omit the prefill prediction unless you pass `--flops`.

Primary sources: Kaplan et al. 2020 (arXiv:2001.08361, §2.1, the 2N/6N
convention and the explicit exclusion of the attention term); NVIDIA
"Mastering LLM Techniques: Inference Optimization" and Yuan et al. 2024
(arXiv:2402.16363) for the prefill-compute-bound / decode-memory-bound split;
NVIDIA architecture whitepapers for GPU FP16; Turner metal-benchmarks and KTH
arXiv:2502.05317 for Apple GPU FP16=FP32; Qwen2.5 `config.json` + Hugging Face
`safetensors` metadata for the exact parameter counts used in the harness.

## Speculative decoding hint

For models >= 7B, pairing a same-family draft model 10-20x smaller
typically yields 1.3-2.0x decode speedups (published speculative-decoding
results cluster in that band; acceptance-rate dependent). The hint is a
heuristic pointer, not a prediction.

## Known limitations

- **Prefill is a compute ceiling, not a guarantee.** It assumes the
  compute-bound regime (moderate prompt length) and the linear `2ND` FLOP
  term; see the prefill section above for what it omits (attention `N²`,
  very short prompts) and how the harness validates it.
- **FP16 peak is less clean than bandwidth.** Memory bandwidth is a
  spec-sheet number; peak FP16 FLOPS, especially for Apple GPUs, is derived
  and approximate. Prefill predictions inherit that uncertainty — the
  harness measures the real achieved MFU rather than trusting the peak.
- **MoE needs `--active-params`.** Mixture-of-experts models read/run only
  the routed experts per token, not the full parameter count, for **both**
  decode and prefill; `--model` alone (as for a dense model) underestimates
  both speeds. Passing `--active-params` alongside `--model` predicts
  decode and prefill from the active count while still sizing memory
  footprint/fit from the total count. There's no auto-detection from a
  model name yet - you supply both numbers.
- **Batch size 1.** Batched serving amortizes weight reads and changes the
  regime entirely.
- **No CPU/GPU split.** Partial offload blends two bandwidth domains; not
  modeled.
- Predictions are estimates for sanity-checking hardware choices, not
  benchmarks. Measure before you buy.
