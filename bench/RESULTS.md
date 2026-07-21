# Benchmark results: measured vs predicted

Real measurements from the `bench/` harness on this project's own hardware,
used to validate `localmodel-fit`'s decode and prefill models. Reproduce with
the commands at the bottom.

**Machine:** Apple M4 (base, 10-core GPU, 16 GB unified memory) — the `m4`
preset: 120 GB/s bandwidth, 4.26 FP16 TFLOPS.
**Runtime:** ollama 0.32.1 (llama.cpp / Metal), macOS.
**Models:** `qwen2.5:0.5b` and `qwen2.5:1.5b`, both `Q4_K_M`, both **dense**.
Exact parameter counts read from the GGUF metadata and confirmed against
Hugging Face `safetensors` metadata: **494,032,768** and **1,543,714,304**.
**Method:** throwaway warmup (nonce-tagged, so no cached-prefix reuse), a
~460-token prompt for the compute-bound prefill measurement, a long-output
prompt capped at 128 tokens for a stable decode rate; throughput from ollama's
own `prompt_eval_*` / `eval_*` counters (model-load time excluded).

## Representative run

### qwen2.5:0.5b
```
localmodel-fit benchmark harness
machine: Apple M4 — 120 GB/s bandwidth, 4.26 FP16 TFLOPS
model:   qwen2.5:0.5b — 494M, Q4_K_M (0.30 GB weights)

PHASE     TOKENS    MEASURED   PREDICTED    ERROR   ACHIEVED
prefill      466    3554.4 t/s    2155.7 t/s   -39.4%   mfu 0.50→0.82
decode       128     172.2 t/s     280.5 t/s   +62.8%   efficiency 0.70→0.43
```

### qwen2.5:1.5b
```
localmodel-fit benchmark harness
machine: Apple M4 — 120 GB/s bandwidth, 4.26 FP16 TFLOPS
model:   qwen2.5:1.5b — 1.5B, Q4_K_M (0.94 GB weights)

PHASE     TOKENS    MEASURED   PREDICTED    ERROR   ACHIEVED
prefill      466    1186.2 t/s     689.9 t/s   -41.8%   mfu 0.50→0.86
decode       128      81.2 t/s      89.8 t/s   +10.5%   efficiency 0.70→0.63
```

## What the measurements validate

Median throughput over several runs each (base M4 is passively cooled, so
sustained runs drift a few percent with temperature):

| model | prompt tok | prefill meas. | decode meas. | prefill achieved MFU | decode achieved eff. |
|---|---|---|---|---|---|
| qwen2.5:0.5b | ~460 | ~3620 t/s | ~168 t/s | ~0.83 | ~0.42 |
| qwen2.5:1.5b | ~460 | ~1183 t/s | ~80 t/s | ~0.86 | ~0.63 |

**1. Prefill scales as 1/params (the core new claim).**
Measured prefill ratio 0.5b/1.5b = **3.06–3.18** across runs; the exact
parameter ratio is **1,543,714,304 / 494,032,768 = 3.1247**. Agreement within
~2%. The compute-bound `prefill tok/s = flops·mfu/(2·P)` form predicts exactly
this inverse-linear scaling.

**2. The prefill compute-bound form holds: achieved MFU is size-independent.**
The recovered MFU is **~0.83 (0.5b) and ~0.86 (1.5b)** — essentially the same
fraction of peak for both model sizes, which is what a compute-bound ceiling
predicts (MFU is a property of the machine+kernel, not the model size). That
consistency is stronger evidence than any single absolute number.

**3. Achieved compute is consistent with the verified 4.26-TFLOPS M4 peak.**
Both models imply ~3.5 TFLOP/s of *achieved* FP16 compute (`2·P·prefill_tok_s`).
Against the 4.26-TFLOPS peak that is MFU ~0.83; against the more conservative
3.76-TFLOPS base-clock reading it would be MFU >0.9, which is implausibly high
— so the measurement supports the 4.26 (boost-clock) figure. The shipped
default MFU (0.5) therefore *under-predicts* M4 prefill — the safe direction
for a screening estimate. Calibrate with `-mfu` / `--mfu` for your hardware.

**4. Decode (bandwidth-bound) is accurate for the larger model, loose for the
tiny one — as expected.**
The 1.5B decode is within **~10–15%** of the default (efficiency 0.7 →
achieved ~0.62–0.69). The 0.5B decode is over-predicted by ~60%: at that size
the vocab embedding (28% of its parameters, tied) dominates the file yet is
*not* streamed in full per token, so the `params × bits/8` weight estimate
overstates the bytes actually moved. This is the known tiny-model limitation
in METHODOLOGY.md; real 7B+ targets, where embeddings are a small fraction,
track the bandwidth model far more tightly (the trend from 0.5B→1.5B, achieved
efficiency 0.42→0.63, already shows this).

## Honest caveats
- Two small models on one machine — this validates the *model form* (scaling,
  size-independent MFU, bandwidth accuracy trend), not a universal constant.
- Apple GPU FP16 peaks carry ~±10% clock ambiguity; the harness measures the
  real achieved rate rather than trusting the peak.
- Base M4 is fanless; sustained runs vary a few percent thermally.

## Reproduce
```
ollama pull qwen2.5:0.5b qwen2.5:1.5b
go run ./bench -model qwen2.5:0.5b -hw m4 -params 494032768
go run ./bench -model qwen2.5:1.5b -hw m4 -params 1543714304
```
Offline (no ollama, no download), replays the committed capture:
```
go run ./bench -response bench/testdata/qwen2.5-0.5b-response.json -hw m4 -params 494032768
```
