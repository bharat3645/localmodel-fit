#!/usr/bin/env bash
# Demo driver for localmodel-fit. Builds the real binary and runs it against
# the exact scenarios in this cast: dense-model advise, the MoE-aware decode
# correction, and prefill/TTFT/end-to-end latency. No network access, no
# ollama — this tool is a pure calculator, so every number here is
# reproducible offline from the committed hardware/quant tables.
set -euo pipefail

say() { printf '\n== %s ==\n' "$*"; }

say "Build the real binary from source"
go build -o /tmp/localmodel-fit .
BIN=/tmp/localmodel-fit
{ $BIN --list-hw | head -5; } || true
echo "..."

say "1. Dense model: which quant fits an 8B model on an M4 Pro, and how fast?"
$BIN --hw m4-pro --model 8b

say "2. MoE-aware correction: Mixtral-8x7B (46.7B total, 12.9B active per token)"
echo "-- naive (treats it as if all 46.7B were read every token) --"
$BIN --hw m2-ultra --model 46.7b
echo "-- correct (--active-params tells the tool only 12.9B is read per token) --"
$BIN --hw m2-ultra --model 46.7b --active-params 12.9b

say "3. Prefill is compute-bound, not bandwidth-bound: TTFT + end-to-end latency"
$BIN --hw m4-pro --model 8b --prompt-tokens 2048 --gen-tokens 256

say "Done. Every number above came from the compiled binary in this recording, not hand-typed."
