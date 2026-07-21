package fit

// Prefill (prompt processing) is compute-bound, not memory-bandwidth-bound.
//
// Decode generates one token at a time and streams ~every weight through the
// memory system per token, so it is bounded by memory bandwidth (see fit.go).
// Prefill instead processes the whole prompt in a single forward pass: that is
// a batched matrix-matrix multiply over N prompt tokens, which reuses each
// weight N times and therefore has high arithmetic intensity — the binding
// constraint is compute throughput (FLOP/s), not bandwidth.
//
// A forward pass over N tokens through a model with P (active) parameters costs
// approximately 2*N*P FLOPs: one multiply and one add per parameter per token
// (Kaplan et al. 2020, "Scaling Laws for Neural Language Models"; the training
// cost C≈6ND is 2ND forward + 4ND backward). The 2ND count is the dense
// matmul work and omits attention's O(N^2) term, which is a small correction at
// short-to-moderate context (see METHODOLOGY.md). Per-token prefill cost is
// thus ~2*P FLOPs, independent of prompt length in this regime, so:
//
//	prefill tok/s = flops * mfu / (2 * P_active)
//
// MFU (model FLOPs utilization) is the achieved fraction of peak, analogous to
// the bandwidth `efficiency` used for decode.

// DefaultMFU is the fraction of peak FP16 compute that real prefill kernels
// achieve (model FLOPs utilization). It is deliberately a conservative
// screening midpoint: measured prefill MFU is hardware-, kernel-, and
// model-dependent, so absolute prefill predictions are softer than the
// bandwidth-bound decode ones. This project's own M4 measured ~0.83 on small
// models (see bench/RESULTS.md); other hardware and larger models will differ.
// The benchmark harness (bench/) measures the achieved value on real hardware
// so this default can be checked, not just asserted (see METHODOLOGY.md). Use
// --mfu / the harness to calibrate for your machine.
const DefaultMFU = 0.5

// PrefillTokS predicts prompt-processing throughput in tokens/sec for a
// compute-bound forward pass. flopsFp16 is peak FP16 throughput in FLOP/s
// (not TFLOPS); mfu is the achieved fraction of that peak. activeParams is the
// parameter count that actually does matmul work per token — the full count
// for a dense model, the routed subset for a Mixture-of-Experts model (only
// the selected experts run, exactly as for decode).
//
// Prefill throughput is essentially independent of quantization: llama.cpp
// dequantizes to compute the matmul, so the FLOP count is set by the model
// dimensions, not the on-disk weight format.
func PrefillTokS(flopsFp16, activeParams, mfu float64) float64 {
	if flopsFp16 <= 0 || activeParams <= 0 {
		return 0
	}
	if mfu <= 0 {
		mfu = DefaultMFU
	}
	return flopsFp16 * mfu / (2 * activeParams)
}

// TTFTSeconds estimates time-to-first-token: the prefill pass over promptTokens
// at prefillTokS tokens/sec. Returns 0 if prefillTokS is non-positive.
func TTFTSeconds(promptTokens, prefillTokS float64) float64 {
	if prefillTokS <= 0 {
		return 0
	}
	return promptTokens / prefillTokS
}

// EndToEndSeconds estimates total latency to produce genTokens after a
// promptTokens prompt: prefill (TTFT) plus genTokens of bandwidth-bound decode.
// It ignores scheduling/sampling overhead and any decode slowdown from a
// growing KV cache — a screening estimate, not a benchmark.
func EndToEndSeconds(promptTokens, genTokens, prefillTokS, decodeTokS float64) float64 {
	ttft := TTFTSeconds(promptTokens, prefillTokS)
	var gen float64
	if decodeTokS > 0 {
		gen = genTokens / decodeTokS
	}
	return ttft + gen
}
