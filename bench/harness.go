package main

// Pure analysis logic for the benchmark harness: turning ollama's raw timing
// counters into measured throughput, and comparing measured against the
// fit-package prediction. Kept free of any I/O so it can be unit-tested in CI
// against a recorded fixture (bench/testdata/) without a running ollama or any
// model download.

import "github.com/bharat3645/localmodel-fit/fit"

// ollamaResponse captures the timing counters ollama returns from
// /api/generate (durations are nanoseconds). Only the fields the harness needs
// are decoded.
type ollamaResponse struct {
	Model              string `json:"model"`
	PromptEvalCount    int    `json:"prompt_eval_count"`    // prompt tokens prefilled
	PromptEvalDuration int64  `json:"prompt_eval_duration"` // ns spent in prefill
	EvalCount          int    `json:"eval_count"`           // tokens generated
	EvalDuration       int64  `json:"eval_duration"`        // ns spent in decode
}

// measuredPrefillTokS is prompt tokens divided by prompt-eval seconds. Returns
// 0 when the counter is absent (e.g. a fully KV-cached prompt reports no work).
func measuredPrefillTokS(r ollamaResponse) float64 {
	if r.PromptEvalDuration <= 0 || r.PromptEvalCount <= 0 {
		return 0
	}
	return float64(r.PromptEvalCount) / (float64(r.PromptEvalDuration) / 1e9)
}

// measuredDecodeTokS is generated tokens divided by decode seconds.
func measuredDecodeTokS(r ollamaResponse) float64 {
	if r.EvalDuration <= 0 || r.EvalCount <= 0 {
		return 0
	}
	return float64(r.EvalCount) / (float64(r.EvalDuration) / 1e9)
}

// phase names one measured-vs-predicted comparison.
type phase struct {
	Name      string  // "prefill" or "decode"
	Measured  float64 // tok/s from ollama's counters
	Predicted float64 // tok/s from the fit model
	// Achieved is the implied real-world constant the prediction assumed: the
	// bandwidth efficiency for decode, or the compute MFU for prefill. It is
	// (measured/predicted) * the assumed constant — what the machine actually
	// delivered, recovered from the measurement.
	AssumedName string  // "efficiency" or "mfu"
	Assumed     float64 // the value the prediction used
	Achieved    float64 // Assumed * Measured/Predicted, or 0 if Predicted==0
	Tokens      int     // sample size (prompt or generated token count)
}

// errorPct is the signed relative error of the prediction vs measurement:
// (predicted-measured)/measured, in percent. Positive = the model over-predicts.
func (p phase) errorPct() float64 {
	if p.Measured == 0 {
		return 0
	}
	return (p.Predicted - p.Measured) / p.Measured * 100
}

func achieved(assumed, measured, predicted float64) float64 {
	if predicted <= 0 {
		return 0
	}
	return assumed * measured / predicted
}

// comparePrefill builds the prefill comparison. weightBytes is unused for
// prefill (compute is quant-independent); prediction uses the machine's peak
// FP16 throughput and the assumed MFU.
func comparePrefill(r ollamaResponse, hw fit.Hardware, activeParams, mfu float64) phase {
	measured := measuredPrefillTokS(r)
	predicted := fit.PrefillTokS(hw.FP16FLOPS(), activeParams, mfu)
	return phase{
		Name: "prefill", Measured: measured, Predicted: predicted,
		AssumedName: "mfu", Assumed: mfu,
		Achieved: achieved(mfu, measured, predicted),
		Tokens:   r.PromptEvalCount,
	}
}

// compareDecode builds the decode comparison. weightBytes is the predicted
// weight footprint the tool streams per token (params * bits/8) for the model's
// quantization; prediction uses the machine's bandwidth and assumed efficiency.
func compareDecode(r ollamaResponse, hw fit.Hardware, weightBytes, efficiency float64) phase {
	measured := measuredDecodeTokS(r)
	predicted := fit.DecodeTokS(hw.BandwidthGBs, weightBytes, efficiency)
	return phase{
		Name: "decode", Measured: measured, Predicted: predicted,
		AssumedName: "efficiency", Assumed: efficiency,
		Achieved: achieved(efficiency, measured, predicted),
		Tokens:   r.EvalCount,
	}
}
