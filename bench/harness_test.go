package main

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/bharat3645/localmodel-fit/fit"
)

func almost(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %v want %v (±%v)", got, want, tol)
	}
}

// loadFixture reads a recorded real ollama response so CI validates the harness
// math deterministically, with no ollama server and no model download.
func loadFixture(t *testing.T) ollamaResponse {
	t.Helper()
	b, err := os.ReadFile("testdata/qwen2.5-0.5b-response.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var r ollamaResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return r
}

func TestMeasuredRatesFromFixture(t *testing.T) {
	r := loadFixture(t)
	// prompt_eval_count=69, prompt_eval_duration=21634000ns -> 3189.42 tok/s
	almost(t, measuredPrefillTokS(r), 3189.4241, 0.01)
	// eval_count=128, eval_duration=743076000ns -> 172.26 tok/s
	almost(t, measuredDecodeTokS(r), 172.2569, 0.01)
}

func TestMeasuredRatesGuardZeroDuration(t *testing.T) {
	if got := measuredPrefillTokS(ollamaResponse{PromptEvalCount: 10}); got != 0 {
		t.Fatalf("zero duration must yield 0, got %v", got)
	}
	if got := measuredDecodeTokS(ollamaResponse{EvalCount: 10}); got != 0 {
		t.Fatalf("zero duration must yield 0, got %v", got)
	}
}

func TestComparePrefillAchievedMFU(t *testing.T) {
	r := loadFixture(t)
	// A hypothetical machine at exactly the measured effective throughput:
	// if predicted == measured, the achieved MFU equals the assumed MFU.
	hw := fit.Hardware{Name: "t", FP16TFLOPS: 3.5}
	activeParams := 0.494e9
	// Choose mfu so predicted == measured, then achieved must round-trip to it.
	measured := measuredPrefillTokS(r)
	mfu := measured * 2 * activeParams / hw.FP16FLOPS()
	p := comparePrefill(r, hw, activeParams, mfu)
	almost(t, p.Predicted, measured, 0.5)
	almost(t, p.errorPct(), 0, 0.1)
	almost(t, p.Achieved, mfu, 1e-6)
}

func TestCompareDecodeErrorAndAchieved(t *testing.T) {
	r := loadFixture(t)
	hw := fit.Hardware{Name: "t", BandwidthGBs: 120}
	// Q4_K_M footprint for ~0.494B params: params*4.85/8 bytes.
	weightBytes := fit.WeightBytes(0.494e9, 4.85)
	eff := 0.7
	d := compareDecode(r, hw, weightBytes, eff)
	// Predicted = 120e9*0.7/weightBytes; check it matches fit.DecodeTokS.
	almost(t, d.Predicted, fit.DecodeTokS(120, weightBytes, eff), 1e-6)
	// Achieved efficiency = eff * measured/predicted; error is self-consistent.
	almost(t, d.Achieved, eff*d.Measured/d.Predicted, 1e-9)
	almost(t, d.errorPct(), (d.Predicted-d.Measured)/d.Measured*100, 1e-9)
}
