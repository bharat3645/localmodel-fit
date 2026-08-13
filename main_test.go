package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// runOut is a small helper: run the CLI entrypoint with args, fail the test
// if it returns an error, and return everything it wrote to stdout.
func runOut(t *testing.T, args ...string) string {
	t.Helper()
	var buf bytes.Buffer
	if err := run(args, &buf); err != nil {
		t.Fatalf("run(%v): unexpected error: %v", args, err)
	}
	return buf.String()
}

// runErr is the inverse: assert run fails and return the error.
func runErr(t *testing.T, args ...string) error {
	t.Helper()
	var buf bytes.Buffer
	err := run(args, &buf)
	if err == nil {
		t.Fatalf("run(%v): expected an error, got none (stdout: %q)", args, buf.String())
	}
	return err
}

func TestRunRequiresModel(t *testing.T) {
	err := runErr(t, "--hw", "m4-pro")
	if !strings.Contains(err.Error(), "--model is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunRequiresHardware(t *testing.T) {
	err := runErr(t, "--model", "8b")
	if !strings.Contains(err.Error(), "need --hw preset or explicit --bandwidth and --mem") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunUnknownHardwarePreset(t *testing.T) {
	err := runErr(t, "--hw", "does-not-exist", "--model", "8b")
	if !strings.Contains(err.Error(), `unknown hardware preset "does-not-exist"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunInvalidModel(t *testing.T) {
	err := runErr(t, "--hw", "m4-pro", "--model", "not-a-number")
	if !strings.Contains(err.Error(), "cannot parse parameter count") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunActiveParamsCannotExceedModel(t *testing.T) {
	err := runErr(t, "--hw", "m4-pro", "--model", "8b", "--active-params", "9b")
	if !strings.Contains(err.Error(), "cannot exceed --model") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunUnknownFlag(t *testing.T) {
	// Each call gets a fresh flag.FlagSet (see run's doc comment), so this
	// - and every other test in this file - must not panic or bleed state
	// into other tests the way a shared flag.CommandLine would.
	runErr(t, "--nope", "8b")
}

func TestRunCustomHardware(t *testing.T) {
	out := runOut(t, "--bandwidth", "273", "--mem", "24", "--model", "8b")
	if !strings.Contains(out, "custom: 273.0 GB/s peak, 24 GB memory") {
		t.Fatalf("expected custom hardware line, got:\n%s", out)
	}
	if !strings.Contains(out, "model: 8.0B parameters\n") {
		t.Fatalf("expected dense model line, got:\n%s", out)
	}
}

func TestRunTableFormatting(t *testing.T) {
	out := runOut(t, "--hw", "m4-pro", "--model", "8b")
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")

	header := "QUANT        WEIGHTS  FITS       DECODE"
	found := false
	for _, l := range lines {
		if l == header {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected table header %q in output:\n%s", header, out)
	}
	if !strings.Contains(out, "F16") || !strings.Contains(out, "Q4_K_M") {
		t.Fatalf("expected known quant names in output:\n%s", out)
	}
	if !strings.Contains(out, "best fit:") {
		t.Fatalf("expected a 'best fit:' summary line, got:\n%s", out)
	}
}

func TestRunMoEOutputFormatting(t *testing.T) {
	// Mixtral-8x7B-shaped: 46.7B total, 12.9B active.
	out := runOut(t, "--hw", "m2-ultra", "--model", "46.7b", "--active-params", "12.9b")
	if !strings.Contains(out, "model: 46.7B parameters total, 12.9B active per token (MoE)") {
		t.Fatalf("expected MoE model summary line, got:\n%s", out)
	}
}

func TestRunNothingFits(t *testing.T) {
	out := runOut(t, "--bandwidth", "100", "--mem", "1", "--model", "70b")
	if !strings.Contains(out, "no quantization fits in 1 GB") {
		t.Fatalf("expected a no-fit message, got:\n%s", out)
	}
	if strings.Contains(out, "best fit:") {
		t.Fatalf("did not expect a 'best fit:' line when nothing fits, got:\n%s", out)
	}
}

func TestRunPrefillAndLatency(t *testing.T) {
	out := runOut(t, "--hw", "m4", "--model", "1.5b", "--prompt-tokens", "512", "--gen-tokens", "128")
	if !strings.Contains(out, "prefill:") || !strings.Contains(out, "tok/s prompt processing") {
		t.Fatalf("expected a prefill line, got:\n%s", out)
	}
	if !strings.Contains(out, "TTFT for a 512-token prompt:") {
		t.Fatalf("expected a TTFT line, got:\n%s", out)
	}
	if !strings.Contains(out, "end-to-end for +128 generated tokens") {
		t.Fatalf("expected an end-to-end latency line, got:\n%s", out)
	}
}

func TestRunNoFP16FigureNotesFlopsFlag(t *testing.T) {
	// ddr5-2ch carries no FP16TFLOPS figure by default.
	out := runOut(t, "--hw", "ddr5-2ch", "--model", "8b")
	if !strings.Contains(out, "pass --flops <TFLOPS>") {
		t.Fatalf("expected a hint to pass --flops, got:\n%s", out)
	}
}

func TestRunListHW(t *testing.T) {
	out := runOut(t, "--list-hw")
	if !strings.Contains(out, "m4-pro") || !strings.Contains(out, "rtx-4090") {
		t.Fatalf("expected known preset names in --list-hw output, got:\n%s", out)
	}
	// --list-hw should short-circuit and not require --model.
	if strings.Contains(out, "QUANT") {
		t.Fatalf("--list-hw should not print the quant table, got:\n%s", out)
	}
}

func TestRunJSONOutput(t *testing.T) {
	out := runOut(t, "--hw", "m4-pro", "--model", "8b", "--json")

	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("--json output is not valid JSON: %v\noutput:\n%s", err, out)
	}

	if doc["moe"] != false {
		t.Fatalf("expected moe=false for a dense model, got %v", doc["moe"])
	}
	if got, want := doc["params"].(float64), 8e9; got != want {
		t.Fatalf("params = %v, want %v", got, want)
	}
	results, ok := doc["results"].([]any)
	if !ok || len(results) == 0 {
		t.Fatalf("expected a non-empty results array, got %v", doc["results"])
	}
	if _, ok := doc["best"]; !ok {
		t.Fatalf("expected a best field when a quant fits, got doc=%v", doc)
	}
	hw, ok := doc["hardware"].(map[string]any)
	if !ok || hw["Name"] != "Apple M4 Pro" {
		t.Fatalf("expected hardware.Name = Apple M4 Pro, got %v", doc["hardware"])
	}
}

func TestRunJSONOutputMoEAndPrefill(t *testing.T) {
	out := runOut(t, "--hw", "m2-ultra", "--model", "46.7b", "--active-params", "12.9b",
		"--prompt-tokens", "256", "--gen-tokens", "64", "--json")

	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("--json output is not valid JSON: %v\noutput:\n%s", err, out)
	}
	if doc["moe"] != true {
		t.Fatalf("expected moe=true, got %v", doc["moe"])
	}
	if got, want := doc["active_params"].(float64), 12.9e9; got != want {
		t.Fatalf("active_params = %v, want %v", got, want)
	}
	prefill, ok := doc["prefill"].(map[string]any)
	if !ok {
		t.Fatalf("expected a prefill object, got %v", doc["prefill"])
	}
	if _, ok := prefill["ttft_s"]; !ok {
		t.Fatalf("expected prefill.ttft_s when --prompt-tokens is set, got %v", prefill)
	}
	if _, ok := prefill["end_to_end_s"]; !ok {
		t.Fatalf("expected prefill.end_to_end_s when --gen-tokens and a best fit are set, got %v", prefill)
	}
}

func TestRunSpeculativeDecodingHintForLargeModels(t *testing.T) {
	big := runOut(t, "--hw", "m4-pro", "--model", "8b")
	if !strings.Contains(big, "speculative decoding:") {
		t.Fatalf("expected a speculative-decoding hint for an 8B model, got:\n%s", big)
	}

	small := runOut(t, "--hw", "m4-pro", "--model", "350m")
	if strings.Contains(small, "speculative decoding:") {
		t.Fatalf("did not expect a speculative-decoding hint for a 350M model, got:\n%s", small)
	}
}
