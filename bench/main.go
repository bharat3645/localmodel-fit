// Command bench measures real ollama prefill/decode throughput and compares it
// against localmodel-fit's predictions, reporting the residual and the
// machine's achieved bandwidth-efficiency / compute-MFU.
//
// Live run (needs a running ollama and the model pulled):
//
//	go run ./bench -model qwen2.5:1.5b -hw m4 -params 1.54b
//
// Offline (replay a captured response, no ollama needed):
//
//	go run ./bench -response bench/testdata/qwen2.5-0.5b-response.json \
//	    -hw m4 -params 0.494b
//
// The harness deliberately measures prefill and decode with separate requests:
// prefill wants a long input prompt (compute-bound regime), decode wants a long
// output (many eval tokens for a stable per-token rate). See METHODOLOGY.md.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/bharat3645/localmodel-fit/fit"
)

const paragraph = "In local large language model inference the memory subsystem and the " +
	"compute units impose two different ceilings on throughput, and which one " +
	"binds depends entirely on the phase of execution being considered. "

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "bench: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	model := flag.String("model", "", "ollama model tag (live run), e.g. qwen2.5:1.5b")
	hwName := flag.String("hw", "", "hardware preset (fit --list-hw)")
	paramsS := flag.String("params", "", "total parameter count, e.g. 1.54b (required)")
	activeS := flag.String("active-params", "", "active params/token for MoE (default: --params)")
	quantName := flag.String("quant", "Q4_K_M", "quantization the model is stored in")
	eff := flag.Float64("efficiency", fit.DefaultEfficiency, "assumed decode bandwidth efficiency")
	mfu := flag.Float64("mfu", fit.DefaultMFU, "assumed prefill compute MFU")
	flops := flag.Float64("flops", 0, "peak FP16 TFLOPS override (else from --hw preset)")
	respFile := flag.String("response", "", "replay a saved ollama response instead of a live run")
	url := flag.String("url", "http://localhost:11434", "ollama base URL")
	jsonOut := flag.String("json", "", "write the live decode response to this file")
	flag.Parse()

	if *paramsS == "" {
		return fmt.Errorf("--params is required (the model's real parameter count)")
	}
	params, err := fit.ParseParams(*paramsS)
	if err != nil {
		return err
	}
	active := params
	if *activeS != "" {
		if active, err = fit.ParseParams(*activeS); err != nil {
			return fmt.Errorf("--active-params: %w", err)
		}
	}
	q, ok := quantByName(*quantName)
	if !ok {
		return fmt.Errorf("unknown quant %q", *quantName)
	}

	hw := fit.Hardware{Name: "custom"}
	if *hwName != "" {
		preset, ok := fit.HardwarePresets[*hwName]
		if !ok {
			return fmt.Errorf("unknown hardware preset %q (fit --list-hw)", *hwName)
		}
		hw = preset
	}
	if *flops > 0 {
		hw.FP16TFLOPS = *flops
	}
	if hw.BandwidthGBs <= 0 {
		return fmt.Errorf("need --hw preset (for bandwidth) — decode prediction requires it")
	}

	// Obtain a prefill measurement and a decode measurement.
	var prefillResp, decodeResp ollamaResponse
	if *respFile != "" {
		r, err := loadResponse(*respFile)
		if err != nil {
			return err
		}
		prefillResp, decodeResp = r, r // one capture drives both phases offline
	} else {
		if *model == "" {
			return fmt.Errorf("--model is required for a live run (or pass --response)")
		}
		prefillResp, decodeResp, err = liveMeasure(*url, *model, *jsonOut)
		if err != nil {
			return err
		}
	}

	// Predictions. Both phases use active params (dense: active == total).
	weightBytes := fit.WeightBytes(active, q.BitsPerWeight)
	pf := comparePrefill(prefillResp, hw, active, *mfu)
	dc := compareDecode(decodeResp, hw, weightBytes, *eff)

	printReport(hw, *model, params, active, q, weightBytes, pf, dc)
	return nil
}

func printReport(hw fit.Hardware, model string, params, active float64, q fit.Quant, weightBytes float64, pf, dc phase) {
	fmt.Printf("localmodel-fit benchmark harness\n")
	fmt.Printf("machine: %s — %.0f GB/s bandwidth", hw.Name, hw.BandwidthGBs)
	if hw.FP16TFLOPS > 0 {
		fmt.Printf(", %.2f FP16 TFLOPS", hw.FP16TFLOPS)
	}
	fmt.Printf("\n")
	if model != "" {
		fmt.Printf("model:   %s — ", model)
	} else {
		fmt.Printf("model:   ")
	}
	if active != params {
		fmt.Printf("%s total / %s active (MoE), ", fit.HumanParams(params), fit.HumanParams(active))
	} else {
		fmt.Printf("%s, ", fit.HumanParams(params))
	}
	fmt.Printf("%s (%.2f GB weights)\n\n", q.Name, weightBytes/1e9)

	fmt.Printf("%-8s %7s %11s %11s %8s   %s\n", "PHASE", "TOKENS", "MEASURED", "PREDICTED", "ERROR", "ACHIEVED")
	for _, p := range []phase{pf, dc} {
		if p.Predicted <= 0 {
			fmt.Printf("%-8s %7d %9.1f t/s %11s %8s   (no %s figure for this preset)\n",
				p.Name, p.Tokens, p.Measured, "—", "—", p.AssumedName)
			continue
		}
		fmt.Printf("%-8s %7d %9.1f t/s %9.1f t/s %+7.1f%%   %s %.2f→%.2f\n",
			p.Name, p.Tokens, p.Measured, p.Predicted, p.errorPct(),
			p.AssumedName, p.Assumed, p.Achieved)
	}
	fmt.Printf("\nERROR = (predicted−measured)/measured; positive means the model over-predicts.\n")
	fmt.Printf("ACHIEVED = the %s / %s the machine actually delivered, recovered from the measurement.\n",
		pf.AssumedName, dc.AssumedName)
}

func quantByName(name string) (fit.Quant, bool) {
	for _, q := range fit.Quants {
		if strings.EqualFold(q.Name, name) {
			return q, true
		}
	}
	return fit.Quant{}, false
}

// --- live ollama I/O ---

func liveMeasure(baseURL, model, jsonOut string) (prefill, decode ollamaResponse, err error) {
	nonce := fmt.Sprintf("%d", time.Now().UnixNano())
	// Warmup: load weights; nonce prevents a cached prefix on the measured run.
	if _, _, err = generate(baseURL, model, "Warmup "+nonce+": reply ok.", 4); err != nil {
		return prefill, decode, fmt.Errorf("warmup (is ollama running and %q pulled?): %w", model, err)
	}
	// Prefill: long unique input prompt, short generation.
	pPrompt := "Doc " + nonce + ". " + strings.Repeat(paragraph, 12) + "Reply with one word: acknowledged."
	if prefill, _, err = generate(baseURL, model, pPrompt, 8); err != nil {
		return prefill, decode, err
	}
	// Decode: prompt that induces a long output, capped at 128 tokens.
	dPrompt := "Session " + nonce + ". Write a detailed 400-word technical explanation of how " +
		"CPU caches, main memory bandwidth, and arithmetic throughput each limit program " +
		"performance in different situations. Use full sentences and be thorough."
	var raw []byte
	if decode, raw, err = generate(baseURL, model, dPrompt, 128); err != nil {
		return prefill, decode, err
	}
	if jsonOut != "" {
		if err := os.WriteFile(jsonOut, raw, 0o644); err != nil {
			return prefill, decode, fmt.Errorf("write --json: %w", err)
		}
	}
	return prefill, decode, nil
}

func generate(baseURL, model, prompt string, numPredict int) (ollamaResponse, []byte, error) {
	body, _ := json.Marshal(map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": false,
		"options": map[string]any{
			"num_predict": numPredict,
			"temperature": 0,
			"seed":        42,
		},
	})
	req, err := http.NewRequest("POST", baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return ollamaResponse{}, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return ollamaResponse{}, nil, err
	}
	defer resp.Body.Close()
	raw, err := readAll(resp)
	if err != nil {
		return ollamaResponse{}, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return ollamaResponse{}, nil, fmt.Errorf("ollama %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var r ollamaResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		return ollamaResponse{}, nil, fmt.Errorf("decode ollama response: %w", err)
	}
	return r, raw, nil
}

func loadResponse(path string) (ollamaResponse, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return ollamaResponse{}, err
	}
	var r ollamaResponse
	if err := json.Unmarshal(b, &r); err != nil {
		return ollamaResponse{}, fmt.Errorf("decode %s: %w", path, err)
	}
	return r, nil
}

func readAll(resp *http.Response) ([]byte, error) {
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
