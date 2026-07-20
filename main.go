// Command localmodel-fit predicts local-LLM decode throughput from memory
// bandwidth and recommends the best quantization that fits.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/bharat3645/localmodel-fit/fit"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "localmodel-fit: %v\n", err)
		os.Exit(2)
	}
}

func run() error {
	hwName := flag.String("hw", "", "hardware preset (see --list-hw)")
	bandwidth := flag.Float64("bandwidth", 0, "peak memory bandwidth GB/s (overrides preset)")
	mem := flag.Float64("mem", 0, "accelerator-visible memory GB (overrides preset)")
	model := flag.String("model", "", "model parameter count, e.g. 8b, 70b, 350m")
	activeModel := flag.String("active-params", "", "active parameter count per token, for MoE models (e.g. 12.9b for Mixtral-8x7B's 46.7b total); defaults to --model (dense)")
	efficiency := flag.Float64("efficiency", fit.DefaultEfficiency, "achievable fraction of peak bandwidth")
	jsonOut := flag.Bool("json", false, "JSON output")
	listHW := flag.Bool("list-hw", false, "list hardware presets and exit")
	flag.Parse()

	if *listHW {
		printPresets()
		return nil
	}

	hw := fit.Hardware{Name: "custom"}
	if *hwName != "" {
		preset, ok := fit.HardwarePresets[*hwName]
		if !ok {
			return fmt.Errorf("unknown hardware preset %q (try --list-hw)", *hwName)
		}
		hw = preset
	}
	if *bandwidth > 0 {
		hw.BandwidthGBs = *bandwidth
	}
	if *mem > 0 {
		hw.MemoryGB = *mem
	}
	if hw.BandwidthGBs <= 0 || hw.MemoryGB <= 0 {
		return fmt.Errorf("need --hw preset or explicit --bandwidth and --mem")
	}
	if *model == "" {
		return fmt.Errorf("--model is required (e.g. --model 8b)")
	}
	params, err := fit.ParseParams(*model)
	if err != nil {
		return err
	}
	activeParams := params
	isMoE := false
	if *activeModel != "" {
		activeParams, err = fit.ParseParams(*activeModel)
		if err != nil {
			return fmt.Errorf("--active-params: %w", err)
		}
		if activeParams > params {
			return fmt.Errorf("--active-params (%s) cannot exceed --model (%s)", fit.HumanParams(activeParams), fit.HumanParams(params))
		}
		isMoE = true
	}

	results := fit.AdviseMoE(params, activeParams, hw, *efficiency)
	best, hasBest := fit.Best(results)

	if *jsonOut {
		doc := map[string]any{
			"hardware":      hw,
			"params":        params,
			"active_params": activeParams,
			"moe":           isMoE,
			"efficiency":    *efficiency,
			"results":       results,
		}
		if hasBest {
			doc["best"] = best
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(doc)
	}

	fmt.Printf("%s: %.1f GB/s peak, %.0f GB memory, efficiency %.2f\n", hw.Name, hw.BandwidthGBs, hw.MemoryGB, *efficiency)
	if isMoE {
		fmt.Printf("model: %s parameters total, %s active per token (MoE)\n\n", fit.HumanParams(params), fit.HumanParams(activeParams))
	} else {
		fmt.Printf("model: %s parameters\n\n", fit.HumanParams(params))
	}
	fmt.Printf("%-8s %11s %5s %12s\n", "QUANT", "WEIGHTS", "FITS", "DECODE")
	for _, r := range results {
		fitStr := "no"
		if r.Fits {
			fitStr = "yes"
		}
		fmt.Printf("%-8s %8.2f GB %5s %8.1f t/s\n", r.Quant.Name, r.WeightGB, fitStr, r.DecodeTokS)
	}
	if hasBest {
		fmt.Printf("\nbest fit: %s (%.2f GB), predicted %.1f tok/s decode\n", best.Quant.Name, best.WeightGB, best.DecodeTokS)
	} else {
		fmt.Printf("\nno quantization fits in %.0f GB: smaller model or more memory\n", hw.MemoryGB)
	}
	if params >= 7e9 {
		fmt.Println("speculative decoding: pair a same-family draft model ~10-20x smaller; typical 1.3-2.0x decode speedup (heuristic, see METHODOLOGY.md)")
	}
	return nil
}

func printPresets() {
	names := make([]string, 0, len(fit.HardwarePresets))
	for name := range fit.HardwarePresets {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		p := fit.HardwarePresets[name]
		fmt.Printf("%-10s %-18s %7.1f GB/s %5.0f GB\n", name, p.Name, p.BandwidthGBs, p.MemoryGB)
	}
}
