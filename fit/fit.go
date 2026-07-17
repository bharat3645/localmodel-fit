// Package fit predicts local-LLM decode throughput from memory bandwidth
// and advises quantization choices that fit a memory budget.
//
// The core observation (see METHODOLOGY.md): single-stream decode is
// memory-bandwidth-bound — every generated token streams approximately the
// whole weight file through the memory system — so predicted tokens/sec is
// bandwidth * efficiency / weight_bytes.
package fit

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Quant describes a llama.cpp-style quantization with its approximate
// bits-per-weight cost (including scales and zero-point overhead).
type Quant struct {
	Name          string
	BitsPerWeight float64
}

// Hardware describes the memory system that bounds decode speed.
type Hardware struct {
	Name         string
	BandwidthGBs float64
	MemoryGB     float64
}

// DefaultEfficiency is the fraction of peak bandwidth real decode loops
// achieve in practice (see METHODOLOGY.md for the observed range).
const DefaultEfficiency = 0.7

// FitOverhead multiplies weight bytes to approximate the runtime footprint
// (KV cache at moderate context, compute buffers, allocator slack).
const FitOverhead = 1.15

// WeightBytes returns the model weight footprint in bytes.
func WeightBytes(params, bitsPerWeight float64) float64 {
	return params * bitsPerWeight / 8
}

// DecodeTokS predicts decode tokens/sec for a bandwidth-bound generation
// loop. bandwidthGBs is the peak spec figure; efficiency scales it to what
// real kernels achieve.
func DecodeTokS(bandwidthGBs, weightBytes, efficiency float64) float64 {
	if weightBytes <= 0 {
		return 0
	}
	return bandwidthGBs * 1e9 * efficiency / weightBytes
}

// Fits reports whether a model of weightBytes runs within memGB, including
// the FitOverhead allowance.
func Fits(weightBytes, memGB float64) bool {
	return weightBytes*FitOverhead <= memGB*1e9
}

// Result is the advice row for one quantization.
type Result struct {
	Quant      Quant
	WeightGB   float64
	Fits       bool
	DecodeTokS float64
}

// Advise evaluates every known quant for a parameter count on hardware.
// Rows are ordered best-quality-first (highest bits per weight). An
// efficiency <= 0 selects DefaultEfficiency.
func Advise(params float64, hw Hardware, efficiency float64) []Result {
	if efficiency <= 0 {
		efficiency = DefaultEfficiency
	}
	results := make([]Result, 0, len(Quants))
	for _, q := range Quants {
		wb := WeightBytes(params, q.BitsPerWeight)
		results = append(results, Result{
			Quant:      q,
			WeightGB:   wb / 1e9,
			Fits:       Fits(wb, hw.MemoryGB),
			DecodeTokS: DecodeTokS(hw.BandwidthGBs, wb, efficiency),
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Quant.BitsPerWeight > results[j].Quant.BitsPerWeight
	})
	return results
}

// Best returns the highest-quality result that fits, or ok=false when
// nothing does.
func Best(results []Result) (Result, bool) {
	for _, r := range results {
		if r.Fits {
			return r, true
		}
	}
	return Result{}, false
}

// ParseParams parses "8b", "70B", "0.5b", "350m", or a raw count.
func ParseParams(s string) (float64, error) {
	t := strings.TrimSpace(strings.ToLower(s))
	mult := 1.0
	switch {
	case strings.HasSuffix(t, "b"):
		mult = 1e9
		t = strings.TrimSuffix(t, "b")
	case strings.HasSuffix(t, "m"):
		mult = 1e6
		t = strings.TrimSuffix(t, "m")
	}
	v, err := strconv.ParseFloat(t, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("cannot parse parameter count %q", s)
	}
	return v * mult, nil
}

// HumanParams renders 8e9 as "8.0B" and 3.5e8 as "350M".
func HumanParams(params float64) string {
	if params >= 1e9 {
		return fmt.Sprintf("%.1fB", params/1e9)
	}
	return fmt.Sprintf("%.0fM", params/1e6)
}
