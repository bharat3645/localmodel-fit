package fit

import (
	"math"
	"testing"
)

func almost(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %v want %v (±%v)", got, want, tol)
	}
}

func TestParseParams(t *testing.T) {
	cases := []struct {
		in   string
		want float64
	}{
		{"8b", 8e9},
		{"70B", 70e9},
		{"0.5b", 5e8},
		{"350m", 3.5e8},
		{"8000000000", 8e9},
	}
	for _, c := range cases {
		got, err := ParseParams(c.in)
		if err != nil {
			t.Fatalf("ParseParams(%q): %v", c.in, err)
		}
		almost(t, got, c.want, c.want*1e-9)
	}
	for _, bad := range []string{"", "abc", "-7b", "0"} {
		if _, err := ParseParams(bad); err == nil {
			t.Fatalf("ParseParams(%q): expected error", bad)
		}
	}
}

func TestHumanParams(t *testing.T) {
	if got := HumanParams(8e9); got != "8.0B" {
		t.Fatalf("got %q", got)
	}
	if got := HumanParams(3.5e8); got != "350M" {
		t.Fatalf("got %q", got)
	}
}

func TestWeightBytes(t *testing.T) {
	almost(t, WeightBytes(8e9, 4.85), 4.85e9, 1)
	almost(t, WeightBytes(8e9, 16), 16e9, 1)
}

func TestDecodeTokS(t *testing.T) {
	// M4 Pro-class: 273 GB/s * 0.7 / 4.85 GB = 39.4 tok/s
	almost(t, DecodeTokS(273, 4.85e9, 0.7), 39.402, 0.01)
	if DecodeTokS(273, 0, 0.7) != 0 {
		t.Fatal("zero weights must predict 0")
	}
}

func TestFits(t *testing.T) {
	if !Fits(20e9, 24) {
		t.Fatal("20 GB * 1.15 = 23 GB should fit in 24 GB")
	}
	if Fits(21e9, 24) {
		t.Fatal("21 GB * 1.15 = 24.15 GB must not fit in 24 GB")
	}
}

func TestAdviseOrderingAndBest(t *testing.T) {
	hw := Hardware{Name: "test", BandwidthGBs: 100, MemoryGB: 8}
	results := Advise(8e9, hw, 0) // 0 selects DefaultEfficiency
	if len(results) != len(Quants) {
		t.Fatalf("want %d rows, got %d", len(Quants), len(results))
	}
	for i := 1; i < len(results); i++ {
		if results[i].Quant.BitsPerWeight > results[i-1].Quant.BitsPerWeight {
			t.Fatal("results must be sorted best-quality-first")
		}
	}
	best, ok := Best(results)
	if !ok {
		t.Fatal("something should fit 8B params in 8 GB")
	}
	// 8 GB budget: bpw <= 8*8/(8*1.15) = 6.96, so Q6_K (6.59) is best.
	if best.Quant.Name != "Q6_K" {
		t.Fatalf("want Q6_K best in 8 GB, got %s", best.Quant.Name)
	}
	if _, ok := Best(Advise(70e9, Hardware{Name: "tiny", BandwidthGBs: 100, MemoryGB: 8}, 0)); ok {
		t.Fatal("70B cannot fit in 8 GB at any known quant")
	}
}

func TestAdviseMoEMatchesAdviseForDenseModel(t *testing.T) {
	hw := Hardware{Name: "test", BandwidthGBs: 100, MemoryGB: 8}
	dense := Advise(8e9, hw, 0.7)
	moe := AdviseMoE(8e9, 8e9, hw, 0.7)
	for i := range dense {
		if dense[i] != moe[i] {
			t.Fatalf("row %d: Advise=%+v AdviseMoE(total==active)=%+v", i, dense[i], moe[i])
		}
	}
}

func TestAdviseMoEUsesActiveParamsForDecodeAndTotalForFit(t *testing.T) {
	// Mixtral-8x7B-shaped: 46.7B total (memory footprint), 12.9B active
	// per token (decode speed) - using total params for decode speed would
	// underestimate it, per the MoE limitation in METHODOLOGY.md.
	hw := Hardware{Name: "test", BandwidthGBs: 273, MemoryGB: 64}
	moe := AdviseMoE(46.7e9, 12.9e9, hw, 0.7)
	dense := AdviseMoE(46.7e9, 46.7e9, hw, 0.7)
	for i := range moe {
		if moe[i].WeightGB != dense[i].WeightGB {
			t.Fatalf("%s: memory footprint must come from total params regardless of active params: moe=%v dense=%v",
				moe[i].Quant.Name, moe[i].WeightGB, dense[i].WeightGB)
		}
		if moe[i].Fits != dense[i].Fits {
			t.Fatalf("%s: Fits must come from total params: moe=%v dense=%v", moe[i].Quant.Name, moe[i].Fits, dense[i].Fits)
		}
		if moe[i].DecodeTokS <= dense[i].DecodeTokS {
			t.Fatalf("%s: active-param decode (%v) should be faster than total-param decode (%v)",
				moe[i].Quant.Name, moe[i].DecodeTokS, dense[i].DecodeTokS)
		}
	}
	// Q4_K_M active bytes = 12.9e9 * 4.85/8; decode = 273e9*0.7/that.
	q4 := moe[0]
	for _, r := range moe {
		if r.Quant.Name == "Q4_K_M" {
			q4 = r
		}
	}
	wantDecode := DecodeTokS(273, WeightBytes(12.9e9, 4.85), 0.7)
	almost(t, q4.DecodeTokS, wantDecode, 0.01)
}

func TestPresetsSane(t *testing.T) {
	for key, hw := range HardwarePresets {
		if hw.BandwidthGBs <= 0 || hw.MemoryGB <= 0 || hw.Name == "" {
			t.Fatalf("preset %q is incomplete: %+v", key, hw)
		}
	}
	seen := map[float64]bool{}
	for _, q := range Quants {
		if q.BitsPerWeight <= 0 || q.BitsPerWeight > 16 {
			t.Fatalf("quant %s has implausible bpw %v", q.Name, q.BitsPerWeight)
		}
		if seen[q.BitsPerWeight] {
			t.Fatalf("duplicate bpw %v", q.BitsPerWeight)
		}
		seen[q.BitsPerWeight] = true
	}
}
