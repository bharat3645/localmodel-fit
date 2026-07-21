package fit

import "testing"

func TestPrefillTokS(t *testing.T) {
	// 10 TFLOPS peak, 1B active params, MFU 0.35:
	// 10e12 * 0.35 / (2 * 1e9) = 3.5e12 / 2e9 = 1750 tok/s.
	almost(t, PrefillTokS(10e12, 1e9, 0.35), 1750, 0.01)

	// Prefill scales as 1/active_params: doubling params halves throughput.
	almost(t, PrefillTokS(10e12, 2e9, 0.35), 875, 0.01)

	// mfu <= 0 selects DefaultMFU.
	if PrefillTokS(10e12, 1e9, 0) != PrefillTokS(10e12, 1e9, DefaultMFU) {
		t.Fatal("mfu<=0 must select DefaultMFU")
	}

	// Degenerate inputs predict 0, never NaN/Inf.
	if PrefillTokS(0, 1e9, 0.35) != 0 {
		t.Fatal("zero flops must predict 0")
	}
	if PrefillTokS(10e12, 0, 0.35) != 0 {
		t.Fatal("zero params must predict 0")
	}
}

func TestPrefillUsesActiveParamsForMoE(t *testing.T) {
	// Mixtral-8x7B shape: 46.7B total resident, 12.9B active per token.
	// Prefill, like decode, only runs the routed experts, so it is bounded by
	// active params — using total would underestimate prefill throughput.
	total := PrefillTokS(50e12, 46.7e9, 0.35)
	active := PrefillTokS(50e12, 12.9e9, 0.35)
	if active <= total {
		t.Fatalf("active-param prefill (%v) must beat total-param prefill (%v)", active, total)
	}
	// Ratio tracks the param ratio exactly (compute is linear in params).
	almost(t, active/total, 46.7/12.9, 1e-9)
}

func TestTTFTAndEndToEnd(t *testing.T) {
	// 512-token prompt at 1750 tok/s prefill = 0.2926 s to first token.
	almost(t, TTFTSeconds(512, 1750), 512.0/1750.0, 1e-9)

	// End-to-end: prefill 512 @ 1750 tok/s + generate 128 @ 40 tok/s decode.
	want := 512.0/1750.0 + 128.0/40.0
	almost(t, EndToEndSeconds(512, 128, 1750, 40), want, 1e-9)

	// Non-positive rates degrade gracefully to 0 for that phase.
	if TTFTSeconds(512, 0) != 0 {
		t.Fatal("zero prefill rate must yield 0 TTFT")
	}
	// decode rate 0 => only the prefill term contributes.
	almost(t, EndToEndSeconds(512, 128, 1750, 0), 512.0/1750.0, 1e-9)
}
