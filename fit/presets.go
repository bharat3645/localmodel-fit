package fit

// Quants lists approximate llama.cpp bits-per-weight including quantization
// overhead, derived from published quantize output sizes (METHODOLOGY.md).
var Quants = []Quant{
	{Name: "F16", BitsPerWeight: 16.0},
	{Name: "Q8_0", BitsPerWeight: 8.5},
	{Name: "Q6_K", BitsPerWeight: 6.59},
	{Name: "Q5_K_M", BitsPerWeight: 5.69},
	{Name: "Q4_K_M", BitsPerWeight: 4.85},
	{Name: "Q4_0", BitsPerWeight: 4.55},
	{Name: "Q3_K_M", BitsPerWeight: 3.91},
	{Name: "Q2_K", BitsPerWeight: 3.35},
}

// HardwarePresets maps preset keys to peak memory bandwidth and the memory
// reachable by the accelerator. Apple figures are unified-memory specs with
// a common capacity config (override with --mem); GPU figures are VRAM.
var HardwarePresets = map[string]Hardware{
	"m1":       {Name: "Apple M1", BandwidthGBs: 68.25, MemoryGB: 16},
	"m1-pro":   {Name: "Apple M1 Pro", BandwidthGBs: 200, MemoryGB: 32},
	"m1-max":   {Name: "Apple M1 Max", BandwidthGBs: 400, MemoryGB: 64},
	"m1-ultra": {Name: "Apple M1 Ultra", BandwidthGBs: 800, MemoryGB: 128},
	"m2":       {Name: "Apple M2", BandwidthGBs: 100, MemoryGB: 16},
	"m2-pro":   {Name: "Apple M2 Pro", BandwidthGBs: 200, MemoryGB: 32},
	"m2-max":   {Name: "Apple M2 Max", BandwidthGBs: 400, MemoryGB: 96},
	"m2-ultra": {Name: "Apple M2 Ultra", BandwidthGBs: 800, MemoryGB: 192},
	"m3":       {Name: "Apple M3", BandwidthGBs: 102.4, MemoryGB: 16},
	"m3-pro":   {Name: "Apple M3 Pro", BandwidthGBs: 150, MemoryGB: 36},
	"m3-max":   {Name: "Apple M3 Max", BandwidthGBs: 400, MemoryGB: 128},
	"m4":       {Name: "Apple M4", BandwidthGBs: 120, MemoryGB: 16},
	"m4-pro":   {Name: "Apple M4 Pro", BandwidthGBs: 273, MemoryGB: 48},
	"m4-max":   {Name: "Apple M4 Max", BandwidthGBs: 546, MemoryGB: 128},
	"rtx-3090": {Name: "NVIDIA RTX 3090", BandwidthGBs: 936, MemoryGB: 24},
	"rtx-4090": {Name: "NVIDIA RTX 4090", BandwidthGBs: 1008, MemoryGB: 24},
	"rtx-5090": {Name: "NVIDIA RTX 5090", BandwidthGBs: 1792, MemoryGB: 32},
	"ddr4-2ch": {Name: "DDR4-3200 dual", BandwidthGBs: 51.2, MemoryGB: 64},
	"ddr5-2ch": {Name: "DDR5-5600 dual", BandwidthGBs: 89.6, MemoryGB: 64},
}
