# Gemma 4 Local Lab

Local Gemma 4 workstation lab with:

- FastAPI backend
- React frontend
- local TTS
- model / quantization switching
- benchmark scripts and benchmark reports

## Benchmark setup

- GPU: `NVIDIA GeForce RTX 5090 32 GB`
- Driver: `581.42`
- CPU: `AMD Ryzen 9 9950X3D`
- Date: `2026-04-04`

Two benchmark paths were measured:

1. `BF16` with the local `Transformers` runtime used by the app backend
2. quantized `GGUF` models with `llama.cpp` CUDA

Important note:

- the Google table you sent lists `BF16 / SFP8 / Q4_0`
- the practical 8-bit benchmark I measured locally is `Q8_0`, not the exact Google `SFP8` format
- the practical 4-bit benchmark I measured locally is `Q4_0`

## Generation throughput graphs

### Bubble chart

![Gemma 4 benchmark bubble chart](benchmark/results/gemma4-benchmark-bubble.svg)

This chart uses:

- `X` = total model size in billions of parameters
- `Y` = measured generation tokens per second
- bubble size = bit-depth, with `16-bit` larger than `8-bit`, and `8-bit` larger than `4-bit`
- `KO` models shown on the right when they did not run cleanly on this workstation

### BF16 gen tok/s

```mermaid
xychart-beta
    title "Gemma 4 BF16 Generation Throughput on RTX 5090"
    x-axis ["E2B BF16", "E4B BF16"]
    y-axis "tokens / second" 0 --> 20
    bar [15.96, 13.23]
```

### Quantized gen tok/s

```mermaid
xychart-beta
    title "Gemma 4 Quantized Generation Throughput on RTX 5090"
    x-axis ["E2B Q8", "E2B Q4", "E4B Q8", "E4B Q4", "26B Q8", "26B Q4", "31B Q4"]
    y-axis "tokens / second" 0 --> 300
    bar [243.41, 285.29, 154.83, 191.27, 166.75, 184.50, 66.12]
```

## BF16 benchmark

Runtime:

- `Transformers`
- text-only generation
- one model per fresh Python process
- warmup before measured runs

| Model | Status | Mean gen tok/s | Best gen tok/s | Load time | VRAM after load |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | ok | 15.96 | 16.20 | 10.69 s | 9.67 GiB |
| Gemma 4 E4B | ok | 13.23 | 13.26 | 11.10 s | 14.90 GiB |
| Gemma 4 26B A4B | failed | - | - | - | - |
| Gemma 4 31B | failed | - | - | - | - |

BF16 failure notes:

- `Gemma 4 26B A4B`: Windows paging file / commit limit failure during load
- `Gemma 4 31B`: Windows paging file / commit limit failure during load

## Quantized GGUF benchmark

Runtime:

- `llama.cpp` CUDA
- prompt benchmark: `256` prompt tokens
- generation benchmark: `128` generated tokens
- repetitions: `2`

| Model | Quant | Source | Prompt tok/s | Gen tok/s | Status |
| --- | --- | --- | ---: | ---: | --- |
| Gemma 4 E2B | Q8_0 | official | 16545.45 | 243.41 | ok |
| Gemma 4 E2B | Q4_0 | community | 13077.21 | 285.29 | ok |
| Gemma 4 E4B | Q8_0 | official | 10537.28 | 154.83 | ok |
| Gemma 4 E4B | Q4_0 | community | 9387.39 | 191.27 | ok |
| Gemma 4 26B A4B | Q8_0 | official | 5795.03 | 166.75 | ok |
| Gemma 4 26B A4B | Q4_0 | community | 5267.01 | 184.50 | ok |
| Gemma 4 31B | Q8_0 | official | - | - | failed |
| Gemma 4 31B | Q4_0 | community | 3530.86 | 66.12 | ok |

Quantized failure notes:

- `Gemma 4 31B Q8_0`: CUDA load failure on this RTX 5090 setup with the tested `llama.cpp` build

## Quick takeaways

- Fastest small model: `Gemma 4 E2B Q4_0` at `285.29 tok/s`
- Best balanced small model: `Gemma 4 E4B Q4_0` at `191.27 tok/s`
- Biggest model that ran cleanly in quantized mode: `Gemma 4 31B Q4_0` at `66.12 tok/s`
- Best large-model compromise on this machine: `Gemma 4 26B A4B Q4_0` at `184.50 tok/s`

## Files

- BF16 summary: `benchmark/results/gemma4-5090-summary-20260404.md`
- GGUF summary: `benchmark/results/gemma4-gguf-benchmark-20260404-202529.md`
- BF16 raw JSON:
  - `benchmark/results/gemma4-5090-benchmark-20260404-194128.json`
  - `benchmark/results/gemma4-5090-benchmark-20260404-194004.json`
  - `benchmark/results/gemma4-5090-benchmark-20260404-194021.json`
  - `benchmark/results/gemma4-5090-benchmark-20260404-194036.json`
- GGUF raw JSON:
  - `benchmark/results/gemma4-gguf-benchmark-20260404-202529.json`

## Re-run

BF16:

```powershell
.\.venv\Scripts\python.exe .\benchmark\benchmark_gemma4.py --models e2b e4b 26b-a4b 31b --max-new-tokens 128 --warmup-tokens 32 --runs 2
```

GGUF:

```powershell
.\.venv\Scripts\python.exe .\benchmark\benchmark_gemma4_gguf.py
```
