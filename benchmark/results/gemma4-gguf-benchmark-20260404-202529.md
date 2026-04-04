# Gemma 4 GGUF Benchmark

- Runtime: `llama.cpp` CUDA
- Prompt benchmark: `256` prompt tokens
- Generation benchmark: `128` generated tokens
- Repetitions: `2`

| Model | Quant | Source | Status | Prompt tok/s | Gen tok/s | Stddev |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Gemma 4 E2B | Q8_0 | official | ok | 16545.45 | 243.41 | 8.29 |
| Gemma 4 E2B | Q4_0 | community | ok | 13077.21 | 285.29 | 4.44 |
| Gemma 4 E4B | Q8_0 | official | ok | 10537.28 | 154.83 | 1.88 |
| Gemma 4 E4B | Q4_0 | community | ok | 9387.39 | 191.27 | 0.09 |
| Gemma 4 26B A4B | Q8_0 | official | ok | 5795.03 | 166.75 | 1.57 |
| Gemma 4 26B A4B | Q4_0 | community | ok | 5267.01 | 184.50 | 2.12 |
| Gemma 4 31B | Q8_0 | official | failed | - | - | - |

- `Gemma 4 31B Q8_0`: D:\a\llama.cpp\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:98: CUDA error

| Gemma 4 31B | Q4_0 | community | ok | 3530.86 | 66.12 | 0.02 |