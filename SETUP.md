# SETUP.md

This file is written for AI coding agents such as Codex or Claude.

If a user says something like:

- "Install this repo on my machine"
- "Set up this Gemma 4 lab locally"
- "Make this project run"

use this file as the default setup playbook.

## What This Repository Is

This repo is a local **Gemma 4 workstation lab**.

It includes:

- a `FastAPI` backend
- a `React` frontend
- local `Piper` TTS
- multiple Gemma 4 model variants
- multiple runtime paths depending on quantization
- optional `WSL + vLLM` support for NVIDIA's `nvidia/Gemma-4-31B-IT-NVFP4`

The main user-facing result is a web app served at:

- `http://127.0.0.1:8000`

## Available Features

These are the main features currently implemented in the lab.

### Core UX

- text chat with Gemma 4
- model switching from the UI
- quantization switching from the UI
- explicit `Load model` flow with load status and progress
- streaming assistant replies
- Markdown rendering in assistant responses
- send on `Enter`
- local chat persistence in the browser
- automatic thread titles for new chats
- chat history sidebar

### Model And Runtime Features

- `BF16` inference through `Transformers`
- `SFP8` slot mapped to practical local `Q8_0 GGUF` via `llama.cpp`
- `Q4_0` inference through `llama.cpp`
- `NVFP4` support through `WSL Ubuntu + vLLM`
- unload/reload logic when switching models
- runtime health reporting through `/api/health`

### Multimodal Features

- text input
- image input on supported models
- audio input on supported small models
- live camera mode in the UI
- recorded voice-turn flow for supported models

Important:

- Gemma 4 in this lab produces **text output**
- local voice playback is generated server-side with `Piper`
- the spoken output is not native audio generation from Gemma itself

### Voice And Audio Features

- local `Piper` TTS generation on the backend
- custom in-app audio player
- optional voice playback for assistant replies
- audio clip serving through the backend

### Generation Controls

- configurable max output tokens
- continuation support
- auto-continue support
- temperature and sampling controls in the app

### Benchmark And Research Features

- benchmark scripts for `BF16`
- benchmark scripts for `GGUF`
- benchmark scripts for `NVFP4`
- benchmark reports in `benchmark/results`
- throughput charts in the repo docs

### Operational Features

- rotating backend log file
- dedicated `llama.cpp` log
- dedicated `WSL vLLM` log
- stale process cleanup for local runtimes
- safer relaunch script for the lab

## Important Runtime Mapping

Do not assume every UI quantization maps to the same backend family.

Current runtime mapping:

| UI quantization | Actual runtime | Notes |
| --- | --- | --- |
| `BF16` | `Transformers` | Official Hugging Face Google checkpoints |
| `SFP8` | `llama.cpp` | In this repo, this is implemented as a practical `Q8_0 GGUF` local path |
| `Q4_0` | `llama.cpp` | `Q4_0 GGUF` |
| `NVFP4` | `vLLM` in `WSL Ubuntu` | For `nvidia/Gemma-4-31B-IT-NVFP4` |

Important:

- `SFP8` in the UI is **not** the raw Google tensor format in this local app
- it is intentionally mapped to `Q8_0 GGUF` so the model can run locally through `llama.cpp`
- `NVFP4` is a separate NVIDIA runtime path and should not be treated like `Q4_0`

## Main Files You Should Know

- `install_and_run_gemma4_lab.ps1`
- `run_gemma4_lab.ps1`
- `backend/app.py`
- `web/package.json`
- `requirements-lab.txt`
- `scripts/prefetch_gemma4_assets.py`
- `logs/gemma4-lab.log`
- `logs/llama-server.log`
- `logs/vllm-wsl.log`

## Preferred Agent Behavior

If the user asks you to install the repo, do **not** rebuild the setup manually first.

Start with the repo's own bootstrap script unless the user explicitly wants a custom install.

Preferred order:

1. inspect the repo state quickly
2. run the official install script
3. verify the lab is reachable on `http://127.0.0.1:8000`
4. verify `/api/health`
5. only do manual repairs if the scripted install fails

## Host Assumptions

This repo is designed primarily for:

- Windows
- PowerShell
- NVIDIA GPU
- local CUDA inference

Optional path:

- `WSL Ubuntu` for the NVIDIA `NVFP4` model through `vLLM`

## Minimum Prerequisites

Check these before running setup:

- `npm`
- `Python 3.12+`
- `PowerShell`
- NVIDIA GPU drivers installed

Optional but required for the `NVFP4` path:

- `wsl`
- a distro named `Ubuntu`

## Fastest Successful Install Path

From the repo root, run:

```powershell
.\install_and_run_gemma4_lab.ps1
```

This is the default full install path.

It will:

- create `.venv` if needed
- install Python dependencies from `requirements-lab.txt`
- install frontend dependencies with `npm ci`
- build the frontend
- install Windows CUDA `llama.cpp` binaries if missing
- prepare the `WSL` Python environment for `NVFP4`
- prefetch model assets into `.\.hf-cache`
- launch the lab on port `8000`

## Faster or Lighter Install Variants

If the user wants a lighter install, use one of these:

### Install only, do not launch

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly
```

### Skip the NVIDIA WSL path

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipWSLNVFP4
```

### Skip model downloads

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipModelDownloads
```

### Skip GGUF models

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipGGUF
```

### Skip NVFP4 model

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipNVFP4
```

### Skip TTS assets

```powershell
.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipTTS
```

## Quick Relaunch After Install

If the environment is already installed, prefer:

```powershell
.\run_gemma4_lab.ps1
```

This script is the correct relaunch entry point.

It is safer than launching `uvicorn` manually because it also:

- rebuilds the frontend
- stops a previous process on port `8000`
- kills stale `llama-server.exe` processes from this workspace

## Health Check After Install

After launch, verify:

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/api/health'
```

What success looks like:

- the HTTP request returns `200`
- `status` is `ready`
- the backend reports the GPU correctly
- the app is reachable in the browser

Note:

- `loaded` can still be `false` if no model is warmed yet
- that is valid for a fresh launch

## Recommended Validation Flow

After the app is up:

1. load a small model first
2. validate one generation call
3. only then try larger variants

Good first validation target:

- `Gemma 4 E2B`
- quantization `SFP8`

If you want a direct API smoke test after a model is loaded:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/models/load' `
  -ContentType 'application/json' `
  -Body (@{ model_key = 'e2b'; quantization_key = 'sfp8' } | ConvertTo-Json)
```

Then:

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/api/health'
```

## How The App Is Structured

### Backend

`backend/app.py` handles:

- model catalog
- model loading
- runtime switching
- FastAPI routes
- TTS
- `llama.cpp` process management
- `WSL vLLM` integration

### Frontend

The React app is in:

- `web/src/App.jsx`
- `web/src/App.css`

### Model Assets

Assets are stored in:

- `.\.hf-cache`

This cache is shared across:

- local `Transformers`
- local `llama.cpp`
- the `WSL vLLM` path when configured by the repo scripts

## Model Families In Practice

Typical local options:

- `E2B / SFP8`
- `E4B / SFP8`
- `26B A4B / SFP8`
- `31B / SFP8`
- `Q4_0` variants for smaller memory pressure

Special NVIDIA option:

- `31B IT NVFP4 / NVFP4`

## Known Heavy Paths

These are the main expensive paths:

- full bootstrap with all model downloads
- `26B A4B`
- `31B`
- `31B IT NVFP4`

If the user only wants the app running quickly, do not force every model download.

## Memory And Process Notes

This repo already contains fixes for stale `llama.cpp` processes.

Still, when debugging load issues, always check:

- RAM usage
- VRAM usage
- lingering `llama-server.exe`
- WSL `vLLM` processes

Useful commands:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue
```

```powershell
nvidia-smi
```

```powershell
Get-NetTCPConnection -LocalPort 8000,8011,8012 -State Listen -ErrorAction SilentlyContinue
```

## Where To Look If Setup Fails

Check these logs:

- `logs/gemma4-lab.log`
- `logs/llama-server.log`
- `logs/vllm-wsl.log`

Use them in this order:

1. `gemma4-lab.log` for high-level orchestration
2. `llama-server.log` for `Q4_0` / `SFP8` local runtime issues
3. `vllm-wsl.log` for `NVFP4` problems

## Common Failure Patterns

### The app starts but no model answers

Check:

- `/api/health`
- current `load_state`
- whether the model is actually loaded

### Memory is full after switching models

Check for orphaned:

- `llama-server.exe`
- WSL `vLLM` engine processes

Prefer relaunching with:

```powershell
.\run_gemma4_lab.ps1
```

### BF16 fails on large models

That is often expected on Windows if commit memory or page file space is insufficient.

Prefer:

- `SFP8`
- `Q4_0`

### NVFP4 fails

Check:

- `WSL Ubuntu` exists
- WSL environment was installed by the repo script
- enough GPU memory is free

## What An AI Agent Should Say To The User

If the user simply asks for setup, a good default plan is:

1. run `.\install_and_run_gemma4_lab.ps1`
2. wait for dependencies and model assets to install
3. verify `http://127.0.0.1:8000`
4. verify `/api/health`
5. load a small model and test one prompt

If the user wants the fastest path with fewer downloads:

1. run `.\install_and_run_gemma4_lab.ps1 -InstallOnly -SkipModelDownloads -SkipWSLNVFP4`
2. run `.\run_gemma4_lab.ps1`
3. only prefetch specific models later if needed

## Suggested Prompt For Another Agent

If you are another agent and need a ready-made instruction, use something like:

> Install this repo on the current Windows machine using the repo's own setup scripts. Prefer the official PowerShell bootstrap over manual reimplementation. Verify the app is reachable on http://127.0.0.1:8000, check /api/health, then load a small Gemma model and run one generation smoke test. If setup fails, inspect logs/gemma4-lab.log, logs/llama-server.log, and logs/vllm-wsl.log before changing code.

## Final Rule

When in doubt:

- prefer the repo scripts
- prefer verification over assumptions
- prefer reading logs before editing code
- prefer small-model validation before large-model benchmarking
