$ErrorActionPreference = "Stop"

param(
    [switch]$InstallOnly,
    [switch]$SkipModelDownloads,
    [switch]$SkipWSLNVFP4,
    [switch]$SkipBF16,
    [switch]$SkipGGUF,
    [switch]$SkipNVFP4,
    [switch]$SkipTTS,
    [string]$WslDistro = "Ubuntu"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$webDir = Join-Path $root "web"
$requirementsPath = Join-Path $root "requirements-lab.txt"
$llamaRoot = Join-Path $root "tools\llama.cpp"
$llamaBin = Join-Path $llamaRoot "bin\llama-server.exe"
$llamaBench = Join-Path $llamaRoot "bin\llama-bench.exe"
$llamaZip = Join-Path $llamaRoot "llama-b8664-bin-win-cuda-13.1-x64.zip"
$cudartZip = Join-Path $llamaRoot "cudart-llama-bin-win-cuda-13.1-x64.zip"
$llamaUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b8664/llama-b8664-bin-win-cuda-13.1-x64.zip"
$cudartUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b8664/cudart-llama-bin-win-cuda-13.1-x64.zip"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Invoke-VenvPython {
    param([string[]]$Args)
    & $venvPython @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Args -join ' ')"
    }
}

function Ensure-Venv {
    if (Test-Path $venvPython) {
        return
    }

    Write-Step "Creating Python virtual environment"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv (Join-Path $root ".venv")
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $root ".venv")
    } else {
        throw "Python 3.12+ is required to create .venv."
    }

    if (-not (Test-Path $venvPython)) {
        throw "Failed to create Python virtual environment."
    }
}

function Ensure-PythonDependencies {
    Write-Step "Installing Python dependencies"
    Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-VenvPython @("-m", "pip", "install", "-r", $requirementsPath)
}

function Ensure-WebDependencies {
    Write-Step "Installing web dependencies"
    Push-Location $webDir
    try {
        & npm ci
        if ($LASTEXITCODE -ne 0) {
            throw "npm ci failed."
        }

        Write-Step "Building the React frontend"
        & npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "npm run build failed."
        }
    }
    finally {
        Pop-Location
    }
}

function Ensure-LlamaCpp {
    if ((Test-Path $llamaBin) -and (Test-Path $llamaBench)) {
        return
    }

    Write-Step "Installing llama.cpp Windows CUDA binaries"
    New-Item -ItemType Directory -Force -Path $llamaRoot | Out-Null

    if (-not (Test-Path $llamaZip)) {
        Invoke-WebRequest -Uri $llamaUrl -OutFile $llamaZip
    }
    if (-not (Test-Path $cudartZip)) {
        Invoke-WebRequest -Uri $cudartUrl -OutFile $cudartZip
    }

    Expand-Archive -LiteralPath $llamaZip -DestinationPath $llamaRoot -Force
    Expand-Archive -LiteralPath $cudartZip -DestinationPath $llamaRoot -Force

    if (-not (Test-Path $llamaBin)) {
        throw "llama.cpp install failed: llama-server.exe is missing."
    }
}

function Ensure-WslNvfp4Environment {
    param([string]$Distro)

    if ($SkipWSLNVFP4) {
        return
    }

    Write-Step "Preparing WSL vLLM environment for NVFP4 in distro '$Distro'"
    $availableDistros = (wsl -l -q) -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if ($availableDistros -notcontains $Distro) {
        throw "WSL distro '$Distro' is not installed. Install Ubuntu first or rerun with -SkipWSLNVFP4."
    }

    $wslCommand = @"
set -euo pipefail
python3 -m venv ~/vllm-gemma4
source ~/vllm-gemma4/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm==0.19.0 nvidia-modelopt huggingface_hub
python -m pip install --upgrade git+https://github.com/huggingface/transformers.git
"@

    & wsl -d $Distro bash -lc $wslCommand
    if ($LASTEXITCODE -ne 0) {
        throw "WSL NVFP4 environment setup failed."
    }
}

function Prefetch-Models {
    if ($SkipModelDownloads) {
        return
    }

    Write-Step "Prefetching Gemma checkpoints, GGUF assets, and local TTS voice"
    $prefetchArgs = @("scripts\prefetch_gemma4_assets.py")
    if ($SkipBF16) {
        $prefetchArgs += "--skip-bf16"
    }
    if ($SkipGGUF) {
        $prefetchArgs += "--skip-gguf"
    }
    if ($SkipNVFP4 -or $SkipWSLNVFP4) {
        $prefetchArgs += "--skip-nvfp4"
    }
    if ($SkipTTS) {
        $prefetchArgs += "--skip-tts"
    }

    Invoke-VenvPython $prefetchArgs
}

function Stop-ExistingLab {
    param([int]$Port = 8000)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        return
    }

    foreach ($processId in ($connections | Select-Object -ExpandProperty OwningProcess -Unique)) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

Push-Location $root
try {
    Require-Command npm
    if (-not $SkipWSLNVFP4) {
        Require-Command wsl
    }
    Ensure-Venv
    Ensure-PythonDependencies
    Ensure-WebDependencies
    Ensure-LlamaCpp
    Ensure-WslNvfp4Environment -Distro $WslDistro
    Prefetch-Models

    if ($InstallOnly) {
        Write-Step "Install finished. Launch later with .\\run_gemma4_lab.ps1"
        return
    }

    Write-Step "Launching Gemma 4 Lab on http://127.0.0.1:8000"
    Stop-ExistingLab -Port 8000
    & $venvPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
