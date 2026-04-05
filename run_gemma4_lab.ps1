$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    $connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($processId in ($connections | Select-Object -ExpandProperty OwningProcess -Unique)) {
            try {
                Stop-Process -Id $processId -Force -ErrorAction Stop
            }
            catch {
            }
        }
    }

    $llamaBin = Join-Path $root "tools\\llama.cpp\\bin\\llama-server.exe"
    $llamaProcesses = Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'" -ErrorAction SilentlyContinue
    if ($llamaProcesses) {
        foreach ($process in $llamaProcesses) {
            $commandLine = [string]$process.CommandLine
            if (-not $commandLine) {
                continue
            }
            if ($commandLine -like "*$llamaBin*" -or $commandLine -like "*--port 8011*") {
                try {
                    Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
                }
                catch {
                }
            }
        }
    }

    Push-Location (Join-Path $root "web")
    try {
        npm run build
    }
    finally {
        Pop-Location
    }

    .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
