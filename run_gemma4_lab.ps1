$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
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
