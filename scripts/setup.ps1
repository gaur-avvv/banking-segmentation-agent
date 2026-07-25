$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$help = $args -contains "--help" -or $args -contains "-h"
if ($help) {
    Write-Host "Usage: .\scripts\setup.ps1 [-h|--help]"
    Write-Host "Installs the project, asks for a dataset path, and offers a start menu."
    exit 0
}
$env:UV_LINK_MODE = if ($env:UV_LINK_MODE) { $env:UV_LINK_MODE } else { "copy" }

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv sync --extra dev --extra adk
} else {
    if (-not (Test-Path .venv)) { py -m venv .venv }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -e ".[dev,adk]"
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example. Add GEMINI_API_KEY before cloud-model use."
} else { Write-Host ".env already exists; leaving it unchanged." }

$datasetPath = Read-Host "Dataset CSV, ZIP, or folder path (leave blank for automatic demo)"
if ($datasetPath) {
    $lines = if (Test-Path .env) { Get-Content .env | Where-Object { $_ -notmatch '^BANKING_DATA_PATH=' } } else { @() }
    ($lines + "BANKING_DATA_PATH=$datasetPath") | Set-Content .env
    Write-Host "Saved dataset path in .env"
}

Write-Host "Setup complete. Examples:"
Write-Host "  uv run banking-agent setup"
Write-Host "  uv run banking-agent chat"
Write-Host "  uv run banking-agent api --port 8000"
Write-Host "  uv run banking-agent adk-web --port 8001"

$choice = Read-Host "Choose start: 1=chat, 2=API, 3=Google ADK Web, 4=one-shot segmentation, 5=exit [5]"
switch ($choice) {
    "1" { & .\.venv\Scripts\banking-agent.exe chat }
    "2" { & .\.venv\Scripts\banking-agent.exe api --port 8000 }
    "3" { & .\.venv\Scripts\banking-agent.exe adk-web --port 8001 }
    "4" { & .\.venv\Scripts\banking-agent.exe run --query "Segment customers into priority, regular, and dormant groups" }
    default { Write-Host "Setup complete. Run a command above when ready." }
}
