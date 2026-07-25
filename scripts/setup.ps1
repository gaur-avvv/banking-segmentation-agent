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

Write-Host "Choose provider: 1) Gemini  2) OpenRouter  3) Groq  4) OpenAI  5) Ollama  6) None"
$providerChoice = Read-Host "Provider [1]"
$provider = switch ($providerChoice) { "2" { "openrouter" } "3" { "groq" } "4" { "openai" } "5" { "ollama" } "6" { "none" } default { "gemini" } }
$keyName = switch ($provider) { "gemini" { "GEMINI_API_KEY" } "openrouter" { "OPENROUTER_API_KEY" } "groq" { "GROQ_API_KEY" } "openai" { "OPENAI_API_KEY" } default { "" } }
$lines = Get-Content .env | Where-Object { $_ -notmatch '^LLM_PROVIDER=' -and ($keyName -eq "" -or $_ -notmatch "^$keyName=") }
$lines += "LLM_PROVIDER=$provider"
if ($keyName) {
    $secure = Read-Host "Enter $keyName (hidden)" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    if ($plain) { $lines += "$keyName=$plain" }
}
$lines | Set-Content .env

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
