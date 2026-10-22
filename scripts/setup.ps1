<#
.SYNOPSIS
    One-time setup: Python environment, web app dependencies, and a local .env file.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Native([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: $exe $($arguments -join ' ')" }
}

# 1. Python 3.12 environment
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating the Python 3.12 environment..." -ForegroundColor Cyan
    Invoke-Native "py" @("-3.12", "-m", "venv", ".venv")
}
$python = Join-Path $root ".venv\Scripts\python.exe"
Invoke-Native $python @("-m", "pip", "install", "--upgrade", "pip", "--quiet")

# 2. PyTorch: the CUDA build when an NVIDIA GPU is present, otherwise the CPU build
& $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    $index = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cu128" } else { "cpu" }
    Write-Host "Installing PyTorch ($index)..." -ForegroundColor Cyan
    Invoke-Native $python @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/$index")
}

# 3. The project and its pipelines
Write-Host "Installing the project..." -ForegroundColor Cyan
Invoke-Native $python @("-m", "pip", "install", "-e", ".[data,train,recsys,search,galaxy,api,copilot,dev]", "--quiet")

# 4. Local settings with a private API key (the key is never printed)
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
$settings = [IO.File]::ReadAllText((Join-Path $root ".env"))
$emptyKey = "(?m)^(EIE_API_KEY=)[ \t]*(?=\r?$)"
if ($settings -match $emptyKey) {
    $key = & $python -c "import secrets; print(secrets.token_urlsafe(32))"
    $settings = $settings -replace $emptyKey, ('${1}' + $key)
    [IO.File]::WriteAllText((Join-Path $root ".env"), $settings)
    Write-Host "Created a private API key in .env" -ForegroundColor Cyan
}

# 5. Web app
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Host "Installing the web app..." -ForegroundColor Cyan
    Push-Location web
    # A clean install the first time; later runs only add what changed.
    $mode = if (Test-Path "node_modules") { "install" } else { "ci" }
    try { Invoke-Native "npm" @($mode, "--no-audit", "--no-fund") } finally { Pop-Location }
} else {
    Write-Warning "Node.js 20 or later is needed for the web app: https://nodejs.org"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next: put your Kaggle token in $HOME\.kaggle, accept the competition rules, then run"
Write-Host "  .\.venv\Scripts\eie.exe build"
