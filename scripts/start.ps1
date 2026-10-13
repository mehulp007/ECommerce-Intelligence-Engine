<#
.SYNOPSIS
    Start the API and the web app, then open http://127.0.0.1:3000. Press Ctrl+C to stop both.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start.ps1
.PARAMETER NoBrowser
    Do not open the browser.
#>
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$eie = Join-Path $root ".venv\Scripts\eie.exe"
$web = Join-Path $root "web"
$logs = Join-Path $root "logs"

if (-not (Test-Path $eie)) { throw "Run scripts\setup.ps1 first." }
if (-not (Test-Path (Join-Path $root "data\serving\manifest.json"))) {
    throw "Build the data first: .\.venv\Scripts\eie.exe build"
}

function Wait-Url([string]$url, [int]$seconds, $process) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        } catch {
            if ($process -and $process.HasExited) { return $false }
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

New-Item -ItemType Directory -Force $logs | Out-Null
$api = $null
$site = $null
try {
    if (-not (Wait-Url "http://127.0.0.1:8000/health" 1 $null)) {
        Write-Host "Starting the API (log: logs\api.log)..." -ForegroundColor Cyan
        $api = Start-Process $eie -ArgumentList "serve", "run" -WorkingDirectory $root -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $logs "api.log") -RedirectStandardError (Join-Path $logs "api-errors.log")
        if (-not (Wait-Url "http://127.0.0.1:8000/health" 180 $api)) {
            throw "The API did not start. See logs\api-errors.log"
        }
    }

    # Build the web app the first time, and again after its source changes.
    $built = Join-Path $web ".next\BUILD_ID"
    $sources = Get-ChildItem (Join-Path $web "app"), (Join-Path $web "components"), (Join-Path $web "lib") -Recurse -File
    $newest = ($sources | Measure-Object LastWriteTime -Maximum).Maximum
    if (-not (Test-Path $built) -or (Get-Item $built).LastWriteTime -lt $newest) {
        Write-Host "Building the web app (first run takes about a minute)..." -ForegroundColor Cyan
        Push-Location $web
        try {
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "The web build failed." }
        } finally { Pop-Location }
    }

    Write-Host "Starting the web app..." -ForegroundColor Cyan
    $site = Start-Process "npm.cmd" -ArgumentList "run", "start" -WorkingDirectory $web -NoNewWindow -PassThru
    if (-not (Wait-Url "http://127.0.0.1:3000" 60 $site)) { throw "The web app did not start." }
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:3000" }
    Write-Host "Running at http://127.0.0.1:3000 - press Ctrl+C to stop." -ForegroundColor Green
    $site.WaitForExit()
} finally {
    foreach ($process in @($site, $api)) {
        if ($process -and -not $process.HasExited) {
            # npm and uvicorn start child processes, so stop the whole tree.
            try { & taskkill /PID $process.Id /T /F | Out-Null } catch { }
        }
    }
}
