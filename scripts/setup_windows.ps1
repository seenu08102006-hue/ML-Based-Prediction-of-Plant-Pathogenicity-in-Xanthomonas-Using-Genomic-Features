# Set up the project on Windows (PowerShell 5.1 or 7+).
#
# Safe to run repeatedly: an existing .venv is reused, never deleted.
# Does NOT modify your PowerShell profile, execution policy or PATH.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Optionally choose an interpreter:  -PythonVersion 3.12

param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Info($msg) { Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[setup] ERROR: $msg" -ForegroundColor Red; exit 1 }

# 1-2. Locate a supported Python (prefer the 'py' launcher with the requested version)
$PythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py "-$PythonVersion" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $PythonCmd = @("py", "-$PythonVersion") }
}
if (-not $PythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) { $PythonCmd = @("python") }
if (-not $PythonCmd) {
    Fail "Python not found. Install Python 3.10-3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')."
}
$exe = $PythonCmd[0]
$pyArgs = @($PythonCmd | Select-Object -Skip 1)
$version = & $exe @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
& $exe @pyArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "Python $version found, but 3.10 or newer is required." }
Info "Using Python $version ($($PythonCmd -join ' '))"

# 3. Create the virtual environment only if missing
if (Test-Path $VenvPython) {
    Info "Reusing existing virtual environment $VenvDir"
} else {
    Info "Creating virtual environment $VenvDir"
    & $exe @pyArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}

# 4-5. Install dependencies and the project in editable (development) mode
Info "Installing dependencies and the project package"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed (see messages above)." }
& $VenvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) { Fail "Project installation failed." }

# 7. Basic environment check
& $VenvPython -m xanthomonas_ml check-env
if ($LASTEXITCODE -ne 0) { Fail "Environment check reported problems." }

# 6 + 8. Activation help and next commands
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the environment in this PowerShell session with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "  (if blocked by the execution policy, run once per session:"
Write-Host "    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass )"
Write-Host ""
Write-Host "Next commands:"
Write-Host "    python -m xanthomonas_ml download-data"
Write-Host "    python -m xanthomonas_ml run-all"
Write-Host "    python -m xanthomonas_ml test"
