#!/usr/bin/env pwsh
[CmdletBinding()]
param(
	[string]$PythonExe = "python"
)

$ErrorActionPreference = 'Stop'

function Test-CommandExists {
	param([Parameter(Mandatory = $true)][string]$Name)
	return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Path 'requirements.txt')) {
	Write-Error "requirements.txt not found in current directory: $((Get-Location).Path)"
	exit 1
}

Write-Host "[setup] Installing Python dependencies from requirements.txt..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt

if (-not (Test-CommandExists -Name 'llama-benchy')) {
	Write-Host "[setup] llama-benchy not found in PATH. Installing with pip..."
	& $PythonExe -m pip install llama-benchy
}

if (-not (Test-CommandExists -Name 'llama-benchy')) {
	Write-Error "llama-benchy installation/lookup failed. Ensure llama-benchy is installed and available on PATH."
	exit 1
}

Write-Host "[setup] Verifying llama-benchy..."
& llama-benchy --help | Out-Null
Write-Host "[setup] Setup completed successfully."
