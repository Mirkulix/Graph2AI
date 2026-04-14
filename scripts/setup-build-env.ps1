<#
.SYNOPSIS
  Configures the build environment for qlang on Windows (GNU toolchain + LLVM 18).

.DESCRIPTION
  - Ensures Rust GNU toolchain is active (stable-x86_64-pc-windows-gnu).
  - Prepends MinGW and MSYS2/mingw64 bin dirs to PATH (user scope + current session).
  - Sets LLVM_SYS_180_PREFIX for llvm-sys / inkwell.
  - Optionally verifies the environment by running llvm-config and cargo check.

.PARAMETER Session
  Apply to current PowerShell session only (no persistent user env changes).

.PARAMETER Verify
  After applying, run `llvm-config --version` and `cargo check --offline` to verify.

.EXAMPLE
  .\scripts\setup-build-env.ps1              # persistent + session
  .\scripts\setup-build-env.ps1 -Session     # session only
  .\scripts\setup-build-env.ps1 -Verify      # persistent + session + sanity check
#>

[CmdletBinding()]
param(
    [switch]$Session,
    [switch]$Verify
)

$ErrorActionPreference = 'Stop'

# --- Paths (edit if your layout differs) ---
$MinGwBin       = 'C:\Users\a.b\mingw64\bin'
$Msys2MingwBin  = 'C:\msys64\mingw64\bin'
$LlvmPrefix     = 'C:\msys64\mingw64'
$RustToolchain  = 'stable-x86_64-pc-windows-gnu'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "OK  $msg"  -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "!!  $msg"  -ForegroundColor Yellow }

# --- 1. Sanity checks ---
Write-Step 'Sanity checks'
foreach ($p in @($MinGwBin, $Msys2MingwBin, $LlvmPrefix)) {
    if (-not (Test-Path $p)) {
        Write-Warn2 "Pfad fehlt: $p  (weiter mit Vorsicht)"
    } else {
        Write-Ok "gefunden: $p"
    }
}

$llvmConfig = Join-Path $LlvmPrefix 'bin\llvm-config.exe'
if (Test-Path $llvmConfig) {
    $ver = & $llvmConfig --version
    Write-Ok "llvm-config: $ver"
    if ($ver -notlike '18.*') {
        Write-Warn2 "Achtung: llvm-config meldet $ver, erwartet 18.x (inkwell = llvm18-0)"
    }
} else {
    Write-Warn2 "llvm-config.exe nicht gefunden: $llvmConfig"
}

# --- 2. Rust toolchain ---
Write-Step "Rust-Toolchain auf $RustToolchain stellen"
if (Get-Command rustup -ErrorAction SilentlyContinue) {
    $installed = (rustup toolchain list) -join "`n"
    if ($installed -notmatch [regex]::Escape($RustToolchain)) {
        Write-Step "Installiere Toolchain $RustToolchain"
        rustup toolchain install $RustToolchain | Out-Null
    }
    rustup default $RustToolchain | Out-Null
    Write-Ok "Default-Toolchain: $(rustup show active-toolchain)"
} else {
    Write-Warn2 'rustup nicht im PATH — Toolchain-Schritt übersprungen'
}

# --- 3. Session PATH + env vars ---
Write-Step 'Session-Umgebung setzen'
$env:LLVM_SYS_180_PREFIX = $LlvmPrefix
$sessionPrepend = @($Msys2MingwBin, $MinGwBin) -join ';'
$env:PATH = "$sessionPrepend;$env:PATH"
Write-Ok "LLVM_SYS_180_PREFIX = $env:LLVM_SYS_180_PREFIX"
Write-Ok 'PATH um MinGW + MSYS2/mingw64 erweitert (Session)'

# --- 4. Persistent user env (default) ---
if (-not $Session) {
    Write-Step 'Benutzer-Umgebungsvariablen dauerhaft setzen'
    [Environment]::SetEnvironmentVariable('LLVM_SYS_180_PREFIX', $LlvmPrefix, 'User')
    Write-Ok "LLVM_SYS_180_PREFIX dauerhaft: $LlvmPrefix"

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ([string]::IsNullOrEmpty($userPath)) { $userPath = '' }
    $existing = $userPath -split ';' | Where-Object { $_ }
    $desired  = @($Msys2MingwBin, $MinGwBin)
    $needed   = $desired | Where-Object { $existing -notcontains $_ }
    if ($needed.Count -gt 0) {
        $newUserPath = (@($needed) + $existing) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
        Write-Ok "Benutzer-PATH um $($needed -join ', ') ergänzt (vorne)"
    } else {
        Write-Ok 'Benutzer-PATH enthält MinGW + MSYS2/mingw64 bereits'
    }
} else {
    Write-Warn2 'Session-Modus: keine dauerhaften Änderungen am Benutzerprofil'
}

# --- 5. Verify (optional) ---
if ($Verify) {
    Write-Step 'Verifikation'
    & $llvmConfig --version --prefix
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
    Push-Location $repoRoot
    try {
        Write-Step 'cargo check --offline (mit LLVM-Feature)'
        cargo check --offline
        if ($LASTEXITCODE -eq 0) { Write-Ok 'cargo check erfolgreich' }
        else                     { Write-Warn2 "cargo check exit $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

Write-Step 'Fertig.'
Write-Host 'Neue Shells erben die dauerhaften Variablen. Aktuelle Shell ist bereits scharfgestellt.' -ForegroundColor Gray
