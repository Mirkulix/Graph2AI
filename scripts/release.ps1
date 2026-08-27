<#
.SYNOPSIS
Cut a release: gate on the CI test suite, build the release binaries, print
SHA-256 checksums, and create the version tag. The operator picks the version;
the script does not bump crate versions automatically (that is a decision).

.EXAMPLE
./scripts/release.ps1 -Version 0.1.0           # gate + release build + tag v0.1.0
./scripts/release.ps1 -Version 0.1.0 -SkipTag  # gate + build, no tag
#>
param(
    [Parameter(Mandatory)][string]$Version,
    [switch]$SkipTag
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version must be MAJOR.MINOR.PATCH, got '$Version'" }

Write-Host "== Release gate v$Version ==" -ForegroundColor Cyan

Write-Host "`n[1/3] CI test gate..." -ForegroundColor Yellow
Push-Location $root
try {
    cargo test -p qo-knowledge -p qo-server -p qo-agents --no-default-features
    if ($LASTEXITCODE -ne 0) { throw "test gate failed" }
} finally { Pop-Location }
Write-Host "  tests green" -ForegroundColor Green

Write-Host "`n[2/3] Release build (--no-default-features, JIT off)..." -ForegroundColor Yellow
# On Windows GNU targets, cargo build scripts (windows-sys, parking_lot_core)
# need dlltool, which the Rust toolchain ships in its self-contained bin dir.
# It is not on PATH by default, so put it there for the release profile (which
# recompiles those crates from scratch, unlike a cached debug build).
$dlltool = Get-ChildItem (Join-Path (rustc --print sysroot).Trim() 'lib\rustlib') -Recurse `
    -Filter 'dlltool.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dlltool) {
    Write-Host "  found dlltool at $($dlltool.DirectoryName)"
    $env:PATH = "$($dlltool.DirectoryName);$env:PATH"
}
Push-Location $root
try {
    cargo build --release --bin qo --bin qlang --no-default-features 2>&1 | Tee-Object -Variable releaseOut | Select-Object -Last 5
    if ($LASTEXITCODE -ne 0) {
        if ($releaseOut -match 'dlltool') {
            Write-Host "`nRELEASE BUILD FAILED: the windows-gnu toolchain here lacks a working dlltool" -ForegroundColor Red
            Write-Host "(the self-contained one cannot create import libraries). Install a complete" -ForegroundColor Red
            Write-Host "mingw-w64 binutils (dlltool) or run this on a full toolchain; no tag was created." -ForegroundColor Red
        } else {
            Write-Host "`nRELEASE BUILD FAILED (see above); no tag was created." -ForegroundColor Red
        }
        throw "release build failed"
    }
} finally { Pop-Location }
foreach ($b in @('target/release/qo.exe', 'target/release/qlang.exe')) {
    if (Test-Path $b) {
        $hash = (Get-FileHash $b -Algorithm SHA256).Hash
        $size = (Get-Item $b).Length
        Write-Host "  $b  $size bytes  sha256=$hash" -ForegroundColor Green
    }
}

if (-not $SkipTag) {
    Write-Host "`n[3/3] Tagging v$Version ..." -ForegroundColor Yellow
    Push-Location $root
    try {
        git tag "v$Version"
        if ($LASTEXITCODE -ne 0) { throw "git tag failed (does v$Version already exist?)" }
        git tag -l "v$Version"
    } finally { Pop-Location }
    Write-Host "  tagged v$Version (push later with: git push origin v$Version)" -ForegroundColor Green
} else {
    Write-Host "`n[3/3] Skipping tag (-SkipTag)." -ForegroundColor Yellow
}

Write-Host "`nRELEASE READY: v$Version. Notes: CHANGES.md. Push: git push origin NewWayLLMHandling && git push origin v$Version" -ForegroundColor Green
