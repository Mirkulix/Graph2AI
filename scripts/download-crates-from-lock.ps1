param(
    [string]$LockFile = "Cargo.lock",
    [string]$OutputDir = ".cargo-offline\crate-cache",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

function Get-CratePackagesFromLock {
    param([string]$Path)

    $lines = Get-Content -LiteralPath $Path
    $packages = @()
    $current = @{}
    $inPackage = $false

    foreach ($line in $lines) {
        if ($line -match '^\[\[package\]\]') {
            if ($inPackage -and $current.name -and $current.version -and $current.source -like 'registry+*') {
                $packages += [pscustomobject]@{
                    Name = $current.name
                    Version = $current.version
                    Source = $current.source
                }
            }
            $current = @{}
            $inPackage = $true
            continue
        }

        if (-not $inPackage) {
            continue
        }

        if ($line -match '^name\s*=\s*"(.+)"$') {
            $current.name = $Matches[1]
            continue
        }

        if ($line -match '^version\s*=\s*"(.+)"$') {
            $current.version = $Matches[1]
            continue
        }

        if ($line -match '^source\s*=\s*"(.+)"$') {
            $current.source = $Matches[1]
            continue
        }
    }

    if ($inPackage -and $current.name -and $current.version -and $current.source -like 'registry+*') {
        $packages += [pscustomobject]@{
            Name = $current.name
            Version = $current.version
            Source = $current.source
        }
    }

    $packages |
        Sort-Object Name, Version -Unique
}

function Get-CrateDownloadUrl {
    param(
        [string]$Name,
        [string]$Version
    )

    "https://static.crates.io/crates/$Name/$Name-$Version.crate"
}

if (-not (Test-Path -LiteralPath $LockFile)) {
    throw "Lock file not found: $LockFile"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$packages = Get-CratePackagesFromLock -Path $LockFile

if (-not $packages -or $packages.Count -eq 0) {
    throw "No registry packages found in $LockFile"
}

Write-Host "Found $($packages.Count) registry packages in $LockFile"
Write-Host "Downloading to $OutputDir"

$downloaded = 0
$skipped = 0
$failed = 0

foreach ($pkg in $packages) {
    $fileName = "$($pkg.Name)-$($pkg.Version).crate"
    $dest = Join-Path $OutputDir $fileName
    $url = Get-CrateDownloadUrl -Name $pkg.Name -Version $pkg.Version

    if ($SkipExisting -and (Test-Path -LiteralPath $dest)) {
        Write-Host "[skip] $fileName"
        $skipped++
        continue
    }

    try {
        Write-Host "[get ] $fileName"
        & curl.exe --ssl-no-revoke -L $url -o $dest --fail --silent --show-error
        if ($LASTEXITCODE -ne 0) {
            throw "curl exit code $LASTEXITCODE"
        }
        $downloaded++
    }
    catch {
        Write-Warning "Failed: $fileName :: $_"
        $failed++
    }
}

Write-Host ""
Write-Host "Done."
Write-Host "Downloaded: $downloaded"
Write-Host "Skipped:    $skipped"
Write-Host "Failed:     $failed"

if ($failed -gt 0) {
    exit 1
}
