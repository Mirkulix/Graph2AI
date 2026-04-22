#requires -Version 5.1
# OrbitQLang one-shot installer (Windows PowerShell 5.1 compatible).
#
# Usage:
#   pwsh -File scripts\install-qlang-ide.ps1                  (release build, install in all IDEs)
#   pwsh -File scripts\install-qlang-ide.ps1 -DebugBinaries   (faster cargo build)
#   pwsh -File scripts\install-qlang-ide.ps1 -SkipBuild       (just reinstall existing VSIX)
#   pwsh -File scripts\install-qlang-ide.ps1 -NoInstall       (build only, no IDE install)
#   pwsh -File scripts\install-qlang-ide.ps1 -StartServer     (also launch qo in new window)

param(
    [switch]$DebugBinaries,
    [switch]$SkipBuild,
    [switch]$StartServer,
    [switch]$NoInstall
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step { param([string]$m); Write-Host ''; Write-Host ('==> ' + $m) -ForegroundColor Cyan }
function Write-Ok   { param([string]$m); Write-Host ('    [OK]   ' + $m) -ForegroundColor Green }
function Write-Warn { param([string]$m); Write-Host ('    [WARN] ' + $m) -ForegroundColor Yellow }
function Write-Err  { param([string]$m); Write-Host ('    [ERR]  ' + $m) -ForegroundColor Red }

function Get-CmdSource {
    param([string]$Name)
    $c = Get-Command $Name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source } else { return $null }
}

function Resolve-First {
    param([string[]]$Candidates, [string]$NotFoundMessage)
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    throw $NotFoundMessage
}

function Resolve-Cargo {
    return Resolve-First @(
        (Get-CmdSource 'cargo'),
        ($env:USERPROFILE + '\.cargo\bin\cargo.exe'),
        'C:\Users\a.b\.cargo\bin\cargo.exe'
    ) 'cargo.exe not found. Install Rust from rustup.rs first.'
}

function Resolve-Node {
    return Resolve-First @(
        (Get-CmdSource 'node'),
        'C:\Program Files\nodejs\node.exe',
        ($env:USERPROFILE + '\pprog\node\node.exe'),
        'C:\Users\a.b\pprog\node\node.exe'
    ) 'node.exe not found. Install Node 18 or newer from nodejs.org first.'
}

function Resolve-Npx {
    $node = Resolve-Node
    $nodeDir = Split-Path $node -Parent
    return Resolve-First @(
        (Get-CmdSource 'npx.cmd'),
        (Join-Path $nodeDir 'npx.cmd')
    ) 'npx.cmd not found alongside node.exe.'
}

function Get-RepoRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir '..')).Path
}

# ---------------------------------------------------------------------------
# IDE detection
# ---------------------------------------------------------------------------

function Find-InstalledIdes {
    $local = $env:LOCALAPPDATA
    $catalog = @(
        @{ Name = 'VS Code';         Candidates = @( (Get-CmdSource 'code'),          'C:\Program Files\Microsoft VS Code\bin\code.cmd',                ($local + '\Programs\Microsoft VS Code\bin\code.cmd') ) },
        @{ Name = 'VS Code Insiders';Candidates = @( (Get-CmdSource 'code-insiders'), 'C:\Program Files\Microsoft VS Code Insiders\bin\code-insiders.cmd' ) },
        @{ Name = 'Cursor';          Candidates = @( (Get-CmdSource 'cursor'),        ($local + '\Programs\cursor\resources\app\bin\cursor.cmd') ) },
        @{ Name = 'Windsurf';        Candidates = @( (Get-CmdSource 'windsurf'),      ($local + '\Programs\Windsurf\bin\windsurf.cmd') ) },
        @{ Name = 'Kiro';            Candidates = @( (Get-CmdSource 'kiro'),          ($local + '\Programs\Kiro\bin\kiro.cmd') ) },
        @{ Name = 'Antigravity';     Candidates = @( (Get-CmdSource 'antigravity'),   ($local + '\Programs\Antigravity\bin\antigravity.cmd') ) },
        @{ Name = 'Trae';            Candidates = @( (Get-CmdSource 'trae'),          ($local + '\Programs\Trae\bin\trae.cmd'), ($local + '\Programs\trae\bin\trae.cmd') ) },
        @{ Name = 'VSCodium';        Candidates = @( (Get-CmdSource 'codium'),        'C:\Program Files\VSCodium\bin\codium.cmd' ) }
    )

    $found = @()
    foreach ($ide in $catalog) {
        foreach ($cand in $ide.Candidates) {
            if ($cand -and (Test-Path $cand)) {
                $found += [pscustomobject]@{ Name = $ide.Name; Cli = $cand }
                break
            }
        }
    }
    return $found
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

$repoRoot = Get-RepoRoot
$cargoBin = $env:USERPROFILE + '\.cargo\bin'
$extDir   = Join-Path $repoRoot 'editors\vscode'

Write-Host ''
Write-Host 'OrbitQLang one-shot installer' -ForegroundColor Magenta
Write-Host ('Repo: ' + $repoRoot)
Write-Host ''

# --- Build binaries ---------------------------------------------------------
if (-not $SkipBuild) {
    Write-Step 'Building Rust binaries (qlang-cli and qo)'
    $cargo = Resolve-Cargo
    $profileName = if ($DebugBinaries) { 'debug' } else { 'release' }
    $profileArgs = if ($DebugBinaries) { @() } else { @('-r') }

    Push-Location $repoRoot
    try {
        $buildArgs = @('build') + $profileArgs + @('--bin','qlang-cli','--bin','qo')
        & $cargo @buildArgs 2>&1 | ForEach-Object { Write-Host ('    ' + $_) }
        if ($LASTEXITCODE -ne 0) { throw ('cargo build failed (exit=' + $LASTEXITCODE + ')') }
        Write-Ok ('Built qlang-cli and qo (' + $profileName + ')')
    } finally {
        Pop-Location
    }

    Write-Step ('Installing binaries into ' + $cargoBin)
    if (-not (Test-Path $cargoBin)) {
        New-Item -ItemType Directory -Path $cargoBin -Force | Out-Null
    }
    foreach ($bin in @('qlang-cli.exe','qo.exe')) {
        $src = Join-Path $repoRoot ('target\' + $profileName + '\' + $bin)
        if (-not (Test-Path $src)) {
            Write-Err ('Built binary not found: ' + $src)
            continue
        }
        $dst = Join-Path $cargoBin $bin
        Copy-Item $src $dst -Force
        Write-Ok ($bin + ' -> ' + $dst)
    }
} else {
    Write-Warn 'Skipping cargo build [SkipBuild flag]'
}

# --- Build VSIX -------------------------------------------------------------
$vsixPath = Join-Path $extDir 'qlang-0.2.0.vsix'

if (-not $SkipBuild) {
    Write-Step 'Compiling extension TypeScript'
    $node = Resolve-Node
    $tscPath = Join-Path $extDir 'node_modules\typescript\bin\tsc'
    if (-not (Test-Path $tscPath)) {
        Write-Step 'Installing extension npm dependencies (one-time)'
        Push-Location $extDir
        try {
            & 'npm.cmd' install 2>&1 | ForEach-Object { Write-Host ('    ' + $_) }
        } finally {
            Pop-Location
        }
    }

    & $node $tscPath '-p' (Join-Path $extDir 'tsconfig.json') 2>&1 | ForEach-Object { Write-Host ('    ' + $_) }
    if ($LASTEXITCODE -ne 0) { throw ('tsc failed (exit=' + $LASTEXITCODE + ')') }
    Write-Ok 'TypeScript compiled to out/'

    Write-Step 'Packaging .vsix'
    if (Test-Path $vsixPath) { Remove-Item $vsixPath -Force }
    $npx = Resolve-Npx
    Push-Location $extDir
    try {
        $vsceArgs = @('-y','@vscode/vsce','package','--allow-missing-repository','--allow-star-activation')
        & $npx @vsceArgs 2>&1 | Select-Object -Last 5 | ForEach-Object { Write-Host ('    ' + $_) }
        if ($LASTEXITCODE -ne 0) { throw ('vsce package failed (exit=' + $LASTEXITCODE + ')') }
    } finally {
        Pop-Location
    }
    if (-not (Test-Path $vsixPath)) { throw ('VSIX not produced at ' + $vsixPath) }
    $vsixSize = [math]::Round((Get-Item $vsixPath).Length / 1MB, 2)
    Write-Ok ('Packaged ' + $vsixPath + ' (' + $vsixSize + ' MB)')
}

if (-not (Test-Path $vsixPath)) {
    Write-Err ('No VSIX at ' + $vsixPath + '. Run without -SkipBuild first.')
    exit 1
}

# --- Detect + install IDEs --------------------------------------------------
Write-Step 'Detecting installed VS Code-family IDEs'
$ides = Find-InstalledIdes
if ($ides.Count -eq 0) {
    Write-Warn ('No VS Code-family IDE detected. VSIX is at ' + $vsixPath + ' (install manually).')
} else {
    Write-Ok ('Found ' + $ides.Count + ' IDE(s):')
    foreach ($ide in $ides) {
        Write-Host ('       - ' + $ide.Name + ' (' + $ide.Cli + ')')
    }
    if (-not $NoInstall) {
        Write-Host ''
        # Local prefs: VSCode CLIs spam stderr warnings (DEP, TLS) that PS 5.1 promotes
        # to NativeCommandError. Switch to Continue and verify success via exit code +
        # post-install --list-extensions probe.
        $savedPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            foreach ($ide in $ides) {
                Write-Host ('    -> Installing into ' + $ide.Name)
                # Stderr to $null so warnings don't trip our reporting; trust the exit code.
                & $ide.Cli '--install-extension' $vsixPath '--force' 2>$null
                $installCode = $LASTEXITCODE
                # Verify: ask the IDE to list extensions and look for qlang.
                $listed = & $ide.Cli '--list-extensions' '--show-versions' 2>$null
                $hit = $listed | Where-Object { $_ -match 'qlang' } | Select-Object -First 1
                if ($hit) {
                    Write-Ok ('Installed in ' + $ide.Name + ' [' + $hit + ']')
                } elseif ($installCode -eq 0) {
                    Write-Warn ($ide.Name + ' install reported success but qlang not in --list-extensions')
                } else {
                    Write-Err ($ide.Name + ' install failed (exit=' + $installCode + ')')
                }
            }
        } finally {
            $ErrorActionPreference = $savedPref
        }
    } else {
        Write-Warn 'Skipping IDE install [NoInstall flag]'
    }
}

# --- Optional: start QO server ---------------------------------------------
if ($StartServer) {
    Write-Step 'Launching QO backend in a new window'
    $qoPath = Join-Path $cargoBin 'qo.exe'
    if (-not (Test-Path $qoPath)) {
        Write-Err ('qo.exe not found in ' + $cargoBin + '. Run without -SkipBuild first.')
    } else {
        $startCmd = '& ' + ([char]34) + $qoPath + ([char]34) + ' ' + ([char]45) + ([char]45) + 'offline'
        Start-Process -FilePath 'powershell.exe' `
            -ArgumentList '-NoExit','-Command',$startCmd `
            -WindowStyle Normal
        Write-Ok 'Launched QO backend window. Cockpit at localhost on port 4646 path /supervisor'
    }
}

Write-Host ''
Write-Host 'Done.' -ForegroundColor Magenta
Write-Host ''
Write-Host 'Next steps:'
Write-Host '  1. Reload your IDE window so the extension picks up the changes'
Write-Host '  2. Start the QO backend if not done: qo (with offline flag)'
Write-Host '  3. Open the cockpit in your browser at localhost port 4646 path /supervisor'
Write-Host '  4. Try a handover via Ctrl+Shift+P then QLMS Handover Graph to Agent'
Write-Host ''
