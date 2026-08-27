<#
.SYNOPSIS
One-command setup for OrbitQO: build the server + CLI, ensure the .qlang
config, start `qo`, and PROVE an agent can reach it over MCP. Windows /
PowerShell. Linux/macOS: `./scripts/install.sh && ./scripts/start-qo.sh`.

This is the "is there a setup?" answer: after this script, any MCP client
(Claude Code, the DeepSeek-harness plugin surface, a script) can talk to the
knowledge graph at POST /mcp/v1.

.EXAMPLE
./scripts/setup.ps1                 # build + start on :4646 + MCP self-test
./scripts/setup.ps1 -Port 4747      # different port
./scripts/setup.ps1 -SkipBuild      # reuse an existing build
./scripts/setup.ps1 -WithCockpit    # also build the React cockpit (needs Node)
#>
param(
    [int]$Port = 4646,
    [switch]$SkipBuild,
    [switch]$WithCockpit
)
# 'Continue', not 'Stop': native commands (cargo, npm) write progress to
# stderr, and 'Stop' would abort on that. Real failures are caught by the
# explicit $LASTEXITCODE checks and throws below.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot

Write-Host "== OrbitQO setup ==" -ForegroundColor Cyan

# ---- 1. Build ----
if (-not $SkipBuild) {
    Write-Host "`n[1/4] Building qo + qlang (--no-default-features, JIT off)..." -ForegroundColor Yellow
    Push-Location $root
    try {
        cargo build --bin qo --bin qlang --no-default-features
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
    } finally { Pop-Location }
} else {
    Write-Host "`n[1/4] Skipping build (-SkipBuild)." -ForegroundColor Yellow
}

# ---- 2. .qlang config (never overwrite existing; no secrets created) ----
Write-Host "`n[2/4] Ensuring .qlang config..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Join-Path $root '.qlang') | Out-Null
$trust = Join-Path $root '.qlang/trusted_delta_producers.json'
if (-not (Test-Path $trust)) {
    Copy-Item (Join-Path $root '.qlang/trusted_delta_producers.example.json') $trust
    Write-Host "  created $trust (empty trust store - nobody may submit signed deltas yet)"
} else {
    Write-Host "  kept existing $trust"
}
# No api_keys.json is created: a fresh local instance stays unauthenticated
# (loopback-bound) until the operator issues seats with `qlang keys issue`.

# ---- 3. Optional cockpit ----
if ($WithCockpit) {
    Write-Host "`n[3/4] Building cockpit (npm run build)..." -ForegroundColor Yellow
    Push-Location (Join-Path $root 'frontend')
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
    } finally { Pop-Location }
} else {
    Write-Host "`n[3/4] Skipping cockpit build (use -WithCockpit to build the UI)." -ForegroundColor Yellow
}

# ---- 4. Start + verify + MCP self-test ----
Write-Host "`n[4/4] Starting qo on port $Port ..." -ForegroundColor Yellow
$env:QO_PORT = "$Port"
$env:QO_DATA_DIR = Join-Path $env:TEMP "orbitqo_data_$Port"
New-Item -ItemType Directory -Force -Path $env:QO_DATA_DIR | Out-Null
$log = Join-Path $env:TEMP "orbitqo_$Port.log"
$bin = Join-Path $root 'target/debug/qo.exe'
if (-not (Test-Path $bin)) { throw "missing $bin - build first or drop -SkipBuild" }
$proc = Start-Process -FilePath $bin -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$base = "http://127.0.0.1:$Port"
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { throw "qo exited early (code $($proc.ExitCode)) - see $log / $log.err" }
    try { Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { }
}
if (-not $ok) { throw "qo did not become healthy on $base - see $log" }
Write-Host "  qo is up: $base (PID $($proc.Id), data in $env:QO_DATA_DIR)" -ForegroundColor Green

Write-Host "`nMCP self-test (POST $base/mcp/v1):" -ForegroundColor Yellow
$mcp = @{ 'Content-Type' = 'application/json' }
$list = @{ jsonrpc = '2.0'; id = 1; method = 'tools/list'; params = @{} } | ConvertTo-Json
$tools = Invoke-RestMethod -Uri "$base/mcp/v1" -Method Post -Headers $mcp -Body $list
Write-Host "  tools/list            -> $($tools.result.tools.Count) tools"
$call = @{ jsonrpc = '2.0'; id = 2; method = 'tools/call'; params = @{ name = 'orbit_graph_health'; arguments = @{} } } | ConvertTo-Json -Depth 5
$health = Invoke-RestMethod -Uri "$base/mcp/v1" -Method Post -Headers $mcp -Body $call
Write-Host "  orbit_graph_health    -> $($health.result.content[0].text)"

Write-Host "`nREADY. Connect any MCP client / agent to:" -ForegroundColor Green
Write-Host "  MCP endpoint : $base/mcp/v1"
Write-Host "  API base     : $base"
Write-Host "  Cockpit      : $base/ (when built with -WithCockpit)"
Write-Host "  CLI          : QO_PORT=$Port qlang graph health"
Write-Host "  Stop server  : Stop-Process -Id $($proc.Id)"
Write-Host "  First steps  : qlang keys issue --label me --role admin  (then set seats)"
