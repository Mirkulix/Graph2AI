<#
.SYNOPSIS
Server end-to-end demo: start qo against a scratch workspace, then watch the
whole knowledge loop work over MCP - propose, stay out of context while
unverified, get deterministically verified against source, enter context,
prove with a receipt, summarize with health. This is the "watch the product
work" answer to "when is it a finished product": the same steps any agent
takes against POST /mcp/v1.

.EXAMPLE
./scripts/e2e-demo.ps1            # port 4646
./scripts/e2e-demo.ps1 -Port 4747
#>
param([int]$Port = 4646)
$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $PSScriptRoot
$ws = Join-Path $env:TEMP "orbitqo_e2e_ws_$Port"
$data = Join-Path $env:TEMP "orbitqo_e2e_data_$Port"
New-Item -ItemType Directory -Force -Path (Join-Path $ws 'src') | Out-Null
$fixture = "// auth hashes passwords with bcrypt`npub fn hash_password() -> String { bcrypt::hash(`"x`") }`n"
Set-Content -Path (Join-Path $ws 'src/auth.rs') -Value $fixture -Encoding ASCII

$env:QO_PORT = "$Port"
$env:QO_DATA_DIR = $data
$env:QO_WORKSPACE = $ws
$log = Join-Path $env:TEMP "orbitqo_e2e_$Port.log"
$proc = Start-Process -FilePath (Join-Path $root 'target/debug/qo.exe') -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$base = "http://127.0.0.1:$Port"
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { Write-Error "qo exited early - see $log"; break }
    try { Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { }
}
if (-not $ok) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "qo did not become healthy on $base - see $log"
}

$mcp = @{ 'Content-Type' = 'application/json' }
function MCP($id, $name, $toolargs) {
    $body = @{ jsonrpc = '2.0'; id = $id; method = 'tools/call'; params = @{ name = $name; arguments = $toolargs } } | ConvertTo-Json -Depth 6
    (Invoke-RestMethod -Uri "$base/mcp/v1" -Method Post -Headers $mcp -Body $body).result.content[0].text
}

Write-Host "`n== 1. Agent proposes a finding (orbit_graph_propose) ==" -ForegroundColor Cyan
$doc = "DELTA|1|d-1`nBY|worker-3|1700000000`n+E|file|src/auth.rs`n+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt`n"
Write-Host (MCP 1 'orbit_graph_propose' @{ document = $doc })

Write-Host "`n== 2. Unverified proposals never reach context ==" -ForegroundColor Cyan
$ctx = MCP 2 'orbit_graph_context' @{ kind = 'file'; name = 'src/auth.rs' }
if ($ctx -match 'auth hashes') { Write-Host "  FAIL: unverified claim leaked into context" } else { Write-Host "  context is empty - correct (proposal is not fact)" }

Write-Host "`n== 3. Deterministic source verification (orbit_graph_verify_source) ==" -ForegroundColor Cyan
Write-Host (MCP 3 'orbit_graph_verify_source' @{ id = 'c1'; by = 'checker' })

Write-Host "`n== 4. Now the context carries it ==" -ForegroundColor Cyan
Write-Host (MCP 4 'orbit_graph_context' @{ kind = 'file'; name = 'src/auth.rs' })

Write-Host "`n== 5. Proof: the receipt (whole trail) ==" -ForegroundColor Cyan
Write-Host (MCP 5 'orbit_graph_receipt' @{ id = 'c1' })

Write-Host "`n== 6. Operator summary ==" -ForegroundColor Cyan
Write-Host (MCP 6 'orbit_graph_health' @{})

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $ws, $data -ErrorAction SilentlyContinue
Remove-Item -Force $log, "$log.err" -ErrorAction SilentlyContinue
Write-Host "`nDemo done (server stopped, scratch cleaned)." -ForegroundColor Green
