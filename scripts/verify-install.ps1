# Verify the local OrbitQLang install end to end.
#
# Answers one question with evidence: "does this actually work on my machine?"
# Every check prints what it did and what came back, so a failure names the
# thing to fix rather than just saying no.
#
# Read-only against the graph: it exercises reads and a handshake, and never
# writes a claim, so running it against a live instance changes nothing.
#
#   .\scripts\verify-install.ps1              # against the default port
#   .\scripts\verify-install.ps1 -Port 5353   # against another instance
#
# Each check is a function: PowerShell 5.1 mis-parses try/catch blocks split
# across comment banners, so structure carries the layout instead of banners.

[CmdletBinding()]
param(
    [int]$Port = $(if ($env:QO_PORT) { [int]$env:QO_PORT } else { 4646 }),
    [int]$StartupTimeout = 90
)

$ErrorActionPreference = 'Stop'
$script:Base     = "http://127.0.0.1:$Port"
$script:Failures = @()
$script:Checks   = 0
$script:Skip     = $false
$script:Started  = $null

function Step($text) { Write-Host "`n$text" -ForegroundColor Cyan }
function Info($text) { Write-Host "         $text" -ForegroundColor DarkGray }
function Pass($text) {
    $script:Checks++
    Write-Host "  [ok]   $text" -ForegroundColor Green
}
function Fail($text, $fix) {
    $script:Checks++
    $script:Failures += [pscustomobject]@{ What = $text; Fix = $fix }
    Write-Host "  [FAIL] $text" -ForegroundColor Red
    if ($fix) { Write-Host "         -> $fix" -ForegroundColor Yellow }
}

# Local mode means no token is needed from this machine. If the operator has
# one configured anyway, send it, so the check works in both setups.
$script:Headers = @{}
if ($env:QO_AUTH_TOKEN) { $script:Headers['Authorization'] = "Bearer $env:QO_AUTH_TOKEN" }

function Invoke-Qo {
    param($Path, $Method = 'GET', $Body = $null)
    $splat = @{
        Uri             = "$script:Base$Path"
        Method          = $Method
        Headers         = $script:Headers
        TimeoutSec      = 20
        UseBasicParsing = $true
    }
    if ($Body) {
        $splat['Body'] = ($Body | ConvertTo-Json -Depth 10 -Compress)
        $splat['ContentType'] = 'application/json'
    }
    Invoke-RestMethod @splat
}

function Get-HttpStatus($ErrorRecord) {
    if ($ErrorRecord.Exception.Response) {
        return $ErrorRecord.Exception.Response.StatusCode.value__
    }
    return 0
}

function Test-ServerUp {
    Step "1/6  Is the server up?"
    try {
        Invoke-Qo '/api/health' | Out-Null
        Pass "server is answering on port $Port"
        return $true
    } catch {
        if ((Get-HttpStatus $_) -eq 401) {
            Fail "server is up but refused the request (401)" "Start it with QO_LOCAL_MODE=1 (start-cockpit.cmd sets this), or set QO_AUTH_TOKEN."
            $script:Skip = $true
            return $true
        }
        Info "nothing is listening - starting one for this check"
        return Start-CheckServer
    }
}

function Start-CheckServer {
    $exe = Join-Path $PSScriptRoot '..\target\release\qo.exe'
    if (-not (Test-Path $exe)) {
        Fail "no server, and target\release\qo.exe is not built" "Run: cargo build --release --bin qo --no-default-features"
        $script:Skip = $true
        return $false
    }
    # A qo from an earlier run keeps the redb lock even when it listens on a
    # different port, so the new process dies instantly with "Database already
    # open". Name that case up front — it is the most common startup failure
    # and the least obvious from a timeout alone.
    $stale = @(Get-Process qo -ErrorAction SilentlyContinue)
    if ($stale.Count -gt 0) {
        Fail "another qo process is already running (pid $($stale[0].Id))" "It holds the database lock. Stop it first: Stop-Process -Id $($stale[0].Id)"
        $script:Skip = $true
        return $false
    }

    $env:QO_LOCAL_MODE = '1'
    $env:QO_PORT = "$Port"
    $script:Started = Start-Process -FilePath $exe -ArgumentList '--offline' -PassThru -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($StartupTimeout)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        # A child that already exited will never answer; say why instead of
        # waiting out the full timeout.
        if ($script:Started.HasExited) {
            Fail "the server exited immediately (code $($script:Started.ExitCode))" "Usually a held database lock or an occupied port. Run it in a console to see the error: .\target\release\qo.exe --offline"
            $script:Skip = $true
            return $false
        }
        try {
            Invoke-Qo '/api/health' | Out-Null
            Pass "started a server for this check (pid $($script:Started.Id))"
            return $false
        } catch { }
    }
    Fail "server did not come up within ${StartupTimeout}s" "Run it in a console to see the error: .\target\release\qo.exe --offline"
    $script:Skip = $true
    return $false
}

# The check that would have caught the original failure: the plugin got 401
# because seats existed, and surfaced it as a JSON parse error.
function Test-Handshake {
    Step "2/6  Can an MCP client attach without a token?"
    try {
        $init = Invoke-Qo '/mcp/v1' 'POST' @{
            jsonrpc = '2.0'
            id      = 1
            method  = 'initialize'
            params  = @{ clientInfo = @{ name = 'install-check'; version = '1.0' } }
        }
        if ($init.result) {
            Pass "MCP handshake accepted (server: $($init.result.serverInfo.name))"
        } else {
            Fail "handshake returned no result" ($init | ConvertTo-Json -Compress)
        }
    } catch {
        if ((Get-HttpStatus $_) -eq 401) {
            Fail "MCP endpoint refused an unauthenticated local client (401)" "This is exactly what breaks the plugin. Start the server with QO_LOCAL_MODE=1."
        } else {
            Fail "MCP handshake failed" $_.Exception.Message
        }
    }
}

function Test-ToolsList {
    Step "3/6  Are the tools actually exposed?"
    try {
        $list = Invoke-Qo '/mcp/v1' 'POST' @{ jsonrpc = '2.0'; id = 2; method = 'tools/list' }
        $tools = @($list.result.tools)
        if ($tools.Count -gt 0) {
            Pass "$($tools.Count) tools exposed"
            $graph = @($tools | Where-Object { $_.name -like 'orbit_graph_*' })
            Info "knowledge-graph tools: $($graph.Count)"
        } else {
            Fail "tools/list returned nothing" "The server is up but exposes no tools - likely a build mismatch."
        }
    } catch {
        Fail "tools/list failed" $_.Exception.Message
    }
}

# A read-only call, so the check never writes to the graph.
function Test-ToolCall {
    Step "4/6  Does a real tool call work?"
    try {
        $call = Invoke-Qo '/mcp/v1' 'POST' @{
            jsonrpc = '2.0'
            id      = 3
            method  = 'tools/call'
            params  = @{
                name       = 'orbit_graph_health'
                arguments  = @{}
                clientInfo = @{ name = 'install-check' }
            }
        }
        if ($call.result) {
            Pass "orbit_graph_health returned a result"
        } else {
            Fail "tool call returned an error" ($call.error | ConvertTo-Json -Compress)
        }
    } catch {
        Fail "tool call failed" $_.Exception.Message
    }
}

# Proves the cockpit's Integrations view is fed by real traffic: this script
# just handshook and called a tool, so it must appear.
function Test-HarnessRegistry {
    Step "5/6  Did the harness registry notice us?"
    try {
        $h = Invoke-Qo '/api/harness'
        $me = @($h.sessions | Where-Object { $_.id -eq 'install-check' })
        if ($me.Count -gt 0) {
            Pass "this check appears as an attached client ($($me[0].calls) call(s) recorded)"
            Info "open the cockpit -> Integrations to see it listed"
        } else {
            Fail "the registry did not record this session" "Expected 'install-check' in /api/harness."
        }
    } catch {
        Fail "/api/harness failed" $_.Exception.Message
    }
}

function Test-PluginConfig {
    Step "6/6  Is the plugin pointed at this server?"
    $mcpConfig = Join-Path $PSScriptRoot '..\plugins\orbitqlang-claude\.mcp.json'
    if (-not (Test-Path $mcpConfig)) {
        Fail "plugin config not found" "Expected $mcpConfig"
        return
    }
    $url = (Get-Content $mcpConfig -Raw | ConvertFrom-Json).mcpServers.orbitqlang.url
    if ($url -match ":$Port/") {
        Pass "plugin points at this server ($url)"
    } else {
        Fail "plugin points at $url, but this check ran against $script:Base" "Start the server on the plugin's port, or edit .mcp.json."
    }
}

Write-Host "OrbitQLang install check -> $script:Base" -ForegroundColor White

$serverWasRunning = Test-ServerUp

if (-not $script:Skip) {
    Test-Handshake
    Test-ToolsList
    Test-ToolCall
    Test-HarnessRegistry
    Test-PluginConfig
}

# Only stop a server this script started; one that was already running belongs
# to whoever launched it.
if ($script:Started -and -not $serverWasRunning) {
    Write-Host "`nStopping the server this check started..." -ForegroundColor DarkGray
    Stop-Process -Id $script:Started.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($script:Failures.Count -eq 0) {
    Write-Host "All $script:Checks checks passed - the install works." -ForegroundColor Green
    Write-Host "Restart Claude Code to pick up the plugin, then ask it to call orbit_graph_health." -ForegroundColor DarkGray
    exit 0
}

Write-Host "$($script:Failures.Count) of $script:Checks checks failed:" -ForegroundColor Red
foreach ($f in $script:Failures) {
    Write-Host "  - $($f.What)" -ForegroundColor Red
    if ($f.Fix) { Write-Host "    -> $($f.Fix)" -ForegroundColor Yellow }
}
exit 1
