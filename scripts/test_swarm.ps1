# QLMS Swarm Test: Send a SIGNED QLMS v2 frame to the qo-server.
# Usage: powershell -File scripts/test_swarm.ps1

$Port = 4646
$Url = "http://localhost:$Port/qlms/v1.1/deliver"

Write-Host "Generating signed QLMS v2 frame using Rust backend..." -ForegroundColor Cyan

# 1. Generate the signed JSON into a temporary file to avoid pipe noise
$TmpFile = Join-Path $env:TEMP "signed_qlms_req.json"

$ProjectRoot = Get-Location
Push-Location "$ProjectRoot\qo\qo-server"
try {
    # Use cmd /c to handle redirection cleanly
    cmd /c "cargo run --example gen_json --quiet > $TmpFile"
    if (-not (Test-Path $TmpFile) -or (Get-Item $TmpFile).Length -eq 0) {
        throw "Failed to generate signed JSON. Check if 'cargo run --example gen_json' works manually."
    }
} finally {
    Pop-Location
}

# 2. Read and deliver
Write-Host "Sending SIGNED QLMS frame to $Url..." -ForegroundColor Cyan
try {
    # InFile is the most robust way to send JSON from a file
    $Response = Invoke-RestMethod -Uri $Url -Method Post -InFile $TmpFile -ContentType "application/json"
    
    Write-Host "Success! Message Verified. ✅" -ForegroundColor Green
    $Response | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Failed to deliver message." -ForegroundColor Red
    if ($_.Exception.Response) {
        $Reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $ErrorBody = $Reader.ReadToEnd()
        Write-Host "Server Error: $ErrorBody" -ForegroundColor Yellow
    } else {
        Write-Host $_.Exception.Message
    }
} finally {
    if (Test-Path $TmpFile) { Remove-Item $TmpFile }
}
