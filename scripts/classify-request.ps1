param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$env:PATH = 'C:\Users\a.b\mingw64\bin;' + $env:PATH
Remove-Item Env:CARGO_HTTP_CAINFO -ErrorAction SilentlyContinue
Remove-Item Env:CARGO_HTTP_PROXY_CAINFO -ErrorAction SilentlyContinue
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
$env:CARGO_HTTP_CHECK_REVOKE = 'false'
$env:CARGO_NET_GIT_FETCH_WITH_CLI = 'true'

& cargo run --bin classify-request --no-default-features --offline -- @ArgsList
