@echo off
REM OrbitQO Cockpit — native desktop launcher.
REM
REM Starts the QO server and opens the cockpit in its own app window
REM (no browser chrome). Closing the window stops the server.
REM
REM Release builds on the windows-gnu toolchain need mingw-w64's dlltool on
REM PATH; adding it here keeps the launcher working from a bare double-click.
setlocal

if exist "%USERPROFILE%\mingw64\bin\dlltool.exe" (
    set "PATH=%USERPROFILE%\mingw64\bin;%PATH%"
)

REM Single-machine install: clients on this machine (the cockpit, Claude Code,
REM Codex, …) authenticate as the operator without a token, and the server
REM binds to 127.0.0.1 only so that is not reachable from the network.
REM Remove this line — and use QO_AUTH_TOKEN or an issued seat — to serve a LAN.
if not defined QO_LOCAL_MODE set "QO_LOCAL_MODE=1"

set "ROOT=%~dp0"
set "LAUNCHER=%ROOT%target\release\qo-desktop.exe"

if not exist "%LAUNCHER%" (
    echo Building the desktop launcher ^(first run only^)...
    pushd "%ROOT%"
    cargo build --release --bin qo-desktop --bin qo --no-default-features
    if errorlevel 1 (
        echo.
        echo Build failed. See the output above.
        popd
        pause
        exit /b 1
    )
    popd
)

if not exist "%ROOT%frontend\dist\index.html" (
    echo Building the cockpit frontend ^(first run only^)...
    pushd "%ROOT%frontend"
    call npm run build
    if errorlevel 1 (
        echo.
        echo Frontend build failed. See the output above.
        popd
        pause
        exit /b 1
    )
    popd
)

"%LAUNCHER%" %*
