# Build

## Windows

Der verifizierte Windows-Pfad fuer dieses Repo ist:

- Rust-Host-Toolchain: `stable-x86_64-pc-windows-gnu`
- MinGW-Linker: `C:\msys64\mingw64\bin\gcc.exe`
- LLVM: `18.1.8`
- `LLVM_SYS_180_PREFIX=C:\msys64\mingw64`

Damit laufen:

- `cargo build --offline`
- `cargo test --no-run --offline`
- `cargo test --workspace --lib --offline`

Hinweis:
- `inkwell 0.5` ist hier auf `llvm18-0` verdrahtet. LLVM 22 reicht nicht; es muss LLVM 18 sein.
- Der Build wurde mit MSYS2 `mingw64` + GNU-Toolchain stabilisiert, nicht mit MSVC.

## Setup

Das Repo enthaelt ein Setup-Skript:

```powershell
.\scripts\setup-build-env.ps1
```

Optional:

```powershell
.\scripts\setup-build-env.ps1 -Session
.\scripts\setup-build-env.ps1 -Verify
```

Das Skript:

- setzt die Rust-GNU-Toolchain
- setzt `LLVM_SYS_180_PREFIX`
- erweitert den `PATH` um `C:\msys64\mingw64\bin` und `C:\Users\a.b\mingw64\bin`
- prueft optional `llvm-config --version`
- fuehrt optional `cargo check --offline` aus

## Cargo-Linker

Die funktionierende Zielkonfiguration liegt in `%USERPROFILE%\.cargo\config.toml` und verwendet:

```toml
[target.x86_64-pc-windows-gnu]
linker = "C:\\msys64\\mingw64\\bin\\gcc.exe"
rustflags = ["-C", "link-arg=-lffi", "-C", "link-arg=-Wl,--allow-multiple-definition"]
```

Das behebt in dieser Umgebung:

- `undefined reference to ffi_*`
- gemischte MinGW-/pthread-Linker-Konflikte

## Offline / Proxy

Der Workspace wurde in einer Proxy-/Sophos-Umgebung stabilisiert.

Wichtige Punkte:

- Cargo-Cache wurde offline vorbereitet
- MSYS2 bekam ein funktionierendes CA-Bundle
- Builds laufen danach mit `--offline`

Wenn Cargo auf einem neuen Rechner keine Crates laden kann:

1. erst `.\scripts\setup-build-env.ps1` ausfuehren
2. dann den Cargo-Cache / das Vendor-Setup pruefen
3. Builds bevorzugt mit `--offline` starten

## Nuetzliche Befehle

Default-Build mit LLVM:

```powershell
cargo build --offline
```

Nur kompilieren, keine Tests ausfuehren:

```powershell
cargo test --no-run --offline
```

Workspace-Library-Tests:

```powershell
cargo test --workspace --lib --offline
```

Hybrid-CLI:

```powershell
cargo run --bin classify-request --no-default-features --offline -- "Please inspect this Rust panic and tell me which file to patch first"
```

## Bekannte offene Punkte

- In `qlang-agent` gibt es aktuell noch 2 Serialization-Test-Failures, die nicht durch LLVM verursacht sind.
- Einige Examples wurden bisher nur ueber Cargo-Autodiscovery gefunden; relevante Beispiele sind jetzt zusaetzlich explizit in `Cargo.toml` eingetragen.
