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

## Linux

Fuer den Umzug auf ein Linux-System sollte dieses Repo nicht als bereits
gebautes Arbeitsverzeichnis kopiert werden, sondern als sauberer Quellstand
ohne lokale Build-Artefakte. Der grosse Platzverbrauch kommt lokal fast komplett
aus `target/` und optional aus `frontend/node_modules/`.

Empfohlener Linux-Pfad:

- Quellcode ueber Git oder als sauberes Archiv uebertragen
- keine Windows-Buildartefakte mitnehmen
- Rust und Node.js auf dem Zielsystem frisch installieren
- Frontend und Rust-Binaries auf Linux neu bauen
- `.env` separat und sicher auf dem Zielsystem bereitstellen

### Linux-Voraussetzungen

Minimal sinnvoll:

- Rust stable toolchain
- `clang` oder `gcc`
- `llvm-18` und `llvm-config-18`, falls LLVM/JIT-Pfade aktiv gebaut werden
- Node.js 20+ und `npm`
- `pkg-config`
- `libssl-dev`, falls spaeter zusaetzliche Netz-/Native-Abhaengigkeiten dazukommen

Beispiel fuer Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential clang pkg-config llvm-18 llvm-18-dev npm
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

### Was mit soll

Fuer eine saubere Portierung reichen im Regelfall:

- gesamter Quellcode
- `.git/`, wenn die Historie erhalten bleiben soll
- `Cargo.toml`, `Cargo.lock`
- `frontend/package.json` und `frontend/package-lock.json`
- `.env.example`
- die echte `.env` nur separat und sicher uebertragen, nicht oeffentlich einchecken

### Was nicht mit soll

Diese Ordner und Dateien sind Ballast fuer den Umzug und sollten vor dem
Transfer entfernt oder ignoriert werden:

- `target/`
- `frontend/node_modules/`
- temporaere Logs
- lokale Screenshots und Debug-Artefakte, wenn sie nicht bewusst Teil des Projekts sind

### Vor dem Transfer aufraeumen

Rust-Buildartefakte entfernen:

```bash
cargo clean
```

Frontend-Abhaengigkeiten entfernen, wenn nur der Quellstand transportiert
werden soll:

```bash
rm -rf frontend/node_modules
```

Unter Windows PowerShell:

```powershell
cargo clean
Remove-Item -Recurse -Force frontend\node_modules
```

### Neuaufbau auf Linux

Nach dem Transfer auf dem Linux-System:

```bash
cargo build --release --no-default-features
cd frontend
npm ci
npm run build
cd ..
cargo run --bin qo -- --offline
```

Wenn LLVM/JIT benoetigt wird, statt `--no-default-features` den normalen Build
mit passender LLVM-18-Installation verwenden.

### Betriebsnotiz fuer Linux

Fuer einen stabilen Dauerbetrieb auf Linux ist dieser Zielzustand sinnvoll:

- `qo` als `systemd`-Service
- Reverse Proxy vor Port `4646`
- `.env` ausserhalb des Repos, z. B. unter `/etc/orbitqo/`
- Datenverzeichnis persistent halten

### Warum der Ordner lokal so gross wird

Die groessten lokalen Speicherfresser sind:

- `target/debug/deps`
- `target/debug/incremental`
- `target/debug/examples`

Das sind kompilierte Rust-Abhaengigkeiten und Caches fuer schnellere lokale
Builds. Sie sind fuer die Portierung nicht noetig und koennen vor dem Umzug
entfernt werden.

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
