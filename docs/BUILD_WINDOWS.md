# Build- und Kompilierungsumgebung von Qlang auf einer Windows-Entwicklungsmaschine

## Abstract

Dieses Dokument beschreibt die funktionierende Kompilierungsumgebung des Projekts `qlang` auf einer Windows-Maschine. Ziel ist es, die tatsächlich verwendeten Build-Einstellungen, Toolchains, Linker-Optionen und LLVM-Abhängigkeiten so zu dokumentieren, dass die Umgebung reproduzierbar auf anderen Systemen nachgebaut werden kann. Die vorliegende Konfiguration basiert auf einer `GNU`-basierten Rust-Toolchain unter Windows, kombiniert mit `MSYS2/MinGW`, einem externen `LLVM 18` für `llvm-sys` bzw. `inkwell` sowie einer angepassten Cargo-Konfiguration für TLS/Proxy-Umgebungen.

## 1. Zielsetzung

Das Projekt `qlang` benötigt für den vollständigen Build nicht nur eine funktionierende Rust-Installation, sondern zusätzlich eine kompatible externe LLVM-Umgebung. Besonders relevant ist dies für Komponenten, die über `llvm-sys` und `inkwell` auf LLVM zugreifen. Auf der untersuchten Maschine wurde daher eine Build-Konfiguration etabliert, die folgende Anforderungen erfüllt:

1. Nutzung einer stabilen Rust-Toolchain unter Windows.
2. Vermeidung des problematischen `MSVC`-Pfades zugunsten einer `GNU`-basierten Toolchain.
3. Konsistente Verwendung von `MSYS2/MinGW` als Linker- und Toolchain-Basis.
4. Bereitstellung eines externen `LLVM 18`, das mit der verwendeten Rust-Bibliothekslandschaft kompatibel ist.
5. Anpassung der Cargo-Netzwerk- und TLS-Konfiguration an die lokale Unternehmens- bzw. Proxy-Umgebung.

## 2. Systemkontext

Die untersuchte Entwicklungsumgebung basiert auf den folgenden Rahmenbedingungen:

- Betriebssystem: Windows
- Shell: PowerShell
- Arbeitsverzeichnis: `C:\Users\a.b\Graph\qlang`

Die Build-Umgebung ist ausdrücklich auf Rust `GNU` unter Windows ausgelegt und nicht auf `MSVC`.

## 3. Aktive Rust-Toolchain

Die auf der Maschine aktiv verwendete Rust-Toolchain ist:

- Toolchain: `stable-x86_64-pc-windows-gnu`
- Host: `x86_64-pc-windows-gnu`
- Installiertes Ziel: `x86_64-pc-windows-gnu`

Die verwendete Rust-Version ist:

- `rustc 1.94.1 (e408947bf 2026-03-25)`

Zusätzlich meldet `rustc -vV` die interne LLVM-Version von Rust:

- internes Rust-LLVM: `21.1.8`

Wichtig ist hier die Unterscheidung zwischen dem internen LLVM von Rust und dem extern eingebundenen LLVM für das Projekt. Das interne LLVM von Rust wird für die Übersetzung durch den Rust-Compiler selbst verwendet. Für `qlang` ist daneben ein separates LLVM relevant, das über `llvm-sys` bzw. `inkwell` eingebunden wird.

## 4. Cargo-Konfiguration

Die globale Cargo-Konfiguration befindet sich unter:

- `C:\Users\a.b\.cargo\config.toml`

Die Konfiguration lautet:

```toml
[http]
cainfo = "C:/Users/a.b/.cargo/cacert-with-sophos.pem"
proxy-cainfo = "C:/Users/a.b/.cargo/cacert-with-sophos.pem"
check-revoke = false
ssl-version = "tlsv1.2"
multiplexing = false

[net]
git-fetch-with-cli = true

[registries.crates-io]
protocol = "git"

[target.x86_64-pc-windows-gnu]
linker = "C:\\msys64\\mingw64\\bin\\gcc.exe"
rustflags = ["-C", "link-arg=-lffi", "-C", "link-arg=-Wl,--allow-multiple-definition"]
```

### 4.1 Bedeutung der HTTP- und TLS-Einstellungen

Die Parameter im Abschnitt `[http]` sind an eine Umgebung angepasst, in der TLS-Verbindungen über ein lokales oder unternehmensspezifisches Zertifikatsbundle abgesichert werden. Dabei gilt:

- `cainfo` und `proxy-cainfo` verweisen auf ein benutzerdefiniertes CA-Bundle.
- `check-revoke = false` deaktiviert Zertifikats-Widerrufsprüfungen.
- `ssl-version = "tlsv1.2"` erzwingt TLS 1.2.
- `multiplexing = false` reduziert potenzielle Probleme in restriktiven Netzwerken.

### 4.2 Netzwerkverhalten von Cargo

Im Abschnitt `[net]` ist gesetzt:

- `git-fetch-with-cli = true`

Damit verwendet Cargo für Git-basierte Zugriffe die lokale Git-CLI statt der eingebauten Bibliotheksimplementierung. Dies ist oft robuster in restriktiven Netzwerkumgebungen.

## 5. Linker- und GNU-Toolchain-Setup

Für den Build des Projekts wird explizit nicht der Standard-Linker einer `MSVC`-Toolchain verwendet, sondern der GCC-Linker aus der `MSYS2/MinGW`-Umgebung:

- Linker: `C:\msys64\mingw64\bin\gcc.exe`

Zusätzlich werden folgende Rust-Linker-Argumente verwendet:

- `-lffi`
- `-Wl,--allow-multiple-definition`

## 6. Externe LLVM-Konfiguration

Neben dem internen LLVM des Rust-Compilers benötigt das Projekt ein externes LLVM für `llvm-sys` und `inkwell`. Die im Projekt verwendete Setup-Logik befindet sich in:

- `scripts/setup-build-env.ps1`

Aus dieser ergibt sich die folgende Soll-Konfiguration:

- `LLVM_SYS_180_PREFIX = C:\msys64\mingw64`

Das Projekt erwartet dabei eine LLVM-Version aus der `18.x`-Reihe. Konkret wird geprüft:

- `C:\msys64\mingw64\bin\llvm-config.exe`

## 7. Reproduzierbarer Build-Ablauf

Für die Nutzung auf dieser Maschine ergibt sich der folgende empfohlene Ablauf:

```powershell
.\scripts\setup-build-env.ps1
cargo build --offline
```

Der Parameter `--offline` ist dabei insbesondere dann sinnvoll, wenn bereits ein lokaler Crate-Cache vorliegt oder die Umgebung nur eingeschränkten Netzwerkzugriff erlaubt.
