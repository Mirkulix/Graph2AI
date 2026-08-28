//! `qo-desktop` — the native desktop shell for the QO cockpit.
//!
//! # Why this exists rather than a browser tab
//!
//! The cockpit is a long-running operator surface: it holds a live delta feed,
//! a knowledge graph and a websocket to the supervisor. In a browser tab it
//! competes with the operator's other tabs, dies on an accidental close, and
//! shares a cookie/storage origin with everything else on `localhost`. As an
//! app window it gets its own taskbar entry, its own WebView2 user-data
//! directory, and a lifecycle tied to the server it drives.
//!
//! # Why not Tauri
//!
//! Tauri links WebView2 through the MSVC toolchain. This machine's active
//! Rust toolchain is `x86_64-pc-windows-gnu` and no Visual Studio / MSVC
//! build tools are installed, so a Tauri build cannot link here. Rather than
//! make a working system depend on a multi-gigabyte toolchain install, this
//! launcher drives the *same* WebView2 runtime that is already present
//! (`Microsoft\EdgeWebView\Application\<version>`) through Edge's documented
//! `--app=` mode. The result is the same thing an end user wants — a native
//! window with no browser chrome — with no new build dependency.
//!
//! # What it guarantees
//!
//! - The server is a supervised child: if the window closes, the server is
//!   stopped; if the server dies, that is reported rather than leaving an
//!   empty window.
//! - An already-running QO is reused rather than fought over, so a second
//!   launch does not produce a port collision or a second database handle.
//! - The window is never opened against a server that is not answering yet;
//!   readiness is polled on `/api/health`.

use std::io::{Read, Write as _};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// How long to wait for the server to answer `/api/health` before giving up.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);
/// Gap between readiness probes. Short enough to feel instant on a warm start.
const PROBE_INTERVAL: Duration = Duration::from_millis(250);
/// Grace period for the server to exit on its own after the window closes.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

fn main() {
    if let Err(err) = run() {
        eprintln!("qo-desktop: {err}");
        // A GUI launcher that dies silently is a support ticket. Keep the
        // console open long enough to read the reason when double-clicked.
        if std::env::var("QO_DESKTOP_NO_PAUSE").is_err() {
            eprintln!("\nPress Enter to close.");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let port: u16 = std::env::var("QO_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(4646);
    let url = format!("http://127.0.0.1:{port}");

    // Reuse an instance that is already up. Two QO processes on one redb file
    // is a corruption risk, not a convenience.
    let mut server: Option<Child> = None;
    if health_ok(port) {
        println!("qo-desktop: reusing the QO already listening on {port}");
    } else {
        if port_in_use(port) {
            return Err(format!(
                "port {port} is occupied by something that is not answering \
                 /api/health. Stop it, or set QO_PORT to a free port."
            ));
        }
        println!("qo-desktop: starting QO on {port} ...");
        server = Some(spawn_server(port)?);
        wait_until_ready(port, server.as_mut().expect("just spawned"))?;
        println!("qo-desktop: QO is ready");
    }

    let webview = locate_webview_runtime()
        .ok_or_else(|| {
            "no WebView2 / Edge runtime found. Install the Microsoft Edge \
             WebView2 Runtime, or open the cockpit in a browser instead."
                .to_string()
        })?;

    let profile = user_data_dir();
    std::fs::create_dir_all(&profile)
        .map_err(|e| format!("cannot create window profile dir {}: {e}", profile.display()))?;

    println!("qo-desktop: opening the cockpit window");
    let status = Command::new(&webview)
        // `--app=` is what removes the browser chrome: no tab strip, no
        // omnibox, own taskbar entry. This is the whole "native" part.
        .arg(format!("--app={url}"))
        // A dedicated profile keeps cockpit storage out of the operator's
        // normal browsing profile, and keeps this window from being adopted
        // by an already-running Edge (which would detach it from our wait).
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg(format!("--window-size={},{}", 1440, 900))
        .status()
        .map_err(|e| format!("cannot launch the window via {}: {e}", webview.display()))?;

    if !status.success() {
        eprintln!("qo-desktop: window exited with {status}");
    }

    // The window is the app's lifetime. Only stop what we started — an
    // instance we merely reused belongs to whoever launched it.
    if let Some(mut child) = server {
        println!("qo-desktop: window closed, stopping QO");
        stop_server(&mut child);
    }
    Ok(())
}

/// Start `qo` as a child process, inheriting this process's environment.
fn spawn_server(port: u16) -> Result<Child, String> {
    let exe = locate_qo_binary()?;
    Command::new(&exe)
        .arg("--offline")
        .env("QO_PORT", port.to_string())
        // Server logs belong in this console; the window has its own devtools.
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))
}

/// Find the `qo` binary: next to this launcher first (the shipped layout),
/// then on PATH (a `cargo install`ed copy).
fn locate_qo_binary() -> Result<PathBuf, String> {
    if let Ok(explicit) = std::env::var("QO_BINARY") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!("QO_BINARY points at {}, which is not a file", path.display()));
    }

    let exe_name = if cfg!(windows) { "qo.exe" } else { "qo" };
    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            let sibling = dir.join(exe_name);
            if sibling.is_file() {
                return Ok(sibling);
            }
        }
    }
    if let Some(found) = which(exe_name) {
        return Ok(found);
    }
    Err(format!(
        "cannot find `{exe_name}` next to this launcher or on PATH. \
         Build it with `cargo build --release --bin qo`, or set QO_BINARY."
    ))
}

/// Locate the WebView2 runtime, falling back to installed Edge.
///
/// WebView2 is checked first: it is the runtime meant to host an embedded app
/// window, and using it avoids attaching to the operator's running browser.
fn locate_webview_runtime() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("QO_WEBVIEW") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }

    let program_files = [
        std::env::var("ProgramFiles(x86)").ok(),
        std::env::var("ProgramFiles").ok(),
    ];

    // WebView2 keeps one directory per installed version; take the newest.
    for base in program_files.iter().flatten() {
        let root = Path::new(base).join("Microsoft/EdgeWebView/Application");
        if let Some(exe) = newest_versioned_binary(&root, "msedgewebview2.exe") {
            return Some(exe);
        }
    }

    for base in program_files.iter().flatten() {
        let edge = Path::new(base).join("Microsoft/Edge/Application/msedge.exe");
        if edge.is_file() {
            return Some(edge);
        }
    }

    // Non-Windows dev machines: any Chromium supports --app=.
    for candidate in [
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    for name in ["microsoft-edge", "google-chrome", "chromium"] {
        if let Some(found) = which(name) {
            return Some(found);
        }
    }
    None
}

/// Newest `<root>/<version>/<binary>`, ordered by parsed version components so
/// that `151.0.4129.107` sorts above `151.0.4129.101` (a lexical sort does not).
fn newest_versioned_binary(root: &Path, binary: &str) -> Option<PathBuf> {
    let mut best: Option<(Vec<u64>, PathBuf)> = None;
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let exe = entry.path().join(binary);
        if !exe.is_file() {
            continue;
        }
        let version = entry
            .file_name()
            .to_string_lossy()
            .split('.')
            .map(|p| p.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>();
        let better = match &best {
            Some((current, _)) => version > *current,
            None => true,
        };
        if better {
            best = Some((version, exe));
        }
    }
    best.map(|(_, path)| path)
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Block until the server answers `/api/health`, or the child dies, or we
/// run out of patience.
fn wait_until_ready(port: u16, server: &mut Child) -> Result<(), String> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        // A crashed child will never become ready; say so instead of waiting
        // out the full timeout.
        if let Ok(Some(status)) = server.try_wait() {
            return Err(format!("QO exited during startup with {status}"));
        }
        if health_ok(port) {
            return Ok(());
        }
        std::thread::sleep(PROBE_INTERVAL);
    }
    let _ = server.kill();
    Err(format!(
        "QO did not answer /api/health on port {port} within {}s",
        STARTUP_TIMEOUT.as_secs()
    ))
}

/// Minimal HTTP/1.1 probe. Deliberately dependency-free: the launcher must
/// build even when the workspace's heavier features are disabled.
fn health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    // Only the status line matters; cap the read so a chatty endpoint cannot
    // stall startup.
    let mut buf = [0u8; 512];
    while response.len() < 4096 {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => response.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
        if response.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    // Auth may reject us (401/403) — that still proves the server is up and
    // routing, which is all readiness means here.
    let head = String::from_utf8_lossy(&response);
    let Some(status_line) = head.lines().next() else {
        return false;
    };
    status_line.starts_with("HTTP/1.1 2")
        || status_line.starts_with("HTTP/1.1 401")
        || status_line.starts_with("HTTP/1.1 403")
}

/// True when something holds the port, whether or not it is a healthy QO.
fn port_in_use(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// Stop the supervised server, giving it a chance to flush redb before a kill.
fn stop_server(child: &mut Child) {
    #[cfg(windows)]
    {
        // No portable SIGTERM on Windows. `taskkill` without /F asks the
        // process to close first, which lets redb finish its write.
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    let deadline = Instant::now() + SHUTDOWN_GRACE;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Per-user directory for the window's WebView2 profile.
fn user_data_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("QO_DESKTOP_PROFILE") {
        return PathBuf::from(explicit);
    }
    let base = std::env::var("LOCALAPPDATA")
        .or_else(|_| std::env::var("XDG_DATA_HOME"))
        .or_else(|_| std::env::var("HOME").map(|h| format!("{h}/.local/share")))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(base).join("OrbitQLang/desktop-profile")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_version_wins_numerically_not_lexically() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Lexically "151.0.4129.101" > "151.0.4129.99", but numerically the
        // .107 build is the newest of the three.
        for version in ["151.0.4129.99", "151.0.4129.101", "151.0.4129.107"] {
            let app = root.join(version);
            std::fs::create_dir_all(&app).unwrap();
            std::fs::write(app.join("msedgewebview2.exe"), b"stub").unwrap();
        }
        let found = newest_versioned_binary(root, "msedgewebview2.exe").unwrap();
        assert!(
            found.to_string_lossy().contains("151.0.4129.107"),
            "expected the newest build, got {}",
            found.display()
        );
    }

    #[test]
    fn missing_runtime_directory_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(newest_versioned_binary(&missing, "msedgewebview2.exe").is_none());
    }

    #[test]
    fn a_version_directory_without_the_binary_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("151.0.4129.107")).unwrap();
        let with_binary = dir.path().join("150.0.1.1");
        std::fs::create_dir_all(&with_binary).unwrap();
        std::fs::write(with_binary.join("msedgewebview2.exe"), b"stub").unwrap();

        let found = newest_versioned_binary(dir.path(), "msedgewebview2.exe").unwrap();
        assert!(
            found.to_string_lossy().contains("150.0.1.1"),
            "a newer directory without the binary must not win"
        );
    }

    #[test]
    fn health_probe_reports_false_on_a_closed_port() {
        // Port 1 is reserved and never bound by QO.
        assert!(!health_ok(1));
        assert!(!port_in_use(1));
    }

    #[test]
    fn profile_dir_is_overridable() {
        std::env::set_var("QO_DESKTOP_PROFILE", "/tmp/qo-profile-test");
        assert_eq!(user_data_dir(), PathBuf::from("/tmp/qo-profile-test"));
        std::env::remove_var("QO_DESKTOP_PROFILE");
    }
}
