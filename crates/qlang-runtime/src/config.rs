//! Production-ready runtime configuration, logging, and execution guards.
//!
//! Uses only `std` for logging (no external logging crates).
//! Uses `serde`/`serde_json` (already workspace deps) for JSON config parsing.

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity levels for log messages, ordered from most to least severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Trace => "TRACE",
        };
        f.write_str(s)
    }
}

impl LogLevel {
    /// Parse a log level from a case-insensitive string.
    pub fn from_str_loose(s: &str) -> Option<LogLevel> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Some(LogLevel::Error),
            "warn" | "warning" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// QlangConfig
// ---------------------------------------------------------------------------

/// Top-level runtime configuration.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct QlangConfig {
    /// Minimum severity that will be emitted by the logger.
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,

    /// Maximum heap memory the runtime is allowed to track (MiB).
    #[serde(default = "default_max_memory_mb")]
    pub max_memory_mb: usize,

    /// Maximum wall-clock execution time (ms) before the guard trips.
    #[serde(default = "default_max_execution_time_ms")]
    pub max_execution_time_ms: u64,

    /// Enable built-in profiling hooks.
    #[serde(default)]
    pub enable_profiling: bool,

    /// Enable JIT compilation path.
    #[serde(default)]
    pub enable_jit: bool,

    /// Number of worker threads.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Enable parallel execution of independent graph nodes via rayon.
    #[serde(default)]
    pub enable_parallel: bool,

    /// Directory for cached model artefacts.
    #[serde(default = "default_model_cache_dir")]
    pub model_cache_dir: String,
}

fn default_log_level() -> LogLevel {
    LogLevel::Info
}
fn default_max_memory_mb() -> usize {
    1024
}
fn default_max_execution_time_ms() -> u64 {
    30_000
}
fn default_num_threads() -> usize {
    1
}
fn default_model_cache_dir() -> String {
    String::from(".qlang/cache")
}

impl Default for QlangConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            max_memory_mb: default_max_memory_mb(),
            max_execution_time_ms: default_max_execution_time_ms(),
            enable_profiling: false,
            enable_jit: false,
            num_threads: default_num_threads(),
            enable_parallel: false,
            model_cache_dir: default_model_cache_dir(),
        }
    }
}

impl QlangConfig {
    /// Build a config by reading environment variables.
    ///
    /// Any variable that is absent or unparseable is left at its default value.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("QLANG_LOG_LEVEL") {
            if let Some(lvl) = LogLevel::from_str_loose(&v) {
                cfg.log_level = lvl;
            }
        }
        if let Ok(v) = std::env::var("QLANG_MAX_MEMORY_MB") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.max_memory_mb = n;
            }
        }
        if let Ok(v) = std::env::var("QLANG_MAX_TIME_MS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.max_execution_time_ms = n;
            }
        }
        if let Ok(v) = std::env::var("QLANG_ENABLE_JIT") {
            cfg.enable_jit = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "yes");
        }
        if let Ok(v) = std::env::var("QLANG_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.num_threads = n;
            }
        }
        if let Ok(v) = std::env::var("QLANG_PARALLEL") {
            cfg.enable_parallel = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "yes");
        }

        cfg
    }

    /// Deserialise a config from a JSON file on disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

/// Logging sink — either stdout or a file, behind a mutex so it is `Send`.
enum LogSink {
    Stdout,
    File(fs::File),
}

/// A simple, timestamped logger that filters by [`LogLevel`].
pub struct Logger {
    level: LogLevel,
    sink: Mutex<LogSink>,
}

impl Logger {
    /// Create a logger that writes to **stdout**.
    pub fn stdout(level: LogLevel) -> Self {
        Self {
            level,
            sink: Mutex::new(LogSink::Stdout),
        }
    }

    /// Create a logger that writes to a **file** (created or truncated).
    pub fn to_file<P: AsRef<Path>>(level: LogLevel, path: P) -> io::Result<Self> {
        let file = fs::File::create(path)?;
        Ok(Self {
            level,
            sink: Mutex::new(LogSink::File(file)),
        })
    }

    /// Emit a log line if `level` passes the configured filter.
    ///
    /// Format: `[YYYY-MM-DD HH:MM:SS] [LEVEL] message`
    pub fn log(&self, level: LogLevel, message: &str) {
        if level > self.level {
            return;
        }
        let ts = format_timestamp();
        let line = format!("[{}] [{}] {}\n", ts, level, message);

        if let Ok(mut sink) = self.sink.lock() {
            match &mut *sink {
                LogSink::Stdout => {
                    let _ = io::stdout().write_all(line.as_bytes());
                }
                LogSink::File(f) => {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
    }

    pub fn error(&self, msg: &str) {
        self.log(LogLevel::Error, msg);
    }
    pub fn warn(&self, msg: &str) {
        self.log(LogLevel::Warn, msg);
    }
    pub fn info(&self, msg: &str) {
        self.log(LogLevel::Info, msg);
    }
    pub fn debug(&self, msg: &str) {
        self.log(LogLevel::Debug, msg);
    }
    pub fn trace(&self, msg: &str) {
        self.log(LogLevel::Trace, msg);
    }

    /// Format a single log line (without writing). Useful for tests.
    pub fn format_line(level: LogLevel, message: &str) -> String {
        let ts = format_timestamp();
        format!("[{}] [{}] {}", ts, level, message)
    }
}

/// Produce a UTC timestamp string `YYYY-MM-DD HH:MM:SS`.
fn format_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    // Simple UTC calendar arithmetic (no chrono dependency).
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;

    // Days since 1970-01-01 → (Y, M, D) — civil calendar algorithm.
    let (y, m, d) = days_to_ymd(days as i64);

    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, m, d, hh, mm, ss)
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from Howard Hinnant (public domain).
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// ExecutionGuard
// ---------------------------------------------------------------------------

/// Runtime resource guard that enforces memory, time, and recursion limits.
///
/// All limits are *soft*: the user code must call the appropriate `check_*`
/// method at appropriate points (e.g. every allocation, every function call).
#[derive(Debug)]
pub struct ExecutionGuard {
    // Memory
    max_memory_bytes: usize,
    allocated_bytes: Arc<AtomicUsize>,

    // Time
    max_execution_ms: u64,
    start: Instant,

    // Recursion
    max_recursion_depth: usize,
    current_depth: usize,
}

/// Errors returned when a resource limit is exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardError {
    MemoryLimitExceeded { used: usize, limit: usize },
    TimeLimitExceeded { elapsed_ms: u64, limit_ms: u64 },
    RecursionLimitExceeded { depth: usize, limit: usize },
}

impl fmt::Display for GuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardError::MemoryLimitExceeded { used, limit } => {
                write!(f, "memory limit exceeded: {} bytes used, {} limit", used, limit)
            }
            GuardError::TimeLimitExceeded { elapsed_ms, limit_ms } => {
                write!(f, "time limit exceeded: {}ms elapsed, {}ms limit", elapsed_ms, limit_ms)
            }
            GuardError::RecursionLimitExceeded { depth, limit } => {
                write!(f, "recursion limit exceeded: depth {}, limit {}", depth, limit)
            }
        }
    }
}

impl std::error::Error for GuardError {}

impl ExecutionGuard {
    /// Create a new guard from a [`QlangConfig`].
    pub fn new(cfg: &QlangConfig) -> Self {
        Self {
            max_memory_bytes: cfg.max_memory_mb * 1024 * 1024,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            max_execution_ms: cfg.max_execution_time_ms,
            start: Instant::now(),
            max_recursion_depth: 512,
            current_depth: 0,
        }
    }

    /// Create a guard with explicit limits (useful for tests).
    pub fn with_limits(max_memory_bytes: usize, max_execution_ms: u64, max_recursion_depth: usize) -> Self {
        Self {
            max_memory_bytes,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            max_execution_ms,
            start: Instant::now(),
            max_recursion_depth,
            current_depth: 0,
        }
    }

    // -- Memory -------------------------------------------------------------

    /// Record an allocation of `bytes` and check the limit.
    pub fn allocate(&self, bytes: usize) -> Result<(), GuardError> {
        let prev = self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        let used = prev + bytes;
        if used > self.max_memory_bytes {
            // Roll back so the counter stays consistent.
            self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
            return Err(GuardError::MemoryLimitExceeded {
                used,
                limit: self.max_memory_bytes,
            });
        }
        Ok(())
    }

    /// Record a deallocation.
    pub fn deallocate(&self, bytes: usize) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current tracked allocation in bytes.
    pub fn allocated(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    // -- Time ---------------------------------------------------------------

    /// Check that execution time has not exceeded the limit.
    pub fn check_time(&self) -> Result<(), GuardError> {
        let elapsed = self.start.elapsed().as_millis() as u64;
        if elapsed > self.max_execution_ms {
            return Err(GuardError::TimeLimitExceeded {
                elapsed_ms: elapsed,
                limit_ms: self.max_execution_ms,
            });
        }
        Ok(())
    }

    /// Elapsed milliseconds since guard creation.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    // -- Recursion ----------------------------------------------------------

    /// Enter a recursive call. Returns `Err` if the limit would be exceeded.
    pub fn enter_recursion(&mut self) -> Result<(), GuardError> {
        if self.current_depth >= self.max_recursion_depth {
            return Err(GuardError::RecursionLimitExceeded {
                depth: self.current_depth,
                limit: self.max_recursion_depth,
            });
        }
        self.current_depth += 1;
        Ok(())
    }

    /// Leave a recursive call.
    pub fn leave_recursion(&mut self) {
        self.current_depth = self.current_depth.saturating_sub(1);
    }

    /// Current recursion depth.
    pub fn recursion_depth(&self) -> usize {
        self.current_depth
    }

    // -- Reset --------------------------------------------------------------

    /// Reset all counters (memory, time, recursion) so the guard can be reused.
    pub fn reset(&mut self) {
        self.allocated_bytes.store(0, Ordering::Relaxed);
        self.start = Instant::now();
        self.current_depth = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    // 1. Default config values
    #[test]
    fn test_default_config() {
        let cfg = QlangConfig::default();
        assert_eq!(cfg.log_level, LogLevel::Info);
        assert_eq!(cfg.max_memory_mb, 1024);
        assert_eq!(cfg.max_execution_time_ms, 30_000);
        assert!(!cfg.enable_profiling);
        assert!(!cfg.enable_jit);
        assert_eq!(cfg.num_threads, 1);
        assert_eq!(cfg.model_cache_dir, ".qlang/cache");
    }

    // 2. Env var parsing
    #[test]
    fn test_config_from_env() {
        // SAFETY: This test is run single-threaded for env var manipulation.
        unsafe {
            std::env::set_var("QLANG_LOG_LEVEL", "debug");
            std::env::set_var("QLANG_MAX_MEMORY_MB", "2048");
            std::env::set_var("QLANG_MAX_TIME_MS", "60000");
            std::env::set_var("QLANG_ENABLE_JIT", "true");
            std::env::set_var("QLANG_THREADS", "4");
        }

        let cfg = QlangConfig::from_env();
        assert_eq!(cfg.log_level, LogLevel::Debug);
        assert_eq!(cfg.max_memory_mb, 2048);
        assert_eq!(cfg.max_execution_time_ms, 60_000);
        assert!(cfg.enable_jit);
        assert_eq!(cfg.num_threads, 4);

        // Clean up.
        unsafe {
            std::env::remove_var("QLANG_LOG_LEVEL");
            std::env::remove_var("QLANG_MAX_MEMORY_MB");
            std::env::remove_var("QLANG_MAX_TIME_MS");
            std::env::remove_var("QLANG_ENABLE_JIT");
            std::env::remove_var("QLANG_THREADS");
        }
    }

    // 3. Logger output format
    #[test]
    fn test_logger_output_format() {
        let tmp = std::env::temp_dir().join("qlang_log_test.log");
        {
            let logger = Logger::to_file(LogLevel::Info, &tmp).unwrap();
            logger.info("hello world");
        }

        let mut contents = String::new();
        fs::File::open(&tmp)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = fs::remove_file(&tmp);

        // Should match pattern: [YYYY-MM-DD HH:MM:SS] [INFO] hello world
        assert!(
            contents.contains("[INFO] hello world"),
            "unexpected log output: {:?}",
            contents
        );
        // Check timestamp bracket structure.
        assert!(contents.starts_with('['), "should start with timestamp bracket");
        // Rough regex-like check: "[nnnn-nn-nn nn:nn:nn]"
        let closing = contents.find(']').expect("closing bracket");
        let ts_part = &contents[1..closing];
        assert_eq!(ts_part.len(), 19, "timestamp should be 19 chars: {}", ts_part);
    }

    // 4. Memory limit enforcement
    #[test]
    fn test_memory_limit() {
        let guard = ExecutionGuard::with_limits(1024, u64::MAX, 512);

        // Allocate within limit.
        assert!(guard.allocate(512).is_ok());
        assert_eq!(guard.allocated(), 512);

        // Allocate beyond limit should fail.
        let err = guard.allocate(1024).unwrap_err();
        assert!(matches!(err, GuardError::MemoryLimitExceeded { .. }));

        // Counter should not have moved after the failed allocation.
        assert_eq!(guard.allocated(), 512);

        // Deallocate and re-allocate.
        guard.deallocate(512);
        assert_eq!(guard.allocated(), 0);
        assert!(guard.allocate(1024).is_ok());
    }

    // 5. Time limit enforcement
    #[test]
    fn test_time_limit() {
        // Use a 1 ms limit — by the time we do anything it should be exceeded.
        let guard = ExecutionGuard::with_limits(usize::MAX, 0, 512);
        // Even a 0-ms limit trips after any measurable delay.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let err = guard.check_time().unwrap_err();
        assert!(matches!(err, GuardError::TimeLimitExceeded { .. }));
    }

    // 6. Config from JSON
    #[test]
    fn test_config_from_json() {
        let tmp = std::env::temp_dir().join("qlang_config_test.json");
        let json = r#"{
            "log_level": "warn",
            "max_memory_mb": 4096,
            "max_execution_time_ms": 10000,
            "enable_profiling": true,
            "enable_jit": false,
            "num_threads": 8,
            "model_cache_dir": "/tmp/models"
        }"#;
        fs::write(&tmp, json).unwrap();

        let cfg = QlangConfig::from_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(cfg.log_level, LogLevel::Warn);
        assert_eq!(cfg.max_memory_mb, 4096);
        assert_eq!(cfg.max_execution_time_ms, 10_000);
        assert!(cfg.enable_profiling);
        assert!(!cfg.enable_jit);
        assert_eq!(cfg.num_threads, 8);
        assert_eq!(cfg.model_cache_dir, "/tmp/models");
    }

    // 7. Log level filtering
    #[test]
    fn test_log_level_filtering() {
        let tmp = std::env::temp_dir().join("qlang_log_filter_test.log");
        {
            let logger = Logger::to_file(LogLevel::Warn, &tmp).unwrap();
            logger.error("err");
            logger.warn("wrn");
            logger.info("inf");   // should be filtered
            logger.debug("dbg");  // should be filtered
            logger.trace("trc");  // should be filtered
        }

        let mut contents = String::new();
        fs::File::open(&tmp)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = fs::remove_file(&tmp);

        assert!(contents.contains("[ERROR] err"));
        assert!(contents.contains("[WARN] wrn"));
        assert!(!contents.contains("[INFO]"));
        assert!(!contents.contains("[DEBUG]"));
        assert!(!contents.contains("[TRACE]"));
    }

    // 8. ExecutionGuard resets
    #[test]
    fn test_execution_guard_reset() {
        let mut guard = ExecutionGuard::with_limits(1024, u64::MAX, 4);

        // Use some resources.
        guard.allocate(512).unwrap();
        guard.enter_recursion().unwrap();
        guard.enter_recursion().unwrap();
        assert_eq!(guard.allocated(), 512);
        assert_eq!(guard.recursion_depth(), 2);

        // Reset.
        guard.reset();
        assert_eq!(guard.allocated(), 0);
        assert_eq!(guard.recursion_depth(), 0);

        // Should be usable again up to original limits.
        assert!(guard.allocate(1024).is_ok());
        assert!(guard.enter_recursion().is_ok());
    }

    // 9. Recursion limit
    #[test]
    fn test_recursion_limit() {
        let mut guard = ExecutionGuard::with_limits(usize::MAX, u64::MAX, 3);
        assert!(guard.enter_recursion().is_ok()); // depth 1
        assert!(guard.enter_recursion().is_ok()); // depth 2
        assert!(guard.enter_recursion().is_ok()); // depth 3
        let err = guard.enter_recursion().unwrap_err();
        assert!(matches!(err, GuardError::RecursionLimitExceeded { .. }));

        guard.leave_recursion(); // depth 2
        assert!(guard.enter_recursion().is_ok()); // depth 3 again
    }

    // 10. JSON config with defaults for missing fields
    #[test]
    fn test_config_json_partial() {
        let tmp = std::env::temp_dir().join("qlang_config_partial.json");
        fs::write(&tmp, r#"{"log_level": "trace"}"#).unwrap();

        let cfg = QlangConfig::from_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(cfg.log_level, LogLevel::Trace);
        // Everything else should be default.
        assert_eq!(cfg.max_memory_mb, 1024);
        assert_eq!(cfg.num_threads, 1);
    }
}
