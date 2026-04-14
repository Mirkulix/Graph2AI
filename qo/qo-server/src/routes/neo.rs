//! Neo — aggregated hardware/memory/status endpoints for the Neo companion UI.
//!
//! - GET /api/neo/hardware    — live GPU/CPU/RAM snapshot via nvidia-smi + /proc
//! - GET /api/neo/memory      — recent memory entries (HDC + organism shared memory)
//! - GET /api/neo/status      — aggregated snapshot of organism + bus + hardware counts
//! - GET /api/neo/agents      — list of Claude Code subagent transcripts
//! - GET /api/neo/agents/{id} — detailed tool-call timeline for one subagent

use axum::{extract::Path as AxPath, extract::State, Json};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

const NEO_TRANSCRIPT_ROOT: &str =
    "/home/mirkulix/.claude/projects/-home-mirkulix-AI-neoqlang-qlang";

#[derive(Serialize, Default)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub util: u32,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
    pub temp: u32,
    pub power: u32,
}

#[derive(Serialize, Default)]
pub struct HardwareSnapshot {
    pub gpus: Vec<GpuInfo>,
    pub cpu_cores: u32,
    pub cpu_util: f32,
    pub ram_gb: f32,
    pub ram_used_gb: f32,
    pub source: String,
}

/// GET /api/neo/hardware
pub async fn hardware() -> Json<HardwareSnapshot> {
    let mut snap = HardwareSnapshot::default();
    snap.source = "unknown".into();

    // ── GPUs via nvidia-smi ───────────────────────────────
    // Query: index,name,util,mem.used,mem.total,temp,power.draw
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(o) = out {
        if o.status.success() {
            snap.source = "nvidia-smi".into();
            let stdout = String::from_utf8_lossy(&o.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
                if parts.len() >= 7 {
                    snap.gpus.push(GpuInfo {
                        index: parts[0].parse().unwrap_or(0),
                        name: parts[1].to_string(),
                        util: parts[2].parse().unwrap_or(0),
                        mem_used_mb: parts[3].parse().unwrap_or(0),
                        mem_total_mb: parts[4].parse().unwrap_or(0),
                        temp: parts[5].parse().unwrap_or(0),
                        power: parts[6].parse::<f32>().unwrap_or(0.0) as u32,
                    });
                }
            }
        }
    }

    // ── CPU cores & util ──────────────────────────────────
    snap.cpu_cores = num_cpus();
    snap.cpu_util = cpu_util_percent().unwrap_or(0.0);

    // ── RAM ───────────────────────────────────────────────
    if let Some((total_gb, used_gb)) = read_meminfo_gb() {
        snap.ram_gb = total_gb;
        snap.ram_used_gb = used_gb;
    }

    Json(snap)
}

fn num_cpus() -> u32 {
    // Read /proc/cpuinfo and count `processor` lines
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        return s.lines().filter(|l| l.starts_with("processor")).count() as u32;
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

/// Estimate CPU utilization by reading /proc/stat twice, ~100ms apart.
fn cpu_util_percent() -> Option<f32> {
    let s1 = read_cpu_jiffies()?;
    std::thread::sleep(std::time::Duration::from_millis(100));
    let s2 = read_cpu_jiffies()?;
    let total_d = s2.0.saturating_sub(s1.0) as f32;
    let idle_d = s2.1.saturating_sub(s1.1) as f32;
    if total_d <= 0.0 {
        return Some(0.0);
    }
    Some(((total_d - idle_d) / total_d) * 100.0)
}

fn read_cpu_jiffies() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/stat").ok()?;
    let line = s.lines().next()?;
    let parts: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    if parts.len() < 5 {
        return None;
    }
    let total: u64 = parts.iter().sum();
    let idle: u64 = parts[3] + parts.get(4).copied().unwrap_or(0);
    Some((total, idle))
}

fn read_meminfo_gb() -> Option<(f32, f32)> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb: u64 = 0;
    let mut avail_kb: u64 = 0;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = rest.trim().split_whitespace().next()?.parse().ok()?;
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = rest.trim().split_whitespace().next()?.parse().ok()?;
        }
    }
    if total_kb == 0 {
        return None;
    }
    let total_gb = total_kb as f32 / 1_048_576.0;
    let used_gb = (total_kb.saturating_sub(avail_kb)) as f32 / 1_048_576.0;
    Some((total_gb, used_gb))
}

#[derive(Serialize)]
pub struct MemEntry {
    pub source: String,
    pub key: String,
    pub preview: String,
}

#[derive(Serialize)]
pub struct MemoryListResponse {
    pub hdc_count: usize,
    pub organism_count: usize,
    pub entries: Vec<MemEntry>,
}

/// GET /api/neo/memory — lists recent entries from the vector store and the organism's shared memory.
pub async fn memory(State(state): State<Arc<AppState>>) -> Json<MemoryListResponse> {
    let mut entries: Vec<MemEntry> = Vec::new();

    // ── Vector memory (MemoryContext) — keys only
    let hdc_count = {
        let mem = state.memory.lock().await;
        let count = mem.count();
        // MemoryContext.recall returns top-K keys; use a broad query to pull some
        let top = mem.recall("", 20);
        for (key, _score) in top {
            let preview = if key.len() > 120 { format!("{}…", &key[..120]) } else { key.clone() };
            entries.push(MemEntry {
                source: "hdc".into(),
                key,
                preview,
            });
        }
        count
    };

    // ── Organism shared memory
    let organism_count = {
        let org = crate::routes::organism::organism_snapshot().await;
        for item in org.items.iter().rev().take(20) {
            let preview = if item.len() > 120 { format!("{}…", &item[..120]) } else { item.clone() };
            entries.push(MemEntry {
                source: "organism".into(),
                key: item.clone(),
                preview,
            });
        }
        org.items.len()
    };

    Json(MemoryListResponse {
        hdc_count,
        organism_count,
        entries,
    })
}

#[derive(Serialize)]
pub struct StatusSnapshot {
    pub server: &'static str,
    pub hdc_memory: usize,
    pub organism_generation: u32,
    pub organism_interactions: usize,
    pub organism_memory_items: usize,
    pub specialists: usize,
    pub gpu_count: usize,
    pub gpu_temps: Vec<u32>,
    pub gpu_utils: Vec<u32>,
}

/// GET /api/neo/status — aggregated snapshot for the Neo top strip.
pub async fn status(State(state): State<Arc<AppState>>) -> Json<StatusSnapshot> {
    let hdc = { state.memory.lock().await.count() };
    let org = crate::routes::organism::organism_snapshot().await;
    let hw = hardware().await.0;

    Json(StatusSnapshot {
        server: "online",
        hdc_memory: hdc,
        organism_generation: org.generation,
        organism_interactions: org.interactions,
        organism_memory_items: org.items.len(),
        specialists: org.specialists,
        gpu_count: hw.gpus.len(),
        gpu_temps: hw.gpus.iter().map(|g| g.temp).collect(),
        gpu_utils: hw.gpus.iter().map(|g| g.util).collect(),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Subagent transcript inspection (GET /api/neo/agents, /api/neo/agents/{id})
// ══════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
pub struct AgentSummary {
    pub id: String,
    pub session_id: String,
    #[serde(rename = "type")]
    pub agent_type: String,
    pub description: String,
    pub status: String,
    pub interactions: usize,
    pub tool_calls: usize,
    pub files_written: Vec<String>,
    pub files_read: Vec<String>,
    pub last_tool: Option<String>,
    pub last_activity_at: Option<String>,
    pub started_at: Option<String>,
    pub duration_secs: Option<i64>,
}

#[derive(Serialize)]
pub struct ToolEvent {
    pub ts: Option<String>,
    pub role: String,
    pub tool: Option<String>,
    pub summary: String,
    pub file_path: Option<String>,
    pub is_error: Option<bool>,
}

#[derive(Serialize)]
pub struct AgentDetail {
    pub summary: AgentSummary,
    pub events: Vec<ToolEvent>,
    pub initial_prompt: Option<String>,
}

fn parse_agent_summary(jsonl: &Path, session_id: &str) -> Option<AgentSummary> {
    let stem = jsonl.file_stem()?.to_str()?.to_string();
    let id = stem.strip_prefix("agent-").unwrap_or(&stem).to_string();

    let meta_path = jsonl.with_extension("meta.json");
    let (agent_type, description) = match std::fs::read_to_string(&meta_path) {
        Ok(s) => {
            let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
            (
                v.get("agentType").and_then(|x| x.as_str()).unwrap_or("general").to_string(),
                v.get("description").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            )
        }
        Err(_) => ("general".to_string(), String::new()),
    };

    let content = std::fs::read_to_string(jsonl).ok()?;
    let mut interactions = 0usize;
    let mut tool_calls = 0usize;
    let mut files_written: Vec<String> = Vec::new();
    let mut files_read: Vec<String> = Vec::new();
    let mut last_tool: Option<String> = None;
    let mut first_ts: Option<String> = None;
    let mut last_ts: Option<String> = None;
    let mut saw_stop = false;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        interactions += 1;
        if let Some(ts) = v.get("timestamp").and_then(|x| x.as_str()) {
            if first_ts.is_none() {
                first_ts = Some(ts.to_string());
            }
            last_ts = Some(ts.to_string());
        }
        let msg = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        if let Some(stop) = msg.get("stop_reason").and_then(|x| x.as_str()) {
            if matches!(stop, "end_turn" | "stop_sequence") {
                saw_stop = true;
            }
        }
        let ca = match msg.get("content") {
            Some(Value::Array(a)) => a,
            _ => continue,
        };
        for block in ca {
            if block.get("type").and_then(|x| x.as_str()) == Some("tool_use") {
                tool_calls += 1;
                let name = block
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                last_tool = Some(name.clone());
                if let Some(fp) = block
                    .get("input")
                    .and_then(|i| i.get("file_path"))
                    .and_then(|x| x.as_str())
                {
                    match name.as_str() {
                        "Write" | "Edit" => {
                            if !files_written.iter().any(|x| x == fp) {
                                files_written.push(fp.to_string());
                            }
                        }
                        "Read" => {
                            if !files_read.iter().any(|x| x == fp) {
                                files_read.push(fp.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Status heuristic: "running" if last-modified within 120s AND no end_turn seen.
    let mtime_secs = std::fs::metadata(jsonl)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let age = (now - mtime_secs).max(0);
    let status = if saw_stop {
        "done"
    } else if age < 120 {
        "running"
    } else {
        "done"
    };

    Some(AgentSummary {
        id,
        session_id: session_id.to_string(),
        agent_type,
        description,
        status: status.to_string(),
        interactions,
        tool_calls,
        files_written,
        files_read,
        last_tool,
        last_activity_at: last_ts,
        started_at: first_ts,
        duration_secs: None,
    })
}

fn scan_all_agents() -> Vec<AgentSummary> {
    let mut out = Vec::new();
    let root = Path::new(NEO_TRANSCRIPT_ROOT);
    let sessions = match std::fs::read_dir(root) {
        Ok(r) => r,
        Err(_) => return out,
    };
    for session in sessions.flatten() {
        let sp = session.path();
        if !sp.is_dir() {
            continue;
        }
        let sid = sp.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let sub = sp.join("subagents");
        let entries = match std::fs::read_dir(&sub) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(s) = parse_agent_summary(&p, &sid) {
                out.push(s);
            }
        }
    }
    out.sort_by(|a, b| b.last_activity_at.cmp(&a.last_activity_at));
    out
}

/// GET /api/neo/agents
pub async fn list_agents() -> Json<Vec<AgentSummary>> {
    Json(scan_all_agents())
}

fn find_agent_jsonl(agent_id: &str) -> Option<(PathBuf, String)> {
    let root = Path::new(NEO_TRANSCRIPT_ROOT);
    for session in std::fs::read_dir(root).ok()?.flatten() {
        let sp = session.path();
        if !sp.is_dir() {
            continue;
        }
        let candidate = sp
            .join("subagents")
            .join(format!("agent-{}.jsonl", agent_id));
        if candidate.exists() {
            let sid = sp.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            return Some((candidate, sid));
        }
    }
    None
}

/// GET /api/neo/agents/{id}
pub async fn get_agent(AxPath(id): AxPath<String>) -> Json<Option<AgentDetail>> {
    let (path, sid) = match find_agent_jsonl(&id) {
        Some(x) => x,
        None => return Json(None),
    };
    let summary = match parse_agent_summary(&path, &sid) {
        Some(s) => s,
        None => return Json(None),
    };
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    let mut events: Vec<ToolEvent> = Vec::new();
    let mut initial_prompt: Option<String> = None;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ts = v.get("timestamp").and_then(|x| x.as_str()).map(String::from);
        let msg = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        let role = msg.get("role").and_then(|x| x.as_str()).unwrap_or("?").to_string();

        if initial_prompt.is_none() && role == "user" {
            if let Some(s) = msg.get("content").and_then(|x| x.as_str()) {
                initial_prompt = Some(s.chars().take(2000).collect());
            }
        }

        let ca = match msg.get("content") {
            Some(Value::Array(a)) => a,
            _ => continue,
        };
        for block in ca {
            let ty = block.get("type").and_then(|x| x.as_str()).unwrap_or("");
            match ty {
                "tool_use" => {
                    let name = block
                        .get("name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let file_path = block
                        .get("input")
                        .and_then(|i| i.get("file_path"))
                        .and_then(|x| x.as_str())
                        .map(String::from);
                    let summary_txt = block
                        .get("input")
                        .and_then(|i| {
                            i.get("description")
                                .or_else(|| i.get("command"))
                                .or_else(|| i.get("pattern"))
                                .or_else(|| i.get("file_path"))
                                .and_then(|x| x.as_str())
                        })
                        .unwrap_or("")
                        .chars()
                        .take(300)
                        .collect::<String>();
                    events.push(ToolEvent {
                        ts: ts.clone(),
                        role: "assistant".into(),
                        tool: Some(name),
                        summary: summary_txt,
                        file_path,
                        is_error: None,
                    });
                }
                "tool_result" => {
                    let is_err = block.get("is_error").and_then(|x| x.as_bool());
                    let txt = match block.get("content") {
                        Some(Value::String(s)) => s.chars().take(200).collect::<String>(),
                        Some(Value::Array(a)) => a
                            .iter()
                            .filter_map(|b| b.get("text").and_then(|x| x.as_str()))
                            .collect::<Vec<_>>()
                            .join(" ")
                            .chars()
                            .take(200)
                            .collect::<String>(),
                        _ => String::new(),
                    };
                    events.push(ToolEvent {
                        ts: ts.clone(),
                        role: "user".into(),
                        tool: Some("tool_result".into()),
                        summary: txt,
                        file_path: None,
                        is_error: is_err,
                    });
                }
                "text" => {
                    if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                        events.push(ToolEvent {
                            ts: ts.clone(),
                            role: role.clone(),
                            tool: None,
                            summary: t.chars().take(400).collect(),
                            file_path: None,
                            is_error: None,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    Json(Some(AgentDetail {
        summary,
        events,
        initial_prompt,
    }))
}
