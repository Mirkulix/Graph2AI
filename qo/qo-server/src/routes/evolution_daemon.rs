//! Evolution Daemon HTTP API — drives the EvolutionDashboard frontend.
//!
//! By default these endpoints operate the **real** evolution daemon
//! (`qlang_runtime::evolution::real_daemon::RealEvolutionDaemon`), which
//! seeds a population of ternary MNIST specialists from real training data
//! and evaluates fitness by measuring accuracy on held-out test samples.
//!
//! If the request body sets `use_real_daemon: false`, the legacy simulated
//! daemon (`evolution::daemon::EvolutionDaemon`) is used instead. That path
//! is retained for backwards compatibility with tooling / UIs that still
//! depend on the simulated trajectory.
//!
//! Endpoints:
//!   POST /api/evolution/start        — start daemon with optional config
//!   POST /api/evolution/stop         — stop the daemon
//!   GET  /api/evolution/status       — current status + last report
//!   GET  /api/evolution/history      — all generation reports (most recent 500)
//!   GET  /api/evolution/specialists  — active + retired specialists with fitness
//!   GET  /api/evolution/lineage/:id  — lineage tree (ancestors + descendants)
//!   GET  /api/evolution/stream       — SSE stream of new GenerationReports

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use qlang_runtime::evolution::daemon::{
    DaemonConfig as StubDaemonConfig, EvolutionDaemon as StubEvolutionDaemon,
    GenerationReport as StubGenerationReport, SharedDaemon as StubSharedDaemon,
    SpecialistInfo as StubSpecialistInfo,
};
use qlang_runtime::evolution::real_daemon::{
    EvolutionConfig as RealEvolutionConfig, RealEvolutionDaemon,
};
use qlang_runtime::mnist::MnistData;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

use crate::AppState;

// ---------------------------------------------------------------------------
// Request / Response payloads
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct StartRequest {
    /// When true (default) the real ternary-MNIST daemon is started.
    /// Set to `false` to fall back to the legacy simulated daemon.
    #[serde(default = "default_use_real_daemon")]
    pub use_real_daemon: bool,

    // --- shared knobs ---
    pub interval_secs: Option<u64>,
    pub population_size: Option<usize>,
    pub mutation_rate: Option<f32>,
    pub retire_fraction: Option<f32>,
    pub max_age: Option<u32>,
    pub seed: Option<u64>,

    // --- real-daemon specific ---
    /// How many MNIST test samples to use per fitness evaluation.
    pub eval_sample_size: Option<usize>,
    /// Offspring spawned per generation.
    pub mutations_per_generation: Option<usize>,
    /// Directory containing MNIST files.
    pub data_path: Option<String>,
}

fn default_use_real_daemon() -> bool {
    true
}

impl StartRequest {
    fn into_stub_config(&self) -> StubDaemonConfig {
        let defaults = StubDaemonConfig::default();
        StubDaemonConfig {
            interval_secs: self.interval_secs.unwrap_or(defaults.interval_secs).max(1),
            population_size: self
                .population_size
                .unwrap_or(defaults.population_size)
                .clamp(4, 500),
            mutation_rate: self
                .mutation_rate
                .unwrap_or(defaults.mutation_rate)
                .clamp(0.0, 1.0),
            retire_fraction: self
                .retire_fraction
                .unwrap_or(defaults.retire_fraction)
                .clamp(0.0, 0.9),
            max_age: self.max_age.unwrap_or(defaults.max_age).max(1),
            seed: self.seed.unwrap_or(defaults.seed),
        }
    }

    fn into_real_config(&self) -> RealEvolutionConfig {
        let mut cfg = RealEvolutionConfig::default();
        if let Some(s) = self.interval_secs {
            cfg.generation_interval_secs = s.max(1);
        } else {
            // HTTP callers usually want snappier feedback than the 300s default.
            cfg.generation_interval_secs = 30;
        }
        if let Some(n) = self.eval_sample_size {
            cfg.eval_sample_size = n.max(1);
        }
        if let Some(n) = self.mutations_per_generation {
            cfg.mutations_per_generation = n.clamp(1, 200);
        }
        if let Some(p) = &self.data_path {
            if !p.is_empty() {
                cfg.data_path = p.clone();
            }
        }
        if let Some(m) = self.mutation_rate {
            cfg.mutation_config.flip_rate = m.clamp(0.0, 1.0);
        }
        if let Some(seed) = self.seed {
            cfg.mutation_config.seed = seed;
        }
        cfg
    }
}

#[derive(Serialize)]
pub struct SimpleResponse {
    pub ok: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// POST /api/evolution/start
// ---------------------------------------------------------------------------

pub async fn start(
    State(state): State<Arc<AppState>>,
    body: Option<Json<StartRequest>>,
) -> Json<SimpleResponse> {
    let req = body.map(|Json(r)| r).unwrap_or_default();
    if req.use_real_daemon {
        start_real(state, req).await
    } else {
        start_stub(state, req).await
    }
}

async fn start_real(state: Arc<AppState>, req: StartRequest) -> Json<SimpleResponse> {
    let population_size = req.population_size.unwrap_or(20).clamp(2, 500);
    let cfg = req.into_real_config();

    // Load MNIST — fall back to synthetic if the real data is missing so the
    // endpoint still comes up in dev environments without `data/mnist`.
    let mnist = match MnistData::load_from_dir(&cfg.data_path) {
        Ok(d) => d,
        Err(_) => MnistData::synthetic(2_000, 500),
    };

    let mut slot = state.real_evolution_daemon.lock().await;
    if let Some(existing) = slot.as_ref() {
        if existing.is_running() {
            return Json(SimpleResponse {
                ok: false,
                message: "real evolution daemon already running".into(),
            });
        }
    }
    let daemon = Arc::new(RealEvolutionDaemon::new(cfg));
    if let Err(e) = daemon.seed_population(population_size, &mnist) {
        return Json(SimpleResponse {
            ok: false,
            message: format!("seed_population failed: {e}"),
        });
    }
    match Arc::clone(&daemon).start() {
        Ok(_handle) => {
            *slot = Some(daemon);
            Json(SimpleResponse {
                ok: true,
                message: format!(
                    "real evolution daemon started — seeded {} specialists on MNIST",
                    population_size
                ),
            })
        }
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: format!("start failed: {e}"),
        }),
    }
}

async fn start_stub(state: Arc<AppState>, req: StartRequest) -> Json<SimpleResponse> {
    let cfg = req.into_stub_config();
    let daemon = {
        let mut slot = state.evolution_daemon.lock().await;
        if slot.is_none() {
            *slot = Some(StubEvolutionDaemon::new());
        }
        slot.as_ref().unwrap().clone()
    };
    match daemon.start(cfg) {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "stub (simulated) evolution daemon started".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e,
        }),
    }
}

// ---------------------------------------------------------------------------
// POST /api/evolution/stop
// ---------------------------------------------------------------------------

pub async fn stop(State(state): State<Arc<AppState>>) -> Json<SimpleResponse> {
    let mut stopped_any = false;
    if let Some(d) = state.real_evolution_daemon.lock().await.as_ref() {
        if d.is_running() {
            d.stop();
            stopped_any = true;
        }
    }
    if let Some(d) = state.evolution_daemon.lock().await.as_ref() {
        if d.is_running() {
            d.stop();
            stopped_any = true;
        }
    }
    Json(SimpleResponse {
        ok: stopped_any,
        message: if stopped_any {
            "stop signal sent".into()
        } else {
            "no daemon running".into()
        },
    })
}

// ---------------------------------------------------------------------------
// GET /api/evolution/status
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct StatusResponse {
    pub initialized: bool,
    /// "real" | "stub" | null
    pub backend: Option<String>,
    pub running: bool,
    pub current_generation: u32,
    pub population_size: usize,
    pub total_generations: u32,
    pub best_fitness: f32,
    pub uptime_secs: u64,
    /// Last generation report, normalised to the stub-daemon schema so
    /// existing UIs can render it unchanged.
    pub last_report: Option<StubGenerationReport>,
}

pub async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    // Prefer the real daemon when present.
    if let Some(d) = state.real_evolution_daemon.lock().await.as_ref().cloned() {
        let s = d.status();
        let last_report = d.get_reports().last().cloned().map(|r| StubGenerationReport {
            generation: r.generation,
            timestamp: r.timestamp,
            population_size: r.population_size,
            best_fitness: r.best_fitness,
            avg_fitness: r.avg_fitness,
            min_fitness: 0.0,
            best_id: String::new(),
            retired: Vec::new(),
            killed: Vec::new(),
            spawned: Vec::new(),
            notes: format!("real daemon: elapsed {}ms", r.elapsed_ms),
        });
        return Json(StatusResponse {
            initialized: true,
            backend: Some("real".into()),
            running: s.running,
            current_generation: s.current_generation,
            population_size: s.population_size,
            total_generations: s.total_generations,
            best_fitness: s.best_fitness,
            uptime_secs: s.uptime_secs,
            last_report,
        });
    }

    if let Some(d) = state.evolution_daemon.lock().await.as_ref().cloned() {
        let s = d.status();
        return Json(StatusResponse {
            initialized: true,
            backend: Some("stub".into()),
            running: s.running,
            current_generation: s.current_generation,
            population_size: s.population_size,
            total_generations: s.current_generation,
            best_fitness: s.best_fitness_ever,
            uptime_secs: s.uptime_secs,
            last_report: s.last_report,
        });
    }

    Json(StatusResponse {
        initialized: false,
        backend: None,
        running: false,
        current_generation: 0,
        population_size: 0,
        total_generations: 0,
        best_fitness: 0.0,
        uptime_secs: 0,
        last_report: None,
    })
}

// ---------------------------------------------------------------------------
// GET /api/evolution/history
// ---------------------------------------------------------------------------

pub async fn history(State(state): State<Arc<AppState>>) -> Json<Vec<StubGenerationReport>> {
    if let Some(d) = state.real_evolution_daemon.lock().await.as_ref().cloned() {
        let out: Vec<StubGenerationReport> = d
            .get_reports()
            .into_iter()
            .map(|r| StubGenerationReport {
                generation: r.generation,
                timestamp: r.timestamp,
                population_size: r.population_size,
                best_fitness: r.best_fitness,
                avg_fitness: r.avg_fitness,
                min_fitness: 0.0,
                best_id: String::new(),
                retired: Vec::new(),
                killed: Vec::new(),
                spawned: Vec::new(),
                notes: format!(
                    "real daemon gen {} — spawned {} retired {} killed {} ({}ms)",
                    r.generation, r.mutations_spawned, r.retired, r.killed, r.elapsed_ms
                ),
            })
            .collect();
        return Json(out);
    }
    if let Some(d) = state.evolution_daemon.lock().await.as_ref() {
        return Json(d.history());
    }
    Json(Vec::new())
}

// ---------------------------------------------------------------------------
// GET /api/evolution/specialists
// ---------------------------------------------------------------------------

pub async fn specialists(State(state): State<Arc<AppState>>) -> Json<Vec<StubSpecialistInfo>> {
    if let Some(d) = state.real_evolution_daemon.lock().await.as_ref().cloned() {
        let snaps = d.snapshot_specialists();
        let out: Vec<StubSpecialistInfo> = snaps
            .into_iter()
            .map(|s| StubSpecialistInfo {
                id: s.id,
                generation_born: s.generation_born,
                parent_id: s.parent_id,
                children: s.children,
                fitness: s.fitness,
                age: 0,
                status: match s.status.as_str() {
                    "retired" => qlang_runtime::evolution::daemon::SpecialistStatus::Retired,
                    "dead" => qlang_runtime::evolution::daemon::SpecialistStatus::Dead,
                    _ => qlang_runtime::evolution::daemon::SpecialistStatus::Active,
                },
                mutations: s.mutations,
            })
            .collect();
        return Json(out);
    }
    if let Some(d) = state.evolution_daemon.lock().await.as_ref() {
        return Json(d.specialists());
    }
    Json(Vec::new())
}

// ---------------------------------------------------------------------------
// GET /api/evolution/lineage/:id
// ---------------------------------------------------------------------------

pub async fn lineage(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<Vec<StubSpecialistInfo>> {
    if let Some(d) = state.real_evolution_daemon.lock().await.as_ref().cloned() {
        let snaps = d.lineage(&id);
        let out: Vec<StubSpecialistInfo> = snaps
            .into_iter()
            .map(|s| StubSpecialistInfo {
                id: s.id,
                generation_born: s.generation_born,
                parent_id: s.parent_id,
                children: s.children,
                fitness: s.fitness,
                age: 0,
                status: match s.status.as_str() {
                    "retired" => qlang_runtime::evolution::daemon::SpecialistStatus::Retired,
                    "dead" => qlang_runtime::evolution::daemon::SpecialistStatus::Dead,
                    _ => qlang_runtime::evolution::daemon::SpecialistStatus::Active,
                },
                mutations: s.mutations,
            })
            .collect();
        return Json(out);
    }
    if let Some(d) = state.evolution_daemon.lock().await.as_ref() {
        return Json(d.lineage(&id));
    }
    Json(Vec::new())
}

// ---------------------------------------------------------------------------
// GET /api/evolution/stream — SSE of new generation reports
// ---------------------------------------------------------------------------
//
// The real daemon does not expose a broadcast channel yet. If the stub is
// running, stream from its broadcast; otherwise fall back to a polling loop
// over the real daemon's report log.

pub async fn stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    // Stub daemon path: real broadcast channel.
    let stub_rx = {
        let mut slot = state.evolution_daemon.lock().await;
        if slot.is_none() {
            *slot = Some(StubEvolutionDaemon::new());
        }
        let _daemon: StubSharedDaemon = slot.as_ref().unwrap().clone();
        // We only subscribe if the stub daemon is actually the active backend.
        let real_active = state
            .real_evolution_daemon
            .lock()
            .await
            .as_ref()
            .map(|d| d.is_running())
            .unwrap_or(false);
        if real_active {
            None
        } else {
            Some(_daemon.subscribe())
        }
    };

    let state_for_stream = Arc::clone(&state);
    let stream = async_stream::stream! {
        if let Some(mut rx) = stub_rx {
            loop {
                match rx.recv().await {
                    Ok(report) => {
                        let data = serde_json::to_string(&report).unwrap_or_default();
                        yield Ok(Event::default().event("generation").data(data));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        } else {
            // Real-daemon polling fallback: emit new reports as they appear.
            let mut last_gen: u32 = 0;
            loop {
                let reports = match state_for_stream
                    .real_evolution_daemon
                    .lock()
                    .await
                    .as_ref()
                    .cloned()
                {
                    Some(d) => d.get_reports(),
                    None => Vec::new(),
                };
                let new_reports: Vec<_> = reports
                    .into_iter()
                    .filter(|r| r.generation > last_gen)
                    .collect();
                for r in new_reports {
                    last_gen = r.generation;
                    let data = serde_json::to_string(&r).unwrap_or_default();
                    yield Ok(Event::default().event("generation").data(data));
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}
