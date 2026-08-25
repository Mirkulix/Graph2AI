use qo_knowledge::{Claim, ClaimId, Entity, EntityId, EntityKind, Evidence, EvidenceKind, KnowledgeStore, Provenance};
use serde::Serialize;
use std::{fs, path::Path, time::{SystemTime, UNIX_EPOCH}};

const MAX_FILES: usize = 5_000;
const MAX_FILE_BYTES: u64 = 1_000_000;

#[derive(Debug, Default, Serialize)]
pub struct IndexReport { pub scanned: usize, pub indexed: usize, pub already_known: usize, pub skipped: usize, pub errors: Vec<String> }

pub fn index_repository(root: &Path, store: &KnowledgeStore) -> IndexReport {
    let mut report = IndexReport::default(); visit(root, root, store, &mut report); report
}

fn visit(root: &Path, dir: &Path, store: &KnowledgeStore, report: &mut IndexReport) {
    let Ok(entries) = fs::read_dir(dir) else { report.errors.push(dir.display().to_string()); return; };
    for entry in entries.flatten() {
        if report.scanned >= MAX_FILES { return; }
        let path = entry.path();
        if path.is_dir() {
            if !matches!(path.file_name().and_then(|n| n.to_str()), Some(".git" | "target" | "node_modules" | ".projectatlas" | "dist" | "data")) { visit(root, &path, store, report); }
        } else { index_file(root, &path, store, report); }
    }
}

fn index_file(root: &Path, path: &Path, store: &KnowledgeStore, report: &mut IndexReport) {
    let Ok(meta) = fs::metadata(path) else { report.skipped += 1; return; };
    if meta.len() > MAX_FILE_BYTES || !matches!(path.extension().and_then(|e| e.to_str()), Some("rs" | "ts" | "tsx" | "js" | "jsx" | "md" | "toml" | "json" | "yaml" | "yml")) { report.skipped += 1; return; }
    let Some(relative) = path.strip_prefix(root).ok().and_then(|p| p.to_str()).map(|p| p.replace('\\', "/")) else { report.skipped += 1; return; };
    let Ok(source) = fs::read_to_string(path) else { report.skipped += 1; return; };
    report.scanned += 1;
    let entity = Entity { id: EntityId::derive(EntityKind::File, &relative), kind: EntityKind::File, name: relative.clone() };
    if store.put_entity(&entity).is_err() { report.errors.push(relative); return; }
    let lines = source.lines().count().max(1) as u32;
    let fingerprint = format!("index-v1:{:016x}", fnv1a(&source));
    let evidence = Evidence { kind: EvidenceKind::Source, locator: relative.clone(), lines: Some((1, lines)), excerpt: Some(fingerprint.clone()), supports: true };
    let provenance = Provenance { producer: "repository-indexer".into(), observed_at: now_secs(), git_revision: None, run_id: None };
    let claim_id = ClaimId(format!("index:file:{relative}"));
    if let Ok(Some(previous)) = store.latest(&claim_id) {
        let changed = previous.evidence.last().and_then(|e| e.excerpt.as_deref()) != Some(fingerprint.as_str());
        if changed {
            let _ = store.mark_stale(&claim_id, provenance.clone());
            match store.refresh_observed(&claim_id, evidence, provenance) { Ok(_) => report.indexed += 1, Err(error) => report.errors.push(error.to_string()) }
        } else { report.already_known += 1; }
        return;
    }
    let claim = Claim::observed(claim_id.0, format!("Repository contains indexed file `{relative}` ({lines} lines)."), entity.id, provenance, vec![evidence]);
    match claim.and_then(|claim| store.add_claim(&claim)) {
        Ok(()) => report.indexed += 1,
        Err(qo_knowledge::Error::ClaimExists(_)) => report.already_known += 1,
        Err(error) => report.errors.push(error.to_string()),
    }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

fn fnv1a(source: &str) -> u64 {
    source.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3))
}
