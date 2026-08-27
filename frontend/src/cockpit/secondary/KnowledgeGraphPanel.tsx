import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CircleDot, Database, Download, FileCheck2, Receipt, RefreshCw, Search, ShieldCheck, Upload } from 'lucide-react';
import {
  api,
  type KnowledgeBackupEntry,
  type KnowledgeClaim,
  type KnowledgeDivergence,
  type KnowledgeEntity,
  type KnowledgeEvent,
  type KnowledgeHealth,
  type KnowledgeReceipt,
  type KnowledgeSnapshot,
  type KnowledgeStats,
  type KnowledgeVerifySourceResult,
} from '../../lib/api';

type Status = KnowledgeClaim['status'];

const ALL_STATUSES: Status[] = ['verified', 'observed', 'proposed', 'stale', 'refuted'];
const STATUS_COLOR: Record<string, string> = {
  verified: 'var(--ok)',
  observed: 'var(--accent)',
  proposed: 'var(--warn)',
  stale: 'var(--ink-muted)',
  refuted: 'var(--danger)',
};

const EMPTY_STATS: KnowledgeStats = {
  verified: 0, observed: 0, proposed: 0, stale: 0, refuted: 0,
  load_bearing: 0, total: 0, entities: 0,
};

export function KnowledgeGraphPanel() {
  const [stats, setStats] = useState<KnowledgeStats>(EMPTY_STATS);
  const [snapshot, setSnapshot] = useState<KnowledgeSnapshot>({ entities: [], claims: [] });
  const [activeStatuses, setActiveStatuses] = useState<Set<Status>>(new Set(ALL_STATUSES));
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [indexResult, setIndexResult] = useState<string | null>(null);
  const [sweeping, setSweeping] = useState(false);
  const [sweepResult, setSweepResult] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshResult, setRefreshResult] = useState<string | null>(null);
  const [healing, setHealing] = useState(false);
  const [healResult, setHealResult] = useState<string | null>(null);
  const [divergences, setDivergences] = useState<KnowledgeDivergence[]>([]);
  const [divergencesOpen, setDivergencesOpen] = useState(false);
  const [health, setHealth] = useState<KnowledgeHealth | null>(null);
  const [backups, setBackups] = useState<KnowledgeBackupEntry[]>([]);
  const [events, setEvents] = useState<KnowledgeEvent[]>([]);
  const [backingUp, setBackingUp] = useState(false);
  const [backupResult, setBackupResult] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);
  const [restoreResult, setRestoreResult] = useState<string | null>(null);
  const [proposeOpen, setProposeOpen] = useState(false);
  const [proposeDoc, setProposeDoc] = useState('');
  const [proposing, setProposing] = useState(false);
  const [proposeResult, setProposeResult] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importResult, setImportResult] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextStats, nextSnapshot, nextDivergences, nextHealth, nextBackups, nextEvents] = await Promise.all([
        api.knowledgeStats(),
        api.knowledgeSnapshot(150),
        api.knowledgeDivergences().catch(() => ({ divergences: [] as KnowledgeDivergence[] })),
        api.knowledgeHealth().catch(() => null),
        api.knowledgeBackups().catch(() => ({ backups: [] as KnowledgeBackupEntry[] })),
        api.knowledgeEvents(15).catch(() => [] as KnowledgeEvent[]),
      ]);
      setStats(nextStats);
      setSnapshot(nextSnapshot);
      setDivergences(nextDivergences.divergences ?? []);
      setHealth(nextHealth);
      setBackups(nextBackups.backups ?? []);
      setEvents(
        (nextEvents ?? [])
          .filter((e) => e.action_type.startsWith('knowledge_') || e.action_type.startsWith('orbit_graph_'))
          .slice(0, 8),
      );
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Knowledge graph unavailable');
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const visibleClaims = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return snapshot.claims.filter((claim) =>
      activeStatuses.has(claim.status)
      && (!needle || [claim.statement, claim.subject, claim.object ?? '', claim.id]
        .some((value) => value.toLowerCase().includes(needle))));
  }, [activeStatuses, query, snapshot.claims]);

  const graphEntities = useMemo(() => {
    const used = new Set<string>();
    for (const claim of visibleClaims) {
      used.add(claim.subject);
      if (claim.object) used.add(claim.object);
    }
    return snapshot.entities.filter((entity) => used.has(entity.id));
  }, [snapshot.entities, visibleClaims]);

  const selectedClaim = visibleClaims.find((claim) => claim.id === selected) ?? null;

  function toggleStatus(status: Status) {
    setActiveStatuses((current) => {
      const next = new Set(current);
      if (next.has(status)) next.delete(status); else next.add(status);
      return next;
    });
  }

  async function indexWorkspace() {
    setIndexing(true);
    try {
      const result = await api.knowledgeIndex();
      setIndexResult(`${result.indexed} new · ${result.already_known} unchanged · ${result.errors.length} errors`);
      const [nextStats, nextSnapshot] = await Promise.all([api.knowledgeStats(), api.knowledgeSnapshot(150)]);
      setStats(nextStats); setSnapshot(nextSnapshot); setError(null);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Indexing failed'); }
    finally { setIndexing(false); }
  }

  async function sweepProposals() {
    setSweeping(true);
    try {
      const result = await api.knowledgeVerifyAll();
      setSweepResult(`${result.verified} verified · ${result.inconclusive} inconclusive · ${result.unavailable} unavailable (of ${result.checked})`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Sweep failed'); }
    finally { setSweeping(false); }
  }

  async function refreshSources() {
    setRefreshing(true);
    try {
      const result = await api.knowledgeRefreshSources();
      setRefreshResult(`${result.stale} stale · ${result.still_current} current (of ${result.checked})`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Source refresh failed'); }
    finally { setRefreshing(false); }
  }

  async function healStale() {
    setHealing(true);
    try {
      const result = await api.knowledgeHealStale();
      setHealResult(`${result.healed} healed · ${result.remained_stale} still stale (of ${result.examined})`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Heal failed'); }
    finally { setHealing(false); }
  }

  async function backupNow() {
    setBackingUp(true);
    try {
      const result = await api.knowledgeBackup();
      setBackupResult(`written: ${result.path.split(/[\\/]/).pop()}`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Backup failed'); }
    finally { setBackingUp(false); }
  }

  async function restoreLatest() {
    if (backups.length === 0) { setRestoreResult('no backups on the server yet'); return; }
    setRestoring(true);
    try {
      const result = await api.knowledgeRestore();
      setRestoreResult(
        `restored ${result.claims_added} claim(s) from ${result.restored_from.split(/[\\/]/).pop()} · ${result.claims_skipped.length} skipped (additive)`,
      );
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Restore failed'); }
    finally { setRestoring(false); }
  }

  async function proposeDocument() {
    if (!proposeDoc.trim()) return;
    setProposing(true);
    try {
      const result = await api.knowledgePropose(proposeDoc);
      setProposeResult(`accepted: ${result.applied} operation(s) from ${result.delta_id} (all claims proposed)`);
      setProposeDoc('');
      await refresh();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Propose failed';
      setProposeResult(`rejected: ${message.length > 320 ? `${message.slice(0, 319)}…` : message}`);
    } finally { setProposing(false); }
  }

  async function exportGraph() {
    setExporting(true);
    try {
      const archive = await api.knowledgeExport();
      const entities = Array.isArray(archive.entities) ? archive.entities.length : 0;
      const claims = Array.isArray(archive.claims) ? archive.claims.length : 0;
      const blob = new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `knowledge-graph-${Date.now()}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      setExportResult(`exported ${entities} entities, ${claims} revisions (${blob.size} bytes) — saved as download`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Export failed'); }
    finally { setExporting(false); }
  }

  async function importGraph(file: File | undefined) {
    if (!file) return;
    setImporting(true);
    try {
      const text = await file.text();
      const archive = JSON.parse(text) as Record<string, unknown>;
      const result = await api.knowledgeImport(archive);
      setImportResult(`imported ${result.entities_added} entities, ${result.claims_added} claims · ${result.claims_skipped.length} skipped (additive)`);
      await refresh();
    } catch (cause) {
      setImportResult(cause instanceof Error ? cause.message : 'Import failed');
    } finally { setImporting(false); }
  }

  return (
    <section style={panel}>
      <div style={heading}>
        <div>
          <div className="eyebrow">durable knowledge graph</div>
          <div style={title}><ShieldCheck size={17} /> verified project memory</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap' }}>
          <button onClick={() => void sweepProposals()} disabled={sweeping} style={indexButton}><ShieldCheck size={12} /> {sweeping ? 'sweeping…' : 'sweep proposals'}</button>
          <button onClick={() => void refreshSources()} disabled={refreshing} style={indexButton}><RefreshCw size={12} /> {refreshing ? 'refreshing…' : 'refresh sources'}</button>
          <button onClick={() => void healStale()} disabled={healing} style={indexButton}><ShieldCheck size={12} /> {healing ? 'healing…' : 'heal stale'}</button>
          <button onClick={() => void indexWorkspace()} disabled={indexing} style={indexButton}><RefreshCw size={12} /> {indexing ? 'scanning…' : 'scan workspace'}</button>
          <button onClick={() => void backupNow()} disabled={backingUp} style={indexButton}><Database size={12} /> {backingUp ? 'backing up…' : 'backup'}</button>
          <button onClick={() => void restoreLatest()} disabled={restoring} style={indexButton}><FileCheck2 size={12} /> {restoring ? 'restoring…' : 'restore'}</button>
          <button onClick={() => void exportGraph()} disabled={exporting} style={indexButton}><Download size={12} /> {exporting ? 'exporting…' : 'export'}</button>
          <button onClick={() => fileInputRef.current?.click()} disabled={importing} style={indexButton}><Upload size={12} /> {importing ? 'importing…' : 'import'}</button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json,.json"
            style={{ display: 'none' }}
            onChange={(event) => { void importGraph(event.target.files?.[0]); event.target.value = ''; }}
          />
          <div style={loadBearing}><FileCheck2 size={14} /> {stats.load_bearing} reliable</div>
        </div>
      </div>
      {indexResult && <div style={indexResultStyle}>last scan: {indexResult}</div>}
      {sweepResult && <div style={indexResultStyle}>sweep: {sweepResult}</div>}
      {refreshResult && <div style={indexResultStyle}>source refresh: {refreshResult}</div>}
      {healResult && <div style={indexResultStyle}>heal: {healResult}</div>}
      {backupResult && <div style={indexResultStyle}>backup: {backupResult}</div>}
      {restoreResult && <div style={indexResultStyle}>restore: {restoreResult}</div>}
      {exportResult && <div style={indexResultStyle}>export: {exportResult}</div>}
      {importResult && <div style={indexResultStyle}>import: {importResult}</div>}
      {health && (
        <div style={healthLine}>
          health: {health.load_bearing} load-bearing · {health.proposed} proposals · {health.stale} stale · {health.refuted} refuted · {health.divergences} divergences · {health.entities} entities
        </div>
      )}
      <div style={{ marginBottom: 10 }}>
        <button onClick={() => setProposeOpen((o) => !o)} style={indexButton}><CircleDot size={12} /> {proposeOpen ? 'hide propose' : 'propose document'}</button>
        {proposeOpen && (
          <div style={{ marginTop: 8 }}>
            <textarea
              value={proposeDoc}
              onChange={(e) => setProposeDoc(e.target.value)}
              placeholder={'DELTA|1|d-1\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|auth uses bcrypt'}
              spellCheck={false}
              style={proposeTextarea}
            />
            <div style={actionRow}>
              <button onClick={() => void proposeDocument()} disabled={proposing} style={actionButton}><CircleDot size={12} /> {proposing ? 'submitting…' : 'submit proposal'}</button>
            </div>
            {proposeResult && <div style={indexResultStyle}>{proposeResult}</div>}
          </div>
        )}
      </div>
      {divergences.length > 0 && (
        <div style={divergenceBox}>
          <button onClick={() => setDivergencesOpen((o) => !o)} style={divergenceHeader}>
            <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{divergences.length} divergent subject(s)</span>
            <span style={{ color: 'var(--ink-faint)' }}>{divergencesOpen ? 'hide' : 'show'}</span>
          </button>
          {divergencesOpen && divergences.map((d) => (
            <div key={d.subject} style={divergenceRow}>
              <div style={{ ...divergenceMeta, color: 'var(--ink-bright)' }}>{d.subject}</div>
              <div style={divergenceMeta}>
                <span style={{ color: 'var(--ok)' }}>+{d.load_bearing.length} {d.load_bearing.map((c) => c.statement).join(' · ')}</span>
              </div>
              <div style={divergenceMeta}>
                <span style={{ color: 'var(--danger)' }}>−{d.refuted.length} {d.refuted.map((c) => c.statement).join(' · ')}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={metrics}>
        <Metric label="entities" value={stats.entities} icon={<CircleDot size={13} />} />
        <Metric label="claims" value={stats.total} icon={<Database size={13} />} />
        <Metric label="verified" value={stats.verified + stats.observed} tone="var(--ok)" icon={<ShieldCheck size={13} />} />
        <Metric label="proposals" value={stats.proposed} tone="var(--warn)" icon={<CircleDot size={13} />} />
      </div>

      {error ? (
        <div style={errorBox}>Graphdaten können noch nicht geladen werden: {error}</div>
      ) : (
        <>
          <div style={controls}>
            <div style={statusFilters}>
              {ALL_STATUSES.map((status) => (
                <button
                  key={status}
                  onClick={() => toggleStatus(status)}
                  style={{ ...statusButton, opacity: activeStatuses.has(status) ? 1 : 0.38, borderColor: STATUS_COLOR[status] }}
                >
                  <span style={{ ...statusDot, background: STATUS_COLOR[status] }} /> {status}
                </button>
              ))}
            </div>
            <label style={searchBox}>
              <Search size={13} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="claims, entities, ids…" style={searchInput} />
            </label>
          </div>

          <div style={contentGrid}>
            <GraphMap entities={graphEntities} claims={visibleClaims} selectedId={selected} onSelect={setSelected} />
            <ClaimInspector key={selectedClaim?.id ?? 'none'} claim={selectedClaim} claims={visibleClaims} onSelect={setSelected} onChanged={refresh} />
          </div>
          {events.length > 0 && (
            <div style={eventsFeed}>
              <div style={eventsTitle}>knowledge events (recent)</div>
              {events.map((event) => (
                <div key={event.id} style={eventsRow}>
                  <span style={{ color: 'var(--ink-faint)' }}>{new Date(event.timestamp * 1000).toLocaleTimeString()}</span>
                  <span style={{ color: 'var(--accent)' }}>{event.action_type.replace('knowledge_', '')}</span>
                  <span style={{ color: 'var(--ink-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{event.description}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function GraphMap({ entities, claims, selectedId, onSelect }: {
  entities: KnowledgeEntity[];
  claims: KnowledgeClaim[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const position = useMemo(() => new Map(entities.map((entity, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(entities.length, 1) - Math.PI / 2;
    return [entity.id, { x: 220 + Math.cos(angle) * 145, y: 150 + Math.sin(angle) * 95 }];
  })), [entities]);

  if (entities.length === 0) {
    return <div style={graphEmpty}>Noch keine Claims im ausgewählten Filter. Claude kann beim Prüfen von Code Claims mit Evidenz speichern.</div>;
  }

  return (
    <div style={graphWrap}>
      <svg viewBox="0 0 440 300" role="img" aria-label="Knowledge graph" style={graphSvg}>
        <defs>
          <marker id="knowledge-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="var(--ink-faint)" />
          </marker>
        </defs>
        {claims.filter((claim) => claim.object && position.has(claim.subject) && position.has(claim.object)).map((claim) => {
          const from = position.get(claim.subject)!;
          const to = position.get(claim.object!)!;
          const isSelected = claim.id === selectedId;
          return <g key={claim.id} onClick={() => onSelect(claim.id)} style={{ cursor: 'pointer' }}>
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={STATUS_COLOR[claim.status]} strokeWidth={isSelected ? 3 : 1.3} opacity={isSelected ? 1 : 0.65} markerEnd="url(#knowledge-arrow)" />
            <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5} textAnchor="middle" fill="var(--ink-muted)" fontSize="8" fontFamily="var(--font-mono)">{claim.relation ?? 'claim'}</text>
          </g>;
        })}
        {entities.map((entity) => {
          const point = position.get(entity.id)!;
          return <g key={entity.id} transform={`translate(${point.x}, ${point.y})`}>
            <circle r="27" fill="var(--bg-raised)" stroke="var(--accent)" strokeWidth="1.5" />
            <text y="-3" textAnchor="middle" fill="var(--ink-bright)" fontSize="9" fontFamily="var(--font-mono)">{short(entity.name, 16)}</text>
            <text y="10" textAnchor="middle" fill="var(--ink-faint)" fontSize="7" fontFamily="var(--font-mono)">{entity.kind}</text>
          </g>;
        })}
      </svg>
      <div style={graphLegend}><span style={{ color: 'var(--ink-faint)' }}>{entities.length} entities</span><span style={{ color: 'var(--ink-faint)' }}>{claims.length} claims</span></div>
    </div>
  );
}

function ClaimInspector({ claim, claims, onSelect, onChanged }: {
  claim: KnowledgeClaim | null;
  claims: KnowledgeClaim[];
  onSelect: (id: string) => void;
  onChanged: () => Promise<void>;
}) {
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<KnowledgeVerifySourceResult | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<KnowledgeReceipt | null>(null);
  const [receiptLoading, setReceiptLoading] = useState(false);

  async function verifySource() {
    if (!claim) return;
    setVerifying(true);
    setVerifyError(null);
    setVerifyResult(null);
    try {
      const result = await api.knowledgeVerifySource(claim.id);
      setVerifyResult(result);
      // The claim's status may have changed to `verified`; refresh the graph.
      await onChanged();
    } catch (cause) {
      setVerifyError(cause instanceof Error ? cause.message : 'Source check failed');
    } finally {
      setVerifying(false);
    }
  }

  async function showReceipt() {
    if (!claim) return;
    setReceiptLoading(true);
    setVerifyError(null);
    try {
      setReceipt(await api.knowledgeReceipt(claim.id));
    } catch (cause) {
      setVerifyError(cause instanceof Error ? cause.message : 'Receipt failed');
    } finally {
      setReceiptLoading(false);
    }
  }

  if (claim) {
    return <aside style={inspector}>
      <ClaimStatus status={claim.status} />
      <div style={claimStatement}>{claim.statement}</div>
      <div style={claimMeta}>{claim.subject}{claim.relation ? `  ${claim.relation}  ${claim.object ?? ''}` : ''}</div>
      <div style={claimMeta}>revision {claim.revision} · by {claim.provenance.producer}</div>

      <div style={actionRow}>
        {claim.status === 'proposed' && (
          <button onClick={() => void verifySource()} disabled={verifying} style={actionButton}>
            <ShieldCheck size={12} /> {verifying ? 'checking source…' : 'check against source'}
          </button>
        )}
        <button onClick={() => void showReceipt()} disabled={receiptLoading} style={actionButton}>
          <Receipt size={12} /> {receiptLoading ? 'loading…' : 'proof receipt'}
        </button>
      </div>

      {verifyError && <div style={verifyErrorStyle}>{verifyError}</div>}
      {verifyResult && <VerifyResultLine result={verifyResult} />}
      {receipt && (
        <div style={{ marginTop: 10 }}>
          <div className="eyebrow" style={{ marginBottom: 4 }}>proof receipt</div>
          <pre style={receiptPre}>{receipt.rendered}</pre>
        </div>
      )}

      <div className="eyebrow" style={{ marginTop: 12 }}>evidence</div>
      {claim.evidence.length === 0 ? <div style={muted}>Noch keine Evidenz — dieser Claim ist nicht belastbar.</div> : claim.evidence.map((evidence, index) => (
        <div key={`${evidence.locator}-${index}`} style={evidenceCard}>
          <div style={{ color: evidence.supports ? 'var(--ok)' : 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>{evidence.kind} · {evidence.supports ? 'supports' : 'refutes'}</div>
          <div style={locator}>{evidence.locator}{evidence.lines ? `:${evidence.lines[0]}-${evidence.lines[1]}` : ''}</div>
          {evidence.excerpt && <div style={excerpt}>{evidence.excerpt}</div>}
        </div>
      ))}
    </aside>;
  }

  return <aside style={inspector}>
    <div className="eyebrow">claims</div>
    {claims.slice(0, 7).map((item) => (
      <button key={item.id} onClick={() => onSelect(item.id)} style={claimRow}>
        <ClaimStatus status={item.status} />
        <span style={claimRowText}>{item.statement}</span>
      </button>
    ))}
    {claims.length === 0 && <div style={muted}>Keine Claims gefunden.</div>}
  </aside>;
}

function VerifyResultLine({ result }: { result: KnowledgeVerifySourceResult }) {
  const kind = result.verdict.kind;
  const color = kind === 'verified' ? 'var(--ok)' : kind === 'inconclusive' ? 'var(--warn)' : 'var(--ink-muted)';
  const detail = kind === 'verified'
    ? `all ${result.matched} term(s) present: ${result.terms.join(', ')}`
    : kind === 'inconclusive'
      ? (result.verdict.reason ?? `${result.matched}/${result.terms.length} terms present`)
      : kind === 'not_proposed'
        ? `already ${result.verdict.status ?? 'settled'} — not re-promoted`
        : (result.verdict.reason ?? 'unavailable');
  return (
    <div style={{ ...verifyResultStyle, color }}>
      <ShieldCheck size={12} /> {kind === 'verified' ? 'verified against source' : kind} — {detail}
    </div>
  );
}

function ClaimStatus({ status }: { status: Status }) {
  return <span style={{ ...claimStatus, color: STATUS_COLOR[status], borderColor: STATUS_COLOR[status] }}>{status}</span>;
}

function Metric({ label, value, icon, tone }: { label: string; value: number; icon: React.ReactNode; tone?: string }) {
  return <div style={metric}><div style={{ ...metricLabel, color: tone ?? 'var(--ink-muted)' }}>{icon} {label}</div><div style={{ ...metricValue, color: tone ?? 'var(--ink-bright)' }}>{value}</div></div>;
}

function short(value: string, limit: number) { return value.length > limit ? `${value.slice(0, limit - 1)}…` : value; }

const panel: React.CSSProperties = { padding: 14, marginBottom: 18, border: '1px solid var(--rule-default)', borderRadius: 5, background: 'var(--bg-panel)' };
const heading: React.CSSProperties = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 };
const title: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 7, color: 'var(--ink-bright)', fontSize: 15, fontFamily: 'var(--font-mono)' };
const loadBearing: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 5, padding: '4px 7px', color: 'var(--ok)', border: '1px solid var(--ok)', borderRadius: 3, fontSize: 10, fontFamily: 'var(--font-mono)' };
const metrics: React.CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 7, marginBottom: 12 };
const metric: React.CSSProperties = { padding: '8px 10px', background: 'var(--bg-raised)', borderRadius: 3 };
const metricLabel: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, textTransform: 'uppercase' };
const metricValue: React.CSSProperties = { fontFamily: 'var(--font-mono)', fontSize: 19, marginTop: 4 };
const controls: React.CSSProperties = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', marginBottom: 10 };
const statusFilters: React.CSSProperties = { display: 'flex', gap: 4, flexWrap: 'wrap' };
const statusButton: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, padding: '3px 6px', border: '1px solid', background: 'transparent', color: 'var(--ink-muted)', borderRadius: 3, cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 9 };
const statusDot: React.CSSProperties = { width: 6, height: 6, borderRadius: 99 };
const searchBox: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, padding: '5px 7px', minWidth: 180, border: '1px solid var(--rule-default)', borderRadius: 3, color: 'var(--ink-faint)' };
const searchInput: React.CSSProperties = { width: '100%', background: 'transparent', border: 0, outline: 0, color: 'var(--ink-bright)', fontFamily: 'var(--font-mono)', fontSize: 10 };
const contentGrid: React.CSSProperties = { display: 'grid', gridTemplateColumns: 'minmax(0, 1.45fr) minmax(210px, .8fr)', gap: 10 };
const graphWrap: React.CSSProperties = { minHeight: 300, border: '1px solid var(--rule-faint)', borderRadius: 3, overflow: 'hidden', background: 'var(--bg-void)' };
const graphSvg: React.CSSProperties = { display: 'block', width: '100%', height: 285 };
const graphLegend: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', padding: '5px 8px', borderTop: '1px solid var(--rule-faint)', fontFamily: 'var(--font-mono)', fontSize: 9 };
const graphEmpty: React.CSSProperties = { minHeight: 178, display: 'grid', placeItems: 'center', padding: 22, color: 'var(--ink-faint)', textAlign: 'center', fontSize: 11, border: '1px dashed var(--rule-default)', borderRadius: 3 };
const inspector: React.CSSProperties = { padding: 10, minWidth: 0, border: '1px solid var(--rule-faint)', borderRadius: 3, background: 'var(--bg-raised)' };
const claimStatus: React.CSSProperties = { display: 'inline-block', padding: '2px 5px', border: '1px solid', borderRadius: 2, fontSize: 9, fontFamily: 'var(--font-mono)', textTransform: 'uppercase' };
const claimStatement: React.CSSProperties = { marginTop: 8, color: 'var(--ink-bright)', fontSize: 12, lineHeight: 1.45 };
const claimMeta: React.CSSProperties = { marginTop: 6, color: 'var(--ink-muted)', fontSize: 9, fontFamily: 'var(--font-mono)', overflowWrap: 'anywhere' };
const evidenceCard: React.CSSProperties = { marginTop: 6, padding: 7, border: '1px solid var(--rule-faint)', borderRadius: 3 };
const locator: React.CSSProperties = { marginTop: 3, color: 'var(--ink-bright)', fontSize: 10, fontFamily: 'var(--font-mono)', overflowWrap: 'anywhere' };
const excerpt: React.CSSProperties = { marginTop: 4, color: 'var(--ink-muted)', fontSize: 10, lineHeight: 1.35 };
const muted: React.CSSProperties = { marginTop: 8, color: 'var(--ink-faint)', fontSize: 10, lineHeight: 1.4 };
const claimRow: React.CSSProperties = { width: '100%', display: 'flex', alignItems: 'flex-start', gap: 6, padding: '7px 0', background: 'transparent', border: 0, borderBottom: '1px solid var(--rule-faint)', cursor: 'pointer', textAlign: 'left' };
const claimRowText: React.CSSProperties = { color: 'var(--ink-muted)', fontSize: 10, lineHeight: 1.35 };
const errorBox: React.CSSProperties = { padding: 10, border: '1px solid var(--danger)', borderRadius: 3, color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 10 };
const indexButton: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, padding: '4px 7px', color: 'var(--ink-bright)', background: 'var(--bg-raised)', border: '1px solid var(--rule-default)', borderRadius: 3, cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 10 };
const indexResultStyle: React.CSSProperties = { marginBottom: 9, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', fontSize: 10 };
const actionRow: React.CSSProperties = { display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' };
const actionButton: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, padding: '4px 7px', color: 'var(--ink-bright)', background: 'var(--bg-raised)', border: '1px solid var(--rule-default)', borderRadius: 3, cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 10 };
const verifyErrorStyle: React.CSSProperties = { marginTop: 8, padding: 6, border: '1px solid var(--danger)', borderRadius: 3, color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 9, overflowWrap: 'anywhere' };
const verifyResultStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 5, marginTop: 8, padding: 6, border: '1px solid var(--rule-faint)', borderRadius: 3, fontFamily: 'var(--font-mono)', fontSize: 9, lineHeight: 1.4 };
const receiptPre: React.CSSProperties = { fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-muted)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0, padding: 8, background: 'var(--bg-void)', borderRadius: 3, lineHeight: 1.5, maxHeight: 340, overflowY: 'auto' };
const divergenceBox: React.CSSProperties = { marginBottom: 12, border: '1px solid var(--danger)', borderRadius: 3, background: 'var(--bg-raised)', overflow: 'hidden' };
const divergenceHeader: React.CSSProperties = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', padding: '7px 10px', background: 'transparent', border: 0, cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 10 };
const divergenceRow: React.CSSProperties = { padding: '6px 10px', borderTop: '1px solid var(--rule-faint)' };
const divergenceMeta: React.CSSProperties = { marginTop: 3, fontFamily: 'var(--font-mono)', fontSize: 9, overflowWrap: 'anywhere', lineHeight: 1.4 };
const healthLine: React.CSSProperties = { marginBottom: 10, padding: '7px 9px', color: 'var(--ok)', border: '1px solid var(--ok)', borderRadius: 3, fontFamily: 'var(--font-mono)', fontSize: 10 };
const proposeTextarea: React.CSSProperties = { width: '100%', minHeight: 96, padding: 8, background: 'var(--bg-void)', color: 'var(--ink-bright)', border: '1px solid var(--rule-default)', borderRadius: 3, fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: 1.5, outline: 0 };
const eventsFeed: React.CSSProperties = { marginTop: 12, padding: 8, border: '1px solid var(--rule-faint)', borderRadius: 3, background: 'var(--bg-raised)' };
const eventsTitle: React.CSSProperties = { fontFamily: 'var(--font-mono)', fontSize: 9, textTransform: 'uppercase', color: 'var(--ink-faint)', marginBottom: 5 };
const eventsRow: React.CSSProperties = { display: 'flex', gap: 8, alignItems: 'baseline', padding: '2px 0', fontFamily: 'var(--font-mono)', fontSize: 9, minWidth: 0 };
