import { useEffect, useMemo, useState } from 'react';
import { CircleDot, Database, FileCheck2, RefreshCw, Search, ShieldCheck } from 'lucide-react';
import {
  api,
  type KnowledgeClaim,
  type KnowledgeEntity,
  type KnowledgeSnapshot,
  type KnowledgeStats,
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

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [nextStats, nextSnapshot] = await Promise.all([
          api.knowledgeStats(),
          api.knowledgeSnapshot(150),
        ]);
        if (!alive) return;
        setStats(nextStats);
        setSnapshot(nextSnapshot);
        setError(null);
      } catch (cause) {
        if (alive) setError(cause instanceof Error ? cause.message : 'Knowledge graph unavailable');
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 4000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);

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

  return (
    <section style={panel}>
      <div style={heading}>
        <div>
          <div className="eyebrow">durable knowledge graph</div>
          <div style={title}><ShieldCheck size={17} /> verified project memory</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <button onClick={() => void indexWorkspace()} disabled={indexing} style={indexButton}><RefreshCw size={12} /> {indexing ? 'scanning…' : 'scan workspace'}</button>
          <div style={loadBearing}><FileCheck2 size={14} /> {stats.load_bearing} reliable</div>
        </div>
      </div>
      {indexResult && <div style={indexResultStyle}>last scan: {indexResult}</div>}

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
            <ClaimInspector claim={selectedClaim} claims={visibleClaims} onSelect={setSelected} />
          </div>
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

function ClaimInspector({ claim, claims, onSelect }: {
  claim: KnowledgeClaim | null;
  claims: KnowledgeClaim[];
  onSelect: (id: string) => void;
}) {
  if (claim) {
    return <aside style={inspector}>
      <ClaimStatus status={claim.status} />
      <div style={claimStatement}>{claim.statement}</div>
      <div style={claimMeta}>{claim.subject}{claim.relation ? `  ${claim.relation}  ${claim.object ?? ''}` : ''}</div>
      <div style={claimMeta}>revision {claim.revision} · by {claim.provenance.producer}</div>
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
