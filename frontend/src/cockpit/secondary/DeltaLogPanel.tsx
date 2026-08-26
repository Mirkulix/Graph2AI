// Live feed of merged OrbitQLang deltas, with a conflict view.
//
// The graph never overwrites a decision silently, which means a conflict is
// something a human has to look at. This panel exists so those do not stay
// buried in a merge report nobody reads: conflicts are counted in the header,
// filterable in one click, and each one names the claim and both sides.

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronRight, GitCommitHorizontal, Layers } from 'lucide-react';
import { api, type KnowledgeConflict, type KnowledgeDelta } from '../../lib/api';

const POLL_INTERVAL_MS = 4000;

const CONFLICT_LABEL: Record<string, string> = {
  unknown_claim: 'unknown claim',
  duplicate_claim_id: 'duplicate id',
  contradictory_status: 'contradiction',
  stale_source_revision: 'stale revision',
  rejected: 'rejected',
};

export function DeltaLogPanel() {
  const [deltas, setDeltas] = useState<KnowledgeDelta[]>([]);
  const [totalConflicts, setTotalConflicts] = useState(0);
  const [conflictsOnly, setConflictsOnly] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const log = await api.knowledgeDeltas(50, conflictsOnly);
        if (!alive) return;
        setDeltas(log.deltas);
        setTotalConflicts(log.unresolved_conflicts);
        setError(null);
      } catch (cause) {
        if (alive) setError(cause instanceof Error ? cause.message : 'Delta log unavailable');
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [conflictsOnly]);

  const applied = useMemo(
    () => deltas.reduce((sum, delta) => sum + delta.applied, 0),
    [deltas],
  );

  function toggle(id: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  return (
    <section style={panel}>
      <div style={heading}>
        <div>
          <div className="eyebrow">agent-to-agent sync</div>
          <div style={title}><GitCommitHorizontal size={17} /> graph deltas</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <button
            onClick={() => setConflictsOnly((value) => !value)}
            style={{
              ...filterButton,
              color: conflictsOnly ? 'var(--danger)' : 'var(--ink-muted)',
              borderColor: conflictsOnly ? 'var(--danger)' : 'var(--rule-default)',
            }}
          >
            <AlertTriangle size={12} /> conflicts only
          </button>
          <div style={{ ...badge, color: totalConflicts > 0 ? 'var(--danger)' : 'var(--ok)', borderColor: totalConflicts > 0 ? 'var(--danger)' : 'var(--ok)' }}>
            {totalConflicts} conflict{totalConflicts === 1 ? '' : 's'}
          </div>
        </div>
      </div>

      {error && <div style={errorBox}>{error}</div>}

      {!error && deltas.length === 0 && (
        <div style={empty}>
          {conflictsOnly
            ? 'No conflicts. Every submitted delta merged cleanly.'
            : 'No deltas yet. Worker sessions submit findings via orbit_graph_commit_delta.'}
        </div>
      )}

      {deltas.length > 0 && (
        <>
          <div style={summary}>
            <span><Layers size={11} /> {deltas.length} deltas</span>
            <span>{applied} operations applied</span>
          </div>

          <div>
            {deltas.map((delta) => {
              const isOpen = expanded.has(delta.delta_id);
              const hasConflicts = delta.conflicts.length > 0;
              return (
                <div key={delta.delta_id} style={row}>
                  <button onClick={() => toggle(delta.delta_id)} style={rowHeader}>
                    {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    <span style={{ ...deltaId, color: hasConflicts ? 'var(--danger)' : 'var(--ink-bright)' }}>
                      {delta.delta_id}
                    </span>
                    <span style={producer}>{delta.producer}</span>
                    <span style={counts}>
                      +{delta.applied}
                      {delta.already_applied > 0 && <span style={{ color: 'var(--ink-faint)' }}> ·{delta.already_applied} known</span>}
                      {hasConflicts && <span style={{ color: 'var(--danger)' }}> ·{delta.conflicts.length} conflict</span>}
                    </span>
                    <span style={when}>{formatTime(delta.emitted_at)}</span>
                  </button>

                  {hasConflicts && (
                    <div style={conflictList}>
                      {delta.conflicts.map((conflict, index) => (
                        <ConflictRow key={`${delta.delta_id}-${index}`} conflict={conflict} />
                      ))}
                    </div>
                  )}

                  {isOpen && (
                    <pre style={document}>{delta.document.trimEnd()}</pre>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

function ConflictRow({ conflict }: { conflict: KnowledgeConflict }) {
  return (
    <div style={conflictCard}>
      <div style={conflictKind}>
        <AlertTriangle size={11} /> {CONFLICT_LABEL[conflict.kind] ?? conflict.kind}
        {conflict.claim_id && <span style={{ color: 'var(--ink-faint)' }}> · {conflict.claim_id}</span>}
      </div>
      <div style={conflictDetail}>{conflict.detail}</div>
    </div>
  );
}

// Deltas carry a producer-supplied unix timestamp; a worker with a wrong clock
// should not render as "in 3 hours", so future stamps clamp to "just now".
function formatTime(unixSeconds: number): string {
  const deltaMs = Date.now() - unixSeconds * 1000;
  if (deltaMs < 60_000) return 'just now';
  const minutes = Math.floor(deltaMs / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const panel: React.CSSProperties = { padding: 14, marginBottom: 18, border: '1px solid var(--rule-default)', borderRadius: 5, background: 'var(--bg-panel)' };
const heading: React.CSSProperties = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 };
const title: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 7, color: 'var(--ink-bright)', fontSize: 15, fontFamily: 'var(--font-mono)' };
const badge: React.CSSProperties = { padding: '4px 7px', border: '1px solid', borderRadius: 3, fontSize: 10, fontFamily: 'var(--font-mono)' };
const filterButton: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, padding: '4px 7px', background: 'transparent', border: '1px solid', borderRadius: 3, cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 10 };
const summary: React.CSSProperties = { display: 'flex', gap: 14, marginBottom: 8, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', fontSize: 9, textTransform: 'uppercase' };
const row: React.CSSProperties = { borderBottom: '1px solid var(--rule-faint)' };
const rowHeader: React.CSSProperties = { width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0', background: 'transparent', border: 0, color: 'var(--ink-muted)', cursor: 'pointer', textAlign: 'left', fontFamily: 'var(--font-mono)', fontSize: 10 };
const deltaId: React.CSSProperties = { minWidth: 70, fontFamily: 'var(--font-mono)', fontSize: 11 };
const producer: React.CSSProperties = { flex: 1, minWidth: 0, color: 'var(--accent)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' };
const counts: React.CSSProperties = { color: 'var(--ok)', fontFamily: 'var(--font-mono)', fontSize: 10 };
const when: React.CSSProperties = { minWidth: 62, color: 'var(--ink-faint)', textAlign: 'right', fontSize: 9 };
const conflictList: React.CSSProperties = { paddingBottom: 8 };
const conflictCard: React.CSSProperties = { marginTop: 4, padding: 7, border: '1px solid var(--danger)', borderRadius: 3, background: 'var(--bg-raised)' };
const conflictKind: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4, color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 9, textTransform: 'uppercase' };
const conflictDetail: React.CSSProperties = { marginTop: 4, color: 'var(--ink-muted)', fontSize: 10, lineHeight: 1.4 };
const document: React.CSSProperties = { margin: '0 0 9px', padding: 9, maxHeight: 220, overflow: 'auto', border: '1px solid var(--rule-faint)', borderRadius: 3, background: 'var(--bg-void)', color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: 1.5, whiteSpace: 'pre' };
const empty: React.CSSProperties = { padding: 22, color: 'var(--ink-faint)', textAlign: 'center', fontSize: 11, border: '1px dashed var(--rule-default)', borderRadius: 3 };
const errorBox: React.CSSProperties = { padding: 10, border: '1px solid var(--danger)', borderRadius: 3, color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 10 };
