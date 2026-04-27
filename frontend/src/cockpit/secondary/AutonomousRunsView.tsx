// AutonomousRunsView — branches created by overnight swarms.
// Polls /api/git/branches every 4s, lets the operator inspect a diff,
// merge a branch into main, or discard it.

import { useEffect, useMemo, useRef, useState } from 'react';
import { GitBranch, ChevronRight, ChevronDown, Check, X, Loader2 } from 'lucide-react';
import { api, type AutoBranchInfo } from '../../lib/api';
import { SecondaryFrame, Stat, Empty } from './SecondaryFrame';

const POLL_INTERVAL_MS = 4000;
const DIFF_INLINE_MAX = 2000;

type TestState = 'passed' | 'failed' | 'unknown';

interface DiffCacheEntry {
  diff: string;
  truncated: boolean;
  loadedAt: number;
}

export default function AutonomousRunsView() {
  const [branches, setBranches] = useState<AutoBranchInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [diffCache, setDiffCache] = useState<Record<string, DiffCacheEntry | 'loading' | 'error'>>({});
  const [busyBranch, setBusyBranch] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const aliveRef = useRef(true);

  // Poll branches every 4s
  useEffect(() => {
    aliveRef.current = true;
    const load = async () => {
      try {
        const list = await api.gitBranches();
        if (!aliveRef.current) return;
        setBranches(Array.isArray(list) ? list : []);
        setError(null);
      } catch (e) {
        if (!aliveRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();
    const t = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(t);
    };
  }, []);

  function flash(kind: 'ok' | 'err', text: string) {
    setToast({ kind, text });
    setTimeout(() => setToast(null), 4000);
  }

  async function toggleRow(branch: AutoBranchInfo) {
    const next = new Set(expanded);
    if (next.has(branch.name)) {
      next.delete(branch.name);
      setExpanded(next);
      return;
    }
    next.add(branch.name);
    setExpanded(next);
    // Load diff lazily on first expand
    if (!diffCache[branch.name] || diffCache[branch.name] === 'error') {
      setDiffCache(prev => ({ ...prev, [branch.name]: 'loading' }));
      try {
        const res = await api.gitDiff(branch.name);
        setDiffCache(prev => ({
          ...prev,
          [branch.name]: { diff: res.diff, truncated: res.truncated, loadedAt: Date.now() },
        }));
      } catch (e) {
        setDiffCache(prev => ({ ...prev, [branch.name]: 'error' }));
        flash('err', `diff failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  async function mergeBranch(branch: AutoBranchInfo) {
    if (busyBranch) return;
    setBusyBranch(branch.name);
    try {
      const res = await api.gitMerge(branch.name);
      if (res.merged) {
        setBranches(prev => prev.filter(b => b.name !== branch.name));
        flash('ok', `merged ${branch.name}`);
      } else {
        flash('err', `merge conflict: ${res.conflict ?? 'unknown'}`);
      }
    } catch (e) {
      flash('err', e instanceof Error ? e.message : String(e));
    } finally {
      setBusyBranch(null);
    }
  }

  async function discardBranch(branch: AutoBranchInfo) {
    if (busyBranch) return;
    // eslint-disable-next-line no-alert
    const ok = window.confirm(`Discard branch "${branch.name}"?\nThis cannot be undone.`);
    if (!ok) return;
    setBusyBranch(branch.name);
    try {
      const res = await api.gitDiscard(branch.name);
      if (res.discarded) {
        setBranches(prev => prev.filter(b => b.name !== branch.name));
        flash('ok', `discarded ${branch.name}`);
      } else {
        flash('err', 'discard failed');
      }
    } catch (e) {
      flash('err', e instanceof Error ? e.message : String(e));
    } finally {
      setBusyBranch(null);
    }
  }

  // ─── Stats (the four big numbers up top) ─────────────────────────
  const stats = useMemo(() => {
    const total = branches.length;
    const passing = branches.filter(b => testStateOf(b) === 'passed').length;
    const awaiting = branches.filter(b => testStateOf(b) !== 'failed').length;
    // "merged today" — we don't get historical merges back; show 0 unless
    // backend later returns merged-today via a header. Stay honest.
    const mergedToday = 0;
    return { total, passing, awaiting, mergedToday };
  }, [branches]);

  return (
    <SecondaryFrame
      title="autonomous runs"
      subtitle="branches created by overnight swarms — review the diff, then merge or discard"
    >
      <div style={statsRowStyle}>
        <Stat label="total branches" value={String(stats.total)} />
        <Stat label="passing tests" value={String(stats.passing)} accent />
        <Stat label="awaiting review" value={String(stats.awaiting)} />
        <Stat label="merged today" value={String(stats.mergedToday)} />
      </div>

      {error && (
        <div style={errorBannerStyle}>
          <strong>backend offline</strong> · {error}
          <div style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            The view will retry automatically every {POLL_INTERVAL_MS / 1000}s.
          </div>
        </div>
      )}

      {branches.length === 0 && !error ? (
        <Empty text="No autonomous branches. Overnight swarms will push their work here." />
      ) : (
        <div style={listStyle}>
          {branches.map(b => (
            <BranchRow
              key={b.name}
              branch={b}
              expanded={expanded.has(b.name)}
              onToggle={() => void toggleRow(b)}
              onMerge={() => void mergeBranch(b)}
              onDiscard={() => void discardBranch(b)}
              busy={busyBranch === b.name}
              diffEntry={diffCache[b.name] ?? null}
            />
          ))}
        </div>
      )}

      {toast && (
        <div style={toastStyle(toast.kind)} role="status" aria-live="polite">
          {toast.text}
        </div>
      )}
    </SecondaryFrame>
  );
}

// ─── One branch row ─────────────────────────────────────────────────

interface RowProps {
  branch: AutoBranchInfo;
  expanded: boolean;
  onToggle: () => void;
  onMerge: () => void;
  onDiscard: () => void;
  busy: boolean;
  diffEntry: DiffCacheEntry | 'loading' | 'error' | null;
}

function BranchRow({ branch, expanded, onToggle, onMerge, onDiscard, busy, diffEntry }: RowProps) {
  const tests = testStateOf(branch);
  const canMerge = tests !== 'failed' && !busy;

  return (
    <article style={rowStyle}>
      <button onClick={onToggle} style={rowHeaderBtn} aria-expanded={expanded}>
        {expanded
          ? <ChevronDown size={14} style={{ color: 'var(--ink-muted)', flexShrink: 0 }} />
          : <ChevronRight size={14} style={{ color: 'var(--ink-muted)', flexShrink: 0 }} />}
        <GitBranch size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span style={branchNameStyle}>{branch.name}</span>
        <TestDot state={tests} />
      </button>

      <div style={metaRowStyle}>
        <span style={shaStyle}>{branch.sha.slice(0, 7)}</span>
        <span style={dateStyle}>{formatDate(branch.date)}</span>
        <span style={messageStyle} title={branch.message}>"{truncate(branch.message, 64)}"</span>
      </div>

      {branch.diff && (
        <div style={diffStatRowStyle}>
          <span style={{ color: 'var(--ok)', fontWeight: 600 }}>+{branch.diff.insertions}</span>
          <span style={{ color: 'var(--danger)', fontWeight: 600 }}>−{branch.diff.deletions}</span>
          <span style={{ color: 'var(--ink-muted)' }}>
            · {branch.diff.files_changed} file{branch.diff.files_changed === 1 ? '' : 's'}
          </span>
        </div>
      )}

      <div style={testsLineStyle(tests)}>
        {tests === 'passed' && <><Check size={11} /> tests passed</>}
        {tests === 'failed' && <><X size={11} /> tests failed{branch.tests_detail ? ` (${branch.tests_detail})` : ''}</>}
        {tests === 'unknown' && <>· tests not run</>}
      </div>

      <div style={actionsRowStyle}>
        <button
          type="button"
          onClick={onToggle}
          className="surface-button"
          style={{ padding: '4px 12px', fontSize: 11 }}
        >
          {expanded ? 'hide diff' : 'view diff'}
        </button>
        <button
          type="button"
          onClick={onMerge}
          disabled={!canMerge}
          className="surface-button surface-button--primary"
          style={{
            padding: '4px 12px',
            fontSize: 11,
            opacity: canMerge ? 1 : 0.4,
            cursor: canMerge ? 'pointer' : 'not-allowed',
          }}
          title={tests === 'failed' ? 'tests failed — fix first' : 'merge into main'}
        >
          {busy ? 'working…' : 'merge to main'}
        </button>
        <button
          type="button"
          onClick={onDiscard}
          disabled={busy}
          style={discardBtnStyle(busy)}
        >
          discard
        </button>
      </div>

      {expanded && (
        <div style={diffPaneStyle}>
          {diffEntry === 'loading' && (
            <div style={diffLoadingStyle}>
              <Loader2 size={12} className="spin" /> loading diff…
            </div>
          )}
          {diffEntry === 'error' && (
            <div style={{ ...diffLoadingStyle, color: 'var(--danger)' }}>
              failed to load diff — click again to retry
            </div>
          )}
          {diffEntry && typeof diffEntry === 'object' && (
            <>
              <pre style={diffPreStyle}>{diffEntry.diff.slice(0, DIFF_INLINE_MAX)}</pre>
              {(diffEntry.truncated || diffEntry.diff.length > DIFF_INLINE_MAX) && (
                <div style={diffTruncStyle}>
                  diff truncated · showing first {DIFF_INLINE_MAX.toLocaleString()} chars
                  {diffEntry.truncated && ' (server-capped)'}
                </div>
              )}
            </>
          )}
        </div>
      )}

      <style>{spinKeyframes}</style>
    </article>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────

function testStateOf(b: AutoBranchInfo): TestState {
  if (b.tests === 'passed' || b.tests === 'failed' || b.tests === 'unknown') return b.tests;
  return 'unknown';
}

function truncate(s: string, n: number): string {
  if (!s) return '';
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + '…';
}

function formatDate(iso: string): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return iso;
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return `today ${d.toTimeString().slice(0, 5)}`;
  }
  // e.g. "apr 24 09:54"
  return d.toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

function TestDot({ state }: { state: TestState }) {
  const color =
    state === 'passed' ? 'var(--ok)' :
    state === 'failed' ? 'var(--danger)' :
                         'var(--ink-faint)';
  return (
    <span
      aria-label={`tests ${state}`}
      title={`tests ${state}`}
      style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
        marginLeft: 'auto',
        flexShrink: 0,
      }}
    />
  );
}

// ─── Styles ─────────────────────────────────────────────────────────

const spinKeyframes = `
@keyframes auto-runs-spin { to { transform: rotate(360deg); } }
.spin { animation: auto-runs-spin 900ms linear infinite; }
`;

const statsRowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  marginBottom: 16,
};

const errorBannerStyle: React.CSSProperties = {
  marginBottom: 12,
  padding: '10px 14px',
  background: 'var(--warn-soft)',
  border: '1px solid var(--warn)',
  borderRadius: 4,
  color: 'var(--warn)',
  fontSize: 12,
  fontFamily: 'var(--font-mono)',
};

const listStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const rowStyle: React.CSSProperties = {
  padding: '12px 14px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderLeft: '3px solid var(--accent)',
  borderRadius: 5,
};

const rowHeaderBtn: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  width: '100%',
  padding: 0,
  background: 'transparent',
  border: 'none',
  cursor: 'pointer',
  textAlign: 'left',
};

const branchNameStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 13,
  fontWeight: 600,
  color: 'var(--ink-bright)',
  flex: 1,
  minWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const metaRowStyle: React.CSSProperties = {
  marginTop: 6,
  display: 'flex',
  alignItems: 'baseline',
  gap: 10,
  flexWrap: 'wrap',
};

const shaStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--ink-muted)',
  padding: '1px 6px',
  background: 'var(--bg-raised)',
  borderRadius: 2,
};

const dateStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--ink-faint)',
};

const messageStyle: React.CSSProperties = {
  fontSize: 12,
  color: 'var(--ink-muted)',
  fontStyle: 'italic',
  flex: 1,
  minWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const diffStatRowStyle: React.CSSProperties = {
  marginTop: 6,
  display: 'flex',
  gap: 10,
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
};

function testsLineStyle(state: TestState): React.CSSProperties {
  const color =
    state === 'passed' ? 'var(--ok)' :
    state === 'failed' ? 'var(--danger)' :
                         'var(--ink-faint)';
  return {
    marginTop: 6,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color,
  };
}

const actionsRowStyle: React.CSSProperties = {
  marginTop: 10,
  display: 'flex',
  gap: 6,
  flexWrap: 'wrap',
};

function discardBtnStyle(busy: boolean): React.CSSProperties {
  return {
    padding: '4px 12px',
    fontSize: 11,
    fontFamily: 'var(--font)',
    color: 'var(--danger)',
    background: 'transparent',
    border: '1px solid var(--danger)',
    borderRadius: 3,
    cursor: busy ? 'not-allowed' : 'pointer',
    opacity: busy ? 0.4 : 1,
  };
}

const diffPaneStyle: React.CSSProperties = {
  marginTop: 10,
  border: '1px solid var(--rule-faint)',
  background: 'var(--bg-raised)',
  borderRadius: 4,
  overflow: 'hidden',
};

const diffLoadingStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '12px 14px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--ink-muted)',
};

const diffPreStyle: React.CSSProperties = {
  margin: 0,
  padding: '10px 12px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.5,
  color: 'var(--ink-bright)',
  whiteSpace: 'pre',
  overflowX: 'auto',
  maxHeight: 320,
  overflowY: 'auto',
};

const diffTruncStyle: React.CSSProperties = {
  padding: '6px 12px',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--ink-faint)',
  borderTop: '1px solid var(--rule-faint)',
  background: 'var(--bg-panel)',
};

function toastStyle(kind: 'ok' | 'err'): React.CSSProperties {
  return {
    position: 'fixed',
    bottom: 24,
    left: '50%',
    transform: 'translateX(-50%)',
    padding: '8px 18px',
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    color: kind === 'ok' ? 'var(--ok)' : 'var(--danger)',
    background: kind === 'ok' ? 'var(--ok-soft)' : 'var(--danger-soft)',
    border: `1px solid ${kind === 'ok' ? 'var(--ok)' : 'var(--danger)'}`,
    borderRadius: 4,
    zIndex: 200,
    boxShadow: 'var(--shadow-md)',
  };
}
