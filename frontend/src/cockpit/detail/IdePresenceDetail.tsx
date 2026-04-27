import { useEffect, useMemo, useRef, useState } from 'react';
import { Cpu, Network, Clock, Sparkles, MessageSquare, ChevronRight, ChevronDown } from 'lucide-react';
import { api, intentOf, nameOf, type BusMessage, type PresenceEntry } from '../../lib/api';

interface Props {
  identity: string;
  // If the parent already has the presence list cached, can pass to skip a fetch.
  presenceCache?: PresenceEntry[];
}

const MSG_POLL_MS = 4000;
const MSG_TAIL = 30;
const MSG_FETCH_N = 200;
const SNIPPET_LEN = 140;

export default function IdePresenceDetail({ identity, presenceCache }: Props) {
  const [entry, setEntry] = useState<PresenceEntry | null>(
    () => presenceCache?.find(p => p.identity === identity) ?? null,
  );
  const [loading, setLoading] = useState(!entry);

  useEffect(() => {
    let alive = true;
    if (entry && entry.identity === identity) {
      // If we already have a match, refresh it in the background but don't show loading
      api.presence().then(list => {
        if (!alive) return;
        const fresh = list.find(p => p.identity === identity);
        if (fresh) setEntry(fresh);
      }).catch(() => {});
      return;
    }
    setLoading(true);
    api.presence().then(list => {
      if (!alive) return;
      const found = list.find(p => p.identity === identity) ?? null;
      setEntry(found);
      setLoading(false);
    }).catch(() => {
      if (alive) { setEntry(null); setLoading(false); }
    });
    return () => { alive = false; };
  }, [identity]);

  if (loading) {
    return <div style={{ padding: 16, color: 'var(--ink-muted)', fontSize: 12 }}>loading…</div>;
  }
  if (!entry) {
    // Not a presence-registered identity; the caller (DetailPane) should fall back to AgentStats.
    // We render NOTHING; DetailPane is responsible for routing.
    return null;
  }

  const isAuto = entry.capabilities?.includes('auto-respond') === true;
  const lastSeen = entry.last_seen_at ? new Date(entry.last_seen_at).toLocaleString() : '—';
  const expiresAt = entry.expires_at ? new Date(entry.expires_at).toLocaleString() : '—';

  return (
    <div style={{ padding: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <Network size={14} strokeWidth={1.6} style={{ color: 'var(--cta)' }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--ink-bright)', fontWeight: 600 }}>
          {entry.identity}
        </span>
        {isAuto && (
          <span className="chip chip-cta" style={{ marginLeft: 'auto' }}>
            <Sparkles size={9} strokeWidth={2} style={{ marginRight: 3 }} />
            AUTO
          </span>
        )}
      </div>

      {/* IDE + Host */}
      <Row label="ide"  value={entry.ide_name ?? '—'} mono />
      <Row label="host" value={entry.host ?? '—'} mono />

      {/* LLM provider — the killer info */}
      <div style={{
        marginTop: 16, padding: 12,
        background: 'var(--bg-2)',
        border: '1px solid var(--line)',
        borderRadius: 8,
      }}>
        <div className="eyebrow" style={{ marginBottom: 6 }}>auto-respond LLM</div>
        {isAuto ? (
          <div>
            <div style={{ fontSize: 14, color: 'var(--ink-bright)', fontWeight: 500 }}>
              {entry.llm_provider ?? 'unknown'}
            </div>
            {entry.llm_model && (
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', marginTop: 2 }}>
                {entry.llm_model}
              </div>
            )}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: 'var(--ink-faint)' }}>
            auto-respond is OFF — this IDE does not auto-answer incoming messages
          </div>
        )}
      </div>

      {/* Capabilities */}
      <div style={{ marginTop: 16 }}>
        <div className="eyebrow" style={{ marginBottom: 6 }}>capabilities</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {(entry.capabilities ?? []).map(c => (
            <span key={c} className="chip" style={{ fontFamily: 'var(--font-mono)' }}>{c}</span>
          ))}
        </div>
      </div>

      {/* Timestamps */}
      <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 6, color: 'var(--ink-muted)' }}>
        <Clock size={11} strokeWidth={1.6} />
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>last seen · {lastSeen}</span>
      </div>
      <div style={{ marginTop: 4, marginLeft: 17, color: 'var(--ink-faint)' }}>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}>expires · {expiresAt}</span>
      </div>

      {/* Hint */}
      <div style={{
        marginTop: 24, padding: 10,
        background: 'var(--cta-soft)',
        borderLeft: '2px solid var(--cta)',
        borderRadius: 4,
        fontSize: 11, color: 'var(--ink-muted)', lineHeight: 1.5,
      }}>
        <Cpu size={11} strokeWidth={1.6} style={{ marginRight: 4, verticalAlign: 'middle' }} />
        Send a handover to <code style={{ color: 'var(--cta)' }}>{entry.identity}</code> from any IDE
        — this instance will auto-answer using <strong>{entry.llm_provider ?? 'its configured provider'}</strong>.
      </div>

      {/* Recent messages — every bus message touching this identity */}
      <RecentMessages identity={entry.identity} />
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'baseline' }}>
      <span className="eyebrow" style={{ minWidth: 50 }}>{label}</span>
      <span style={{
        fontSize: 12,
        color: 'var(--ink-bright)',
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)',
      }}>
        {value}
      </span>
    </div>
  );
}

// ─── Recent messages section ────────────────────────────────────────

function RecentMessages({ identity }: { identity: string }) {
  const [all, setAll] = useState<BusMessage[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    setExpanded(new Set());
    setLoaded(false);
    const load = async () => {
      try {
        const list = await api.recentMessages(MSG_FETCH_N);
        if (!aliveRef.current) return;
        const filtered = (list ?? []).filter(m => touches(m, identity));
        setTotalMatches(filtered.length);
        setAll(filtered.slice(-MSG_TAIL).reverse());
        setLoaded(true);
      } catch {
        if (!aliveRef.current) return;
        setLoaded(true);
        // keep prior state on transient errors
      }
    };
    void load();
    const t = setInterval(load, MSG_POLL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(t);
    };
  }, [identity]);

  const toggle = (id: string) => {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id); else next.add(id);
    setExpanded(next);
  };

  const showCount = all.length;

  function openInKnowledge() {
    // The Knowledge view uses a search input. We can't deep-link without
    // adding a router; the simplest, honest behavior is to drop a hint
    // into sessionStorage and open the menu manually. For now, just emit
    // a custom event the App can pick up later. Fallback: alert-free no-op
    // so the button is never broken.
    window.dispatchEvent(new CustomEvent('orbit:open-knowledge', { detail: { search: identity } }));
  }

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 8 }}>
        <MessageSquare size={11} strokeWidth={1.6} style={{ color: 'var(--ink-muted)' }} />
        <span className="eyebrow">
          recent messages
        </span>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-faint)' }}>
          {showCount} of {totalMatches}
        </span>
      </div>

      {!loaded ? (
        <div style={msgEmptyStyle}>loading…</div>
      ) : showCount === 0 ? (
        <div style={msgEmptyStyle}>
          No bus messages involving this IDE in the last {MSG_FETCH_N}.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {all.map((m, idx) => (
            <MessageRow
              key={messageKey(m, idx)}
              message={m}
              identity={identity}
              expanded={expanded.has(messageKey(m, idx))}
              onToggle={() => toggle(messageKey(m, idx))}
            />
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={openInKnowledge}
        style={knowledgeLinkStyle}
      >
        view all in Knowledge →
      </button>
    </div>
  );
}

interface MsgRowProps {
  message: BusMessage;
  identity: string;
  expanded: boolean;
  onToggle: () => void;
}

function MessageRow({ message, identity, expanded, onToggle }: MsgRowProps) {
  const fromName = nameOf(message.from);
  const toName = nameOf(message.to);
  const incoming = toName === identity;
  const counterpart = incoming ? fromName : toName;
  const intent = intentOf(message.intent);
  const auto = !!message.auto_triggered;
  const content = message.content ?? '';
  const snippet = useMemo(() => {
    if (!content) return '';
    return content.length <= SNIPPET_LEN ? content : content.slice(0, SNIPPET_LEN - 1) + '…';
  }, [content]);
  const time = formatMsgTime(message.ts);
  const expandable = content.length > SNIPPET_LEN;

  return (
    <div style={msgRowStyle}>
      <button
        type="button"
        onClick={onToggle}
        style={msgHeaderBtn}
        aria-expanded={expanded}
        disabled={!expandable && !content}
      >
        {expandable
          ? (expanded
              ? <ChevronDown size={11} style={{ color: 'var(--ink-faint)', flexShrink: 0 }} />
              : <ChevronRight size={11} style={{ color: 'var(--ink-faint)', flexShrink: 0 }} />)
          : <span style={{ width: 11, flexShrink: 0 }} />}
        <span style={msgTimeStyle}>{time}</span>
        <span style={msgArrowStyle(incoming)}>{incoming ? '←' : '→'}</span>
        <span style={msgCounterpartStyle} title={counterpart}>{truncateName(counterpart)}</span>
        {intent && <span style={intentBadgeStyle}>{intent}</span>}
        {auto && <span style={autoBadgeStyle}>AUTO</span>}
        {snippet && (
          <span style={msgSnippetStyle} title={content}>"{snippet}"</span>
        )}
      </button>
      {expanded && content && (
        <pre style={msgFullStyle}>{content}</pre>
      )}
    </div>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────

function touches(m: BusMessage, identity: string): boolean {
  return nameOf(m.from) === identity || nameOf(m.to) === identity;
}

function messageKey(m: BusMessage, idx: number): string {
  if (m.id != null) return `id-${m.id}`;
  return `idx-${idx}-${m.ts ?? ''}`;
}

function truncateName(s: string): string {
  if (!s) return '?';
  if (s.length <= 18) return s;
  return s.slice(0, 8) + '…' + s.slice(-6);
}

function formatMsgTime(ts: string | undefined): string {
  if (!ts) return '--:--:--';
  const d = new Date(ts);
  if (!Number.isFinite(d.getTime())) return ts.slice(0, 8);
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  const s = d.getSeconds().toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
}

// ─── Styles for messages section ────────────────────────────────────

const msgEmptyStyle: React.CSSProperties = {
  padding: '12px 14px',
  fontSize: 11,
  color: 'var(--ink-faint)',
  background: 'var(--bg-raised)',
  border: '1px dashed var(--rule-faint)',
  borderRadius: 4,
  textAlign: 'center',
};

const msgRowStyle: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 3,
  overflow: 'hidden',
};

const msgHeaderBtn: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  width: '100%',
  padding: '5px 8px',
  background: 'transparent',
  border: 'none',
  cursor: 'pointer',
  textAlign: 'left',
  minHeight: 24,
};

const msgTimeStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--ink-faint)',
  flexShrink: 0,
  minWidth: 56,
};

function msgArrowStyle(incoming: boolean): React.CSSProperties {
  return {
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    fontWeight: 700,
    color: incoming ? 'var(--cta)' : 'var(--ok)',
    flexShrink: 0,
    width: 12,
    textAlign: 'center',
  };
}

const msgCounterpartStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--ink-bright)',
  fontWeight: 600,
  flexShrink: 0,
  minWidth: 100,
  maxWidth: 130,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const intentBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '0 5px',
  fontFamily: 'var(--font-mono)',
  fontSize: 9,
  fontWeight: 600,
  color: 'var(--accent)',
  background: 'var(--accent-soft)',
  borderRadius: 2,
  letterSpacing: 0.04,
  textTransform: 'uppercase',
  flexShrink: 0,
};

const autoBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '0 5px',
  fontFamily: 'var(--font-mono)',
  fontSize: 9,
  fontWeight: 700,
  color: 'var(--cta)',
  background: 'var(--cta-soft)',
  borderRadius: 2,
  letterSpacing: 0.06,
  flexShrink: 0,
};

const msgSnippetStyle: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  fontSize: 11,
  color: 'var(--ink-muted)',
  fontStyle: 'italic',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const msgFullStyle: React.CSSProperties = {
  margin: 0,
  padding: '8px 10px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.5,
  color: 'var(--ink-bright)',
  background: 'var(--bg-raised)',
  borderTop: '1px solid var(--rule-faint)',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  maxHeight: 240,
  overflowY: 'auto',
};

const knowledgeLinkStyle: React.CSSProperties = {
  marginTop: 8,
  padding: '4px 10px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--cta)',
  background: 'transparent',
  border: '1px solid var(--cta)',
  borderRadius: 3,
  cursor: 'pointer',
};
