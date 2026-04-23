import { useEffect, useState } from 'react';
import { Cpu, Network, Clock, Sparkles } from 'lucide-react';
import { api, type PresenceEntry } from '../../lib/api';

interface Props {
  identity: string;
  // If the parent already has the presence list cached, can pass to skip a fetch.
  presenceCache?: PresenceEntry[];
}

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
