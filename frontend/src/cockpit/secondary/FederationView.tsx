import { useEffect, useState } from 'react';
import { api } from '../../lib/api';

interface Peer { id: string; address?: string; last_seen?: string; rounds?: number }

export default function FederationView() {
  const [peers, setPeers] = useState<Peer[]>([]);
  const [stats, setStats] = useState<{ peer_count?: number; rounds_completed?: number; convergence?: number } | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      Promise.all([api.federationPeers().catch(() => []), api.federationStats().catch(() => null)])
        .then(([p, s]) => { if (alive) { setPeers(p ?? []); setStats(s); } });
    };
    load();
    const t = setInterval(load, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <SecondaryFrame title="federation" subtitle="ternary majority-vote consensus across peers">
      <div style={{ display: 'flex', gap: 12, marginBottom: 22 }}>
        <Stat label="peers"        value={String(stats?.peer_count ?? peers.length)} />
        <Stat label="rounds"       value={String(stats?.rounds_completed ?? 0)} />
        <Stat label="convergence"  value={stats?.convergence != null ? `${(stats.convergence * 100).toFixed(0)}%` : '—'} accent />
      </div>

      <h3 className="eyebrow" style={{ margin: '0 0 8px' }}>active peers</h3>
      {peers.length === 0 ? (
        <Empty text="No peers connected. Federation activates when other QO instances are reachable." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {peers.map(p => (
            <div key={p.id} style={rowStyle}>
              <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--verified)', boxShadow: '0 0 6px var(--verified-glow)' }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, color: 'var(--ink-bright)', fontFamily: 'var(--font-mono)' }}>{p.id}</div>
                {p.address && <div style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>{p.address}</div>}
              </div>
              {p.rounds != null && (
                <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
                  {p.rounds}r
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </SecondaryFrame>
  );
}

export function SecondaryFrame({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div style={{ padding: '24px 28px', maxWidth: 720, margin: '0 auto' }}>
      <div className="eyebrow" style={{ color: 'var(--ink-faint)' }}>view</div>
      <h2 style={{
        fontFamily: 'var(--font-display)',
        fontSize: 28,
        fontWeight: 500,
        margin: '4px 0 4px',
        color: 'var(--ink-bright)',
        letterSpacing: -0.01,
      }}>
        {title}
      </h2>
      {subtitle && <p style={{ fontSize: 13, color: 'var(--ink-muted)', margin: '0 0 24px' }}>{subtitle}</p>}
      {children}
    </div>
  );
}

export function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{
      flex: 1,
      padding: '12px 14px',
      background: 'var(--bg-panel)',
      border: '1px solid var(--rule-default)',
      borderRadius: 4,
    }}>
      <div className="eyebrow">{label}</div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 22,
        color: accent ? 'var(--verified-bright)' : 'var(--ink-bright)',
        marginTop: 4,
      }}>
        {value}
      </div>
    </div>
  );
}

export function Empty({ text }: { text: string }) {
  return (
    <div style={{ padding: 32, fontSize: 12, color: 'var(--ink-faint)', textAlign: 'center', border: '1px dashed var(--rule-default)', borderRadius: 4 }}>
      {text}
    </div>
  );
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  padding: '8px 10px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
};
