import { useEffect, useState } from 'react';
import { api, type AgentInfo } from '../../lib/api';

export default function AgentStats({ name }: { name: string }) {
  const [info, setInfo] = useState<AgentInfo | null>(null);

  useEffect(() => {
    let alive = true;
    api.busAgents()
      .then(list => { if (alive) setInfo(list?.find(a => a.name === name) ?? null); })
      .catch(() => { if (alive) setInfo(null); });
  }, [name]);

  if (!info) {
    return (
      <div style={{ padding: 8, fontSize: 11, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
        loading {name}…
      </div>
    );
  }

  const dotColor = info.online === false ? 'var(--ink-faint)' : 'var(--verified)';

  return (
    <div>
      <header style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 10, height: 10, borderRadius: 5, background: dotColor,
            boxShadow: info.online === false ? 'none' : `0 0 10px ${dotColor}`,
          }} />
          <h2 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 22,
            fontWeight: 500,
            margin: 0,
            color: 'var(--ink-bright)',
            letterSpacing: -0.01,
          }}>
            {info.name}
          </h2>
        </div>
        {info.role && (
          <div style={{ marginTop: 4, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
            role · {info.role}
          </div>
        )}
      </header>

      <div style={{ display: 'flex', gap: 8, marginBottom: 18 }}>
        <Stat label="mailbox" value={String(info.mailbox_size ?? 0)} />
        <Stat label="status" value={info.online === false ? 'offline' : 'online'} accent={info.online !== false} />
      </div>

      {info.capabilities && info.capabilities.length > 0 && (
        <section style={{ marginBottom: 18 }}>
          <h3 className="eyebrow" style={{ margin: '0 0 8px' }}>capabilities</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {info.capabilities.map(c => (
              <span key={c} style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                padding: '3px 8px',
                background: 'var(--bg-panel)',
                border: '1px solid var(--rule-default)',
                borderRadius: 3,
                color: 'var(--ink-primary)',
              }}>
                {c}
              </span>
            ))}
          </div>
        </section>
      )}

      {info.last_seen && (
        <section>
          <h3 className="eyebrow" style={{ margin: '0 0 4px' }}>last seen</h3>
          <p style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', margin: 0 }}>
            {info.last_seen}
          </p>
        </section>
      )}

      <section style={{ marginTop: 24, padding: 12, background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4 }}>
        <h3 className="eyebrow" style={{ margin: '0 0 6px' }}>signing identity</h3>
        <p style={{ fontSize: 11, color: 'var(--ink-muted)', margin: 0, lineHeight: 1.5 }}>
          Messages from this agent are signed with HMAC-SHA256.
          The recurring{' '}
          <span style={{ color: 'var(--verified-bright)', fontFamily: 'var(--font-mono)' }}>signed</span>
          {' '}badge in the conversation thread shows the first 6 hex chars of each signature.
        </p>
      </section>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{
      flex: 1,
      padding: '8px 10px',
      background: 'var(--bg-panel)',
      border: '1px solid var(--rule-default)',
      borderRadius: 4,
    }}>
      <div className="eyebrow" style={{ fontSize: 9 }}>{label}</div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 16,
        color: accent ? 'var(--verified-bright)' : 'var(--ink-bright)',
        marginTop: 2,
      }}>
        {value}
      </div>
    </div>
  );
}
