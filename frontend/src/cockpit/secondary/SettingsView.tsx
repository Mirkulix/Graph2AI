import { useState } from 'react';
import { SecondaryFrame } from './FederationView';

export default function SettingsView() {
  const [token, setToken] = useState(() => localStorage.getItem('qo_auth_token') ?? '');
  const [seed,  setSeed]  = useState(() => localStorage.getItem('qo_signing_seed') ?? '');
  const [saved, setSaved] = useState<string | null>(null);

  function save() {
    if (token.trim()) localStorage.setItem('qo_auth_token', token.trim());
    else localStorage.removeItem('qo_auth_token');
    if (seed.trim())  localStorage.setItem('qo_signing_seed', seed.trim());
    else localStorage.removeItem('qo_signing_seed');
    setSaved('Saved. Reload to apply auth changes.');
    setTimeout(() => setSaved(null), 3000);
  }

  return (
    <SecondaryFrame title="settings" subtitle="local cockpit preferences (stored in browser)">
      <Field label="QO auth token" hint="Bearer token for /api/* requests. Stored in localStorage; never leaves your browser.">
        <input
          type="password"
          value={token}
          onChange={e => setToken(e.target.value)}
          placeholder="empty = no auth header"
          style={inputStyle}
        />
      </Field>

      <Field label="signing seed (64 hex)" hint="HMAC seed for outbound QLMS messages from this cockpit. Empty = use server-default 0x000…">
        <input
          type="text"
          value={seed}
          onChange={e => setSeed(e.target.value)}
          placeholder="64-character hex string"
          maxLength={64}
          style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }}
        />
      </Field>

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
        {saved && (
          <span style={{ fontSize: 11, color: 'var(--verified-bright)', fontFamily: 'var(--font-mono)', alignSelf: 'center' }}>
            {saved}
          </span>
        )}
        <button onClick={save} className="surface-button surface-button--primary">save</button>
      </div>

      <section style={{ marginTop: 32, padding: 16, background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4 }}>
        <h3 className="eyebrow" style={{ margin: '0 0 8px' }}>about</h3>
        <p style={{ fontSize: 12, color: 'var(--ink-muted)', margin: 0, lineHeight: 1.6 }}>
          OrbitQLang Cockpit · v1.1 protocol · signed-graph control plane.<br />
          Source: {' '}
          <a href="https://github.com/anthropics/orbitqlang" target="_blank" rel="noopener noreferrer">repository</a>
        </p>
      </section>
    </SecondaryFrame>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <label className="eyebrow" style={{ display: 'block', marginBottom: 4 }}>{label}</label>
      {children}
      {hint && (
        <p style={{ fontSize: 10, color: 'var(--ink-faint)', margin: '4px 0 0', fontFamily: 'var(--font-ui)', lineHeight: 1.5 }}>
          {hint}
        </p>
      )}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 10px',
  background: 'var(--bg-elevated)',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  color: 'var(--ink-bright)',
  fontFamily: 'var(--font-ui)',
  fontSize: 13,
};
