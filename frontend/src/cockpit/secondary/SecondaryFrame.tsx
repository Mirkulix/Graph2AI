// Shared secondary-view chrome used by all panels in /cockpit/secondary/.
// Originally lived in FederationView.tsx; extracted when Federation was
// removed so the shared layout primitives survive independently.

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
