import { useEffect, useState } from 'react';
import { Plus, Power, Trash2, Sparkles } from 'lucide-react';
import { api, type ProviderConfig, type ProviderTemplate } from '../../lib/api';

const DEEPSEEK_ID = 'deepseek';

export default function ProvidersDetail() {
  const [configured, setConfigured] = useState<ProviderConfig[]>([]);
  const [templates, setTemplates] = useState<ProviderTemplate[]>([]);
  const [adding, setAdding] = useState<ProviderTemplate | null>(null);

  async function refresh() {
    try {
      const [c, t] = await Promise.all([api.providers(), api.providerTemplates()]);
      setConfigured(c ?? []);
      setTemplates(t ?? []);
    } catch {
      setConfigured([]);
      setTemplates([]);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  // Promote DeepSeek to first position in templates list (it's the new shiny)
  const sortedTemplates = [...templates].sort((a, b) => {
    if (a.id === DEEPSEEK_ID) return -1;
    if (b.id === DEEPSEEK_ID) return 1;
    return (a.name ?? a.id).localeCompare(b.name ?? b.id);
  });

  const configuredIds = new Set(configured.map(c => c.provider_type.toLowerCase()));

  return (
    <div>
      <section style={{ marginBottom: 22 }}>
        <h3 className="eyebrow" style={{ margin: '0 0 8px' }}>configured · {configured.length}</h3>
        {configured.length === 0 ? (
          <div style={{ fontSize: 11, color: 'var(--ink-faint)', padding: '8px 0' }}>
            No providers configured yet. Pick one below.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {configured.map(c => (
              <ConfiguredRow key={c.id} cfg={c} onChange={refresh} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h3 className="eyebrow" style={{ margin: '0 0 8px' }}>available templates</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {sortedTemplates.map(t => {
            const featured = t.id === DEEPSEEK_ID;
            const installed = configuredIds.has(t.id.toLowerCase());
            return (
              <TemplateRow
                key={t.id}
                template={t}
                featured={featured}
                installed={installed}
                onAdd={() => setAdding(t)}
              />
            );
          })}
        </div>
      </section>

      {adding && (
        <AddProviderModal
          template={adding}
          onClose={() => setAdding(null)}
          onAdded={() => { setAdding(null); refresh(); }}
        />
      )}
    </div>
  );
}

function ConfiguredRow({ cfg, onChange }: { cfg: ProviderConfig; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [latency, setLatency] = useState<number | undefined>(undefined);

  async function toggle() {
    setBusy(true);
    try {
      await api.providerToggle(cfg.id, !cfg.enabled);
      onChange();
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm(`Remove ${cfg.name ?? cfg.provider_type}?`)) return;
    setBusy(true);
    try {
      await api.providerDelete(cfg.id);
      onChange();
    } finally {
      setBusy(false);
    }
  }

  async function test() {
    setTesting(true);
    setOnline(null);
    try {
      const result = await api.providerTest(cfg.id);
      setOnline(result.success);
      setLatency(result.latency_ms);
    } catch {
      setOnline(false);
    } finally {
      setTesting(false);
    }
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '8px 10px',
      background: 'var(--bg-panel)',
      border: '1px solid var(--rule-default)',
      borderRadius: 4,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: 3,
        background: online === false ? 'var(--alert)' : (online === true ? 'var(--verified)' : (cfg.enabled ? 'var(--verified)' : 'var(--ink-faint)')),
        boxShadow: (online !== false && cfg.enabled) ? '0 0 6px var(--verified-glow)' : 'none',
      }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: 'var(--ink-bright)', fontWeight: 500 }}>
          {cfg.name ?? cfg.provider_type}
        </div>
        {cfg.model && (
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-faint)' }}>
            {cfg.model}
          </div>
        )}
        {online != null && (
          <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', marginTop: 2, color: online ? 'var(--verified)' : 'var(--alert-bright)' }}>
            {online ? `connection online${latency != null ? ` · ${latency}ms` : ''}` : 'connection offline'}
          </div>
        )}
      </div>
      {cfg.enabled && (
        <button onClick={test} disabled={testing} style={testBtn} title="test the live connection">
          {testing ? 'testing…' : 'test'}
        </button>
      )}
      <button
        onClick={toggle}
        disabled={busy}
        title={cfg.enabled ? 'disable' : 'enable'}
        style={iconBtn}
      >
        <Power size={12} strokeWidth={1.6} style={{ color: cfg.enabled ? 'var(--verified)' : 'var(--ink-muted)' }} />
      </button>
      <button onClick={remove} disabled={busy} title="remove" style={iconBtn}>
        <Trash2 size={12} strokeWidth={1.6} style={{ color: 'var(--ink-muted)' }} />
      </button>
    </div>
  );
}

function TemplateRow({ template, featured, installed, onAdd }:
  { template: ProviderTemplate; featured: boolean; installed: boolean; onAdd: () => void }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '8px 10px',
      background: featured ? 'var(--signal-soft)' : 'transparent',
      border: featured ? '1px solid var(--signal)' : '1px solid var(--rule-faint)',
      borderRadius: 4,
      transition: 'all 120ms',
    }}>
      {featured && <Sparkles size={11} strokeWidth={1.6} style={{ color: 'var(--signal-bright)' }} />}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: 'var(--ink-bright)', fontWeight: featured ? 600 : 500 }}>
          {template.name ?? template.id}
          {featured && (
            <span style={{ marginLeft: 6, fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--signal-bright)', letterSpacing: 0.06, textTransform: 'uppercase' }}>
              new
            </span>
          )}
        </div>
        {template.description && (
          <div style={{ fontSize: 10, color: 'var(--ink-faint)', marginTop: 2 }}>
            {template.description}
          </div>
        )}
        {template.models.length > 0 && (
          <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', marginTop: 4 }}>
            {template.models.slice(0, 3).map(m => m.id).join(' · ')}
            {template.models.length > 3 && ` · +${template.models.length - 3}`}
          </div>
        )}
      </div>
      <button
        onClick={onAdd}
        disabled={installed}
        className={installed ? 'surface-button' : 'surface-button surface-button--primary'}
        style={{ opacity: installed ? 0.5 : 1, cursor: installed ? 'default' : 'pointer' }}
      >
        {installed ? 'installed' : (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <Plus size={11} strokeWidth={2} /> add
          </span>
        )}
      </button>
    </div>
  );
}

function AddProviderModal({ template, onClose, onAdded }: { template: ProviderTemplate; onClose: () => void; onAdded: () => void }) {
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(template.models[0]?.id ?? '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const requiresKey = template.free !== true;

  async function submit() {
    setBusy(true);
    setErr(null);
    try {
      await api.providerAdd({
        template_id: template.id,
        api_key: requiresKey ? apiKey : '',
        model,
      });
      onAdded();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0,
      background: 'var(--bg-overlay)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 100,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--bg-raised)',
        border: '1px solid var(--rule-strong)',
        borderRadius: 8,
        padding: 24,
        width: 420,
        boxShadow: 'var(--shadow-deep)',
      }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 500, margin: 0, marginBottom: 4, color: 'var(--ink-bright)' }}>
          Add {template.name ?? template.id}
        </h2>
        {template.description && (
          <p style={{ fontSize: 12, color: 'var(--ink-muted)', marginTop: 0, marginBottom: 16 }}>
            {template.description}
          </p>
        )}

        {requiresKey && (
          <Field label="API key">
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder="sk-…"
              style={inputStyle}
              autoFocus
            />
          </Field>
        )}

        <Field label="default model">
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            style={inputStyle}
          >
            {template.models.map(m => (
              <option key={m.id} value={m.id}>
                {m.name ?? m.id}{m.cost_per_1k != null && m.cost_per_1k > 0 ? ` · $${m.cost_per_1k}/1k` : ' · free'}
              </option>
            ))}
          </select>
        </Field>

        {err && (
          <div style={{ fontSize: 11, color: 'var(--alert-bright)', background: 'var(--alert-soft)', padding: '6px 10px', borderRadius: 4, marginBottom: 12, fontFamily: 'var(--font-mono)' }}>
            {err}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose} className="surface-button">cancel</button>
          <button
            onClick={submit}
            disabled={busy || (requiresKey && !apiKey.trim())}
            className="surface-button surface-button--primary"
            style={{ opacity: (busy || (requiresKey && !apiKey.trim())) ? 0.5 : 1 }}
          >
            {busy ? 'saving…' : 'save'}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label className="eyebrow" style={{ display: 'block', marginBottom: 4 }}>{label}</label>
      {children}
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
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
};

const iconBtn: React.CSSProperties = {
  width: 24,
  height: 24,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  borderRadius: 3,
  transition: 'background 120ms',
};

const testBtn: React.CSSProperties = {
  padding: '3px 7px',
  fontSize: 9,
  fontFamily: 'var(--font-mono)',
  color: 'var(--ink-bright)',
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-default)',
  borderRadius: 3,
  cursor: 'pointer',
};
