import { useEffect, useState } from 'react'
import { Server, Plus, Trash2, Power, Zap, Key, Check, X, Edit3, DollarSign, Activity } from 'lucide-react'

interface CostSummary {
  total_cost_usd: number; cloud_cost_usd: number; groq_cost_usd: number
  local_cost_usd: number; total_requests: number; savings_vs_all_cloud_usd: number
}
interface ModelOption { id: string; name: string; cost_per_1k: number; recommended: boolean }
interface ProviderTemplate { id: string; name: string; provider_type: string; base_url: string; models: ModelOption[]; tier: number; free: boolean; description: string }
interface ConfiguredProvider {
  id: string; name: string; provider_type: string; model: string; base_url: string | null
  enabled: boolean; tier: number; cost_per_1k_tokens: number
  requests?: number; tokens?: number; cost_usd?: number; avg_latency_ms?: number; source?: string
}
interface TestResult { success: boolean; latency_ms: number; message: string }

const fmtCost = (v: number) => `$${v.toFixed(4)}`
const fmtTokens = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

const S = {
  row: { display: 'flex', alignItems: 'center', gap: '12px' } as React.CSSProperties,
  between: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } as React.CSSProperties,
  col: { display: 'flex', flexDirection: 'column' as const, gap: '10px' },
  icon: { width: 40, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 'var(--radius-md)', background: 'color-mix(in srgb, var(--accent-primary) 10%, transparent)', color: 'var(--accent-primary)' } as React.CSSProperties,
  freeBadge: { fontSize: '11px', padding: '2px 8px', borderRadius: '999px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', fontWeight: 600 } as React.CSSProperties,
  testResult: (ok: boolean) => ({ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', padding: '6px 10px', borderRadius: 'var(--radius-sm)', background: `color-mix(in srgb, ${ok ? 'var(--accent-success)' : 'var(--accent-danger)'} 10%, transparent)`, color: ok ? 'var(--accent-success)' : 'var(--accent-danger)' } as React.CSSProperties),
  editBox: { background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', padding: '12px', display: 'flex', flexDirection: 'column' as const, gap: '10px' },
  actionBtn: { flex: 1, fontSize: '12px', minHeight: '44px' } as React.CSSProperties,
}

function ProviderCard({ provider, maxLatency, onToggle, onDelete, onUpdate }: {
  provider: ConfiguredProvider; maxLatency: number
  onToggle: (id: string) => void; onDelete: (id: string) => void; onUpdate: () => void
}) {
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const [editing, setEditing] = useState(false)
  const [editKey, setEditKey] = useState('')
  const [editModel, setEditModel] = useState(provider.model)
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const latencyPct = maxLatency > 0 ? Math.min(((provider.avg_latency_ms ?? 0) / maxLatency) * 100, 100) : 0

  const handleTest = async () => {
    setTesting(true); setTestResult(null)
    try {
      const res = await fetch('/api/providers/test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: provider.id }) })
      setTestResult(await res.json())
    } catch { setTestResult({ success: false, latency_ms: 0, message: 'Netzwerkfehler' }) }
    finally { setTesting(false) }
  }

  const handleSave = async () => {
    setSaving(true)
    const body: Record<string, string> = {}
    if (editKey) body.api_key = editKey
    if (editModel !== provider.model) body.model = editModel
    if (Object.keys(body).length > 0)
      await fetch(`/api/providers/${provider.id}/edit`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    setSaving(false); setEditing(false); setEditKey(''); onUpdate()
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px', opacity: provider.enabled ? 1 : 0.6 }}>
      <div style={S.between}>
        <div style={S.row}>
          <div style={S.icon}><Server size={20} /></div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 600 }}>{provider.name}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>{provider.model}</div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {provider.cost_per_1k_tokens === 0 && <span style={S.freeBadge}>Kostenlos</span>}
          <span className={`badge ${provider.enabled ? 'badge-idle' : 'badge-info'}`}>
            <span className="badge-dot" />{provider.enabled ? 'Aktiv' : 'Inaktiv'}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        <div><div className="label">Requests</div><div className="stat-value" style={{ fontSize: '16px', color: 'var(--accent-primary)' }}>{provider.requests ?? 0}</div></div>
        <div><div className="label">Tokens</div><div className="stat-value" style={{ fontSize: '16px', color: 'var(--accent-primary)' }}>{fmtTokens(provider.tokens ?? 0)}</div></div>
        <div><div className="label">Kosten</div><div className="stat-value mono" style={{ fontSize: '16px' }}>{fmtCost(provider.cost_usd ?? 0)}</div></div>
        <div><div className="label">Latenz</div><div className="stat-value mono" style={{ fontSize: '16px' }}>{(provider.avg_latency_ms ?? 0) > 0 ? `${provider.avg_latency_ms}ms` : '—'}</div></div>
      </div>

      {(provider.avg_latency_ms ?? 0) > 0 && (
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${latencyPct}%`, background: (provider.avg_latency_ms ?? 0) > 1000 ? 'var(--accent-warning)' : 'var(--accent-success)' }} />
        </div>
      )}

      {provider.source === 'env' && <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>Quelle: Umgebungsvariable</div>}

      {testResult && (
        <div style={S.testResult(testResult.success)}>
          {testResult.success ? <Check size={12} /> : <X size={12} />}
          {testResult.success ? `OK — ${testResult.latency_ms}ms` : testResult.message}
        </div>
      )}

      {editing && (
        <div style={S.editBox}>
          <div>
            <label className="label" style={{ display: 'block', marginBottom: '4px' }}>API Key</label>
            <div style={{ display: 'flex', gap: '6px' }}>
              <input className="input" type={showKey ? 'text' : 'password'} value={editKey} onChange={e => setEditKey(e.target.value)} placeholder="Neuer API Key (leer = unverändert)" style={{ flex: 1 }} />
              <button className="btn btn-ghost btn-icon" onClick={() => setShowKey(k => !k)}>{showKey ? <X size={14} /> : <Key size={14} />}</button>
            </div>
          </div>
          <div>
            <label className="label" style={{ display: 'block', marginBottom: '4px' }}>Modell</label>
            <input className="input" value={editModel} onChange={e => setEditModel(e.target.value)} placeholder="Modell-ID" />
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn btn-primary" style={{ flex: 1, fontSize: '12px' }} disabled={saving} onClick={handleSave}><Check size={13} /> Speichern</button>
            <button className="btn btn-ghost" style={{ fontSize: '12px' }} onClick={() => { setEditing(false); setEditKey('') }}>Abbrechen</button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px' }}>
        <button className="btn btn-ghost" style={S.actionBtn} onClick={handleTest} disabled={testing}><Zap size={13} /> {testing ? 'Teste…' : 'Testen'}</button>
        <button className="btn btn-ghost" style={S.actionBtn} onClick={() => { setEditing(e => !e); setEditModel(provider.model) }}><Edit3 size={13} /> Bearbeiten</button>
        <button className="btn btn-ghost" style={S.actionBtn} onClick={() => onToggle(provider.id)}><Power size={13} /> {provider.enabled ? 'Aus' : 'An'}</button>
        <button className="btn btn-ghost btn-icon" style={{ color: 'var(--accent-danger)', minHeight: '44px' }} onClick={() => onDelete(provider.id)}><Trash2 size={13} /></button>
      </div>
    </div>
  )
}

function AddProviderPanel({ templates, onAdded, onClose }: { templates: ProviderTemplate[]; onAdded: () => void; onClose: () => void }) {
  const sorted = [...templates].sort((a, b) => (a.free === b.free ? a.tier - b.tier : a.free ? -1 : 1))
  const [selected, setSelected] = useState<ProviderTemplate | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [model, setModel] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<string | null>(null)

  const selectTemplate = (t: ProviderTemplate) => {
    setSelected(t)
    setModel((t.models.find(m => m.recommended) ?? t.models[0])?.id ?? '')
    setApiKey(''); setTestResult(null); setAddError(null)
  }

  const handleTest = async () => {
    if (!selected) return
    setTesting(true); setTestResult(null)
    try {
      await fetch('/api/providers/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ template_id: selected.id, api_key: apiKey, model }) })
      const res = await fetch('/api/providers/test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: selected.id }) })
      setTestResult(await res.json())
    } catch { setTestResult({ success: false, latency_ms: 0, message: 'Netzwerkfehler' }) }
    finally { setTesting(false) }
  }

  const handleAdd = async () => {
    if (!selected) return
    setAdding(true); setAddError(null)
    try {
      const res = await fetch('/api/providers/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ template_id: selected.id, api_key: apiKey, model }) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onAdded(); onClose()
    } catch (e: unknown) { setAddError(e instanceof Error ? e.message : 'Fehler') }
    finally { setAdding(false) }
  }

  const needsKey = selected && selected.id !== 'ollama'

  return (
    <div className="card-static" style={{ marginTop: '16px' }}>
      <div style={{ ...S.between, marginBottom: '16px' }}>
        <h3 className="heading" style={{ fontSize: '14px', margin: 0 }}>Provider hinzufügen</h3>
        <button className="btn btn-ghost btn-icon" onClick={onClose}><X size={16} /></button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '10px', marginBottom: '16px' }}>
        {sorted.map(t => (
          <button key={t.id} onClick={() => selectTemplate(t)} style={{ cursor: 'pointer', textAlign: 'left', padding: '12px', borderRadius: 'var(--radius-md)', border: `1px solid ${selected?.id === t.id ? 'var(--accent-primary)' : 'var(--border)'}`, background: selected?.id === t.id ? 'color-mix(in srgb, var(--accent-primary) 6%, var(--bg-elevated))' : 'var(--bg-elevated)', minHeight: '44px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600, fontSize: '13px' }}>{t.name}</span>
              {t.free && <span style={{ ...S.freeBadge, fontSize: '10px', padding: '1px 6px' }}>FREE</span>}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>{t.description}</div>
          </button>
        ))}
      </div>

      {selected && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px', ...S.col }}>
          <div style={{ fontSize: '13px', fontWeight: 600 }}>{selected.name} konfigurieren</div>
          {needsKey && (
            <div>
              <label className="label" style={{ display: 'block', marginBottom: '4px' }}><Key size={11} style={{ display: 'inline', marginRight: '4px' }} />API Key</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input className="input" type={showKey ? 'text' : 'password'} value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder={`${selected.name} API Key`} style={{ flex: 1 }} />
                <button className="btn btn-ghost" style={{ minHeight: '44px', padding: '0 12px', fontSize: '11px' }} onClick={() => setShowKey(v => !v)}>{showKey ? 'Verstecken' : 'Zeigen'}</button>
              </div>
            </div>
          )}
          <div>
            <label className="label" style={{ display: 'block', marginBottom: '4px' }}>Modell</label>
            <select value={model} onChange={e => setModel(e.target.value)} className="input" style={{ appearance: 'auto' }}>
              {selected.models.map(m => <option key={m.id} value={m.id}>{m.name}{m.recommended ? ' (empfohlen)' : ''}{m.cost_per_1k === 0 ? ' — kostenlos' : ` — $${m.cost_per_1k}/1K`}</option>)}
            </select>
          </div>
          {testResult && (
            <div style={S.testResult(testResult.success)}>
              {testResult.success ? <Check size={14} /> : <X size={14} />}
              {testResult.success ? `Verbindung erfolgreich — ${testResult.latency_ms}ms` : `Fehler: ${testResult.message}`}
            </div>
          )}
          {addError && <div style={{ fontSize: '12px', color: 'var(--accent-danger)' }}>{addError}</div>}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn btn-ghost" onClick={handleTest} disabled={testing || (!!needsKey && !apiKey)} style={{ fontSize: '13px' }}><Zap size={14} /> {testing ? 'Teste…' : 'Testen'}</button>
            <button className="btn btn-primary" onClick={handleAdd} disabled={adding || (!!needsKey && !apiKey)} style={{ fontSize: '13px', marginLeft: 'auto' }}><Plus size={14} /> {adding ? 'Wird hinzugefügt…' : 'Hinzufügen'}</button>
          </div>
        </div>
      )}
    </div>
  )
}

function CostBar({ label, value, maxVal, color }: { label: string; value: number; maxVal: number; color: string }) {
  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px' }}>
        <span>{label}</span><span className="mono" style={{ color: 'var(--text-secondary)' }}>{fmtCost(value)}</span>
      </div>
      <div className="progress-track" style={{ height: '8px' }}>
        <div className="progress-fill" style={{ width: `${maxVal > 0 ? (value / maxVal) * 100 : 0}%`, minWidth: value > 0 ? '4px' : '0', background: color }} />
      </div>
    </div>
  )
}

export default function ProviderView() {
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null)
  const [configured, setConfigured] = useState<ConfiguredProvider[]>([])
  const [templates, setTemplates] = useState<ProviderTemplate[]>([])
  const [totalTokens, setTotalTokens] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [showAdd, setShowAdd] = useState(false)

  const fetchData = async () => {
    try {
      const [costRes, confRes, tplRes] = await Promise.all([fetch('/api/providers/costs'), fetch('/api/providers/configured'), fetch('/api/providers/templates')])
      if (!costRes.ok) throw new Error('API error')
      const [cost, conf, tpls] = await Promise.all([costRes.json(), confRes.json(), tplRes.json()])
      const confArr: ConfiguredProvider[] = Array.isArray(conf) ? conf : []
      setCostSummary(cost)
      setConfigured(confArr)
      setTotalTokens(confArr.reduce((s, p) => s + (p.tokens ?? 0), 0))
      setTemplates(Array.isArray(tpls) ? tpls : [])
      setError(null)
    } catch { setError('Fehler beim Laden der Provider-Daten') }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchData(); const t = setInterval(fetchData, 10000); return () => clearInterval(t) }, [])

  const handleToggle = async (id: string) => { await fetch(`/api/providers/${id}/toggle`, { method: 'PUT' }); fetchData() }
  const handleDelete = async (id: string) => { await fetch(`/api/providers/${id}`, { method: 'DELETE' }); fetchData() }

  const maxLatency = Math.max(...configured.map(p => p.avg_latency_ms ?? 0), 1)
  const costMax = costSummary ? Math.max(costSummary.groq_cost_usd, costSummary.cloud_cost_usd, costSummary.local_cost_usd, 0.0001) : 0.0001

  return (
    <div className="view">
      <div className="card-static" style={{ marginBottom: '20px' }}>
        {loading ? (
          <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Lade Daten…</div>
        ) : error ? (
          <div style={{ color: 'var(--accent-danger)', fontSize: '14px' }}>{error}</div>
        ) : costSummary ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '32px', alignItems: 'center' }}>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}><DollarSign size={11} style={{ display: 'inline', marginRight: '4px' }} />Gesamtkosten</div>
              <div className="stat-value mono" style={{ fontSize: '28px', color: 'var(--accent-primary)' }}>{fmtCost(costSummary.total_cost_usd)}</div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}><Activity size={11} style={{ display: 'inline', marginRight: '4px' }} />Requests</div>
              <div className="stat-value mono" style={{ fontSize: '22px' }}>{costSummary.total_requests}</div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Tokens (est.)</div>
              <div className="stat-value mono" style={{ fontSize: '22px' }}>~{fmtTokens(totalTokens)}</div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Gespart vs. Cloud</div>
              <div className="stat-value mono" style={{ fontSize: '22px', color: 'var(--accent-success)' }}>~{fmtCost(costSummary.savings_vs_all_cloud_usd)}</div>
            </div>
          </div>
        ) : null}
      </div>

      <div style={{ marginBottom: '24px' }}>
        <div style={{ ...S.between, marginBottom: '14px' }}>
          <h3 className="heading" style={{ fontSize: '14px', margin: 0 }}>LLM Provider ({configured.length})</h3>
          <button className="btn btn-primary" onClick={() => setShowAdd(v => !v)} style={{ fontSize: '13px', padding: '0 16px' }}>
            <Plus size={14} /> Provider hinzufügen
          </button>
        </div>
        {configured.length === 0 && !showAdd && (
          <div style={{ color: 'var(--text-secondary)', fontSize: '13px', padding: '12px 0' }}>
            Noch keine Provider konfiguriert. Klicke "Provider hinzufügen" um zu starten.
          </div>
        )}
        <div className="grid-3">
          {configured.map(p => (
            <ProviderCard key={p.id} provider={p} maxLatency={maxLatency} onToggle={handleToggle} onDelete={handleDelete} onUpdate={fetchData} />
          ))}
        </div>
        {showAdd && <AddProviderPanel templates={templates} onAdded={fetchData} onClose={() => setShowAdd(false)} />}
      </div>

      {costSummary && (
        <div className="card-static">
          <h3 className="heading" style={{ fontSize: '14px', marginBottom: '16px' }}>Kostenverteilung</h3>
          <CostBar label="Groq (Free)" value={costSummary.groq_cost_usd} maxVal={costMax} color="var(--accent-info)" />
          <CostBar label="Cloud LLM" value={costSummary.cloud_cost_usd} maxVal={costMax} color="var(--accent-warning)" />
          <CostBar label="QO-LLM (IGQK)" value={costSummary.local_cost_usd} maxVal={costMax} color="var(--accent-success)" />
        </div>
      )}
    </div>
  )
}
