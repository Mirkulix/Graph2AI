import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { Dna, Play, Square, Activity, Users, Award, Zap, TrendingUp } from 'lucide-react'

// ============================================================
// Types — mirror qlang_runtime::evolution::daemon
// ============================================================

interface DaemonConfig {
  interval_secs: number
  population_size: number
  mutation_rate: number
  retire_fraction: number
  max_age: number
  seed: number
}

interface GenerationReport {
  generation: number
  timestamp: number
  population_size: number
  best_fitness: number
  avg_fitness: number
  min_fitness: number
  best_id: string
  retired: string[]
  killed: string[]
  spawned: string[]
  notes: string
}

interface SpecialistInfo {
  id: string
  generation_born: number
  parent_id: string | null
  children: string[]
  fitness: number
  age: number
  status: 'active' | 'retired' | 'dead'
  mutations: number
}

interface StatusResponse {
  initialized: boolean
  running?: boolean
  current_generation?: number
  uptime_secs?: number
  population_size?: number
  best_fitness_ever?: number
  total_born?: number
  total_retired?: number
  total_killed?: number
  config?: DaemonConfig
  last_report?: GenerationReport | null
}

// ============================================================
// Helpers
// ============================================================

const fitnessColor = (f: number): string => {
  const clamped = Math.max(0, Math.min(1, f))
  const hue = clamped * 130 // red 0 → green 130
  return `hsl(${hue}, 72%, 45%)`
}

const fmtDuration = (secs: number): string => {
  if (secs < 60) return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`
  const h = Math.floor(secs / 3600)
  return `${h}h ${Math.floor((secs % 3600) / 60)}m`
}

// ============================================================
// SVG Fitness Chart
// ============================================================

function FitnessChart({ history }: { history: GenerationReport[] }) {
  if (history.length < 2) {
    return (
      <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
        Warte auf mindestens 2 Generationen...
      </div>
    )
  }
  const w = 720, h = 220, pad = { t: 16, r: 20, b: 28, l: 44 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b
  const maxGen = history[history.length - 1].generation || 1
  const minGen = history[0].generation || 0
  const genRange = Math.max(1, maxGen - minGen)
  const x = (g: number) => pad.l + ((g - minGen) / genRange) * cw
  const y = (v: number) => pad.t + ch - Math.max(0, Math.min(1, v)) * ch

  const bestPath = history.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(r.generation).toFixed(1)},${y(r.best_fitness).toFixed(1)}`).join(' ')
  const avgPath = history.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(r.generation).toFixed(1)},${y(r.avg_fitness).toFixed(1)}`).join(' ')

  // Annotate generations with large population delta
  const deltas: { g: number; label: string }[] = []
  for (let i = 1; i < history.length; i++) {
    const delta = history[i].population_size - history[i - 1].population_size
    if (Math.abs(delta) >= 5) {
      deltas.push({ g: history[i].generation, label: `${delta > 0 ? '+' : ''}${delta}` })
    }
  }

  const gridLines = [0, 0.25, 0.5, 0.75, 1]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      {gridLines.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={y(v)} x2={w - pad.r} y2={y(v)} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3,4" />
          <text x={pad.l - 6} y={y(v) + 3} textAnchor="end" fontSize={10} fill="var(--text-muted)">{v.toFixed(2)}</text>
        </g>
      ))}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" />
      <path d={avgPath} fill="none" stroke="var(--accent-info)" strokeWidth={1.5} strokeDasharray="4,4" />
      <path d={bestPath} fill="none" stroke="var(--accent-success)" strokeWidth={2} />
      {deltas.map((d, i) => (
        <g key={i}>
          <line x1={x(d.g)} y1={pad.t} x2={x(d.g)} y2={h - pad.b} stroke="var(--accent-warning)" strokeOpacity={0.4} strokeDasharray="2,3" />
          <text x={x(d.g) + 2} y={pad.t + 10} fontSize={9} fill="var(--accent-warning)">pop {d.label}</text>
        </g>
      ))}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">Generation</text>
      <g transform={`translate(${w - pad.r - 130}, ${pad.t})`}>
        <rect x={0} y={0} width={126} height={38} fill="var(--bg-secondary)" stroke="var(--border)" rx={4} />
        <line x1={8} y1={13} x2={24} y2={13} stroke="var(--accent-success)" strokeWidth={2} />
        <text x={28} y={16} fontSize={10} fill="var(--text-primary)">best fitness</text>
        <line x1={8} y1={28} x2={24} y2={28} stroke="var(--accent-info)" strokeWidth={1.5} strokeDasharray="4,4" />
        <text x={28} y={31} fontSize={10} fill="var(--text-primary)">avg fitness</text>
      </g>
    </svg>
  )
}

// ============================================================
// SVG Lineage Tree
// ============================================================

function LineageTree({ nodes, focusId }: { nodes: SpecialistInfo[]; focusId: string }) {
  if (nodes.length === 0) {
    return <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>Keine Daten.</div>
  }
  // Assign depth (generation - min generation in subset).
  const minGen = Math.min(...nodes.map(n => n.generation_born))
  const maxGen = Math.max(...nodes.map(n => n.generation_born))
  const layers: Record<number, SpecialistInfo[]> = {}
  for (const n of nodes) {
    const depth = n.generation_born - minGen
    if (!layers[depth]) layers[depth] = []
    layers[depth].push(n)
  }
  const depthCount = Math.max(1, maxGen - minGen + 1)
  const w = 720, h = Math.max(180, depthCount * 90)
  const layerH = h / depthCount

  // Position nodes: spread across x axis within their layer.
  const positions: Record<string, { x: number; y: number }> = {}
  for (let d = 0; d < depthCount; d++) {
    const layer = layers[d] || []
    const n = layer.length
    layer.forEach((node, i) => {
      const x = ((i + 1) / (n + 1)) * w
      const y = d * layerH + layerH / 2
      positions[node.id] = { x, y }
    })
  }

  const edges: { x1: number; y1: number; x2: number; y2: number; color: string }[] = []
  const idSet = new Set(nodes.map(n => n.id))
  for (const n of nodes) {
    if (n.parent_id && idSet.has(n.parent_id) && positions[n.parent_id] && positions[n.id]) {
      edges.push({
        x1: positions[n.parent_id].x,
        y1: positions[n.parent_id].y + 10,
        x2: positions[n.id].x,
        y2: positions[n.id].y - 10,
        color: 'var(--border)',
      })
    }
  }

  const statusColor = (s: SpecialistInfo['status']) =>
    s === 'active' ? 'var(--accent-success)' : s === 'retired' ? 'var(--text-muted)' : '#1a1a1a'

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      {edges.map((e, i) => (
        <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={e.color} strokeWidth={1} />
      ))}
      {nodes.map(n => {
        const p = positions[n.id]
        if (!p) return null
        const isFocus = n.id === focusId
        return (
          <g key={n.id}>
            <circle
              cx={p.x} cy={p.y} r={isFocus ? 11 : 8}
              fill={statusColor(n.status)}
              stroke={isFocus ? 'var(--accent-primary)' : 'var(--bg-primary)'}
              strokeWidth={isFocus ? 3 : 1.5}
            />
            <text x={p.x} y={p.y + 24} textAnchor="middle" fontSize={9} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
              {n.id.slice(-6)}
            </text>
            <text x={p.x} y={p.y + 35} textAnchor="middle" fontSize={9} fill={fitnessColor(n.fitness)}>
              f={n.fitness.toFixed(2)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ============================================================
// Main component
// ============================================================

export default function EvolutionView() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [history, setHistory] = useState<GenerationReport[]>([])
  const [specialists, setSpecialists] = useState<SpecialistInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [lineage, setLineage] = useState<SpecialistInfo[]>([])
  const [serverError, setServerError] = useState<string | null>(null)

  const [cfg, setCfg] = useState<DaemonConfig>({
    interval_secs: 10,
    population_size: 50,
    mutation_rate: 0.01,
    retire_fraction: 0.2,
    max_age: 30,
    seed: 42,
  })

  const streamAbort = useRef<AbortController | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/evolution/status')
      if (!r.ok) throw new Error(`status ${r.status}`)
      const s: StatusResponse = await r.json()
      setStatus(s)
      setServerError(null)
    } catch (e) {
      setServerError((e as Error).message)
    }
  }, [])

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch('/api/evolution/history')
      if (r.ok) setHistory(await r.json())
    } catch { /* ignore */ }
  }, [])

  const fetchSpecialists = useCallback(async () => {
    try {
      const r = await fetch('/api/evolution/specialists')
      if (r.ok) setSpecialists(await r.json())
    } catch { /* ignore */ }
  }, [])

  const fetchLineage = useCallback(async (id: string) => {
    try {
      const r = await fetch(`/api/evolution/lineage/${id}`)
      if (r.ok) setLineage(await r.json())
    } catch { /* ignore */ }
  }, [])

  // Initial load + slow poll
  useEffect(() => {
    fetchStatus()
    fetchHistory()
    fetchSpecialists()
    const t = setInterval(() => {
      fetchStatus()
      fetchSpecialists()
    }, 5000)
    return () => clearInterval(t)
  }, [fetchStatus, fetchHistory, fetchSpecialists])

  // SSE — reconnects on disconnect
  useEffect(() => {
    let cancelled = false
    const connect = () => {
      if (cancelled) return
      const ctrl = new AbortController()
      streamAbort.current = ctrl
      fetch('/api/evolution/stream', { signal: ctrl.signal })
        .then(async res => {
          if (!res.ok || !res.body || cancelled) return
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ''
          let currentEvent = ''
          while (!cancelled) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
              if (line.startsWith('event: ')) currentEvent = line.slice(7).trim()
              else if (line.startsWith('data: ')) {
                try {
                  const parsed = JSON.parse(line.slice(6)) as GenerationReport
                  if (currentEvent === 'generation') {
                    setHistory(prev => [...prev, parsed].slice(-500))
                    fetchStatus()
                    fetchSpecialists()
                  }
                } catch { /* skip */ }
              }
            }
          }
        })
        .catch(() => { /* reconnect below */ })
        .finally(() => {
          if (!cancelled) setTimeout(connect, 3000)
        })
    }
    connect()
    return () => { cancelled = true; streamAbort.current?.abort() }
  }, [fetchStatus, fetchSpecialists])

  const onStart = async () => {
    try {
      await fetch('/api/evolution/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      })
      fetchStatus()
    } catch (e) {
      setServerError((e as Error).message)
    }
  }

  const onStop = async () => {
    try {
      await fetch('/api/evolution/stop', { method: 'POST' })
      fetchStatus()
    } catch (e) {
      setServerError((e as Error).message)
    }
  }

  const onSelect = (id: string) => {
    setSelectedId(id)
    fetchLineage(id)
  }

  const running = !!status?.running
  const initialized = !!status?.initialized

  const activeSpecialists = useMemo(
    () => specialists.filter(s => s.status === 'active').slice(0, 100),
    [specialists],
  )
  const feedReports = useMemo(() => [...history].reverse().slice(0, 25), [history])

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '6px 10px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg-primary)',
    color: 'var(--text-primary)', fontSize: 13,
  }

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* --- Control Panel ------------------------------------------------ */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <Dna size={18} style={{ color: 'var(--accent-primary)' }} />
          <h3 className="heading" style={{ margin: 0, fontSize: 14 }}>Evolution Daemon</h3>
          {running && (
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 10,
              background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)',
              color: 'var(--accent-success)', fontWeight: 600 }}>
              <Zap size={10} style={{ marginRight: 4 }} /> RUNNING
            </span>
          )}
          {!initialized && !serverError && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Daemon nicht initialisiert — Start klicken
            </span>
          )}
          {serverError && (
            <span style={{ fontSize: 11, color: 'var(--accent-danger)' }}>
              Server nicht erreichbar: {serverError}
            </span>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginBottom: 12 }}>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Intervall (s)</span>
            <input type="range" min={1} max={60} step={1} value={cfg.interval_secs}
              onChange={e => setCfg(c => ({ ...c, interval_secs: parseInt(e.target.value) }))}
              disabled={running} style={{ width: '100%' }} />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{cfg.interval_secs}s</span>
          </label>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Population</span>
            <input type="range" min={4} max={200} step={1} value={cfg.population_size}
              onChange={e => setCfg(c => ({ ...c, population_size: parseInt(e.target.value) }))}
              disabled={running} style={{ width: '100%' }} />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{cfg.population_size}</span>
          </label>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Mutation Rate</span>
            <input type="range" min={0} max={0.2} step={0.005} value={cfg.mutation_rate}
              onChange={e => setCfg(c => ({ ...c, mutation_rate: parseFloat(e.target.value) }))}
              disabled={running} style={{ width: '100%' }} />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{cfg.mutation_rate.toFixed(3)}</span>
          </label>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Retire Frac</span>
            <input type="range" min={0} max={0.8} step={0.01} value={cfg.retire_fraction}
              onChange={e => setCfg(c => ({ ...c, retire_fraction: parseFloat(e.target.value) }))}
              disabled={running} style={{ width: '100%' }} />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{(cfg.retire_fraction * 100).toFixed(0)}%</span>
          </label>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Max Age (gens)</span>
            <input type="number" min={1} max={500} value={cfg.max_age}
              onChange={e => setCfg(c => ({ ...c, max_age: parseInt(e.target.value) || 1 }))}
              disabled={running} style={inputStyle} />
          </label>
          <label style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: 3 }}>Seed</span>
            <input type="number" value={cfg.seed}
              onChange={e => setCfg(c => ({ ...c, seed: parseInt(e.target.value) || 0 }))}
              disabled={running} style={inputStyle} />
          </label>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={onStart} disabled={running}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 20px', fontSize: 13 }}>
            <Play size={14} /> Start
          </button>
          <button className="btn" onClick={onStop} disabled={!running}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 20px', fontSize: 13,
              background: running ? 'color-mix(in srgb, var(--accent-danger) 15%, transparent)' : 'var(--bg-secondary)',
              color: running ? 'var(--accent-danger)' : 'var(--text-muted)',
              border: '1px solid var(--border)', borderRadius: 6, cursor: running ? 'pointer' : 'not-allowed' }}>
            <Square size={14} /> Stop
          </button>
          {status?.current_generation != null && (
            <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)',
              display: 'flex', alignItems: 'center', gap: 16, fontFamily: 'var(--font-mono)' }}>
              <span>Gen <b style={{ color: 'var(--text-primary)' }}>{status.current_generation}</b></span>
              <span>Uptime <b style={{ color: 'var(--text-primary)' }}>{fmtDuration(status.uptime_secs || 0)}</b></span>
              <span>Best <b style={{ color: 'var(--accent-success)' }}>{(status.best_fitness_ever || 0).toFixed(3)}</b></span>
            </div>
          )}
        </div>
      </div>

      {/* --- Stats cards -------------------------------------------------- */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
        <StatCard label="Generations" value={`${status?.current_generation ?? 0}`} icon={TrendingUp} color="var(--accent-primary)" />
        <StatCard label="Population" value={`${status?.population_size ?? 0}`} icon={Users} color="var(--accent-info)" />
        <StatCard label="Best ever" value={(status?.best_fitness_ever ?? 0).toFixed(3)} icon={Award} color="var(--accent-success)" />
        <StatCard label="Born" value={`${status?.total_born ?? 0}`} icon={Activity} color="var(--accent-purple)" />
        <StatCard label="Retired" value={`${status?.total_retired ?? 0}`} icon={Activity} color="var(--accent-warning)" />
        <StatCard label="Killed" value={`${status?.total_killed ?? 0}`} icon={Activity} color="var(--accent-danger)" />
      </div>

      {/* --- Fitness chart ------------------------------------------------ */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <Activity size={16} style={{ color: 'var(--accent-success)' }} />
          <span style={{ fontSize: 13, fontWeight: 600 }}>Fitness über Generationen</span>
          {running && <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10,
            background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)',
            color: 'var(--accent-success)', fontWeight: 600 }}>LIVE</span>}
          <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' }}>
            {history.length} Reports
          </span>
        </div>
        <FitnessChart history={history} />
      </div>

      {/* --- Two-column layout: specialists + reports --------------------- */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Users size={16} style={{ color: 'var(--accent-info)' }} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Specialists (aktiv)</span>
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' }}>
              {activeSpecialists.length} gezeigt
            </span>
          </div>
          {activeSpecialists.length === 0 ? (
            <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
              Keine aktiven Specialists. Start den Daemon.
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: 8 }}>
              {activeSpecialists.map(s => (
                <button
                  key={s.id}
                  onClick={() => onSelect(s.id)}
                  style={{
                    padding: 8, borderRadius: 8,
                    border: s.id === selectedId ? '2px solid var(--accent-primary)' : '1px solid var(--border)',
                    background: 'var(--bg-primary)', textAlign: 'left',
                    cursor: 'pointer', transition: 'transform 0.15s',
                    display: 'flex', flexDirection: 'column', gap: 2,
                  }}
                >
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                    {s.id.slice(-8)}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 10, height: 10, borderRadius: '50%', background: fitnessColor(s.fitness) }} />
                    <span style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                      {s.fitness.toFixed(3)}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    gen {s.generation_born} · age {s.age}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Activity size={16} style={{ color: 'var(--accent-purple)' }} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Generation Feed</span>
          </div>
          {feedReports.length === 0 ? (
            <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
              Noch keine Reports.
            </div>
          ) : (
            <div style={{ maxHeight: 420, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {feedReports.map(r => (
                <div key={r.generation} style={{
                  padding: 8, borderRadius: 6, background: 'var(--bg-primary)',
                  border: '1px solid var(--border)', fontSize: 11,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                    <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>Gen {r.generation}</span>
                    <span style={{ color: fitnessColor(r.best_fitness), fontFamily: 'var(--font-mono)' }}>
                      best={r.best_fitness.toFixed(3)}
                    </span>
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                    {r.retired.length} retired · {r.killed.length} killed · {r.spawned.length} spawned · pop {r.population_size}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* --- Lineage viewer ---------------------------------------------- */}
      {selectedId && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Dna size={16} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Lineage: {selectedId}</span>
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' }}>
              {lineage.length} Knoten
            </span>
            <button onClick={() => { setSelectedId(null); setLineage([]) }}
              style={{ fontSize: 11, padding: '3px 8px', border: '1px solid var(--border)',
                borderRadius: 4, background: 'var(--bg-primary)', color: 'var(--text-muted)',
                cursor: 'pointer' }}>
              Schließen
            </button>
          </div>
          <LineageTree nodes={lineage} focusId={selectedId} />
        </div>
      )}
    </div>
  )
}

// ============================================================
// StatCard helper
// ============================================================

function StatCard({ label, value, icon: Icon, color }: {
  label: string; value: string; icon: typeof Dna; color: string
}) {
  return (
    <div className="card" style={{ padding: 14, textAlign: 'center' }}>
      <Icon size={16} style={{ color, marginBottom: 4 }} />
      <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
    </div>
  )
}
