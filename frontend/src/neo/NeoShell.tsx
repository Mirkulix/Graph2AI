import { useEffect, useState } from 'react'
import {
  Sparkles, Brain, Dna, Users, Database,
  MessageCircle, Cpu, Send,
} from 'lucide-react'
import NeoMind from './NeoMind'
import NeoAgents from './NeoAgents'
import NeoBody from './NeoBody'
import NeoMemory from './NeoMemory'
import NeoMessages from './NeoMessages'
import NeoDialogue from './NeoDialogue'

// ── Tab definitions ───────────────────────────────────────────────
type NeoTab = 'mind' | 'evolution' | 'agents' | 'memory' | 'messages' | 'body' | 'dialogue'

const NEO_TABS: { id: NeoTab; label: string; icon: typeof Brain }[] = [
  { id: 'mind',      label: 'Mind',      icon: Brain },
  { id: 'evolution', label: 'Evolution', icon: Dna },
  { id: 'agents',    label: 'Agents',    icon: Users },
  { id: 'memory',    label: 'Memory',    icon: Database },
  { id: 'messages',  label: 'Messages',  icon: Send },
  { id: 'body',      label: 'Body',      icon: Cpu },
  { id: 'dialogue',  label: 'Dialogue',  icon: MessageCircle },
]

// ── Shared types for status strip / activity feed ─────────────────
interface NeoStatus {
  server: string
  hdc_memory: number
  organism_generation: number
  organism_interactions: number
  organism_memory_items: number
  specialists: number
  gpu_count: number
  gpu_temps: number[]
  gpu_utils: number[]
}

interface ActivityItem {
  ts: number
  text: string
  kind: 'info' | 'qlms' | 'warn'
}

// ── Small inline Evolution view (uses /api/evolution/daemon status) ──
function NeoEvolutionInline() {
  const [gen, setGen] = useState<number | null>(null)
  const [specialists, setSpecialists] = useState<number>(0)
  const [interactions, setInteractions] = useState<number>(0)

  useEffect(() => {
    let alive = true
    const load = () => {
      fetch('/api/organism/status')
        .then(r => r.ok ? r.json() : Promise.reject(new Error(String(r.status))))
        .then(d => {
          if (!alive) return
          setGen(d.generation ?? 0)
          setSpecialists(Array.isArray(d.specialists) ? d.specialists.length : 0)
          setInteractions(d.total_interactions ?? 0)
        })
        .catch(() => {})
    }
    load()
    const iv = setInterval(load, 4000)
    return () => { alive = false; clearInterval(iv) }
  }, [])

  return (
    <div style={{ padding: 24, color: '#d4d8e4', fontFamily: 'JetBrains Mono' }}>
      <h2 style={{ fontFamily: 'Outfit', fontWeight: 700, fontSize: 24, margin: 0 }}>Evolution</h2>
      <div style={{ fontSize: 11, color: '#7a8199', marginTop: 4, marginBottom: 20 }}>
        Organism generational progress.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16 }}>
        <MetricCard label="GENERATION" value={gen?.toString() ?? '—'} color="#4f8eff" />
        <MetricCard label="SPECIALISTS" value={specialists.toString()} color="#2dd4a0" />
        <MetricCard label="INTERACTIONS" value={interactions.toString()} color="#d4508e" />
      </div>
      <div style={{ marginTop: 24, padding: 20, background: '#0a0d14', border: '1px solid #1a2030', borderRadius: 14 }}>
        <div style={{ fontSize: 11, color: '#7a8199', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 10 }}>
          Lifecycle
        </div>
        <div style={{ fontSize: 12, color: '#d4d8e4', lineHeight: 1.6 }}>
          The organism grows by interaction. Each chat turn reinforces one specialist; evolution periodically
          prunes, mutates and reinstantiates specialist ensembles against the shared HDC memory. Trigger
          <code style={{ margin: '0 6px', padding: '2px 6px', background: '#141820', border: '1px solid #1a2030', borderRadius: 4, color: '#4f8eff' }}>
            POST /api/organism/evolve
          </code>
          to advance.
        </div>
      </div>
    </div>
  )
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      padding: 20,
      background: 'linear-gradient(145deg,#0a0d14,#0f1219)',
      border: `1px solid ${color}33`, borderRadius: 14,
      boxShadow: `0 0 24px ${color}15, inset 0 0 20px #0006`,
    }}>
      <div style={{ fontSize: 10, color: '#7a8199', letterSpacing: 1.5 }}>{label}</div>
      <div style={{ fontSize: 34, fontWeight: 700, color, fontFamily: 'Outfit', marginTop: 6 }}>
        {value}
      </div>
    </div>
  )
}

// ── Status strip across the top ───────────────────────────────────
function StatusStrip({ status }: { status: NeoStatus | null }) {
  const online = status?.server === 'online'
  const maxTemp = status?.gpu_temps?.length ? Math.max(...status.gpu_temps) : 0
  const tempColor = maxTemp < 50 ? '#2dd4a0' : maxTemp < 70 ? '#f0b429' : '#ff4f6e'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 20,
      padding: '12px 24px',
      background: 'linear-gradient(90deg,#04060b,#0a0d14,#04060b)',
      borderBottom: '1px solid #161b27',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Sparkles size={22} style={{ color: '#4f8eff', filter: 'drop-shadow(0 0 8px #4f8eff)' }} />
        <div>
          <div style={{ fontFamily: 'Outfit', fontWeight: 800, fontSize: 22, letterSpacing: 2, color: '#d4d8e4' }}>
            NEO
          </div>
          <div style={{ fontSize: 9, color: '#7a8199', letterSpacing: 2, marginTop: -2 }}>COMPANION</div>
        </div>
      </div>

      <div style={{ width: 1, height: 36, background: '#161b27' }} />

      <Pill label="SERVER" value={online ? 'online' : 'offline'} color={online ? '#2dd4a0' : '#ff4f6e'} />
      <Pill label="GPU TEMP" value={maxTemp ? `${maxTemp}°C` : '—'} color={tempColor} />
      <Pill label="GPUs" value={status ? status.gpu_count.toString() : '—'} color="#4f8eff" />
      <Pill label="SPECIALISTS" value={status ? status.specialists.toString() : '—'} color="#7c5fcf" />
      <Pill label="GENERATION" value={status ? status.organism_generation.toString() : '—'} color="#d4508e" />
      <Pill label="MEMORY" value={status ? String(status.hdc_memory + status.organism_memory_items) : '—'} color="#2dd4a0" />
    </div>
  )
}

function Pill({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      padding: '4px 12px', borderRadius: 8,
      background: '#0a0d14', border: `1px solid ${color}33`,
      boxShadow: `inset 0 0 12px ${color}10`,
    }}>
      <span style={{ fontSize: 9, color: '#7a8199', letterSpacing: 1.5 }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color, fontFamily: 'JetBrains Mono' }}>{value}</span>
    </div>
  )
}

// ── Left sidebar (Neo-local tabs) ─────────────────────────────────
function LeftSidebar({ active, onSelect }: { active: NeoTab; onSelect: (t: NeoTab) => void }) {
  return (
    <nav style={{
      width: 180, flexShrink: 0,
      padding: '18px 12px',
      background: '#04060b',
      borderRight: '1px solid #161b27',
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      {NEO_TABS.map(tab => {
        const Icon = tab.icon
        const on = active === tab.id
        return (
          <button
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '10px 14px', border: 'none', borderRadius: 10,
              background: on
                ? 'linear-gradient(135deg,#4f8eff22,#7c5fcf22)'
                : 'transparent',
              color: on ? '#d4d8e4' : '#7a8199',
              boxShadow: on ? '0 0 16px #4f8eff22, inset 0 0 12px #4f8eff15' : 'none',
              cursor: 'pointer',
              fontFamily: 'Outfit', fontSize: 12, fontWeight: 600,
              letterSpacing: 1, textTransform: 'uppercase',
              transition: 'all 180ms ease',
            }}
            onMouseEnter={e => {
              if (!on) e.currentTarget.style.background = '#0a0d14'
            }}
            onMouseLeave={e => {
              if (!on) e.currentTarget.style.background = 'transparent'
            }}
          >
            <Icon size={16} style={{ color: on ? '#4f8eff' : '#7a8199' }} />
            {tab.label}
          </button>
        )
      })}
    </nav>
  )
}

// ── Right sidebar (SSE activity) ──────────────────────────────────
function RightActivity({ items }: { items: ActivityItem[] }) {
  return (
    <aside style={{
      width: 280, flexShrink: 0,
      background: '#04060b',
      borderLeft: '1px solid #161b27',
      padding: '18px 14px',
      overflowY: 'auto',
      fontFamily: 'JetBrains Mono',
      color: '#d4d8e4',
    }}>
      <div style={{
        fontFamily: 'Outfit', fontSize: 11, fontWeight: 700,
        letterSpacing: 2, textTransform: 'uppercase',
        color: '#7a8199', marginBottom: 12,
      }}>
        Activity
      </div>
      {items.length === 0 ? (
        <div style={{ fontSize: 11, color: '#3d4455', textAlign: 'center', padding: 20 }}>
          Awaiting events…
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {items.slice().reverse().slice(0, 80).map((it, i) => {
            const color = it.kind === 'qlms' ? '#4f8eff' : it.kind === 'warn' ? '#ff4f6e' : '#2dd4a0'
            const time = new Date(it.ts).toLocaleTimeString('en-GB', { hour12: false })
            return (
              <div key={i} style={{
                fontSize: 10, lineHeight: 1.4,
                padding: '6px 8px',
                background: '#0a0d14',
                border: `1px solid ${color}22`,
                borderLeft: `2px solid ${color}`,
                borderRadius: 4,
              }}>
                <div style={{ color: '#3d4455', fontSize: 9 }}>{time}</div>
                <div style={{ color: '#d4d8e4', wordBreak: 'break-word' }}>{it.text}</div>
              </div>
            )
          })}
        </div>
      )}
    </aside>
  )
}

// ── Main shell component ──────────────────────────────────────────
export default function NeoShell() {
  const [tab, setTab] = useState<NeoTab>('mind')
  const [status, setStatus] = useState<NeoStatus | null>(null)
  const [activity, setActivity] = useState<ActivityItem[]>([])

  // Poll /api/neo/status
  useEffect(() => {
    let alive = true
    const load = () => {
      fetch('/api/neo/status')
        .then(r => r.ok ? r.json() : Promise.reject(new Error(String(r.status))))
        .then(d => { if (alive) setStatus(d) })
        .catch(() => { if (alive) setStatus(null) })
    }
    load()
    const iv = setInterval(load, 3000)
    return () => { alive = false; clearInterval(iv) }
  }, [])

  // Subscribe to QLMS stream for activity feed
  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    es.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data)
        const text = `${m.from ?? '?'} → ${m.to ?? '?'} · ${m.intent ?? 'msg'}`
        setActivity(a => {
          const next = [...a, { ts: Date.now(), text, kind: 'qlms' as const }]
          return next.slice(-200)
        })
      } catch { /* ignore */ }
    }
    es.onerror = () => {
      setActivity(a => {
        const last = a[a.length - 1]
        if (last?.text === 'QLMS stream disconnected') return a
        return [...a, { ts: Date.now(), text: 'QLMS stream disconnected', kind: 'warn' as const }]
      })
    }
    return () => es.close()
  }, [])

  // Emit heartbeat lines from status polls
  useEffect(() => {
    if (!status) return
    setActivity(a => {
      const text = `hb · spec=${status.specialists} gen=${status.organism_generation} mem=${status.hdc_memory + status.organism_memory_items}`
      const last = a[a.length - 1]
      if (last?.text === text) return a
      return [...a, { ts: Date.now(), text, kind: 'info' as const }].slice(-200)
    })
  }, [status?.organism_generation, status?.hdc_memory, status?.organism_memory_items, status?.specialists])

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', minHeight: 0,
      background: '#04060b',
      color: '#d4d8e4',
    }}>
      <StatusStrip status={status} />

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <LeftSidebar active={tab} onSelect={setTab} />

        <main style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
          {tab === 'mind'      && <NeoMind />}
          {tab === 'evolution' && <NeoEvolutionInline />}
          {tab === 'agents'    && <NeoAgents />}
          {tab === 'memory'    && <NeoMemory />}
          {tab === 'messages'  && <NeoMessages />}
          {tab === 'body'      && <NeoBody />}
          {tab === 'dialogue'  && <NeoDialogue />}
        </main>

        <RightActivity items={activity} />
      </div>
    </div>
  )
}
