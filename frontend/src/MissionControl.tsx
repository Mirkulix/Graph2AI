// MissionControl.tsx — live operations room for the agent swarm.
//
// Layout:
//   ┌──────────────────────────────────────────┬─────────────────┐
//   │  Live DAG (react-flow)                   │ Agent Roster    │
//   │                                          │ Werte-Radar     │
//   ├──────────────────────────────────────────┤                 │
//   │  Event Timeline (last 30 GraphEvents)    │ Demo Trigger    │
//   └──────────────────────────────────────────┴─────────────────┘
//
// Data sources:
//   • GET /api/agents       (3 s poll — roster + task counters)
//   • WS  /ws/graph-stream  (GraphEvent broadcast from message bus)
//   • POST /api/goals       (Demo-Trigger button)

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  Activity,
  PlayCircle,
  Signal,
  SignalHigh,
  SignalLow,
  Zap,
} from 'lucide-react'
import ValuesRadar from './ValuesRadar'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GraphEvent {
  timestamp_ms: number
  from: string
  to: string
  intent: string
  size_bytes: number
  status?: string | null
}

interface AgentRow {
  role: string
  status: string
  tasks_completed: number
  tasks_failed: number
}

type NodeStatus = 'idle' | 'active' | 'error'

interface NodeData extends Record<string, unknown> {
  label: string
  status: NodeStatus
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PULSE_MS = 1_500
const MAX_EDGES = 40
const MAX_TIMELINE = 30
const BYTES_AT_MAX_WIDTH = 8_192

const AGENT_ORDER = ['ceo', 'researcher', 'developer', 'guardian', 'strategist', 'artisan']

const AGENT_BLURB: Record<string, string> = {
  ceo: 'Orchestriert · zerlegt Ziele in Teilaufgaben',
  researcher: 'Recherche · sammelt Fakten',
  developer: 'Technik · entwirft Code & Architektur',
  guardian: 'Prüft Aktionen gegen die 5 Kernwerte',
  strategist: 'Evaluiert Optionen · wählt Vorgehen',
  artisan: 'Formuliert · schreibt Texte',
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function circlePosition(index: number, total: number): { x: number; y: number } {
  if (total <= 1) return { x: 280, y: 220 }
  const radius = 160
  const angle = (index / total) * 2 * Math.PI - Math.PI / 2
  return { x: 280 + Math.cos(angle) * radius, y: 220 + Math.sin(angle) * radius }
}

function edgeWidthFromBytes(bytes: number): number {
  const clamped = Math.max(0, Math.min(BYTES_AT_MAX_WIDTH, bytes))
  return 1 + (clamped / BYTES_AT_MAX_WIDTH) * 4
}

function cssColour(status: NodeStatus): string {
  if (status === 'active') return 'var(--accent)'
  if (status === 'error') return 'var(--err)'
  return 'var(--border-strong)'
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function formatAgoMs(ms: number): string {
  const diffS = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (diffS < 5) return 'jetzt'
  if (diffS < 60) return `${diffS}s`
  if (diffS < 3600) return `${Math.floor(diffS / 60)}m`
  return `${Math.floor(diffS / 3600)}h`
}

function idleNodeStyle(): React.CSSProperties {
  return {
    border: `1.4px solid ${cssColour('idle')}`,
    background: 'var(--surface)',
    color: 'var(--text)',
    borderRadius: 10,
    padding: '6px 12px',
    fontSize: 12,
    fontWeight: 500,
    boxShadow: 'var(--shadow-card)',
  }
}

function activeNodeStyle(status: NodeStatus): React.CSSProperties {
  const soft = status === 'error' ? 'var(--err-soft)' : 'var(--accent-soft)'
  return {
    ...idleNodeStyle(),
    border: `1.4px solid ${cssColour(status)}`,
    boxShadow: `0 0 0 3px ${soft}`,
  }
}

function isGraphEvent(value: unknown): value is GraphEvent {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return (
    typeof v.from === 'string' &&
    typeof v.to === 'string' &&
    typeof v.intent === 'string' &&
    typeof v.size_bytes === 'number' &&
    typeof v.timestamp_ms === 'number'
  )
}

function readAuthToken(): string | null {
  const v = localStorage.getItem('qo_auth_token')
  return v && v.trim().length > 0 ? v.trim() : null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function MissionControlInner() {
  const [nodes, setNodes] = useState<Node<NodeData>[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [timeline, setTimeline] = useState<GraphEvent[]>([])
  const [connState, setConnState] = useState<'idle' | 'live' | 'error'>('idle')
  const [eventCount, setEventCount] = useState(0)
  const [lastError, setLastError] = useState<string | null>(null)
  const [agents, setAgents] = useState<AgentRow[]>([])
  const [demoBusy, setDemoBusy] = useState(false)
  const [demoToast, setDemoToast] = useState<string | null>(null)

  const pulseTimers = useRef<Map<string, number>>(new Map())
  const eventsInLastMin = useRef<number[]>([])

  // ─── Agent roster poll ──────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    async function pull() {
      try {
        const r = await fetch('/api/agents')
        if (!r.ok) return
        const body = await r.json()
        if (!cancelled && Array.isArray(body.agents)) {
          setAgents(body.agents as AgentRow[])
        }
      } catch {
        /* ignore — roster is best-effort */
      }
    }
    void pull()
    const h = window.setInterval(pull, 3_000)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  // ─── Pulse animation helpers ────────────────────────────────────────
  const clearPulseTimer = useCallback((id: string) => {
    const t = pulseTimers.current.get(id)
    if (t !== undefined) {
      window.clearTimeout(t)
      pulseTimers.current.delete(id)
    }
  }, [])

  const schedulePulse = useCallback(
    (id: string, status: NodeStatus) => {
      clearPulseTimer(id)
      const handle = window.setTimeout(() => {
        setNodes((prev) =>
          prev.map((n) =>
            n.id === id ? { ...n, data: { ...n.data, status: 'idle' }, style: idleNodeStyle() } : n,
          ),
        )
        pulseTimers.current.delete(id)
      }, PULSE_MS)
      pulseTimers.current.set(id, handle)
      setNodes((prev) =>
        prev.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, status }, style: activeNodeStyle(status) } : n,
        ),
      )
    },
    [clearPulseTimer],
  )

  // ─── Event ingestion ────────────────────────────────────────────────
  const ingest = useCallback(
    (evt: GraphEvent) => {
      setEventCount((c) => c + 1)
      eventsInLastMin.current = [
        ...eventsInLastMin.current.filter((t) => Date.now() - t < 60_000),
        Date.now(),
      ]
      setTimeline((prev) => [evt, ...prev].slice(0, MAX_TIMELINE))

      const { from, to, intent, size_bytes } = evt
      const failed = evt.status === 'error'

      setNodes((prev) => {
        const known = new Set(prev.map((n) => n.id))
        const out = [...prev]
        let changed = false
        for (const id of [from, to]) {
          if (!known.has(id)) {
            out.push({
              id,
              position: circlePosition(out.length, out.length + 1),
              data: { label: id, status: 'idle' },
              style: idleNodeStyle(),
              draggable: true,
            })
            known.add(id)
            changed = true
          }
        }
        if (!changed) return prev
        return out.map((n, i, arr) => ({ ...n, position: circlePosition(i, arr.length) }))
      })

      setEdges((prev) => {
        const id = `${from}->${to}#${evt.timestamp_ms}`
        const next: Edge = {
          id,
          source: from,
          target: to,
          label: `${intent} · ${formatBytes(size_bytes)}`,
          animated: !failed,
          style: {
            stroke: failed ? 'var(--err)' : 'var(--accent)',
            strokeWidth: edgeWidthFromBytes(size_bytes),
            opacity: 0.9,
          },
          labelStyle: { fill: 'var(--text-muted)', fontSize: 10 },
        }
        const combined = [...prev, next]
        if (combined.length > MAX_EDGES) combined.splice(0, combined.length - MAX_EDGES)
        return combined
      })

      schedulePulse(to, failed ? 'error' : 'active')
    },
    [schedulePulse],
  )

  // ─── WebSocket wiring ───────────────────────────────────────────────
  useEffect(() => {
    const token = readAuthToken()
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const base = `${proto}//${window.location.host}/ws/graph-stream`
    const url = token ? `${base}?token=${encodeURIComponent(token)}` : base

    const ws = new WebSocket(url)
    ws.onopen = () => {
      setConnState('live')
      setLastError(null)
    }
    ws.onclose = () => setConnState('idle')
    ws.onerror = () => {
      setConnState('error')
      setLastError('WebSocket-Fehler')
    }
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data)
        if (data && typeof data === 'object' && 'type' in data) {
          if (data.type === 'lagged') {
            setLastError(`Stream hinkt nach: ${data.missed ?? '?'} Events verloren`)
            return
          }
          if (data.type === 'hello') return
        }
        if (isGraphEvent(data)) ingest(data)
      } catch (e) {
        console.warn('graph-stream parse failed', e)
      }
    }

    return () => {
      ws.close()
      for (const handle of pulseTimers.current.values()) window.clearTimeout(handle)
      pulseTimers.current.clear()
    }
  }, [ingest])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) =>
      setNodes((nds) => applyNodeChanges(changes, nds) as Node<NodeData>[]),
    [],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  )

  // ─── Derived stats ──────────────────────────────────────────────────
  const perMinute = useMemo(() => {
    const fresh = eventsInLastMin.current.filter((t) => Date.now() - t < 60_000)
    return fresh.length
  }, [eventCount])

  const lastIntent = timeline[0]?.intent ?? '—'

  const connChip = useMemo(() => {
    if (connState === 'error') {
      return (
        <span className="pill pill--err">
          <span className="dot dot--err" /> Fehler
        </span>
      )
    }
    if (connState === 'live') {
      return (
        <span className="pill pill--ok">
          <span className="dot dot--ok dot--pulse" /> Live
        </span>
      )
    }
    return <span className="pill">Verbinde …</span>
  }, [connState])

  // ─── Demo trigger ───────────────────────────────────────────────────
  // POSTs against /api/chat with a goal-shaped prompt so the intent
  // classifier routes it to execute_goal_background() — that is the
  // orchestrator path that actually fires QLMS messages between agents.
  const triggerDemoGoal = useCallback(async () => {
    if (demoBusy) return
    setDemoBusy(true)
    setDemoToast(null)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message:
            'Entwickle einen knappen 3-Punkte-Plan, was QLANG besonders macht. Aktiviere dabei alle Agenten.',
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setDemoToast(
        'Ziel gestartet — CEO dekomponiert gerade. Events fliessen gleich ein.',
      )
    } catch (e) {
      setDemoToast(`Fehler: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setDemoBusy(false)
      window.setTimeout(() => setDemoToast(null), 8_000)
    }
  }, [demoBusy])

  // ─── Render ─────────────────────────────────────────────────────────
  return (
    <div className="mission">
      {/* LEFT column: graph + timeline */}
      <div className="mission__left">
        <div className="mission__stats">
          <div className="mission__stat">
            <span className="mission__stat-label">Status</span>
            <span className="mission__stat-value">{connChip}</span>
          </div>
          <div className="mission__stat">
            <span className="mission__stat-label">Events gesamt</span>
            <span className="mission__stat-value mission__stat-value--num">
              {eventCount}
            </span>
          </div>
          <div className="mission__stat">
            <span className="mission__stat-label">/ Minute</span>
            <span className="mission__stat-value mission__stat-value--num">
              {perMinute}
            </span>
          </div>
          <div className="mission__stat">
            <span className="mission__stat-label">Letzter Intent</span>
            <span className="mission__stat-value mission__stat-value--code">
              {lastIntent}
            </span>
          </div>
          <button
            type="button"
            className="mission__demo-btn"
            onClick={triggerDemoGoal}
            disabled={demoBusy}
            title="Löst ein echtes Ziel aus, das alle Agenten aktiviert"
          >
            <PlayCircle size={14} />
            {demoBusy ? 'Starte …' : 'Demo auslösen'}
          </button>
        </div>

        {demoToast ? <div className="mission__toast">{demoToast}</div> : null}
        {lastError ? <div className="mission__toast mission__toast--warn">{lastError}</div> : null}

        <div className="mission__graph">
          {nodes.length === 0 ? (
            <div className="mission__empty">
              <Activity size={28} strokeWidth={1.5} />
              <div>
                Noch keine Agent-Kommunikation beobachtet.
                <br />
                Starte ein Ziel über <b>Demo auslösen</b> oder tippe auf{' '}
                <b>Home</b> eine Aufgabe ein.
              </div>
            </div>
          ) : null}

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView={nodes.length > 0}
            minZoom={0.4}
            maxZoom={1.8}
            proOptions={{ hideAttribution: true }}
            nodesDraggable
            nodesConnectable={false}
            elementsSelectable={false}
          >
            <Background color="rgba(120,128,140,0.18)" gap={24} />
          </ReactFlow>
        </div>

        <div className="mission__timeline">
          <div className="mission__panel-header">
            <Zap size={14} />
            <span>Event-Verlauf</span>
            <span className="mission__panel-sub">
              letzte {MAX_TIMELINE} · neueste zuerst
            </span>
          </div>
          {timeline.length === 0 ? (
            <div className="mission__timeline-empty">
              Sobald Agenten Nachrichten austauschen, erscheinen sie hier.
            </div>
          ) : (
            <ul className="mission__timeline-list">
              {timeline.map((e, i) => (
                <li key={`${e.timestamp_ms}-${i}`} className="mission__timeline-item">
                  <span className="mission__timeline-time">{formatAgoMs(e.timestamp_ms)}</span>
                  <span className="mission__timeline-route">
                    <code>{e.from}</code>
                    <span className="mission__timeline-arrow">→</span>
                    <code>{e.to}</code>
                  </span>
                  <span className="mission__timeline-intent">{e.intent}</span>
                  <span className="mission__timeline-size">{formatBytes(e.size_bytes)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* RIGHT sidebar: roster + radar */}
      <aside className="mission__side">
        <div className="mission__roster">
          <div className="mission__panel-header">
            <Signal size={14} />
            <span>Agenten ({agents.length})</span>
          </div>
          <ul className="mission__roster-list">
            {[...agents]
              .sort(
                (a, b) =>
                  AGENT_ORDER.indexOf(a.role.toLowerCase()) -
                  AGENT_ORDER.indexOf(b.role.toLowerCase()),
              )
              .map((a) => (
                <li key={a.role} className="mission__roster-item">
                  <AgentStatusIcon status={a.status} />
                  <div className="mission__roster-text">
                    <div className="mission__roster-name">{a.role}</div>
                    <div className="mission__roster-blurb">
                      {AGENT_BLURB[a.role.toLowerCase()] ?? '—'}
                    </div>
                  </div>
                  <div className="mission__roster-counts" title="fertig / fehlgeschlagen">
                    <span className="mission__roster-ok">{a.tasks_completed}</span>
                    {a.tasks_failed > 0 ? (
                      <span className="mission__roster-fail"> / {a.tasks_failed}</span>
                    ) : null}
                  </div>
                </li>
              ))}
          </ul>
        </div>

        <ValuesRadar />
      </aside>
    </div>
  )
}

function AgentStatusIcon({ status }: { status: string }) {
  const lower = status.toLowerCase()
  if (lower.includes('active') || lower.includes('working')) {
    return (
      <span className="mission__roster-dot mission__roster-dot--active" title={status}>
        <SignalHigh size={14} />
      </span>
    )
  }
  if (lower.includes('error') || lower.includes('fail')) {
    return (
      <span className="mission__roster-dot mission__roster-dot--error" title={status}>
        <Signal size={14} />
      </span>
    )
  }
  return (
    <span className="mission__roster-dot mission__roster-dot--idle" title={status}>
      <SignalLow size={14} />
    </span>
  )
}

export default function MissionControl() {
  return (
    <ReactFlowProvider>
      <MissionControlInner />
    </ReactFlowProvider>
  )
}
