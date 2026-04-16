// MissionControl.tsx — PRD Task 6.1 / UI-Konzept "Mission Control".
//
// Subscribes to /ws/graph-stream, renders a live DAG of QLANG agents
// using @xyflow/react. Each incoming GraphEvent:
//   * ensures the `from` and `to` agents exist as nodes
//   * adds (or refreshes) the edge between them
//   * flashes the receiving agent as "active" for ~1.5s
//   * styles edge thickness proportional to `size_bytes`
//
// Nodes live on a deterministic circular layout so the same agent
// always sits in the same slot — easier to follow the flow than a
// force-directed animation that re-shuffles on every event.
//
// The right-hand sidebar hosts the Werte-Radar so the Guardian's
// current assessment is always visible next to the live traffic.

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
import { Activity } from 'lucide-react'
import ValuesRadar from './ValuesRadar'

interface GraphEvent {
  timestamp_ms: number
  from: string
  to: string
  intent: string
  size_bytes: number
  status?: string | null
}

type NodeStatus = 'idle' | 'active' | 'error'

interface NodeData extends Record<string, unknown> {
  label: string
  status: NodeStatus
  lastIntent?: string
}

const PULSE_MS = 1_500
const MAX_EDGE_WIDTH = 5
const MIN_EDGE_WIDTH = 1
const BYTES_AT_MAX_WIDTH = 8_192
const MAX_EDGES = 50 // prevent unbounded growth

/**
 * Lay out `n` agents evenly around a circle. The x/y offsets keep the
 * whole layout comfortably inside the default viewport.
 */
function circlePosition(index: number, total: number): { x: number; y: number } {
  if (total <= 1) {
    return { x: 280, y: 220 }
  }
  const radius = 160
  const angle = (index / total) * 2 * Math.PI - Math.PI / 2
  return {
    x: 280 + Math.cos(angle) * radius,
    y: 220 + Math.sin(angle) * radius,
  }
}

function edgeWidthFromBytes(bytes: number): number {
  const clamped = Math.max(0, Math.min(BYTES_AT_MAX_WIDTH, bytes))
  const ratio = clamped / BYTES_AT_MAX_WIDTH
  return MIN_EDGE_WIDTH + ratio * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH)
}

function cssColour(status: NodeStatus): string {
  switch (status) {
    case 'active':
      return 'var(--accent-sinn, #3b7de8)'
    case 'error':
      return 'var(--accent-error, #e8545d)'
    default:
      return 'rgba(255, 255, 255, 0.24)'
  }
}

function MissionControlInner() {
  const [nodes, setNodes] = useState<Node<NodeData>[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [connState, setConnState] = useState<'idle' | 'live' | 'error'>('idle')
  const [eventCount, setEventCount] = useState(0)
  const [lastError, setLastError] = useState<string | null>(null)

  // Per-node pulse timers so an "active" flash clears cleanly even if a
  // fresh event arrives mid-flight.
  const pulseTimers = useRef<Map<string, number>>(new Map())

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
          n.id === id
            ? { ...n, data: { ...n.data, status }, style: activeNodeStyle(status) }
            : n,
        ),
      )
    },
    [clearPulseTimer],
  )

  const ingest = useCallback(
    (evt: GraphEvent) => {
      setEventCount((c) => c + 1)
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
        // Re-layout so new additions are evenly distributed.
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
            stroke: failed ? 'var(--accent-error, #e8545d)' : 'var(--accent-sinn, #3b7de8)',
            strokeWidth: edgeWidthFromBytes(size_bytes),
            opacity: 0.9,
          },
          labelStyle: { fill: 'var(--text-secondary, #a5a9b3)', fontSize: 10 },
        }
        const combined = [...prev, next]
        if (combined.length > MAX_EDGES) {
          combined.splice(0, combined.length - MAX_EDGES)
        }
        return combined
      })

      schedulePulse(to, failed ? 'error' : 'active')
    },
    [schedulePulse],
  )

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
        // Control frames: { type: "hello" } and { type: "lagged", missed }
        if (data && typeof data === 'object' && 'type' in data) {
          if (data.type === 'lagged') {
            setLastError(`Stream hinkt nach: ${data.missed ?? '?'} Events verloren`)
            return
          }
          if (data.type === 'hello') {
            return // handshake — already reflected by onopen
          }
        }
        if (isGraphEvent(data)) {
          ingest(data)
        }
      } catch (e) {
        console.warn('graph-stream parse failed', e)
      }
    }

    return () => {
      ws.close()
      // Cancel any lingering pulse timers so state does not leak.
      for (const handle of pulseTimers.current.values()) {
        window.clearTimeout(handle)
      }
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

  const status = useMemo(() => {
    if (connState === 'error') return { cls: 'error', label: lastError ?? 'Fehler' }
    if (connState === 'live') return { cls: 'live', label: `Live · ${eventCount} Events` }
    return { cls: 'idle', label: 'Offline' }
  }, [connState, eventCount, lastError])

  return (
    <div className="mission-control">
      <div className="mission-control__graph">
        <div className="mission-control__header">
          <span className={`mission-control__status-dot mission-control__status-dot--${status.cls}`} />
          <Activity size={14} />
          <h2>Mission Control</h2>
          <span>{status.label}</span>
        </div>

        {nodes.length === 0 ? (
          <div className="mission-control__empty">
            Noch keine Agenten-Kommunikation.
            <br />
            Sobald Events über <code>/ws/graph-stream</code> fliessen, erscheinen sie hier.
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
          <Background color="rgba(255,255,255,0.05)" gap={24} />
        </ReactFlow>
      </div>

      <aside className="mission-control__side">
        <ValuesRadar />
      </aside>
    </div>
  )
}

export default function MissionControl() {
  return (
    <ReactFlowProvider>
      <MissionControlInner />
    </ReactFlowProvider>
  )
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function idleNodeStyle(): React.CSSProperties {
  return {
    border: `1.4px solid ${cssColour('idle')}`,
    background: 'rgba(20, 24, 32, 0.88)',
    color: 'var(--text-primary, #e6e8ec)',
    borderRadius: 10,
    padding: '6px 12px',
    fontSize: 12,
    fontWeight: 500,
  }
}

function activeNodeStyle(status: NodeStatus): React.CSSProperties {
  const colour = cssColour(status)
  return {
    ...idleNodeStyle(),
    border: `1.4px solid ${colour}`,
    boxShadow: `0 0 0 3px ${colour.replace('var(', 'rgba(').replace(')', ', 0.22)')}`,
  }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
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
