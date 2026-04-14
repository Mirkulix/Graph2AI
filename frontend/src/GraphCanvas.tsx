import { useState, useEffect, useRef, useCallback } from 'react'

// ============================================================
// Types
// ============================================================

interface AgentNode {
  id: string
  x: number
  y: number
  color: string
  icon: string
  active: boolean
}

interface MessageEdge {
  id: string
  from: string
  to: string
  intent: string
  timestamp: number
  age: number // 0..1, fades out
}

interface LiveMessage {
  id: number
  from: string
  to: string
  intent: string
  graph_name: string
  timestamp: number
}

// ============================================================
// Constants
// ============================================================

const AGENT_COLORS: Record<string, string> = {
  ceo: '#3b82f6',
  researcher: '#8b5cf6',
  developer: '#22c55e',
  guardian: '#f59e0b',
  strategist: '#a78bfa',
  artisan: '#f472b6',
}

const AGENT_ICONS: Record<string, string> = {
  ceo: 'C',
  researcher: 'R',
  developer: 'D',
  guardian: 'G',
  strategist: 'S',
  artisan: 'A',
}

const NODE_RADIUS = 32
const EDGE_LIFETIME = 8000 // ms before edge fades

// ============================================================
// Layout: arrange agents in a circle
// ============================================================

function circleLayout(agents: string[], cx: number, cy: number, r: number): AgentNode[] {
  return agents.map((id, i) => {
    const angle = (i / agents.length) * Math.PI * 2 - Math.PI / 2
    return {
      id,
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
      color: AGENT_COLORS[id] ?? '#6b7280',
      icon: AGENT_ICONS[id] ?? id[0].toUpperCase(),
      active: false,
    }
  })
}

// ============================================================
// Component
// ============================================================

export default function GraphCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [agents, setAgents] = useState<string[]>([])
  const nodesRef = useRef<AgentNode[]>([])
  const edgesRef = useRef<MessageEdge[]>([])
  const dragRef = useRef<{ nodeId: string; offsetX: number; offsetY: number } | null>(null)
  const animRef = useRef<number>(0)
  const sizeRef = useRef<{ w: number; h: number }>({ w: 800, h: 480 })
  const [canvasReady, setCanvasReady] = useState(false)
  const [messageCount, setMessageCount] = useState(0)
  const [lastMessage, setLastMessage] = useState('')

  // Fetch agents and layout once canvas is sized
  useEffect(() => {
    fetch('/api/messages/agents')
      .then(r => r.json())
      .then(data => {
        const list = Array.isArray(data) ? data : []
        setAgents(list)
      })
      .catch(() => {})
  }, [])

  // Layout agents when both agents are loaded AND canvas is sized
  useEffect(() => {
    if (agents.length > 0 && canvasReady) {
      const { w, h } = sizeRef.current
      nodesRef.current = circleLayout(agents, w / 2, h / 2, Math.min(w, h) * 0.3)
    }
  }, [agents, canvasReady])

  // SSE stream for live messages
  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    es.onmessage = (event) => {
      try {
        const msg: LiveMessage = JSON.parse(event.data)
        const edge: MessageEdge = {
          id: `${msg.id}-${Date.now()}`,
          from: msg.from,
          to: msg.to,
          intent: msg.intent,
          timestamp: Date.now(),
          age: 0,
        }
        edgesRef.current.push(edge)
        setMessageCount(c => c + 1)

        const intentShort = msg.intent.includes('Result') ? 'Result' : msg.intent.includes('Execute') ? 'Execute' : msg.intent.slice(0, 12)
        setLastMessage(`${msg.from} → ${msg.to} [${intentShort}]`)

        // Flash active state on both agents
        const fromNode = nodesRef.current.find(n => n.id === msg.from)
        const toNode = nodesRef.current.find(n => n.id === msg.to)
        if (fromNode) fromNode.active = true
        if (toNode) toNode.active = true
        setTimeout(() => {
          if (fromNode) fromNode.active = false
          if (toNode) toNode.active = false
        }, 800)
      } catch {}
    }
    return () => es.close()
  }, [])

  // Canvas resize
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const resize = () => {
      const rect = parent.getBoundingClientRect()
      const w = Math.max(Math.floor(rect.width), 400)
      const h = 480 // fixed height for reliable rendering
      canvas.width = w * 2 // retina
      canvas.height = h * 2
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      sizeRef.current = { w, h }
      setCanvasReady(true)
    }

    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(parent)
    return () => observer.disconnect()
  }, [agents])

  // Drag handling
  const getMousePos = useCallback((e: React.MouseEvent): { x: number; y: number } => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }, [])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    const pos = getMousePos(e)
    for (const node of nodesRef.current) {
      const dx = pos.x - node.x
      const dy = pos.y - node.y
      if (dx * dx + dy * dy < NODE_RADIUS * NODE_RADIUS) {
        dragRef.current = { nodeId: node.id, offsetX: dx, offsetY: dy }
        return
      }
    }
  }, [getMousePos])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return
    const pos = getMousePos(e)
    const node = nodesRef.current.find(n => n.id === dragRef.current!.nodeId)
    if (node) {
      node.x = pos.x - dragRef.current.offsetX
      node.y = pos.y - dragRef.current.offsetY
    }
  }, [getMousePos])

  const onMouseUp = useCallback(() => {
    dragRef.current = null
  }, [])

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const draw = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const dpr = 2
      const { w, h } = sizeRef.current
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const now = Date.now()

      // Update edge ages, remove old ones
      edgesRef.current = edgesRef.current.filter(e => {
        e.age = (now - e.timestamp) / EDGE_LIFETIME
        return e.age < 1
      })

      // Draw edges (animated)
      for (const edge of edgesRef.current) {
        const fromNode = nodesRef.current.find(n => n.id === edge.from)
        const toNode = nodesRef.current.find(n => n.id === edge.to)
        if (!fromNode || !toNode) continue

        const alpha = 1 - edge.age
        const isResult = edge.intent.includes('Result')

        // Line
        ctx.beginPath()
        ctx.moveTo(fromNode.x, fromNode.y)
        ctx.lineTo(toNode.x, toNode.y)
        ctx.strokeStyle = isResult
          ? `rgba(34, 197, 94, ${alpha * 0.8})`
          : `rgba(59, 130, 246, ${alpha * 0.8})`
        ctx.lineWidth = 2.5 * alpha + 0.5
        ctx.stroke()

        // Animated dot traveling along the edge
        const t = Math.min(edge.age * 3, 1) // dot travels in first third of lifetime
        const dotX = fromNode.x + (toNode.x - fromNode.x) * t
        const dotY = fromNode.y + (toNode.y - fromNode.y) * t
        ctx.beginPath()
        ctx.arc(dotX, dotY, 5 * alpha + 2, 0, Math.PI * 2)
        ctx.fillStyle = isResult
          ? `rgba(34, 197, 94, ${alpha})`
          : `rgba(59, 130, 246, ${alpha})`
        ctx.fill()

        // Arrow head at midpoint
        const mx = fromNode.x + (toNode.x - fromNode.x) * 0.65
        const my = fromNode.y + (toNode.y - fromNode.y) * 0.65
        const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x)
        const arrowSize = 8 * alpha + 3
        ctx.beginPath()
        ctx.moveTo(mx + Math.cos(angle) * arrowSize, my + Math.sin(angle) * arrowSize)
        ctx.lineTo(mx + Math.cos(angle + 2.5) * arrowSize, my + Math.sin(angle + 2.5) * arrowSize)
        ctx.lineTo(mx + Math.cos(angle - 2.5) * arrowSize, my + Math.sin(angle - 2.5) * arrowSize)
        ctx.closePath()
        ctx.fillStyle = isResult
          ? `rgba(34, 197, 94, ${alpha * 0.9})`
          : `rgba(59, 130, 246, ${alpha * 0.9})`
        ctx.fill()

        // Intent label near midpoint
        if (alpha > 0.3) {
          const label = isResult ? 'Result' : 'Execute'
          ctx.font = `${11}px system-ui, sans-serif`
          ctx.fillStyle = `rgba(200, 200, 220, ${alpha * 0.8})`
          ctx.fillText(label, mx + 10, my - 8)
        }
      }

      // Draw nodes
      for (const node of nodesRef.current) {
        // Glow when active
        if (node.active) {
          ctx.beginPath()
          ctx.arc(node.x, node.y, NODE_RADIUS + 8, 0, Math.PI * 2)
          ctx.fillStyle = node.color + '30'
          ctx.fill()
        }

        // Circle
        ctx.beginPath()
        ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = '#1a1a2e'
        ctx.fill()
        ctx.strokeStyle = node.active ? node.color : node.color + '80'
        ctx.lineWidth = node.active ? 3 : 2
        ctx.stroke()

        // Icon letter
        ctx.font = 'bold 18px system-ui, sans-serif'
        ctx.fillStyle = node.color
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(node.icon, node.x, node.y)

        // Label
        ctx.font = '11px system-ui, sans-serif'
        ctx.fillStyle = '#9ca3af'
        ctx.fillText(node.id, node.x, node.y + NODE_RADIUS + 14)
      }

      animRef.current = requestAnimationFrame(draw)
    }

    animRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animRef.current)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '12px' }}>
      {/* Controls bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Messages: <strong style={{ color: 'var(--accent-primary)' }}>{messageCount}</strong>
        </span>
        {lastMessage && (
          <span style={{ fontSize: '12px', color: 'var(--accent-success)' }}>
            {lastMessage}
          </span>
        )}
        <div style={{ flex: 1 }} />
        <button
          className="btn btn-primary"
          style={{ fontSize: '12px', padding: '6px 14px' }}
          onClick={async () => {
            try {
              await fetch('/api/proof/tensor-exchange', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  text_a: 'Rust ist eine Systemprogrammiersprache',
                  text_b: 'Python ist eine Skriptsprache',
                }),
              })
            } catch {}
          }}
        >
          Tensor-Proof senden
        </button>
        <button
          className="btn"
          style={{ fontSize: '12px', padding: '6px 14px' }}
          onClick={() => {
            const { w, h } = sizeRef.current
            nodesRef.current = circleLayout(agents, w / 2, h / 2, Math.min(w, h) * 0.32)
          }}
        >
          Reset Layout
        </button>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', background: '#0d1117', minHeight: '400px' }}>
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          style={{ cursor: dragRef.current ? 'grabbing' : 'grab', display: 'block', width: '100%' }}
        />
      </div>

      <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center' }}>
        Agenten mit der Maus verschieben. Blaue Linien = Execute, Gruene = Result. Punkte zeigen Datenfluss.
      </div>
    </div>
  )
}
