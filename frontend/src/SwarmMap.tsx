// SwarmMap.tsx — PRD Task 6.4 / UI-Konzept "Swarm & Federation".
//
// 3D visualisation of the QO federation: every known peer becomes a node,
// local nodes light up blue, remote peers stay grey. The center of the
// graph is ringed with pulse animations whenever the peer-discovery loop
// ticks. The top-bar metrics come from /api/federation/stats.
//
// The heavy lifting (3D rendering, force simulation) is delegated to
// react-force-graph-3d + three.js, both already pulled in by
// KnowledgeGraph3DView. No new dependencies for this tab.

import { useEffect, useMemo, useRef, useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — react-force-graph-3d ships its own .d.ts that imports the
// (untyped) three namespace. We don't hit that surface from our code, but
// the generic type parameter would otherwise require @types/three.
import ForceGraph3D, { ForceGraphMethods } from 'react-force-graph-3d'
import { Globe, Wifi, WifiOff } from 'lucide-react'

interface PeerEntry {
  id: string
  addr: string
  is_local: boolean
}

interface PeersResponse {
  total_nodes: number
  local_nodes: number
  peers: PeerEntry[]
}

interface FederationStats {
  rounds_completed: number
  rounds_failed: number
  gossip_rate_per_sec: number
  mean_interval_ms: number
  bandwidth_saved_pct: number
  peer_discovery_enabled: boolean
}

interface GraphNode {
  id: string
  addr: string
  is_local: boolean
  size: number
}

interface GraphLink {
  source: string
  target: string
}

const PEERS_POLL_MS = 15_000
const STATS_POLL_MS = 5_000

export default function SwarmMap() {
  const [peers, setPeers] = useState<PeerEntry[]>([])
  const [stats, setStats] = useState<FederationStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined)

  // --- poll peers -----------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    async function tick() {
      try {
        const r = await fetch('/api/federation/peers')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const body = (await r.json()) as PeersResponse
        if (!cancelled) {
          setPeers(body.peers)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }
    void tick()
    const h = window.setInterval(tick, PEERS_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  // --- poll stats -----------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    async function tick() {
      try {
        const r = await fetch('/api/federation/stats')
        if (!r.ok) return
        const body = (await r.json()) as FederationStats
        if (!cancelled) setStats(body)
      } catch {
        /* ignore — stats are best-effort */
      }
    }
    void tick()
    const h = window.setInterval(tick, STATS_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  // --- graph data ----------------------------------------------------------
  const graphData = useMemo(() => {
    const nodes: GraphNode[] = peers.map((p) => ({
      id: p.id,
      addr: p.addr,
      is_local: p.is_local,
      size: p.is_local ? 8 : 5,
    }))
    // Full mesh: every pair gets a link, so force-layout spreads nodes
    // in a ring. Cheap to compute at N ≤ ~50 (realistic swarm size).
    const links: GraphLink[] = []
    for (let i = 0; i < peers.length; i++) {
      for (let j = i + 1; j < peers.length; j++) {
        links.push({ source: peers[i].id, target: peers[j].id })
      }
    }
    return { nodes, links }
  }, [peers])

  // Small auto-fit when peers change so the camera always frames the swarm.
  useEffect(() => {
    if (!fgRef.current) return
    const handle = window.setTimeout(() => {
      try {
        fgRef.current?.zoomToFit(600, 40)
      } catch {
        /* ignore transient render-before-ready */
      }
    }, 400)
    return () => window.clearTimeout(handle)
  }, [graphData])

  const connectedBadge = stats?.peer_discovery_enabled ? (
    <span className="swarm-map__badge swarm-map__badge--on">
      <Wifi size={12} /> Peer Discovery aktiv
    </span>
  ) : (
    <span className="swarm-map__badge swarm-map__badge--off">
      <WifiOff size={12} /> PEER_DISCOVERY_SEEDS unset
    </span>
  )

  return (
    <div className="swarm-map">
      <header className="swarm-map__header">
        <div className="swarm-map__title">
          <Globe size={16} />
          <h2>Swarm &amp; Federation</h2>
        </div>
        <div className="swarm-map__metrics">
          <Metric label="Total Nodes" value={peers.length} />
          <Metric
            label="Local"
            value={peers.filter((p) => p.is_local).length}
          />
          <Metric
            label="Gossip Rate"
            value={`${(stats?.gossip_rate_per_sec ?? 0).toFixed(2)}/s`}
          />
          <Metric
            label="Bandwidth saved vs JSON"
            value={`${(stats?.bandwidth_saved_pct ?? 0).toFixed(0)}%`}
          />
          <Metric
            label="Rounds"
            value={stats ? `${stats.rounds_completed}/${stats.rounds_completed + stats.rounds_failed}` : '—'}
          />
        </div>
        {connectedBadge}
      </header>

      {error ? (
        <div className="swarm-map__error">Fehler: {error}</div>
      ) : null}

      <div className="swarm-map__canvas">
        {peers.length === 0 ? (
          <div className="swarm-map__empty">
            Noch keine Peers bekannt. Setze <code>QO_NODE_ADDR</code> und <code>PEER_DISCOVERY_SEEDS</code> oder starte die Swarm-Compose:
            <br />
            <code>docker compose -f docker-compose.swarm.yml up</code>
          </div>
        ) : (
          <ForceGraph3D
            ref={fgRef}
            graphData={graphData}
            nodeLabel={(n: GraphNode) => `${n.id} (${n.addr})`}
            nodeVal={(n: GraphNode) => n.size}
            nodeColor={(n: GraphNode) =>
              n.is_local ? '#3b7de8' : '#70757f'
            }
            nodeOpacity={0.95}
            linkColor={() => 'rgba(255, 255, 255, 0.15)'}
            linkOpacity={0.3}
            backgroundColor="rgba(12, 14, 20, 1)"
            showNavInfo={false}
            controlType="orbit"
            enableNodeDrag={false}
          />
        )}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="swarm-map__metric">
      <span className="swarm-map__metric-label">{label}</span>
      <span className="swarm-map__metric-value">{value}</span>
    </div>
  )
}
