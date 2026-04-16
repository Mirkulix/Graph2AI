// SwarmMap.tsx — federation / swarm overview.
//
// Layout:
//   ┌──────────────────────────────────────────────┬────────────────┐
//   │  3D peer-graph (react-force-graph-3d)        │ Bandwidth card │
//   │  Local nodes blue, remote grey               │ Gossip-History │
//   ├──────────────────────────────────────────────┤ Trigger btn    │
//   │  Metric bar: total/local/rate/rounds/saved   │                │
//   └──────────────────────────────────────────────┴────────────────┘
//
// Data sources:
//   • GET  /api/federation/peers  (15 s poll)
//   • GET  /api/federation/stats  (5 s poll)
//   • POST /api/qlms/federation/gossip { peers: [] } — trigger round

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — react-force-graph-3d .d.ts imports `three` types we don't need
import ForceGraph3D, { ForceGraphMethods } from 'react-force-graph-3d'
import {
  CircleDashed,
  Globe,
  History as HistoryIcon,
  PlayCircle,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface GossipEntry {
  when: number
  peers_ok: number
  peers_failed: number
  message?: string
  error?: string
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PEERS_POLL_MS = 15_000
const STATS_POLL_MS = 5_000
const MAX_HISTORY = 12

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SwarmMap() {
  const [peers, setPeers] = useState<PeerEntry[]>([])
  const [stats, setStats] = useState<FederationStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<GossipEntry[]>([])
  const [gossipBusy, setGossipBusy] = useState(false)
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined)
  const lastRoundsRef = useRef<number>(0)

  // ─── Polls ──────────────────────────────────────────────────────────
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

  useEffect(() => {
    let cancelled = false
    async function tick() {
      try {
        const r = await fetch('/api/federation/stats')
        if (!r.ok) return
        const body = (await r.json()) as FederationStats
        if (cancelled) return
        setStats((prev) => {
          // When rounds_completed increments, add an auto-history entry so
          // the history panel reflects background peer-discovery rounds
          // even when the user didn't click the manual trigger.
          if (prev && body.rounds_completed > prev.rounds_completed) {
            const diff = body.rounds_completed - prev.rounds_completed
            setHistory((h) =>
              [
                {
                  when: Date.now(),
                  peers_ok: 0,
                  peers_failed: 0,
                  message: `Auto-Gossip ×${diff}`,
                },
                ...h,
              ].slice(0, MAX_HISTORY),
            )
          }
          lastRoundsRef.current = body.rounds_completed
          return body
        })
      } catch {
        /* ignore */
      }
    }
    void tick()
    const h = window.setInterval(tick, STATS_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  // ─── Graph data ─────────────────────────────────────────────────────
  const graphData = useMemo(() => {
    const nodes: GraphNode[] = peers.map((p) => ({
      id: p.id,
      addr: p.addr,
      is_local: p.is_local,
      size: p.is_local ? 8 : 5,
    }))
    const links: GraphLink[] = []
    for (let i = 0; i < peers.length; i++) {
      for (let j = i + 1; j < peers.length; j++) {
        links.push({ source: peers[i].id, target: peers[j].id })
      }
    }
    return { nodes, links }
  }, [peers])

  useEffect(() => {
    if (!fgRef.current) return
    const handle = window.setTimeout(() => {
      try {
        fgRef.current?.zoomToFit(600, 40)
      } catch {
        /* ignore */
      }
    }, 400)
    return () => window.clearTimeout(handle)
  }, [graphData])

  // ─── Manual gossip trigger ──────────────────────────────────────────
  const triggerGossip = useCallback(async () => {
    if (gossipBusy) return
    setGossipBusy(true)
    try {
      const remotePeers = peers.filter((p) => !p.is_local).map((p) => p.addr)
      const res = await fetch('/api/qlms/federation/gossip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ peers: remotePeers }),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        setHistory((h) =>
          [
            {
              when: Date.now(),
              peers_ok: 0,
              peers_failed: remotePeers.length,
              error: `HTTP ${res.status}${text ? ': ' + text.slice(0, 120) : ''}`,
            },
            ...h,
          ].slice(0, MAX_HISTORY),
        )
      } else {
        const body = await res.json()
        setHistory((h) =>
          [
            {
              when: Date.now(),
              peers_ok: body.peers_ok?.length ?? 0,
              peers_failed: body.peers_failed?.length ?? 0,
              message: body.weight_changes !== undefined ? `Δ ${body.weight_changes} Gewichte` : undefined,
            },
            ...h,
          ].slice(0, MAX_HISTORY),
        )
      }
    } catch (e) {
      setHistory((h) =>
        [
          {
            when: Date.now(),
            peers_ok: 0,
            peers_failed: 0,
            error: e instanceof Error ? e.message : String(e),
          },
          ...h,
        ].slice(0, MAX_HISTORY),
      )
    } finally {
      setGossipBusy(false)
    }
  }, [gossipBusy, peers])

  // ─── Derived ────────────────────────────────────────────────────────
  const total = peers.length
  const local = peers.filter((p) => p.is_local).length
  const remote = total - local
  const totalRounds = (stats?.rounds_completed ?? 0) + (stats?.rounds_failed ?? 0)
  const failRate =
    totalRounds > 0 ? ((stats?.rounds_failed ?? 0) / totalRounds) * 100 : 0

  const connectedBadge = stats?.peer_discovery_enabled ? (
    <span className="swarm__badge swarm__badge--on">
      <Wifi size={12} /> Peer Discovery aktiv
    </span>
  ) : (
    <span className="swarm__badge swarm__badge--off">
      <WifiOff size={12} /> Peer Discovery aus
    </span>
  )

  return (
    <div className="swarm">
      {/* ── Top: title + metrics + trigger ── */}
      <header className="swarm__header">
        <div className="swarm__title">
          <Globe size={18} />
          <h2>Swarm &amp; Föderation</h2>
        </div>

        <div className="swarm__metrics">
          <Metric label="Nodes" value={total} sub={`${local} lokal · ${remote} remote`} />
          <Metric
            label="Gossip-Rate"
            value={`${(stats?.gossip_rate_per_sec ?? 0).toFixed(2)}/s`}
            sub={`Intervall ~${Math.round((stats?.mean_interval_ms ?? 0) / 1000)}s`}
          />
          <Metric
            label="Runden"
            value={stats ? `${stats.rounds_completed}` : '—'}
            sub={`${stats?.rounds_failed ?? 0} Fehler · ${failRate.toFixed(0)} %`}
          />
          <Metric
            label="Bandbreite"
            value={`−${Math.round(stats?.bandwidth_saved_pct ?? 0)} %`}
            sub="vs JSON (gemessen)"
            accent
          />
        </div>

        {connectedBadge}

        <button
          type="button"
          className="swarm__trigger"
          onClick={triggerGossip}
          disabled={gossipBusy || peers.length === 0}
          title={
            peers.length === 0
              ? 'Noch keine Peers bekannt'
              : 'Löst einen Gossip-Round gegen alle Remote-Peers aus'
          }
        >
          <PlayCircle size={14} />
          {gossipBusy ? 'Gossip läuft …' : 'Gossip auslösen'}
        </button>
      </header>

      {error ? <div className="swarm__error">Fehler: {error}</div> : null}

      {/* ── Main: 3D canvas + side panels ── */}
      <div className="swarm__body">
        <div className="swarm__canvas">
          {peers.length === 0 ? (
            <div className="swarm__empty">
              <Globe size={32} strokeWidth={1.5} />
              <div>
                Noch keine Peers bekannt.
                <br />
                Setze <code>QO_NODE_ADDR</code> und{' '}
                <code>PEER_DISCOVERY_SEEDS</code> oder starte die Swarm-Compose:
                <br />
                <code>docker compose -f docker-compose.swarm.yml up</code>
              </div>
            </div>
          ) : (
            <ForceGraph3D
              ref={fgRef}
              graphData={graphData}
              nodeLabel={(n: GraphNode) => `${n.id} (${n.addr})`}
              nodeVal={(n: GraphNode) => n.size}
              nodeColor={(n: GraphNode) => (n.is_local ? '#5B5BF0' : '#9098A6')}
              nodeOpacity={0.95}
              linkColor={() => 'rgba(140, 147, 163, 0.4)'}
              linkOpacity={0.4}
              backgroundColor="rgba(0,0,0,0)"
              showNavInfo={false}
              controlType="orbit"
              enableNodeDrag={false}
            />
          )}
        </div>

        <aside className="swarm__side">
          {/* Bandwidth savings card */}
          <div className="swarm__card">
            <div className="swarm__card-label">
              <Zap size={12} />
              BANDBREITE gespart
            </div>
            <div className="swarm__card-value">
              {Math.round(stats?.bandwidth_saved_pct ?? 0)} %
            </div>
            <div className="swarm__card-sub">
              QLMS-Binary ist messbar 2.3–2.54× kleiner als äquivalentes
              JSON (Spec §13.2). Jede Gossip-Runde spart damit
              ~60 % Netzwerk-Bandbreite gegenüber klassischen
              Föderations-Protokollen.
            </div>
          </div>

          {/* Gossip history */}
          <div className="swarm__card">
            <div className="swarm__card-label">
              <HistoryIcon size={12} />
              Gossip-Verlauf
            </div>
            {history.length === 0 ? (
              <div className="swarm__card-sub">
                Noch keine Runden. Klicke <b>Gossip auslösen</b> oder setze
                Peer-Discovery an.
              </div>
            ) : (
              <ul className="swarm__history">
                {history.map((h, i) => (
                  <li
                    key={`${h.when}-${i}`}
                    className={
                      'swarm__history-item' +
                      (h.error ? ' swarm__history-item--err' : '')
                    }
                  >
                    <span className="swarm__history-time">{formatAgo(h.when)}</span>
                    {h.error ? (
                      <span className="swarm__history-msg swarm__history-msg--err">
                        {h.error}
                      </span>
                    ) : (
                      <span className="swarm__history-msg">
                        {h.message ?? 'Runde abgeschlossen'}
                        {h.peers_ok + h.peers_failed > 0
                          ? ` · ${h.peers_ok} ok / ${h.peers_failed} ausgefallen`
                          : ''}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Consensus / pulse placeholder */}
          <div className="swarm__card swarm__card--pulse">
            <div className="swarm__card-label">
              <CircleDashed size={12} />
              KONSENS
            </div>
            <div className="swarm__card-sub">
              {local <= 1 && remote === 0
                ? 'Single-Node-Modus — Konsens trivial (1 Stimme).'
                : `Letzte Majority-Vote-Runde: ${
                    stats?.rounds_completed ?? 0
                  } erfolgreich, Tie-Break → 0 (konservativ).`}
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

function Metric({
  label,
  value,
  sub,
  accent,
}: {
  label: string
  value: string | number
  sub?: string
  accent?: boolean
}) {
  return (
    <div className="swarm__metric">
      <span className="swarm__metric-label">{label}</span>
      <span
        className={
          'swarm__metric-value' + (accent ? ' swarm__metric-value--accent' : '')
        }
      >
        {value}
      </span>
      {sub ? <span className="swarm__metric-sub">{sub}</span> : null}
    </div>
  )
}

function formatAgo(ms: number): string {
  const diff = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (diff < 5) return 'jetzt'
  if (diff < 60) return `vor ${diff}s`
  if (diff < 3600) return `vor ${Math.floor(diff / 60)}m`
  return `vor ${Math.floor(diff / 3600)}h`
}
