// GraphInspectorView.tsx — PRD Task 6.2 / UI-Konzept "Graph Inspector".
//
// Split-view debugger for every QLANG graph the QO server has stored:
//   * Left pane: paginated list of StoredGraph entries from /api/graphs.
//   * Right pane: selected graph's header (id, title, timestamp, type),
//     structured metadata, node tree and edge list.
//
// The GraphStore we hit already persists binary QLBG payloads internally,
// but the REST view exposes only the high-level StoredGraph JSON. This
// component mirrors that shape; a future iteration can add a "Replay"
// button that ships the raw frame back through /qlms/v1.1/deliver.

import { useEffect, useMemo, useState } from 'react'
import {
  Archive,
  Box,
  Check,
  CircleDashed,
  Clock,
  Hash,
  Play,
  Shield,
  X,
} from 'lucide-react'

type GraphType = 'Chat' | 'GoalExecution' | 'AgentTask' | 'Evolution' | 'ValueCheck'
type NodeType = 'Input' | 'Output' | 'Llm' | 'Deterministic' | 'Memory' | 'Values'
type NodeStatus = 'Pending' | 'Running' | 'Completed' | 'Failed'

interface QlangNode {
  id: string
  op: string
  node_type: NodeType
  label: string
  agent: string | null
  status: NodeStatus
  duration_ms: number | null
  input_type: string | null
  output_type: string | null
}

interface QlangEdge {
  from: string
  to: string
  data_type: string
}

interface GraphMetadata {
  total_duration_ms: number | null
  llm_tier: string | null
  tokens_estimated: number | null
  cost_usd: number | null
}

interface StoredGraph {
  id: number
  timestamp: number
  graph_type: GraphType
  title: string
  nodes: QlangNode[]
  edges: QlangEdge[]
  metadata: GraphMetadata
}

const PAGE_SIZE = 50

export default function GraphInspectorView() {
  const [graphs, setGraphs] = useState<StoredGraph[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<StoredGraph | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  async function loadList() {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(`/api/graphs?limit=${PAGE_SIZE}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const body = (await r.json()) as StoredGraph[]
      setGraphs(body)
      // Auto-select the newest graph on first load so the detail pane is
      // never empty when graphs exist.
      if (body.length > 0 && selectedId === null) {
        setSelectedId(body[0].id)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadList()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (selectedId === null) {
      setDetail(null)
      return
    }
    let cancelled = false
    setDetailLoading(true)
    ;(async () => {
      try {
        const r = await fetch(`/api/graphs/${selectedId}`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const body = (await r.json()) as StoredGraph
        if (!cancelled) {
          setDetail(body)
          setDetailLoading(false)
        }
      } catch (e) {
        if (!cancelled) {
          setDetail(null)
          setDetailLoading(false)
          setError(e instanceof Error ? e.message : String(e))
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const formattedList = useMemo(
    () =>
      graphs.map((g) => ({
        ...g,
        ageLabel: formatTimestamp(g.timestamp),
      })),
    [graphs],
  )

  return (
    <div className="inspector">
      <aside className="inspector__list">
        <div className="inspector__list-header">
          <Archive size={14} />
          <span>Graphs ({graphs.length})</span>
          <button type="button" onClick={() => void loadList()}>Neu laden</button>
        </div>

        {loading ? (
          <div className="inspector__empty">Lade …</div>
        ) : error ? (
          <div className="inspector__empty inspector__empty--error">
            {error}
          </div>
        ) : graphs.length === 0 ? (
          <div className="inspector__empty">
            Noch keine gespeicherten Graphs. Sobald Agenten Graphs ausführen,
            erscheinen sie hier.
          </div>
        ) : (
          <ul>
            {formattedList.map((g) => (
              <li
                key={g.id}
                className={selectedId === g.id ? 'is-selected' : undefined}
              >
                <button type="button" onClick={() => setSelectedId(g.id)}>
                  <span className="inspector__list-title">{g.title || `Graph #${g.id}`}</span>
                  <span className="inspector__list-meta">
                    <code>{g.graph_type}</code>
                    <span>·</span>
                    <span>{g.nodes.length}n / {g.edges.length}e</span>
                    <span>·</span>
                    <span>{g.ageLabel}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <section className="inspector__detail">
        {selectedId === null ? (
          <EmptyDetail />
        ) : detailLoading ? (
          <div className="inspector__empty">Lade Details …</div>
        ) : detail ? (
          <GraphDetailPane graph={detail} />
        ) : (
          <div className="inspector__empty">Keine Details verfügbar.</div>
        )}
      </section>
    </div>
  )
}

function EmptyDetail() {
  return (
    <div className="inspector__empty">
      Wähle links einen Graph, um Header, Security, Knoten und Kanten zu inspizieren.
    </div>
  )
}

function GraphDetailPane({ graph }: { graph: StoredGraph }) {
  const hasMetadata =
    graph.metadata.total_duration_ms !== null ||
    graph.metadata.llm_tier !== null ||
    graph.metadata.tokens_estimated !== null ||
    graph.metadata.cost_usd !== null

  return (
    <>
      <header className="inspector__detail-header">
        <div>
          <h2>{graph.title || `Graph #${graph.id}`}</h2>
          <div className="inspector__detail-sub">
            <span><Hash size={12} /> {graph.id}</span>
            <span><Clock size={12} /> {formatTimestamp(graph.timestamp)}</span>
            <code>{graph.graph_type}</code>
          </div>
        </div>
        <button type="button" className="inspector__replay" disabled title="Replay via /qlms/v1.1/deliver kommt in einem folgenden PR">
          <Play size={12} />
          Replay
        </button>
      </header>

      <section className="inspector__panel">
        <h3>Header</h3>
        <dl>
          <dt>Nodes</dt>
          <dd>{graph.nodes.length}</dd>
          <dt>Edges</dt>
          <dd>{graph.edges.length}</dd>
          <dt>Timestamp (UNIX)</dt>
          <dd><code>{graph.timestamp}</code></dd>
          <dt>Graph-Typ</dt>
          <dd><code>{graph.graph_type}</code></dd>
        </dl>
      </section>

      {hasMetadata ? (
        <section className="inspector__panel">
          <h3>
            <Shield size={12} /> Ausführungs-Metadaten
          </h3>
          <dl>
            {graph.metadata.total_duration_ms !== null ? (
              <>
                <dt>Gesamtdauer</dt>
                <dd>{graph.metadata.total_duration_ms} ms</dd>
              </>
            ) : null}
            {graph.metadata.llm_tier !== null ? (
              <>
                <dt>LLM-Tier</dt>
                <dd><code>{graph.metadata.llm_tier}</code></dd>
              </>
            ) : null}
            {graph.metadata.tokens_estimated !== null ? (
              <>
                <dt>Tokens (geschätzt)</dt>
                <dd>{graph.metadata.tokens_estimated.toLocaleString('de-DE')}</dd>
              </>
            ) : null}
            {graph.metadata.cost_usd !== null ? (
              <>
                <dt>Kosten</dt>
                <dd>${graph.metadata.cost_usd.toFixed(4)}</dd>
              </>
            ) : null}
          </dl>
        </section>
      ) : null}

      <section className="inspector__panel">
        <h3>
          <Box size={12} /> Knoten
        </h3>
        <ul className="inspector__nodes">
          {graph.nodes.map((n) => (
            <li key={n.id} className={`inspector__node inspector__node--${statusClass(n.status)}`}>
              <div className="inspector__node-head">
                <StatusBadge status={n.status} />
                <code>{n.id}</code>
                <strong>{n.op}</strong>
                <span className="inspector__node-label">{n.label}</span>
              </div>
              <div className="inspector__node-meta">
                <span><code>{n.node_type}</code></span>
                {n.agent ? <span>Agent: <code>{n.agent}</code></span> : null}
                {n.duration_ms !== null ? <span>{n.duration_ms} ms</span> : null}
                {n.input_type ? <span>in: <code>{n.input_type}</code></span> : null}
                {n.output_type ? <span>out: <code>{n.output_type}</code></span> : null}
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="inspector__panel">
        <h3>Kanten</h3>
        {graph.edges.length === 0 ? (
          <p className="inspector__muted">Keine Kanten.</p>
        ) : (
          <ul className="inspector__edges">
            {graph.edges.map((e, i) => (
              <li key={`${e.from}->${e.to}-${i}`}>
                <code>{e.from}</code>
                <span> → </span>
                <code>{e.to}</code>
                <span className="inspector__muted"> : {e.data_type}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  )
}

function StatusBadge({ status }: { status: NodeStatus }) {
  switch (status) {
    case 'Completed':
      return (
        <span className="inspector__status inspector__status--ok" title="Abgeschlossen">
          <Check size={12} />
        </span>
      )
    case 'Failed':
      return (
        <span className="inspector__status inspector__status--err" title="Fehlgeschlagen">
          <X size={12} />
        </span>
      )
    case 'Running':
      return (
        <span className="inspector__status inspector__status--run" title="Läuft">
          <CircleDashed size={12} />
        </span>
      )
    default:
      return (
        <span className="inspector__status inspector__status--pending" title="Ausstehend">
          <CircleDashed size={12} />
        </span>
      )
  }
}

function statusClass(s: NodeStatus): string {
  if (s === 'Completed') return 'ok'
  if (s === 'Failed') return 'err'
  if (s === 'Running') return 'run'
  return 'pending'
}

function formatTimestamp(unix: number): string {
  if (!unix || !Number.isFinite(unix)) return '—'
  const d = new Date(unix * 1000)
  return d.toLocaleString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
