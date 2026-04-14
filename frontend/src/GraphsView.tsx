import { useState, useEffect } from 'react'
import { GitBranch, Timer, ArrowRight, Circle, RefreshCw, Download } from 'lucide-react'

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

interface GraphNode {
  id: string
  op: string
  node_type: string
  label: string
  agent?: string | null
  status?: string
  duration_ms?: number | null
  input_type?: string | null
  output_type?: string | null
}

interface GraphEdge {
  from: string
  to: string
  data_type?: string
}

interface GraphMetadata {
  total_duration_ms?: number
  llm_tier?: string
  tokens_estimated?: number
}

interface StoredGraph {
  id: number
  timestamp: number
  graph_type: string
  title: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  metadata?: GraphMetadata
}

interface GraphStats {
  total_graphs: number
  by_type: Record<string, number>
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function relativeTime(timestamp: number): string {
  const now = Math.floor(Date.now() / 1000)
  const diff = now - timestamp
  if (diff < 60) return 'gerade eben'
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`
  return `vor ${Math.floor(diff / 86400)} Tagen`
}

function nodeTypeColor(nodeType: string): string {
  switch (nodeType) {
    case 'Llm':           return 'var(--accent-primary)'
    case 'Deterministic': return 'var(--accent-info)'
    case 'Values':        return 'var(--accent-achtsamkeit)'
    case 'Memory':        return 'var(--accent-anerkennung)'
    default:              return 'var(--text-muted)'
  }
}

function nodeTypeBorder(nodeType: string): string {
  const isInputOutput = nodeType === 'Input' || nodeType === 'Output'
  if (isInputOutput) return '1px dashed var(--text-muted)'
  return `1px solid ${nodeTypeColor(nodeType)}`
}

function nodeTypeGlow(nodeType: string): string {
  if (nodeType === 'Llm') return `0 0 8px ${nodeTypeColor(nodeType)}40`
  return 'none'
}

function statusColor(status?: string): string {
  switch (status) {
    case 'Completed': return 'var(--accent-success)'
    case 'Failed':    return 'var(--accent-danger)'
    case 'Running':   return 'var(--accent-primary)'
    default:          return 'var(--text-muted)'
  }
}

function typeBadgeClass(graphType: string): string {
  switch (graphType) {
    case 'Chat':          return 'badge-info'
    case 'GoalExecution': return 'badge-pending'
    case 'Evolution':     return 'badge-idle'
    default:              return 'badge-info'
  }
}

function graphTypeLabel(graphType: string): string {
  switch (graphType) {
    case 'Chat':          return 'Chat'
    case 'GoalExecution': return 'Goal'
    case 'Evolution':     return 'Evolution'
    default:              return graphType
  }
}

// ─── Graph layout helpers ────────────────────────────────────────────────────

/**
 * Build ordered columns for horizontal flow layout.
 * Nodes in the same "parallel group" (same source → fan-out) share a column.
 */
function buildColumns(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[][] {
  if (nodes.length === 0) return []

  // Map: nodeId → set of predecessors
  const predecessors: Map<string, Set<string>> = new Map()
  const successors: Map<string, Set<string>> = new Map()
  for (const n of nodes) {
    predecessors.set(n.id, new Set())
    successors.set(n.id, new Set())
  }
  for (const e of edges) {
    predecessors.get(e.to)?.add(e.from)
    successors.get(e.from)?.add(e.to)
  }

  // Assign depth via BFS from roots (nodes with no predecessors)
  const depth: Map<string, number> = new Map()
  const queue: string[] = []
  for (const n of nodes) {
    if ((predecessors.get(n.id)?.size ?? 0) === 0) {
      depth.set(n.id, 0)
      queue.push(n.id)
    }
  }

  let qi = 0
  while (qi < queue.length) {
    const cur = queue[qi++]
    const curDepth = depth.get(cur) ?? 0
    for (const succ of successors.get(cur) ?? []) {
      const existing = depth.get(succ)
      if (existing === undefined || existing < curDepth + 1) {
        depth.set(succ, curDepth + 1)
        queue.push(succ)
      }
    }
  }

  // Any node not reached gets depth = last depth + 1
  const maxDepth = Math.max(0, ...Array.from(depth.values()))
  for (const n of nodes) {
    if (!depth.has(n.id)) depth.set(n.id, maxDepth)
  }

  // Group into columns
  const colCount = Math.max(0, ...Array.from(depth.values())) + 1
  const columns: GraphNode[][] = Array.from({ length: colCount }, () => [])
  for (const n of nodes) {
    columns[depth.get(n.id) ?? 0].push(n)
  }

  return columns
}

// ─── Node Box ────────────────────────────────────────────────────────────────

function NodeBox({ node }: { node: GraphNode }) {
  const isRunning = node.status === 'Running'

  return (
    <div
      className="graph-node"
      style={{
        border: nodeTypeBorder(node.node_type),
        boxShadow: nodeTypeGlow(node.node_type),
        color: node.node_type === 'Input' || node.node_type === 'Output'
          ? 'var(--text-muted)'
          : 'var(--text-primary)',
      }}
    >
      {/* Status dot */}
      <div style={{ position: 'absolute', top: 6, right: 6 }}>
        <Circle
          size={8}
          fill={statusColor(node.status)}
          color={statusColor(node.status)}
          className={isRunning ? 'status-pulse' : undefined}
        />
      </div>

      {/* Label */}
      <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4, paddingRight: 14 }}>
        {node.label}
      </div>

      {/* Agent name */}
      {node.agent && (
        <div style={{ fontSize: 10, color: nodeTypeColor(node.node_type), marginBottom: 2 }}>
          {node.agent}
        </div>
      )}

      {/* Duration */}
      {node.duration_ms != null && (
        <div className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 2 }}>
          <Timer size={9} />
          {node.duration_ms}ms
        </div>
      )}

      {/* Types row */}
      {(node.input_type || node.output_type) && (
        <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {node.input_type && <span>in: {node.input_type}</span>}
          {node.output_type && <span>out: {node.output_type}</span>}
        </div>
      )}
    </div>
  )
}

// ─── Graph Diagram ───────────────────────────────────────────────────────────

function GraphDiagram({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const columns = buildColumns(nodes, edges)

  return (
    <div className="graph-diagram-scroll">
      <div className="graph-flow">
        {columns.map((col, ci) => (
          <div key={ci} style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 0 }}>
            {/* Column of nodes (stacked vertically for parallel branches) */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {col.map(node => (
                <NodeBox key={node.id} node={node} />
              ))}
            </div>

            {/* Arrow connector to next column (skip last) */}
            {ci < columns.length - 1 && (
              <div style={{ display: 'flex', alignItems: 'center', padding: '0 4px', color: 'var(--text-muted)' }}>
                <ArrowRight size={14} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Graph Card ──────────────────────────────────────────────────────────────

function GraphCard({ graph }: { graph: StoredGraph }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="graph-card">
      {/* Collapsed header (always visible) */}
      <button
        className="graph-card-header"
        onClick={() => setExpanded(x => !x)}
        aria-expanded={expanded}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1, minWidth: 0 }}>
          <span className={`badge ${typeBadgeClass(graph.graph_type)}`}>
            {graphTypeLabel(graph.graph_type)}
          </span>
          <span style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {graph.title}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          <span className="label" style={{ fontSize: 10 }}>
            {graph.nodes.length} Nodes / {graph.edges.length} Edges
          </span>
          {graph.metadata?.total_duration_ms != null && (
            <span className="mono label" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
              <Timer size={10} />
              {graph.metadata.total_duration_ms}ms
            </span>
          )}
          <span className="label" style={{ fontSize: 10 }}>
            {relativeTime(graph.timestamp)}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="graph-card-body">
          <GraphDiagram nodes={graph.nodes} edges={graph.edges} />

          {/* Metadata bar */}
          <div className="graph-meta-bar">
            {graph.metadata?.total_duration_ms != null && (
              <span className="graph-meta-item">
                <Timer size={11} />
                {graph.metadata.total_duration_ms}ms
              </span>
            )}
            {graph.metadata?.llm_tier && (
              <span className="graph-meta-item">
                tier: {graph.metadata.llm_tier}
              </span>
            )}
            {graph.metadata?.tokens_estimated != null && (
              <span className="graph-meta-item">
                ~{graph.metadata.tokens_estimated} tokens
              </span>
            )}
            <span className={`badge ${typeBadgeClass(graph.graph_type)}`} style={{ marginLeft: 'auto' }}>
              {graphTypeLabel(graph.graph_type)}
            </span>
            <button
              className="btn btn-ghost btn-icon"
              style={{ padding: '2px 6px', minHeight: 'unset', marginLeft: 8 }}
              title="Graph exportieren"
              aria-label="Graph als JSON exportieren"
              onClick={e => {
                e.stopPropagation()
                const date = new Date(graph.timestamp * 1000).toISOString().slice(0, 10)
                const payload = JSON.stringify({
                  id: graph.id,
                  timestamp: graph.timestamp,
                  graph_type: graph.graph_type,
                  title: graph.title,
                  nodes: graph.nodes,
                  edges: graph.edges,
                  metadata: graph.metadata,
                }, null, 2)
                downloadFile(`qo-graph-${date}-${graph.id}.json`, payload, 'application/json')
              }}
            >
              <Download size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main View ───────────────────────────────────────────────────────────────

type TypeFilter = 'all' | 'Chat' | 'GoalExecution' | 'Evolution'

const typeFilters: { id: TypeFilter; label: string }[] = [
  { id: 'all',          label: 'Alle' },
  { id: 'Chat',         label: 'Chat' },
  { id: 'GoalExecution',label: 'Goal' },
  { id: 'Evolution',    label: 'Evolution' },
]

export default function GraphsView() {
  const [graphs, setGraphs] = useState<StoredGraph[]>([])
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('all')

  const fetchData = async () => {
    try {
      const [graphsRes, statsRes] = await Promise.all([
        fetch('/api/graphs?limit=50').catch(() => null),
        fetch('/api/graphs/stats').catch(() => null),
      ])

      if (graphsRes?.ok) {
        const data = await graphsRes.json()
        setGraphs(Array.isArray(data) ? data : [])
      }
      if (statsRes?.ok) {
        const data = await statsRes.json()
        setStats(data)
      }
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [])

  const filtered = typeFilter === 'all'
    ? graphs
    : graphs.filter(g => g.graph_type === typeFilter)

  // newest first
  const sorted = [...filtered].sort((a, b) => b.timestamp - a.timestamp)

  return (
    <div className="view">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="graphs-icon-wrapper">
            <GitBranch size={24} />
          </div>
          <div>
            <h2 className="heading" style={{ fontSize: 18, margin: 0 }}>QLANG Graphs</h2>
            {stats && (
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                {stats.total_graphs} Graphs gespeichert
                {stats.by_type && Object.entries(stats.by_type).map(([k, v]) => (
                  <span key={k} style={{ marginLeft: 8 }}>
                    {graphTypeLabel(k)}: {v}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <button
          className="btn btn-ghost btn-icon"
          onClick={fetchData}
          title="Aktualisieren"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {typeFilters.map(f => (
          <button
            key={f.id}
            className={`btn btn-ghost graphs-filter-btn${typeFilter === f.id ? ' active' : ''}`}
            onClick={() => setTypeFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {loading && (
        <div className="empty-state">
          <GitBranch size={40} className="empty-icon" />
          <div className="empty-title">Lade Graphs...</div>
        </div>
      )}

      {!loading && error && (
        <div style={{ fontSize: 12, color: 'var(--accent-danger)', marginBottom: 16 }}>
          Fehler: {error}
        </div>
      )}

      {!loading && !error && sorted.length === 0 && (
        <div className="empty-state">
          <GitBranch size={40} className="empty-icon" />
          <div className="empty-title">Keine Graphs vorhanden</div>
          <div className="empty-hint">
            Graphs werden gespeichert, wenn Chat- oder Goal-Ausf&uuml;hrungen stattfinden.
          </div>
        </div>
      )}

      {!loading && sorted.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {sorted.map(g => (
            <GraphCard key={g.id} graph={g} />
          ))}
        </div>
      )}

      <style>{`
        .graphs-icon-wrapper {
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
          color: var(--accent-primary);
        }

        .graphs-filter-btn {
          min-height: 32px;
          padding: 4px 14px;
          font-size: 12px;
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-secondary);
        }

        .graphs-filter-btn.active {
          background: color-mix(in srgb, var(--accent-primary) 12%, transparent);
          border-color: var(--accent-primary);
          color: var(--accent-primary);
        }

        .graph-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          overflow: hidden;
          transition: border-color var(--transition);
        }

        .graph-card:hover {
          border-color: var(--border-active);
        }

        .graph-card-header {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 14px 18px;
          background: transparent;
          border: none;
          cursor: pointer;
          text-align: left;
          outline: none;
          font-family: 'JetBrains Mono', monospace;
          color: var(--text-primary);
          transition: background var(--transition);
        }

        .graph-card-header:hover {
          background: var(--bg-elevated);
        }

        .graph-card-header:focus-visible {
          outline: 2px solid var(--accent-primary);
          outline-offset: -2px;
        }

        .graph-card-body {
          border-top: 1px solid var(--border);
          padding: 16px 18px 0;
          animation: fadeIn 200ms ease-out;
        }

        .graph-diagram-scroll {
          overflow-x: auto;
          padding-bottom: 16px;
        }

        .graph-flow {
          display: flex;
          flex-direction: row;
          align-items: center;
          gap: 0;
          min-width: max-content;
        }

        .graph-node {
          position: relative;
          background: var(--bg-elevated);
          border-radius: var(--radius-md);
          padding: 10px 12px;
          min-width: 100px;
          max-width: 160px;
          font-size: 11px;
          flex-shrink: 0;
        }

        .status-pulse {
          animation: pulse 1.5s ease-in-out infinite;
        }

        .graph-meta-bar {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 0 14px;
          border-top: 1px solid var(--border);
          margin-top: 12px;
          font-size: 11px;
          color: var(--text-secondary);
          flex-wrap: wrap;
        }

        .graph-meta-item {
          display: flex;
          align-items: center;
          gap: 4px;
          font-variant-numeric: tabular-nums;
        }
      `}</style>
    </div>
  )
}
