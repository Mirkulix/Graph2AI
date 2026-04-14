import React, { useState, useEffect } from 'react'
import { Target, ChevronDown, ChevronUp, ArrowRight, Download } from 'lucide-react'

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

interface Subtask {
  description: string
  assigned_agent: string
  status: string
  result?: string
}

interface GraphNode {
  id: string
  label: string
  node_type: string
  agent?: string
  status: string
  duration_ms?: number
}

interface GraphEdge {
  from: string
  to: string
  data_type: string
}

interface ExecutionGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface Goal {
  id: string
  description: string
  status: 'Pending' | 'InProgress' | 'Completed' | 'Failed'
  subtasks: Subtask[]
  result?: string
  created_at: string
  execution_graph?: ExecutionGraph
}

const statusBadgeClass: Record<string, string> = {
  Pending: 'badge badge-pending',
  InProgress: 'badge badge-active',
  Completed: 'badge badge-completed',
  Failed: 'badge badge-error',
}

const statusLabel: Record<string, string> = {
  Pending: 'Ausstehend',
  InProgress: 'In Arbeit',
  Completed: 'Erledigt',
  Failed: 'Fehlgeschlagen',
}

const nodeStatusColor: Record<string, string> = {
  completed: 'var(--accent-success)',
  failed: 'var(--accent-danger)',
  'in-progress': 'var(--accent-primary)',
  pending: 'var(--text-muted)',
}

function StatusBadge({ status }: { status: string }) {
  const cls = statusBadgeClass[status] ?? 'badge badge-info'
  return (
    <span className={cls}>
      {status === 'InProgress' && <span className="badge-dot" />}
      {statusLabel[status] ?? status}
    </span>
  )
}

function ExecutionGraphView({ graph }: { graph: ExecutionGraph }) {
  const orderedIds = [
    'input',
    'ceo_decompose',
    ...graph.nodes.filter(n => n.id.startsWith('subtask_')).map(n => n.id),
    'ceo_summary',
    'output',
  ]
  const orderedNodes = orderedIds
    .map(id => graph.nodes.find(n => n.id === id))
    .filter(Boolean) as GraphNode[]

  return (
    <div className="exec-graph">
      <div className="label" style={{ marginBottom: '8px' }}>Ausf&uuml;hrungsgraph</div>
      <div className="exec-graph-flow">
        {orderedNodes.map((node, i) => {
          const color = nodeStatusColor[node.status] ?? 'var(--text-muted)'
          const isDashed = node.node_type === 'input' || node.node_type === 'output'
          return (
            <React.Fragment key={node.id}>
              <div
                className="exec-graph-node"
                style={{
                  borderColor: color,
                  borderStyle: isDashed ? 'dashed' : 'solid',
                  background: `color-mix(in srgb, ${color} 10%, transparent)`,
                }}
              >
                <div style={{ fontSize: '10px', color, fontWeight: 600 }}>
                  {node.label.length > 30 ? node.label.slice(0, 30) + '\u2026' : node.label}
                </div>
                {node.duration_ms != null && (
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {node.duration_ms < 1000 ? `${node.duration_ms}ms` : `${(node.duration_ms / 1000).toFixed(1)}s`}
                  </div>
                )}
              </div>
              {i < orderedNodes.length - 1 && (
                <ArrowRight size={14} style={{ color: 'var(--border)', flexShrink: 0 }} />
              )}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}

function GoalCard({ goal }: { goal: Goal }) {
  const [expanded, setExpanded] = useState(false)
  const hasSubtasks = goal.subtasks && goal.subtasks.length > 0

  return (
    <div className="card" style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '14px', marginBottom: '8px' }}>{goal.description}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <StatusBadge status={goal.status} />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {new Date(goal.created_at).toLocaleString('de-DE', {
                hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit',
              })}
            </span>
          </div>
        </div>
        {hasSubtasks && (
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setExpanded(e => !e)}
            aria-label={expanded ? 'Einklappen' : 'Aufklappen'}
          >
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        )}
      </div>

      {expanded && hasSubtasks && (
        <div style={{ marginTop: '16px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
          {goal.subtasks.map((sub, i) => (
            <div key={i} className="subtask-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px' }}>{sub.description}</span>
                <StatusBadge status={sub.status} />
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Agent: <span style={{ color: 'var(--accent-primary)' }}>{sub.assigned_agent}</span>
              </div>
              {sub.result && (
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Ergebnis: <span style={{ color: 'var(--text-primary)' }}>{sub.result}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {goal.status === 'Completed' && goal.result && (
        <div className="goal-result" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
          <span>{goal.result}</span>
          <button
            className="btn btn-ghost btn-icon"
            style={{ flexShrink: 0, padding: '2px 6px', minHeight: 'unset' }}
            title="Ziel exportieren"
            aria-label="Ziel als Markdown exportieren"
            onClick={() => {
              const date = new Date(goal.created_at).toISOString().slice(0, 10)
              const lines: string[] = [
                `# Ziel: ${goal.description}`,
                '',
                `**Status:** ${statusLabel[goal.status] ?? goal.status}`,
                `**Erstellt:** ${new Date(goal.created_at).toLocaleString('de-DE')}`,
                '',
              ]
              if (goal.subtasks?.length) {
                lines.push('## Teilaufgaben', '')
                for (const sub of goal.subtasks) {
                  lines.push(`### ${sub.description}`)
                  lines.push(`- Agent: ${sub.assigned_agent}`)
                  lines.push(`- Status: ${statusLabel[sub.status] ?? sub.status}`)
                  if (sub.result) lines.push(`- Ergebnis: ${sub.result}`)
                  lines.push('')
                }
              }
              lines.push('## Ergebnis', '', goal.result ?? '')
              downloadFile(`qo-goal-${date}-${goal.id}.md`, lines.join('\n'), 'text/markdown')
            }}
          >
            <Download size={14} />
          </button>
        </div>
      )}

      {expanded && goal.execution_graph && (
        <ExecutionGraphView graph={goal.execution_graph} />
      )}
    </div>
  )
}

export default function GoalsView() {
  const [goals, setGoals] = useState<Goal[]>([])
  const [input, setInput] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchGoals = () => {
    fetch('/api/goals')
      .then(r => r.json())
      .then(data => setGoals(Array.isArray(data) ? data : []))
      .catch(() => {})
  }

  useEffect(() => {
    fetchGoals()
    const interval = setInterval(fetchGoals, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch('/api/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: trimmed }),
      })
      if (!res.ok) throw new Error('Fehler beim Erstellen')
      setInput('')
      fetchGoals()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="view">
      <form onSubmit={handleSubmit} style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <Target size={20} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          <input
            className="input"
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Neues Ziel..."
            disabled={submitting}
          />
          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting || !input.trim()}
          >
            {submitting ? '...' : 'Starten'}
          </button>
        </div>
        {error && (
          <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--accent-danger)' }}>{error}</div>
        )}
      </form>

      {goals.length === 0 ? (
        <div className="empty-state">
          <Target size={40} className="empty-icon" />
          <div className="empty-title">Noch keine Ziele</div>
          <div className="empty-hint">Erstelle ein neues Ziel oben, um loszulegen.</div>
        </div>
      ) : (
        goals.map(goal => <GoalCard key={goal.id} goal={goal} />)
      )}

      <style>{`
        .subtask-item {
          padding: 8px 12px;
          background: var(--bg-primary);
          border-radius: var(--radius-sm);
          margin-bottom: 6px;
          border-left: 3px solid var(--border);
        }
        .goal-result {
          margin-top: 12px;
          padding: 10px 14px;
          background: color-mix(in srgb, var(--accent-success) 8%, transparent);
          border: 1px solid color-mix(in srgb, var(--accent-success) 20%, transparent);
          border-radius: var(--radius-sm);
          font-size: 12px;
          color: var(--accent-success);
        }
        .exec-graph {
          margin-top: 16px;
          border-top: 1px solid var(--border);
          padding-top: 12px;
        }
        .exec-graph-flow {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: 6px;
          overflow-x: auto;
          padding-bottom: 4px;
        }
        .exec-graph-node {
          border: 1px solid;
          border-radius: var(--radius-sm);
          padding: 4px 8px;
          min-width: 80px;
          max-width: 140px;
          text-align: center;
        }
      `}</style>
    </div>
  )
}
