import { useEffect, useMemo, useState, useRef } from 'react'

// ──────────────────────────────────────────────────────────────────────
// Types mirroring /api/neo/agents and /api/neo/agents/{id}
// ──────────────────────────────────────────────────────────────────────

interface AgentSummary {
  id: string
  session_id: string
  type: string
  description: string
  status: 'running' | 'done' | 'idle'
  interactions: number
  tool_calls: number
  files_written: string[]
  files_read: string[]
  last_tool: string | null
  last_activity_at: string | null
  started_at: string | null
  duration_secs: number | null
}

interface ToolEvent {
  ts: string | null
  role: 'assistant' | 'user' | string
  tool: string | null
  summary: string
  file_path: string | null
  is_error: boolean | null
}

interface AgentDetail {
  summary: AgentSummary
  events: ToolEvent[]
  initial_prompt: string | null
}

// ──────────────────────────────────────────────────────────────────────
// Visual helpers
// ──────────────────────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  coder: '#00ffb2',
  reviewer: '#ffb020',
  tester: '#20d3ff',
  planner: '#b28fff',
  researcher: '#ff6ec7',
  'cicd-engineer': '#f0df55',
  'security-auditor': '#ff4d6d',
  'performance-engineer': '#ff9f43',
  'backend-dev': '#00b8a9',
  'frontend-dev': '#7bd3ea',
  general: '#888fa8',
}

const colorFor = (t: string) => TYPE_COLORS[t] || '#888fa8'

const relTime = (iso: string | null): string => {
  if (!iso) return 'unknown'
  const d = Date.parse(iso)
  if (isNaN(d)) return iso
  const diff = Math.max(0, Date.now() - d)
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h}h ago`
  const days = Math.floor(h / 24)
  return `${days}d ago`
}

const fmtDuration = (a: string | null, b: string | null): string => {
  if (!a || !b) return '—'
  const da = Date.parse(a), db = Date.parse(b)
  if (isNaN(da) || isNaN(db)) return '—'
  const s = Math.max(0, Math.round((db - da) / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60), rs = s % 60
  return `${m}m ${rs}s`
}

const basename = (p: string) => p.split('/').pop() || p

// ──────────────────────────────────────────────────────────────────────
// Card
// ──────────────────────────────────────────────────────────────────────

function AgentCard({ a, onOpen }: { a: AgentSummary; onOpen: () => void }) {
  const col = colorFor(a.type)
  const running = a.status === 'running'
  return (
    <button
      onClick={onOpen}
      className="neo-agent-card"
      style={{
        background: 'linear-gradient(160deg, #0f1530 0%, #0a0e27 100%)',
        border: `1px solid ${col}33`,
        borderRadius: 14,
        padding: '16px 18px',
        textAlign: 'left',
        cursor: 'pointer',
        color: '#e2e7ff',
        transition: 'all 0.2s ease',
        boxShadow: running
          ? `0 0 0 1px ${col}55, 0 0 24px ${col}33, inset 0 1px 0 ${col}22`
          : `0 2px 10px #00000055`,
        position: 'relative',
        overflow: 'hidden',
        minHeight: 170,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
      onMouseEnter={(e) => {
        const el = e.currentTarget
        el.style.transform = 'translateY(-2px)'
        el.style.boxShadow = `0 0 0 1px ${col}88, 0 10px 30px ${col}44`
        el.style.borderColor = `${col}aa`
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget
        el.style.transform = 'translateY(0)'
        el.style.boxShadow = running
          ? `0 0 0 1px ${col}55, 0 0 24px ${col}33, inset 0 1px 0 ${col}22`
          : `0 2px 10px #00000055`
        el.style.borderColor = `${col}33`
      }}
    >
      {/* Top: type tag + status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: col,
            padding: '3px 8px',
            border: `1px solid ${col}66`,
            borderRadius: 4,
            background: `${col}15`,
          }}
        >
          {a.type}
        </span>
        <StatusPill status={a.status} color={col} />
      </div>

      {/* Description */}
      <div
        style={{
          fontSize: 13,
          lineHeight: 1.4,
          color: '#c8d0f0',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          minHeight: 36,
        }}
      >
        {a.description || <span style={{ opacity: 0.5 }}>No description</span>}
      </div>

      {/* Last tool line */}
      <div
        style={{
          fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
          fontSize: 11,
          color: '#7f88b3',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          minHeight: 16,
        }}
      >
        <span style={{ color: col }}>›</span>
        {running ? (
          <TypingLine text={a.last_tool || 'thinking'} />
        ) : (
          <span style={{ color: '#9aa3cc' }}>{a.last_tool || '—'}</span>
        )}
      </div>

      {/* Stats row */}
      <div
        style={{
          display: 'flex',
          gap: 14,
          fontSize: 11,
          color: '#7f88b3',
          marginTop: 'auto',
          flexWrap: 'wrap',
        }}
      >
        <Stat label="tools" value={a.tool_calls} />
        <Stat label="files" value={a.files_written.length} />
        <Stat label="steps" value={a.interactions} />
        <span style={{ marginLeft: 'auto' }}>{relTime(a.last_activity_at)}</span>
      </div>
    </button>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <span>
      <span style={{ color: '#e2e7ff', fontWeight: 600 }}>{value}</span>
      <span style={{ opacity: 0.6, marginLeft: 3 }}>{label}</span>
    </span>
  )
}

function StatusPill({ status, color }: { status: string; color: string }) {
  const running = status === 'running'
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 10,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: running ? color : '#7f88b3',
        fontWeight: 700,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: running ? color : '#3a4270',
          boxShadow: running ? `0 0 8px ${color}, 0 0 2px ${color}` : 'none',
          animation: running ? 'neoPulse 1.2s ease-in-out infinite' : 'none',
        }}
      />
      {status}
    </span>
  )
}

function TypingLine({ text }: { text: string }) {
  const [dots, setDots] = useState('')
  useEffect(() => {
    const i = setInterval(() => setDots((d) => (d.length >= 3 ? '' : d + '.')), 350)
    return () => clearInterval(i)
  }, [])
  return (
    <span style={{ color: '#b0b8e0' }}>
      {text}
      <span style={{ color: '#00ffb2' }}>{dots}</span>
    </span>
  )
}

// ──────────────────────────────────────────────────────────────────────
// Detail panel (timeline)
// ──────────────────────────────────────────────────────────────────────

function DetailPanel({ agentId, onClose }: { agentId: string; onClose: () => void }) {
  const [data, setData] = useState<AgentDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancel = false
    setData(null)
    setErr(null)
    fetch(`/api/neo/agents/${agentId}`)
      .then((r) => r.json())
      .then((d: AgentDetail | null) => {
        if (cancel) return
        if (!d) setErr('not found')
        else setData(d)
      })
      .catch((e) => !cancel && setErr(String(e)))
    return () => {
      cancel = true
    }
  }, [agentId])

  const col = data ? colorFor(data.summary.type) : '#888fa8'

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(5, 8, 20, 0.82)',
        zIndex: 100,
        backdropFilter: 'blur(6px)',
        display: 'flex',
        justifyContent: 'flex-end',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(720px, 95vw)',
          height: '100%',
          background: '#0a0e27',
          borderLeft: `1px solid ${col}55`,
          boxShadow: `-20px 0 60px ${col}22`,
          overflowY: 'auto',
          padding: '22px 26px',
          color: '#e2e7ff',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0, fontSize: 16, color: col, letterSpacing: '0.05em' }}>
            {data ? data.summary.type.toUpperCase() : 'Loading…'}
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: `1px solid #2a3058`,
              color: '#9aa3cc',
              padding: '4px 10px',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            close (esc)
          </button>
        </div>

        {err && <div style={{ marginTop: 20, color: '#ff6b6b' }}>Error: {err}</div>}

        {data && (
          <>
            <p style={{ color: '#c8d0f0', marginTop: 8, fontSize: 13, lineHeight: 1.5 }}>
              {data.summary.description || <em style={{ opacity: 0.6 }}>(no description)</em>}
            </p>
            <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', marginTop: 10, fontSize: 11, color: '#9aa3cc' }}>
              <span>id: <code style={{ color: col }}>{data.summary.id.slice(0, 12)}</code></span>
              <span>status: <span style={{ color: col }}>{data.summary.status}</span></span>
              <span>steps: {data.summary.interactions}</span>
              <span>tools: {data.summary.tool_calls}</span>
              <span>duration: {fmtDuration(data.summary.started_at, data.summary.last_activity_at)}</span>
            </div>

            {data.initial_prompt && (
              <details style={{ marginTop: 16 }}>
                <summary style={{ cursor: 'pointer', color: '#9aa3cc', fontSize: 12 }}>
                  initial prompt ({data.initial_prompt.length} chars)
                </summary>
                <pre
                  style={{
                    marginTop: 8,
                    background: '#070a1c',
                    border: '1px solid #1a1f3d',
                    padding: 10,
                    borderRadius: 6,
                    fontSize: 11,
                    whiteSpace: 'pre-wrap',
                    color: '#b0b8e0',
                    maxHeight: 240,
                    overflow: 'auto',
                  }}
                >
                  {data.initial_prompt}
                </pre>
              </details>
            )}

            {data.summary.files_written.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <div style={{ color: col, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Files Written ({data.summary.files_written.length})
                </div>
                <ul style={{ margin: '6px 0 0', paddingLeft: 18, fontSize: 12 }}>
                  {data.summary.files_written.map((f) => (
                    <li key={f} style={{ color: '#c8d0f0', fontFamily: 'ui-monospace, monospace' }}>
                      <span style={{ color: '#7f88b3' }}>{f.replace(basename(f), '')}</span>
                      <span style={{ color: col }}>{basename(f)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div style={{ marginTop: 22 }}>
              <div style={{ color: col, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Timeline ({data.events.length})
              </div>
              <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {data.events.map((ev, i) => (
                  <EventRow key={i} ev={ev} color={col} />
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function EventRow({ ev, color }: { ev: ToolEvent; color: string }) {
  const isErr = ev.is_error === true
  const isTool = ev.tool && ev.tool !== 'tool_result'
  const bg = isErr ? '#2a0f1f' : ev.role === 'assistant' ? '#0d1330' : '#0a0f22'
  const border = isErr ? '#ff4d6d66' : ev.role === 'assistant' ? `${color}33` : '#1a1f3d'
  return (
    <div
      style={{
        padding: '7px 10px',
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 5,
        fontSize: 12,
        display: 'flex',
        gap: 10,
        alignItems: 'flex-start',
      }}
    >
      <div style={{ minWidth: 70, fontFamily: 'ui-monospace, monospace', fontSize: 10, color: '#7f88b3' }}>
        {ev.ts ? new Date(ev.ts).toLocaleTimeString() : ''}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, marginBottom: 2 }}>
          <span
            style={{
              color: isTool ? color : '#7f88b3',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            {ev.tool || ev.role}
          </span>
          {isErr && (
            <span style={{ marginLeft: 8, color: '#ff6b6b', fontSize: 10 }}>ERROR</span>
          )}
          {ev.file_path && (
            <span
              style={{
                marginLeft: 8,
                color: '#9aa3cc',
                fontFamily: 'ui-monospace, monospace',
                fontSize: 10,
              }}
            >
              {ev.file_path}
            </span>
          )}
        </div>
        {ev.summary && (
          <div
            style={{
              color: '#c8d0f0',
              fontFamily: ev.tool ? 'ui-monospace, monospace' : 'inherit',
              fontSize: 11,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              opacity: 0.9,
            }}
          >
            {ev.summary}
          </div>
        )}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────
// Main view
// ──────────────────────────────────────────────────────────────────────

export default function NeoAgents() {
  const [agents, setAgents] = useState<AgentSummary[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'running' | 'done'>('all')
  const [query, setQuery] = useState('')
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false
    const load = () => {
      fetch('/api/neo/agents')
        .then((r) => r.json())
        .then((d: AgentSummary[]) => {
          if (stopped.current) return
          setAgents(d)
          setErr(null)
        })
        .catch((e) => !stopped.current && setErr(String(e)))
    }
    load()
    const iv = setInterval(load, 3000)
    return () => {
      stopped.current = true
      clearInterval(iv)
    }
  }, [])

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenId(null)
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])

  const filtered = useMemo(() => {
    if (!agents) return []
    return agents.filter((a) => {
      if (filter !== 'all' && a.status !== filter) return false
      if (query) {
        const q = query.toLowerCase()
        if (
          !a.type.toLowerCase().includes(q) &&
          !a.description.toLowerCase().includes(q) &&
          !a.id.toLowerCase().includes(q)
        )
          return false
      }
      return true
    })
  }, [agents, filter, query])

  const runningCount = agents?.filter((a) => a.status === 'running').length ?? 0
  const totalCount = agents?.length ?? 0

  return (
    <div
      style={{
        minHeight: '100%',
        background: '#0a0e27',
        color: '#e2e7ff',
        padding: '24px 28px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <style>{`
        @keyframes neoPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.55; transform: scale(1.3); }
        }
      `}</style>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <h1
            style={{
              margin: 0,
              fontSize: 24,
              fontWeight: 700,
              letterSpacing: '0.02em',
              background: 'linear-gradient(90deg, #00ffb2 0%, #20d3ff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Neo Agents
          </h1>
          <div style={{ color: '#7f88b3', fontSize: 12, marginTop: 2 }}>
            Live Claude Code · Ruflo subagents · auto-refresh 3s
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, marginLeft: 'auto', alignItems: 'center', flexWrap: 'wrap' }}>
          <HeaderStat label="total" value={totalCount} />
          <HeaderStat label="running" value={runningCount} color="#00ffb2" glow />
          <FilterPills filter={filter} setFilter={setFilter} />
          <input
            placeholder="search type / description / id"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              background: '#0f1530',
              color: '#e2e7ff',
              border: '1px solid #2a3058',
              borderRadius: 6,
              padding: '6px 10px',
              fontSize: 12,
              width: 240,
              outline: 'none',
            }}
          />
        </div>
      </div>

      {err && (
        <div
          style={{
            padding: 12,
            background: '#2a0f1f',
            border: '1px solid #ff4d6d66',
            borderRadius: 6,
            color: '#ffb3c0',
            marginBottom: 16,
            fontSize: 13,
          }}
        >
          Error: {err}
        </div>
      )}

      {agents === null && !err && (
        <div style={{ color: '#7f88b3', fontSize: 13 }}>Loading transcripts…</div>
      )}

      {agents && filtered.length === 0 && (
        <div
          style={{
            textAlign: 'center',
            padding: '60px 20px',
            border: '1px dashed #2a3058',
            borderRadius: 10,
            color: '#7f88b3',
          }}
        >
          <div style={{ fontSize: 48, opacity: 0.4, marginBottom: 10 }}>◇</div>
          <div style={{ fontSize: 14, marginBottom: 4 }}>
            {agents.length === 0 ? 'No subagent transcripts yet' : 'No agents match current filters'}
          </div>
          <div style={{ fontSize: 11, opacity: 0.7 }}>
            {agents.length === 0
              ? 'Subagents spawned by Claude Code will appear here in real-time.'
              : 'Try adjusting the filter or clearing the search.'}
          </div>
        </div>
      )}

      {/* Grid */}
      {filtered.length > 0 && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: 14,
          }}
        >
          {filtered.map((a) => (
            <AgentCard key={a.id} a={a} onOpen={() => setOpenId(a.id)} />
          ))}
        </div>
      )}

      {openId && <DetailPanel agentId={openId} onClose={() => setOpenId(null)} />}
    </div>
  )
}

function HeaderStat({ label, value, color, glow }: { label: string; value: number; color?: string; glow?: boolean }) {
  const c = color || '#e2e7ff'
  return (
    <div
      style={{
        padding: '6px 12px',
        background: '#0f1530',
        border: `1px solid ${glow ? c + '55' : '#2a3058'}`,
        borderRadius: 6,
        fontSize: 12,
        boxShadow: glow ? `0 0 14px ${c}33` : 'none',
      }}
    >
      <span style={{ color: c, fontWeight: 700, marginRight: 6 }}>{value}</span>
      <span style={{ color: '#7f88b3' }}>{label}</span>
    </div>
  )
}

function FilterPills({
  filter,
  setFilter,
}: {
  filter: 'all' | 'running' | 'done'
  setFilter: (f: 'all' | 'running' | 'done') => void
}) {
  const opts: Array<'all' | 'running' | 'done'> = ['all', 'running', 'done']
  return (
    <div style={{ display: 'flex', background: '#0f1530', border: '1px solid #2a3058', borderRadius: 6, overflow: 'hidden' }}>
      {opts.map((o) => (
        <button
          key={o}
          onClick={() => setFilter(o)}
          style={{
            background: filter === o ? '#1a1f3d' : 'transparent',
            color: filter === o ? '#00ffb2' : '#9aa3cc',
            border: 'none',
            padding: '6px 12px',
            cursor: 'pointer',
            fontSize: 11,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          {o}
        </button>
      ))}
    </div>
  )
}
