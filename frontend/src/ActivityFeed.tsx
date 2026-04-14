import { useState, useEffect, useRef } from 'react'
import { Minus } from 'lucide-react'

interface ActivityEntry {
  id: number
  message: string
  agent?: string
  level: string
  timestamp: number
}

const levelColor: Record<string, string> = {
  info: 'var(--text-secondary)',
  success: 'var(--accent-success)',
  error: 'var(--accent-danger)',
  progress: 'var(--accent-primary)',
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  return `${hh}:${mm}:${ss}`
}

const MAX_ENTRIES = 50
let _entryCounter = 0

export default function ActivityFeed() {
  const [entries, setEntries] = useState<ActivityEntry[]>([])
  const [minimized, setMinimized] = useState(false)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const es = new EventSource('/api/consciousness/stream')
    esRef.current = es

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'activity') {
          const entry: ActivityEntry = {
            id: ++_entryCounter,
            message: data.message,
            agent: data.agent ?? undefined,
            level: data.level ?? 'info',
            timestamp: data.timestamp,
          }
          setEntries(prev => {
            const next = [...prev, entry]
            return next.length > MAX_ENTRIES ? next.slice(next.length - MAX_ENTRIES) : next
          })
        }
      } catch {}
    }

    es.onerror = () => {}

    return () => { es.close() }
  }, [])

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [entries])

  return (
    <div className="activity-feed">
      {/* Header */}
      <div className="activity-header">
        <h3 className="heading" style={{ fontSize: '11px', letterSpacing: '0.08em' }}>
          Aktivit&auml;t
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {entries.length > 0 && (
            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
              {entries.length}
            </span>
          )}
          <button
            className="btn btn-ghost btn-icon"
            style={{ minHeight: '28px', minWidth: '28px', padding: '2px' }}
            onClick={() => setMinimized(m => !m)}
            aria-label={minimized ? 'Aufklappen' : 'Einklappen'}
          >
            <Minus size={14} />
          </button>
        </div>
      </div>

      {/* Log entries */}
      {!minimized && (
        <div className="activity-log">
          {entries.length === 0 ? (
            <div className="activity-empty">Warte auf Aktivit&auml;t...</div>
          ) : (
            entries.map(entry => (
              <div key={entry.id} className="activity-entry">
                <span className="activity-time mono">{formatTime(entry.timestamp)}</span>
                <span
                  className="activity-dot"
                  style={{ background: levelColor[entry.level] ?? 'var(--text-secondary)' }}
                />
                {entry.agent && (
                  <span className="activity-agent">[{entry.agent}]</span>
                )}
                <span
                  className="activity-message"
                  style={{ color: levelColor[entry.level] ?? 'var(--text-secondary)' }}
                >
                  {entry.message}
                </span>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      )}

      <style>{`
        .activity-feed {
          display: flex;
          flex-direction: column;
          height: 100%;
        }
        .activity-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 6px 16px;
          border-bottom: 1px solid var(--border);
          background: var(--bg-elevated);
          flex-shrink: 0;
        }
        .activity-log {
          flex: 1;
          overflow-y: auto;
          padding: 4px 0;
        }
        .activity-empty {
          padding: 8px 16px;
          font-size: 12px;
          color: var(--text-muted);
        }
        .activity-entry {
          display: flex;
          align-items: baseline;
          gap: 8px;
          padding: 2px 16px;
          font-size: 12px;
          line-height: 1.5;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .activity-time {
          color: var(--text-muted);
          flex-shrink: 0;
        }
        .activity-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
          align-self: center;
        }
        .activity-agent {
          color: var(--accent-primary);
          font-weight: 600;
          flex-shrink: 0;
        }
        .activity-message {
          overflow: hidden;
          text-overflow: ellipsis;
        }
      `}</style>
    </div>
  )
}
