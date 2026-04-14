import { useState, useEffect } from 'react'
import {
  MessageSquare,
  Target,
  Users,
  Dna,
  CheckCircle,
  XCircle,
  ClipboardList,
  ChevronDown,
  ChevronUp,
  Clock,
} from 'lucide-react'

interface HistoryEntry {
  id: number
  timestamp: number
  action_type: string
  description: string
  details: string
}

const typeConfig: Record<string, { icon: typeof MessageSquare; color: string; label: string }> = {
  chat:               { icon: MessageSquare, color: 'var(--accent-info)',    label: 'Chat' },
  goal_created:       { icon: Target,        color: 'var(--accent-warning)', label: 'Ziel erstellt' },
  goal_completed:     { icon: CheckCircle,   color: 'var(--accent-success)', label: 'Ziel erledigt' },
  goal_failed:        { icon: XCircle,       color: 'var(--accent-danger)',  label: 'Ziel fehlgeschlagen' },
  agent_executed:     { icon: Users,         color: 'var(--accent-anerkennung)', label: 'Agent ausgef\u00FChrt' },
  evolution_analyzed: { icon: Dna,           color: 'var(--accent-primary)', label: 'Evolution analysiert' },
  proposal_approved:  { icon: CheckCircle,   color: 'var(--accent-success)', label: 'Vorschlag genehmigt' },
  proposal_rejected:  { icon: XCircle,       color: 'var(--accent-danger)',  label: 'Vorschlag abgelehnt' },
}

function relativeTime(timestamp: number): string {
  const now = Math.floor(Date.now() / 1000)
  const diff = now - timestamp
  if (diff < 60) return 'gerade eben'
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`
  return `vor ${Math.floor(diff / 86400)} Tagen`
}

function HistoryItem({ entry, onNavigate }: { entry: HistoryEntry; onNavigate?: (tab: string) => void }) {
  const [expanded, setExpanded] = useState(false)
  const cfg = typeConfig[entry.action_type] ?? {
    icon: ClipboardList,
    color: 'var(--text-secondary)',
    label: entry.action_type,
  }
  const Icon = cfg.icon

  return (
    <div className="timeline-item">
      {/* Timeline dot + line */}
      <div className="timeline-track">
        <div className="timeline-dot" style={{ borderColor: cfg.color, color: cfg.color }}>
          <Icon size={12} />
        </div>
        <div className="timeline-line" />
      </div>

      {/* Content */}
      <div className="timeline-content" style={{ borderLeftColor: cfg.color }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
          <div style={{ flex: 1 }}>
            <span className="badge" style={{
              background: `color-mix(in srgb, ${cfg.color} 12%, transparent)`,
              color: cfg.color,
              border: `1px solid color-mix(in srgb, ${cfg.color} 25%, transparent)`,
              marginRight: '8px',
            }}>
              {cfg.label}
            </span>
            <span style={{ fontSize: '12px' }}>
              {entry.description.length > 80 ? entry.description.slice(0, 80) + '\u2026' : entry.description}
            </span>
          </div>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', flexShrink: 0 }}>
            {relativeTime(entry.timestamp)}
          </span>
        </div>
        <div style={{ marginTop: '6px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {entry.details && (
            <button
              className="btn btn-ghost"
              style={{ minHeight: '28px', padding: '2px 8px', fontSize: '10px' }}
              onClick={() => setExpanded(e => !e)}
            >
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {expanded ? 'Ausblenden' : 'Details'}
            </button>
          )}
          {entry.action_type === 'chat' && onNavigate && (
            <button
              className="btn btn-ghost"
              style={{ minHeight: '28px', padding: '2px 8px', fontSize: '10px', color: 'var(--accent-info)' }}
              onClick={() => onNavigate('chat')}
            >
              <MessageSquare size={12} style={{ marginRight: '4px' }} />
              Zum Chat
            </button>
          )}
          {(entry.action_type === 'goal_created' || entry.action_type === 'goal_completed' || entry.action_type === 'goal_failed') && onNavigate && (
            <button
              className="btn btn-ghost"
              style={{ minHeight: '28px', padding: '2px 8px', fontSize: '10px', color: 'var(--accent-warning)' }}
              onClick={() => onNavigate('goals')}
            >
              <Target size={12} style={{ marginRight: '4px' }} />
              Zu Zielen
            </button>
          )}
        </div>
        {expanded && entry.details && (
          <div className="timeline-details" style={{ marginTop: '6px' }}>
            {entry.details}
          </div>
        )}
      </div>
    </div>
  )
}

export default function HistorieView({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)

  const fetchHistory = () => {
    fetch('/api/history?limit=100')
      .then(r => r.json())
      .then(data => {
        setEntries(Array.isArray(data) ? data : [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }

  useEffect(() => {
    fetchHistory()
    const interval = setInterval(fetchHistory, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="view">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Clock size={20} style={{ color: 'var(--accent-primary)' }} />
          <h2 className="heading" style={{ margin: 0, fontSize: '16px' }}>Aktionsverlauf</h2>
        </div>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{entries.length} Eintr&auml;ge</span>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="empty-title">Laden...</div>
        </div>
      ) : entries.length === 0 ? (
        <div className="empty-state">
          <Clock size={40} className="empty-icon" />
          <div className="empty-title">Noch keine Aktionen aufgezeichnet.</div>
        </div>
      ) : (
        <div className="timeline">
          {entries.map(entry => (
            <HistoryItem key={entry.id} entry={entry} onNavigate={onNavigate} />
          ))}
        </div>
      )}

      <style>{`
        .timeline {
          display: flex;
          flex-direction: column;
        }
        .timeline-item {
          display: flex;
          gap: 12px;
          animation: fadeIn 300ms ease-out;
        }
        .timeline-track {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 28px;
          flex-shrink: 0;
        }
        .timeline-dot {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          border: 2px solid;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-primary);
          flex-shrink: 0;
        }
        .timeline-line {
          flex: 1;
          width: 2px;
          background: var(--border);
          margin-top: 4px;
          min-height: 12px;
        }
        .timeline-content {
          flex: 1;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-left: 3px solid;
          border-radius: var(--radius-md);
          padding: 10px 14px;
          margin-bottom: 8px;
        }
        .timeline-details {
          margin-top: 6px;
          font-size: 11px;
          color: var(--text-secondary);
          background: var(--bg-primary);
          border-radius: var(--radius-sm);
          padding: 8px 10px;
          word-break: break-word;
        }
      `}</style>
    </div>
  )
}
