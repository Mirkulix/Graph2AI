import { useState, useEffect } from 'react'
import {
  Crown,
  Search,
  Code,
  Shield,
  BarChart3,
  Palette,
  Bot,
} from 'lucide-react'

interface Agent {
  role: string
  status: string
  tasks_completed: number
  tasks_failed: number
}

const agentMeta: Record<string, { icon: typeof Crown; color: string }> = {
  CEO:        { icon: Crown, color: 'var(--accent-info)' },
  Researcher: { icon: Search, color: 'var(--accent-secondary)' },
  Developer:  { icon: Code, color: 'var(--accent-success)' },
  Guardian:   { icon: Shield, color: 'var(--accent-warning)' },
  Strategist: { icon: BarChart3, color: 'var(--accent-entwicklung)' },
  Artisan:    { icon: Palette, color: 'var(--accent-anerkennung)' },
}

const statusConfig: Record<string, { cls: string; label: string }> = {
  Idle:   { cls: 'badge-idle', label: 'Bereit' },
  Active: { cls: 'badge-active', label: 'Aktiv' },
  Error:  { cls: 'badge-error', label: 'Fehler' },
}

function AgentCard({ agent }: { agent: Agent }) {
  const meta = agentMeta[agent.role] ?? { icon: Bot, color: 'var(--accent-primary)' }
  const status = statusConfig[agent.status] ?? { cls: 'badge-info', label: agent.status }
  const Icon = meta.icon
  const total = agent.tasks_completed + agent.tasks_failed
  const completePct = total > 0 ? (agent.tasks_completed / total) * 100 : 0

  return (
    <div className="card agent-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="agent-icon-wrapper" style={{ color: meta.color }}>
            <Icon size={24} />
          </div>
          <div>
            <h3 className="heading agent-role" style={{ color: meta.color }}>{agent.role}</h3>
          </div>
        </div>
        <span className={`badge ${status.cls}`}>
          <span className="badge-dot" />
          {status.label}
        </span>
      </div>

      <div className="agent-stats">
        <div className="agent-stat">
          <span className="stat-value" style={{ color: 'var(--accent-success)', fontSize: '20px' }}>
            {agent.tasks_completed}
          </span>
          <span className="label">erledigt</span>
        </div>
        <div className="agent-stat-divider" />
        <div className="agent-stat">
          <span className="stat-value" style={{ color: 'var(--accent-danger)', fontSize: '20px' }}>
            {agent.tasks_failed}
          </span>
          <span className="label">fehlgesch.</span>
        </div>
      </div>

      {total > 0 && (
        <div className="progress-track" style={{ marginTop: '4px' }}>
          <div
            className="progress-fill"
            style={{
              width: `${completePct}%`,
              background: 'var(--accent-success)',
            }}
          />
        </div>
      )}
    </div>
  )
}

export default function AgentsView() {
  const [agents, setAgents] = useState<Agent[]>([])

  useEffect(() => {
    const fetchAgents = () => {
      fetch('/api/agents')
        .then(r => r.json())
        .then(data => setAgents(Array.isArray(data) ? data : (data?.agents ?? [])))
        .catch(() => {})
    }
    fetchAgents()
    const interval = setInterval(fetchAgents, 3000)
    return () => clearInterval(interval)
  }, [])

  if (agents.length === 0) {
    return (
      <div className="view">
        <div className="empty-state">
          <Bot size={40} className="empty-icon" />
          <div className="empty-title">Lade Agenten...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="view">
      <div className="grid-3">
        {agents.map(agent => (
          <AgentCard key={agent.role} agent={agent} />
        ))}
      </div>

      <style>{`
        .agent-card {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .agent-icon-wrapper {
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          background: color-mix(in srgb, currentColor 10%, transparent);
        }
        .agent-role {
          font-size: 14px;
          margin: 0;
        }
        .agent-stats {
          display: flex;
          gap: 16px;
          align-items: flex-end;
        }
        .agent-stat {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .agent-stat-divider {
          width: 1px;
          height: 32px;
          background: var(--border);
        }
      `}</style>
    </div>
  )
}
