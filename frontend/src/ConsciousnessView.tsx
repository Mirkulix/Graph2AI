import { useState, useEffect, useRef } from 'react'
import {
  BookOpen,
  Crosshair,
  AlertTriangle,
  Sparkles,
  Eye,
  HelpCircle,
  Heart,
  Activity,
} from 'lucide-react'

interface ConsciousnessState {
  mood: string
  energy: number
  heartbeat: number
  agents_active: number
  agents_idle: number
  tasks_completed: number
  tasks_failed: number
  values: {
    achtsamkeit: number
    anerkennung: number
    aufmerksamkeit: number
    entwicklung: number
    sinn: number
  }
}

const moodIcons: Record<string, typeof BookOpen> = {
  Learning: BookOpen,
  Focused: Crosshair,
  Restless: AlertTriangle,
  Creating: Sparkles,
  Reflecting: Eye,
}

const valueConfig: Record<string, { label: string; color: string }> = {
  achtsamkeit:     { label: 'Achtsamkeit',     color: 'var(--accent-achtsamkeit)' },
  anerkennung:     { label: 'Anerkennung',     color: 'var(--accent-anerkennung)' },
  aufmerksamkeit:  { label: 'Aufmerksamkeit',  color: 'var(--accent-aufmerksamkeit)' },
  entwicklung:     { label: 'Entwicklung',      color: 'var(--accent-entwicklung)' },
  sinn:            { label: 'Sinn',             color: 'var(--accent-sinn)' },
}

function energyColor(energy: number): string {
  if (energy > 60) return 'var(--accent-success)'
  if (energy > 30) return 'var(--accent-warning)'
  return 'var(--accent-danger)'
}

export default function ConsciousnessView() {
  const [state, setState] = useState<ConsciousnessState | null>(null)
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const applyState = (data: ConsciousnessState) => {
    setState(data)
  }

  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(() => {
      fetch('/api/consciousness/state')
        .then(r => r.json())
        .then(applyState)
        .catch(() => {})
    }, 2000)
    fetch('/api/consciousness/state')
      .then(r => r.json())
      .then(applyState)
      .catch(() => {})
  }

  useEffect(() => {
    // Always load state immediately via polling (SSE only fires on events)
    fetch('/api/consciousness/state')
      .then(r => r.json())
      .then(applyState)
      .catch(() => {})

    // Poll every 5 seconds as baseline (SSE supplements with real-time events)
    pollRef.current = setInterval(() => {
      fetch('/api/consciousness/state')
        .then(r => r.json())
        .then(applyState)
        .catch(() => {})
    }, 5000)

    // SSE for real-time events (chat, goals, evolution trigger state updates)
    const es = new EventSource('/api/consciousness/stream')
    esRef.current = es

    es.onopen = () => {
      setConnected(true)
    }

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'state') {
          applyState(data.state as ConsciousnessState)
        }
      } catch {}
    }

    es.onerror = () => {
      setConnected(false)
      es.close()
    }

    return () => {
      es.close()
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  if (!state) {
    return (
      <div className="view">
        <div className="empty-state">
          <Activity size={40} className="empty-icon" />
          <div className="empty-title">Verbinde mit Bewusstseinsstrom...</div>
        </div>
      </div>
    )
  }

  const MoodIcon = moodIcons[state.mood] ?? HelpCircle

  return (
    <div className="view">
      {/* Connection indicator */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '16px' }}>
        <span className={`badge ${connected ? 'badge-idle' : 'badge-pending'}`}>
          <span className="badge-dot" />
          {connected ? 'Live-Stream' : 'Polling'}
        </span>
      </div>

      {/* Mood */}
      <div className="card-static consciousness-mood">
        <div className="mood-icon-wrapper">
          <MoodIcon size={48} />
        </div>
        <div className="label" style={{ marginTop: '12px' }}>Stimmung</div>
        <h2 className="heading" style={{ fontSize: '24px', color: 'var(--accent-primary)', marginTop: '4px' }}>
          {state.mood}
        </h2>
      </div>

      {/* Energy + Heartbeat */}
      <div className="grid-2" style={{ marginTop: '16px' }}>
        <div className="card-static">
          <div className="label" style={{ marginBottom: '10px' }}>Energie</div>
          <div className="stat-value mono" style={{ color: energyColor(state.energy), marginBottom: '10px' }}>
            {Math.round(state.energy)}%
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, state.energy)}%`,
                background: energyColor(state.energy),
              }}
            />
          </div>
        </div>

        <div className="card-static">
          <div className="label" style={{ marginBottom: '10px' }}>Herzschlag</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Heart size={20} style={{ color: 'var(--accent-danger)' }} />
            <span className="stat-value mono" style={{ color: 'var(--accent-danger)' }}>
              {state.heartbeat}
            </span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Ticks</div>
        </div>
      </div>

      {/* Agents + Tasks */}
      <div className="grid-2" style={{ marginTop: '16px' }}>
        <div className="card-static">
          <div className="label" style={{ marginBottom: '12px' }}>Agenten</div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <div>
              <div className="stat-value" style={{ fontSize: '22px', color: 'var(--accent-success)' }}>
                {state.agents_active}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>aktiv</div>
            </div>
            <div>
              <div className="stat-value" style={{ fontSize: '22px', color: 'var(--text-secondary)' }}>
                {state.agents_idle}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>bereit</div>
            </div>
          </div>
        </div>

        <div className="card-static">
          <div className="label" style={{ marginBottom: '12px' }}>Aufgaben</div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <div>
              <div className="stat-value" style={{ fontSize: '22px', color: 'var(--accent-success)' }}>
                {state.tasks_completed}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>erledigt</div>
            </div>
            <div>
              <div className="stat-value" style={{ fontSize: '22px', color: 'var(--accent-danger)' }}>
                {state.tasks_failed}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>fehlgesch.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Values */}
      <div className="card-static" style={{ marginTop: '16px' }}>
        <div className="label" style={{ marginBottom: '16px' }}>Werte</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {Object.entries(state.values).map(([key, val]) => {
            const cfg = valueConfig[key]
            if (!cfg) return null
            return (
              <div key={key}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{ fontSize: '13px' }}>{cfg.label}</span>
                  <span className="mono" style={{ fontSize: '13px', color: cfg.color }}>
                    {(val * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(100, val * 100)}%`,
                      background: cfg.color,
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <style>{`
        .consciousness-mood {
          text-align: center;
          padding: 32px 24px;
        }
        .mood-icon-wrapper {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 80px;
          height: 80px;
          border-radius: 50%;
          background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
          color: var(--accent-primary);
        }
      `}</style>
    </div>
  )
}
