import { useState, useEffect } from 'react'
import { Dna, Play, Check, X, AlertTriangle, TrendingUp, FlaskConical, Star } from 'lucide-react'

interface EvolutionState {
  strategy_weights?: Record<string, number>
  generation?: number
  fitness_score?: number
  entropy?: number
}

interface Pattern {
  id: string
  name: string
  description: string
  severity: string
  detected_at?: string
}

interface Proposal {
  id: string
  title: string
  description: string
  status: string
  proposed_at?: string
}

interface StrategyScore {
  strategy_index: number
  name: string
  avg_score: number
  success_rate: number
  avg_value_alignment: number
  avg_duration_ms: number
  avg_cost: number
  top_risks: string[]
  top_benefits: string[]
}

interface SimulationPrediction {
  scenario_id: number
  recommended_strategy: number
  recommended_name: string
  confidence: number
  strategy_scores: StrategyScore[]
}

const severityBadge: Record<string, string> = {
  low: 'badge-info',
  medium: 'badge-pending',
  high: 'badge-error',
  info: 'badge-info',
}

export default function EvolutionView() {
  const [evoState, setEvoState] = useState<EvolutionState | null>(null)
  const [patterns, setPatterns] = useState<Pattern[]>([])
  const [proposals, setProposals] = useState<Proposal[]>([])
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Simulation state
  const [simDescription, setSimDescription] = useState('')
  const [simCount, setSimCount] = useState(20)
  const [simulating, setSimulating] = useState(false)
  const [simError, setSimError] = useState<string | null>(null)
  const [simResult, setSimResult] = useState<SimulationPrediction | null>(null)

  const fetchAll = async () => {
    try {
      const [stateRes, patternsRes, proposalsRes] = await Promise.all([
        fetch('/api/evolution/state').catch(() => null),
        fetch('/api/evolution/patterns').catch(() => null),
        fetch('/api/evolution/proposals').catch(() => null),
      ])

      if (stateRes?.ok) {
        const data = await stateRes.json()
        setEvoState(data)
      }
      if (patternsRes?.ok) {
        const data = await patternsRes.json()
        setPatterns(Array.isArray(data) ? data : (data?.patterns ?? []))
      }
      if (proposalsRes?.ok) {
        const data = await proposalsRes.json()
        setProposals(Array.isArray(data) ? data : (data?.proposals ?? []))
      }
    } catch {
      // Silently handle — evolution endpoints may not exist yet
    }
  }

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleAnalyze = async () => {
    setAnalyzing(true)
    setError(null)
    try {
      const res = await fetch('/api/evolution/analyze', { method: 'POST' })
      if (!res.ok) throw new Error('Analyse fehlgeschlagen')
      await fetchAll()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setAnalyzing(false)
    }
  }

  const handleProposal = async (id: string, action: 'approve' | 'reject') => {
    try {
      await fetch(`/api/evolution/proposals/${id}/${action}`, { method: 'POST' })
      fetchAll()
    } catch {}
  }

  const handleSimulate = async () => {
    if (!simDescription.trim()) return
    setSimulating(true)
    setSimError(null)
    setSimResult(null)
    try {
      const res = await fetch('/api/simulation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: simDescription, num_simulations: simCount }),
      })
      if (!res.ok) throw new Error('Simulation fehlgeschlagen')
      const data: SimulationPrediction = await res.json()
      setSimResult(data)
    } catch (err) {
      setSimError((err as Error).message)
    } finally {
      setSimulating(false)
    }
  }

  const hasData = evoState?.strategy_weights || patterns.length > 0 || proposals.length > 0

  return (
    <div className="view">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="evo-icon-wrapper">
            <Dna size={24} />
          </div>
          <div>
            <h2 className="heading" style={{ fontSize: '18px', margin: 0 }}>Evolution</h2>
            {evoState?.generation != null && (
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Generation {evoState.generation}
                {evoState.fitness_score != null && ` \u00b7 Fitness: ${(evoState.fitness_score * 100).toFixed(1)}%`}
                {evoState.entropy != null && ` \u00b7 Entropie: ${evoState.entropy.toFixed(4)}`}
              </div>
            )}
          </div>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleAnalyze}
          disabled={analyzing}
        >
          <Play size={16} />
          {analyzing ? 'Analysiere...' : 'Analyse starten'}
        </button>
      </div>

      {error && (
        <div style={{ marginBottom: '16px', fontSize: '12px', color: 'var(--accent-danger)' }}>{error}</div>
      )}

      {!hasData && (
        <div className="empty-state">
          <Dna size={40} className="empty-icon" />
          <div className="empty-title">Evolution -- Phase 4</div>
          <div className="empty-hint">
            Starte eine Analyse, um Muster zu erkennen und Verbesserungsvorschl&auml;ge zu generieren.
          </div>
        </div>
      )}

      {hasData && (
        <>
          {/* Strategy weights */}
          {evoState?.strategy_weights && Object.keys(evoState.strategy_weights).length > 0 && (
            <div className="card-static" style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <TrendingUp size={16} style={{ color: 'var(--accent-primary)' }} />
                <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Strategie-Gewichte</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(evoState.strategy_weights).map(([key, val]) => (
                  <div key={key}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '12px' }}>{key}</span>
                      <span className="mono" style={{ fontSize: '12px', color: 'var(--accent-primary)' }}>
                        {(val * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${Math.min(100, val * 100)}%`,
                          background: 'var(--accent-primary)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patterns */}
          {patterns.length > 0 && (
            <div className="card-static" style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <AlertTriangle size={16} style={{ color: 'var(--accent-warning)' }} />
                <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Erkannte Muster</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {patterns.map(p => (
                  <div key={p.id} className="evo-pattern-item">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 500 }}>{p.name}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {p.description}
                        </div>
                      </div>
                      <span className={`badge ${severityBadge[p.severity] ?? 'badge-info'}`}>
                        {p.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Proposals */}
          {proposals.length > 0 && (
            <div className="card-static">
              <h3 className="heading" style={{ fontSize: '13px', margin: '0 0 16px' }}>Vorschl&auml;ge</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {proposals.map(p => (
                  <div key={p.id} className="evo-proposal-card">
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 500 }}>{p.title}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        {p.description}
                      </div>
                    </div>
                    {p.status === 'pending' && (
                      <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                        <button
                          className="btn btn-ghost"
                          style={{ color: 'var(--accent-success)', minHeight: '36px', padding: '4px 12px' }}
                          onClick={() => handleProposal(p.id, 'approve')}
                        >
                          <Check size={16} /> Annehmen
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ color: 'var(--accent-danger)', minHeight: '36px', padding: '4px 12px' }}
                          onClick={() => handleProposal(p.id, 'reject')}
                        >
                          <X size={16} /> Ablehnen
                        </button>
                      </div>
                    )}
                    {p.status !== 'pending' && (
                      <span
                        className={`badge ${p.status === 'approved' ? 'badge-completed' : 'badge-error'}`}
                        style={{ marginTop: '8px', alignSelf: 'flex-start' }}
                      >
                        {p.status === 'approved' ? 'Genehmigt' : 'Abgelehnt'}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Simulation Section */}
      <div className="card-static evo-simulation-section" style={{ marginTop: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <FlaskConical size={18} style={{ color: 'var(--accent-info)' }} />
          <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Szenario-Simulation</h3>
        </div>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
          <input
            className="input"
            style={{ flex: 1, fontSize: '13px' }}
            placeholder="Beschreibe ein Szenario..."
            value={simDescription}
            onChange={e => setSimDescription(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSimulate()}
          />
          <input
            type="number"
            className="input"
            style={{ width: '80px', fontSize: '13px' }}
            title="Anzahl Simulationen"
            min={1}
            max={200}
            value={simCount}
            onChange={e => setSimCount(Number(e.target.value))}
          />
          <button
            className="btn btn-primary"
            onClick={handleSimulate}
            disabled={simulating || !simDescription.trim()}
            style={{ minWidth: '110px' }}
          >
            <Play size={15} />
            {simulating ? 'Simuliere...' : 'Simulieren'}
          </button>
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
          Anzahl Simulationen: {simCount}
        </div>

        {simError && (
          <div style={{ fontSize: '12px', color: 'var(--accent-danger)', marginBottom: '12px' }}>{simError}</div>
        )}

        {simResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Confidence meter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Konfidenz</span>
              <div className="progress-track" style={{ flex: 1 }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.round(simResult.confidence * 100)}%`,
                    background: simResult.confidence > 0.7 ? 'var(--accent-success)' : 'var(--accent-warning)',
                  }}
                />
              </div>
              <span className="mono" style={{ fontSize: '12px', color: 'var(--accent-primary)', minWidth: '38px', textAlign: 'right' }}>
                {Math.round(simResult.confidence * 100)}%
              </span>
            </div>

            {/* Strategy cards */}
            {simResult.strategy_scores.map(s => {
              const isRecommended = s.strategy_index === simResult.recommended_strategy
              return (
                <div
                  key={s.strategy_index}
                  className="evo-sim-strategy-card"
                  style={isRecommended ? { borderColor: 'var(--accent-success)', background: 'color-mix(in srgb, var(--accent-success) 6%, var(--bg-primary))' } : {}}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {isRecommended && <Star size={13} style={{ color: 'var(--accent-success)' }} fill="currentColor" />}
                      <span style={{ fontSize: '13px', fontWeight: 600, color: isRecommended ? 'var(--accent-success)' : undefined }}>
                        {s.name}
                      </span>
                    </div>
                    {isRecommended && (
                      <span className="badge badge-completed" style={{ fontSize: '10px' }}>Empfohlen</span>
                    )}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', marginBottom: '8px' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div className="mono" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                        {Math.round(s.success_rate * 100)}%
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Erfolg</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div className="mono" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                        {s.avg_score.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Ø Score</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div className="mono" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                        {Math.round(s.avg_value_alignment * 100)}%
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Werte</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: s.top_risks.length > 0 || s.top_benefits.length > 0 ? '8px' : 0 }}>
                    <span>{Math.round(s.avg_duration_ms / 1000)}s Dauer</span>
                    <span>{s.avg_cost === 0 ? 'kostenlos' : `$${s.avg_cost.toFixed(4)}`}</span>
                  </div>

                  {s.top_benefits.length > 0 && (
                    <div style={{ marginBottom: '4px' }}>
                      {s.top_benefits.map((b, i) => (
                        <div key={i} style={{ fontSize: '11px', color: 'var(--accent-success)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Check size={10} /> {b}
                        </div>
                      ))}
                    </div>
                  )}
                  {s.top_risks.length > 0 && (
                    <div>
                      {s.top_risks.map((r, i) => (
                        <div key={i} style={{ fontSize: '11px', color: 'var(--accent-danger)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <AlertTriangle size={10} /> {r}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      <style>{`
        .evo-icon-wrapper {
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
          color: var(--accent-primary);
        }
        .evo-pattern-item {
          padding: 10px 14px;
          background: var(--bg-primary);
          border-radius: var(--radius-sm);
          border-left: 3px solid var(--accent-warning);
        }
        .evo-proposal-card {
          padding: 14px 16px;
          background: var(--bg-primary);
          border-radius: var(--radius-md);
          border: 1px solid var(--border);
          display: flex;
          flex-direction: column;
        }
        .evo-simulation-section {
          border-color: var(--accent-info);
          border-left: 3px solid var(--accent-info);
        }
        .evo-sim-strategy-card {
          padding: 12px 14px;
          background: var(--bg-primary);
          border-radius: var(--radius-md);
          border: 1px solid var(--border);
        }
      `}</style>
    </div>
  )
}
