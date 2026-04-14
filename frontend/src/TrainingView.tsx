import { useState, useEffect, useRef } from 'react'
import { Play, Activity, Cpu, BarChart3, Zap, CheckCircle, Circle } from 'lucide-react'

// ============================================================
// Types
// ============================================================

interface MonitorEpoch {
  epoch: number
  total_epochs: number
  f32_accuracy: number
  ternary_accuracy: number
  pos_goodness: number
  neg_goodness: number
}

interface MonitorData {
  running: boolean
  epochs: MonitorEpoch[]
  current_epoch: number
  total_epochs: number
  best_f32: number
  best_ternary: number
}

interface TrainResult {
  success: boolean
  f32_accuracy: number
  ternary_accuracy: number
  total_params: number
  f32_size_kb: number
  ternary_size_kb: number
  compression_ratio: number
  train_time_secs: number
  model_file: string | null
  epochs: MonitorEpoch[]
}

// ============================================================
// SVG Accuracy Chart
// ============================================================

function AccuracyChart({ epochs }: { epochs: MonitorEpoch[] }) {
  if (epochs.length === 0) return null

  const w = 600, h = 200, pad = { t: 20, r: 20, b: 30, l: 45 }
  const cw = w - pad.l - pad.r
  const ch = h - pad.t - pad.b

  const maxEpoch = Math.max(...epochs.map(e => e.epoch), 1)
  const x = (epoch: number) => pad.l + (epoch / maxEpoch) * cw
  const y = (acc: number) => pad.t + ch - acc * ch

  const f32Path = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${x(e.epoch).toFixed(1)},${y(e.f32_accuracy).toFixed(1)}`).join(' ')
  const ternPath = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${x(e.epoch).toFixed(1)},${y(e.ternary_accuracy).toFixed(1)}`).join(' ')

  const gridLines = [0.2, 0.4, 0.6, 0.8, 1.0]

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      {/* Grid */}
      {gridLines.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={y(v)} x2={w - pad.r} y2={y(v)} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
          <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">{(v * 100).toFixed(0)}%</text>
        </g>
      ))}

      {/* Axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Epoch</text>

      {/* f32 line */}
      <path d={f32Path} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

      {/* Ternary line */}
      <path d={ternPath} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

      {/* Ternary fill */}
      {epochs.length > 1 && (
        <path
          d={`${ternPath} L${x(epochs[epochs.length - 1].epoch)},${y(0)} L${x(epochs[0].epoch)},${y(0)} Z`}
          fill="#3b82f6" fillOpacity="0.08"
        />
      )}

      {/* Data points */}
      {epochs.map(e => (
        <g key={e.epoch}>
          <circle cx={x(e.epoch)} cy={y(e.f32_accuracy)} r="3" fill="#8b5cf6" />
          <circle cx={x(e.epoch)} cy={y(e.ternary_accuracy)} r="3.5" fill="#3b82f6" />
        </g>
      ))}

      {/* Legend */}
      <g transform={`translate(${pad.l + 10}, ${pad.t + 10})`}>
        <line x1="0" y1="0" x2="16" y2="0" stroke="#8b5cf6" strokeWidth="2" />
        <text x="20" y="4" fontSize="10" fill="var(--text-secondary)">f32 Shadow</text>
        <line x1="0" y1="14" x2="16" y2="14" stroke="#3b82f6" strokeWidth="2.5" />
        <text x="20" y="18" fontSize="10" fill="var(--text-secondary)">Ternary {'{-1,0,+1}'}</text>
      </g>

      {/* Latest value labels */}
      {epochs.length > 0 && (
        <g>
          <text x={x(epochs[epochs.length - 1].epoch) + 6} y={y(epochs[epochs.length - 1].f32_accuracy) + 4}
            fontSize="11" fontWeight="600" fill="#8b5cf6">
            {(epochs[epochs.length - 1].f32_accuracy * 100).toFixed(1)}%
          </text>
          <text x={x(epochs[epochs.length - 1].epoch) + 6} y={y(epochs[epochs.length - 1].ternary_accuracy) + 4}
            fontSize="11" fontWeight="600" fill="#3b82f6">
            {(epochs[epochs.length - 1].ternary_accuracy * 100).toFixed(1)}%
          </text>
        </g>
      )}
    </svg>
  )
}

// ============================================================
// Goodness Chart (pos vs neg separation)
// ============================================================

function GoodnessChart({ epochs }: { epochs: MonitorEpoch[] }) {
  if (epochs.length === 0) return null

  const w = 600, h = 120, pad = { t: 15, r: 20, b: 25, l: 45 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  const maxEpoch = Math.max(...epochs.map(e => e.epoch), 1)
  const allVals = epochs.flatMap(e => [e.pos_goodness, e.neg_goodness])
  const maxVal = Math.max(...allVals, 1)

  const x = (epoch: number) => pad.l + (epoch / maxEpoch) * cw
  const y = (val: number) => pad.t + ch - (val / maxVal) * ch

  const posPath = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${x(e.epoch).toFixed(1)},${y(e.pos_goodness).toFixed(1)}`).join(' ')
  const negPath = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${x(e.epoch).toFixed(1)},${y(e.neg_goodness).toFixed(1)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <path d={posPath} fill="none" stroke="#22c55e" strokeWidth="2" />
      <path d={negPath} fill="none" stroke="#ef4444" strokeWidth="2" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="0.5" />
      <g transform={`translate(${pad.l + 10}, ${pad.t + 4})`}>
        <circle cx="0" cy="0" r="3" fill="#22c55e" /><text x="8" y="4" fontSize="9" fill="var(--text-muted)">Positive (echte Daten)</text>
        <circle cx="130" cy="0" r="3" fill="#ef4444" /><text x="138" y="4" fontSize="9" fill="var(--text-muted)">Negative (falsche Labels)</text>
      </g>
    </svg>
  )
}

// ============================================================
// Phase Explainer
// ============================================================

function PhaseExplainer({ running, epochs }: { running: boolean; epochs: MonitorEpoch[] }) {
  const lastEpoch = epochs.length > 0 ? epochs[epochs.length - 1] : null
  const phase1Done = epochs.length > 0
  const phase2Active = running && epochs.length > 0
  const done = !running && epochs.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Phase 1 */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', opacity: phase1Done ? 1 : 0.4 }}>
        {phase1Done ? <CheckCircle size={18} style={{ color: 'var(--accent-success)', flexShrink: 0, marginTop: '2px' }} /> : <Circle size={18} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: '2px' }} />}
        <div>
          <div style={{ fontSize: '13px', fontWeight: 600 }}>Phase 1: Statistische Initialisierung</div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5, marginTop: '2px' }}>
            Fuer jede Klasse (0-9) wird der Mittelwert aller Trainingsbilder berechnet.
            Die Differenz zum Gesamtmittelwert wird ternaer quantisiert: starke positive Pixel werden +1,
            starke negative -1, der Rest 0. Das erzeugt sofort einen brauchbaren Classifier — ohne
            einen einzigen Trainingsschritt.
          </div>
        </div>
      </div>

      {/* Phase 2 */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', opacity: phase2Active || done ? 1 : 0.4 }}>
        {done ? <CheckCircle size={18} style={{ color: 'var(--accent-success)', flexShrink: 0, marginTop: '2px' }} /> :
         phase2Active ? <Activity size={18} style={{ color: 'var(--accent-primary)', flexShrink: 0, marginTop: '2px', animation: 'pulse 1.5s ease-in-out infinite' }} /> :
         <Circle size={18} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: '2px' }} />}
        <div>
          <div style={{ fontSize: '13px', fontWeight: 600 }}>Phase 2: Forward-Forward Training</div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5, marginTop: '2px' }}>
            Jede Schicht lernt lokal: Echte Daten (Bild + richtiges Label) sollen hohe "Goodness"
            erzeugen, falsche Daten (Bild + falsches Label) niedrige. Die Differenz treibt die
            Gewichts-Updates. Kein Gradient fliesst ueber Schichten hinweg — jede Schicht optimiert sich selbst.
            Die f32 Shadow-Weights werden nach jeder Epoche zu ternaeren Gewichten synchronisiert.
          </div>
        </div>
      </div>

      {/* Result interpretation */}
      {lastEpoch && (
        <div style={{ background: 'color-mix(in srgb, var(--accent-primary) 5%, transparent)', borderRadius: '8px', padding: '12px', fontSize: '12px', lineHeight: 1.6 }}>
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>Was die Zahlen bedeuten:</div>
          <div><strong style={{ color: '#8b5cf6' }}>f32 ({(lastEpoch.f32_accuracy * 100).toFixed(1)}%)</strong> — Genauigkeit mit den vollen Fliesskomma-Gewichten. Das ist die Obergrenze.</div>
          <div><strong style={{ color: '#3b82f6' }}>Ternary ({(lastEpoch.ternary_accuracy * 100).toFixed(1)}%)</strong> — Genauigkeit mit nur drei Werten pro Gewicht: -1, 0, +1. Das ist 16x kleiner und braucht keine Multiplikation bei Inference.</div>
          <div style={{ marginTop: '4px' }}><strong>Goodness-Gap:</strong> Positiv={lastEpoch.pos_goodness.toFixed(2)}, Negativ={lastEpoch.neg_goodness.toFixed(2)} — je groesser die Luecke, desto besser trennt das Netzwerk echte von falschen Daten.</div>
        </div>
      )}
    </div>
  )
}

// ============================================================
// Main Component
// ============================================================

export default function TrainingView() {
  const [dataset, setDataset] = useState('mnist')
  const [epochs, setEpochs] = useState(10)
  const [samples, setSamples] = useState(10000)
  const [hidden, setHidden] = useState('256, 128')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [monitor, setMonitor] = useState<MonitorData | null>(null)

  // Poll live monitor
  useEffect(() => {
    const poll = () => {
      fetch('/api/training/monitor')
        .then(r => r.json())
        .then(setMonitor)
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [])

  const startTraining = async () => {
    setRunning(true)
    setResult(null)
    try {
      const hiddenLayers = hidden.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
      const res = await fetch('/api/training/qlang', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset, epochs, train_samples: samples, hidden_layers: hiddenLayers }),
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setResult({ success: false, f32_accuracy: 0, ternary_accuracy: 0, total_params: 0,
        f32_size_kb: 0, ternary_size_kb: 0, compression_ratio: 0, train_time_secs: 0,
        model_file: null, epochs: [] })
    }
    setRunning(false)
  }

  // Choose which epochs to display: result > monitor
  const displayEpochs = result?.epochs ?? monitor?.epochs ?? []
  const isLive = !result && (monitor?.running ?? false)
  const bestTernary = Math.max(...displayEpochs.map(e => e.ternary_accuracy), 0)
  const bestF32 = Math.max(...displayEpochs.map(e => e.f32_accuracy), 0)
  const currentEpoch = displayEpochs.length > 0 ? displayEpochs[displayEpochs.length - 1].epoch : 0
  const totalEpochs = displayEpochs.length > 0 ? displayEpochs[displayEpochs.length - 1].total_epochs : 0

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header + Config */}
      <div className="card" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
          <Cpu size={18} style={{ color: 'var(--accent-primary)' }} />
          <h3 className="heading" style={{ margin: 0, fontSize: '14px' }}>QLANG Ternary Training</h3>
          {isLive && <span style={{ fontSize: '11px', color: 'var(--accent-success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}><Zap size={12} /> LIVE</span>}
          <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>
            Gewichte: nur -1, 0, +1 | Kein Backpropagation
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px', marginBottom: '12px' }}>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Dataset</span>
            <select value={dataset} onChange={e => setDataset(e.target.value)} style={{ width: '100%', padding: '7px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }}>
              <option value="mnist">MNIST (Handschrift)</option>
              <option value="synthetic">Synthetisch</option>
            </select>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Epochen</span>
            <input type="number" value={epochs} onChange={e => setEpochs(parseInt(e.target.value) || 10)} min={1} max={100} style={{ width: '100%', padding: '7px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Samples</span>
            <input type="number" value={samples} onChange={e => setSamples(parseInt(e.target.value) || 5000)} min={100} max={60000} step={1000} style={{ width: '100%', padding: '7px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Hidden Layers</span>
            <input type="text" value={hidden} onChange={e => setHidden(e.target.value)} style={{ width: '100%', padding: '7px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </label>
        </div>
        <button className="btn btn-primary" onClick={startTraining} disabled={running} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 20px', fontSize: '13px' }}>
          <Play size={14} /> {running ? 'Training laeuft...' : 'Training starten'}
        </button>
      </div>

      {/* Metrics Row */}
      {displayEpochs.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '26px', fontWeight: 700, color: '#3b82f6', fontFamily: 'var(--font-mono)' }}>{(bestTernary * 100).toFixed(1)}%</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Best Ternary</div>
          </div>
          <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '26px', fontWeight: 700, color: '#8b5cf6', fontFamily: 'var(--font-mono)' }}>{(bestF32 * 100).toFixed(1)}%</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Best f32</div>
          </div>
          <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '26px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{currentEpoch}<span style={{ fontSize: '14px', color: 'var(--text-muted)' }}>/{totalEpochs}</span></div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Epoch</div>
          </div>
          <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--accent-success)', fontFamily: 'var(--font-mono)' }}>16x</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Kompression</div>
          </div>
        </div>
      )}

      {/* Accuracy Chart */}
      {displayEpochs.length > 0 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <BarChart3 size={16} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Accuracy (f32 vs Ternary)</span>
            {isLive && <span style={{ fontSize: '10px', padding: '2px 8px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', borderRadius: '10px', fontWeight: 600 }}>LIVE</span>}
          </div>
          <AccuracyChart epochs={displayEpochs} />
        </div>
      )}

      {/* Goodness Chart */}
      {displayEpochs.length > 0 && displayEpochs[0].pos_goodness > 0 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Activity size={16} style={{ color: '#22c55e' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Goodness Separation</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Je weiter auseinander, desto besser</span>
          </div>
          <GoodnessChart epochs={displayEpochs} />
        </div>
      )}

      {/* Explainer */}
      <div className="card" style={{ padding: '16px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px' }}>Was passiert hier?</div>
        <PhaseExplainer running={isLive || running} epochs={displayEpochs} />
      </div>

      {/* Result Details */}
      {result && result.success && (
        <div className="card" style={{ padding: '16px', borderLeft: '3px solid var(--accent-success)' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px', color: 'var(--accent-success)' }}>Training abgeschlossen</div>
          <div style={{ fontSize: '12px', lineHeight: 1.8, fontFamily: 'var(--font-mono)' }}>
            <div>Methode:     Forward-Forward Ternary (kein Backprop)</div>
            <div>Parameter:   {result.total_params.toLocaleString()}</div>
            <div>f32 Size:    {result.f32_size_kb.toFixed(1)} KB</div>
            <div>Ternary:     {result.ternary_size_kb.toFixed(1)} KB ({result.compression_ratio.toFixed(1)}x kleiner)</div>
            <div>Zeit:        {result.train_time_secs.toFixed(1)}s</div>
            {result.model_file && <div>Modell:      {result.model_file}</div>}
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
    </div>
  )
}
