import React, { useState, useCallback, useRef } from 'react'
import { Brain, Play, Zap, Activity, BarChart3, Clock, Loader } from 'lucide-react'

// ============================================================
// Types
// ============================================================

interface RunResponse {
  spike_raster: number[][]
  spike_counts: number[]
  classification: number
  membrane_trace: number[][]
}

interface TrainProgress {
  epoch: number
  total_epochs: number
  accuracy: number
  avg_spikes: number
  elapsed_secs: number
}

type ViewState = 'idle' | 'running' | 'training' | 'done'

// ============================================================
// Spike Raster Plot (SVG)
// ============================================================

const LAYER_COLORS = ['#06b6d4', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444']

function SpikeRaster({ raster, layers }: { raster: number[][]; layers: number[] }) {
  const totalNeurons = raster.length
  if (totalNeurons === 0) return null
  const timesteps = raster[0]?.length || 0
  if (timesteps === 0) return null

  const w = 620, h = 220, pad = { t: 20, r: 20, b: 30, l: 50 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  // Build layer offsets for coloring
  const layerOffsets: number[] = []
  let off = 0
  for (const s of layers) { layerOffsets.push(off); off += s }

  const getNeuronColor = (idx: number) => {
    for (let l = layers.length - 1; l >= 0; l--) {
      if (idx >= layerOffsets[l]) return LAYER_COLORS[l % LAYER_COLORS.length]
    }
    return LAYER_COLORS[0]
  }

  // Sample neurons if too many (max 60 displayed)
  const maxDisplay = 60
  const step = totalNeurons > maxDisplay ? Math.ceil(totalNeurons / maxDisplay) : 1
  const displayNeurons: number[] = []
  for (let i = 0; i < totalNeurons; i += step) displayNeurons.push(i)

  const dots: React.ReactElement[] = []
  const tStep = Math.max(1, Math.floor(timesteps / 300))

  for (const ni of displayNeurons) {
    const ny = pad.t + (displayNeurons.indexOf(ni) / displayNeurons.length) * ch
    for (let t = 0; t < timesteps; t += tStep) {
      if (raster[ni][t] === 1) {
        const tx = pad.l + (t / timesteps) * cw
        dots.push(
          <rect key={`${ni}-${t}`} x={tx} y={ny} width={1.5} height={2}
            fill={getNeuronColor(ni)} opacity={0.8} />
        )
      }
    }
  }

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Timestep</text>
      <text x={10} y={h / 2} textAnchor="middle" fontSize="10" fill="var(--text-muted)"
        transform={`rotate(-90, 10, ${h / 2})`}>Neuron</text>
      {dots}
      {/* Layer legend */}
      {layers.map((size, l) => (
        <g key={l}>
          <rect x={pad.l + l * 80} y={4} width={8} height={8} rx={2} fill={LAYER_COLORS[l % LAYER_COLORS.length]} />
          <text x={pad.l + l * 80 + 12} y={12} fontSize="9" fill="var(--text-muted)">L{l} ({size})</text>
        </g>
      ))}
    </svg>
  )
}

// ============================================================
// Membrane Potential Trace (SVG)
// ============================================================

function MembraneTrace({ traces, neuronIndices }: { traces: number[][]; neuronIndices: number[] }) {
  if (traces.length === 0 || neuronIndices.length === 0) return null
  const timesteps = traces[0]?.length || 0
  if (timesteps === 0) return null

  const w = 620, h = 180, pad = { t: 20, r: 20, b: 30, l: 50 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b
  const threshold = 1.0

  // Find global min/max
  let mn = 0, mx = threshold
  for (const idx of neuronIndices) {
    if (!traces[idx]) continue
    for (const v of traces[idx]) { if (v < mn) mn = v; if (v > mx) mx = v }
  }
  const range = mx - mn || 1

  const x = (t: number) => pad.l + (t / timesteps) * cw
  const y = (v: number) => pad.t + ch - ((v - mn) / range) * ch

  const colors = ['#06b6d4', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444']

  // Downsample for performance
  const step = Math.max(1, Math.floor(timesteps / 300))

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      {/* Threshold line */}
      <line x1={pad.l} y1={y(threshold)} x2={w - pad.r} y2={y(threshold)}
        stroke="#ef4444" strokeWidth="1" strokeDasharray="6,3" opacity={0.6} />
      <text x={w - pad.r + 2} y={y(threshold) + 4} fontSize="9" fill="#ef4444">thresh</text>
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Timestep</text>
      {neuronIndices.map((ni, ci) => {
        if (!traces[ni]) return null
        const d = traces[ni]
        const path = Array.from({ length: Math.ceil(timesteps / step) }, (_, i) => {
          const t = i * step
          return `${i === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(d[t]).toFixed(1)}`
        }).join(' ')
        return (
          <g key={ni}>
            <path d={path} fill="none" stroke={colors[ci % colors.length]}
              strokeWidth="1.5" strokeLinecap="round" opacity={0.8} />
            <text x={w - pad.r - ci * 55 - 4} y={pad.t + 12} fontSize="9"
              fill={colors[ci % colors.length]} textAnchor="end">N{ni}</text>
          </g>
        )
      })}
    </svg>
  )
}

// ============================================================
// Accuracy Chart (for training)
// ============================================================

function AccuracyChart({ data }: { data: { epoch: number; accuracy: number }[] }) {
  if (data.length < 1) return null
  const w = 620, h = 160, pad = { t: 20, r: 20, b: 30, l: 50 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  const maxEpoch = data[data.length - 1].epoch || 1
  const x = (e: number) => pad.l + (e / maxEpoch) * cw
  const y = (a: number) => pad.t + ch - a * ch

  const path = data.map((d, i) =>
    `${i === 0 ? 'M' : 'L'}${x(d.epoch).toFixed(1)},${y(d.accuracy).toFixed(1)}`
  ).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
        <g key={v}>
          <line x1={pad.l} y1={y(v)} x2={w - pad.r} y2={y(v)} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
          <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="9" fill="var(--text-muted)">{(v * 100).toFixed(0)}%</text>
        </g>
      ))}
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Epoch</text>
      <path d={path} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {data.map((d, i) => (
        <circle key={i} cx={x(d.epoch)} cy={y(d.accuracy)} r="3" fill="#22c55e" stroke="var(--bg-primary)" strokeWidth="1.5" />
      ))}
      {data.length > 0 && (
        <text x={x(data[data.length - 1].epoch) + 6} y={y(data[data.length - 1].accuracy) + 4}
          fontSize="10" fontWeight="600" fill="#22c55e">
          {(data[data.length - 1].accuracy * 100).toFixed(1)}%
        </text>
      )}
    </svg>
  )
}

// ============================================================
// Stat Card
// ============================================================

function StatCard({ label, value, icon: Icon, color }: { label: string; value: string; icon: typeof Brain; color?: string }) {
  return (
    <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
      <Icon size={16} style={{ color: color || 'var(--text-muted)', marginBottom: '4px' }} />
      <div style={{ fontSize: '24px', fontWeight: 700, color: color || 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{value}</div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{label}</div>
    </div>
  )
}

// ============================================================
// Digit Canvas (28x28 drawing grid)
// ============================================================

function DigitCanvas({ onInput }: { onInput: (pixels: number[]) => void }) {
  const [grid, setGrid] = useState<number[]>(new Array(784).fill(0))
  const drawing = useRef(false)

  const paint = (idx: number) => {
    setGrid(prev => {
      const next = [...prev]
      next[idx] = 1.0
      // Also paint neighbors for thicker stroke
      const r = Math.floor(idx / 28), c = idx % 28
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const nr = r + dr, nc = c + dc
        if (nr >= 0 && nr < 28 && nc >= 0 && nc < 28) next[nr * 28 + nc] = Math.max(next[nr * 28 + nc], 0.5)
      }
      return next
    })
  }

  const handleMouse = (idx: number, down?: boolean) => {
    if (down !== undefined) drawing.current = down
    if (drawing.current) paint(idx)
  }

  const clear = () => setGrid(new Array(784).fill(0))
  const submit = () => onInput(grid)
  const randomize = () => {
    const g = new Array(784).fill(0)
    const cx = 10 + Math.floor(Math.random() * 8)
    const cy = 10 + Math.floor(Math.random() * 8)
    for (let i = 0; i < 80; i++) {
      const r = cy + Math.floor(Math.random() * 8 - 4)
      const c = cx + Math.floor(Math.random() * 8 - 4)
      if (r >= 0 && r < 28 && c >= 0 && c < 28) g[r * 28 + c] = Math.random() * 0.5 + 0.5
    }
    setGrid(g)
    onInput(g)
  }

  return (
    <div>
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(28, 1fr)', gap: 0,
        width: '168px', height: '168px', border: '1px solid var(--border)', borderRadius: '4px',
        overflow: 'hidden', cursor: 'crosshair',
      }}>
        {grid.map((v, i) => (
          <div key={i}
            onMouseDown={() => handleMouse(i, true)}
            onMouseUp={() => handleMouse(i, false)}
            onMouseEnter={() => handleMouse(i)}
            style={{ background: v > 0 ? `rgba(6, 182, 212, ${v})` : 'transparent' }}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
        <button className="btn" onClick={submit}
          style={{ fontSize: '11px', padding: '4px 10px' }}>Use</button>
        <button className="btn" onClick={randomize}
          style={{ fontSize: '11px', padding: '4px 10px' }}>Random</button>
        <button className="btn" onClick={clear}
          style={{ fontSize: '11px', padding: '4px 10px' }}>Clear</button>
      </div>
    </div>
  )
}

// ============================================================
// Main Component
// ============================================================

export default function SpikingView() {
  const [viewState, setViewState] = useState<ViewState>('idle')
  const [timesteps, setTimesteps] = useState(100)
  const [runResult, setRunResult] = useState<RunResponse | null>(null)
  const [inputPixels, setInputPixels] = useState<number[]>([])
  const [trainHistory, setTrainHistory] = useState<TrainProgress[]>([])
  const [trainComplete, setTrainComplete] = useState(false)
  const layers = [784, 256, 10]

  const runNetwork = useCallback(async () => {
    const input = inputPixels.length === 784 ? inputPixels : new Array(784).fill(0).map(() => Math.random())
    setViewState('running')
    try {
      const res = await fetch('/api/spiking/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input, timesteps }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: RunResponse = await res.json()
      setRunResult(data)
      setViewState('done')
    } catch {
      setViewState('idle')
    }
  }, [inputPixels, timesteps])

  const startTraining = useCallback(async () => {
    setViewState('training')
    setTrainHistory([])
    setTrainComplete(false)
    try {
      const res = await fetch('/api/spiking/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: 'mnist', timesteps: 50, epochs: 5, layers }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = '', currentEvent = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('event: ')) currentEvent = line.slice(7).trim()
          else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (currentEvent === 'progress') setTrainHistory(prev => [...prev, data as TrainProgress])
              if (currentEvent === 'complete') setTrainComplete(true)
            } catch { /* skip */ }
          }
        }
      }
      setViewState('done')
    } catch {
      setViewState('idle')
    }
  }, [timesteps]) // eslint-disable-line react-hooks/exhaustive-deps

  const totalSpikes = runResult ? runResult.spike_counts.reduce((a, b) => a + b, 0) : 0
  const totalNeurons = runResult ? runResult.spike_counts.length : 0
  const activeNeurons = runResult ? runResult.spike_counts.filter(c => c > 0).length : 0
  const avgRate = runResult && timesteps > 0 ? (totalSpikes / totalNeurons / timesteps).toFixed(3) : '0'

  // Pick 4 interesting neurons for membrane trace (from different layers)
  const traceNeurons = runResult ? [0, Math.floor(layers[0] / 2), layers[0] + 5, layers[0] + layers[1] + 1] : []

  const isRunning = viewState === 'running' || viewState === 'training'

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header + Controls */}
      <div className="card" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
          <Brain size={18} style={{ color: '#8b5cf6' }} />
          <h3 className="heading" style={{ margin: 0, fontSize: '14px' }}>Spiking Neural Network</h3>
          {isRunning && <Loader size={14} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />}
        </div>

        <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <DigitCanvas onInput={setInputPixels} />
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ fontSize: '12px', display: 'block', marginBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Timesteps: {timesteps}</span>
              <input type="range" min={20} max={200} step={10} value={timesteps}
                onChange={e => setTimesteps(parseInt(e.target.value))} disabled={isRunning}
                style={{ width: '100%' }} />
            </label>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
              Network: {layers.join(' -> ')} ({layers.reduce((a, b) => a + b, 0)} neurons)
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button className="btn btn-primary" onClick={runNetwork} disabled={isRunning}
                style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', fontSize: '13px' }}>
                <Play size={14} /> Run
              </button>
              <button className="btn" onClick={startTraining} disabled={isRunning}
                style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', fontSize: '13px',
                  background: 'color-mix(in srgb, #8b5cf6 12%, transparent)', color: '#8b5cf6',
                  border: '1px solid color-mix(in srgb, #8b5cf6 30%, transparent)', borderRadius: '6px', cursor: 'pointer' }}>
                <Zap size={14} /> Train STDP
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      {runResult && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          <StatCard label="Total Spikes" value={totalSpikes.toLocaleString()} icon={Zap} color="#06b6d4" />
          <StatCard label="Avg Firing Rate" value={avgRate} icon={Activity} color="#8b5cf6" />
          <StatCard label="Classification" value={String(runResult.classification)} icon={Brain} color="#22c55e" />
          <StatCard label="Active Neurons" value={`${((activeNeurons / totalNeurons) * 100).toFixed(0)}%`} icon={BarChart3} color="#f59e0b" />
        </div>
      )}

      {/* Spike Raster */}
      {runResult && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Zap size={16} style={{ color: '#06b6d4' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Spike Raster</span>
            <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>
              {totalSpikes} spikes / {timesteps} steps
            </span>
          </div>
          <SpikeRaster raster={runResult.spike_raster} layers={layers} />
        </div>
      )}

      {/* Membrane Trace */}
      {runResult && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Activity size={16} style={{ color: '#8b5cf6' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Membrane Potential</span>
            <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>
              Showing {traceNeurons.length} neurons
            </span>
          </div>
          <MembraneTrace traces={runResult.membrane_trace} neuronIndices={traceNeurons} />
        </div>
      )}

      {/* Training Progress */}
      {trainHistory.length > 0 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <BarChart3 size={16} style={{ color: '#22c55e' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Training (STDP)</span>
            {!trainComplete && <span style={{ fontSize: '10px', padding: '2px 8px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', borderRadius: '10px', fontWeight: 600 }}>LIVE</span>}
            {trainComplete && <span style={{ fontSize: '10px', padding: '2px 8px', background: 'color-mix(in srgb, #22c55e 15%, transparent)', color: '#22c55e', borderRadius: '10px', fontWeight: 600 }}>DONE</span>}
          </div>
          <AccuracyChart data={trainHistory.map(h => ({ epoch: h.epoch, accuracy: h.accuracy }))} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '10px' }}>
            <StatCard label="Epoch" value={`${trainHistory[trainHistory.length - 1].epoch}/${trainHistory[trainHistory.length - 1].total_epochs}`} icon={Clock} />
            <StatCard label="Accuracy" value={`${(trainHistory[trainHistory.length - 1].accuracy * 100).toFixed(1)}%`} icon={Brain} color="#22c55e" />
            <StatCard label="Avg Spikes" value={trainHistory[trainHistory.length - 1].avg_spikes.toFixed(1)} icon={Zap} color="#06b6d4" />
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
