import { useState, useRef, useCallback, useEffect } from 'react'
import { Cpu, Zap, Activity, Play, Square, BarChart3, Clock, Download, Terminal } from 'lucide-react'

// ============================================================
// Types
// ============================================================

interface TrainingConfig {
  model: 'mamba' | 'ff'
  dataset: 'wikitext2' | 'mnist'
  n_steps: number
  d_model: number
  n_layers: number
  lr: number
}

interface ProgressEvent {
  step: number
  total_steps: number
  loss: number
  ppl: number
  steps_per_sec: number
  eta_secs: number
  generated: string
  elapsed_secs: number
}

interface CheckpointEvent {
  step: number
  path: string
  ppl: number
}

interface CompleteEvent {
  final_ppl: number
  init_ppl: number
  improvement: string
  total_time_secs: number
  model_path: string
  ternary_ppl: number
  ternary_path: string
}

type TrainingState = 'idle' | 'training' | 'complete' | 'error'

// ============================================================
// SVG Loss Chart
// ============================================================

function LossChart({ data }: { data: { step: number; loss: number }[] }) {
  if (data.length < 2) return null

  const w = 620, h = 200, pad = { t: 20, r: 20, b: 30, l: 50 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  const maxStep = data[data.length - 1].step || 1
  const minLoss = Math.min(...data.map(d => d.loss))
  const maxLoss = Math.max(...data.map(d => d.loss))
  const lossRange = maxLoss - minLoss || 1

  const x = (step: number) => pad.l + (step / maxStep) * cw
  const y = (loss: number) => pad.t + ch - ((loss - minLoss) / lossRange) * ch

  const path = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(d.step).toFixed(1)},${y(d.loss).toFixed(1)}`).join(' ')
  const fillPath = `${path} L${x(data[data.length - 1].step)},${pad.t + ch} L${x(data[0].step)},${pad.t + ch} Z`

  const minIdx = data.reduce((mi, d, i) => d.loss < data[mi].loss ? i : mi, 0)
  const gridCount = 4
  const gridLines = Array.from({ length: gridCount + 1 }, (_, i) => minLoss + (lossRange * i) / gridCount)

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <defs>
        <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {gridLines.map((v, i) => (
        <g key={i}>
          <line x1={pad.l} y1={y(v)} x2={w - pad.r} y2={y(v)} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
          <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">{v.toFixed(2)}</text>
        </g>
      ))}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Step</text>
      <path d={fillPath} fill="url(#lossFill)" />
      <path d={path} fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={x(data[minIdx].step)} cy={y(data[minIdx].loss)} r="4" fill="#06b6d4" stroke="var(--bg-primary)" strokeWidth="2" />
      <text x={x(data[minIdx].step)} y={y(data[minIdx].loss) - 8} textAnchor="middle" fontSize="10" fontWeight="600" fill="#06b6d4">
        {data[minIdx].loss.toFixed(3)}
      </text>
    </svg>
  )
}

// ============================================================
// SVG Perplexity Chart
// ============================================================

function PplChart({ data, ternaryPpl }: { data: { step: number; ppl: number }[]; ternaryPpl?: number }) {
  if (data.length < 2) return null

  const w = 620, h = 180, pad = { t: 20, r: 20, b: 30, l: 55 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  const maxStep = data[data.length - 1].step || 1
  const allPpl = data.map(d => d.ppl)
  const minPpl = Math.min(...allPpl)
  const maxPpl = Math.max(...allPpl)
  const range = maxPpl - minPpl || 1

  const x = (step: number) => pad.l + (step / maxStep) * cw
  const y = (ppl: number) => pad.t + ch - ((ppl - minPpl) / range) * ch

  const path = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(d.step).toFixed(1)},${y(d.ppl).toFixed(1)}`).join(' ')

  const initPpl = data[0]?.ppl ?? 0
  const curPpl = data[data.length - 1]?.ppl ?? 0
  const improvement = initPpl > 0 ? (initPpl / curPpl).toFixed(1) : '?'

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="var(--border)" strokeWidth="1" />
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Step</text>
      <path d={path} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {ternaryPpl != null && (
        <line x1={pad.l} y1={y(ternaryPpl)} x2={w - pad.r} y2={y(ternaryPpl)} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="6,4" />
      )}
      <text x={w - pad.r - 4} y={pad.t + 14} textAnchor="end" fontSize="11" fontWeight="600" fill="#8b5cf6">
        {improvement}x improvement
      </text>
      {data.length > 0 && (
        <text x={x(data[data.length - 1].step) + 4} y={y(curPpl) + 4} fontSize="11" fontWeight="600" fill="#8b5cf6">
          {curPpl.toFixed(1)}
        </text>
      )}
      {ternaryPpl != null && (
        <text x={w - pad.r - 4} y={y(ternaryPpl) - 6} textAnchor="end" fontSize="10" fill="#f59e0b">Ternary: {ternaryPpl.toFixed(1)}</text>
      )}
    </svg>
  )
}

// ============================================================
// Metric Card
// ============================================================

function MetricCard({ label, value, icon: Icon, color }: { label: string; value: string; icon: typeof Cpu; color?: string }) {
  return (
    <div className="card" style={{ padding: '14px', textAlign: 'center' }}>
      <Icon size={16} style={{ color: color || 'var(--text-muted)', marginBottom: '4px' }} />
      <div style={{ fontSize: '26px', fontWeight: 700, color: color || 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{value}</div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{label}</div>
    </div>
  )
}

// ============================================================
// Main Component
// ============================================================

export default function GpuTrainingView() {
  const [config, setConfig] = useState<TrainingConfig>({
    model: 'mamba', dataset: 'wikitext2', n_steps: 5000, d_model: 256, n_layers: 4, lr: 0.001,
  })
  const [state, setState] = useState<TrainingState>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([])
  const [pplHistory, setPplHistory] = useState<{ step: number; ppl: number }[]>([])
  const [checkpoints, setCheckpoints] = useState<CheckpointEvent[]>([])
  const [generations, setGenerations] = useState<string[]>([])
  const [currentMetrics, setCurrentMetrics] = useState<ProgressEvent | null>(null)
  const [completeData, setCompleteData] = useState<CompleteEvent | null>(null)
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // On mount: check if training is already running, replay history, connect SSE
  useEffect(() => {
    let cancelled = false
    const streamRef = { current: null as AbortController | null }

    const connectStream = () => {
      const ctrl = new AbortController()
      streamRef.current = ctrl
      abortRef.current = ctrl
      fetch('/api/training/gpu/stream', { signal: ctrl.signal })
        .then(async (res) => {
          if (!res.ok || cancelled) return
          const reader = res.body!.getReader()
          const decoder = new TextDecoder()
          let buffer = '', currentEvent = ''
          while (!cancelled) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
              if (line.startsWith('event: ')) currentEvent = line.slice(7).trim()
              else if (line.startsWith('data: ')) {
                try { handleSSE(currentEvent, JSON.parse(line.slice(6))) } catch { /* skip */ }
              }
            }
          }
        })
        .catch(() => { /* connection closed or aborted */ })
    }

    fetch('/api/training/gpu/status')
      .then(r => r.json())
      .then(d => {
        if (cancelled) return
        setGpuAvailable(true)
        if (d.running) {
          setState('training')
          // Replay history into charts
          if (d.history && d.history.length > 0) {
            setLossHistory(d.history.map((h: { step: number; loss: number }) => ({ step: h.step, loss: h.loss })))
            setPplHistory(d.history.map((h: { step: number; ppl: number }) => ({ step: h.step, ppl: h.ppl })))
            setGenerations(d.history.filter((h: { generated: string }) => h.generated).map((h: { generated: string }) => h.generated).reverse().slice(0, 10))
            const last = d.history[d.history.length - 1]
            setCurrentMetrics({ step: last.step, total_steps: d.total_steps, loss: last.loss, ppl: last.ppl, steps_per_sec: last.steps_per_sec, eta_secs: last.eta_secs, elapsed_secs: last.elapsed_secs, generated: last.generated })
          }
          // Connect to live stream
          connectStream()
        }
      })
      .catch(() => setGpuAvailable(null))

    return () => { cancelled = true; streamRef.current?.abort() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const updateConfig = useCallback(<K extends keyof TrainingConfig>(key: K, val: TrainingConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: val }))
  }, [])

  const startTraining = async () => {
    setState('training')
    setErrorMsg('')
    setLossHistory([])
    setPplHistory([])
    setCheckpoints([])
    setGenerations([])
    setCurrentMetrics(null)
    setCompleteData(null)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      // Fire-and-forget POST to start training
      const startRes = await fetch('/api/training/gpu', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...config, use_gpu: true }),
        signal: controller.signal,
      })

      if (!startRes.ok) {
        setState('error')
        setErrorMsg(`HTTP ${startRes.status}: ${startRes.statusText}`)
        return
      }

      // Read the POST SSE stream (same as before)
      const reader = startRes.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('event: ')) currentEvent = line.slice(7).trim()
          else if (line.startsWith('data: ')) {
            try { handleSSE(currentEvent, JSON.parse(line.slice(6))) } catch { /* skip */ }
          }
        }
      }

      if (state !== 'error') setState(prev => prev === 'training' ? 'complete' : prev)
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        setState('error')
        setErrorMsg((err as Error).message || 'Connection failed')
      }
    }
  }

  const handleSSE = (event: string, data: Record<string, unknown>) => {
    switch (event) {
      case 'progress': {
        const p = data as unknown as ProgressEvent
        setCurrentMetrics(p)
        setLossHistory(prev => {
          if (prev.length > 0 && prev[prev.length - 1].step === p.step) return prev
          return [...prev, { step: p.step, loss: p.loss }]
        })
        setPplHistory(prev => {
          if (prev.length > 0 && prev[prev.length - 1].step === p.step) return prev
          return [...prev, { step: p.step, ppl: p.ppl }]
        })
        if (p.generated) {
          setGenerations(prev => [p.generated, ...prev].slice(0, 10))
        }
        break
      }
      case 'checkpoint':
        setCheckpoints(prev => [...prev, data as unknown as CheckpointEvent])
        break
      case 'complete':
        setCompleteData(data as unknown as CompleteEvent)
        setState('complete')
        break
      case 'error':
        setState('error')
        setErrorMsg((data as { message: string }).message)
        break
    }
  }

  const stopTraining = async () => {
    abortRef.current?.abort()
    abortRef.current = null
    try { await fetch('/api/training/gpu/stop', { method: 'POST' }) } catch { /* ignore */ }
    setState('idle')
  }

  const formatEta = (secs: number) => {
    if (secs < 60) return `${Math.round(secs)}s`
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
  }

  const lossTrend = lossHistory.length >= 2
    ? lossHistory[lossHistory.length - 1].loss < lossHistory[lossHistory.length - 2].loss ? '#22c55e' : '#ef4444'
    : 'var(--text-primary)'

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '7px 10px', borderRadius: '6px',
    border: '1px solid var(--border)', background: 'var(--bg-primary)',
    color: 'var(--text-primary)', fontSize: '13px',
  }

  const isRunning = state === 'training'

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Config Panel */}
      <div className="card" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
          <Cpu size={18} style={{ color: 'var(--accent-primary)' }} />
          <h3 className="heading" style={{ margin: 0, fontSize: '14px' }}>GPU Training — Mamba LM</h3>
          {isRunning && <span style={{ fontSize: '11px', color: 'var(--accent-success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}><Zap size={12} /> TRAINING</span>}
          <span style={{ marginLeft: 'auto', fontSize: '11px', padding: '2px 8px', borderRadius: '10px', background: gpuAvailable === true ? 'color-mix(in srgb, #22c55e 15%, transparent)' : gpuAvailable === false ? 'color-mix(in srgb, #f59e0b 15%, transparent)' : 'color-mix(in srgb, var(--text-muted) 15%, transparent)', color: gpuAvailable === true ? '#22c55e' : gpuAvailable === false ? '#f59e0b' : 'var(--text-muted)', fontWeight: 600 }}>
            {gpuAvailable === true ? 'GPU' : gpuAvailable === false ? 'CPU Fallback' : 'Checking...'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px', marginBottom: '12px' }}>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Model</span>
            <select value={config.model} onChange={e => updateConfig('model', e.target.value as 'mamba' | 'ff')} style={inputStyle} disabled={isRunning}>
              <option value="mamba">Mamba LM (30M)</option>
              <option value="ff">Forward-Forward</option>
            </select>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Dataset</span>
            <select value={config.dataset} onChange={e => updateConfig('dataset', e.target.value as 'wikitext2' | 'mnist')} style={inputStyle} disabled={isRunning}>
              <option value="wikitext2">WikiText-2</option>
              <option value="mnist">MNIST</option>
            </select>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Steps</span>
            <input type="range" min={1000} max={50000} step={1000} value={config.n_steps} onChange={e => updateConfig('n_steps', parseInt(e.target.value))} disabled={isRunning} style={{ width: '100%' }} />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{config.n_steps.toLocaleString()}</span>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Model Dim</span>
            <select value={config.d_model} onChange={e => updateConfig('d_model', parseInt(e.target.value))} style={inputStyle} disabled={isRunning}>
              {[64, 128, 256, 512].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Layers</span>
            <select value={config.n_layers} onChange={e => updateConfig('n_layers', parseInt(e.target.value))} style={inputStyle} disabled={isRunning}>
              {[1, 2, 3, 4, 5, 6].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label style={{ fontSize: '12px' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '3px' }}>Learning Rate</span>
            <select value={config.lr} onChange={e => updateConfig('lr', parseFloat(e.target.value))} style={inputStyle} disabled={isRunning}>
              {[0.0001, 0.0003, 0.001, 0.003, 0.01].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-primary" onClick={startTraining} disabled={isRunning} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 20px', fontSize: '13px' }}>
            <Play size={14} /> {isRunning ? 'Training laeuft...' : 'Training starten'}
          </button>
          {isRunning && (
            <button className="btn" onClick={stopTraining} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', fontSize: '13px', background: 'color-mix(in srgb, #ef4444 15%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)', borderRadius: '6px', cursor: 'pointer' }}>
              <Square size={14} /> Stop
            </button>
          )}
        </div>
      </div>

      {/* Error banner */}
      {state === 'error' && (
        <div className="card" style={{ padding: '12px 16px', borderLeft: '3px solid #ef4444', color: '#ef4444', fontSize: '13px' }}>
          Error: {errorMsg}
        </div>
      )}

      {/* Metrics Cards */}
      {currentMetrics && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          <MetricCard label="Loss" value={currentMetrics.loss.toFixed(3)} icon={Activity} color={lossTrend} />
          <MetricCard label="Perplexity" value={currentMetrics.ppl.toFixed(1)} icon={BarChart3} color="#8b5cf6" />
          <MetricCard label="Steps/sec" value={currentMetrics.steps_per_sec.toFixed(1)} icon={Zap} color="#06b6d4" />
          <MetricCard label="ETA" value={formatEta(currentMetrics.eta_secs)} icon={Clock} color="var(--text-muted)" />
        </div>
      )}

      {/* Progress bar */}
      {isRunning && currentMetrics && (
        <div style={{ background: 'var(--bg-secondary)', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
          <div style={{ width: `${(currentMetrics.step / currentMetrics.total_steps) * 100}%`, height: '100%', background: 'linear-gradient(90deg, #06b6d4, #8b5cf6)', borderRadius: '4px', transition: 'width 0.3s ease' }} />
        </div>
      )}

      {/* Loss Chart */}
      {lossHistory.length > 1 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <BarChart3 size={16} style={{ color: '#06b6d4' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Loss Curve</span>
            {isRunning && <span style={{ fontSize: '10px', padding: '2px 8px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', borderRadius: '10px', fontWeight: 600 }}>LIVE</span>}
            <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>
              Min: {Math.min(...lossHistory.map(d => d.loss)).toFixed(3)}
            </span>
          </div>
          <LossChart data={lossHistory} />
        </div>
      )}

      {/* Perplexity Chart */}
      {pplHistory.length > 1 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Activity size={16} style={{ color: '#8b5cf6' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Perplexity</span>
          </div>
          <PplChart data={pplHistory} ternaryPpl={completeData?.ternary_ppl} />
        </div>
      )}

      {/* Generated Text */}
      {generations.length > 0 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Terminal size={16} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Generated Text</span>
            <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>Last {generations.length} samples</span>
          </div>
          <div style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {generations.map((text, i) => (
              <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', padding: '8px 10px', background: 'var(--bg-primary)', borderRadius: '6px', border: '1px solid var(--border)', color: i === 0 ? 'var(--text-primary)' : 'var(--text-muted)', lineHeight: 1.5 }}>
                {text || '(empty)'}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Checkpoint Timeline */}
      {checkpoints.length > 0 && (
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
            <Download size={16} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Checkpoints</span>
          </div>
          <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
            {checkpoints.map((cp, i) => (
              <div key={i} style={{ flexShrink: 0, padding: '8px 12px', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '12px', textAlign: 'center', minWidth: '80px' }}>
                <div style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>Step {cp.step}</div>
                <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>PPL {cp.ppl.toFixed(1)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Summary */}
      {state === 'complete' && completeData && (
        <div className="card" style={{ padding: '16px', borderLeft: '3px solid var(--accent-success)' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px', color: 'var(--accent-success)' }}>Training abgeschlossen</div>
          <div style={{ fontSize: '12px', lineHeight: 1.8, fontFamily: 'var(--font-mono)' }}>
            <div>Init PPL:      {completeData.init_ppl.toFixed(1)}</div>
            <div>Final PPL:     {completeData.final_ppl.toFixed(1)}</div>
            <div>Improvement:   {completeData.improvement}</div>
            <div>Total Time:    {formatEta(completeData.total_time_secs)}</div>
            <div>Model (f32):   {completeData.model_path}</div>
            {completeData.ternary_path && <div>Ternary PPL:   {completeData.ternary_ppl.toFixed(1)}</div>}
            {completeData.ternary_path && <div>Ternary Path:  {completeData.ternary_path}</div>}
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
    </div>
  )
}
