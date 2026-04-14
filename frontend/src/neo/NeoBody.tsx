import { useEffect, useState } from 'react'

// ── Types mirroring /api/neo/hardware ─────────────────────────────
interface GpuInfo {
  index: number
  name: string
  util: number
  mem_used_mb: number
  mem_total_mb: number
  temp: number
  power: number
}

interface Hardware {
  gpus: GpuInfo[]
  cpu_cores: number
  cpu_util: number
  ram_gb: number
  ram_used_gb: number
  source: string
}

// Temperature → color (green → amber → red)
function tempColor(temp: number): string {
  if (temp < 50) return '#2dd4a0'
  if (temp < 70) return '#f0b429'
  return '#ff4f6e'
}

// A glowing ring whose stroke-arc length = utilization %.
function GpuRing({ gpu }: { gpu: GpuInfo }) {
  const size = 180
  const stroke = 14
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const util = Math.min(100, Math.max(0, gpu.util))
  const color = tempColor(gpu.temp)
  const memPct = gpu.mem_total_mb > 0 ? (gpu.mem_used_mb / gpu.mem_total_mb) * 100 : 0

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 10, padding: 20,
      background: 'linear-gradient(145deg,#0a0d14,#0f1219)',
      border: '1px solid #1a2030', borderRadius: 18,
      boxShadow: `0 0 30px ${color}15, inset 0 0 20px #0006`,
      minWidth: 240,
    }}>
      <div style={{ fontSize: 11, color: '#7a8199', letterSpacing: 1, textTransform: 'uppercase' }}>
        GPU {gpu.index} · {gpu.name.replace('NVIDIA ', '')}
      </div>
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={size / 2} cy={size / 2} r={r} stroke="#161b27" strokeWidth={stroke} fill="none" />
          <circle
            cx={size / 2} cy={size / 2} r={r}
            stroke={color} strokeWidth={stroke} fill="none"
            strokeDasharray={c}
            strokeDashoffset={c - (c * util) / 100}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color})`, transition: 'stroke-dashoffset 500ms ease' }}
          />
        </svg>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ fontSize: 36, fontWeight: 700, color: '#d4d8e4', fontFamily: 'Outfit' }}>
            {util}<span style={{ fontSize: 16, color: '#7a8199' }}>%</span>
          </div>
          <div style={{ fontSize: 10, color: '#7a8199', letterSpacing: 1 }}>UTILIZATION</div>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, width: '100%' }}>
        <Stat label="TEMP" value={`${gpu.temp}°C`} color={color} />
        <Stat label="POWER" value={`${gpu.power}W`} />
        <Stat label="MEM" value={`${(gpu.mem_used_mb / 1024).toFixed(1)}/${(gpu.mem_total_mb / 1024).toFixed(1)}G`} />
      </div>
      <div style={{ width: '100%', height: 6, background: '#141820', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${memPct}%`, height: '100%',
          background: `linear-gradient(90deg, #4f8eff, ${color})`,
          boxShadow: `0 0 8px ${color}80`, transition: 'width 500ms ease',
        }} />
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: '#3d4455', letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 13, color: color ?? '#d4d8e4', fontWeight: 600, fontFamily: 'JetBrains Mono' }}>{value}</div>
    </div>
  )
}

// ── CPU as a radial spiral of cores ───────────────────────────────
function CpuSpiral({ cores, util }: { cores: number; util: number }) {
  const size = 220
  const cx = size / 2
  const cy = size / 2
  const n = Math.max(1, cores)
  const maxR = size / 2 - 18

  const dots = Array.from({ length: n }, (_, i) => {
    const t = i / n
    const angle = t * Math.PI * 6 // 3 full turns
    const radius = 14 + t * (maxR - 14)
    const x = cx + Math.cos(angle) * radius
    const y = cy + Math.sin(angle) * radius
    // Highlight roughly (util%) of cores as "active"
    const activeThreshold = util / 100
    const active = t <= activeThreshold
    return { x, y, active, i }
  })

  return (
    <div style={{
      padding: 20,
      background: 'linear-gradient(145deg,#0a0d14,#0f1219)',
      border: '1px solid #1a2030', borderRadius: 18,
      boxShadow: '0 0 30px #4f8eff12, inset 0 0 20px #0006',
    }}>
      <div style={{ fontSize: 11, color: '#7a8199', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 12 }}>
        CPU · {cores} cores
      </div>
      <div style={{ position: 'relative', width: size, height: size, margin: '0 auto' }}>
        <svg width={size} height={size}>
          {dots.map(d => (
            <circle
              key={d.i}
              cx={d.x} cy={d.y}
              r={d.active ? 4 : 2.5}
              fill={d.active ? '#4f8eff' : '#2a3042'}
              style={{ filter: d.active ? 'drop-shadow(0 0 4px #4f8eff)' : 'none', transition: 'all 400ms ease' }}
            />
          ))}
        </svg>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
        }}>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#d4d8e4', fontFamily: 'Outfit' }}>
            {util.toFixed(0)}<span style={{ fontSize: 14, color: '#7a8199' }}>%</span>
          </div>
          <div style={{ fontSize: 10, color: '#7a8199', letterSpacing: 1 }}>LOAD</div>
        </div>
      </div>
    </div>
  )
}

// ── RAM bar ───────────────────────────────────────────────────────
function RamBar({ total, used }: { total: number; used: number }) {
  const pct = total > 0 ? (used / total) * 100 : 0
  return (
    <div style={{
      padding: 20,
      background: 'linear-gradient(145deg,#0a0d14,#0f1219)',
      border: '1px solid #1a2030', borderRadius: 18,
      boxShadow: '0 0 30px #7c5fcf12, inset 0 0 20px #0006',
    }}>
      <div style={{ fontSize: 11, color: '#7a8199', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 12 }}>
        RAM · {total.toFixed(0)} GB
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
        <div style={{ fontSize: 42, fontWeight: 700, color: '#d4d8e4', fontFamily: 'Outfit' }}>
          {used.toFixed(1)}
        </div>
        <div style={{ fontSize: 16, color: '#7a8199' }}>
          / {total.toFixed(1)} GB ({pct.toFixed(0)}%)
        </div>
      </div>
      <div style={{ width: '100%', height: 10, background: '#141820', borderRadius: 5, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: 'linear-gradient(90deg,#7c5fcf,#d4508e)',
          boxShadow: '0 0 12px #7c5fcf80', transition: 'width 500ms ease',
        }} />
      </div>
      <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#3d4455' }}>
        <span>USED {used.toFixed(1)} GB</span>
        <span>FREE {(total - used).toFixed(1)} GB</span>
      </div>
    </div>
  )
}

export default function NeoBody() {
  const [hw, setHw] = useState<Hardware | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = () => {
      fetch('/api/neo/hardware')
        .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
        .then(d => { if (alive) { setHw(d); setErr(null) } })
        .catch(e => { if (alive) setErr(String(e)) })
    }
    load()
    const iv = setInterval(load, 2500)
    return () => { alive = false; clearInterval(iv) }
  }, [])

  if (err && !hw) {
    return <div style={{ padding: 40, color: '#ff4f6e' }}>Hardware unavailable: {err}</div>
  }
  if (!hw) {
    return <div style={{ padding: 40, color: '#7a8199' }}>Reading sensors…</div>
  }

  return (
    <div style={{ padding: 24, color: '#d4d8e4', fontFamily: 'JetBrains Mono' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontFamily: 'Outfit', fontWeight: 700, fontSize: 24, margin: 0, color: '#d4d8e4' }}>
          Body
        </h2>
        <div style={{ fontSize: 11, color: '#7a8199', marginTop: 4 }}>
          Source: {hw.source} · {hw.gpus.length} GPU(s) · {hw.cpu_cores} cores
        </div>
      </div>

      {hw.gpus.length > 0 ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
          {hw.gpus.map(g => <GpuRing key={g.index} gpu={g} />)}
        </div>
      ) : (
        <div style={{
          padding: 20, marginBottom: 20,
          background: '#0a0d14', border: '1px dashed #1a2030', borderRadius: 14,
          color: '#7a8199', fontSize: 12,
        }}>
          No GPUs detected (nvidia-smi not available).
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <CpuSpiral cores={hw.cpu_cores} util={hw.cpu_util} />
        <RamBar total={hw.ram_gb} used={hw.ram_used_gb} />
      </div>
    </div>
  )
}
