import { useEffect, useState } from 'react'

interface MemEntry {
  source: string
  key: string
  preview: string
}

interface MemResp {
  hdc_count: number
  organism_count: number
  entries: MemEntry[]
}

function Card({ entry, idx }: { entry: MemEntry; idx: number }) {
  const isHdc = entry.source === 'hdc'
  const color = isHdc ? '#4f8eff' : '#d4508e'
  // Pseudo-3D tilt: alternate rows tilt differently, cards slightly offset in Y.
  const tilt = (idx % 2 === 0 ? 1 : -1) * 2
  const lift = (idx % 3) * 2

  return (
    <div
      style={{
        position: 'relative',
        padding: 16,
        background: `linear-gradient(145deg, #0a0d14, #0f1219)`,
        border: `1px solid ${color}33`,
        borderRadius: 14,
        boxShadow: `0 8px 24px #0008, 0 0 20px ${color}15, inset 0 1px 0 ${color}22`,
        transform: `perspective(900px) rotateX(${tilt}deg) translateY(${-lift}px)`,
        transformStyle: 'preserve-3d',
        transition: 'transform 300ms ease, box-shadow 300ms ease',
        cursor: 'default',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = `perspective(900px) rotateX(0deg) translateY(-6px) scale(1.02)`
        e.currentTarget.style.boxShadow = `0 16px 32px #000a, 0 0 32px ${color}40`
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = `perspective(900px) rotateX(${tilt}deg) translateY(${-lift}px)`
        e.currentTarget.style.boxShadow = `0 8px 24px #0008, 0 0 20px ${color}15, inset 0 1px 0 ${color}22`
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <span style={{
          fontSize: 9, letterSpacing: 1.5, textTransform: 'uppercase',
          color, fontWeight: 700, fontFamily: 'Outfit',
          padding: '3px 8px', background: `${color}15`,
          border: `1px solid ${color}40`, borderRadius: 3,
        }}>
          {entry.source}
        </span>
        <span style={{ fontSize: 9, color: '#3d4455' }}>#{idx.toString().padStart(3, '0')}</span>
      </div>
      <div style={{
        fontSize: 12, color: '#d4d8e4',
        fontFamily: 'JetBrains Mono', lineHeight: 1.5,
        wordBreak: 'break-word', whiteSpace: 'pre-wrap',
        maxHeight: 96, overflow: 'hidden',
      }}>
        {entry.preview || <span style={{ color: '#3d4455' }}>(empty)</span>}
      </div>
    </div>
  )
}

export default function NeoMemory() {
  const [data, setData] = useState<MemResp | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = () => {
      fetch('/api/neo/memory')
        .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
        .then(d => { if (alive) { setData(d); setErr(null) } })
        .catch(e => { if (alive) setErr(String(e)) })
    }
    load()
    const iv = setInterval(load, 5000)
    return () => { alive = false; clearInterval(iv) }
  }, [])

  if (err && !data) {
    return <div style={{ padding: 40, color: '#ff4f6e' }}>Memory unavailable: {err}</div>
  }
  if (!data) {
    return <div style={{ padding: 40, color: '#7a8199' }}>Loading memories…</div>
  }

  return (
    <div style={{ padding: 24, color: '#d4d8e4', fontFamily: 'JetBrains Mono' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontFamily: 'Outfit', fontWeight: 700, fontSize: 24, margin: 0 }}>
          Memory
        </h2>
        <div style={{ fontSize: 11, color: '#7a8199', marginTop: 4 }}>
          HDC vector store: <span style={{ color: '#4f8eff' }}>{data.hdc_count}</span> ·{' '}
          Organism shared: <span style={{ color: '#d4508e' }}>{data.organism_count}</span> ·{' '}
          showing {data.entries.length}
        </div>
      </div>

      {data.entries.length === 0 ? (
        <div style={{
          padding: 40, textAlign: 'center',
          background: '#0a0d14', border: '1px dashed #1a2030', borderRadius: 14,
          color: '#7a8199',
        }}>
          No memory entries yet. Chat with the organism or store an embedding.
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
          gap: 16,
          perspective: 1200,
        }}>
          {data.entries.map((e, i) => <Card key={i} entry={e} idx={i} />)}
        </div>
      )}
    </div>
  )
}
