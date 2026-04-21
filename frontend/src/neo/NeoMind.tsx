import React, { useEffect, useMemo, useRef, useState } from 'react'

interface NeoStatusSnapshot {
  server: string
  hdc_memory: number
  organism_generation: number
  organism_interactions: number
  organism_memory_items: number
  specialists: number
  gpu_count: number
  gpu_temps: number[]
  gpu_utils: number[]
}

interface BusMessage {
  id: number
  from: string
  to: string
  intent: string
  graph_name: string
  timestamp: number
}

export const NeoMind: React.FC = () => {
  const [status, setStatus] = useState<NeoStatusSnapshot | null>(null)
  const [messages, setMessages] = useState<BusMessage[]>([])
  const [statusError, setStatusError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let cancelled = false
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/neo/status')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: NeoStatusSnapshot = await res.json()
        if (!cancelled) {
          setStatus(data)
          setStatusError(null)
        }
      } catch (err) {
        if (!cancelled) setStatusError((err as Error).message)
      }
    }

    void fetchStatus()
    const id = window.setInterval(fetchStatus, 1000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    es.onmessage = (ev) => {
      try {
        const msg: BusMessage = JSON.parse(ev.data)
        setMessages((prev) => [...prev, msg].slice(-20))
      } catch {
        // ignore malformed bus payloads
      }
    }
    return () => es.close()
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const memoryTotal = useMemo(() => {
    if (!status) return 0
    return (status.hdc_memory ?? 0) + (status.organism_memory_items ?? 0)
  }, [status])

  return (
    <div style={{
      background: '#0a0e27',
      color: '#d0e4ff',
      fontFamily: "'JetBrains Mono', monospace",
      minHeight: 720,
      padding: 16,
      borderRadius: 12,
    }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0, color: '#00e5ff', letterSpacing: 2 }}>NEO MIND</h2>
          <div style={{ fontSize: 11, color: '#6a7aa8', marginTop: 4 }}>
            live neo topology - aggregate snapshot via <code>/api/neo/status</code> and QLMS bus
          </div>
        </div>
        {statusError ? (
          <div style={{ color: '#ff6b6b', fontSize: 12 }}>api error: {statusError}</div>
        ) : null}
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 12, marginBottom: 16 }}>
        <MetricCard label='Generation' value={String(status?.organism_generation ?? 0)} accent='#ff2bd6' />
        <MetricCard label='Specialists' value={String(status?.specialists ?? 0)} accent='#64ffda' />
        <MetricCard label='Interactions' value={String(status?.organism_interactions ?? 0)} accent='#ffd54a' />
        <MetricCard label='Memory' value={String(memoryTotal)} accent='#b388ff' />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 16 }}>
        <section style={{
          border: '1px solid #1a2347',
          borderRadius: 10,
          background: 'radial-gradient(ellipse at center, #0d1436 0%, #050714 100%)',
          padding: 16,
          minHeight: 520,
        }}>
          <div style={{ fontSize: 11, letterSpacing: 2, color: '#6a7aa8', marginBottom: 12 }}>SYSTEM SNAPSHOT</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
            <SnapshotRow label='Server' value={status?.server ?? 'offline'} />
            <SnapshotRow label='GPU count' value={String(status?.gpu_count ?? 0)} />
            <SnapshotRow label='GPU utils' value={(status?.gpu_utils ?? []).join(', ') || '-'} />
            <SnapshotRow label='GPU temps' value={(status?.gpu_temps ?? []).join(', ') || '-'} />
            <SnapshotRow label='HDC memory' value={String(status?.hdc_memory ?? 0)} />
            <SnapshotRow label='Legacy organism memory' value={String(status?.organism_memory_items ?? 0)} />
          </div>

          <div style={{ marginTop: 18, padding: 16, border: '1px dashed #26345d', borderRadius: 10, color: '#6a7aa8', fontSize: 12, lineHeight: 1.6 }}>
            The previous evolution and organism endpoints are no longer active. This view now reflects the surviving runtime state instead of polling removed APIs.
          </div>
        </section>

        <aside style={{
          border: '1px solid #1a2347',
          borderRadius: 10,
          background: '#0d1436',
          padding: 16,
          minHeight: 520,
        }}>
          <div style={{ fontSize: 11, letterSpacing: 2, color: '#00e5ff', marginBottom: 12 }}>RECENT BUS MESSAGES</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 460, overflowY: 'auto' }}>
            {messages.length === 0 ? (
              <div style={{ color: '#6a7aa8', fontSize: 12 }}>No QLMS bus traffic yet.</div>
            ) : messages.map((msg) => (
              <div key={msg.id} style={{ borderBottom: '1px solid #1a2347', paddingBottom: 8 }}>
                <div style={{ color: '#ffd54a', fontSize: 11 }}>#{msg.id} · {msg.intent}</div>
                <div style={{ color: '#d0e4ff', fontSize: 12 }}>{msg.from} -&gt; {msg.to}</div>
                <div style={{ color: '#6a7aa8', fontSize: 11 }}>{msg.graph_name}</div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        </aside>
      </div>
    </div>
  )
}

function MetricCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div style={{ border: '1px solid #1a2347', borderRadius: 10, background: '#0d1436', padding: 14 }}>
      <div style={{ color: '#6a7aa8', fontSize: 10, letterSpacing: 2 }}>{label}</div>
      <div style={{ color: accent, fontSize: 28, fontWeight: 700, marginTop: 6 }}>{value}</div>
    </div>
  )
}

function SnapshotRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ padding: 12, border: '1px solid #1a2347', borderRadius: 8, background: '#0b1028' }}>
      <div style={{ color: '#6a7aa8', fontSize: 10, letterSpacing: 1.5 }}>{label}</div>
      <div style={{ color: '#d0e4ff', fontSize: 14, marginTop: 6, wordBreak: 'break-word' }}>{value}</div>
    </div>
  )
}

export default NeoMind
