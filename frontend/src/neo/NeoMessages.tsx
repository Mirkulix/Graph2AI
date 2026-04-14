import { useEffect, useRef, useState } from 'react'

interface QlmsEvent {
  id: number
  from: string
  to: string
  intent: string
  content?: string
  signed?: boolean
  verified?: boolean
  ts?: string
  [k: string]: unknown
}

function Bubble({ msg, idx }: { msg: QlmsEvent; idx: number }) {
  const side = idx % 2 === 0 ? 'left' : 'right'
  const align = side === 'right' ? 'flex-end' : 'flex-start'
  const bg = side === 'right'
    ? 'linear-gradient(135deg,#1a2340,#15182a)'
    : 'linear-gradient(135deg,#201429,#18152b)'
  const accent = side === 'right' ? '#4f8eff' : '#d4508e'
  const content = (msg.content ?? JSON.stringify(msg.intent)) as string

  return (
    <div style={{ display: 'flex', justifyContent: align, margin: '10px 0' }}>
      <div style={{
        maxWidth: '72%',
        padding: '10px 14px',
        background: bg,
        border: `1px solid ${accent}33`,
        borderRadius: 14,
        boxShadow: `0 4px 16px #0007, 0 0 16px ${accent}15`,
        color: '#d4d8e4',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6,
          fontSize: 10, color: '#7a8199', letterSpacing: 1,
          textTransform: 'uppercase', fontFamily: 'Outfit',
        }}>
          <span style={{ color: accent, fontWeight: 700 }}>{msg.from}</span>
          <span>→</span>
          <span>{msg.to}</span>
          <span style={{ marginLeft: 'auto' }}>{msg.intent}</span>
        </div>
        <div style={{
          fontSize: 12, fontFamily: 'JetBrains Mono', lineHeight: 1.5,
          wordBreak: 'break-word',
        }}>
          {content}
        </div>
        <div style={{
          marginTop: 8, display: 'flex', gap: 6,
          fontSize: 9, letterSpacing: 1,
        }}>
          {msg.signed && (
            <span style={{
              padding: '2px 6px', borderRadius: 3,
              background: '#2dd4a020', color: '#2dd4a0',
              border: '1px solid #2dd4a055',
            }}>SIGNED</span>
          )}
          {msg.verified && (
            <span style={{
              padding: '2px 6px', borderRadius: 3,
              background: '#4f8eff20', color: '#4f8eff',
              border: '1px solid #4f8eff55',
            }}>VERIFIED</span>
          )}
          {msg.ts && (
            <span style={{ color: '#3d4455', marginLeft: 'auto' }}>{msg.ts}</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default function NeoMessages() {
  const [msgs, setMsgs] = useState<QlmsEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    es.onopen = () => setConnected(true)
    es.onerror = () => { setConnected(false); setErr('SSE disconnected') }
    es.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data)
        setMsgs(prev => {
          const next = [...prev, { id: prev.length, ...m }]
          return next.slice(-200)
        })
      } catch {
        // ignore malformed
      }
    }
    return () => es.close()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs])

  return (
    <div style={{ padding: 24, color: '#d4d8e4', fontFamily: 'JetBrains Mono', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontFamily: 'Outfit', fontWeight: 700, fontSize: 24, margin: 0 }}>
          Messages
        </h2>
        <div style={{ fontSize: 11, color: '#7a8199', marginTop: 4, display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: connected ? '#2dd4a0' : '#ff4f6e',
            boxShadow: connected ? '0 0 8px #2dd4a0' : '0 0 8px #ff4f6e',
          }} />
          <span>QLMS stream · {msgs.length} message(s)</span>
          {err && !connected && <span style={{ color: '#ff4f6e' }}>· {err}</span>}
        </div>
      </div>

      <div style={{
        flex: 1, overflowY: 'auto',
        background: '#04060b',
        border: '1px solid #161b27', borderRadius: 14,
        padding: 20,
      }}>
        {msgs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 60, color: '#7a8199' }}>
            No QLMS traffic yet.
          </div>
        ) : (
          <>
            {msgs.map(m => <Bubble key={m.id} msg={m} idx={m.id} />)}
            <div ref={bottomRef} />
          </>
        )}
      </div>
    </div>
  )
}
