import { useState, useEffect, useRef } from 'react'
import {
  Mail,
  Users,
  Activity,
  ArrowRight,
  Radio,
  LayoutGrid,
  List,
} from 'lucide-react'
import GraphCanvas from './GraphCanvas'

interface BusStats {
  total_messages: number
  delivered: number
  failed: number
  active_agents: number
  active_conversations: number
}

interface Conversation {
  key: string
  participants: string[]
}

interface MessageEvent {
  id: number
  from: string
  to: string
  intent: string
  graph_name: string
  timestamp: number
}

const agentColor: Record<string, string> = {
  ceo: 'var(--accent-info)',
  researcher: 'var(--accent-secondary)',
  developer: 'var(--accent-success)',
  guardian: 'var(--accent-warning)',
  strategist: 'var(--accent-entwicklung, #a78bfa)',
  artisan: 'var(--accent-anerkennung, #f472b6)',
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

function intentLabel(raw: string): string {
  if (raw.includes('Execute')) return 'Execute'
  if (raw.includes('Result')) return 'Result'
  if (raw.includes('Compress')) return 'Compress'
  if (raw.includes('Optimize')) return 'Optimize'
  if (raw.includes('Train')) return 'Train'
  if (raw.includes('Verify')) return 'Verify'
  if (raw.includes('Compose')) return 'Compose'
  return raw.slice(0, 20)
}

function StatCard({ label, value, color }: { label: string; value: number | string; color?: string }) {
  return (
    <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
      <div style={{ fontSize: '24px', fontWeight: 700, color: color ?? 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
        {value}
      </div>
      <div className="label" style={{ marginTop: '4px' }}>{label}</div>
    </div>
  )
}

interface ProofResult {
  success: boolean
  description: string
  similarity: number
  total_us: number
  total_bytes_transferred: number
  messages_sent: number
  steps: { step: number; agent: string; action: string; duration_us: number; data_bytes: number; detail: string }[]
}

export default function MessagesView() {
  const [stats, setStats] = useState<BusStats | null>(null)
  const [agents, setAgents] = useState<string[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [messages, setMessages] = useState<MessageEvent[]>([])
  const [streaming, setStreaming] = useState(false)
  const [proof, setProof] = useState<ProofResult | null>(null)
  const [proofRunning, setProofRunning] = useState(false)
  const [textA, setTextA] = useState('Rust ist eine Systemprogrammiersprache')
  const [textB, setTextB] = useState('Python ist eine Skriptsprache')
  const [viewMode, setViewMode] = useState<'list' | 'canvas'>('canvas')
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const esRef = useRef<EventSource | null>(null)

  // Poll stats + agents + conversations
  useEffect(() => {
    const fetchData = () => {
      fetch('/api/messages/stats').then(r => r.json()).then(setStats).catch(() => {})
      fetch('/api/messages/agents').then(r => r.json()).then(d => setAgents(Array.isArray(d) ? d : [])).catch(() => {})
      fetch('/api/messages/conversations').then(r => r.json()).then(d => setConversations(Array.isArray(d) ? d : [])).catch(() => {})
    }
    fetchData()
    const interval = setInterval(fetchData, 3000)
    return () => clearInterval(interval)
  }, [])

  // SSE stream for live messages
  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    esRef.current = es
    setStreaming(true)

    es.onmessage = (event) => {
      try {
        const msg: MessageEvent = JSON.parse(event.data)
        setMessages(prev => {
          const next = [...prev, msg]
          return next.length > 200 ? next.slice(next.length - 200) : next
        })
      } catch {}
    }

    es.onerror = () => setStreaming(false)
    es.onopen = () => setStreaming(true)

    return () => { es.close(); setStreaming(false) }
  }, [])

  // Auto-scroll message log
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="view">
      {/* View toggle */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <button
          className={`btn ${viewMode === 'canvas' ? 'btn-primary' : ''}`}
          onClick={() => setViewMode('canvas')}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', padding: '6px 14px' }}
        >
          <LayoutGrid size={14} /> Canvas
        </button>
        <button
          className={`btn ${viewMode === 'list' ? 'btn-primary' : ''}`}
          onClick={() => setViewMode('list')}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', padding: '6px 14px' }}
        >
          <List size={14} /> Liste
        </button>
      </div>

      {viewMode === 'canvas' ? (
        <div style={{ flex: 1, minHeight: '500px' }}>
          <GraphCanvas />
        </div>
      ) : (
      <>
      {/* Stats row */}
      <div className="grid-4" style={{ marginBottom: '20px' }}>
        <StatCard label="Nachrichten" value={stats?.total_messages ?? 0} color="var(--accent-primary)" />
        <StatCard label="Zugestellt" value={stats?.delivered ?? 0} color="var(--accent-success)" />
        <StatCard label="Fehlgeschlagen" value={stats?.failed ?? 0} color="var(--accent-danger)" />
        <StatCard label="Konversationen" value={stats?.active_conversations ?? 0} color="var(--accent-info)" />
      </div>

      {/* Proof trigger */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Activity size={16} />
          <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>Tensor-Austausch Proof</h3>
        </div>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
          <input
            type="text"
            value={textA}
            onChange={e => setTextA(e.target.value)}
            placeholder="Text A (Researcher)"
            style={{
              flex: 1, minWidth: '200px', padding: '8px 12px', borderRadius: '8px',
              border: '1px solid var(--border)', background: 'var(--bg-primary)',
              color: 'var(--text-primary)', fontSize: '13px',
            }}
          />
          <input
            type="text"
            value={textB}
            onChange={e => setTextB(e.target.value)}
            placeholder="Text B (Developer)"
            style={{
              flex: 1, minWidth: '200px', padding: '8px 12px', borderRadius: '8px',
              border: '1px solid var(--border)', background: 'var(--bg-primary)',
              color: 'var(--text-primary)', fontSize: '13px',
            }}
          />
          <button
            className="btn btn-primary"
            disabled={proofRunning}
            onClick={async () => {
              setProofRunning(true)
              setProof(null)
              try {
                const res = await fetch('/api/proof/tensor-exchange', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ text_a: textA, text_b: textB }),
                })
                const data = await res.json()
                setProof(data)
              } catch (e) {
                setProof({ success: false, description: String(e), similarity: 0, total_us: 0, total_bytes_transferred: 0, messages_sent: 0, steps: [] })
              }
              setProofRunning(false)
            }}
            style={{ whiteSpace: 'nowrap' }}
          >
            {proofRunning ? 'Laeuft...' : 'Tensor-Proof starten'}
          </button>
        </div>
        {proof && (
          <div style={{ fontSize: '12px', lineHeight: 1.8 }}>
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '8px' }}>
              <span style={{ fontWeight: 700, color: proof.success ? 'var(--accent-success)' : 'var(--accent-danger)' }}>
                {proof.success ? 'Erfolgreich' : 'Fehlgeschlagen'}
              </span>
              <span>Similarity: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>{proof.similarity.toFixed(4)}</strong></span>
              <span>{(proof.total_us / 1000).toFixed(1)}ms</span>
              <span>{proof.total_bytes_transferred} Bytes</span>
              <span>{proof.messages_sent} QLMS Messages</span>
              <span style={{ color: 'var(--accent-success)', fontWeight: 600 }}>0 LLM Calls</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {proof.steps.map(s => (
                <div key={s.step} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="mono" style={{ color: 'var(--text-muted)', width: '18px', textAlign: 'right' }}>{s.step}.</span>
                  <span style={{ color: agentColor[s.agent] ?? 'var(--accent-primary)', fontWeight: 600, width: '80px' }}>{s.agent}</span>
                  <span className="badge" style={{ fontSize: '10px', padding: '1px 6px' }}>{s.action}</span>
                  <span className="mono" style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                    {s.duration_us > 1000 ? `${(s.duration_us / 1000).toFixed(1)}ms` : `${s.duration_us}us`}
                  </span>
                  <span style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.detail}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Agents on bus */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Users size={16} />
          <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>Registrierte Agenten ({agents.length})</h3>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Radio size={12} style={{ color: streaming ? 'var(--accent-success)' : 'var(--accent-danger)' }} />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {streaming ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {agents.map(name => (
            <span
              key={name}
              className="badge"
              style={{
                background: `color-mix(in srgb, ${agentColor[name] ?? 'var(--accent-primary)'} 15%, transparent)`,
                color: agentColor[name] ?? 'var(--accent-primary)',
                padding: '4px 10px',
                borderRadius: '12px',
                fontSize: '12px',
                fontWeight: 600,
              }}
            >
              {name}
            </span>
          ))}
          {agents.length === 0 && (
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Keine Agenten registriert</span>
          )}
        </div>
      </div>

      {/* Conversations */}
      {conversations.length > 0 && (
        <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <Activity size={16} />
            <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>Aktive Konversationen</h3>
          </div>
          {conversations.map(conv => (
            <div key={conv.key} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 0', fontSize: '13px' }}>
              <span style={{ color: agentColor[conv.participants[0]] ?? 'var(--accent-primary)', fontWeight: 600 }}>
                {conv.participants[0]}
              </span>
              <ArrowRight size={14} style={{ color: 'var(--text-muted)' }} />
              <span style={{ color: agentColor[conv.participants[1]] ?? 'var(--accent-secondary)', fontWeight: 600 }}>
                {conv.participants[1]}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Live message log */}
      <div className="card" style={{ padding: '0', flex: 1 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
        }}>
          <Mail size={16} />
          <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>QLMS Nachrichtenlog</h3>
          <span className="mono" style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>
            {messages.length} Nachrichten
          </span>
        </div>
        <div style={{ maxHeight: '400px', overflowY: 'auto', padding: '4px 0' }}>
          {messages.length === 0 ? (
            <div style={{ padding: '24px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
              Warte auf QLMS-Nachrichten... Erstelle ein Ziel im Ziele-Tab um Agent-Kommunikation zu sehen.
            </div>
          ) : (
            messages.map((msg, i) => (
              <div
                key={`${msg.id}-${i}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '3px 16px',
                  fontSize: '12px',
                  lineHeight: 1.6,
                }}
              >
                <span className="mono" style={{ color: 'var(--text-muted)', flexShrink: 0, width: '60px' }}>
                  {formatTime(msg.timestamp)}
                </span>
                <span style={{
                  color: agentColor[msg.from] ?? 'var(--accent-primary)',
                  fontWeight: 600,
                  flexShrink: 0,
                  width: '80px',
                  textAlign: 'right',
                }}>
                  {msg.from}
                </span>
                <ArrowRight size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                <span style={{
                  color: agentColor[msg.to] ?? 'var(--accent-secondary)',
                  fontWeight: 600,
                  flexShrink: 0,
                  width: '80px',
                }}>
                  {msg.to}
                </span>
                <span className="badge" style={{
                  fontSize: '10px',
                  padding: '1px 6px',
                  background: msg.intent.includes('Result')
                    ? 'color-mix(in srgb, var(--accent-success) 15%, transparent)'
                    : 'color-mix(in srgb, var(--accent-primary) 15%, transparent)',
                  color: msg.intent.includes('Result') ? 'var(--accent-success)' : 'var(--accent-primary)',
                  flexShrink: 0,
                }}>
                  {intentLabel(msg.intent)}
                </span>
                <span className="mono" style={{ color: 'var(--text-muted)', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {msg.graph_name}
                </span>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <style>{`
        .grid-4 {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
        }
        @media (max-width: 768px) {
          .grid-4 { grid-template-columns: repeat(2, 1fr); }
        }
      `}</style>
      </>
      )}
    </div>
  )
}
