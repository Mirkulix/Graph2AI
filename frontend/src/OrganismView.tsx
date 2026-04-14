import { useState, useEffect, useRef } from 'react'
import { Send, Brain, Cpu, Database, Zap, ChevronRight } from 'lucide-react'

interface SpecialistInfo { name: string; invocations: number; success_rate: number }
interface ChatMessage { role: 'user' | 'organism'; text: string; specialist?: string; reasoning?: string[]; confidence?: number }
interface OrganismStatus { generation: number; specialists: SpecialistInfo[]; total_interactions: number; memory_items: number }

const specialistColor: Record<string, string> = {
  responder: '#3b82f6', memory: '#8b5cf6', adapter: '#22c55e',
  digit_classifier: '#f59e0b', organism: '#6b7280',
}

export default function OrganismView() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState<OrganismStatus | null>(null)
  const [sending, setSending] = useState(false)
  const [showReasoning, setShowReasoning] = useState<number | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const poll = () => { fetch('/api/organism/status').then(r => r.json()).then(setStatus).catch(() => {}) }
    poll()
    const i = setInterval(poll, 5000)
    return () => clearInterval(i)
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async () => {
    if (!input.trim() || sending) return
    const userMsg = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: userMsg }])
    setSending(true)

    try {
      const res = await fetch('/api/organism/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'organism', text: data.text, specialist: data.specialist,
        reasoning: data.reasoning, confidence: data.confidence,
      }])
      setStatus(prev => prev ? {
        ...prev, total_interactions: data.total_interactions,
        memory_items: data.memory_items, specialists: data.specialists,
      } : null)
    } catch { setMessages(prev => [...prev, { role: 'organism', text: 'Error connecting to organism.' }]) }
    setSending(false)
  }

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '12px' }}>
      {/* Status bar */}
      {status && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <div className="card" style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
            <Brain size={14} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: '12px' }}>Gen <strong>{status.generation}</strong></span>
          </div>
          <div className="card" style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
            <Cpu size={14} style={{ color: '#22c55e' }} />
            <span style={{ fontSize: '12px' }}><strong>{status.specialists.length}</strong> Specialists</span>
          </div>
          <div className="card" style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
            <Database size={14} style={{ color: '#8b5cf6' }} />
            <span style={{ fontSize: '12px' }}><strong>{status.memory_items}</strong> Memories</span>
          </div>
          <div className="card" style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
            <Zap size={14} style={{ color: '#f59e0b' }} />
            <span style={{ fontSize: '12px' }}><strong>{status.total_interactions}</strong> Interactions</span>
          </div>
        </div>
      )}

      {/* Specialists */}
      {status && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {status.specialists.map(s => (
            <span key={s.name} style={{
              fontSize: '11px', padding: '3px 10px', borderRadius: '12px', fontWeight: 600,
              background: `color-mix(in srgb, ${specialistColor[s.name] ?? '#6b7280'} 12%, transparent)`,
              color: specialistColor[s.name] ?? '#6b7280',
            }}>
              {s.name} ({s.invocations})
            </span>
          ))}
        </div>
      )}

      {/* Chat */}
      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px', padding: '40px 20px' }}>
              Der QLANG Organismus besteht aus spezialisierten ternaeren Modellen.<br />
              Schreibe eine Nachricht um mit ihm zu interagieren.
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} style={{ marginBottom: '10px', display: 'flex', flexDirection: 'column',
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
              <div style={{
                maxWidth: '80%', padding: '8px 14px', borderRadius: '12px', fontSize: '13px', lineHeight: 1.5,
                background: msg.role === 'user' ? 'var(--accent-primary)' : 'var(--bg-elevated)',
                color: msg.role === 'user' ? '#fff' : 'var(--text-primary)',
              }}>
                {msg.text}
              </div>
              {msg.specialist && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '3px', fontSize: '11px' }}>
                  <span style={{ color: specialistColor[msg.specialist] ?? 'var(--text-muted)', fontWeight: 600 }}>
                    {msg.specialist}
                  </span>
                  {msg.confidence !== undefined && (
                    <span style={{ color: 'var(--text-muted)' }}>{(msg.confidence * 100).toFixed(0)}%</span>
                  )}
                  {msg.reasoning && msg.reasoning.length > 0 && (
                    <button onClick={() => setShowReasoning(showReasoning === i ? null : i)}
                      style={{ background: 'none', border: 'none', color: 'var(--accent-primary)',
                        cursor: 'pointer', fontSize: '11px', padding: 0 }}>
                      {showReasoning === i ? 'hide' : 'reasoning'}
                    </button>
                  )}
                </div>
              )}
              {showReasoning === i && msg.reasoning && (
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', padding: '6px 10px',
                  background: 'var(--bg-primary)', borderRadius: '6px', maxWidth: '80%' }}>
                  {msg.reasoning.map((r, j) => (
                    <div key={j} style={{ display: 'flex', gap: '4px', alignItems: 'baseline' }}>
                      <ChevronRight size={10} style={{ flexShrink: 0 }} /> {r}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ borderTop: '1px solid var(--border)', padding: '10px 12px', display: 'flex', gap: '8px' }}>
          <input type="text" value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="Nachricht an den Organismus..."
            style={{ flex: 1, padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border)',
              background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          <button className="btn btn-primary" onClick={send} disabled={sending}
            style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Send size={14} /> {sending ? '...' : 'Senden'}
          </button>
        </div>
      </div>
    </div>
  )
}
