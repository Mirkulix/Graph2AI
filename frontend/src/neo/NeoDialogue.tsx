import { useEffect, useRef, useState } from 'react'

interface OrganismResponse {
  text: string
  specialist: string
  confidence: number
  reasoning: string[]
  memory_stored: boolean
  total_interactions: number
  generation: number
  memory_items: number
}

interface Turn {
  role: 'user' | 'organism'
  text: string
  meta?: {
    specialist?: string
    confidence?: number
    latencyMs?: number
    reasoning?: string[]
    memory_stored?: boolean
  }
}

export default function NeoDialogue() {
  const [input, setInput] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns])

  async function send() {
    const msg = input.trim()
    if (!msg || busy) return
    setTurns(t => [...t, { role: 'user', text: msg }])
    setInput('')
    setBusy(true)
    setErr(null)
    const t0 = performance.now()
    try {
      const r = await fetch('/api/organism/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data: OrganismResponse = await r.json()
      const latencyMs = Math.round(performance.now() - t0)
      const cleanText = (data.text ?? '').trim()
      setTurns(t => [...t, {
        role: 'organism',
        text: cleanText.length === 0 ? '(no response — organism returned empty)' : cleanText,
        meta: {
          specialist: data.specialist,
          confidence: data.confidence,
          latencyMs,
          reasoning: data.reasoning,
          memory_stored: data.memory_stored,
        },
      }])
    } catch (e) {
      setErr(String(e))
      setTurns(t => [...t, { role: 'organism', text: `(error: ${String(e)})` }])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  function onKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div style={{
      padding: 24, color: '#d4d8e4', fontFamily: 'JetBrains Mono',
      display: 'flex', flexDirection: 'column', height: '100%',
    }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontFamily: 'Outfit', fontWeight: 700, fontSize: 24, margin: 0 }}>
          Dialogue
        </h2>
        <div style={{ fontSize: 11, color: '#7a8199', marginTop: 4 }}>
          Direct channel to the organism. Responses are real — unknown tokens are shown honestly.
        </div>
      </div>

      <div ref={logRef} style={{
        flex: 1, overflowY: 'auto',
        background: '#04060b',
        border: '1px solid #161b27', borderRadius: 14,
        padding: 20, marginBottom: 12,
      }}>
        {turns.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: '#7a8199' }}>
            Say something to the organism…
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} style={{
            display: 'flex',
            justifyContent: t.role === 'user' ? 'flex-end' : 'flex-start',
            margin: '10px 0',
          }}>
            <div style={{
              maxWidth: '78%',
              padding: '12px 16px',
              borderRadius: 14,
              background: t.role === 'user'
                ? 'linear-gradient(135deg,#1a2340,#15182a)'
                : 'linear-gradient(135deg,#0f2820,#0c1a24)',
              border: t.role === 'user'
                ? '1px solid #4f8eff33'
                : '1px solid #2dd4a033',
              boxShadow: t.role === 'user'
                ? '0 4px 16px #0007, 0 0 16px #4f8eff15'
                : '0 4px 16px #0007, 0 0 16px #2dd4a015',
            }}>
              <div style={{
                fontSize: 9, letterSpacing: 1.5, textTransform: 'uppercase',
                color: t.role === 'user' ? '#4f8eff' : '#2dd4a0',
                marginBottom: 6, fontFamily: 'Outfit', fontWeight: 700,
              }}>
                {t.role === 'user' ? 'You' : (t.meta?.specialist ?? 'Organism')}
              </div>
              <div style={{
                fontSize: 13, lineHeight: 1.6,
                whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                color: '#d4d8e4',
              }}>
                {t.text}
              </div>
              {t.meta && (
                <div style={{
                  marginTop: 10, paddingTop: 8,
                  borderTop: '1px solid #1a2030',
                  display: 'flex', flexWrap: 'wrap', gap: 10,
                  fontSize: 10, color: '#7a8199',
                }}>
                  {typeof t.meta.confidence === 'number' && (
                    <span>conf: <span style={{ color: '#2dd4a0' }}>{(t.meta.confidence * 100).toFixed(0)}%</span></span>
                  )}
                  {typeof t.meta.latencyMs === 'number' && (
                    <span>lat: <span style={{ color: '#4f8eff' }}>{t.meta.latencyMs}ms</span></span>
                  )}
                  {t.meta.memory_stored && (
                    <span style={{ color: '#d4508e' }}>· stored</span>
                  )}
                  {t.meta.reasoning && t.meta.reasoning.length > 0 && (
                    <span style={{ width: '100%', marginTop: 4, color: '#3d4455', fontStyle: 'italic' }}>
                      ↪ {t.meta.reasoning.slice(0, 3).join(' · ')}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {err && (
        <div style={{ fontSize: 11, color: '#ff4f6e', marginBottom: 8 }}>
          {err}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder={busy ? 'Organism is thinking…' : 'Message the organism…'}
          disabled={busy}
          style={{
            flex: 1, padding: '12px 16px',
            background: '#0a0d14',
            border: '1px solid #1a2030',
            borderRadius: 10, color: '#d4d8e4',
            fontFamily: 'JetBrains Mono', fontSize: 13,
            outline: 'none',
          }}
          onFocus={e => { e.currentTarget.style.borderColor = '#4f8eff'; e.currentTarget.style.boxShadow = '0 0 16px #4f8eff30' }}
          onBlur={e => { e.currentTarget.style.borderColor = '#1a2030'; e.currentTarget.style.boxShadow = 'none' }}
        />
        <button
          onClick={send}
          disabled={busy || !input.trim()}
          style={{
            padding: '12px 20px',
            background: busy ? '#1a2030' : 'linear-gradient(135deg,#4f8eff,#7c5fcf)',
            border: 'none', borderRadius: 10,
            color: '#fff', fontFamily: 'Outfit', fontWeight: 600,
            fontSize: 13, letterSpacing: 1, textTransform: 'uppercase',
            cursor: busy ? 'not-allowed' : 'pointer',
            boxShadow: busy ? 'none' : '0 4px 16px #4f8eff40',
            transition: 'transform 120ms ease',
          }}
        >
          {busy ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
