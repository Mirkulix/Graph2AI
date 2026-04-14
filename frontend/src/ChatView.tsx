import { useState, useEffect, useRef } from 'react'
import { Send, RefreshCw, Download } from 'lucide-react'

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tier?: string
  timestamp?: number
  error?: boolean
  originalText?: string
}

interface ChatEntry {
  id?: string
  role: string
  content: string
  tier?: string
  timestamp?: number
}

interface ChatResponse {
  response: string
  tier?: string
}

function LoadingDots() {
  return (
    <div className="chat-bubble chat-bubble-assistant">
      <span className="loading-dot" style={{ animationDelay: '0ms' }} />
      <span className="loading-dot" style={{ animationDelay: '200ms' }} />
      <span className="loading-dot" style={{ animationDelay: '400ms' }} />
    </div>
  )
}

function formatRelativeTime(ts: number): string {
  const nowSec = Math.floor(Date.now() / 1000)
  const diff = nowSec - ts
  if (diff < 60) return 'gerade eben'
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`
  return `vor ${Math.floor(diff / 86400)} Tagen`
}

export default function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [healthError, setHealthError] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    fetch('/api/chat/history')
      .then(r => r.json())
      .then((history: any[]) => {
        if (!Array.isArray(history)) return
        const loaded: ChatMessage[] = []
        for (const entry of history) {
          const userText = entry.user ?? entry.content
          const assistantText = entry.assistant ?? entry.response
          const ts: number | undefined = typeof entry.timestamp === 'number' ? entry.timestamp : undefined
          if (userText) {
            loaded.push({ id: `${entry.id ?? 0}-user`, role: 'user', content: userText, timestamp: ts })
          }
          if (assistantText) {
            loaded.push({ id: `${entry.id ?? 0}-assistant`, role: 'assistant', content: assistantText, tier: entry.tier, timestamp: ts })
          }
        }
        setMessages(loaded)
      })
      .catch(() => {})
  }, [])

  // Poll /api/health every 30 seconds
  useEffect(() => {
    const checkHealth = () => {
      fetch('/api/health')
        .then(r => { setHealthError(!r.ok) })
        .catch(() => setHealthError(true))
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (text?: string) => {
    const msgText = text ?? input.trim()
    if (!msgText || loading) return

    const nowSec = Math.floor(Date.now() / 1000)
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: msgText,
      timestamp: nowSec,
    }
    if (!text) setInput('')
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msgText }),
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data: ChatResponse = await res.json()
      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        tier: data.tier,
        timestamp: Math.floor(Date.now() / 1000),
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Fehler: Anfrage fehlgeschlagen. Bitte erneut versuchen.',
        error: true,
        originalText: msgText,
        timestamp: Math.floor(Date.now() / 1000),
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const retryMessage = (originalText: string, errorMsgId: string) => {
    setMessages(prev => prev.filter(m => m.id !== errorMsgId))
    sendMessage(originalText)
  }

  // Listen to global keyboard events dispatched by App
  useEffect(() => {
    const onFocus = () => inputRef.current?.focus()
    const onSend = () => { if (input.trim() && !loading) sendMessage() }
    const onEscape = () => inputRef.current?.blur()

    window.addEventListener('qo:focus-chat-input', onFocus)
    window.addEventListener('qo:send-chat', onSend)
    window.addEventListener('qo:escape', onEscape)
    return () => {
      window.removeEventListener('qo:focus-chat-input', onFocus)
      window.removeEventListener('qo:send-chat', onSend)
      window.removeEventListener('qo:escape', onEscape)
    }
  }, [input, loading])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
    // Shift+Enter creates a newline — default textarea behavior, no override needed
  }

  const handleExport = () => {
    const today = new Date().toISOString().slice(0, 10)
    const lines: string[] = [`# QO Chat Export — ${today}`, '']
    for (const msg of messages) {
      const time = msg.timestamp
        ? new Date(msg.timestamp * 1000).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })
        : '--:--'
      const role = msg.role === 'user' ? 'User' : `QO${msg.tier ? ` (${msg.tier})` : ''}`
      lines.push(`## ${time} — ${role}`)
      lines.push(msg.content)
      lines.push('')
    }
    downloadFile(`qo-chat-${today}.md`, lines.join('\n'), 'text/markdown')
  }

  return (
    <div className="chat-container">
      {healthError && (
        <div className="health-banner">
          Verbindung verloren — Server nicht erreichbar
        </div>
      )}

      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-title">Starte eine Konversation...</div>
            <div className="empty-hint">Schreibe eine Nachricht, um mit QO zu kommunizieren.</div>
          </div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`chat-row ${msg.role === 'user' ? 'chat-row-user' : 'chat-row-assistant'}`}
          >
            <div className={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-user' : msg.error ? 'chat-bubble-error' : 'chat-bubble-assistant'}`}>
              {msg.content}
            </div>
            <div className="chat-meta">
              {msg.role === 'assistant' && msg.tier && (
                <span className="chat-tier-badge">{msg.tier}</span>
              )}
              {msg.timestamp && (
                <span className="chat-timestamp">{formatRelativeTime(msg.timestamp)}</span>
              )}
              {msg.error && msg.originalText && (
                <button
                  className="chat-retry-btn"
                  onClick={() => retryMessage(msg.originalText!, msg.id)}
                  title="Erneut versuchen"
                >
                  <RefreshCw size={12} />
                  <span>Wiederholen</span>
                </button>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-row chat-row-assistant">
            <LoadingDots />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          className="input chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nachricht eingeben... (Enter zum Senden)"
          rows={1}
        />
        <button
          className="btn btn-primary chat-send-btn"
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          aria-label="Senden"
        >
          <Send size={18} />
        </button>
        {messages.length > 0 && (
          <button
            className="btn btn-ghost btn-icon"
            onClick={handleExport}
            title="Chat exportieren"
            aria-label="Chat als Markdown exportieren"
          >
            <Download size={18} />
          </button>
        )}
      </div>

      <style>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .health-banner {
          background: #7f1d1d;
          color: #fca5a5;
          text-align: center;
          padding: 8px 16px;
          font-size: 13px;
          font-weight: 500;
          flex-shrink: 0;
        }
        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .chat-row {
          display: flex;
          flex-direction: column;
          animation: fadeIn 300ms ease-out;
        }
        .chat-row-user {
          align-items: flex-end;
        }
        .chat-row-assistant {
          align-items: flex-start;
        }
        .chat-bubble {
          max-width: 70%;
          padding: 12px 16px;
          font-size: 14px;
          line-height: 1.6;
          word-break: break-word;
        }
        .chat-bubble-user {
          background: var(--accent-primary);
          color: #ffffff;
          border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
        }
        .chat-bubble-assistant {
          background: var(--bg-surface);
          color: var(--text-primary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .chat-bubble-error {
          background: #450a0a;
          color: #fca5a5;
          border: 1px solid #7f1d1d;
          border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
        }
        .chat-meta {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 4px;
          flex-wrap: wrap;
        }
        .chat-tier-badge {
          font-size: 10px;
          color: var(--text-muted);
          background: var(--bg-surface);
          border: 1px solid var(--border);
          padding: 1px 8px;
          border-radius: 12px;
        }
        .chat-timestamp {
          font-size: 11px;
          color: var(--text-muted);
        }
        .chat-retry-btn {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          font-size: 11px;
          color: #fca5a5;
          background: transparent;
          border: 1px solid #7f1d1d;
          border-radius: 8px;
          padding: 2px 8px;
          cursor: pointer;
        }
        .chat-retry-btn:hover {
          background: #7f1d1d;
        }
        .chat-input-area {
          padding: 16px 24px;
          border-top: 1px solid var(--border);
          background: var(--bg-surface);
          display: flex;
          gap: 12px;
          align-items: flex-end;
        }
        .chat-input {
          flex: 1;
          max-height: 120px;
          overflow-y: auto;
        }
        .chat-send-btn {
          flex-shrink: 0;
          min-width: 44px;
          padding: 8px 16px;
        }
        .loading-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent-primary);
          display: inline-block;
          animation: dotPulse 1.2s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
}
