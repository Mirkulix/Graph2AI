// ChatView.tsx — polished chat surface on top of /api/chat.
//
// Features:
//   • Markdown rendering for assistant replies (bold, lists, headings,
//     code blocks, tables via remark-gfm).
//   • Distinct user vs assistant bubbles — users see their own text on
//     the right, branded accent colour.
//   • Tier badge on every assistant reply so the provider path is
//     transparent (groq / qlang-ollama / cloud / …).
//   • Image + text-file attachments (local preview + base64 upload).
//     Vision-capable models only — a clear notice appears if the
//     currently-routed tier cannot handle images.
//   • Retry button on failed responses, Markdown export for the whole
//     conversation.
//   • `pendingPrompt` hand-over from Home so typed prompts survive the
//     tab switch.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  Download,
  ImageIcon,
  Paperclip,
  RefreshCw,
  Send,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Attachment {
  id: string
  name: string
  kind: 'image' | 'text'
  size: number
  mime: string
  dataUrl: string // image: base64 data URL; text: data URL of text content
  previewText?: string // for text attachments — first ~200 chars
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tier?: string
  timestamp?: number
  error?: boolean
  originalText?: string
  attachments?: Attachment[]
}

interface ChatResponse {
  response: string
  tier?: string
}

interface ChatViewProps {
  pendingPrompt?: string | null
  onPendingConsumed?: () => void
}

// Limits kept conservative so dropping a 20 MB photo does not freeze
// the tab. Bigger inputs should go through /api/graphs or the QLMS path.
const MAX_IMAGE_BYTES = 4 * 1024 * 1024
const MAX_TEXT_BYTES = 200 * 1024

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatRelativeTime(ts: number): string {
  const nowSec = Math.floor(Date.now() / 1000)
  const diff = Math.max(0, nowSec - ts)
  if (diff < 60) return 'gerade eben'
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} min`
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} h`
  return `vor ${Math.floor(diff / 86400)} T`
}

function downloadMarkdown(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function fileIsImage(mime: string): boolean {
  return mime.startsWith('image/')
}

function fileIsText(mime: string): boolean {
  return (
    mime.startsWith('text/') ||
    mime === 'application/json' ||
    mime === 'application/xml' ||
    mime === ''
  )
}

async function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

async function readAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatView(props: ChatViewProps = {}) {
  const { pendingPrompt, onPendingConsumed } = props

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [loading, setLoading] = useState(false)
  const [healthError, setHealthError] = useState(false)
  const [visionWarning, setVisionWarning] = useState<string | null>(null)

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pendingFiredRef = useRef<boolean>(false)

  // ─── Load history + health ponger ──────────────────────────────────

  useEffect(() => {
    fetch('/api/chat/history')
      .then((r) => r.json())
      .then((history: unknown[]) => {
        if (!Array.isArray(history)) return
        const loaded: ChatMessage[] = []
        for (const raw of history) {
          const entry = raw as Record<string, unknown>
          const userText = (entry.user ?? entry.content) as string | undefined
          const assistantText = (entry.assistant ?? entry.response) as
            | string
            | undefined
          const ts = typeof entry.timestamp === 'number' ? (entry.timestamp as number) : undefined
          const id = (entry.id as number | undefined) ?? loaded.length
          if (userText) {
            loaded.push({ id: `${id}-user`, role: 'user', content: userText, timestamp: ts })
          }
          if (assistantText) {
            loaded.push({
              id: `${id}-assistant`,
              role: 'assistant',
              content: assistantText,
              tier: entry.tier as string | undefined,
              timestamp: ts,
            })
          }
        }
        setMessages(loaded)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    const check = () =>
      fetch('/api/health')
        .then((r) => setHealthError(!r.ok))
        .catch(() => setHealthError(true))
    void check()
    const h = window.setInterval(check, 30_000)
    return () => window.clearInterval(h)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // ─── Send ──────────────────────────────────────────────────────────

  const sendMessage = useCallback(
    async (text?: string, explicitAttachments?: Attachment[]) => {
      const msgText = (text ?? input).trim()
      const attached = explicitAttachments ?? attachments
      if ((!msgText && attached.length === 0) || loading) return

      const nowSec = Math.floor(Date.now() / 1000)
      const userMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: msgText,
        timestamp: nowSec,
        attachments: attached.length > 0 ? attached : undefined,
      }
      if (!text) setInput('')
      if (!explicitAttachments) setAttachments([])
      setMessages((prev) => [...prev, userMsg])
      setLoading(true)

      // When the prompt carries attachments, inline their text payload
      // into the outgoing message since the backend /api/chat schema is
      // still text-only. Images are noted as placeholders; a future
      // revision can POST them as base64 when a vision-capable tier
      // (OpenAI/Claude/Gemini) is configured.
      let payloadMessage = msgText
      if (attached.length > 0) {
        const extra: string[] = []
        for (const a of attached) {
          if (a.kind === 'text' && a.previewText) {
            extra.push(`\n\n[Anhang: ${a.name}]\n\`\`\`\n${a.previewText}\n\`\`\``)
          } else if (a.kind === 'image') {
            extra.push(`\n\n[Anhang: Bild ${a.name} (${Math.round(a.size / 1024)} KB) — vision-fähiges Modell nötig]`)
          }
        }
        payloadMessage = payloadMessage + extra.join('')
      }

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: payloadMessage }),
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = (await res.json()) as ChatResponse
        const assistantMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.response,
          tier: data.tier,
          timestamp: Math.floor(Date.now() / 1000),
        }
        setMessages((prev) => [...prev, assistantMsg])

        // Vision warning: if user attached images and the response came
        // from a tier we know cannot see them, surface a one-time hint.
        if (
          attached.some((a) => a.kind === 'image') &&
          data.tier &&
          !['openai', 'claude', 'anthropic', 'gemini'].includes(data.tier.toLowerCase())
        ) {
          setVisionWarning(
            `Aktueller Provider ("${data.tier}") ignoriert Bilder. Für Bildverarbeitung einen Vision-fähigen Provider (OpenAI / Claude / Gemini) konfigurieren.`,
          )
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: `Fehler: ${err instanceof Error ? err.message : String(err)}`,
            error: true,
            originalText: msgText,
            timestamp: Math.floor(Date.now() / 1000),
          },
        ])
      } finally {
        setLoading(false)
        inputRef.current?.focus()
      }
    },
    [attachments, input, loading],
  )

  // ─── Pending prompt from Home ──────────────────────────────────────

  useEffect(() => {
    if (!pendingPrompt || pendingFiredRef.current) return
    const text = pendingPrompt.trim()
    if (!text) return
    pendingFiredRef.current = true
    void sendMessage(text, [])
    onPendingConsumed?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPrompt])

  // ─── File attachments ──────────────────────────────────────────────

  const addFiles = useCallback(async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    const queue: Attachment[] = []
    for (const file of Array.from(fileList)) {
      if (fileIsImage(file.type)) {
        if (file.size > MAX_IMAGE_BYTES) {
          alert(`Bild ${file.name} ist > ${MAX_IMAGE_BYTES / 1_048_576} MB und wird übersprungen.`)
          continue
        }
        const dataUrl = await readAsDataUrl(file)
        queue.push({
          id: `${Date.now()}-${queue.length}`,
          name: file.name,
          kind: 'image',
          size: file.size,
          mime: file.type,
          dataUrl,
        })
      } else if (fileIsText(file.type) || file.name.match(/\.(md|txt|json|csv|tsv|xml|yml|yaml|ts|tsx|js|jsx|rs|py|go|java|c|cpp|h|hpp)$/i)) {
        if (file.size > MAX_TEXT_BYTES) {
          alert(`Text-Datei ${file.name} ist > ${MAX_TEXT_BYTES / 1024} KB und wird übersprungen.`)
          continue
        }
        const dataUrl = await readAsDataUrl(file)
        const preview = await readAsText(file)
        queue.push({
          id: `${Date.now()}-${queue.length}`,
          name: file.name,
          kind: 'text',
          size: file.size,
          mime: file.type || 'text/plain',
          dataUrl,
          previewText: preview.slice(0, 2_000),
        })
      } else {
        alert(`Dateityp ${file.type || file.name} wird nicht unterstützt.`)
      }
    }
    setAttachments((prev) => [...prev, ...queue])
  }, [])

  // ─── Retry + export ────────────────────────────────────────────────

  const retry = useCallback((originalText: string, errorMsgId: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== errorMsgId))
    void sendMessage(originalText, [])
  }, [sendMessage])

  const handleExport = useCallback(() => {
    const lines: string[] = ['# QO Chat-Export', '']
    for (const m of messages) {
      const header = m.role === 'user' ? '**Du**' : `**QO${m.tier ? ` (${m.tier})` : ''}**`
      const ts = m.timestamp ? ` _${formatRelativeTime(m.timestamp)}_` : ''
      lines.push(`### ${header}${ts}`, '', m.content, '')
    }
    downloadMarkdown(`qo-chat-${Date.now()}.md`, lines.join('\n'))
  }, [messages])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        void sendMessage()
      }
    },
    [sendMessage],
  )

  // ─── Render ────────────────────────────────────────────────────────

  const conversationLead = useMemo(
    () => (messages.length === 0 ? 'Neue Konversation' : null),
    [messages.length],
  )

  return (
    <div className="chat">
      {/* Header */}
      <header className="chat__header">
        <div className="chat__header-title">
          <span className="chat__header-label">
            {conversationLead ?? `${Math.ceil(messages.length / 2)} Runden`}
          </span>
        </div>
        {messages.length > 0 ? (
          <button
            type="button"
            className="chat__header-btn"
            onClick={handleExport}
            title="Chat als Markdown exportieren"
          >
            <Download size={14} />
            Export
          </button>
        ) : null}
      </header>

      {/* Banners */}
      {healthError ? (
        <div className="chat__banner chat__banner--err">
          <AlertCircle size={14} />
          Server nicht erreichbar — prüfe, ob <code>qo</code> läuft auf Port 4646.
        </div>
      ) : null}
      {visionWarning ? (
        <div className="chat__banner chat__banner--warn">
          <AlertCircle size={14} />
          {visionWarning}
          <button
            type="button"
            onClick={() => setVisionWarning(null)}
            aria-label="Schließen"
            className="chat__banner-close"
          >
            <X size={14} />
          </button>
        </div>
      ) : null}

      {/* Messages */}
      <div className="chat__messages">
        {messages.length === 0 && !loading ? (
          <div className="chat__empty">
            <div className="chat__empty-title">Stelle eine Frage oder beschreibe eine Aufgabe.</div>
            <div className="chat__empty-hint">
              Aufgaben („Entwickle …", „Baue …") triggern den autonomen Agent-Flow
              — sichtbar im <b>Live</b>-Tab. Texte oder Bilder als Kontext anhängen
              über das <b>Büroklammer</b>-Icon.
            </div>
          </div>
        ) : null}

        {messages.map((msg) => (
          <MessageRow key={msg.id} msg={msg} onRetry={retry} />
        ))}

        {loading ? (
          <div className="chat__row chat__row--assistant">
            <div className="chat__bubble chat__bubble--assistant chat__bubble--loading">
              <span className="chat__dot" style={{ animationDelay: '0ms' }} />
              <span className="chat__dot" style={{ animationDelay: '200ms' }} />
              <span className="chat__dot" style={{ animationDelay: '400ms' }} />
            </div>
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>

      {/* Attachment tray */}
      {attachments.length > 0 ? (
        <div className="chat__attachments">
          {attachments.map((a) => (
            <div key={a.id} className="chat__attach-chip">
              {a.kind === 'image' ? (
                <img src={a.dataUrl} alt={a.name} className="chat__attach-thumb" />
              ) : (
                <span className="chat__attach-icon"><Paperclip size={12} /></span>
              )}
              <span className="chat__attach-name">{a.name}</span>
              <span className="chat__attach-size">{Math.round(a.size / 1024)} KB</span>
              <button
                type="button"
                onClick={() => setAttachments((prev) => prev.filter((x) => x.id !== a.id))}
                aria-label={`${a.name} entfernen`}
                className="chat__attach-remove"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {/* Composer */}
      <div className="chat__composer">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,text/*,.md,.json,.csv,.tsv,.py,.rs,.ts,.tsx,.js,.jsx,.go,.java,.c,.cpp,.h,.hpp,.yml,.yaml,.xml"
          multiple
          onChange={(e) => {
            void addFiles(e.target.files)
            e.target.value = ''
          }}
          style={{ display: 'none' }}
        />
        <button
          type="button"
          className="chat__composer-btn"
          onClick={() => fileInputRef.current?.click()}
          title="Bild oder Text-Datei anhängen"
          aria-label="Anhang"
        >
          <Paperclip size={16} />
        </button>

        <textarea
          ref={inputRef}
          className="chat__composer-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Schreibe eine Nachricht … (Enter zum Senden, Shift+Enter für Zeilenumbruch)"
          rows={1}
        />

        <button
          type="button"
          className="chat__composer-send"
          onClick={() => sendMessage()}
          disabled={loading || (!input.trim() && attachments.length === 0)}
          aria-label="Senden"
        >
          <Send size={16} />
          <span>Senden</span>
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Single message row — handles user vs assistant and markdown
// ---------------------------------------------------------------------------

function MessageRow({
  msg,
  onRetry,
}: {
  msg: ChatMessage
  onRetry: (originalText: string, errorMsgId: string) => void
}) {
  const isUser = msg.role === 'user'
  const bubbleClass = msg.error
    ? 'chat__bubble chat__bubble--err'
    : isUser
      ? 'chat__bubble chat__bubble--user'
      : 'chat__bubble chat__bubble--assistant'

  return (
    <div className={`chat__row ${isUser ? 'chat__row--user' : 'chat__row--assistant'}`}>
      <div className={bubbleClass}>
        {/* Image attachments preview inside bubble */}
        {msg.attachments?.some((a) => a.kind === 'image') ? (
          <div className="chat__bubble-images">
            {msg.attachments
              .filter((a) => a.kind === 'image')
              .map((a) => (
                <img
                  key={a.id}
                  src={a.dataUrl}
                  alt={a.name}
                  className="chat__bubble-image"
                />
              ))}
          </div>
        ) : null}

        {msg.content ? (
          isUser ? (
            <div className="chat__bubble-text">{msg.content}</div>
          ) : (
            <div className="chat__markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            </div>
          )
        ) : null}
      </div>

      <div className="chat__meta">
        {!isUser && msg.tier ? (
          <span className="chat__tier">{msg.tier}</span>
        ) : null}
        {msg.timestamp ? (
          <span className="chat__time">{formatRelativeTime(msg.timestamp)}</span>
        ) : null}
        {msg.error && msg.originalText ? (
          <button
            type="button"
            className="chat__retry"
            onClick={() => onRetry(msg.originalText!, msg.id)}
            title="Erneut versuchen"
          >
            <RefreshCw size={11} /> Wiederholen
          </button>
        ) : null}
        {!isUser && msg.attachments?.some((a) => a.kind === 'image') ? (
          <span className="chat__meta-hint">
            <ImageIcon size={11} />
            Bild-Eingabe: nur mit Vision-Modell
          </span>
        ) : null}
      </div>
    </div>
  )
}
