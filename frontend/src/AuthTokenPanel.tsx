// AuthTokenPanel.tsx — floating settings panel that lets the user set or
// clear the Bearer token persisted in localStorage under 'qo_auth_token'.
// main.tsx's fetch interceptor already reads that key, so this component
// only has to manage the value (and reveal the last 4 chars for feedback).
//
// No props. Default-exports a self-contained component with its own
// trigger button + panel.

import { useEffect, useRef, useState } from 'react'
import { KeyRound, Trash2, Save } from 'lucide-react'

const STORAGE_KEY = 'qo_auth_token'

function readStoredToken(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? ''
  } catch {
    return ''
  }
}

function last4(tok: string): string {
  if (!tok) return ''
  if (tok.length <= 4) return tok
  return tok.slice(-4)
}

export default function AuthTokenPanel() {
  const [open, setOpen] = useState(false)
  const [stored, setStored] = useState<string>(() => readStoredToken())
  const [value, setValue] = useState<string>('')
  const [toast, setToast] = useState<string | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  // When the panel opens, refresh from storage and seed the input with a
  // masked preview of the current token (so the user can see it's set
  // without exposing the full value).
  useEffect(() => {
    if (!open) return
    const current = readStoredToken()
    setStored(current)
    setValue(current)
    // Focus the input for quick editing.
    const h = window.setTimeout(() => inputRef.current?.focus(), 0)
    return () => window.clearTimeout(h)
  }, [open])

  // Outside-click and ESC close the panel.
  useEffect(() => {
    if (!open) return
    function onDocClick(ev: MouseEvent) {
      const target = ev.target as Node | null
      if (!target) return
      if (wrapRef.current && !wrapRef.current.contains(target)) {
        setOpen(false)
      }
    }
    function onKey(ev: KeyboardEvent) {
      if (ev.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  // Auto-dismiss toast.
  useEffect(() => {
    if (!toast) return
    const h = window.setTimeout(() => setToast(null), 1800)
    return () => window.clearTimeout(h)
  }, [toast])

  function handleSave() {
    const trimmed = value.trim()
    if (!trimmed) {
      setToast('leer — nichts gespeichert')
      return
    }
    try {
      localStorage.setItem(STORAGE_KEY, trimmed)
      setStored(trimmed)
      setToast('gespeichert')
      setOpen(false)
    } catch {
      setToast('Speichern fehlgeschlagen')
    }
  }

  function handleRemove() {
    try {
      localStorage.removeItem(STORAGE_KEY)
    } catch {
      /* ignore */
    }
    setStored('')
    setValue('')
    setToast('gelöscht')
  }

  const wrapStyle: React.CSSProperties = {
    position: 'relative',
    display: 'inline-flex',
  }

  const panelStyle: React.CSSProperties = {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    zIndex: 50,
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-raised)',
    borderRadius: 'var(--radius-card)',
    padding: '14px 16px',
    minWidth: 300,
    color: 'var(--text)',
  }

  const headingStyle: React.CSSProperties = {
    fontSize: 13,
    fontWeight: 600,
    color: 'var(--text)',
    marginBottom: 10,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  }

  const inputStyle: React.CSSProperties = {
    width: '100%',
    boxSizing: 'border-box',
    background: 'var(--surface-sunken)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    padding: '8px 10px',
    fontSize: 12,
    color: 'var(--text)',
    fontFamily:
      'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    outline: 'none',
  }

  const rowStyle: React.CSSProperties = {
    display: 'flex',
    gap: 8,
    marginTop: 10,
  }

  const btnBase: React.CSSProperties = {
    flex: 1,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    padding: '8px 10px',
    fontSize: 12,
    fontWeight: 500,
    borderRadius: 8,
    border: '1px solid var(--border)',
    background: 'var(--surface)',
    color: 'var(--text)',
    cursor: 'pointer',
  }

  const btnPrimary: React.CSSProperties = {
    ...btnBase,
    background: 'var(--accent, #4f46e5)',
    borderColor: 'transparent',
    color: '#fff',
  }

  const statusStyle: React.CSSProperties = {
    marginTop: 12,
    paddingTop: 10,
    borderTop: '1px solid var(--border)',
    fontSize: 12,
    color: 'var(--text-muted)',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  }

  const dotActive: React.CSSProperties = {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: 'var(--ok, #10b981)',
    display: 'inline-block',
  }

  const dotInactive: React.CSSProperties = {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: 'var(--border-strong, #cbd1db)',
    display: 'inline-block',
  }

  const toastStyle: React.CSSProperties = {
    marginTop: 8,
    fontSize: 11,
    color: 'var(--text-muted)',
    fontStyle: 'italic',
  }

  return (
    <div ref={wrapRef} style={wrapStyle}>
      <button
        type="button"
        className="header__icon-btn"
        onClick={() => setOpen((v) => !v)}
        aria-label="Auth-Token"
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Auth-Token"
      >
        <KeyRound size={16} />
      </button>

      {open ? (
        <div role="dialog" aria-label="Auth-Token" style={panelStyle}>
          <div style={headingStyle}>
            <KeyRound size={14} />
            Auth-Token
          </div>

          <input
            ref={inputRef}
            type="password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Bearer-Token einfügen…"
            style={inputStyle}
            autoComplete="off"
            spellCheck={false}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                handleSave()
              }
            }}
          />

          <div style={rowStyle}>
            <button
              type="button"
              style={btnPrimary}
              onClick={handleSave}
              title="Token speichern"
            >
              <Save size={14} />
              Speichern
            </button>
            <button
              type="button"
              style={btnBase}
              onClick={handleRemove}
              title="Token entfernen"
              disabled={!stored}
            >
              <Trash2 size={14} />
              Entfernen
            </button>
          </div>

          <div style={statusStyle}>
            {stored ? (
              <>
                <span style={dotActive} aria-hidden />
                <span>
                  Token aktiv (&hellip;{last4(stored)})
                </span>
              </>
            ) : (
              <>
                <span style={dotInactive} aria-hidden />
                <span>Kein Token gesetzt</span>
              </>
            )}
          </div>

          {toast ? <div style={toastStyle}>{toast}</div> : null}
        </div>
      ) : null}
    </div>
  )
}
