// CostBadge.tsx — small header chip that polls /api/providers every 15s and
// shows total session cost + total requests. Clicking it opens a popover
// with a per-provider breakdown (model, requests, cost, avg latency).
//
// No props. Default-exports a self-contained component.

import { useEffect, useRef, useState } from 'react'
import { DollarSign, Zap } from 'lucide-react'

interface ProviderStats {
  name: string
  model: string
  requests: number
  total_tokens_estimate: number
  cost_usd: number
  avg_latency_ms: number
  status: string
}

interface ProvidersResponse {
  providers: ProviderStats[]
  total_cost_usd: number
  total_requests: number
}

function fmtUsd(v: number): string {
  if (!Number.isFinite(v)) return '$0.00'
  if (v < 0.01 && v > 0) return `$${v.toFixed(4)}`
  return `$${v.toFixed(2)}`
}

function fmtLatency(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '—'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  return `${Math.round(ms)}ms`
}

export default function CostBadge() {
  const [data, setData] = useState<ProvidersResponse | null>(null)
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let cancelled = false

    async function pull() {
      try {
        const r = await fetch('/api/providers')
        if (!r.ok) return
        const body = (await r.json()) as ProvidersResponse
        if (!cancelled) setData(body)
      } catch {
        /* silent — badge degrades to placeholder */
      }
    }

    void pull()
    const h = window.setInterval(pull, 15_000)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  // Close popover on outside click — listener only attached while open.
  useEffect(() => {
    if (!open) return
    function onDocClick(ev: MouseEvent) {
      const target = ev.target as Node | null
      if (!target) return
      if (wrapRef.current && !wrapRef.current.contains(target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  const totalCost = data?.total_cost_usd ?? 0
  const totalReq = data?.total_requests ?? 0
  const providers = data?.providers ?? []

  const wrapStyle: React.CSSProperties = {
    position: 'relative',
    display: 'inline-flex',
  }

  const chipStyle: React.CSSProperties = {
    fontFamily:
      'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    fontSize: 12,
    fontVariantNumeric: 'tabular-nums',
  }

  const popoverStyle: React.CSSProperties = {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    zIndex: 50,
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-raised)',
    borderRadius: 'var(--radius-card)',
    padding: '14px 16px',
    minWidth: 280,
    color: 'var(--text)',
  }

  const popoverHeaderStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 10,
    paddingBottom: 10,
    borderBottom: '1px solid var(--border)',
    fontSize: 12,
    color: 'var(--text-muted)',
  }

  const popoverTotalStyle: React.CSSProperties = {
    fontFamily:
      'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    fontSize: 13,
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--text)',
    fontWeight: 600,
  }

  const rowStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: '1fr auto auto',
    columnGap: 14,
    rowGap: 2,
    padding: '8px 0',
    borderTop: '1px solid var(--border)',
    alignItems: 'baseline',
  }

  const rowFirstStyle: React.CSSProperties = {
    ...rowStyle,
    borderTop: 'none',
    paddingTop: 0,
  }

  const nameStyle: React.CSSProperties = {
    fontSize: 13,
    color: 'var(--text)',
    fontWeight: 500,
  }

  const modelStyle: React.CSSProperties = {
    fontSize: 11,
    color: 'var(--text-muted)',
    gridColumn: '1 / -1',
    marginTop: 2,
  }

  const numStyle: React.CSSProperties = {
    fontFamily:
      'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    fontSize: 12,
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--text-muted)',
  }

  const emptyStyle: React.CSSProperties = {
    padding: '10px 0 2px',
    fontSize: 12,
    color: 'var(--text-muted)',
    textAlign: 'center',
  }

  return (
    <div ref={wrapRef} style={wrapStyle}>
      <button
        type="button"
        className="header__chip"
        style={chipStyle}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Session-Kosten und Requests"
      >
        <DollarSign size={14} />
        <span>
          {fmtUsd(totalCost)} &middot; {totalReq} calls
        </span>
      </button>

      {open ? (
        <div role="dialog" aria-label="Provider-Kosten" style={popoverStyle}>
          <div style={popoverHeaderStyle}>
            <span>Provider &mdash; Session</span>
            <span style={popoverTotalStyle}>
              {fmtUsd(totalCost)} &middot; {totalReq} calls
            </span>
          </div>

          {providers.length === 0 ? (
            <div style={emptyStyle}>Noch keine Requests</div>
          ) : (
            providers.map((p, idx) => (
              <div
                key={`${p.name}-${p.model}-${idx}`}
                style={idx === 0 ? rowFirstStyle : rowStyle}
              >
                <span style={nameStyle}>{p.name}</span>
                <span style={numStyle}>{p.requests} req</span>
                <span style={numStyle}>{fmtUsd(p.cost_usd)}</span>
                <span style={modelStyle}>
                  {p.model}
                  <span style={{ marginLeft: 8, display: 'inline-flex', alignItems: 'center', gap: 3, verticalAlign: 'middle' }}>
                    <Zap size={10} />
                    {fmtLatency(p.avg_latency_ms)}
                  </span>
                </span>
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  )
}
