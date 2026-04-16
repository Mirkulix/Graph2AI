// WerteStatus.tsx — persistent 5-Werte indicator for the app header.
// Shows five tiny coloured dots (one per Kernwert) plus the running
// average. Clicking the chip takes the user to the full radar view.
//
// Poll cadence is deliberately slow (10 s) because the Guardian agent
// writes to /api/values only when it forms a new assessment — there is
// no reason to hammer the endpoint.

import { useEffect, useState } from 'react'

interface ValueScores {
  achtsamkeit: number
  anerkennung: number
  aufmerksamkeit: number
  entwicklung: number
  sinn: number
}

interface ValuesResponse {
  scores: ValueScores
  average: number
}

const KEYS: [keyof ValueScores, string, string][] = [
  ['achtsamkeit', 'Achtsamkeit', 'var(--werte-achtsamkeit)'],
  ['anerkennung', 'Anerkennung', 'var(--werte-anerkennung)'],
  ['aufmerksamkeit', 'Aufmerksamkeit', 'var(--werte-aufmerksamkeit)'],
  ['entwicklung', 'Entwicklung', 'var(--werte-entwicklung)'],
  ['sinn', 'Sinn', 'var(--werte-sinn)'],
]

interface WerteStatusProps {
  onOpen: () => void
}

export default function WerteStatus({ onOpen }: WerteStatusProps) {
  const [resp, setResp] = useState<ValuesResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    async function pull() {
      try {
        const r = await fetch('/api/values')
        if (!r.ok) return
        const body = (await r.json()) as ValuesResponse
        if (!cancelled) setResp(body)
      } catch {
        /* silent — chip degrades to placeholder */
      }
    }
    void pull()
    const h = window.setInterval(pull, 10_000)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  const avg = resp ? Math.round(resp.average * 100) : null

  const title = resp
    ? KEYS.map(
        ([k, label]) => `${label}: ${(resp.scores[k] * 100).toFixed(0)}%`,
      ).join(' · ') + `  ⌀ ${avg}%`
    : 'Werte werden geladen …'

  return (
    <button
      type="button"
      className="werte-chip"
      onClick={onOpen}
      title={title}
      aria-label={`Werte-Radar öffnen. ${title}`}
    >
      <span className="werte-chip__dots" aria-hidden="true">
        {KEYS.map(([k, , color]) => {
          const score = resp?.scores[k] ?? 0.5
          // opacity follows the score so a weak value is visibly duller
          // without requiring users to read numbers.
          const opacity = Math.max(0.25, score)
          return (
            <span
              key={k}
              className="werte-chip__dot"
              style={{ background: color, opacity }}
            />
          )
        })}
      </span>
      <span className="werte-chip__value">
        {avg !== null ? `${avg}%` : '—'}
      </span>
    </button>
  )
}
