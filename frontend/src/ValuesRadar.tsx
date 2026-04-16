// ValuesRadar.tsx — PRD Task 6.3 / UI-Konzept "Werte-Radar".
//
// Renders the five Kernwerte (Achtsamkeit / Anerkennung / Aufmerksamkeit /
// Entwicklung / Sinn) as a pentagon radar chart. Pure SVG, no chart
// library — the shape is simple enough that pulling in recharts (~400 KB)
// would be disproportionate.
//
// Data source: GET /api/values → { scores: ValueScores, average: f32 }.
// Auto-refresh every 5 s. The five accent colours come from CSS custom
// properties declared in styles.css (`--accent-achtsamkeit`, …).
//
// Intended consumers:
//  * standalone tab in App.tsx (for now)
//  * permanent sidebar in MissionControl (Task 6.1) once that lands.

import { useEffect, useMemo, useState } from 'react'
import { Shield } from 'lucide-react'

type ValueKey =
  | 'achtsamkeit'
  | 'anerkennung'
  | 'aufmerksamkeit'
  | 'entwicklung'
  | 'sinn'

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

const VALUE_KEYS: ValueKey[] = [
  'achtsamkeit',
  'anerkennung',
  'aufmerksamkeit',
  'entwicklung',
  'sinn',
]

const VALUE_LABELS: Record<ValueKey, string> = {
  achtsamkeit: 'Achtsamkeit',
  anerkennung: 'Anerkennung',
  aufmerksamkeit: 'Aufmerksamkeit',
  entwicklung: 'Entwicklung',
  sinn: 'Sinn',
}

const VALUE_COLORS: Record<ValueKey, string> = {
  achtsamkeit: 'var(--werte-achtsamkeit)',
  anerkennung: 'var(--werte-anerkennung)',
  aufmerksamkeit: 'var(--werte-aufmerksamkeit)',
  entwicklung: 'var(--werte-entwicklung)',
  sinn: 'var(--werte-sinn)',
}

const POLL_MS = 5_000
const AXIS_LABEL_RADIUS = 115
const CHART_RADIUS = 90
const SIZE = 260
const CENTER = SIZE / 2
const RINGS = 4 // 0.25, 0.5, 0.75, 1.0

/**
 * Pentagon vertex for axis i (0..4), scaled by `ratio` in [0,1].
 * Angle starts at -π/2 (top) and advances clockwise.
 */
function polar(i: number, ratio: number, radius: number): [number, number] {
  const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 5
  return [CENTER + Math.cos(angle) * radius * ratio, CENTER + Math.sin(angle) * radius * ratio]
}

function axisEnd(i: number): [number, number] {
  return polar(i, 1, CHART_RADIUS)
}

function polygonPoints(ratios: number[], radius: number): string {
  return ratios
    .map((r, i) => polar(i, r, radius).join(','))
    .join(' ')
}

export default function ValuesRadar() {
  const [resp, setResp] = useState<ValuesResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [fetchedAt, setFetchedAt] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false

    async function tick() {
      try {
        const r = await fetch('/api/values')
        if (!r.ok) {
          throw new Error(`HTTP ${r.status}`)
        }
        const body = (await r.json()) as ValuesResponse
        if (cancelled) return
        setResp(body)
        setError(null)
        setFetchedAt(Date.now())
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      }
    }

    tick()
    const handle = window.setInterval(tick, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(handle)
    }
  }, [])

  const ratios = useMemo(() => {
    if (!resp) return [0, 0, 0, 0, 0]
    return VALUE_KEYS.map((k) => clamp01(resp.scores[k]))
  }, [resp])

  const average = resp?.average ?? 0

  return (
    <div className="values-radar">
      <header className="values-radar__header">
        <Shield size={18} />
        <h2>Werte-Radar</h2>
        <span className="values-radar__avg" title="Durchschnittswert (Guardian)">
          ⌀ {average.toFixed(2)}
        </span>
      </header>

      <svg
        className="values-radar__svg"
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        width={SIZE}
        height={SIZE}
        role="img"
        aria-label={`Werte-Radar: Achtsamkeit ${ratios[0].toFixed(2)}, Anerkennung ${ratios[1].toFixed(2)}, Aufmerksamkeit ${ratios[2].toFixed(2)}, Entwicklung ${ratios[3].toFixed(2)}, Sinn ${ratios[4].toFixed(2)}`}
      >
        {/* concentric guide rings (0.25/0.5/0.75/1.0) */}
        {Array.from({ length: RINGS }, (_, r) => {
          const ratio = (r + 1) / RINGS
          return (
            <polygon
              key={`ring-${r}`}
              points={polygonPoints([ratio, ratio, ratio, ratio, ratio], CHART_RADIUS)}
              fill="none"
              stroke="var(--border)"
              strokeWidth={ratio === 1 ? 1.4 : 0.6}
              strokeDasharray={ratio === 1 ? undefined : '2 3'}
            />
          )
        })}

        {/* axes */}
        {VALUE_KEYS.map((_, i) => {
          const [x, y] = axisEnd(i)
          return (
            <line
              key={`axis-${i}`}
              x1={CENTER}
              y1={CENTER}
              x2={x}
              y2={y}
              stroke="var(--border)"
              strokeWidth={0.6}
            />
          )
        })}

        {/* data polygon */}
        <polygon
          points={polygonPoints(ratios, CHART_RADIUS)}
          fill="var(--accent)"
          fillOpacity={0.14}
          stroke="var(--accent)"
          strokeWidth={1.6}
        />

        {/* per-axis data dots — coloured per value */}
        {VALUE_KEYS.map((key, i) => {
          const [x, y] = polar(i, ratios[i], CHART_RADIUS)
          return (
            <circle
              key={`dot-${key}`}
              cx={x}
              cy={y}
              r={4}
              fill={VALUE_COLORS[key]}
              stroke="var(--surface)"
              strokeWidth={1.4}
            >
              <title>
                {VALUE_LABELS[key]}: {ratios[i].toFixed(2)}
              </title>
            </circle>
          )
        })}

        {/* axis labels */}
        {VALUE_KEYS.map((key, i) => {
          const [x, y] = polar(i, 1, AXIS_LABEL_RADIUS)
          return (
            <text
              key={`label-${key}`}
              x={x}
              y={y}
              fill="var(--text-muted)"
              fontSize={11}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {VALUE_LABELS[key]}
            </text>
          )
        })}
      </svg>

      <footer className="values-radar__footer">
        {error ? (
          <span className="values-radar__status values-radar__status--error">
            Fehler: {error}
          </span>
        ) : fetchedAt ? (
          <span className="values-radar__status">
            Aktualisiert {formatAge(fetchedAt)}
          </span>
        ) : (
          <span className="values-radar__status">Verbinde …</span>
        )}
      </footer>
    </div>
  )
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function formatAge(at: number): string {
  const s = Math.max(0, Math.round((Date.now() - at) / 1000))
  if (s < 5) return 'gerade eben'
  if (s < 60) return `vor ${s}s`
  const m = Math.floor(s / 60)
  return `vor ${m}min`
}
