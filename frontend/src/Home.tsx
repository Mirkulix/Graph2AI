// Home.tsx — first screen a user sees. Apple-simple: one big prompt,
// one row of live stats, a short list of "what can I do" shortcuts.
// Everything else lives behind the sidebar.

import { useEffect, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Gauge,
  Globe,
  MessageCircle,
  Sparkles,
  Users,
} from 'lucide-react'

interface ValuesResponse {
  scores: {
    achtsamkeit: number
    anerkennung: number
    aufmerksamkeit: number
    entwicklung: number
    sinn: number
  }
  average: number
}

interface FederationStats {
  rounds_completed: number
  rounds_failed: number
  gossip_rate_per_sec: number
  bandwidth_saved_pct: number
  peer_discovery_enabled: boolean
}

interface PeersResponse {
  total_nodes: number
  local_nodes: number
  peers: unknown[]
}

interface HomeProps {
  onNavigate: (tab: string) => void
  onAsk?: (prompt: string) => void
}

const SUGGESTIONS = [
  'Welche Graphen wurden heute ausgeführt?',
  'Wie ist der Zustand des Schwarms?',
  'Zeig mir den letzten signierten QLMS-Frame',
]

export default function Home({ onNavigate, onAsk }: HomeProps) {
  const [prompt, setPrompt] = useState('')
  const [values, setValues] = useState<ValuesResponse | null>(null)
  const [stats, setStats] = useState<FederationStats | null>(null)
  const [peers, setPeers] = useState<PeersResponse | null>(null)
  const [health, setHealth] = useState<'ok' | 'err' | 'loading'>('loading')

  useEffect(() => {
    let cancelled = false

    async function pull() {
      const endpoints: [string, (data: unknown) => void][] = [
        ['/api/values', (d) => setValues(d as ValuesResponse)],
        ['/api/federation/stats', (d) => setStats(d as FederationStats)],
        ['/api/federation/peers', (d) => setPeers(d as PeersResponse)],
      ]
      let anyOk = false
      await Promise.all(
        endpoints.map(async ([url, setter]) => {
          try {
            const r = await fetch(url)
            if (!r.ok) return
            setter(await r.json())
            anyOk = true
          } catch {
            /* ignore — degrade gracefully */
          }
        }),
      )
      if (!cancelled) setHealth(anyOk ? 'ok' : 'err')
    }

    void pull()
    const handle = window.setInterval(pull, 5_000)
    return () => {
      cancelled = true
      window.clearInterval(handle)
    }
  }, [])

  function submit() {
    const v = prompt.trim()
    if (!v) return
    if (onAsk) {
      onAsk(v)
    } else {
      onNavigate('chat')
    }
    setPrompt('')
  }

  const averagePct = values ? Math.round(values.average * 100) : null
  const healthPill =
    health === 'loading' ? (
      <span className="pill">Verbinde …</span>
    ) : health === 'ok' ? (
      <span className="pill pill--ok">
        <span className="dot dot--ok dot--pulse" />
        System bereit
      </span>
    ) : (
      <span className="pill pill--err">
        <span className="dot dot--err" />
        Keine Verbindung
      </span>
    )

  return (
    <div className="home">
      {/* ────────────── Hero ────────────── */}
      <section className="home__hero">
        <span className="home__eyebrow">
          <Sparkles size={12} /> QLANG A2A · Live
        </span>
        <h1 className="home__title">Was möchtest du tun?</h1>
        <p className="home__subtitle">
          Dein Schwarm aus spezialisierten Agenten steht bereit. Frag etwas — der CEO-Agent
          verteilt die Aufgabe, der Guardian prüft sie gegen die fünf Kernwerte.
        </p>

        <form
          className="home__prompt"
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
        >
          <input
            className="home__prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Beschreibe eine Aufgabe, stelle eine Frage …"
            aria-label="Task eingeben"
          />
          <button type="submit" className="home__prompt-btn" aria-label="Absenden">
            Senden <ArrowRight size={14} />
          </button>
        </form>

        <div className="home__suggestions">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              className="home__suggestion"
              onClick={() => {
                setPrompt(s)
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </section>

      {/* ────────────── Live-Status ────────────── */}
      <div className="home__section-title">
        Live-Status <span>{healthPill}</span>
      </div>

      <section className="home__stats" aria-label="Live-Status">
        <StatCard
          label="Agenten"
          value={peers ? `${peers.total_nodes}` : '—'}
          meta="6 Rollen aktiv"
          icon={<Users size={14} />}
        />
        <StatCard
          label="Werte-Mittel"
          value={averagePct !== null ? `${averagePct}%` : '—'}
          meta={
            values
              ? `Achtsamkeit · Anerkennung · Aufmerksamkeit · Entwicklung · Sinn`
              : ''
          }
          icon={<Gauge size={14} />}
          accent={averagePct && averagePct >= 70 ? 'ok' : undefined}
        />
        <StatCard
          label="Gossip-Rate"
          value={stats ? `${stats.gossip_rate_per_sec.toFixed(2)}/s` : '—'}
          meta={
            stats?.peer_discovery_enabled
              ? `${stats.rounds_completed} erfolgreich`
              : 'Peer-Discovery deaktiviert'
          }
          icon={<Activity size={14} />}
        />
        <StatCard
          label="Bandbreite gespart"
          value={stats ? `${Math.round(stats.bandwidth_saved_pct)}%` : '—'}
          meta="vs. JSON (QLMS §13.2)"
          icon={<Globe size={14} />}
          accent="accent"
        />
      </section>

      {/* ────────────── Shortcuts ────────────── */}
      <div className="home__section-title">Weitermachen</div>
      <section className="home__quicklinks">
        <Quicklink
          icon={<MessageCircle size={18} />}
          title="Chat"
          desc="Sprich direkt mit dem Schwarm. Antworten sind kryptografisch signiert."
          onClick={() => onNavigate('chat')}
        />
        <Quicklink
          icon={<Activity size={18} />}
          title="Live verfolgen"
          desc="Sieh den DAG laufender Agenten-Nachrichten in Echtzeit."
          onClick={() => onNavigate('mission')}
        />
        <Quicklink
          icon={<Globe size={18} />}
          title="Netzwerk"
          desc="3D-Karte des Föderations-Schwarms, lokale Knoten blau hervorgehoben."
          onClick={() => onNavigate('swarm')}
        />
      </section>
    </div>
  )
}

function StatCard({
  label,
  value,
  meta,
  icon,
  accent,
}: {
  label: string
  value: string
  meta?: string
  icon?: React.ReactNode
  accent?: 'ok' | 'accent'
}) {
  return (
    <div className="home__stat">
      <div className="home__stat-label">
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          {icon}
          {label}
        </span>
      </div>
      <div
        className="home__stat-value"
        style={
          accent === 'ok'
            ? { color: 'var(--ok)' }
            : accent === 'accent'
              ? { color: 'var(--accent-strong)' }
              : undefined
        }
      >
        {value}
      </div>
      {meta ? <div className="home__stat-meta">{meta}</div> : null}
    </div>
  )
}

function Quicklink({
  icon,
  title,
  desc,
  onClick,
}: {
  icon: React.ReactNode
  title: string
  desc: string
  onClick: () => void
}) {
  return (
    <button type="button" className="home__quicklink" onClick={onClick}>
      <div className="home__quicklink-icon">{icon}</div>
      <h3 className="home__quicklink-title">{title}</h3>
      <p className="home__quicklink-desc">{desc}</p>
    </button>
  )
}
