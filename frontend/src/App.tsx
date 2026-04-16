import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ChevronDown,
  FolderOpen,
  Globe,
  History as HistoryIcon,
  Home as HomeIcon,
  ListTodo,
  MessageCircle,
  Moon,
  Shield,
  ShieldCheck,
  ShieldX,
  Sparkles,
  Sun,
} from 'lucide-react'
import Home from './Home'
import WerteStatus from './WerteStatus'
import MissionControl from './MissionControl'
import GraphInspectorView from './GraphInspectorView'
import SwarmMap from './SwarmMap'
import ValuesRadar from './ValuesRadar'
import ChatView from './ChatView'
import WorkspaceView from './WorkspaceView'
import GoalsBrowserView from './GoalsBrowserView'
import CostBadge from './CostBadge'
import AuthTokenPanel from './AuthTokenPanel'
import NeoShell from './neo/NeoShell'
import GoalsView from './GoalsView'
import AgentsView from './AgentsView'
import ConsciousnessView from './ConsciousnessView'
import ProviderView from './ProviderView'
import EvolutionView from './EvolutionView'
import GraphsView from './GraphsView'
import KnowledgeGraph3DView from './KnowledgeGraph3DView'
import MessagesView from './MessagesView'
import TrainingView from './TrainingView'
import GpuTrainingView from './GpuTrainingView'
import SpikingView from './SpikingView'
import OrganismView from './OrganismView'
import HistorieView from './HistorieView'

type Tab =
  | 'home'
  | 'chat'
  | 'mission'
  | 'inspector'
  | 'workspace'
  | 'goals-browser'
  | 'swarm'
  | 'werte'
  | 'neo'
  | 'agents'
  | 'goals'
  | 'provider'
  | 'evolution'
  | 'graphs'
  | 'knowledge-3d'
  | 'messages'
  | 'training'
  | 'gpu-training'
  | 'spiking'
  | 'organism'
  | 'historie'
  | 'consciousness'

type NavKind = 'primary' | 'advanced'

interface NavItem {
  id: Tab
  label: string
  icon: typeof HomeIcon
  kind: NavKind
  badge?: string
}

const NAV: NavItem[] = [
  { id: 'home', label: 'Home', icon: HomeIcon, kind: 'primary' },
  { id: 'chat', label: 'Chat', icon: MessageCircle, kind: 'primary' },
  { id: 'mission', label: 'Live', icon: Activity, kind: 'primary' },
  { id: 'workspace', label: 'Workspace', icon: FolderOpen, kind: 'primary' },
  { id: 'goals-browser', label: 'Ziele', icon: ListTodo, kind: 'primary' },
  { id: 'inspector', label: 'Verlauf', icon: HistoryIcon, kind: 'primary' },
  { id: 'swarm', label: 'Netzwerk', icon: Globe, kind: 'primary' },
  { id: 'werte', label: 'Werte-Radar', icon: Shield, kind: 'primary' },

  // Everything below is collapsed under "Erweitert" by default.
  { id: 'neo', label: 'Neo Shell', icon: Sparkles, kind: 'advanced' },
  { id: 'agents', label: 'Agenten', icon: HomeIcon, kind: 'advanced' },
  { id: 'goals', label: 'Ziele', icon: HomeIcon, kind: 'advanced' },
  { id: 'graphs', label: 'QLANG-Editor', icon: HomeIcon, kind: 'advanced' },
  { id: 'messages', label: 'Messages', icon: HomeIcon, kind: 'advanced' },
  { id: 'provider', label: 'Provider', icon: HomeIcon, kind: 'advanced' },
  { id: 'evolution', label: 'Evolution', icon: HomeIcon, kind: 'advanced' },
  { id: 'training', label: 'Training', icon: HomeIcon, kind: 'advanced' },
  { id: 'gpu-training', label: 'GPU Training', icon: HomeIcon, kind: 'advanced' },
  { id: 'spiking', label: 'Spiking', icon: HomeIcon, kind: 'advanced' },
  { id: 'organism', label: 'Organismus', icon: HomeIcon, kind: 'advanced' },
  { id: 'consciousness', label: 'Bewusstsein', icon: HomeIcon, kind: 'advanced' },
  { id: 'knowledge-3d', label: 'Knowledge 3D', icon: HomeIcon, kind: 'advanced' },
  { id: 'historie', label: 'Historie', icon: HomeIcon, kind: 'advanced' },
]

const THEME_KEY = 'qo_theme'
const ADVANCED_OPEN_KEY = 'qo_nav_advanced_open'

function readTheme(): 'light' | 'dark' {
  const stored = localStorage.getItem(THEME_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('home')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => readTheme())
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(() => {
    return localStorage.getItem(ADVANCED_OPEN_KEY) === '1'
  })
  const [qlmsStatus, setQlmsStatus] = useState<'checking' | 'ok' | 'warn' | 'err'>(
    'checking',
  )
  // Prompt the user typed on the Home hero — handed to ChatView on the
  // next navigation. ChatView clears it via onPendingConsumed so it
  // does not re-fire if the user flips back to chat later.
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  // Whether the Home toggle "Als Ziel an den Schwarm senden" was on
  // when the hero prompt was submitted. ChatView reads this once via
  // pendingForceGoal and attaches force_intent: "goal" on the outgoing
  // POST /api/chat body.
  const [forceGoalPending, setForceGoalPending] = useState<boolean>(false)

  // theme persistence
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem(ADVANCED_OPEN_KEY, advancedOpen ? '1' : '0')
  }, [advancedOpen])

  // Listen for qo:navigate CustomEvents dispatched by children (e.g. the
  // Chat decision-trace "Ziel #N ansehen" button). Payload:
  //   { tab: Tab, goalId?: number }  — tab is required.
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as
        | { tab?: string; goalId?: number }
        | undefined
      // ChatView emits tab: "goals" for the trace-link; normalise to our
      // actual tab id "goals-browser".
      if (detail?.tab === 'goals' || typeof detail?.goalId === 'number') {
        setActiveTab('goals-browser')
        return
      }
      if (detail?.tab) setActiveTab(detail.tab as Tab)
    }
    window.addEventListener('qo:navigate', handler as EventListener)
    return () => window.removeEventListener('qo:navigate', handler as EventListener)
  }, [])

  // QLMS round-trip probe for the header badge
  useEffect(() => {
    let cancelled = false

    async function probe() {
      try {
        const reply = await fetch('/qlms/v1.1/reply', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: [] }),
        })
        if (!reply.ok) throw new Error(`reply HTTP ${reply.status}`)
        const body = await reply.json()
        const deliver = await fetch('/qlms/v1.1/deliver', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ encoding: 'base64', frame: body.frame }),
        })
        if (!deliver.ok) {
          if (!cancelled) setQlmsStatus('warn')
          return
        }
        const d = await deliver.json()
        if (!cancelled) {
          // unsigned by default; reply+deliver succeeding is the healthy state
          setQlmsStatus(d.signature_verified ? 'ok' : 'ok')
        }
      } catch {
        if (!cancelled) setQlmsStatus('err')
      }
    }

    void probe()
    const h = window.setInterval(probe, 30_000)
    return () => {
      cancelled = true
      window.clearInterval(h)
    }
  }, [])

  const primary = useMemo(() => NAV.filter((n) => n.kind === 'primary'), [])
  const advanced = useMemo(() => NAV.filter((n) => n.kind === 'advanced'), [])

  return (
    <div className="app">
      {/* ── Logo / brand ── */}
      <div className="app__logo">
        <span className="app__logo-mark">Q</span>
        <span>
          <span className="app__logo-name">QLANG</span>
          <span className="app__logo-name-beta">A2A</span>
        </span>
      </div>

      {/* ── Header chrome ── */}
      <header className="app__header">
        <WerteStatus onOpen={() => setActiveTab('werte')} />
        <QlmsBadge status={qlmsStatus} />
        <CostBadge />
        <AuthTokenPanel />
        <button
          type="button"
          className="header__icon-btn"
          onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
          aria-label="Theme umschalten"
          title="Theme umschalten"
        >
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
      </header>

      {/* ── Sidebar nav ── */}
      <nav className="app__nav" aria-label="Hauptnavigation">
        {primary.map((n) => (
          <NavButton
            key={n.id}
            item={n}
            active={activeTab === n.id}
            onClick={() => setActiveTab(n.id)}
          />
        ))}

        <div className="nav__accordion" data-open={advancedOpen ? 'true' : 'false'}>
          <button
            type="button"
            className="nav__accordion-header"
            onClick={() => setAdvancedOpen((v) => !v)}
            aria-expanded={advancedOpen}
          >
            <span>Erweitert</span>
            <ChevronDown size={14} className="nav__accordion-chevron" />
          </button>
          {advancedOpen
            ? advanced.map((n) => (
                <NavButton
                  key={n.id}
                  item={n}
                  active={activeTab === n.id}
                  onClick={() => setActiveTab(n.id)}
                />
              ))
            : null}
        </div>
      </nav>

      {/* ── Main content ── */}
      <main className="app__main">
        <View
          tab={activeTab}
          setActiveTab={setActiveTab}
          pendingPrompt={pendingPrompt}
          pendingForceGoal={forceGoalPending}
          onAsk={(text, forceGoal) => {
            setPendingPrompt(text)
            setForceGoalPending(forceGoal ?? false)
            setActiveTab('chat')
          }}
          onPendingConsumed={() => {
            setPendingPrompt(null)
            setForceGoalPending(false)
          }}
        />
      </main>
    </div>
  )
}

function NavButton({
  item,
  active,
  onClick,
}: {
  item: NavItem
  active: boolean
  onClick: () => void
}) {
  const Icon = item.icon
  return (
    <button
      type="button"
      className={`nav__item${active ? ' is-active' : ''}`}
      onClick={onClick}
    >
      <Icon size={16} className="nav__item-icon" />
      <span className="nav__item-label">{item.label}</span>
      {item.badge ? <span className="nav__item-badge">{item.badge}</span> : null}
    </button>
  )
}

function QlmsBadge({ status }: { status: 'checking' | 'ok' | 'warn' | 'err' }) {
  if (status === 'ok') {
    return (
      <span className="header__chip header__chip--ok" title="QLMS-Verbindung OK">
        <ShieldCheck size={14} />
        QLMS
      </span>
    )
  }
  if (status === 'warn') {
    return (
      <span
        className="header__chip header__chip--warn"
        title="QLMS antwortet, Signatur nicht verifiziert"
      >
        <Shield size={14} />
        QLMS
      </span>
    )
  }
  if (status === 'err') {
    return (
      <span
        className="header__chip header__chip--err"
        title="QLMS nicht erreichbar"
      >
        <ShieldX size={14} />
        QLMS
      </span>
    )
  }
  return (
    <span className="header__chip" title="Prüfe QLMS …">
      <Shield size={14} />
      QLMS
    </span>
  )
}

function View({
  tab,
  setActiveTab,
  pendingPrompt,
  pendingForceGoal,
  onAsk,
  onPendingConsumed,
}: {
  tab: Tab
  setActiveTab: (t: Tab) => void
  pendingPrompt: string | null
  pendingForceGoal: boolean
  onAsk: (text: string, forceGoal?: boolean) => void
  onPendingConsumed: () => void
}) {
  switch (tab) {
    case 'home':
      return (
        <div className="view">
          <Home
            onNavigate={(t) => setActiveTab(t as Tab)}
            onAsk={onAsk}
          />
        </div>
      )
    case 'mission':
      return (
        <div className="view view--flush">
          <MissionControl />
        </div>
      )
    case 'inspector':
      return (
        <div className="view view--flush">
          <GraphInspectorView />
        </div>
      )
    case 'workspace':
      return (
        <div className="view view--flush">
          <WorkspaceView />
        </div>
      )
    case 'goals-browser':
      return (
        <div className="view view--flush">
          <GoalsBrowserView />
        </div>
      )
    case 'swarm':
      return (
        <div className="view view--flush">
          <SwarmMap />
        </div>
      )
    case 'werte':
      return (
        <div className="view">
          <div className="view__header">
            <div>
              <h1 className="view__title">Werte-Radar</h1>
              <div className="view__subtitle">
                Der Guardian-Agent bewertet jede Aktion gegen fünf Kernwerte.
              </div>
            </div>
          </div>
          <ValuesRadar />
        </div>
      )
    case 'chat':
      return (
        <div className="view view--flush">
          <ChatView
            pendingPrompt={pendingPrompt}
            pendingForceGoal={pendingForceGoal}
            onPendingConsumed={onPendingConsumed}
          />
        </div>
      )
    case 'neo':
      return <NeoShell />
    case 'agents':
      return (
        <div className="view">
          <AgentsView />
        </div>
      )
    case 'goals':
      return (
        <div className="view">
          <GoalsView />
        </div>
      )
    case 'consciousness':
      return (
        <div className="view">
          <ConsciousnessView />
        </div>
      )
    case 'provider':
      return (
        <div className="view">
          <ProviderView />
        </div>
      )
    case 'evolution':
      return (
        <div className="view">
          <EvolutionView />
        </div>
      )
    case 'graphs':
      return (
        <div className="view">
          <GraphsView />
        </div>
      )
    case 'knowledge-3d':
      return (
        <div className="view view--flush">
          <KnowledgeGraph3DView />
        </div>
      )
    case 'messages':
      return (
        <div className="view">
          <MessagesView />
        </div>
      )
    case 'training':
      return (
        <div className="view">
          <TrainingView />
        </div>
      )
    case 'gpu-training':
      return (
        <div className="view">
          <GpuTrainingView />
        </div>
      )
    case 'spiking':
      return (
        <div className="view">
          <SpikingView />
        </div>
      )
    case 'organism':
      return (
        <div className="view">
          <OrganismView />
        </div>
      )
    case 'historie':
      return (
        <div className="view">
          <HistorieView onNavigate={(t) => setActiveTab(t as Tab)} />
        </div>
      )
    default:
      return (
        <div className="view simple-view">
          <div className="simple-view__empty">
            Diese Ansicht existiert noch nicht.
          </div>
        </div>
      )
  }
}
