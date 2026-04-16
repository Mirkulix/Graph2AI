// GoalsBrowserView.tsx — Split-view browser for every goal the orchestrator
// has run. Left pane: list with status chips and relative ages. Right pane:
// detail view with subtasks, final result (markdown-rendered), and a
// "Weitermachen"-box to extend a goal with a follow-up instruction.
//
// Styling cues come from GraphInspectorView.tsx (inspector__list /
// inspector__detail split-pane conventions) but we use NEW class names
// prefixed with `goals-browser__` so the host project can restyle later.
// A tiny inline <style> block keeps the view usable until proper styles
// land in styles.css.
//
// The component polls /api/goals every 6 s and is self-contained (no props).

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Circle,
  ListTodo,
  Loader2,
  Send,
} from 'lucide-react'

type GoalStatus = 'Pending' | 'InProgress' | 'Completed' | 'Failed'

interface SubTask {
  description: string
  assigned_to: string
  status: GoalStatus
  result?: string
  duration_ms?: number
}

interface Goal {
  id: number
  description: string
  status: GoalStatus
  subtasks: SubTask[]
  created_at: number
  result?: string
  execution_graph?: unknown
}

interface ContinueResponse {
  goal_id?: number
  status?: string
  message?: string
}

const POLL_INTERVAL_MS = 6000
const DESCRIPTION_TRUNCATE = 60

export default function GoalsBrowserView() {
  const [goals, setGoals] = useState<Goal[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const showToast = useCallback((msg: string) => {
    setToast(msg)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 4000)
  }, [])

  const fetchGoals = useCallback(async () => {
    try {
      const r = await fetch('/api/goals')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const body = (await r.json()) as Goal[]
      setGoals(Array.isArray(body) ? body : [])
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchGoals()
    const iv = setInterval(() => void fetchGoals(), POLL_INTERVAL_MS)
    return () => {
      clearInterval(iv)
      if (toastTimer.current) clearTimeout(toastTimer.current)
    }
  }, [fetchGoals])

  // Auto-select newest goal once the list has arrived.
  useEffect(() => {
    if (selectedId === null && goals.length > 0) {
      setSelectedId(goals[0].id)
    }
  }, [goals, selectedId])

  const selected = useMemo(
    () => goals.find((g) => g.id === selectedId) ?? null,
    [goals, selectedId],
  )

  const handleContinue = useCallback(
    async (goalId: number, followUp: string) => {
      const trimmed = followUp.trim()
      if (!trimmed) return
      try {
        const r = await fetch(`/api/goals/${goalId}/continue`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ follow_up: trimmed }),
        })
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const body = (await r.json()) as ContinueResponse
        showToast(body.message ?? body.status ?? 'Fortsetzung gestartet.')
        // Refresh eagerly so the new subtask appears without waiting for
        // the next 6 s tick.
        void fetchGoals()
      } catch (e) {
        showToast(
          `Fehler: ${e instanceof Error ? e.message : String(e)}`,
        )
      }
    },
    [fetchGoals, showToast],
  )

  return (
    <div className="goals-browser">
      <aside className="goals-browser__list">
        <div className="goals-browser__list-header">
          <ListTodo size={14} />
          <span>Ziele ({goals.length})</span>
          <button type="button" onClick={() => void fetchGoals()}>
            Neu laden
          </button>
        </div>

        {loading && goals.length === 0 ? (
          <div className="goals-browser__empty">Lade …</div>
        ) : error && goals.length === 0 ? (
          <div className="goals-browser__empty goals-browser__empty--error">
            {error}
          </div>
        ) : goals.length === 0 ? (
          <div className="goals-browser__empty">
            Noch keine Ziele. Starte eines über Home oder Chat.
          </div>
        ) : (
          <ul>
            {goals.map((g) => (
              <li
                key={g.id}
                className={selectedId === g.id ? 'is-selected' : undefined}
              >
                <button type="button" onClick={() => setSelectedId(g.id)}>
                  <div className="goals-browser__list-row">
                    <StatusChip status={g.status} />
                    <span className="goals-browser__list-id">#{g.id}</span>
                    <span className="goals-browser__list-age">
                      {formatRelativeAge(g.created_at)}
                    </span>
                  </div>
                  <div className="goals-browser__list-desc">
                    {truncate(g.description, DESCRIPTION_TRUNCATE)}
                  </div>
                  <div className="goals-browser__list-meta">
                    {g.subtasks.length} Subtask
                    {g.subtasks.length === 1 ? '' : 's'}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <section className="goals-browser__detail">
        {selected ? (
          <GoalDetailPane goal={selected} onContinue={handleContinue} />
        ) : goals.length === 0 ? (
          <div className="goals-browser__empty">
            Noch keine Ziele. Starte eines über Home oder Chat.
          </div>
        ) : (
          <div className="goals-browser__empty">
            Wähle links ein Ziel, um Details zu sehen.
          </div>
        )}
      </section>

      {toast ? <div className="goals-browser__toast">{toast}</div> : null}

      <style>{STYLE}</style>
    </div>
  )
}

interface DetailProps {
  goal: Goal
  onContinue: (id: number, followUp: string) => void | Promise<void>
}

function GoalDetailPane({ goal, onContinue }: DetailProps) {
  const [followUp, setFollowUp] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    const trimmed = followUp.trim()
    if (!trimmed || submitting) return
    setSubmitting(true)
    try {
      await onContinue(goal.id, trimmed)
      setFollowUp('')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <header className="goals-browser__detail-header">
        <div className="goals-browser__detail-title">
          <StatusChip status={goal.status} />
          <h2>Ziel #{goal.id}</h2>
        </div>
        <p className="goals-browser__detail-desc">{goal.description}</p>
        <div className="goals-browser__detail-sub">
          <span>{formatTimestamp(goal.created_at)}</span>
          <span>·</span>
          <span>
            {goal.subtasks.length} Subtask
            {goal.subtasks.length === 1 ? '' : 's'}
          </span>
        </div>
      </header>

      {goal.result ? (
        <section className="goals-browser__panel">
          <h3>Ergebnis</h3>
          <div className="goals-browser__markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {goal.result}
            </ReactMarkdown>
          </div>
        </section>
      ) : null}

      <section className="goals-browser__panel">
        <h3>Teilaufgaben</h3>
        {goal.subtasks.length === 0 ? (
          <p className="goals-browser__muted">Keine Teilaufgaben.</p>
        ) : (
          <ul className="goals-browser__subtasks">
            {goal.subtasks.map((s, i) => (
              <SubTaskRow key={`${goal.id}-${i}`} sub={s} />
            ))}
          </ul>
        )}
      </section>

      <section className="goals-browser__panel goals-browser__continue">
        <h3>
          <ArrowRight size={14} /> Weitermachen
        </h3>
        <p className="goals-browser__muted">
          Ergänze das Ziel um eine Folgeanweisung. Der Orchestrator hängt daraus
          neue Teilaufgaben an.
        </p>
        <textarea
          className="goals-browser__textarea"
          value={followUp}
          onChange={(e) => setFollowUp(e.target.value)}
          placeholder="z. B. „Bitte das Ergebnis auf Deutsch zusammenfassen."
          rows={3}
          disabled={submitting}
        />
        <div className="goals-browser__continue-actions">
          <button
            type="button"
            className="goals-browser__btn"
            onClick={() => void handleSubmit()}
            disabled={submitting || !followUp.trim()}
          >
            <Send size={12} />
            {submitting ? 'Sende …' : 'Fortsetzen'}
          </button>
        </div>
      </section>
    </>
  )
}

function SubTaskRow({ sub }: { sub: SubTask }) {
  const [expanded, setExpanded] = useState(false)
  const hasResult = Boolean(sub.result && sub.result.length > 0)
  return (
    <li className={`goals-browser__subtask goals-browser__subtask--${statusKey(sub.status)}`}>
      <div className="goals-browser__subtask-head">
        <StatusChip status={sub.status} />
        <span className="goals-browser__subtask-desc">{sub.description}</span>
      </div>
      <div className="goals-browser__subtask-meta">
        <span className="goals-browser__agent-chip">{sub.assigned_to}</span>
        {typeof sub.duration_ms === 'number' ? (
          <span className="goals-browser__muted">
            {sub.duration_ms < 1000
              ? `${sub.duration_ms} ms`
              : `${(sub.duration_ms / 1000).toFixed(1)} s`}
          </span>
        ) : null}
        {hasResult ? (
          <button
            type="button"
            className="goals-browser__linklike"
            onClick={() => setExpanded((x) => !x)}
          >
            {expanded ? 'Ergebnis ausblenden' : 'Ergebnis anzeigen'}
          </button>
        ) : null}
      </div>
      {expanded && hasResult ? (
        <div className="goals-browser__subtask-result">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {sub.result ?? ''}
          </ReactMarkdown>
        </div>
      ) : null}
    </li>
  )
}

function StatusChip({ status }: { status: GoalStatus }) {
  const key = statusKey(status)
  const icon = statusIcon(status)
  return (
    <span
      className={`goals-browser__chip goals-browser__chip--${key}`}
      title={statusLabel(status)}
    >
      {icon}
      <span>{statusLabel(status)}</span>
    </span>
  )
}

function statusIcon(status: GoalStatus) {
  switch (status) {
    case 'Completed':
      return <CheckCircle2 size={12} />
    case 'Failed':
      return <AlertCircle size={12} />
    case 'InProgress':
      return <Loader2 size={12} className="goals-browser__spin" />
    default:
      return <Circle size={12} />
  }
}

function statusKey(status: GoalStatus): string {
  switch (status) {
    case 'Completed':
      return 'ok'
    case 'Failed':
      return 'err'
    case 'InProgress':
      return 'run'
    default:
      return 'pending'
  }
}

function statusLabel(status: GoalStatus): string {
  switch (status) {
    case 'Completed':
      return 'Erledigt'
    case 'Failed':
      return 'Fehlgeschlagen'
    case 'InProgress':
      return 'In Arbeit'
    default:
      return 'Ausstehend'
  }
}

function truncate(s: string, max: number): string {
  if (!s) return ''
  if (s.length <= max) return s
  return s.slice(0, max - 1) + '…'
}

function formatTimestamp(unix: number): string {
  if (!unix || !Number.isFinite(unix)) return '—'
  const d = new Date(unix * 1000)
  return d.toLocaleString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatRelativeAge(unix: number): string {
  if (!unix || !Number.isFinite(unix)) return '—'
  const nowSec = Date.now() / 1000
  const deltaSec = Math.max(0, nowSec - unix)
  if (deltaSec < 60) return 'gerade eben'
  const mins = Math.floor(deltaSec / 60)
  if (mins < 60) return `vor ${mins} min`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `vor ${hours} h`
  const days = Math.floor(hours / 24)
  if (days < 30) return `vor ${days} d`
  const months = Math.floor(days / 30)
  if (months < 12) return `vor ${months} mo`
  const years = Math.floor(days / 365)
  return `vor ${years} y`
}

/* ─────────────────────────── inline styles ─────────────────────────── */

const STYLE = `
.goals-browser {
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: 0;
  height: 100%;
  width: 100%;
  background: var(--bg);
  color: var(--text);
  position: relative;
}

.goals-browser__list {
  border-right: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.goals-browser__list-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.goals-browser__list-header > span { flex: 1; }

.goals-browser__list-header button {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius-button);
  color: var(--text-muted);
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.goals-browser__list-header button:hover {
  border-color: var(--border-strong);
  color: var(--text);
}

.goals-browser__list ul {
  list-style: none;
  margin: 0;
  padding: 6px;
  overflow-y: auto;
  flex: 1;
  min-height: 0;
}

.goals-browser__list li { margin: 0; }

.goals-browser__list li button {
  width: 100%;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-button);
  padding: 8px 10px;
  text-align: left;
  cursor: pointer;
  color: inherit;
  display: block;
}

.goals-browser__list li button:hover {
  background: var(--surface-sunken);
}

.goals-browser__list li.is-selected button {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.goals-browser__list-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.goals-browser__list-id {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-muted);
}

.goals-browser__list-age {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-dim);
}

.goals-browser__list-desc {
  font-size: 13px;
  color: var(--text);
  line-height: 1.35;
  margin-bottom: 4px;
  word-break: break-word;
}

.goals-browser__list-meta {
  font-size: 11px;
  color: var(--text-dim);
}

.goals-browser__detail {
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
}

.goals-browser__detail-header {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

.goals-browser__detail-title {
  display: flex;
  align-items: center;
  gap: 10px;
}

.goals-browser__detail-title h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.goals-browser__detail-desc {
  margin: 0;
  font-size: 14px;
  color: var(--text);
  line-height: 1.4;
}

.goals-browser__detail-sub {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
}

.goals-browser__panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 14px 16px;
}

.goals-browser__panel h3 {
  margin: 0 0 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.goals-browser__markdown {
  font-size: 13px;
  line-height: 1.5;
  color: var(--text);
}

.goals-browser__markdown p { margin: 0 0 8px; }
.goals-browser__markdown p:last-child { margin-bottom: 0; }
.goals-browser__markdown code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  background: var(--surface-sunken);
  padding: 1px 4px;
  border-radius: 4px;
}
.goals-browser__markdown pre {
  background: var(--surface-sunken);
  padding: 10px;
  border-radius: var(--radius-button);
  overflow-x: auto;
}

.goals-browser__subtasks {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.goals-browser__subtask {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-button);
  background: var(--surface-sunken);
}

.goals-browser__subtask--run { border-left: 3px solid var(--accent); }
.goals-browser__subtask--ok { border-left: 3px solid var(--ok); }
.goals-browser__subtask--err { border-left: 3px solid var(--err); }
.goals-browser__subtask--pending { border-left: 3px solid var(--border-strong); }

.goals-browser__subtask-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.goals-browser__subtask-desc {
  font-size: 13px;
  color: var(--text);
  flex: 1;
  word-break: break-word;
}

.goals-browser__subtask-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 11px;
}

.goals-browser__agent-chip {
  background: var(--accent-soft);
  color: var(--accent-strong);
  border-radius: var(--radius-chip);
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 500;
}

.goals-browser__subtask-result {
  margin-top: 8px;
  padding: 8px 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-button);
  font-size: 12px;
  line-height: 1.5;
}

.goals-browser__linklike {
  background: transparent;
  border: none;
  padding: 0;
  color: var(--accent-strong);
  cursor: pointer;
  font-size: 11px;
  text-decoration: underline;
}

.goals-browser__continue h3 { color: var(--accent-strong); }

.goals-browser__textarea {
  width: 100%;
  background: var(--surface-sunken);
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  padding: 8px 10px;
  font: inherit;
  font-size: 13px;
  color: var(--text);
  resize: vertical;
  min-height: 64px;
  margin-top: 8px;
}

.goals-browser__textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.goals-browser__continue-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}

.goals-browser__btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--accent);
  color: var(--text-inverse);
  border: none;
  border-radius: var(--radius-button);
  padding: 8px 14px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
}

.goals-browser__btn:hover:not(:disabled) { background: var(--accent-strong); }
.goals-browser__btn:disabled { opacity: 0.6; cursor: not-allowed; }

.goals-browser__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border-radius: var(--radius-chip);
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 500;
  border: 1px solid transparent;
  white-space: nowrap;
}

.goals-browser__chip--ok { background: var(--ok-soft); color: var(--ok); }
.goals-browser__chip--err { background: var(--err-soft); color: var(--err); }
.goals-browser__chip--run { background: var(--accent-soft); color: var(--accent-strong); }
.goals-browser__chip--pending {
  background: var(--surface-sunken);
  color: var(--text-muted);
  border-color: var(--border);
}

.goals-browser__muted { color: var(--text-muted); font-size: 12px; margin: 0; }

.goals-browser__empty {
  padding: 24px;
  color: var(--text-muted);
  font-size: 13px;
  text-align: center;
}

.goals-browser__empty--error { color: var(--err); }

.goals-browser__toast {
  position: absolute;
  bottom: 16px;
  right: 16px;
  background: var(--surface-elevated);
  color: var(--text);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-button);
  padding: 10px 14px;
  font-size: 12px;
  box-shadow: var(--shadow-raised);
  max-width: 320px;
  z-index: 20;
}

.goals-browser__spin { animation: goals-browser-spin 1s linear infinite; }

@keyframes goals-browser-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
`
