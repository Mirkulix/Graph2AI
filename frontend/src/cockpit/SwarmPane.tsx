import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  api,
  type SwarmState,
  type SwarmRound,
  type SwarmSubtask,
  type SubtaskStatus,
  type SwarmStatus,
  type PresenceEntry,
} from '../lib/api';
import { SecondaryFrame } from './secondary/SecondaryFrame';
import AutonomousPanel from './AutonomousPanel';

const POLL_INTERVAL_MS = 2000;
const RESPONSE_INLINE_MAX = 200;

export default function SwarmPane() {
  const [active, setActive] = useState<SwarmState[]>([]);
  const [goal, setGoal] = useState('');
  const [maxRounds, setMaxRounds] = useState(3);
  const [maxTokens, setMaxTokens] = useState(20000);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const [eligibleIdes, setEligibleIdes] = useState<PresenceEntry[]>([]);
  const [targetIde, setTargetIde] = useState<string>('');
  // Broadcast (fan-out) state — separate from swarm flow.
  const [bcPrompt, setBcPrompt] = useState('');
  const [bcSelected, setBcSelected] = useState<Set<string>>(new Set());
  const [bcBusy, setBcBusy] = useState(false);
  const [bcFeedback, setBcFeedback] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const aliveRef = useRef(true);

  // ─── Load eligible IDE list once on mount ───────────────────────────
  useEffect(() => {
    let alive = true;
    api.presence()
      .then(list => {
        if (!alive) return;
        setEligibleIdes((list ?? []).filter(p => p.eligible_for_swarms !== false));
      })
      .catch(() => { if (alive) setEligibleIdes([]); });
    return () => { alive = false; };
  }, []);

  // ─── Poll active swarms every 2s ────────────────────────────────────
  useEffect(() => {
    aliveRef.current = true;
    const load = async () => {
      try {
        const list = await api.swarmActive();
        if (aliveRef.current) setActive(Array.isArray(list) ? list : []);
      } catch {
        // Quiet on errors; the cockpit should keep working even if swarm
        // backend is offline.
      }
    };
    void load();
    const t = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(t);
    };
  }, []);

  async function startSwarm() {
    const trimmed = goal.trim();
    if (!trimmed || busy) return;
    // Optionally prefix the goal so the strategist scopes the swarm to one IDE.
    const finalGoal = targetIde
      ? `[ Dispatch to IDE: ${targetIde} ] ${trimmed}`
      : trimmed;
    setBusy(true);
    setFeedback(null);
    try {
      const res = await api.swarmStart(finalGoal, { maxRounds, maxTokens });
      setFeedback({ kind: 'ok', text: `swarm #${res.swarm_id} started` });
      setGoal('');
      // Optimistic refresh — don't wait for poll
      try {
        const list = await api.swarmActive();
        setActive(Array.isArray(list) ? list : []);
      } catch {
        // ignore
      }
    } catch (e) {
      setFeedback({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      setTimeout(() => setFeedback(null), 4000);
    }
  }

  async function runBroadcast() {
    const trimmed = bcPrompt.trim();
    const targets = Array.from(bcSelected);
    if (!trimmed || targets.length === 0 || bcBusy) return;
    setBcBusy(true);
    setBcFeedback(null);
    try {
      const res = await api.broadcast(trimmed, targets);
      const failedCount = res.failures.length;
      const text = failedCount === 0
        ? `broadcast → ${res.sent} IDE${res.sent === 1 ? '' : 's'}`
        : `broadcast → ${res.sent} ok, ${failedCount} failed`;
      setBcFeedback({ kind: failedCount === 0 ? 'ok' : 'err', text });
      if (failedCount === 0) setBcPrompt('');
    } catch (e) {
      setBcFeedback({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBcBusy(false);
      setTimeout(() => setBcFeedback(null), 4000);
    }
  }

  async function stopSwarm(id: number) {
    try {
      await api.swarmStop(id);
      setFeedback({ kind: 'ok', text: `stop requested for #${id}` });
    } catch (e) {
      setFeedback({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setTimeout(() => setFeedback(null), 4000);
    }
  }

  return (
    <SecondaryFrame title="swarm" subtitle="autonomous multi-agent runs — strategist plans, agents execute, evaluator decides">
      <AutonomousPanel />

      <div style={{
        margin: '24px 0',
        height: 1,
        background: 'var(--rule-faint)',
        position: 'relative',
      }}>
        <span style={{
          position: 'absolute',
          top: -9,
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '0 12px',
          background: 'var(--bg-void)',
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--ink-faint)',
          letterSpacing: 0.08,
          textTransform: 'uppercase',
        }}>
          manual
        </span>
      </div>

      <SwarmStarter
        goal={goal}
        setGoal={setGoal}
        maxRounds={maxRounds}
        setMaxRounds={setMaxRounds}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
        busy={busy}
        feedback={feedback}
        onStart={() => void startSwarm()}
        eligibleIdes={eligibleIdes}
        targetIde={targetIde}
        setTargetIde={setTargetIde}
      />

      <BroadcastStarter
        prompt={bcPrompt}
        setPrompt={setBcPrompt}
        eligibleIdes={eligibleIdes}
        selected={bcSelected}
        setSelected={setBcSelected}
        busy={bcBusy}
        feedback={bcFeedback}
        onSend={() => void runBroadcast()}
      />

      <h3 className="eyebrow" style={{ margin: '24px 0 10px' }}>
        active swarms {active.length > 0 && <span style={{ color: 'var(--ink-faint)' }}>· {active.length}</span>}
      </h3>

      {active.length === 0 ? (
        <EmptyActive />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {active.map(s => (
            <SwarmCard key={s.id} swarm={s} onStop={() => void stopSwarm(s.id)} maxTokens={maxTokens} />
          ))}
        </div>
      )}

      <style>{spinKeyframes}</style>
    </SecondaryFrame>
  );
}

// ─── Starter card ───────────────────────────────────────────────────

interface StarterProps {
  goal: string;
  setGoal: (s: string) => void;
  maxRounds: number;
  setMaxRounds: (n: number) => void;
  maxTokens: number;
  setMaxTokens: (n: number) => void;
  busy: boolean;
  feedback: { kind: 'ok' | 'err'; text: string } | null;
  onStart: () => void;
  eligibleIdes: PresenceEntry[];
  targetIde: string;
  setTargetIde: (s: string) => void;
}

function SwarmStarter(p: StarterProps) {
  const canStart = !p.busy && p.goal.trim().length > 0;
  return (
    <section style={starterStyle} aria-label="start a swarm">
      <div className="eyebrow" style={{ marginBottom: 10 }}>start a swarm</div>

      <textarea
        value={p.goal}
        onChange={e => p.setGoal(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (canStart) p.onStart();
          }
        }}
        placeholder="Goal — e.g. Refactor consensus.rs into smaller modules and add doctests…"
        rows={3}
        style={{
          width: '100%',
          background: 'var(--bg-panel)',
          border: '1px solid var(--rule-default)',
          borderRadius: 4,
          color: 'var(--ink-bright)',
          fontFamily: 'var(--font-ui)',
          fontSize: 13,
          padding: '10px 12px',
          resize: 'vertical',
          minHeight: 64,
          lineHeight: 1.5,
        }}
      />

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 10, flexWrap: 'wrap' }}>
        <label style={labelStyle}>
          <span className="eyebrow">max rounds</span>
          <input
            type="number"
            min={1}
            max={20}
            value={p.maxRounds}
            onChange={e => p.setMaxRounds(clampInt(e.target.value, 1, 20, 3))}
            style={numberInputStyle}
          />
        </label>
        <label style={labelStyle}>
          <span className="eyebrow">max tokens</span>
          <input
            type="number"
            min={1000}
            max={1_000_000}
            step={1000}
            value={p.maxTokens}
            onChange={e => p.setMaxTokens(clampInt(e.target.value, 1000, 1_000_000, 20000))}
            style={{ ...numberInputStyle, width: 110 }}
          />
        </label>
        <label style={labelStyle}>
          <span className="eyebrow">target ide</span>
          <select
            value={p.targetIde}
            onChange={e => p.setTargetIde(e.target.value)}
            style={selectInputStyle}
            title="Optionally scope the swarm to a single IDE — adds a dispatch hint to the goal."
          >
            <option value="">(any agent)</option>
            {p.eligibleIdes.map(ide => (
              <option key={ide.identity} value={ide.identity}>
                {ide.identity}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={p.onStart}
          disabled={!canStart}
          className="surface-button surface-button--primary"
          style={{
            marginLeft: 'auto',
            padding: '8px 16px',
            opacity: canStart ? 1 : 0.4,
            cursor: canStart ? 'pointer' : 'not-allowed',
          }}
        >
          {p.busy ? 'starting…' : 'START SWARM'}
        </button>
      </div>

      {p.feedback && (
        <div style={{
          marginTop: 10,
          padding: '6px 10px',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          color: p.feedback.kind === 'ok' ? 'var(--verified-bright)' : 'var(--alert-bright)',
          background: p.feedback.kind === 'ok' ? 'var(--verified-soft)' : 'var(--alert-soft)',
          borderRadius: 4,
        }}>
          {p.feedback.text}
        </div>
      )}
    </section>
  );
}

// ─── Broadcast starter (mesh fan-out) ───────────────────────────────

interface BroadcastProps {
  prompt: string;
  setPrompt: (s: string) => void;
  eligibleIdes: PresenceEntry[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  busy: boolean;
  feedback: { kind: 'ok' | 'err'; text: string } | null;
  onSend: () => void;
}

function BroadcastStarter(p: BroadcastProps) {
  const canSend = !p.busy && p.prompt.trim().length > 0 && p.selected.size > 0;
  const allSelected = p.eligibleIdes.length > 0 && p.selected.size === p.eligibleIdes.length;

  function toggle(identity: string) {
    const next = new Set(p.selected);
    if (next.has(identity)) next.delete(identity);
    else next.add(identity);
    p.setSelected(next);
  }
  function selectAll() {
    p.setSelected(new Set(p.eligibleIdes.map(e => e.identity)));
  }
  function selectNone() {
    p.setSelected(new Set());
  }

  return (
    <section style={{ ...starterStyle, marginTop: 16 }} aria-label="broadcast to selected IDEs">
      <div className="eyebrow" style={{ marginBottom: 10 }}>
        broadcast — fan out one prompt to selected IDEs
      </div>

      <textarea
        value={p.prompt}
        onChange={e => p.setPrompt(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (canSend) p.onSend();
          }
        }}
        placeholder="Prompt — every selected IDE receives this and (with auto-respond enabled) starts working autonomously…"
        rows={3}
        style={{
          width: '100%',
          background: 'var(--bg-panel)',
          border: '1px solid var(--rule-default)',
          borderRadius: 4,
          color: 'var(--ink-bright)',
          fontFamily: 'var(--font-ui)',
          fontSize: 13,
          padding: '10px 12px',
          resize: 'vertical',
          minHeight: 64,
          lineHeight: 1.5,
        }}
      />

      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
        <span className="eyebrow">target IDEs · {p.selected.size}/{p.eligibleIdes.length}</span>
        <button
          type="button"
          onClick={allSelected ? selectNone : selectAll}
          disabled={p.eligibleIdes.length === 0}
          className="surface-button"
          style={{ padding: '4px 10px', fontSize: 11 }}
        >
          {allSelected ? 'select none' : 'select all'}
        </button>
      </div>

      {p.eligibleIdes.length === 0 ? (
        <div style={{
          marginTop: 8,
          padding: '8px 10px',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          color: 'var(--ink-faint)',
          background: 'var(--bg-raised)',
          border: '1px dashed var(--rule-faint)',
          borderRadius: 4,
        }}>
          no eligible IDEs registered. Open a project in VS Code with the QLang extension to register one.
        </div>
      ) : (
        <div style={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          maxHeight: 200,
          overflowY: 'auto',
          padding: '8px 10px',
          background: 'var(--bg-raised)',
          border: '1px solid var(--rule-faint)',
          borderRadius: 4,
        }}>
          {p.eligibleIdes.map(ide => {
            const checked = p.selected.has(ide.identity);
            return (
              <label
                key={ide.identity}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '4px 6px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 12,
                  color: 'var(--ink-bright)',
                  cursor: 'pointer',
                  borderRadius: 3,
                  background: checked ? 'var(--cta-soft)' : 'transparent',
                }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggle(ide.identity)}
                  style={{ margin: 0 }}
                />
                <span style={{ fontWeight: 600 }}>{ide.identity}</span>
                {ide.ide_name && (
                  <span style={{ color: 'var(--ink-faint)' }}>· {ide.ide_name}</span>
                )}
                {ide.host && (
                  <span style={{ color: 'var(--ink-faint)' }}>· {ide.host}</span>
                )}
              </label>
            );
          })}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', marginTop: 10 }}>
        <button
          type="button"
          onClick={p.onSend}
          disabled={!canSend}
          className="surface-button surface-button--primary"
          style={{
            marginLeft: 'auto',
            padding: '8px 16px',
            opacity: canSend ? 1 : 0.4,
            cursor: canSend ? 'pointer' : 'not-allowed',
          }}
        >
          {p.busy ? 'sending…' : `BROADCAST → ${p.selected.size}`}
        </button>
      </div>

      {p.feedback && (
        <div style={{
          marginTop: 10,
          padding: '6px 10px',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          color: p.feedback.kind === 'ok' ? 'var(--verified-bright)' : 'var(--alert-bright)',
          background: p.feedback.kind === 'ok' ? 'var(--verified-soft)' : 'var(--alert-soft)',
          borderRadius: 4,
        }}>
          {p.feedback.text}
        </div>
      )}
    </section>
  );
}

// ─── One swarm card ─────────────────────────────────────────────────

function SwarmCard({ swarm, onStop, maxTokens }: { swarm: SwarmState; onStop: () => void; maxTokens: number }) {
  const tokenBudget = useMemo(() => {
    // Use the bigger of the local maxTokens or whatever the swarm itself
    // reports as its ceiling (we don't get the swarm's max back, so fall
    // back to the starter's max).
    return Math.max(maxTokens, 1);
  }, [maxTokens]);
  const usedRatio = swarm.tokens_used / tokenBudget;
  const costMeterColor =
    usedRatio < 0.5 ? 'var(--ok)' :
    usedRatio < 0.9 ? 'var(--warn)' :
                       'var(--danger)';
  const meterPct = Math.min(1, Math.max(0, usedRatio)) * 100;
  const isTerminal = swarm.status === 'done' || swarm.status === 'stopped' || swarm.status === 'error';

  return (
    <article style={swarmCardStyle}>
      <header style={swarmHeaderStyle}>
        <span style={swarmIdStyle}>#{swarm.id}</span>
        <span style={statusChipStyle(swarm.status)}>{swarm.status}</span>
        {swarm.stop_requested && !isTerminal && (
          <span className="chip chip-warn" style={{ fontSize: 10 }}>stop requested</span>
        )}
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--ink-faint)',
          marginLeft: 'auto',
        }}>
          started {formatRelative(swarm.started_at)}
          {swarm.finished_at != null && ` · ended ${formatRelative(swarm.finished_at)}`}
        </span>
      </header>

      <div style={goalLineStyle}>
        <span className="eyebrow" style={{ color: 'var(--ink-faint)', marginRight: 6 }}>goal</span>
        <span style={{ color: 'var(--ink-bright)', fontSize: 13 }}>{swarm.goal}</span>
      </div>

      <div style={meterRowStyle}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={meterTrackStyle}>
            <div style={{
              width: `${meterPct}%`,
              height: '100%',
              background: costMeterColor,
              transition: 'width 320ms var(--ease)',
            }} />
          </div>
          <div style={meterLabelRowStyle}>
            <span>
              <span style={{ color: 'var(--ink-bright)', fontWeight: 600 }}>{formatNum(swarm.tokens_used)}</span>
              <span style={{ color: 'var(--ink-faint)' }}> / {formatNum(tokenBudget)} tokens</span>
            </span>
            <span style={{ color: costMeterColor, fontWeight: 600 }}>
              ${swarm.cost_usd.toFixed(4)}
            </span>
          </div>
        </div>

        {!isTerminal && (
          <button
            type="button"
            onClick={onStop}
            disabled={swarm.stop_requested}
            style={stopButtonStyle(swarm.stop_requested)}
            title={swarm.stop_requested ? 'stop already requested' : 'stop this swarm'}
          >
            <span style={{ display: 'inline-block', width: 8, height: 8, background: 'currentColor' }} />
            STOP
          </button>
        )}
      </div>

      {swarm.rounds.length === 0 ? (
        <div style={{
          marginTop: 14,
          padding: '12px 14px',
          fontSize: 12,
          fontFamily: 'var(--font-mono)',
          color: 'var(--ink-faint)',
          background: 'var(--bg-raised)',
          border: '1px dashed var(--rule-faint)',
          borderRadius: 4,
        }}>
          {swarm.status === 'planning' ? 'strategist is planning…' : 'no rounds yet'}
        </div>
      ) : (
        <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {swarm.rounds.map(r => (
            <RoundView key={r.round_number} round={r} />
          ))}
        </div>
      )}
    </article>
  );
}

// ─── One round (plan + subtasks + eval) ─────────────────────────────

function RoundView({ round }: { round: SwarmRound }) {
  return (
    <section style={roundStyle} aria-label={`round ${round.round_number}`}>
      <header style={roundHeaderStyle}>
        <span style={roundLabelStyle}>ROUND {round.round_number}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-faint)' }}>
          {round.subtasks.length} subtask{round.subtasks.length === 1 ? '' : 's'}
        </span>
      </header>

      {round.plan && (
        <div style={planBlockStyle}>
          <div className="eyebrow" style={{ marginBottom: 4 }}>plan</div>
          <div className="md-reply" style={{ fontSize: 12, lineHeight: 1.55, color: 'var(--ink-bright)' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{round.plan}</ReactMarkdown>
          </div>
        </div>
      )}

      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {round.subtasks.map(st => (
          <SubtaskView key={st.id} subtask={st} />
        ))}
      </div>

      {round.eval && round.eval.trim().length > 0 && (
        <div style={evalBlockStyle}>
          <div className="eyebrow" style={{ marginBottom: 4, color: 'var(--cta)' }}>evaluation</div>
          <div className="md-reply" style={{ fontSize: 12, lineHeight: 1.55, color: 'var(--ink-bright)' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{round.eval}</ReactMarkdown>
          </div>
        </div>
      )}
    </section>
  );
}

// ─── One subtask row ────────────────────────────────────────────────

function SubtaskView({ subtask }: { subtask: SwarmSubtask }) {
  const response = subtask.response ?? '';
  const inline = response.length > 0 && response.length <= RESPONSE_INLINE_MAX;
  const expandable = response.length > RESPONSE_INLINE_MAX;
  return (
    <div style={subtaskRowStyle}>
      <div style={subtaskHeaderStyle}>
        <StatusIcon status={subtask.status} />
        <span style={subtaskIdStyle}>{subtask.id}</span>
        <span style={subtaskAgentStyle}>{subtask.assigned_to}</span>
        {subtask.started_at != null && subtask.finished_at != null && (
          <span style={{
            marginLeft: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-faint)',
          }}>
            {(subtask.finished_at - subtask.started_at).toFixed(1)}s
          </span>
        )}
      </div>

      {subtask.prompt && (
        <pre style={subtaskPromptStyle}>{truncate(subtask.prompt, 240)}</pre>
      )}

      {inline && (
        <div style={subtaskResponseStyle}>
          <div className="md-reply" style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--ink-bright)' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
          </div>
        </div>
      )}

      {expandable && (
        <details style={{ marginTop: 6 }}>
          <summary style={{
            cursor: 'pointer',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-muted)',
            letterSpacing: 0.04,
            textTransform: 'uppercase',
          }}>
            response · {response.length} chars
          </summary>
          <div style={{ ...subtaskResponseStyle, marginTop: 6 }}>
            <div className="md-reply" style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--ink-bright)' }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
            </div>
          </div>
        </details>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: SubtaskStatus }) {
  switch (status) {
    case 'running':
      return (
        <span
          aria-label="running"
          title="running"
          style={{
            display: 'inline-block',
            width: 12,
            height: 12,
            border: '2px solid var(--cta-soft)',
            borderTopColor: 'var(--cta)',
            borderRadius: '50%',
            animation: 'swarm-spin 900ms linear infinite',
          }}
        />
      );
    case 'done':
      return (
        <span aria-label="done" title="done" style={{ color: 'var(--ok)', fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, lineHeight: 1, width: 12, textAlign: 'center' }}>
          ✓
        </span>
      );
    case 'error':
      return (
        <span aria-label="error" title="error" style={{ color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, lineHeight: 1, width: 12, textAlign: 'center' }}>
          ✗
        </span>
      );
    case 'pending':
    default:
      return (
        <span aria-label="pending" title="pending" style={{ color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, lineHeight: 1, width: 12, textAlign: 'center' }}>
          ·
        </span>
      );
  }
}

// ─── Helpers ────────────────────────────────────────────────────────

function EmptyActive() {
  return (
    <div style={{
      padding: '20px 24px',
      background: 'var(--bg-panel)',
      border: '1px dashed var(--rule-default)',
      borderRadius: 6,
      textAlign: 'center',
      fontSize: 12,
      color: 'var(--ink-muted)',
    }}>
      No active swarms. Set a goal above and press <strong>START SWARM</strong>.
    </div>
  );
}

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + '…';
}

function formatNum(n: number): string {
  return n.toLocaleString('en-US');
}

function formatRelative(unixSec: number): string {
  const now = Date.now() / 1000;
  const diff = Math.max(0, now - unixSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function statusChipStyle(status: SwarmStatus): React.CSSProperties {
  const palette: Record<SwarmStatus, { bg: string; fg: string }> = {
    planning:    { bg: 'var(--cta-soft)',     fg: 'var(--cta)' },
    dispatching: { bg: 'var(--cta-soft)',     fg: 'var(--cta)' },
    evaluating:  { bg: 'var(--warn-soft)',    fg: 'var(--warn)' },
    done:        { bg: 'var(--ok-soft)',      fg: 'var(--ok)' },
    stopped:     { bg: 'var(--bg-raised)',    fg: 'var(--ink-muted)' },
    error:       { bg: 'var(--danger-soft)',  fg: 'var(--danger)' },
  };
  const { bg, fg } = palette[status];
  return {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '2px 8px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 600,
    color: fg,
    background: bg,
    borderRadius: 999,
    letterSpacing: 0.05,
    textTransform: 'uppercase',
  };
}

// ─── Inline keyframes (scoped so we don't pollute tokens.css) ──────

const spinKeyframes = `
@keyframes swarm-spin {
  to { transform: rotate(360deg); }
}
`;

// ─── Styles ─────────────────────────────────────────────────────────

const starterStyle: React.CSSProperties = {
  padding: '16px 18px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 6,
};

const labelStyle: React.CSSProperties = {
  display: 'inline-flex',
  flexDirection: 'column',
  gap: 4,
};

const numberInputStyle: React.CSSProperties = {
  width: 80,
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  color: 'var(--ink-bright)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  padding: '6px 8px',
};

const selectInputStyle: React.CSSProperties = {
  minWidth: 160,
  maxWidth: 240,
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  color: 'var(--ink-bright)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  padding: '6px 8px',
};

const swarmCardStyle: React.CSSProperties = {
  padding: '14px 16px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderLeft: '3px solid var(--accent)',
  borderRadius: 6,
};

const swarmHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
};

const swarmIdStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 13,
  fontWeight: 600,
  color: 'var(--ink-bright)',
  letterSpacing: 0.02,
};

const goalLineStyle: React.CSSProperties = {
  marginTop: 8,
  display: 'flex',
  alignItems: 'baseline',
  gap: 4,
  flexWrap: 'wrap',
};

const meterRowStyle: React.CSSProperties = {
  marginTop: 12,
  display: 'flex',
  gap: 12,
  alignItems: 'center',
};

const meterTrackStyle: React.CSSProperties = {
  width: '100%',
  height: 6,
  background: 'var(--bg-raised)',
  borderRadius: 999,
  overflow: 'hidden',
  border: '1px solid var(--rule-faint)',
};

const meterLabelRowStyle: React.CSSProperties = {
  marginTop: 4,
  display: 'flex',
  justifyContent: 'space-between',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--ink-muted)',
};

function stopButtonStyle(disabled: boolean): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 0.06,
    color: 'white',
    background: 'var(--danger)',
    border: '1px solid var(--danger)',
    borderRadius: 4,
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.4 : 1,
    flexShrink: 0,
  };
}

const roundStyle: React.CSSProperties = {
  padding: '12px 14px',
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 5,
};

const roundHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  marginBottom: 8,
};

const roundLabelStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  fontWeight: 700,
  color: 'var(--ink-bright)',
  letterSpacing: 0.06,
};

const planBlockStyle: React.CSSProperties = {
  padding: '8px 10px',
  borderLeft: '2px solid var(--rule-default)',
  background: 'var(--bg-panel)',
  borderRadius: '0 4px 4px 0',
};

const evalBlockStyle: React.CSSProperties = {
  marginTop: 10,
  padding: '8px 10px',
  borderLeft: '3px solid var(--cta)',
  background: 'var(--cta-soft)',
  borderRadius: '0 4px 4px 0',
};

const subtaskRowStyle: React.CSSProperties = {
  padding: '8px 10px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 4,
};

const subtaskHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
};

const subtaskIdStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  fontWeight: 600,
  color: 'var(--ink-muted)',
  padding: '1px 6px',
  background: 'var(--bg-raised)',
  borderRadius: 2,
};

const subtaskAgentStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  fontWeight: 600,
  color: 'var(--ink-bright)',
};

const subtaskPromptStyle: React.CSSProperties = {
  margin: '6px 0 0',
  padding: '6px 8px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.5,
  color: 'var(--ink-muted)',
  background: 'var(--bg-raised)',
  borderRadius: 3,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
};

const subtaskResponseStyle: React.CSSProperties = {
  marginTop: 6,
  padding: '6px 10px',
  borderLeft: '2px solid var(--ok)',
  background: 'var(--ok-soft)',
  borderRadius: '0 3px 3px 0',
};
