import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type AutonomousConfig, type AutonomousState, type AutonomousRun } from '../lib/api';

const STATUS_POLL_MS = 5000;
const HOT_UPDATE_DEBOUNCE_MS = 200;

const DEFAULT_CONFIG: AutonomousConfig = {
  interval_seconds: 1800,        // 30 min
  daily_budget_usd: 0.5,
  max_swarms_per_hour: 4,
  goals_queue: [],
  meta_agent_enabled: true,
  swarm_max_rounds: 2,
  swarm_max_tokens: 20000,
};

export default function AutonomousPanel() {
  const [state, setState] = useState<AutonomousState | null>(null);
  const [config, setConfig] = useState<AutonomousConfig>(DEFAULT_CONFIG);
  const [queueText, setQueueText] = useState<string>('');
  const [queueDirty, setQueueDirty] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const [now, setNow] = useState(() => Math.floor(Date.now() / 1000));
  const aliveRef = useRef(true);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Poll status every 5s
  useEffect(() => {
    aliveRef.current = true;
    const load = async () => {
      try {
        const s = await api.autonomousStatus();
        if (!aliveRef.current) return;
        setState(s);
        // Sync local config from server (but only if not mid-edit of queue text)
        setConfig(prev => ({ ...prev, ...s.config }));
        setQueueText(prevText => {
          if (queueDirty) return prevText;
          return (s.config.goals_queue ?? []).join('\n');
        });
      } catch {
        // Quiet — backend may be offline; keep prior state
      }
    };
    void load();
    const t = setInterval(load, STATUS_POLL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(t);
    };
    // queueDirty intentionally not in deps — load is stable enough
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queueDirty]);

  // 1Hz tick for the countdown only
  useEffect(() => {
    const t = setInterval(() => setNow(Math.floor(Date.now() / 1000)), 1000);
    return () => clearInterval(t);
  }, []);

  // Debounced hot-update: when enabled and config changes, push to backend
  function scheduleHotUpdate(next: AutonomousConfig) {
    if (!state?.enabled) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void api.autonomousStart(next).then(s => {
        if (aliveRef.current) setState(s);
      }).catch(e => {
        flash('err', e instanceof Error ? e.message : String(e));
      });
    }, HOT_UPDATE_DEBOUNCE_MS);
  }

  function flash(kind: 'ok' | 'err', text: string) {
    setFeedback({ kind, text });
    setTimeout(() => setFeedback(null), 4000);
  }

  function updateConfig<K extends keyof AutonomousConfig>(key: K, value: AutonomousConfig[K]) {
    const next = { ...config, [key]: value };
    setConfig(next);
    scheduleHotUpdate(next);
  }

  async function toggleEnabled() {
    try {
      if (state?.enabled) {
        const s = await api.autonomousStop();
        setState(s);
        flash('ok', 'autonomous mode stopped');
      } else {
        const s = await api.autonomousStart(config);
        setState(s);
        flash('ok', 'autonomous mode started');
      }
    } catch (e) {
      flash('err', e instanceof Error ? e.message : String(e));
    }
  }

  async function saveQueue() {
    const lines = queueText
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);
    try {
      const s = await api.autonomousSetQueue(lines);
      setState(s);
      setQueueDirty(false);
      flash('ok', `queue saved (${lines.length} goal${lines.length === 1 ? '' : 's'})`);
    } catch (e) {
      flash('err', e instanceof Error ? e.message : String(e));
    }
  }

  const enabled = !!state?.enabled;
  const paused = state?.paused_reason;
  const countdown = useMemo(() => {
    if (!state?.next_swarm_at) return null;
    const remaining = state.next_swarm_at - now;
    return remaining > 0 ? formatCountdown(remaining) : '00:00';
  }, [state?.next_swarm_at, now]);

  const intervalMin = Math.round(config.interval_seconds / 60);
  const dailyCap = config.daily_budget_usd;
  const spent = state?.spent_today_usd ?? 0;
  const spentRatio = dailyCap > 0 ? Math.min(1, spent / dailyCap) : 0;

  return (
    <section style={panelStyle(enabled, !!paused)} aria-label="autonomous mode">
      <header style={headerStyle}>
        <div>
          <div className="eyebrow" style={{ color: enabled ? 'var(--ok)' : 'var(--ink-faint)' }}>
            autonomous mode
          </div>
          <div style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 2, maxWidth: 540 }}>
            When on, the system runs swarms automatically — picks a goal from your queue (FIFO),
            or asks the meta-agent to invent one (if enabled), runs a swarm, waits, repeats.
          </div>
        </div>
        <button
          type="button"
          onClick={() => void toggleEnabled()}
          style={toggleStyle(enabled)}
          aria-pressed={enabled}
          title={enabled ? 'click to stop autonomous mode' : 'click to start autonomous mode'}
        >
          <span style={toggleDotStyle(enabled)} />
          {enabled ? 'ON' : 'OFF'}
        </button>
      </header>

      {paused && (
        <div style={pausedBannerStyle}>
          <span style={{ fontWeight: 700 }}>AUTONOMOUS PAUSED</span>
          <span style={{ marginLeft: 8 }}>{paused}</span>
        </div>
      )}

      <div style={statusRowStyle}>
        <StatusPill
          dotColor={enabled ? 'var(--ok)' : 'var(--ink-faint)'}
          label={enabled ? 'ON' : 'OFF'}
          value={enabled ? `next swarm in ${countdown ?? '—'}` : 'idle'}
        />
        <StatusPill
          dotColor={spentRatio > 0.9 ? 'var(--danger)' : spentRatio > 0.5 ? 'var(--warn)' : 'var(--ok)'}
          label="spent today"
          value={`$${spent.toFixed(4)} / $${dailyCap.toFixed(2)}`}
        />
        <StatusPill
          dotColor="var(--accent)"
          label="swarms today"
          value={String(state?.swarms_today ?? 0)}
        />
        {state?.current_swarm_id != null && (
          <StatusPill dotColor="var(--accent)" label="active" value={`#${state.current_swarm_id}`} />
        )}
      </div>

      <div style={controlsGridStyle}>
        <SliderField
          label="interval"
          value={intervalMin}
          unit="min"
          min={5}
          max={60}
          step={5}
          inputMin={300}
          inputMax={3600}
          inputStep={300}
          inputValue={config.interval_seconds}
          onChange={(secs) => updateConfig('interval_seconds', secs)}
        />
        <SliderField
          label="daily cap"
          value={dailyCap}
          unit="$"
          unitPrefix
          min={0.1}
          max={5}
          step={0.1}
          inputMin={0.1}
          inputMax={5}
          inputStep={0.1}
          inputValue={dailyCap}
          format={(v) => v.toFixed(2)}
          onChange={(v) => updateConfig('daily_budget_usd', Number(v.toFixed(2)))}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={config.meta_agent_enabled}
              onChange={(e) => updateConfig('meta_agent_enabled', e.target.checked)}
              style={{ accentColor: 'var(--cta)' }}
            />
            <span style={{ fontSize: 12, color: 'var(--ink-bright)' }}>
              meta-agent
            </span>
          </label>
          <span style={{ fontSize: 11, color: 'var(--ink-faint)' }}>
            generate goals when queue is empty
          </span>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={queueHeadStyle}>
          <div className="eyebrow">goal queue (one per line, FIFO)</div>
          <button
            type="button"
            className="surface-button surface-button--primary"
            onClick={() => void saveQueue()}
            disabled={!queueDirty}
            style={{ padding: '4px 12px', fontSize: 11, opacity: queueDirty ? 1 : 0.4 }}
          >
            SAVE QUEUE
          </button>
        </div>
        <textarea
          value={queueText}
          onChange={(e) => { setQueueText(e.target.value); setQueueDirty(true); }}
          rows={4}
          placeholder="Refactor frontend/src/lib/sse.ts for clarity&#10;Audit qo/qo-server/src/tools.rs for path-traversal&#10;Document the swarm orchestrator in docs/SWARM.md"
          style={textareaStyle}
        />
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="eyebrow" style={{ marginBottom: 8 }}>
          recent runs {state?.history && state.history.length > 0 && (
            <span style={{ color: 'var(--ink-faint)' }}>· last {Math.min(state.history.length, 20)}</span>
          )}
        </div>
        <RunsList runs={state?.history ?? []} />
      </div>

      {feedback && (
        <div style={feedbackStyle(feedback.kind)}>
          {feedback.text}
        </div>
      )}

      <style>{pulseKeyframes}</style>
    </section>
  );
}

// ─── Sub-components ─────────────────────────────────────────────────

interface SliderFieldProps {
  label: string;
  value: number;
  unit: string;
  unitPrefix?: boolean;
  min: number;
  max: number;
  step: number;
  inputMin: number;
  inputMax: number;
  inputStep: number;
  inputValue: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

function SliderField(p: SliderFieldProps) {
  const display = p.format ? p.format(p.value) : String(p.value);
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span className="eyebrow">{p.label}</span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          fontWeight: 600,
          color: 'var(--ink-bright)',
          marginLeft: 'auto',
        }}>
          {p.unitPrefix ? p.unit : ''}{display}{p.unitPrefix ? '' : ' ' + p.unit}
        </span>
      </div>
      <input
        type="range"
        min={p.inputMin}
        max={p.inputMax}
        step={p.inputStep}
        value={p.inputValue}
        onChange={(e) => p.onChange(Number(e.target.value))}
        style={{ accentColor: 'var(--cta)', width: '100%' }}
      />
    </label>
  );
}

function StatusPill({ dotColor, label, value }: { dotColor: string; label: string; value: string }) {
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '4px 10px',
      background: 'var(--bg-raised)',
      border: '1px solid var(--rule-faint)',
      borderRadius: 4,
      fontSize: 11,
      fontFamily: 'var(--font-mono)',
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%',
        background: dotColor,
        flexShrink: 0,
      }} />
      <span style={{ color: 'var(--ink-faint)' }}>{label}</span>
      <span style={{ color: 'var(--ink-bright)', fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function RunsList({ runs }: { runs: AutonomousRun[] }) {
  if (runs.length === 0) {
    return (
      <div style={{
        padding: '14px 16px',
        background: 'var(--bg-raised)',
        border: '1px dashed var(--rule-faint)',
        borderRadius: 4,
        textAlign: 'center',
        fontSize: 11,
        color: 'var(--ink-faint)',
      }}>
        No runs yet. Turn autonomous mode on and add goals to your queue.
      </div>
    );
  }
  const visible = runs.slice(-20).reverse();
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      maxHeight: 240,
      overflowY: 'auto',
      padding: 4,
      background: 'var(--bg-raised)',
      border: '1px solid var(--rule-faint)',
      borderRadius: 4,
    }}>
      {visible.map(r => (
        <div key={r.swarm_id} style={runRowStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--ink-bright)',
              fontWeight: 600,
            }}>
              #{r.swarm_id}
            </span>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--ok)',
              fontWeight: 600,
            }}>
              ${r.cost_usd.toFixed(4)}
            </span>
            <span style={sourceChipStyle(r.goal_source)}>{r.goal_source}</span>
            <span style={runStatusStyle(r.status)}>{r.status}</span>
          </div>
          <div style={{
            marginTop: 2,
            fontSize: 11,
            color: 'var(--ink-muted)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }} title={r.goal}>
            "{truncate(r.goal, 80)}"
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + '…';
}

// ─── Styles ─────────────────────────────────────────────────────────

function panelStyle(enabled: boolean, paused: boolean): React.CSSProperties {
  const borderColor = paused ? 'var(--danger)' : enabled ? 'var(--ok)' : 'var(--rule-default)';
  return {
    padding: '16px 18px',
    background: 'var(--bg-panel)',
    border: `1px solid ${borderColor}`,
    borderRadius: 6,
    boxShadow: enabled ? '0 0 0 1px ' + borderColor : 'none',
    animation: enabled && !paused ? 'auto-pulse 2.4s var(--ease) infinite' : 'none',
    opacity: enabled || paused ? 1 : 0.94,
    transition: 'border-color var(--t-base-d) var(--ease), opacity var(--t-base-d) var(--ease)',
  };
}

const headerStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'flex-start',
  gap: 16,
  marginBottom: 12,
};

function toggleStyle(on: boolean): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 16px',
    fontFamily: 'var(--font-mono)',
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 0.06,
    color: on ? 'white' : 'var(--ink-bright)',
    background: on ? 'var(--ok)' : 'var(--bg-raised)',
    border: `1px solid ${on ? 'var(--ok)' : 'var(--rule-default)'}`,
    borderRadius: 999,
    cursor: 'pointer',
    flexShrink: 0,
    minWidth: 90,
    justifyContent: 'center',
    transition: 'background var(--t-base-d) var(--ease), color var(--t-base-d) var(--ease)',
  };
}

function toggleDotStyle(on: boolean): React.CSSProperties {
  return {
    width: 8, height: 8, borderRadius: '50%',
    background: on ? 'white' : 'var(--ink-faint)',
    boxShadow: on ? '0 0 8px rgba(255,255,255,0.6)' : 'none',
  };
}

const pausedBannerStyle: React.CSSProperties = {
  marginBottom: 12,
  padding: '10px 14px',
  background: 'var(--danger-soft)',
  border: '1px solid var(--danger)',
  borderRadius: 4,
  color: 'var(--danger)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  letterSpacing: 0.04,
};

const statusRowStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 8,
  marginBottom: 14,
};

const controlsGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
  gap: 16,
  padding: '12px',
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 4,
};

const queueHeadStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 6,
};

const textareaStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  color: 'var(--ink-bright)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  padding: '10px 12px',
  resize: 'vertical',
  minHeight: 80,
  lineHeight: 1.5,
};

const runRowStyle: React.CSSProperties = {
  padding: '6px 10px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 3,
};

function sourceChipStyle(source: 'queue' | 'meta-agent'): React.CSSProperties {
  const bg = source === 'queue' ? 'var(--cta-soft)' : 'var(--accent-soft)';
  const fg = source === 'queue' ? 'var(--cta)' : 'var(--accent)';
  return {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '1px 6px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 600,
    background: bg,
    color: fg,
    borderRadius: 2,
    letterSpacing: 0.04,
    textTransform: 'uppercase',
  };
}

function runStatusStyle(status: string): React.CSSProperties {
  let color = 'var(--ink-muted)';
  let bg = 'var(--bg-raised)';
  if (status === 'done') { color = 'var(--ok)'; bg = 'var(--ok-soft)'; }
  else if (status === 'error' || status === 'failed') { color = 'var(--danger)'; bg = 'var(--danger-soft)'; }
  else if (status === 'stopped') { color = 'var(--warn)'; bg = 'var(--warn-soft)'; }
  else if (status === 'running' || status === 'planning' || status === 'dispatching' || status === 'evaluating') {
    color = 'var(--cta)'; bg = 'var(--cta-soft)';
  }
  return {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '1px 6px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 600,
    background: bg,
    color,
    borderRadius: 2,
    letterSpacing: 0.04,
    textTransform: 'uppercase',
    marginLeft: 'auto',
  };
}

function feedbackStyle(kind: 'ok' | 'err'): React.CSSProperties {
  return {
    marginTop: 12,
    padding: '6px 10px',
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    color: kind === 'ok' ? 'var(--ok)' : 'var(--danger)',
    background: kind === 'ok' ? 'var(--ok-soft)' : 'var(--danger-soft)',
    borderRadius: 4,
  };
}

const pulseKeyframes = `
@keyframes auto-pulse {
  0%, 100% { box-shadow: 0 0 0 1px var(--ok); }
  50%      { box-shadow: 0 0 0 3px var(--ok-soft); }
}
`;
