import { useEffect, useMemo, useRef, useState } from 'react';
import './styles/tokens.css';
import TopBar from './cockpit/TopBar';
import AgentsPane from './cockpit/AgentsPane';
import ConversationPane from './cockpit/ConversationPane';
import DetailPane, { type DetailContext } from './cockpit/DetailPane';
import ProfileMenu, { type SecondaryView } from './cockpit/ProfileMenu';
import HardwareView from './cockpit/secondary/HardwareView';
import KnowledgeView from './cockpit/secondary/KnowledgeView';
import SettingsView from './cockpit/secondary/SettingsView';
import AutonomousRunsView from './cockpit/secondary/AutonomousRunsView';
import SwarmPane from './cockpit/SwarmPane';
import { subscribeSSE, throttle } from './lib/sse';
import { api, type BusMessage, type AutonomousState, nameOf } from './lib/api';
import { loadTail, saveTail } from './lib/history';

const TAIL_CAP = 200;

export default function App() {
  // Cockpit state
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailContext>({ kind: 'empty' });
  const [collapsed, setCollapsed] = useState(false);
  // liveTail is seeded from localStorage so the cockpit shows recent history
  // on reload instead of an empty view. SSE keeps appending new messages on top.
  // NOTE: ConversationPane currently displays a "Reload clears the view." banner
  // which is no longer accurate — leaving it for now to avoid conflict with the
  // in-flight pipelines edit on that file. Update the banner in a follow-up.
  const [liveTail, setLiveTail] = useState<BusMessage[]>(() => loadTail());
  const [online, setOnline] = useState<boolean>(false);

  // Profile menu + secondary view
  const [menuOpen, setMenuOpen] = useState(false);
  const [secondary, setSecondary] = useState<SecondaryView | null>(null);

  // Pulse animation key — bumps to retrigger animation on each beat
  const [pulseKey, setPulseKey] = useState(0);
  const pulseRef = useRef<HTMLDivElement>(null);

  // Agents list (passed to composer for picker)
  const [agentNames, setAgentNames] = useState<string[]>([]);

  // Autonomous mode global state — polled here for the safety banner
  const [autonomous, setAutonomous] = useState<AutonomousState | null>(null);
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000));

  // ─── Open Knowledge view via custom event (from IDE detail) ─────
  useEffect(() => {
    function onOpenKnowledge() {
      setSecondary('knowledge3d');
      setMenuOpen(false);
    }
    window.addEventListener('orbit:open-knowledge', onOpenKnowledge as EventListener);
    return () => window.removeEventListener('orbit:open-knowledge', onOpenKnowledge as EventListener);
  }, []);

  // ─── Autonomous mode poll (every 5s) for global banner ─────────
  useEffect(() => {
    let alive = true;
    const load = () => {
      api.autonomousStatus()
        .then(s => { if (alive) setAutonomous(s); })
        .catch(() => { if (alive) setAutonomous(null); });
    };
    load();
    const t = setInterval(load, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // 1Hz tick so the banner countdown ticks
  useEffect(() => {
    if (!autonomous?.enabled) return;
    const t = setInterval(() => setNowSec(Math.floor(Date.now() / 1000)), 1000);
    return () => clearInterval(t);
  }, [autonomous?.enabled]);

  async function stopAutonomous() {
    try {
      const s = await api.autonomousStop();
      setAutonomous(s);
    } catch {
      // best-effort; the panel inside SwarmPane will surface errors
    }
  }

  // ─── Health ping ─────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    const ping = () => {
      api.health()
        .then(() => { if (alive) setOnline(true); })
        .catch(() => { if (alive) setOnline(false); });
    };
    ping();
    const t = setInterval(ping, 10000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // ─── Agent name list ─────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    const load = () => {
      api.busAgents()
        .then(list => { if (alive) setAgentNames((list ?? []).map(a => a.name)); })
        .catch(() => {});
    };
    load();
    const t = setInterval(load, 8000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // ─── Live SSE bus stream ─────────────────────────────────────────
  useEffect(() => {
    const beatPulse = throttle(() => setPulseKey(k => k + 1), 800);
    const sub = subscribeSSE<BusMessage>('/api/messages/stream', (m) => {
      setLiveTail(prev => {
        const next = [...prev, { ...m, ts: m.ts ?? new Date().toISOString() }];
        return next.length > TAIL_CAP ? next.slice(-TAIL_CAP) : next;
      });
      beatPulse();
    });
    return () => sub.close();
  }, []);

  // ─── Persist liveTail to localStorage (debounced) ───────────────
  useEffect(() => {
    const t = setTimeout(() => saveTail(liveTail), 500);
    return () => clearTimeout(t);
  }, [liveTail]);

  // ─── Auto-detail when agent changes ──────────────────────────────
  useEffect(() => {
    if (selectedAgent) {
      // Only set agent stats if no graph is already pinned
      if (detail.kind !== 'graph') setDetail({ kind: 'agent', name: selectedAgent });
    } else {
      if (detail.kind === 'agent') setDetail({ kind: 'empty' });
    }
    // intentionally not depending on `detail` to avoid loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAgent]);

  // ─── Open Providers from menu => right-pane detail ───────────────
  function selectSecondary(v: SecondaryView | null) {
    setMenuOpen(false);
    if (v === 'providers') {
      // Providers stays in the right pane (in-cockpit overlay)
      setDetail({ kind: 'providers' });
      setCollapsed(false);
      setSecondary(null);
      return;
    }
    setSecondary(v);
  }

  function openGraph(msg: BusMessage) {
    setDetail({ kind: 'graph', message: msg });
    setCollapsed(false);
  }

  function clearDetail() {
    if (selectedAgent) setDetail({ kind: 'agent', name: selectedAgent });
    else setDetail({ kind: 'empty' });
  }

  // Memoize visible filter
  const visibleAgentList = useMemo(() => agentNames, [agentNames]);

  return (
    <div style={shellStyle}>
      <div ref={pulseRef} key={pulseKey} className="live-pulse live-pulse--beat" />

      {/* Visible build-stamp — proves which version is loaded */}
      <div className="build-stamp" title="Hard-reload (Ctrl+Shift+R) if this stamp doesn't change after a build.">
        BUILD · SWISS-V1 · 2026-04-22
      </div>

      {autonomous?.enabled && (
        <AutonomousBanner state={autonomous} nowSec={nowSec} onStop={() => void stopAutonomous()} />
      )}

      <TopBar
        online={online}
        onProfileClick={() => setMenuOpen(o => !o)}
        profileOpen={menuOpen}
      />

      <ProfileMenu
        open={menuOpen}
        active={secondary}
        onSelect={selectSecondary}
        onClose={() => setMenuOpen(false)}
      />

      <div style={bodyStyle}>
        {secondary ? (
          <SecondaryHost view={secondary} onBack={() => setSecondary(null)} />
        ) : (
          <>
            <AgentsPane
              selectedAgent={selectedAgent}
              onSelectAgent={(name) => {
                setSelectedAgent(name);
                if (detail.kind === 'graph') setDetail({ kind: 'empty' });
              }}
              liveTail={liveTail}
            />
            <ConversationPane
              selectedAgent={selectedAgent}
              liveTail={liveTail}
              onOpenGraph={openGraph}
              agents={visibleAgentList}
            />
            <DetailPane
              ctx={detail}
              collapsed={collapsed}
              onToggleCollapse={() => setCollapsed(c => !c)}
              onClose={clearDetail}
            />
          </>
        )}
      </div>
    </div>
  );
}

function SecondaryHost({ view, onBack }: { view: SecondaryView; onBack: () => void }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-void)' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 18px',
        borderBottom: '1px solid var(--rule-faint)',
        background: 'var(--bg-deep)',
      }}>
        <button onClick={onBack} className="surface-button" style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          ← back to cockpit
        </button>
        <span className="eyebrow" style={{ marginLeft: 8 }}>secondary view · {view}</span>
      </div>
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {view === 'hardware'        && <HardwareView />}
        {view === 'knowledge3d'     && <KnowledgeView />}
        {view === 'swarm'           && <SwarmPane />}
        {view === 'autonomous-runs' && <AutonomousRunsView />}
        {view === 'settings'        && <SettingsView />}
      </div>
    </div>
  );
}

const shellStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  width: '100%',
  background: 'var(--bg-void)',
  position: 'relative',
};

const bodyStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  minHeight: 0,
  overflow: 'hidden',
};

// ─── Global autonomous-mode banner ──────────────────────────────────

function AutonomousBanner({ state, nowSec, onStop }: { state: AutonomousState; nowSec: number; onStop: () => void }) {
  const paused = !!state.paused_reason;
  const remaining = state.next_swarm_at != null ? Math.max(0, state.next_swarm_at - nowSec) : null;
  const countdown = remaining != null ? formatBannerCountdown(remaining) : '—';
  const spent = state.spent_today_usd ?? 0;
  const cap = state.config?.daily_budget_usd ?? 0;

  return (
    <div style={paused ? bannerPausedStyle : bannerActiveStyle} role="alert" aria-live="polite">
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: paused ? 'var(--danger)' : 'var(--ok)',
        boxShadow: paused ? '0 0 8px var(--danger)' : '0 0 8px var(--ok)',
        animation: paused ? 'none' : 'banner-pulse 1.4s ease-in-out infinite',
        flexShrink: 0,
      }} />
      <span style={{ fontWeight: 700, letterSpacing: 0.04 }}>
        {paused ? 'AUTONOMOUS PAUSED' : 'AUTONOMOUS MODE ACTIVE'}
      </span>
      {paused ? (
        <span style={{ marginLeft: 4 }}>— {state.paused_reason}</span>
      ) : (
        <>
          <span style={{ color: 'var(--ink-muted)' }}>—</span>
          <span>next swarm in <strong style={{ color: 'var(--ink-bright)' }}>{countdown}</strong></span>
          <span style={{ color: 'var(--ink-muted)' }}>—</span>
          <span>spent today <strong style={{ color: 'var(--ink-bright)' }}>${spent.toFixed(4)} / ${cap.toFixed(2)}</strong></span>
          {state.current_swarm_id != null && (
            <>
              <span style={{ color: 'var(--ink-muted)' }}>—</span>
              <span>active <strong style={{ color: 'var(--ink-bright)' }}>#{state.current_swarm_id}</strong></span>
            </>
          )}
        </>
      )}
      <button
        type="button"
        onClick={onStop}
        style={bannerStopStyle}
        title="stop autonomous mode"
      >
        STOP
      </button>
      <style>{`@keyframes banner-pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }`}</style>
    </div>
  );
}

function formatBannerCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

const bannerActiveStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  padding: '8px 16px',
  background: 'var(--accent-soft)',
  borderBottom: '2px solid var(--accent)',
  color: 'var(--ink-bright)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  flexShrink: 0,
  zIndex: 40,
};

const bannerPausedStyle: React.CSSProperties = {
  ...bannerActiveStyle,
  background: 'var(--danger-soft)',
  borderBottom: '2px solid var(--danger)',
  color: 'var(--danger)',
};

const bannerStopStyle: React.CSSProperties = {
  marginLeft: 'auto',
  padding: '4px 14px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: 0.08,
  color: 'white',
  background: 'var(--danger)',
  border: '1px solid var(--danger)',
  borderRadius: 3,
  cursor: 'pointer',
  flexShrink: 0,
};
