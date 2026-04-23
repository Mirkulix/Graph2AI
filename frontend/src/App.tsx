import { useEffect, useMemo, useRef, useState } from 'react';
import './styles/tokens.css';
import TopBar from './cockpit/TopBar';
import AgentsPane from './cockpit/AgentsPane';
import ConversationPane from './cockpit/ConversationPane';
import DetailPane, { type DetailContext } from './cockpit/DetailPane';
import ProfileMenu, { type SecondaryView } from './cockpit/ProfileMenu';
import FederationView from './cockpit/secondary/FederationView';
import WerteRadar from './cockpit/secondary/WerteRadar';
import HardwareView from './cockpit/secondary/HardwareView';
import Knowledge3DView from './cockpit/secondary/Knowledge3DView';
import SettingsView from './cockpit/secondary/SettingsView';
import { subscribeSSE, throttle } from './lib/sse';
import { api, type BusMessage, nameOf } from './lib/api';
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
        {view === 'federation'  && <FederationView />}
        {view === 'werte'       && <WerteRadar />}
        {view === 'hardware'    && <HardwareView />}
        {view === 'knowledge3d' && <Knowledge3DView />}
        {view === 'settings'    && <SettingsView />}
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
