import { useEffect, useMemo, useState } from 'react';
import { Search, ArrowRight, Laptop } from 'lucide-react';
import { api, type AgentInfo, type BusMessage, type PresenceEntry, nameOf, intentOf } from '../lib/api';

interface Props {
  selectedAgent: string | null;
  onSelectAgent: (name: string | null) => void;
  liveTail: BusMessage[];
}

export default function AgentsPane({ selectedAgent, onSelectAgent, liveTail }: Props) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [presence, setPresence] = useState<PresenceEntry[]>([]);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    let alive = true;
    const load = () => {
      api.busAgents()
        .then(list => { if (alive) setAgents(list ?? []); })
        .catch(() => { if (alive) setAgents([]); });
      api.presence()
        .then(list => { if (alive) setPresence(list ?? []); })
        .catch(() => { if (alive) setPresence([]); });
    };
    load();
    const t = setInterval(load, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const filtered = useMemo(() => {
    if (!filter.trim()) return agents;
    const f = filter.toLowerCase();
    return agents.filter(a => a.name.toLowerCase().includes(f) || (a.role ?? '').toLowerCase().includes(f));
  }, [agents, filter]);

  const filteredPresence = useMemo(() => {
    if (!filter.trim()) return presence;
    const f = filter.toLowerCase();
    return presence.filter(p =>
      p.identity.toLowerCase().includes(f) ||
      (p.ide_name ?? '').toLowerCase().includes(f) ||
      (p.host ?? '').toLowerCase().includes(f),
    );
  }, [presence, filter]);

  return (
    <aside style={paneStyle}>
      <div style={searchWrap}>
        <Search size={12} strokeWidth={1.6} style={{ color: 'var(--ink-faint)' }} />
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="filter agents…"
          style={searchInput}
        />
        {filter && (
          <button onClick={() => setFilter('')} style={{ color: 'var(--ink-faint)', fontSize: 11, padding: '0 4px' }}>×</button>
        )}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        <div style={sectionHeaderStyle}>
          <span className="eyebrow">agents · {agents.length}</span>
        </div>

        <button
          onClick={() => onSelectAgent(null)}
          style={{
            ...rowStyle,
            background: selectedAgent === null ? 'var(--signal-soft)' : 'transparent',
            borderLeft: selectedAgent === null ? '2px solid var(--signal)' : '2px solid transparent',
          }}
        >
          <div style={{ width: 8, height: 8, borderRadius: 4, background: 'var(--signal)', boxShadow: '0 0 8px var(--signal-glow)' }} />
          <div style={{ flex: 1, textAlign: 'left' }}>
            <div style={{ fontSize: 13, color: 'var(--ink-bright)', fontWeight: 500 }}>All Activity</div>
            <div style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>broadcast</div>
          </div>
        </button>

        {filtered.length === 0 ? (
          <div style={{ padding: '24px 16px', fontSize: 12, color: 'var(--ink-faint)', textAlign: 'center' }}>
            No agents registered. Start the QO server.
          </div>
        ) : (
          filtered.map(a => {
            const active = selectedAgent === a.name;
            const dot = a.online === false ? 'var(--ink-faint)' : 'var(--verified)';
            return (
              <button
                key={a.name}
                onClick={() => onSelectAgent(a.name)}
                style={{
                  ...rowStyle,
                  background: active ? 'var(--signal-soft)' : 'transparent',
                  borderLeft: active ? '2px solid var(--signal)' : '2px solid transparent',
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: 4,
                  background: dot,
                  boxShadow: a.online === false ? 'none' : `0 0 8px ${dot}`,
                  flexShrink: 0,
                }} />
                <div style={{ flex: 1, textAlign: 'left', minWidth: 0 }}>
                  <div style={{ fontSize: 13, color: active ? 'var(--ink-bright)' : 'var(--ink-primary)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {a.name}
                  </div>
                  {a.role && (
                    <div style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {a.role}
                    </div>
                  )}
                </div>
                {!!a.mailbox_size && a.mailbox_size > 0 && (
                  <span style={{
                    fontSize: 10,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--signal-bright)',
                    background: 'var(--signal-soft)',
                    padding: '1px 6px',
                    borderRadius: 8,
                  }}>
                    {a.mailbox_size}
                  </span>
                )}
              </button>
            );
          })
        )}

        <div style={{ ...sectionHeaderStyle, marginTop: 16, borderTop: '1px solid var(--rule-faint)', paddingTop: 12 }}>
          <Laptop size={10} strokeWidth={1.6} style={{ color: 'var(--ink-faint)', marginRight: 4 }} />
          <span className="eyebrow">connected ides · {filteredPresence.length}</span>
        </div>

        {filteredPresence.length === 0 ? (
          <div style={{ padding: '8px 16px', fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
            No IDEs registered.
          </div>
        ) : (
          filteredPresence.map(p => {
            const active = selectedAgent === p.identity;
            const autoRespond = p.capabilities?.includes('auto-respond') === true;
            return (
              <button
                key={p.identity}
                onClick={() => onSelectAgent(p.identity)}
                style={{
                  ...rowStyle,
                  background: active ? 'var(--signal-soft)' : 'transparent',
                  borderLeft: active ? '2px solid var(--signal)' : '2px solid transparent',
                }}
                title={`${p.identity}\n${p.ide_name ?? ''} · ${p.host ?? ''}\n${p.llm_provider ? `${p.llm_provider}/${p.llm_model ?? ''}` : 'no auto-respond'}`}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: 4,
                  background: autoRespond ? 'var(--cta)' : 'var(--ink-muted)',
                  boxShadow: autoRespond ? '0 0 8px var(--cta-glow)' : 'none',
                  flexShrink: 0,
                }} />
                <div style={{ flex: 1, textAlign: 'left', minWidth: 0 }}>
                  <div style={{
                    fontSize: 12,
                    color: active ? 'var(--ink-bright)' : 'var(--ink-primary)',
                    fontWeight: 500,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    {p.identity}
                  </div>
                  {(p.ide_name || p.host) && (
                    <div style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {[p.ide_name, p.host].filter(Boolean).join(' · ')}
                    </div>
                  )}
                </div>
                {autoRespond && (
                  <span style={{
                    fontSize: 9,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--cta)',
                    background: 'var(--cta-soft)',
                    padding: '1px 6px',
                    borderRadius: 8,
                    letterSpacing: 0.04,
                    textTransform: 'uppercase',
                  }}>
                    auto
                  </span>
                )}
              </button>
            );
          })
        )}

        <div style={{ ...sectionHeaderStyle, marginTop: 16, borderTop: '1px solid var(--rule-faint)', paddingTop: 12 }}>
          <span className="eyebrow">live</span>
          <span style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
            {liveTail.length}
          </span>
        </div>

        {liveTail.length === 0 ? (
          <div style={{ padding: '12px 16px', fontSize: 11, color: 'var(--ink-faint)' }}>
            Awaiting traffic…
          </div>
        ) : (
          <div style={{ padding: '0 8px' }}>
            {liveTail.slice().reverse().slice(0, 24).map((m, i) => (
              <LiveRow key={`${m.id ?? i}-${i}`} msg={m} />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

function LiveRow({ msg }: { msg: BusMessage }) {
  const from = nameOf(msg.from);
  const to = nameOf(msg.to);
  const intent = intentOf(msg.intent);
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      padding: '4px 6px',
      fontSize: 10,
      fontFamily: 'var(--font-mono)',
      color: 'var(--ink-muted)',
      borderRadius: 3,
    }}>
      <span style={{ color: 'var(--ink-primary)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{from}</span>
      <ArrowRight size={9} strokeWidth={2} style={{ color: 'var(--ink-faint)', flexShrink: 0 }} />
      <span style={{ color: 'var(--ink-primary)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{to}</span>
      <span style={{ color: 'var(--process-bright)', marginLeft: 'auto', fontSize: 9 }}>{intent}</span>
    </div>
  );
}

const paneStyle: React.CSSProperties = {
  width: 'var(--pane-left-w)',
  flexShrink: 0,
  background: 'var(--bg-deep)',
  borderRight: '1px solid var(--rule-default)',
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  minHeight: 0,
};

const searchWrap: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '10px 12px',
  borderBottom: '1px solid var(--rule-faint)',
  flexShrink: 0,
};

const searchInput: React.CSSProperties = {
  flex: 1,
  padding: '4px 0',
  fontSize: 12,
  color: 'var(--ink-primary)',
  background: 'transparent',
};

const sectionHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  padding: '12px 16px 6px',
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  width: '100%',
  padding: '8px 14px',
  cursor: 'pointer',
  transition: 'background 120ms',
};
