// Replaces Knowledge3DView. A timeline-first conversation ledger:
// pulls swarms, bus messages, chats and graph-store entries into one
// ordered stream with filters, search, agent-activity sparklines and
// inline drill-down. No three.js — no 1.3 MB chunk.

import { useEffect, useMemo, useState } from 'react';
import { Search, ChevronDown, ChevronRight } from 'lucide-react';
import { api, type SwarmState, type BusMessage } from '../../lib/api';
import { SecondaryFrame } from './SecondaryFrame';

type ItemKind = 'swarm' | 'message' | 'chat' | 'graph';
type TimeBucket = 'today' | '7d' | '30d' | 'all';

interface LedgerItem {
  id: string;
  kind: ItemKind;
  ts: number;                  // unix seconds
  title: string;
  subtitle?: string;
  agents: string[];            // for activity sparklines + filtering
  cost_usd?: number;
  status?: string;
  payload: unknown;            // full source object for drill-down
}

const POLL_INTERVAL_MS = 4000;
const KIND_LABELS: Record<ItemKind, string> = { swarm: 'SWARM', message: 'MSG', chat: 'CHAT', graph: 'GRAPH' };
const KIND_COLORS: Record<ItemKind, string> = {
  swarm: 'var(--cta)',
  message: 'var(--accent)',
  chat: 'var(--ok)',
  graph: 'var(--warn)',
};

export default function KnowledgeView() {
  const [items, setItems] = useState<LedgerItem[]>([]);
  const [filterKind, setFilterKind] = useState<ItemKind | 'all'>('all');
  const [timeBucket, setTimeBucket] = useState<TimeBucket>('today');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    let alive = true;
    const load = async () => {
      const [swarms, msgs, chats, graphs] = await Promise.all([
        api.swarmActive().catch(() => [] as SwarmState[]),
        api.recentMessages(200).catch(() => [] as BusMessage[]),
        api.chatHistory().catch(() => ({ messages: [] as Array<{ role: string; content: string; ts?: string }> })),
        api.graphs().catch(() => []),
      ]);
      if (!alive) return;
      const merged: LedgerItem[] = [];

      for (const s of swarms) {
        const agents = collectSwarmAgents(s);
        merged.push({
          id: `swarm-${s.id}`,
          kind: 'swarm',
          ts: s.started_at,
          title: s.goal,
          subtitle: `${s.rounds.length} round${s.rounds.length === 1 ? '' : 's'} · ${countSubtasks(s)} subtask${countSubtasks(s) === 1 ? '' : 's'}`,
          agents,
          cost_usd: s.cost_usd,
          status: s.status,
          payload: s,
        });
      }

      for (const m of msgs) {
        const from = nameOf(m.from);
        const to = nameOf(m.to);
        merged.push({
          id: `msg-${m.id ?? `${from}-${to}-${m.ts ?? Math.random()}`}`,
          kind: 'message',
          ts: parseTs(m.ts) ?? Math.floor(Date.now() / 1000),
          title: m.content?.slice(0, 200) || `${from} → ${to}`,
          subtitle: `${from} → ${to}  ·  ${intentLabel(m.intent)}${m.auto_triggered ? '  ·  AUTO' : ''}`,
          agents: [from, to].filter(Boolean),
          payload: m,
        });
      }

      const chatMsgs = chats?.messages ?? [];
      for (let i = 0; i < chatMsgs.length; i += 2) {
        const u = chatMsgs[i];
        const a = chatMsgs[i + 1];
        if (!u) continue;
        merged.push({
          id: `chat-${i}-${parseTs(u.ts) ?? i}`,
          kind: 'chat',
          ts: parseTs(u.ts) ?? Math.floor(Date.now() / 1000),
          title: u.content.slice(0, 200),
          subtitle: a ? `→ ${a.content.slice(0, 120)}` : undefined,
          agents: ['user', a ? 'assistant' : ''].filter(Boolean),
          payload: { user: u, assistant: a },
        });
      }

      for (const g of graphs) {
        merged.push({
          id: `graph-${g.id}`,
          kind: 'graph',
          ts: parseTs(g.created_at) ?? 0,
          title: g.name ?? `graph ${g.id}`,
          subtitle: `${g.node_count ?? '?'} nodes · ${g.edge_count ?? '?'} edges`,
          agents: [],
          payload: g,
        });
      }

      merged.sort((a, b) => b.ts - a.ts);
      setItems(merged);
    };
    void load();
    const t = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const visible = useMemo(() => {
    const cutoff = bucketCutoff(timeBucket);
    const q = search.trim().toLowerCase();
    return items.filter((i) =>
      (filterKind === 'all' || i.kind === filterKind)
      && (cutoff === 0 || i.ts >= cutoff)
      && (q.length === 0
        || i.title.toLowerCase().includes(q)
        || (i.subtitle ?? '').toLowerCase().includes(q)
        || i.agents.some((a) => a.toLowerCase().includes(q))));
  }, [items, filterKind, timeBucket, search]);

  const stats = useMemo(() => {
    const totalGraphs = items.length;
    const totalAgentCalls = items.reduce((acc, i) => acc + (i.kind === 'message' || i.kind === 'swarm' ? 1 : 0), 0);
    const counts = new Map<string, number>();
    for (const i of items) for (const a of i.agents) {
      if (a && a !== 'user' && a !== 'assistant') counts.set(a, (counts.get(a) ?? 0) + 1);
    }
    let busiest = '—';
    let busiestCount = 0;
    counts.forEach((v, k) => { if (v > busiestCount) { busiest = k; busiestCount = v; } });
    return { totalGraphs, totalAgentCalls, busiest, busiestCount };
  }, [items]);

  const agentSparklines = useMemo(() => buildAgentSparklines(items), [items]);
  const totalCost = useMemo(() => items.reduce((acc, i) => acc + (i.cost_usd ?? 0), 0), [items]);

  return (
    <SecondaryFrame title="knowledge" subtitle="conversation ledger — every swarm, message, chat and graph that flowed through this qo">
      <div style={statsRow}>
        <Stat label="items" value={String(stats.totalGraphs)} />
        <Stat label="agent calls" value={String(stats.totalAgentCalls)} />
        <Stat label="busiest" value={`${stats.busiest} · ${stats.busiestCount}`} />
        <Stat label="total cost" value={`$${totalCost.toFixed(4)}`} accent />
      </div>

      <div style={controlsRow}>
        <Pills value={filterKind} onChange={setFilterKind} options={[
          { id: 'all',     label: 'all' },
          { id: 'swarm',   label: 'swarms' },
          { id: 'message', label: 'messages' },
          { id: 'chat',    label: 'chats' },
          { id: 'graph',   label: 'graphs' },
        ]} />
        <Pills value={timeBucket} onChange={setTimeBucket} options={[
          { id: 'today', label: 'today' },
          { id: '7d',    label: '7d' },
          { id: '30d',   label: '30d' },
          { id: 'all',   label: 'all' },
        ]} />
        <div style={searchWrap}>
          <Search size={12} style={{ color: 'var(--ink-faint)' }} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="filter agents, content, ids…"
            style={searchInput}
          />
        </div>
      </div>

      <div style={layout}>
        <div style={{ minWidth: 0 }}>
          {visible.length === 0 ? (
            <div style={emptyStyle}>
              No matching items in <strong>{timeBucket}</strong>{filterKind !== 'all' ? ` · filter "${filterKind}"` : ''}{search ? ` · search "${search}"` : ''}.
              <div style={{ fontSize: 11, color: 'var(--ink-faint)', marginTop: 6 }}>
                The ledger fills as you talk to agents, run swarms, or as the bus carries messages.
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {visible.map((i) => (
                <Row
                  key={i.id}
                  item={i}
                  expanded={expanded.has(i.id)}
                  onToggle={() => {
                    const next = new Set(expanded);
                    if (next.has(i.id)) next.delete(i.id); else next.add(i.id);
                    setExpanded(next);
                  }}
                />
              ))}
            </div>
          )}
        </div>
        <aside style={sidebar}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>agents (this view)</div>
          {agentSparklines.length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--ink-faint)' }}>No activity yet.</div>
          ) : agentSparklines.map((a) => (
            <div key={a.name} style={agentRow}>
              <div style={{ fontSize: 11, color: 'var(--ink-bright)', fontFamily: 'var(--font-mono)' }}>{a.name}</div>
              <Sparkline buckets={a.buckets} />
              <div style={{ fontSize: 10, color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)' }}>{a.total}</div>
            </div>
          ))}
        </aside>
      </div>
    </SecondaryFrame>
  );
}

// ───────────────────────────────────────────────────────────── helpers

function nameOf(v: unknown): string {
  if (!v) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'object' && v !== null && 'name' in v) return String((v as { name: unknown }).name ?? '');
  return '';
}
function intentLabel(v: unknown): string {
  if (!v) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'object' && v !== null && 'kind' in v) return String((v as { kind: unknown }).kind ?? '');
  return '';
}
function parseTs(v: unknown): number | null {
  if (typeof v === 'number') return Math.floor(v > 1e12 ? v / 1000 : v);
  if (typeof v === 'string') { const n = Date.parse(v); return Number.isFinite(n) ? Math.floor(n / 1000) : null; }
  return null;
}
function bucketCutoff(b: TimeBucket): number {
  const now = Math.floor(Date.now() / 1000);
  if (b === 'today') return now - 86400;
  if (b === '7d') return now - 7 * 86400;
  if (b === '30d') return now - 30 * 86400;
  return 0;
}
function countSubtasks(s: SwarmState): number {
  return s.rounds.reduce((acc, r) => acc + r.subtasks.length, 0);
}
function collectSwarmAgents(s: SwarmState): string[] {
  const set = new Set<string>();
  for (const r of s.rounds) for (const t of r.subtasks) if (t.assigned_to) set.add(t.assigned_to);
  return Array.from(set);
}
function buildAgentSparklines(items: LedgerItem[]) {
  const BUCKETS = 12;                       // 12 buckets across the visible window
  const now = Math.floor(Date.now() / 1000);
  const span = 86400;                       // last 24h
  const start = now - span;
  const counts = new Map<string, number[]>();
  for (const i of items) {
    if (i.ts < start) continue;
    const slot = Math.min(BUCKETS - 1, Math.max(0, Math.floor(((i.ts - start) / span) * BUCKETS)));
    for (const a of i.agents) {
      if (!a || a === 'user' || a === 'assistant') continue;
      let arr = counts.get(a);
      if (!arr) { arr = new Array(BUCKETS).fill(0); counts.set(a, arr); }
      arr[slot] += 1;
    }
  }
  return Array.from(counts.entries())
    .map(([name, buckets]) => ({ name, buckets, total: buckets.reduce((a, b) => a + b, 0) }))
    .sort((a, b) => b.total - a.total);
}

// ───────────────────────────────────────────────────────────── tiny UI primitives

function Row({ item, expanded, onToggle }: { item: LedgerItem; expanded: boolean; onToggle: () => void }) {
  return (
    <div style={rowStyle}>
      <button onClick={onToggle} style={rowHeader}>
        <span style={{ ...kindBadge, background: KIND_COLORS[item.kind] }}>{KIND_LABELS[item.kind]}</span>
        <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
          <div style={{ fontSize: 12, color: 'var(--ink-bright)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {item.title || '(untitled)'}
          </div>
          {item.subtitle && (
            <div style={{ fontSize: 10, color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
              {item.subtitle}
            </div>
          )}
        </div>
        {item.cost_usd != null && item.cost_usd > 0 && (
          <span style={metaPill}>${item.cost_usd.toFixed(4)}</span>
        )}
        {item.status && <span style={metaPill}>{item.status}</span>}
        <span style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', minWidth: 84, textAlign: 'right' }}>
          {fmtTime(item.ts)}
        </span>
        {expanded ? <ChevronDown size={14} style={{ color: 'var(--ink-faint)' }} /> : <ChevronRight size={14} style={{ color: 'var(--ink-faint)' }} />}
      </button>
      {expanded && (
        <div style={{ padding: '8px 12px 12px 12px', borderTop: '1px solid var(--rule-faint)' }}>
          {item.kind === 'swarm' && <SwarmDetail s={item.payload as SwarmState} />}
          {item.kind === 'message' && <MessageDetail m={item.payload as BusMessage} />}
          {item.kind === 'chat' && <ChatDetail c={item.payload as { user: { content: string }; assistant?: { content: string } }} />}
          {item.kind === 'graph' && <pre style={preStyle}>{JSON.stringify(item.payload, null, 2)}</pre>}
        </div>
      )}
    </div>
  );
}

function SwarmDetail({ s }: { s: SwarmState }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {s.rounds.map((r) => (
        <div key={r.round_number} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div className="eyebrow">round {r.round_number}</div>
          <pre style={preStyle}>{r.plan.slice(0, 600)}</pre>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {r.subtasks.map((t) => (
              <div key={t.id} style={subtaskRow}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-muted)', minWidth: 36 }}>{t.id}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-bright)', minWidth: 90 }}>{t.assigned_to}</span>
                <span style={{ fontSize: 10, color: 'var(--ink-faint)', minWidth: 50 }}>{t.status}</span>
                <span style={{ fontSize: 11, color: 'var(--ink-muted)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {t.response ?? '—'}
                </span>
              </div>
            ))}
          </div>
          {r.eval && <div style={{ fontSize: 11, color: 'var(--ok)', fontStyle: 'italic' }}>eval: {r.eval}</div>}
        </div>
      ))}
    </div>
  );
}
function MessageDetail({ m }: { m: BusMessage }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <FlowMini steps={[nameOf(m.from) || 'src', intentLabel(m.intent) || 'intent', nameOf(m.to) || 'dst']} />
      {m.content && <pre style={preStyle}>{m.content.slice(0, 1200)}</pre>}
    </div>
  );
}
function ChatDetail({ c }: { c: { user: { content: string }; assistant?: { content: string } } }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <pre style={preStyle}><b>user:</b> {c.user.content}</pre>
      {c.assistant && <pre style={preStyle}><b>assistant:</b> {c.assistant.content.slice(0, 1200)}</pre>}
    </div>
  );
}
function FlowMini({ steps }: { steps: string[] }) {
  return (
    <svg width="100%" height="42" viewBox={`0 0 ${steps.length * 110} 42`} style={{ maxWidth: 600 }}>
      {steps.map((s, i) => (
        <g key={i} transform={`translate(${i * 110}, 0)`}>
          <rect x={0} y={10} width={90} height={22} rx={4} fill="var(--bg-raised)" stroke="var(--rule-default)" />
          <text x={45} y={25} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink-bright)">{s.slice(0, 12)}</text>
          {i < steps.length - 1 && <line x1={90} y1={21} x2={110} y2={21} stroke="var(--rule-default)" strokeWidth={1} markerEnd="url(#arr)" />}
        </g>
      ))}
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <polygon points="0 0, 6 3, 0 6" fill="var(--ink-faint)" />
        </marker>
      </defs>
    </svg>
  );
}

function Pills<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: Array<{ id: T; label: string }> }) {
  return (
    <div style={{ display: 'flex', gap: 4 }}>
      {options.map((o) => (
        <button key={o.id} onClick={() => onChange(o.id)} style={value === o.id ? pillActive : pill}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
function Sparkline({ buckets }: { buckets: number[] }) {
  const max = Math.max(1, ...buckets);
  return (
    <svg width="80" height="14" style={{ flex: '0 0 auto' }}>
      {buckets.map((v, i) => {
        const h = (v / max) * 12;
        const x = i * 7;
        return <rect key={i} x={x} y={14 - h} width={5} height={h} fill="var(--cta)" />;
      })}
    </svg>
  );
}
function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={statBox}>
      <div className="eyebrow">{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: accent ? 'var(--cta)' : 'var(--ink-bright)', marginTop: 4 }}>{value}</div>
    </div>
  );
}
function fmtTime(ts: number): string {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  if (sameDay) return d.toTimeString().slice(0, 8);
  return d.toISOString().slice(5, 16).replace('T', ' ');
}

// ───────────────────────────────────────────────────────────── styles

const statsRow: React.CSSProperties = { display: 'flex', gap: 8, marginBottom: 12 };
const statBox: React.CSSProperties = { flex: 1, padding: '10px 12px', background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4 };
const controlsRow: React.CSSProperties = { display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 };
const pill: React.CSSProperties = { padding: '4px 10px', fontSize: 11, background: 'transparent', border: '1px solid var(--rule-default)', color: 'var(--ink-muted)', cursor: 'pointer', borderRadius: 3, fontFamily: 'var(--font-mono)' };
const pillActive: React.CSSProperties = { ...pill, background: 'var(--cta)', color: 'var(--bg-void)', borderColor: 'var(--cta)' };
const searchWrap: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, flex: 1, minWidth: 200, padding: '4px 8px', background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 3 };
const searchInput: React.CSSProperties = { flex: 1, fontSize: 12, background: 'transparent', border: 'none', outline: 'none', color: 'var(--ink-bright)', fontFamily: 'var(--font-mono)' };
const layout: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 240px', gap: 16 };
const sidebar: React.CSSProperties = { padding: '12px 14px', background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4, alignSelf: 'start', position: 'sticky', top: 12 };
const agentRow: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', justifyContent: 'space-between' };
const rowStyle: React.CSSProperties = { background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4 };
const rowHeader: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 10, width: '100%', padding: '8px 12px', background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left' };
const kindBadge: React.CSSProperties = { padding: '2px 6px', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--bg-void)', borderRadius: 2, letterSpacing: 0.5, fontWeight: 600 };
const metaPill: React.CSSProperties = { padding: '1px 6px', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', border: '1px solid var(--rule-faint)', borderRadius: 2 };
const subtaskRow: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', background: 'var(--bg-raised)', borderRadius: 3 };
const preStyle: React.CSSProperties = { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-muted)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0, padding: 8, background: 'var(--bg-raised)', borderRadius: 3, lineHeight: 1.5 };
const emptyStyle: React.CSSProperties = { padding: '32px 24px', textAlign: 'center', color: 'var(--ink-muted)', fontSize: 13, border: '1px dashed var(--rule-default)', borderRadius: 4 };
