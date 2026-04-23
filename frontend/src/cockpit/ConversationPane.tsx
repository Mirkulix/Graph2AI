import { useEffect, useMemo, useRef, useState } from 'react';
import { Radio, Send, Shield, ShieldOff, Users, X, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, type BusMessage, type ConsensusResponse, type ConsensusReply, nameOf, intentOf, bytesToHex } from '../lib/api';
import GraphThumbnail from './GraphThumbnail';

type ComposerMode = 'single' | 'consensus';

interface Props {
  selectedAgent: string | null;
  liveTail: BusMessage[];
  onOpenGraph: (msg: BusMessage) => void;
  agents: string[];
}

export default function ConversationPane({ selectedAgent, liveTail, onOpenGraph, agents }: Props) {
  const visible = useMemo(() => {
    if (selectedAgent === null) return liveTail;
    return liveTail.filter(m =>
      nameOf(m.from) === selectedAgent || nameOf(m.to) === selectedAgent,
    );
  }, [liveTail, selectedAgent]);

  const [consensusResult, setConsensusResult] = useState<ConsensusResponse | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [visible.length, consensusResult]);

  return (
    <main style={paneStyle}>
      <div style={threadHeaderStyle}>
        <div>
          <div className="eyebrow" style={{ color: 'var(--ink-faint)' }}>thread</div>
          <h2 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 22,
            fontWeight: 500,
            margin: '4px 0 0',
            color: 'var(--ink-bright)',
            letterSpacing: -0.01,
          }}>
            {selectedAgent ?? 'All Activity'}
          </h2>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div className="eyebrow" style={{ color: 'var(--ink-faint)' }}>messages</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, color: 'var(--ink-primary)', marginTop: 2 }}>
            {visible.length}
          </div>
        </div>
      </div>

      <div ref={scrollRef} style={scrollStyle}>
        {consensusResult && (
          <ConsensusResultPane
            result={consensusResult}
            onDismiss={() => setConsensusResult(null)}
          />
        )}
        {visible.length === 0 && !consensusResult ? (
          <EmptyState selectedAgent={selectedAgent} />
        ) : visible.length > 0 ? (
          <>
            <div style={liveBannerStyle}>
              <Radio size={10} strokeWidth={2.2} style={{ color: 'var(--cta)' }} />
              <span style={{ color: 'var(--cta)', fontWeight: 600, letterSpacing: 0.06 }}>LIVE STREAM</span>
              <span style={{ color: 'var(--ink-faint)' }}>·</span>
              <span>this view shows messages received since the cockpit opened. Reload clears the view.</span>
            </div>
            {visible.map((m, i) => (
              <MessageCard key={`${m.id ?? i}-${i}`} msg={m} onOpenGraph={onOpenGraph} />
            ))}
          </>
        ) : null}
      </div>

      <Composer
        defaultTo={selectedAgent ?? agents[0] ?? 'developer'}
        agents={agents}
        onConsensus={setConsensusResult}
      />
    </main>
  );
}

function MessageCard({ msg, onOpenGraph }: { msg: BusMessage; onOpenGraph: (m: BusMessage) => void }) {
  const from = nameOf(msg.from);
  const to = nameOf(msg.to);
  const intent = intentOf(msg.intent);
  const sigHex = bytesToHex(msg.signature, 8);
  const isSigned = !!msg.signed || !!sigHex;
  const verified = msg.signature_verified !== false; // default true if signed

  return (
    <article style={{
      ...cardStyle,
      borderLeft: msg.auto_triggered ? '3px solid var(--warn)' : cardStyle.borderLeft,
    }}>
      <header style={cardHeaderStyle}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: 'var(--ink-bright)' }}>
          {from}
        </span>
        <span style={{ color: 'var(--ink-faint)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>→</span>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: 'var(--ink-bright)' }}>
          {to}
        </span>

        <span style={intentBadgeStyle}>{intent}</span>

        {msg.auto_triggered && (
          <span style={autoTriggerBadgeStyle} title={`Sent automatically by trigger: ${msg.trigger_kind || 'unknown'}`}>
            <Zap size={9} strokeWidth={2.2} />
            auto
            {msg.trigger_kind && (
              <span style={{ marginLeft: 4, opacity: 0.75, fontFamily: 'var(--font-mono)' }}>
                · {msg.trigger_kind}
              </span>
            )}
          </span>
        )}

        {isSigned ? (
          <span className={verified ? 'shield-mark' : 'shield-mark shield-mark--invalid'}>
            <Shield size={9} strokeWidth={2.2} />
            {verified ? 'signed' : 'invalid'}
            {sigHex && <span style={{ opacity: 0.6, marginLeft: 2 }}>{sigHex.slice(0, 6)}</span>}
          </span>
        ) : (
          <span className="shield-mark shield-mark--unsigned">
            <ShieldOff size={9} strokeWidth={2.2} />
            unsigned
          </span>
        )}

        {msg.ts && (
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
            {formatTs(msg.ts)}
          </span>
        )}
      </header>

      {msg.content && msg.content.trim().length > 0 && (
        <MessageContent content={msg.content} isReply={msg.is_reply === true} />
      )}

      {msg.graph != null && (
        <div style={{ marginTop: 10 }}>
          <GraphThumbnail graph={msg.graph} onClick={() => onOpenGraph(msg)} />
        </div>
      )}

      {msg.in_reply_to != null && (
        <div style={{ marginTop: 8, fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
          in reply to #{msg.in_reply_to}
        </div>
      )}
    </article>
  );
}

function MessageContent({ content, isReply }: { content: string; isReply: boolean }) {
  if (isReply) {
    return (
      <div style={replyContentStyle}>
        <div className="md-reply" style={{ fontSize: 13, lineHeight: 1.55, color: 'var(--ink-bright)' }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    );
  }
  return (
    <div style={promptContentStyle}>
      <div className="eyebrow" style={{ color: 'var(--ink-faint)', marginBottom: 4 }}>prompt</div>
      <pre style={promptPreStyle}>{content}</pre>
    </div>
  );
}

const CONSENSUS_DEFAULTS = ['developer', 'guardian', 'strategist'];

function pickDefaultConsensusAgents(agents: string[]): string[] {
  const present = CONSENSUS_DEFAULTS.filter(a => agents.includes(a));
  if (present.length > 0) return present.slice(0, 3);
  return agents.slice(0, Math.min(3, agents.length));
}

function Composer({
  defaultTo,
  agents,
  onConsensus,
}: {
  defaultTo: string;
  agents: string[];
  onConsensus: (result: ConsensusResponse) => void;
}) {
  const [mode, setMode] = useState<ComposerMode>('single');
  const [text, setText] = useState('');
  const [target, setTarget] = useState(defaultTo);
  const [selectedAgents, setSelectedAgents] = useState<string[]>(() => pickDefaultConsensusAgents(agents));
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  useEffect(() => { setTarget(defaultTo); }, [defaultTo]);
  // Re-seed default consensus agents when the agent list first becomes non-empty
  useEffect(() => {
    if (selectedAgents.length === 0 && agents.length > 0) {
      setSelectedAgents(pickDefaultConsensusAgents(agents));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agents.join('|')]);

  async function send() {
    if (!text.trim() || busy) return;
    setBusy(true);
    setFeedback(null);
    try {
      const msg = {
        id: Math.floor(Math.random() * 1_000_000),
        from: { name: 'cockpit', capabilities: ['Execute'] },
        to: { name: target, capabilities: ['Execute'] },
        graph: { type: 'prompt', source: text },
        inputs: {},
        intent: 'Execute',
        in_reply_to: null,
        signature: null,
        signer_pubkey: null,
        graph_hash: null,
      };
      const seedHex = '0'.repeat(64); // dev-time signing key — set via QO_SEED_HEX env in prod
      const reply = await api.qlmsReply([msg], seedHex);
      const delivered = await api.qlmsDeliver(reply.frame);
      if (delivered.signature_verified) {
        setFeedback({ kind: 'ok', text: `delivered → ${target}` });
        setText('');
      } else {
        setFeedback({ kind: 'err', text: 'signature not verified' });
      }
    } catch (e) {
      setFeedback({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      setTimeout(() => setFeedback(null), 4000);
    }
  }

  async function sendConsensus() {
    if (!text.trim() || busy || selectedAgents.length === 0) return;
    setBusy(true);
    setFeedback(null);
    try {
      const result = await api.consensus(text, selectedAgents);
      onConsensus(result);
      setFeedback({ kind: 'ok', text: `consensus · ${result.summary.successful}/${result.summary.total_replies} replies · score ${result.summary.consensus_score.toFixed(2)}` });
      setText('');
    } catch (e) {
      setFeedback({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      setTimeout(() => setFeedback(null), 4000);
    }
  }

  function toggleAgent(name: string) {
    setSelectedAgents(prev =>
      prev.includes(name) ? prev.filter(a => a !== name) : [...prev, name],
    );
  }

  const canSend = mode === 'single'
    ? !!text.trim()
    : !!text.trim() && selectedAgents.length > 0;

  return (
    <div style={composerStyle}>
      {feedback && (
        <div style={{
          padding: '6px 10px',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          color: feedback.kind === 'ok' ? 'var(--verified-bright)' : 'var(--alert-bright)',
          background: feedback.kind === 'ok' ? 'var(--verified-soft)' : 'var(--alert-soft)',
          borderRadius: 4,
          marginBottom: 8,
        }}>
          {feedback.text}
        </div>
      )}

      <div style={modeToggleRowStyle}>
        <div style={modeToggleStyle} role="tablist" aria-label="composer mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'single'}
            onClick={() => setMode('single')}
            style={modeButtonStyle(mode === 'single')}
          >
            <Send size={10} strokeWidth={2} />
            single
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'consensus'}
            onClick={() => setMode('consensus')}
            style={modeButtonStyle(mode === 'consensus', true)}
          >
            <Users size={10} strokeWidth={2} />
            consensus
          </button>
        </div>
        {mode === 'consensus' && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-faint)' }}>
            {selectedAgents.length} of {agents.length} agents selected
          </span>
        )}
      </div>

      {mode === 'consensus' && (
        <div style={agentMultiSelectStyle}>
          {agents.length === 0 ? (
            <span style={{ fontSize: 11, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
              no agents available
            </span>
          ) : agents.map(a => {
            const on = selectedAgents.includes(a);
            return (
              <button
                key={a}
                type="button"
                onClick={() => toggleAgent(a)}
                className={on ? 'chip chip-cta' : 'chip'}
                style={{
                  cursor: 'pointer',
                  border: on ? '1px solid transparent' : '1px solid var(--rule-default)',
                }}
              >
                {a}
              </button>
            );
          })}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        {mode === 'single' && (
          <select
            value={target}
            onChange={e => setTarget(e.target.value)}
            style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--rule-default)',
              borderRadius: 4,
              color: 'var(--ink-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              padding: '6px 8px',
              cursor: 'pointer',
            }}
          >
            {agents.length === 0 ? <option value="developer">developer</option> : null}
            {agents.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        )}
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void (mode === 'single' ? send() : sendConsensus());
            }
          }}
          placeholder={
            mode === 'single'
              ? 'Compose a signed message…   (Cmd/Ctrl+Enter to send)'
              : 'Ask N agents the same question…   (Cmd/Ctrl+Enter to fan out)'
          }
          rows={2}
          style={{
            flex: 1,
            background: 'var(--bg-elevated)',
            border: '1px solid var(--rule-default)',
            borderRadius: 4,
            color: 'var(--ink-bright)',
            fontFamily: 'var(--font-ui)',
            fontSize: 13,
            padding: '8px 10px',
            resize: 'vertical',
            minHeight: 32,
            lineHeight: 1.5,
          }}
        />
        <button
          onClick={() => void (mode === 'single' ? send() : sendConsensus())}
          disabled={busy || !canSend}
          className="surface-button surface-button--primary"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '8px 14px',
            opacity: (busy || !canSend) ? 0.4 : 1,
            cursor: (busy || !canSend) ? 'not-allowed' : 'pointer',
          }}
        >
          {mode === 'single' ? <Send size={12} strokeWidth={2} /> : <Users size={12} strokeWidth={2} />}
          {mode === 'single'
            ? (busy ? 'signing…' : 'send')
            : (busy ? 'asking…' : `ask ${selectedAgents.length} agent${selectedAgents.length === 1 ? '' : 's'}`)}
        </button>
      </div>
    </div>
  );
}

// ─── Consensus result ────────────────────────────────────────────────

function ConsensusResultPane({
  result,
  onDismiss,
}: {
  result: ConsensusResponse;
  onDismiss: () => void;
}) {
  const { summary, replies } = result;
  const cols = Math.min(replies.length || 1, 3);
  const score = Number.isFinite(summary.consensus_score)
    ? summary.consensus_score.toFixed(2)
    : '—';
  const labelClass = consensusChipClass(summary.consensus_label);

  return (
    <section style={consensusBlockStyle} aria-label="consensus result">
      <header style={consensusHeaderStyle}>
        <Users size={12} strokeWidth={2.2} style={{ color: 'var(--process)' }} />
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 14,
          fontWeight: 500,
          color: 'var(--ink-bright)',
          letterSpacing: -0.01,
        }}>
          consensus
        </span>
        <span style={{ color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
          · {summary.total_replies} agent{summary.total_replies === 1 ? '' : 's'}
        </span>
        <span style={consensusScoreStyle} title="Jaccard agreement score (0–1)">
          score {score}
        </span>
        <span className={`chip ${labelClass}`} style={{ fontSize: 10 }}>
          {summary.consensus_label}
        </span>
        {summary.failed > 0 && (
          <span className="chip chip-danger" style={{ fontSize: 10 }}>
            {summary.failed} failed
          </span>
        )}
        <span style={{
          marginLeft: 'auto',
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--ink-faint)',
        }}>
          avg {Math.round(summary.avg_latency_ms)}ms
        </span>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="dismiss consensus"
          title="dismiss"
          style={dismissButtonStyle}
        >
          <X size={12} strokeWidth={2.2} />
          dismiss
        </button>
      </header>

      <div style={{
        marginTop: 12,
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gap: 12,
      }}>
        {replies.map((r, i) => (
          <ConsensusReplyCard key={`${r.agent}-${i}`} reply={r} />
        ))}
      </div>
    </section>
  );
}

function ConsensusReplyCard({ reply }: { reply: ConsensusReply }) {
  const failed = !reply.ok;
  return (
    <article style={{
      ...consensusCardStyle,
      borderLeft: failed ? '3px solid var(--alert)' : '3px solid var(--process)',
      opacity: failed ? 0.78 : 1,
    }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 13,
          fontWeight: 600,
          color: 'var(--ink-bright)',
        }}>
          {reply.agent}
        </span>
        <span className="chip chip-mono" style={{ fontSize: 10 }}>
          {Math.round(reply.latency_ms)}ms
        </span>
        {failed && (
          <span className="chip chip-danger" style={{ fontSize: 10 }}>error</span>
        )}
      </header>
      <div style={{ marginTop: 8 }}>
        {failed ? (
          <pre style={{
            margin: 0,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--alert-bright)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {reply.error || 'unknown error'}
          </pre>
        ) : (
          <div className="md-reply" style={{ fontSize: 13, lineHeight: 1.55, color: 'var(--ink-bright)' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{reply.content || ''}</ReactMarkdown>
          </div>
        )}
      </div>
    </article>
  );
}

function consensusChipClass(label: string): string {
  switch (label) {
    case 'strong-agreement':  return 'chip-ok';
    case 'majority-agrees':   return 'chip-cta'; // cta maps to process/violet via tokens
    case 'mixed-signals':     return 'chip-warn';
    case 'diverse-opinions':  return 'chip-danger';
    default:                  return 'chip-cta';
  }
}

function EmptyState({ selectedAgent }: { selectedAgent: string | null }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      flex: 1,
      gap: 12,
      padding: 48,
      textAlign: 'center',
    }}>
      <Shield size={32} strokeWidth={1.2} style={{ color: 'var(--ink-faint)' }} />
      <div className="eyebrow">{selectedAgent ? `nothing for ${selectedAgent}` : 'no traffic yet'}</div>
      <p style={{ maxWidth: 360, fontSize: 13, color: 'var(--ink-muted)', lineHeight: 1.6, margin: 0 }}>
        Compose a signed message below or wait for incoming graph handovers from any connected IDE.
      </p>
    </div>
  );
}

function formatTs(ts: string): string {
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return ts;
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return ts;
  }
}

const paneStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  minHeight: 0,
  background: 'var(--bg-void)',
};

const threadHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'flex-end',
  padding: '20px 32px 16px',
  borderBottom: '1px solid var(--rule-faint)',
  flexShrink: 0,
};

const scrollStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '20px 32px',
  display: 'flex',
  flexDirection: 'column',
  gap: 14,
  minHeight: 0,
};

const cardStyle: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderRadius: 6,
  padding: '12px 14px',
};

const cardHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
};

const intentBadgeStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  letterSpacing: 0.04,
  color: 'var(--process-bright)',
  background: 'var(--process-soft)',
  padding: '2px 6px',
  borderRadius: 2,
};

const autoTriggerBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '2px 7px',
  fontSize: 10,
  fontWeight: 500,
  borderRadius: 999,
  background: 'var(--warn-soft)',
  color: 'var(--warn)',
  letterSpacing: 0.04,
  textTransform: 'uppercase',
};

const composerStyle: React.CSSProperties = {
  padding: '14px 32px 18px',
  borderTop: '1px solid var(--rule-default)',
  background: 'var(--bg-deep)',
  flexShrink: 0,
};

const liveBannerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 10,
  fontFamily: 'var(--font-mono)',
  color: 'var(--ink-muted)',
  padding: '6px 10px',
  borderRadius: 3,
  background: 'var(--cta-soft)',
  border: '1px solid var(--rule-faint)',
  marginBottom: 4,
  flexShrink: 0,
};

const replyContentStyle: React.CSSProperties = {
  marginTop: 10,
  padding: '8px 12px',
  borderLeft: '3px solid var(--cta)',
  background: 'var(--cta-soft)',
  borderRadius: '0 4px 4px 0',
  maxHeight: 280,
  overflowY: 'auto',
};

const promptContentStyle: React.CSSProperties = {
  marginTop: 10,
  padding: '6px 10px',
  borderLeft: '2px solid var(--rule-default)',
  maxHeight: 280,
  overflowY: 'auto',
};

const promptPreStyle: React.CSSProperties = {
  margin: 0,
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.5,
  color: 'var(--ink-muted)',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
};

const modeToggleRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  marginBottom: 8,
};

const modeToggleStyle: React.CSSProperties = {
  display: 'inline-flex',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  overflow: 'hidden',
  background: 'var(--bg-elevated)',
};

function modeButtonStyle(active: boolean, isConsensus = false): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '5px 10px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: 0.04,
    textTransform: 'uppercase',
    background: active
      ? (isConsensus ? 'var(--process-soft)' : 'var(--cta-soft)')
      : 'transparent',
    color: active
      ? (isConsensus ? 'var(--process)' : 'var(--cta)')
      : 'var(--ink-muted)',
    border: 'none',
    cursor: 'pointer',
    fontWeight: active ? 600 : 500,
  };
}

const agentMultiSelectStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  marginBottom: 8,
  padding: '6px 8px',
  background: 'var(--bg-elevated)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 4,
};

const consensusBlockStyle: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  borderLeft: '3px solid var(--process)',
  borderRadius: 6,
  padding: '14px 16px',
  flexShrink: 0,
};

const consensusHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
};

const consensusScoreStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '2px 8px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  fontWeight: 600,
  color: 'var(--process)',
  background: 'var(--process-soft)',
  borderRadius: 999,
  letterSpacing: 0.03,
};

const consensusCardStyle: React.CSSProperties = {
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-faint)',
  borderRadius: 6,
  padding: '10px 12px',
  minWidth: 0,
  overflow: 'hidden',
};

const dismissButtonStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '3px 8px',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 0.05,
  color: 'var(--ink-muted)',
  background: 'transparent',
  border: '1px solid var(--rule-default)',
  borderRadius: 4,
  cursor: 'pointer',
};
