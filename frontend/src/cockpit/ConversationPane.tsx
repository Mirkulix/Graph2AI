import { useEffect, useMemo, useRef, useState } from 'react';
import { Radio, Send, Shield, ShieldOff } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, type BusMessage, nameOf, intentOf, bytesToHex } from '../lib/api';
import GraphThumbnail from './GraphThumbnail';

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

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [visible.length]);

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
        {visible.length === 0 ? (
          <EmptyState selectedAgent={selectedAgent} />
        ) : (
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
        )}
      </div>

      <Composer
        defaultTo={selectedAgent ?? agents[0] ?? 'developer'}
        agents={agents}
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
    <article style={cardStyle}>
      <header style={cardHeaderStyle}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: 'var(--ink-bright)' }}>
          {from}
        </span>
        <span style={{ color: 'var(--ink-faint)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>→</span>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: 'var(--ink-bright)' }}>
          {to}
        </span>

        <span style={intentBadgeStyle}>{intent}</span>

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

function Composer({ defaultTo, agents }: { defaultTo: string; agents: string[] }) {
  const [text, setText] = useState('');
  const [target, setTarget] = useState(defaultTo);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  useEffect(() => { setTarget(defaultTo); }, [defaultTo]);

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
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
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
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder="Compose a signed message…   (Cmd/Ctrl+Enter to send)"
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
          onClick={() => void send()}
          disabled={busy || !text.trim()}
          className="surface-button surface-button--primary"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '8px 14px',
            opacity: (busy || !text.trim()) ? 0.4 : 1,
            cursor: (busy || !text.trim()) ? 'not-allowed' : 'pointer',
          }}
        >
          <Send size={12} strokeWidth={2} />
          {busy ? 'signing…' : 'send'}
        </button>
      </div>
    </div>
  );
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
