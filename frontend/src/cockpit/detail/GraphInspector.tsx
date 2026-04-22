import { useState } from 'react';
import { type BusMessage, nameOf, intentOf, bytesToHex } from '../../lib/api';

export default function GraphInspector({ message }: { message: BusMessage }) {
  const [showRaw, setShowRaw] = useState(false);
  const sigHex = bytesToHex(message.signature, 16);
  const pubHex = bytesToHex(message.signer_pubkey, 16);
  const hashHex = bytesToHex(message.graph_hash, 16);

  const graph = message.graph as Record<string, unknown> | null;
  const nodes = Array.isArray(graph?.nodes) ? (graph!.nodes as unknown[]) : [];
  const edges = Array.isArray(graph?.edges) ? (graph!.edges as unknown[]) : [];

  return (
    <div>
      <Section title="message">
        <Row k="from"   v={nameOf(message.from)} />
        <Row k="to"     v={nameOf(message.to)} />
        <Row k="intent" v={intentOf(message.intent)} accent />
        {message.id != null && <Row k="id" v={String(message.id)} mono />}
        {message.in_reply_to != null && <Row k="reply→" v={String(message.in_reply_to)} mono />}
      </Section>

      <Section title="signature">
        {sigHex ? (
          <Row k="sig" v={sigHex + '…'} mono accent />
        ) : (
          <Row k="sig" v="unsigned" warning />
        )}
        {pubHex && <Row k="pub" v={pubHex + '…'} mono />}
        {hashHex && <Row k="hash" v={hashHex + '…'} mono />}
        {message.signature_verified !== undefined && (
          <Row k="verified" v={message.signature_verified ? 'yes' : 'no'} accent={message.signature_verified} warning={!message.signature_verified} />
        )}
      </Section>

      <Section title="graph">
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          <Stat label="nodes" value={nodes.length} />
          <Stat label="edges" value={edges.length} />
        </div>

        {nodes.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="eyebrow" style={{ marginBottom: 4, fontSize: 9 }}>nodes</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {(nodes as Array<Record<string, unknown>>).slice(0, 12).map((n, i) => (
                <NodeRow key={i} index={i} node={n} />
              ))}
              {nodes.length > 12 && (
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-faint)', padding: '4px 0' }}>
                  +{nodes.length - 12} more
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      <Section title="raw envelope">
        <button
          onClick={() => setShowRaw(s => !s)}
          className="surface-button"
          style={{ marginBottom: 8 }}
        >
          {showRaw ? 'hide' : 'show'} JSON
        </button>
        {showRaw && (
          <pre style={{
            margin: 0,
            padding: 10,
            background: 'var(--bg-void)',
            border: '1px solid var(--rule-default)',
            borderRadius: 4,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-muted)',
            maxHeight: 280,
            overflow: 'auto',
            lineHeight: 1.5,
          }}>
            {JSON.stringify(message, null, 2)}
          </pre>
        )}
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 18 }}>
      <h3 className="eyebrow" style={{ margin: 0, marginBottom: 8 }}>{title}</h3>
      <div>{children}</div>
    </section>
  );
}

function Row({ k, v, mono, accent, warning }: { k: string; v: string; mono?: boolean; accent?: boolean; warning?: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 10, padding: '3px 0', alignItems: 'baseline' }}>
      <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-faint)', width: 56, flexShrink: 0 }}>
        {k}
      </span>
      <span style={{
        fontSize: 11,
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)',
        color: warning ? 'var(--caution)' : accent ? 'var(--verified-bright)' : 'var(--ink-primary)',
        wordBreak: 'break-all',
      }}>
        {v}
      </span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{
      flex: 1,
      padding: '8px 10px',
      background: 'var(--bg-panel)',
      border: '1px solid var(--rule-default)',
      borderRadius: 4,
    }}>
      <div className="eyebrow" style={{ fontSize: 9 }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: 'var(--ink-bright)', marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

function NodeRow({ index, node }: { index: number; node: Record<string, unknown> }) {
  const op = typeof node.op === 'string' ? node.op : (typeof node.label === 'string' ? node.label : '?');
  const id = typeof node.id !== 'undefined' ? String(node.id) : `#${index}`;
  return (
    <div style={{
      display: 'flex',
      gap: 8,
      padding: '4px 6px',
      background: 'var(--bg-panel)',
      borderRadius: 3,
      fontSize: 10,
      fontFamily: 'var(--font-mono)',
    }}>
      <span style={{ color: 'var(--ink-faint)', width: 24 }}>{id}</span>
      <span style={{ color: 'var(--process-bright)', flex: 1 }}>{op}</span>
    </div>
  );
}
