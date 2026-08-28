// Integrations — which coding systems are attached to this control plane.
//
// This is the cockpit's answer to "who is driving the graph": Claude Code,
// Codex, Gemini, deepseek-harness, or any other MCP client. It is deliberately
// NOT the Providers view: a provider is a model QO calls outward, a harness is
// a system that calls into QO and runs its tools.
//
// The list is fed passively by MCP traffic (handshake + tool calls), so what
// is shown is what actually happened — never a declared configuration. A
// system QO does not recognise still appears, under its own reported name;
// that is the extension path for a modified deepseek-harness or an in-house
// runner, and the connect card documents it.

import { useEffect, useMemo, useState } from 'react';
import { Check, Copy, Plug, Terminal } from 'lucide-react';
import { api, type HarnessOverview, type HarnessSession } from '../../lib/api';
import { SecondaryFrame, Empty, Stat } from './SecondaryFrame';

const POLL_MS = 5000;

export default function IntegrationsView() {
  const [data, setData] = useState<HarnessOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api.harnesses()
        .then(next => {
          if (!alive) return;
          setData(next);
          setError(null);
          setLoading(false);
        })
        .catch((e: unknown) => {
          if (!alive) return;
          // Keep the last good picture rather than blanking the view on one
          // dropped poll; only a first-load failure is surfaced.
          //
          // Distinguish auth from unreachable: a 401 here means the cockpit
          // has no token, which is a different fix from "start the server",
          // and reporting it as the latter sends the operator down the wrong path.
          const msg = e instanceof Error ? e.message : String(e);
          setError(
            /HTTP 40[13]/.test(msg)
              ? 'Not authorised. Set the API token in Settings, then reload.'
              : 'Cannot reach the server. Is qo running?',
          );
          setLoading(false);
        });
    };
    load();
    const t = setInterval(load, POLL_MS);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const endpoint = useMemo(() => {
    const base = window.location.origin;
    return `${base}${data?.mcp_endpoint ?? '/mcp/v1'}`;
  }, [data?.mcp_endpoint]);

  if (loading) {
    return (
      <SecondaryFrame title="Integrations" subtitle="coding systems attached to this control plane">
        <div style={{ padding: 24, fontSize: 12, color: 'var(--ink-muted)' }}>loading…</div>
      </SecondaryFrame>
    );
  }

  if (error && !data) {
    return (
      <SecondaryFrame title="Integrations" subtitle="coding systems attached to this control plane">
        <Empty text={error} />
      </SecondaryFrame>
    );
  }

  const sessions = data?.sessions ?? [];
  const online = data?.online ?? 0;

  return (
    <SecondaryFrame
      title="Integrations"
      subtitle="which coding systems are driving this graph — fed by real MCP traffic, not configuration"
    >
      <div style={{ display: 'flex', gap: 10, marginBottom: 22 }}>
        <Stat label="attached now" value={String(online)} accent={online > 0} />
        <Stat label="seen this session" value={String(sessions.length)} />
        <Stat label="tools exposed" value={String(data?.tools ?? 0)} />
      </div>

      {/* Supported systems: shows what CAN attach, not only what has. An
          operator seeing "Codex — not attached" learns the integration exists. */}
      <SectionLabel>supported systems</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8, marginBottom: 26 }}>
        {(data?.known_kinds ?? []).map(k => (
          <div
            key={k.kind}
            style={{
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              border: `1px solid ${k.attached ? 'var(--verified-bright)' : 'var(--rule-default)'}`,
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <span
              aria-hidden
              style={{
                width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                background: k.attached ? 'var(--verified-bright)' : 'var(--ink-faint)',
              }}
            />
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 12, color: 'var(--ink-bright)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {k.label}
              </div>
              <div className="eyebrow" style={{ color: k.attached ? 'var(--verified-bright)' : 'var(--ink-faint)' }}>
                {k.attached ? 'attached' : 'not attached'}
              </div>
            </div>
          </div>
        ))}
      </div>

      <SectionLabel>sessions</SectionLabel>
      {sessions.length === 0 ? (
        <Empty text="No coding system has connected yet. Point one at the MCP endpoint below." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 26 }}>
          {sessions.map(s => <SessionRow key={s.id} session={s} />)}
        </div>
      )}

      <ConnectCard endpoint={endpoint} />
    </SecondaryFrame>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="eyebrow" style={{ color: 'var(--ink-faint)', marginBottom: 8 }}>
      {children}
    </div>
  );
}

function SessionRow({ session }: { session: HarnessSession }) {
  const idle = !session.online;
  return (
    <div
      style={{
        padding: '11px 13px',
        background: 'var(--bg-panel)',
        border: '1px solid var(--rule-default)',
        borderRadius: 4,
        opacity: idle ? 0.55 : 1,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span
          aria-hidden
          style={{
            width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
            background: session.online ? 'var(--ok, var(--verified-bright))' : 'var(--ink-faint)',
          }}
        />
        <span style={{ fontSize: 13, color: 'var(--ink-bright)' }}>{session.label}</span>
        {session.version && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)' }}>
            v{session.version}
          </span>
        )}
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-muted)' }}>
          {session.calls} {session.calls === 1 ? 'call' : 'calls'}
        </span>
        <span className="eyebrow" style={{ color: session.online ? 'var(--verified-bright)' : 'var(--ink-faint)' }}>
          {session.online ? 'live' : relative(session.last_seen_at)}
        </span>
      </div>

      {/* The id is only worth showing when it differs from the label — for a
          recognised product they are the same and repeating it is noise. */}
      {session.label !== session.id && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)', marginTop: 4 }}>
          {session.id}
        </div>
      )}

      {session.recent_tools.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          {session.recent_tools.map(tool => (
            <span
              key={tool}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                padding: '2px 6px',
                borderRadius: 3,
                background: 'var(--bg-deep)',
                border: '1px solid var(--rule-faint)',
                color: 'var(--ink-muted)',
              }}
            >
              {tool}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// How to attach a system. Concrete commands beat prose here: the operator's
// next action is pasting one of these.
function ConnectCard({ endpoint }: { endpoint: string }) {
  const snippets: { label: string; icon: typeof Terminal; code: string }[] = [
    {
      label: 'Claude Code',
      icon: Terminal,
      code: `claude mcp add --transport http orbitq ${endpoint}`,
    },
    {
      label: 'Codex / Gemini / any MCP client',
      icon: Plug,
      code: `{
  "mcpServers": {
    "orbitq": { "type": "http", "url": "${endpoint}" }
  }
}`,
    },
    {
      label: 'deepseek-harness (or your own runner)',
      icon: Plug,
      code: `POST ${endpoint}
{"jsonrpc":"2.0","id":1,"method":"initialize",
 "params":{"clientInfo":{"name":"deepseek-harness","version":"1.0"}}}`,
    },
  ];

  return (
    <div
      style={{
        padding: '14px 16px',
        background: 'var(--bg-panel)',
        border: '1px solid var(--rule-default)',
        borderRadius: 4,
      }}
    >
      <div style={{ fontSize: 13, color: 'var(--ink-bright)', marginBottom: 4 }}>
        Attach a coding system
      </div>
      <p style={{ fontSize: 12, color: 'var(--ink-muted)', margin: '0 0 14px', lineHeight: 1.5 }}>
        Any MCP client can attach. It appears here the first time it calls a tool —
        a client QO does not recognise is listed under the name it reports, so an
        extended <code style={{ fontFamily: 'var(--font-mono)' }}>deepseek-harness</code> or
        an in-house runner needs no change on this side.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {snippets.map(s => <Snippet key={s.label} {...s} />)}
      </div>
    </div>
  );
}

function Snippet({ label, icon: Icon, code }: { label: string; icon: typeof Terminal; code: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard is unavailable in some embedded contexts; the code is
      // selectable either way, so a failure needs no error state.
    }
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <Icon size={12} style={{ color: 'var(--ink-faint)' }} aria-hidden />
        <span className="eyebrow" style={{ color: 'var(--ink-faint)' }}>{label}</span>
        <span style={{ flex: 1 }} />
        <button
          onClick={copy}
          className="surface-button"
          style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, padding: '2px 7px' }}
          aria-label={`Copy the ${label} snippet`}
        >
          {copied ? <Check size={11} aria-hidden /> : <Copy size={11} aria-hidden />}
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre
        style={{
          margin: 0,
          padding: '9px 11px',
          background: 'var(--bg-deep)',
          border: '1px solid var(--rule-faint)',
          borderRadius: 3,
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--ink-muted)',
          overflowX: 'auto',
          whiteSpace: 'pre',
          lineHeight: 1.5,
        }}
      >
        {code}
      </pre>
    </div>
  );
}

function relative(unixSeconds: number): string {
  const delta = Math.max(0, Math.floor(Date.now() / 1000) - unixSeconds);
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}
