import { useEffect, useState } from 'react';
import { PanelRightClose, PanelRightOpen } from 'lucide-react';
import { api, type BusMessage, type PresenceEntry } from '../lib/api';
import GraphInspector from './detail/GraphInspector';
import AgentStats from './detail/AgentStats';
import IdePresenceDetail from './detail/IdePresenceDetail';
import ProvidersDetail from './detail/ProvidersDetail';

export type DetailContext =
  | { kind: 'empty' }
  | { kind: 'graph'; message: BusMessage }
  | { kind: 'agent'; name: string }
  | { kind: 'providers' };

interface Props {
  ctx: DetailContext;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onClose: () => void;
}

export default function DetailPane({ ctx, collapsed, onToggleCollapse, onClose }: Props) {
  if (collapsed) {
    return (
      <div style={{
        width: 28,
        borderLeft: '1px solid var(--rule-default)',
        background: 'var(--bg-deep)',
        display: 'flex',
        justifyContent: 'center',
        padding: '10px 0',
        flexShrink: 0,
      }}>
        <button onClick={onToggleCollapse} style={{ color: 'var(--ink-muted)' }}>
          <PanelRightOpen size={14} strokeWidth={1.6} />
        </button>
      </div>
    );
  }

  const heading =
    ctx.kind === 'graph'    ? 'graph inspector' :
    ctx.kind === 'agent'    ? 'agent stats' :
    ctx.kind === 'providers' ? 'providers' :
    'detail';

  return (
    <aside style={paneStyle}>
      <header style={headerStyle}>
        <span className="eyebrow" style={{ color: 'var(--ink-muted)' }}>{heading}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {ctx.kind !== 'empty' && (
            <button onClick={onClose} style={iconBtnStyle} title="clear">
              <span style={{ fontSize: 12, color: 'var(--ink-faint)' }}>×</span>
            </button>
          )}
          <button onClick={onToggleCollapse} style={iconBtnStyle} title="collapse">
            <PanelRightClose size={13} strokeWidth={1.6} style={{ color: 'var(--ink-muted)' }} />
          </button>
        </div>
      </header>

      <div style={scrollStyle}>
        {ctx.kind === 'empty'     && <EmptyDetail />}
        {ctx.kind === 'graph'     && <GraphInspector message={ctx.message} />}
        {ctx.kind === 'agent'     && <AgentDetailRouter identity={ctx.name} />}
        {ctx.kind === 'providers' && <ProvidersDetail />}
      </div>
    </aside>
  );
}

function AgentDetailRouter({ identity }: { identity: string }) {
  const [presenceList, setPresenceList] = useState<PresenceEntry[] | null>(null);
  useEffect(() => {
    let alive = true;
    api.presence()
      .then(list => { if (alive) setPresenceList(list); })
      .catch(() => { if (alive) setPresenceList([]); });
    return () => { alive = false; };
  }, [identity]);
  if (presenceList === null) {
    return <div style={{ padding: 16, color: 'var(--ink-muted)', fontSize: 12 }}>resolving…</div>;
  }
  const isIde = presenceList.some(p => p.identity === identity);
  if (isIde) {
    return <IdePresenceDetail identity={identity} presenceCache={presenceList} />;
  }
  return <AgentStats name={identity} />;
}

function EmptyDetail() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: 10,
      padding: 32,
      textAlign: 'center',
    }}>
      <div style={{
        width: 48,
        height: 48,
        borderRadius: '50%',
        border: '1px dashed var(--rule-strong)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--ink-ghost)',
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
      }}>
        —
      </div>
      <p style={{ fontSize: 12, color: 'var(--ink-faint)', maxWidth: 240, lineHeight: 1.6, margin: 0 }}>
        Click an agent or a graph preview to inspect here.
      </p>
    </div>
  );
}

const paneStyle: React.CSSProperties = {
  width: 'var(--pane-right-w)',
  flexShrink: 0,
  background: 'var(--bg-deep)',
  borderLeft: '1px solid var(--rule-default)',
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  minHeight: 0,
};

const headerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  padding: '10px 14px',
  borderBottom: '1px solid var(--rule-faint)',
  flexShrink: 0,
};

const scrollStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '14px 16px',
  minHeight: 0,
};

const iconBtnStyle: React.CSSProperties = {
  width: 22,
  height: 22,
  borderRadius: 3,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  transition: 'background 120ms',
};
