import { useEffect, useRef } from 'react';
import { Bot, Cable, Compass, Cpu, GitBranch, Network, Settings as SettingsIcon } from 'lucide-react';

export type SecondaryView =
  | 'providers'
  | 'hardware'
  | 'knowledge3d'
  | 'multi-agent'
  | 'swarm'
  | 'autonomous-runs'
  | 'settings';

interface Item {
  id: SecondaryView;
  label: string;
  hint: string;
  icon: typeof Compass;
}

const ITEMS: Item[] = [
  { id: 'providers',       label: 'Providers',       hint: 'configure LLMs (DeepSeek, Claude, OpenAI…)',      icon: Cable },
  { id: 'hardware',        label: 'Hardware',        hint: 'CPU · GPU · memory',                              icon: Cpu },
  { id: 'knowledge3d',     label: 'Knowledge',       hint: 'conversation ledger · timeline · drill-down',     icon: Compass },
  { id: 'multi-agent',     label: 'Multi-Agent',     hint: 'local planner → worker → reviewer product path',  icon: Bot },
  { id: 'swarm',           label: 'Swarm',           hint: 'autonomous multi-agent runs',                     icon: Network },
  { id: 'autonomous-runs', label: 'Autonomous Runs', hint: 'review branches created by overnight swarms',     icon: GitBranch },
  { id: 'settings',        label: 'Settings',        hint: 'token · seed · about',                            icon: SettingsIcon },
];

interface Props {
  open: boolean;
  onSelect: (id: SecondaryView | null) => void;
  onClose: () => void;
  active: SecondaryView | null;
}

export default function ProfileMenu({ open, onSelect, onClose, active }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div ref={ref} style={menuStyle}>
      <div style={{ padding: '10px 14px 6px', borderBottom: '1px solid var(--rule-faint)' }}>
        <div className="eyebrow">secondary views</div>
      </div>
      {active && (
        <button
          onClick={() => onSelect(null)}
          style={{ ...itemStyle, color: 'var(--signal-bright)' }}
        >
          <span style={{ width: 14, color: 'var(--signal-bright)' }}>←</span>
          <div style={{ flex: 1, textAlign: 'left' }}>
            <div style={{ fontSize: 12, fontWeight: 500 }}>Back to cockpit</div>
            <div style={{ fontSize: 10, color: 'var(--ink-faint)' }}>3-pane Inbox view</div>
          </div>
        </button>
      )}
      {ITEMS.map(item => {
        const Icon = item.icon;
        const on = active === item.id;
        return (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            style={{
              ...itemStyle,
              background: on ? 'var(--signal-soft)' : 'transparent',
              color: on ? 'var(--ink-bright)' : 'var(--ink-primary)',
            }}
          >
            <Icon size={14} strokeWidth={1.6} style={{ color: on ? 'var(--signal-bright)' : 'var(--ink-muted)' }} />
            <div style={{ flex: 1, textAlign: 'left' }}>
              <div style={{ fontSize: 12, fontWeight: 500 }}>{item.label}</div>
              <div style={{ fontSize: 10, color: 'var(--ink-faint)' }}>{item.hint}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

const menuStyle: React.CSSProperties = {
  position: 'absolute',
  top: 'calc(var(--topbar-h) + 4px)',
  right: 12,
  width: 280,
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-strong)',
  borderRadius: 6,
  boxShadow: 'var(--shadow-pop)',
  zIndex: 60,
  overflow: 'hidden',
  paddingBottom: 4,
};

const itemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  padding: '8px 14px',
  cursor: 'pointer',
  width: '100%',
  transition: 'background 120ms',
};
