import { useEffect, useState } from 'react';
import { Shield, ChevronDown } from 'lucide-react';
import { api, type BusStats, type KnowledgeHealth } from '../lib/api';

interface Props {
  online: boolean;
  onProfileClick: () => void;
  profileOpen: boolean;
}

export default function TopBar({ online, onProfileClick, profileOpen }: Props) {
  const [stats, setStats] = useState<BusStats | null>(null);
  const [health, setHealth] = useState<KnowledgeHealth | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api.busStats()
        .then(s => { if (alive) setStats(s); })
        .catch(() => { /* silent — pulse will show degraded */ });
      api.knowledgeHealth()
        .then(h => { if (alive) setHealth(h); })
        .catch(() => { /* not available for non-admin or when the graph is empty */ });
    };
    load();
    const t = setInterval(load, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <header style={topbarStyle}>
      <div style={brandStyle}>
        <Logo />
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 500, letterSpacing: -0.01, color: 'var(--ink-bright)' }}>
          Orbit<span style={{ color: 'var(--signal)' }}>QO</span>
        </span>
        <span className="eyebrow" style={{ color: 'var(--ink-faint)' }}>v1.1 / signed</span>
      </div>

      <div style={statusGroupStyle}>
        <Pill
          dot={online ? 'var(--verified)' : 'var(--alert)'}
          label={online ? 'live' : 'offline'}
          mono
        />
        <Pill label={`${stats?.active_agents ?? 0} agents`} />
        <Pill label={`${(stats?.msgs_per_minute ?? 0).toFixed(0)} msg/min`} mono />
        {stats?.uptime_seconds != null && (
          <Pill label={`uptime ${formatUptime(stats.uptime_seconds)}`} mono subtle />
        )}
        {health && (
          <Pill
            dot={health.divergences > 0 ? 'var(--alert)' : 'var(--verified)'}
            label={`${health.load_bearing} reliable${health.divergences > 0 ? ` · ${health.divergences} divergent` : ''}`}
            mono
          />
        )}
      </div>

      <button
        onClick={onProfileClick}
        style={{
          ...profileBtnStyle,
          background: profileOpen ? 'var(--bg-elevated)' : 'transparent',
          borderColor: profileOpen ? 'var(--rule-strong)' : 'var(--rule-default)',
        }}
      >
        <Shield size={13} strokeWidth={1.6} style={{ color: 'var(--verified)' }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-primary)' }}>
          guardian
        </span>
        <ChevronDown size={13} strokeWidth={1.6} style={{ color: 'var(--ink-muted)', transform: profileOpen ? 'rotate(180deg)' : 'none', transition: 'transform 180ms' }} />
      </button>
    </header>
  );
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="11" cy="11" r="3" fill="var(--signal)" />
      <circle cx="11" cy="11" r="7" stroke="var(--signal)" strokeOpacity="0.4" strokeWidth="0.8" />
      <circle cx="11" cy="11" r="10" stroke="var(--verified)" strokeOpacity="0.6" strokeWidth="0.6" strokeDasharray="2 3" />
      <circle cx="18" cy="6" r="1.4" fill="var(--verified)" />
      <circle cx="4" cy="14" r="1" fill="var(--process)" />
    </svg>
  );
}

function Pill({ dot, label, mono, subtle }: { dot?: string; label: string; mono?: boolean; subtle?: boolean }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '3px 8px',
      borderRadius: 4,
      background: subtle ? 'transparent' : 'var(--bg-raised)',
      border: subtle ? 'none' : '1px solid var(--rule-default)',
      fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)',
      fontSize: 11,
      color: subtle ? 'var(--ink-faint)' : 'var(--ink-primary)',
      letterSpacing: mono ? 0.04 : 0,
    }}>
      {dot && (
        <span style={{
          width: 6, height: 6, borderRadius: 3,
          background: dot,
          boxShadow: `0 0 8px ${dot}`,
        }} />
      )}
      {label}
    </span>
  );
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
  return `${Math.floor(sec / 86400)}d`;
}

const topbarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 24,
  height: 'var(--topbar-h)',
  padding: '0 16px',
  background: 'var(--bg-deep)',
  borderBottom: '1px solid var(--rule-default)',
  flexShrink: 0,
};

const brandStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10,
  flexShrink: 0,
};

const statusGroupStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8,
  flex: 1,
  justifyContent: 'center',
};

const profileBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8,
  padding: '5px 10px',
  border: '1px solid var(--rule-default)',
  borderRadius: 6,
  cursor: 'pointer',
  transition: 'all 120ms',
  flexShrink: 0,
};
