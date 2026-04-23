import { useEffect, useState } from 'react';
import { api } from '../../lib/api';
import { SecondaryFrame, Empty } from './FederationView';

const WERTE_LABELS: Record<string, { de: string; color: string }> = {
  achtsamkeit:    { de: 'Achtsamkeit',    color: '#2EA07A' },
  anerkennung:    { de: 'Anerkennung',    color: '#8369D0' },
  aufmerksamkeit: { de: 'Aufmerksamkeit', color: '#D97747' },
  entwicklung:    { de: 'Entwicklung',    color: '#C99216' },
  sinn:           { de: 'Sinn',           color: '#5275D6' },
};

export default function WerteRadar() {
  const [values, setValues] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    let alive = true;
    api.values()
      .then(v => { if (alive) setValues(v); })
      .catch(() => { if (alive) setValues(null); });
  }, []);

  if (!values) {
    return (
      <SecondaryFrame title="Werte" subtitle="Guardian-agent compass — five values track conversational health">
        <Empty text="Loading values…" />
      </SecondaryFrame>
    );
  }

  const keys = Object.keys(WERTE_LABELS);
  const size = 320;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.36;

  const points = keys.map((k, i) => {
    const angle = (Math.PI * 2 * i) / keys.length - Math.PI / 2;
    const value = values[k] ?? 0;
    const r = radius * Math.min(Math.max(value, 0), 1);
    return {
      key: k,
      angle,
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
      labelX: cx + Math.cos(angle) * (radius + 28),
      labelY: cy + Math.sin(angle) * (radius + 28),
      value,
    };
  });

  const polygon = points.map(p => `${p.x},${p.y}`).join(' ');

  return (
    <SecondaryFrame title="Werte" subtitle="Guardian-agent compass — five values track conversational health">
      <div style={{ display: 'flex', justifyContent: 'center', padding: 16 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {[0.25, 0.5, 0.75, 1].map(scale => (
            <polygon
              key={scale}
              points={keys.map((_, i) => {
                const angle = (Math.PI * 2 * i) / keys.length - Math.PI / 2;
                return `${cx + Math.cos(angle) * radius * scale},${cy + Math.sin(angle) * radius * scale}`;
              }).join(' ')}
              fill="none"
              stroke="var(--rule-default)"
              strokeWidth="0.5"
            />
          ))}
          {keys.map((k, i) => {
            const angle = (Math.PI * 2 * i) / keys.length - Math.PI / 2;
            return (
              <line
                key={k}
                x1={cx} y1={cy}
                x2={cx + Math.cos(angle) * radius}
                y2={cy + Math.sin(angle) * radius}
                stroke="var(--rule-default)"
                strokeWidth="0.5"
              />
            );
          })}
          <polygon
            points={polygon}
            fill="var(--signal-soft)"
            stroke="var(--signal)"
            strokeWidth="1.5"
          />
          {points.map(p => (
            <g key={p.key}>
              <circle cx={p.x} cy={p.y} r={3} fill={WERTE_LABELS[p.key].color} />
              <text
                x={p.labelX}
                y={p.labelY}
                textAnchor="middle"
                style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fill: 'var(--ink-muted)' }}
              >
                {WERTE_LABELS[p.key].de}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 16 }}>
        {keys.map(k => (
          <div key={k} style={{
            padding: '10px 12px',
            background: 'var(--bg-panel)',
            border: '1px solid var(--rule-default)',
            borderRadius: 4,
            textAlign: 'center',
          }}>
            <div className="eyebrow" style={{ fontSize: 9 }}>{WERTE_LABELS[k].de}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: WERTE_LABELS[k].color, marginTop: 4 }}>
              {((values[k] ?? 0) * 100).toFixed(0)}
            </div>
          </div>
        ))}
      </div>
    </SecondaryFrame>
  );
}
