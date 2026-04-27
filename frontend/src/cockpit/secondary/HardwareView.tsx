import { useEffect, useState } from 'react';
import { Cpu, HardDrive } from 'lucide-react';
import { api } from '../../lib/api';
import { SecondaryFrame, Stat, Empty } from './SecondaryFrame';

interface HardwareData {
  cpu?: { model?: string; cores?: number; load?: number };
  memory?: { total_mb?: number; used_mb?: number };
  gpu?: Array<{ name?: string; temp_c?: number; util?: number; mem_mb?: number }>;
}

export default function HardwareView() {
  const [hw, setHw] = useState<HardwareData | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api.hardware()
        .then(h => { if (alive) setHw(h); })
        .catch(() => { if (alive) setHw(null); });
    };
    load();
    const t = setInterval(load, 3000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (!hw) {
    return (
      <SecondaryFrame title="hardware" subtitle="host telemetry — CPU, memory, GPU">
        <Empty text="Loading hardware metrics…" />
      </SecondaryFrame>
    );
  }

  const memUsed = hw.memory?.used_mb ?? 0;
  const memTotal = hw.memory?.total_mb ?? 0;
  const memPct = memTotal > 0 ? Math.round((memUsed / memTotal) * 100) : 0;

  return (
    <SecondaryFrame title="hardware" subtitle="host telemetry — CPU, memory, GPU">
      <section style={{ marginBottom: 24 }}>
        <h3 className="eyebrow" style={{ margin: '0 0 8px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Cpu size={11} strokeWidth={1.6} />
          processor
        </h3>
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          <Stat label="cores" value={String(hw.cpu?.cores ?? '—')} />
          <Stat label="load"  value={hw.cpu?.load != null ? `${(hw.cpu.load * 100).toFixed(0)}%` : '—'} accent />
          <Stat label="ram"   value={`${memPct}%`} />
        </div>
        {hw.cpu?.model && (
          <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
            {hw.cpu.model}
          </div>
        )}
        {memTotal > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="eyebrow" style={{ fontSize: 9, marginBottom: 4 }}>memory</div>
            <div style={{
              width: '100%', height: 6, background: 'var(--bg-panel)', borderRadius: 3, overflow: 'hidden',
              border: '1px solid var(--rule-default)',
            }}>
              <div style={{
                width: `${memPct}%`, height: '100%',
                background: memPct > 80 ? 'var(--alert)' : memPct > 60 ? 'var(--caution)' : 'var(--verified)',
                transition: 'width 480ms',
              }} />
            </div>
            <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', marginTop: 4 }}>
              {(memUsed / 1024).toFixed(1)} / {(memTotal / 1024).toFixed(1)} GB
            </div>
          </div>
        )}
      </section>

      <section>
        <h3 className="eyebrow" style={{ margin: '0 0 8px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <HardDrive size={11} strokeWidth={1.6} />
          gpu · {hw.gpu?.length ?? 0}
        </h3>
        {!hw.gpu || hw.gpu.length === 0 ? (
          <Empty text="No GPU detected." />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {hw.gpu.map((g, i) => (
              <div key={i} style={{
                padding: '10px 12px',
                background: 'var(--bg-panel)',
                border: '1px solid var(--rule-default)',
                borderRadius: 4,
              }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, color: 'var(--ink-bright)' }}>{g.name ?? `GPU ${i}`}</span>
                  {g.temp_c != null && (
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 11,
                      color: g.temp_c > 80 ? 'var(--alert)' : g.temp_c > 65 ? 'var(--caution)' : 'var(--verified)',
                      marginLeft: 'auto',
                    }}>
                      {g.temp_c}°C
                    </span>
                  )}
                </div>
                {g.util != null && (
                  <div style={{
                    width: '100%', height: 4, background: 'var(--bg-deep)', borderRadius: 2, overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${Math.min(100, g.util * 100)}%`, height: '100%',
                      background: 'var(--process)',
                    }} />
                  </div>
                )}
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', marginTop: 4 }}>
                  {g.util != null && `util ${(g.util * 100).toFixed(0)}%`}
                  {g.mem_mb != null && ` · ${(g.mem_mb / 1024).toFixed(1)} GB`}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </SecondaryFrame>
  );
}
