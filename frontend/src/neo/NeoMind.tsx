// NeoMind — holographic visualization of the QLANG Organism.
// Data sources (real API only):
//   GET  /api/evolution/status + /api/evolution/specialists (preferred, 1Hz)
//   GET  /api/organism/status  (fallback, 1Hz)
//   GET  /api/messages/stream  (SSE bus)
// Pure React + inline SVG, custom force sim @ 60fps, no external deps.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------
interface ApiSpecialist { name: string; invocations: number; success_rate: number }
interface OrganismStatus { generation: number; specialists: ApiSpecialist[]; total_interactions: number; memory_items: number }
type SpecialistStatus = "Active" | "Retired" | "Dead";
interface EvolutionSpecialist {
  id: string; generation_born: number; parent_id: string | null; children: string[];
  fitness: number; age: number; status: SpecialistStatus; mutations: number;
}
interface GenerationReport {
  generation: number; best_fitness: number; avg_fitness: number; min_fitness: number;
  best_id: string; killed: string[];
}
interface EvolutionStatus {
  initialized: boolean; running?: boolean; current_generation?: number;
  population_size?: number; best_fitness_ever?: number;
  total_born?: number; total_retired?: number; total_killed?: number;
  last_report?: GenerationReport | null;
}
interface BusMessage { id: number; from: string; to: string; intent: string; graph_name: string; timestamp: number }

// ---------------------------------------------------------------------------
// Sim types
// ---------------------------------------------------------------------------
interface SimNode {
  id: string; label: string; role: string; isOrchestrator: boolean;
  invocations: number; successRate: number; lastActivity: number;
  x: number; y: number; vx: number; vy: number;
  fitness: number; age: number; generationBorn: number;
  status: SpecialistStatus; parentId: string | null; rankPct: number;
  deathAt: number | null;
  invocationsPerSec: number; lastInvocations: number; lastInvocationsTs: number;
}
interface SimEdge { from: string; to: string; birth: number; intent: string }
interface DetailState { id: string; messages: BusMessage[] }
interface HoverState { id: string }

// ---------------------------------------------------------------------------
// Role + color helpers
// ---------------------------------------------------------------------------
const ROLE_COLORS: Record<string, string> = {
  orchestrator: "#00e5ff", classifier: "#ff2bd6", memory: "#b388ff",
  language: "#ffd54a", math: "#64ffda", generic: "#7dffb0",
};
function roleOf(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("orchestr")) return "orchestrator";
  if (n.includes("class") || n.includes("topic")) return "classifier";
  if (n.includes("mem") || n.includes("recall")) return "memory";
  if (n.includes("lm") || n.includes("mamba") || n.includes("lang")) return "language";
  if (n.includes("math") || n.includes("calc")) return "math";
  return "generic";
}
const roleColor = (role: string) => ROLE_COLORS[role] ?? ROLE_COLORS.generic;

// Fitness → red 0 → yellow 0.5 → green 1.0
function fitnessColor(f: number): string {
  const x = Math.max(0, Math.min(1, f));
  if (x < 0.5) {
    const t = x / 0.5;
    return `rgb(255,${Math.round(80 + 175 * t)},60)`;
  }
  const t = (x - 0.5) / 0.5;
  return `rgb(${Math.round(255 - 175 * t)},230,${Math.round(60 + 80 * t)})`;
}
const statusGlow = (s: SpecialistStatus) =>
  s === "Dead" ? "#ff3b5c" : s === "Retired" ? "#8892b0" : "#00e5ff";

const WIDTH = 960, HEIGHT = 620, CX = WIDTH / 2, CY = HEIGHT / 2;

// ---------------------------------------------------------------------------
// Force sim
// ---------------------------------------------------------------------------
function stepSim(nodes: SimNode[], edges: SimEdge[], dt: number) {
  const REPULSION = 9000, CENTER_PULL = 0.008, ORCH_PULL = 0.012;
  const DAMPING = 0.82, EDGE_SPRING = 0.02, EDGE_REST = 180;
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const d2 = dx * dx + dy * dy + 0.01;
      const d = Math.sqrt(d2), f = REPULSION / d2;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      if (!a.isOrchestrator) { a.vx += fx * dt; a.vy += fy * dt; }
      if (!b.isOrchestrator) { b.vx -= fx * dt; b.vy -= fy * dt; }
    }
  }
  const orch = nodes.find(n => n.isOrchestrator);
  if (orch) { orch.x = CX; orch.y = CY; orch.vx = 0; orch.vy = 0; }
  for (const n of nodes) {
    if (n.isOrchestrator) continue;
    n.vx += (CX - n.x) * CENTER_PULL * dt;
    n.vy += (CY - n.y) * CENTER_PULL * dt;
    if (orch) { n.vx += (orch.x - n.x) * ORCH_PULL * dt; n.vy += (orch.y - n.y) * ORCH_PULL * dt; }
  }
  const now = performance.now();
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const e of edges) {
    if (now - e.birth > 1500) continue;
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
    const delta = (d - EDGE_REST) * EDGE_SPRING;
    const fx = (dx / d) * delta, fy = (dy / d) * delta;
    if (!a.isOrchestrator) { a.vx += fx * dt; a.vy += fy * dt; }
    if (!b.isOrchestrator) { b.vx -= fx * dt; b.vy -= fy * dt; }
  }
  for (const n of nodes) {
    if (n.isOrchestrator) continue;
    n.vx *= DAMPING; n.vy *= DAMPING;
    n.x += n.vx * dt; n.y += n.vy * dt;
    const p = 40;
    if (n.x < p) { n.x = p; n.vx = 0; }
    if (n.x > WIDTH - p) { n.x = WIDTH - p; n.vx = 0; }
    if (n.y < p) { n.y = p; n.vy = 0; }
    if (n.y > HEIGHT - p) { n.y = HEIGHT - p; n.vy = 0; }
  }
}

function makeOrchestrator(): SimNode {
  return { id: "orchestrator", label: "Orchestrator", role: "orchestrator", isOrchestrator: true,
    invocations: 0, successRate: 1, lastActivity: 0, x: CX, y: CY, vx: 0, vy: 0,
    fitness: 1, age: 0, generationBorn: 0, status: "Active", parentId: null, rankPct: 1,
    deathAt: null, invocationsPerSec: 0, lastInvocations: 0, lastInvocationsTs: 0 };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export const NeoMind: React.FC = () => {
  const [status, setStatus] = useState<OrganismStatus | null>(null);
  const [evoStatus, setEvoStatus] = useState<EvolutionStatus | null>(null);
  const [prevBestFitness, setPrevBestFitness] = useState<number | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);
  const [detail, setDetail] = useState<DetailState | null>(null);
  const [, setTick] = useState(0);

  const nodesRef = useRef<SimNode[]>([]);
  const edgesRef = useRef<SimEdge[]>([]);
  const messagesRef = useRef<BusMessage[]>([]);
  const activityRef = useRef<Record<string, number>>({});
  const prevStatusRef = useRef<Record<string, SpecialistStatus>>({});

  const syncFromEvolution = useCallback((evo: EvolutionSpecialist[], gen: number) => {
    const sorted = [...evo].sort((a, b) => b.fitness - a.fitness);
    const rankMap = new Map<string, number>();
    sorted.forEach((s, i) => rankMap.set(s.id, sorted.length > 1 ? 1 - i / (sorted.length - 1) : 1));
    const existing = new Map(nodesRef.current.map(n => [n.id, n]));
    const next: SimNode[] = [];
    const orch = existing.get("orchestrator") ?? makeOrchestrator();
    next.push(orch);
    const nowMs = performance.now();
    evo.forEach((s, i) => {
      const prev = existing.get(s.id);
      const angle = (i / Math.max(1, evo.length)) * Math.PI * 2;
      const r = 230;
      const prevStatus = prevStatusRef.current[s.id];
      const becameDead = s.status !== "Active" && prevStatus === "Active";
      const deathAt = s.status !== "Active"
        ? (prev?.deathAt ?? (becameDead ? nowMs : nowMs))
        : null;
      next.push({
        id: s.id, label: s.id, role: roleOf(s.id), isOrchestrator: false,
        invocations: prev?.invocations ?? 0, successRate: s.fitness, lastActivity: prev?.lastActivity ?? 0,
        x: prev?.x ?? CX + Math.cos(angle) * r, y: prev?.y ?? CY + Math.sin(angle) * r,
        vx: prev?.vx ?? 0, vy: prev?.vy ?? 0, fitness: s.fitness,
        age: Math.max(0, gen - s.generation_born), generationBorn: s.generation_born,
        status: s.status, parentId: s.parent_id, rankPct: rankMap.get(s.id) ?? 0, deathAt,
        invocationsPerSec: prev?.invocationsPerSec ?? 0,
        lastInvocations: prev?.lastInvocations ?? 0, lastInvocationsTs: prev?.lastInvocationsTs ?? nowMs,
      });
      prevStatusRef.current[s.id] = s.status;
    });
    nodesRef.current = next;
  }, []);

  const syncFromOrganism = useCallback((data: OrganismStatus) => {
    const existing = new Map(nodesRef.current.map(n => [n.id, n]));
    const nowMs = performance.now();
    const ranks = [...data.specialists].sort((a, b) => b.success_rate - a.success_rate);
    const rankMap = new Map<string, number>();
    ranks.forEach((s, i) => rankMap.set(s.name, ranks.length > 1 ? 1 - i / (ranks.length - 1) : 1));
    const next: SimNode[] = [];
    const orch = existing.get("orchestrator") ?? makeOrchestrator();
    orch.invocations = data.total_interactions;
    next.push(orch);
    data.specialists.forEach((s, i) => {
      const prev = existing.get(s.name);
      const angle = (i / Math.max(1, data.specialists.length)) * Math.PI * 2;
      const r = 230;
      const dtSec = Math.max(0.25, (nowMs - (prev?.lastInvocationsTs ?? nowMs)) / 1000);
      const delta = Math.max(0, s.invocations - (prev?.lastInvocations ?? s.invocations));
      const ips = prev ? delta / dtSec : 0;
      next.push({
        id: s.name, label: s.name, role: roleOf(s.name), isOrchestrator: false,
        invocations: s.invocations, successRate: s.success_rate, lastActivity: prev?.lastActivity ?? 0,
        x: prev?.x ?? CX + Math.cos(angle) * r, y: prev?.y ?? CY + Math.sin(angle) * r,
        vx: prev?.vx ?? 0, vy: prev?.vy ?? 0,
        fitness: Math.max(0, Math.min(1, s.success_rate)), age: prev?.age ?? 0,
        generationBorn: prev?.generationBorn ?? (data.generation ?? 0),
        status: "Active", parentId: null, rankPct: rankMap.get(s.name) ?? 0, deathAt: null,
        invocationsPerSec: ips, lastInvocations: s.invocations, lastInvocationsTs: nowMs,
      });
    });
    nodesRef.current = next;
  }, []);

  // Polling — evolution preferred, organism fallback
  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      try {
        const [evoStatusRes, evoSpecRes] = await Promise.all([
          fetch("/api/evolution/status").catch(() => null),
          fetch("/api/evolution/specialists").catch(() => null),
        ]);
        let usedEvolution = false;
        if (evoStatusRes?.ok && evoSpecRes?.ok) {
          const st: EvolutionStatus = await evoStatusRes.json();
          const sp: EvolutionSpecialist[] = await evoSpecRes.json();
          if (st.initialized && sp.length > 0) {
            if (cancelled) return;
            setEvoStatus(prev => {
              if (prev?.last_report && st.last_report &&
                  prev.last_report.generation !== st.last_report.generation) {
                setPrevBestFitness(prev.last_report.best_fitness);
              }
              return st;
            });
            syncFromEvolution(sp, st.current_generation ?? 0);
            setStatusError(null);
            usedEvolution = true;
          }
        }
        if (!usedEvolution) {
          const res = await fetch("/api/organism/status");
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data: OrganismStatus = await res.json();
          if (cancelled) return;
          setStatus(data); setEvoStatus(null);
          syncFromOrganism(data); setStatusError(null);
        } else {
          try {
            const res = await fetch("/api/organism/status");
            if (res.ok) setStatus(await res.json());
          } catch { /* ignore */ }
        }
      } catch (err) {
        if (!cancelled) setStatusError((err as Error).message);
      }
    };
    fetchAll();
    const id = window.setInterval(fetchAll, 1000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [syncFromEvolution, syncFromOrganism]);

  // SSE bus
  useEffect(() => {
    const es = new EventSource("/api/messages/stream");
    es.onmessage = (ev) => {
      try {
        const msg: BusMessage = JSON.parse(ev.data);
        const buf = messagesRef.current;
        buf.push(msg);
        if (buf.length > 500) buf.shift();
        activityRef.current[msg.from] = performance.now();
        activityRef.current[msg.to] = performance.now();
        edgesRef.current.push({ from: msg.from, to: msg.to, birth: performance.now(), intent: msg.intent });
        if (edgesRef.current.length > 200) edgesRef.current.shift();
      } catch { /* ignore */ }
    };
    return () => es.close();
  }, []);

  // 60fps loop
  useEffect(() => {
    let raf = 0, last = performance.now();
    const loop = (t: number) => {
      const dt = Math.min(2, (t - last) / 16.67); last = t;
      const act = activityRef.current;
      for (const n of nodesRef.current) {
        const a = act[n.id];
        if (a && a > n.lastActivity) n.lastActivity = a;
      }
      stepSim(nodesRef.current, edgesRef.current, dt);
      const now = performance.now();
      edgesRef.current = edgesRef.current.filter(e => now - e.birth < 1500);
      nodesRef.current = nodesRef.current.filter(n =>
        n.isOrchestrator || n.status === "Active" || !n.deathAt || (now - n.deathAt < 1200));
      setTick(x => (x + 1) & 0xffff);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Detail panel refresh
  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    const refresh = () => {
      const related = messagesRef.current
        .filter(m => m.from === selectedId || m.to === selectedId)
        .slice(-10).reverse();
      setDetail({ id: selectedId, messages: related });
    };
    refresh();
    const id = window.setInterval(refresh, 500);
    return () => window.clearInterval(id);
  }, [selectedId]);

  const population = useMemo(() => {
    const specs = nodesRef.current.filter(n => !n.isOrchestrator && n.status === "Active");
    const healthy = specs.filter(n => n.fitness > 0.7).length;
    const struggling = specs.filter(n => n.fitness >= 0.3 && n.fitness <= 0.7).length;
    const dying = specs.filter(n => n.fitness < 0.3).length;
    const total = Math.max(1, healthy + struggling + dying);
    return { healthy, struggling, dying, total,
      pctHealthy: (healthy / total) * 100,
      pctStruggling: (struggling / total) * 100,
      pctDying: (dying / total) * 100 };
  }, [nodesRef.current.length, evoStatus, status]);

  const specialistCount = status?.specialists.length ?? evoStatus?.population_size ?? 0;
  const idle = nodesRef.current.filter(n => !n.isOrchestrator).length === 0;
  const selectedNode = useMemo(
    () => nodesRef.current.find(n => n.id === selectedId) ?? null,
    [selectedId, status, evoStatus]
  );
  const hoveredNode = hover ? nodesRef.current.find(n => n.id === hover.id) ?? null : null;
  const now = performance.now();

  const currentGen = evoStatus?.current_generation ?? status?.generation ?? 0;
  const bestFitnessNow = evoStatus?.last_report?.best_fitness ?? null;
  const deltaFitness = (bestFitnessNow !== null && prevBestFitness !== null)
    ? bestFitnessNow - prevBestFitness : null;

  const wrapStyle: React.CSSProperties = { background: "#0a0e27", color: "#d0e4ff", fontFamily: "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace", minHeight: HEIGHT + 120, padding: 16, borderRadius: 12, position: "relative" };
  const svgStyle: React.CSSProperties = { background: "radial-gradient(ellipse at center, #0d1436 0%, #050714 100%)", border: "1px solid #1a2347", borderRadius: 8, boxShadow: "inset 0 0 80px rgba(0,229,255,0.04)", flex: 1, maxWidth: WIDTH };
  const asideStyle: React.CSSProperties = { width: 280, background: "#0d1436", border: "1px solid #1a2347", borderRadius: 8, padding: 14, fontSize: 12, minHeight: HEIGHT, boxShadow: "inset 0 0 20px rgba(0,229,255,0.04)" };

  return (
    <div style={wrapStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10, gap: 16 }}>
        <div>
          <h2 style={{ margin: 0, fontWeight: 600, letterSpacing: 2, color: "#00e5ff", textShadow: "0 0 12px rgba(0,229,255,0.6)" }}>NEO MIND</h2>
          <div style={{ fontSize: 11, color: "#6a7aa8", marginTop: 2 }}>
            live organism topology — {evoStatus?.initialized ? "evolution daemon" : "organism api"} · 1 Hz + QLMS SSE
          </div>
        </div>
        <div style={{ background: "rgba(13,20,54,0.85)", border: "1px solid #1a2347", borderRadius: 6,
          padding: "6px 12px", fontSize: 11, lineHeight: 1.6, minWidth: 160 }}>
          <div style={{ color: "#6a7aa8", fontSize: 9, letterSpacing: 2 }}>GENERATION</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <div style={{ fontSize: 22, color: "#ff2bd6", fontWeight: 600 }}>{currentGen}</div>
            {bestFitnessNow !== null && (
              <div style={{ color: "#64ffda", fontSize: 11 }}>best {bestFitnessNow.toFixed(3)}</div>
            )}
          </div>
          {deltaFitness !== null && (
            <div style={{ color: deltaFitness >= 0 ? "#7dffb0" : "#ff6b6b", fontSize: 10, display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block",
                animation: deltaFitness > 0 ? "neoArrowUp 1.4s ease-in-out infinite" : undefined }}>
                {deltaFitness >= 0 ? "▲" : "▼"}
              </span>
              {deltaFitness >= 0 ? "+" : ""}{deltaFitness.toFixed(4)} vs last gen
            </div>
          )}
        </div>
        <div style={{ fontSize: 12, textAlign: "right", lineHeight: 1.5 }}>
          <div>
            <span style={{ color: "#6a7aa8" }}>spec </span><span style={{ color: "#64ffda" }}>{specialistCount}</span>{"   "}
            <span style={{ color: "#6a7aa8" }}>int </span><span style={{ color: "#ffd54a" }}>{status?.total_interactions ?? 0}</span>{"   "}
            <span style={{ color: "#6a7aa8" }}>mem </span><span style={{ color: "#b388ff" }}>{status?.memory_items ?? 0}</span>
          </div>
          {evoStatus?.initialized && (
            <div style={{ fontSize: 10, color: "#6a7aa8", marginTop: 2 }}>
              born {evoStatus.total_born ?? 0} · retired {evoStatus.total_retired ?? 0} · killed {evoStatus.total_killed ?? 0}
            </div>
          )}
          {statusError && <div style={{ color: "#ff6b6b", fontSize: 10, marginTop: 4 }}>api error: {statusError}</div>}
        </div>
      </header>

      {/* Population health bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, fontSize: 10, color: "#6a7aa8" }}>
        <span style={{ letterSpacing: 2, minWidth: 90 }}>POPULATION</span>
        <div style={{ flex: 1, height: 10, background: "#0d1436", borderRadius: 5,
          border: "1px solid #1a2347", overflow: "hidden", display: "flex" }}>
          <div style={{ width: `${population.pctHealthy}%`, background: "linear-gradient(90deg,#7dffb0,#64ffda)",
            boxShadow: "0 0 8px rgba(125,255,176,0.6)", transition: "width 400ms" }} />
          <div style={{ width: `${population.pctStruggling}%`, background: "linear-gradient(90deg,#ffd54a,#ffb74a)",
            boxShadow: "0 0 6px rgba(255,213,74,0.5)", transition: "width 400ms" }} />
          <div style={{ width: `${population.pctDying}%`, background: "linear-gradient(90deg,#ff3b5c,#d41648)",
            boxShadow: "0 0 6px rgba(255,59,92,0.5)", transition: "width 400ms" }} />
        </div>
        <span style={{ color: "#7dffb0" }}>{population.healthy} healthy</span>
        <span style={{ color: "#ffd54a" }}>{population.struggling} struggling</span>
        <span style={{ color: "#ff3b5c" }}>{population.dying} dying</span>
      </div>

      <div style={{ display: "flex", gap: 16, position: "relative" }}>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" style={svgStyle} onClick={() => setSelectedId(null)}>
          <defs>
            <filter id="nmBlur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" /></filter>
            <filter id="nmDeadBlur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" /></filter>
          </defs>
          <g stroke="#12204a" strokeWidth="0.5" opacity="0.5">
            {Array.from({ length: 12 }, (_, i) => <line key={`v${i}`} x1={(WIDTH / 12) * i} y1={0} x2={(WIDTH / 12) * i} y2={HEIGHT} />)}
            {Array.from({ length: 8 }, (_, i) => <line key={`h${i}`} x1={0} y1={(HEIGHT / 8) * i} x2={WIDTH} y2={(HEIGHT / 8) * i} />)}
          </g>
          <g>
            {edgesRef.current.map((e, i) => {
              const a = nodesRef.current.find(n => n.id === e.from);
              const b = nodesRef.current.find(n => n.id === e.to);
              if (!a || !b) return null;
              const alpha = Math.max(0, 1 - (now - e.birth) / 1500);
              return <line key={`e${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke="#00e5ff" strokeWidth={1.5} opacity={alpha * 0.8} strokeLinecap="round" />;
            })}
          </g>
          <g>
            {nodesRef.current.map(n => {
              const role = n.isOrchestrator ? "orchestrator" : n.role;
              const rCol = roleColor(role);
              const fitCol = n.isOrchestrator ? rCol : fitnessColor(n.fitness);
              const glow = n.isOrchestrator ? rCol : statusGlow(n.status);
              const since = now - n.lastActivity;
              const busPulse = since < 1200 ? Math.max(0, 1 - since / 1200) : 0;
              const ipsPulse = Math.min(1, n.invocationsPerSec / 3);
              const pulse = Math.max(busPulse, ipsPulse);
              const baseR = n.isOrchestrator ? 28 : 12 + n.fitness * 14;
              const r = baseR + pulse * 8;
              const ringThickness = n.isOrchestrator ? 2.5
                : Math.min(6, 1.5 + Math.sqrt(Math.max(0, n.age)) * 0.9);
              const dyingAge = n.deathAt ? (now - n.deathAt) / 1000 : 0;
              const alive = n.status === "Active" || !n.deathAt;
              const fadeOpacity = alive ? 1 : Math.max(0, 1 - dyingAge);
              const isSelected = selectedId === n.id;

              return (
                <g key={n.id} transform={`translate(${n.x},${n.y})`}
                  style={{ cursor: "pointer", opacity: fadeOpacity }}
                  onClick={(ev) => { ev.stopPropagation(); setSelectedId(n.id); }}
                  onMouseEnter={() => setHover({ id: n.id })}
                  onMouseLeave={() => setHover(h => h?.id === n.id ? null : h)}>
                  {!alive && (
                    <circle r={r + 18} fill="#ff3b5c"
                      opacity={0.4 * (1 - dyingAge)} filter="url(#nmDeadBlur)" />
                  )}
                  <circle r={r + 14} fill={glow} opacity={0.10 + pulse * 0.28} filter="url(#nmBlur)" />
                  {pulse > 0 && alive && (
                    <circle r={r + 6 + pulse * 10} fill="none"
                      stroke={glow} strokeWidth={1} opacity={pulse * 0.7} />
                  )}
                  <circle r={r + 3} fill="none" stroke={fitCol} strokeWidth={ringThickness}
                    opacity={n.status === "Retired" ? 0.45 : 0.9} />
                  <circle r={r} fill="#0a0e27" stroke={rCol} strokeWidth={isSelected ? 3 : 1.5} />
                  <circle r={r - 4} fill={rCol} opacity={n.status === "Retired" ? 0.1 : 0.2} />
                  {n.status === "Retired" && (
                    <circle r={r - 1} fill="none" stroke="#8892b0"
                      strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
                  )}
                  <text y={r + 18} textAnchor="middle"
                    fontSize={n.isOrchestrator ? 13 : 11}
                    fill={rCol} style={{ letterSpacing: 1 }}>{n.label}</text>
                  {!n.isOrchestrator && (
                    <text y={r + 32} textAnchor="middle" fontSize={9} fill="#6a7aa8">
                      f {n.fitness.toFixed(2)} · age {n.age}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
          {idle && (
            <g>
              <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#0a0e27" opacity={0.7} />
              <text x={CX} y={CY - 12} textAnchor="middle" fontSize={22} fill="#00e5ff" style={{ letterSpacing: 6 }}>ORGANISM IDLE</text>
              <text x={CX} y={CY + 16} textAnchor="middle" fontSize={12} fill="#6a7aa8">
                {statusError ? "cannot reach /api/evolution or /api/organism" : "no specialists registered — send a message to wake the swarm"}
              </text>
            </g>
          )}
        </svg>

        {hoveredNode && !hoveredNode.isOrchestrator && (
          <HoverTooltip node={hoveredNode}
            messages={messagesRef.current
              .filter(m => m.from === hoveredNode.id || m.to === hoveredNode.id)
              .slice(-3).reverse()} />
        )}

        <aside style={asideStyle}>
          {selectedNode ? (
            <div>
              <div style={{ color: roleColor(selectedNode.role), fontSize: 14, letterSpacing: 2, marginBottom: 8, textShadow: `0 0 8px ${roleColor(selectedNode.role)}55` }}>
                {selectedNode.label.toUpperCase()}
              </div>
              <div style={{ color: "#6a7aa8", marginBottom: 10 }}>
                <div>id: {selectedNode.id}</div>
                <div>role: {selectedNode.role}</div>
                <div>status: <span style={{ color: statusGlow(selectedNode.status) }}>{selectedNode.status}</span></div>
                <div>fitness: <span style={{ color: fitnessColor(selectedNode.fitness) }}>{selectedNode.fitness.toFixed(3)}</span></div>
                {!selectedNode.isOrchestrator && (
                  <>
                    <div>rank: top {((1 - selectedNode.rankPct) * 100).toFixed(0)}%</div>
                    <div>born: gen {selectedNode.generationBorn}</div>
                    <div>age: {selectedNode.age} gen</div>
                    {selectedNode.parentId && <div>parent: {selectedNode.parentId}</div>}
                  </>
                )}
                <div>invocations: {selectedNode.invocations}</div>
                {selectedNode.invocationsPerSec > 0 && (
                  <div>rate: {selectedNode.invocationsPerSec.toFixed(2)}/s</div>
                )}
              </div>
              <div style={{ color: "#00e5ff", fontSize: 10, letterSpacing: 2, marginBottom: 6 }}>LAST 10 MESSAGES</div>
              {detail && detail.messages.length > 0 ? (
                <ul style={{ listStyle: "none", padding: 0, margin: 0, maxHeight: 420, overflowY: "auto" }}>
                  {detail.messages.map(m => (
                    <li key={m.id} style={{ padding: "4px 0", borderBottom: "1px solid #1a2347", fontSize: 10 }}>
                      <div style={{ color: "#ffd54a" }}>#{m.id} · {m.intent}</div>
                      <div style={{ color: "#6a7aa8" }}>{m.from} → {m.to}</div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div style={{ color: "#6a7aa8", fontStyle: "italic" }}>no recent messages on the bus</div>
              )}
            </div>
          ) : (
            <div style={{ color: "#6a7aa8" }}>
              <div style={{ color: "#00e5ff", letterSpacing: 2, marginBottom: 8 }}>INSPECT</div>
              hover any node for a quick summary, or click to pin it and inspect its lineage + bus traffic.
            </div>
          )}
        </aside>
      </div>

      <style>{`@keyframes neoArrowUp {0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}`}</style>
    </div>
  );
};

// Hover tooltip — DOM overlay for crisp text wrapping
const HoverTooltip: React.FC<{ node: SimNode; messages: BusMessage[] }> = ({ node, messages }) => {
  const leftPct = (node.x / WIDTH) * 100;
  const topPx = node.y - 40;
  const style: React.CSSProperties = {
    position: "absolute", left: `calc(${leftPct}% + 18px)`, top: topPx,
    pointerEvents: "none", background: "rgba(5,7,20,0.95)",
    border: `1px solid ${fitnessColor(node.fitness)}`, borderRadius: 6,
    padding: "8px 10px", fontSize: 11, color: "#d0e4ff",
    minWidth: 200, maxWidth: 260,
    boxShadow: `0 0 16px ${fitnessColor(node.fitness)}55`, zIndex: 20, lineHeight: 1.5,
  };
  return (
    <div style={style}>
      <div style={{ color: roleColor(node.role), letterSpacing: 1.5, fontSize: 12, marginBottom: 4 }}>
        {node.label}
      </div>
      <div style={{ color: "#6a7aa8", fontSize: 10 }}>role · {node.role}</div>
      <div>fitness <span style={{ color: fitnessColor(node.fitness) }}>{node.fitness.toFixed(3)}</span>
        {" · top "}{((1 - node.rankPct) * 100).toFixed(0)}%</div>
      <div>born gen {node.generationBorn} · age {node.age}</div>
      <div>status <span style={{ color: statusGlow(node.status) }}>{node.status}</span>
        {node.parentId && <span style={{ color: "#6a7aa8" }}> · parent {node.parentId}</span>}</div>
      {messages.length > 0 && (
        <>
          <div style={{ color: "#00e5ff", fontSize: 9, letterSpacing: 2, margin: "6px 0 2px" }}>RECENT MESSAGES</div>
          {messages.map(m => (
            <div key={m.id} style={{ fontSize: 10, color: "#6a7aa8" }}>
              <span style={{ color: "#ffd54a" }}>#{m.id}</span> {m.intent} — {m.from} → {m.to}
            </div>
          ))}
        </>
      )}
    </div>
  );
};

export default NeoMind;
