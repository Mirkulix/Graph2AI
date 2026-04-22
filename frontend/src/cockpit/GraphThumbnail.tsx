// Live-rendered SVG thumbnail of a QLMS graph. Tries to extract nodes/edges
// from the graph payload; falls back to a node-count badge if the shape is
// unfamiliar.

interface Props {
  graph: unknown;
  width?: number;
  height?: number;
  onClick?: () => void;
}

interface Extracted {
  nodes: Array<{ id: string; label?: string }>;
  edges: Array<{ from: string; to: string }>;
}

function extract(graph: unknown): Extracted | null {
  if (!graph || typeof graph !== 'object') return null;
  const g = graph as Record<string, unknown>;

  // Common shapes: { nodes: [...], edges: [...] } or { type: 'script', source: '...' }
  const nodesRaw = g.nodes;
  const edgesRaw = g.edges;
  if (!Array.isArray(nodesRaw)) return null;

  const nodes = (nodesRaw as Array<Record<string, unknown>>).map((n, i) => ({
    id: String(n.id ?? i),
    label: typeof n.op === 'string' ? n.op : (typeof n.label === 'string' ? n.label : undefined),
  }));

  let edges: Extracted['edges'] = [];
  if (Array.isArray(edgesRaw)) {
    edges = (edgesRaw as Array<Record<string, unknown>>)
      .map(e => ({ from: String(e.from ?? e.src ?? ''), to: String(e.to ?? e.dst ?? '') }))
      .filter(e => e.from && e.to);
  }

  return { nodes, edges };
}

export default function GraphThumbnail({ graph, width = 220, height = 64, onClick }: Props) {
  const data = extract(graph);

  if (!data || data.nodes.length === 0) {
    // Fallback: descriptive chip
    const desc = describe(graph);
    return (
      <button
        onClick={onClick}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 8px',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--rule-default)',
          borderRadius: 4,
          fontSize: 10,
          fontFamily: 'var(--font-mono)',
          color: 'var(--ink-muted)',
          cursor: onClick ? 'pointer' : 'default',
        }}
      >
        <svg width="9" height="9" viewBox="0 0 9 9" style={{ color: 'var(--link-bright)', flexShrink: 0 }}>
          <rect x="2" y="2" width="5" height="5" transform="rotate(45 4.5 4.5)" fill="currentColor" />
        </svg>
        {desc}
      </button>
    );
  }

  // Layout: simple horizontal chain or grid
  const n = data.nodes.length;
  const cols = Math.min(Math.ceil(Math.sqrt(n)), 6);
  const rows = Math.ceil(n / cols);
  const pad = 8;
  const cellW = (width - pad * 2) / cols;
  const cellH = (height - pad * 2) / Math.max(rows, 1);
  const r = Math.min(cellW, cellH) * 0.18;

  const positions = new Map<string, { x: number; y: number }>();
  data.nodes.forEach((node, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    positions.set(node.id, {
      x: pad + cellW * col + cellW / 2,
      y: pad + cellH * row + cellH / 2,
    });
  });

  return (
    <button
      onClick={onClick}
      style={{
        padding: 0,
        background: 'var(--bg-deep)',
        border: '1px solid var(--rule-default)',
        borderRadius: 4,
        cursor: onClick ? 'pointer' : 'default',
        display: 'block',
        overflow: 'hidden',
      }}
    >
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
        {/* edges */}
        {data.edges.map((e, i) => {
          const a = positions.get(e.from);
          const b = positions.get(e.to);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke="var(--rule-strong)"
              strokeWidth="0.8"
            />
          );
        })}
        {/* nodes */}
        {data.nodes.map(node => {
          const p = positions.get(node.id);
          if (!p) return null;
          return (
            <g key={node.id}>
              <circle cx={p.x} cy={p.y} r={r} fill="var(--process)" opacity="0.85" />
              <circle cx={p.x} cy={p.y} r={r * 0.45} fill="var(--process-bright)" />
            </g>
          );
        })}
      </svg>
      <div style={{
        padding: '3px 6px',
        borderTop: '1px solid var(--rule-faint)',
        fontSize: 9,
        fontFamily: 'var(--font-mono)',
        color: 'var(--ink-faint)',
        letterSpacing: 0.04,
        textAlign: 'left',
      }}>
        {data.nodes.length} nodes · {data.edges.length} edges
      </div>
    </button>
  );
}

function describe(graph: unknown): string {
  if (!graph) return 'empty';
  if (typeof graph === 'string') return `text · ${graph.length} ch`;
  if (typeof graph === 'object') {
    const g = graph as Record<string, unknown>;
    if (typeof g.type === 'string' && typeof g.source === 'string') {
      const lang = typeof g.language === 'string' ? g.language : 'src';
      return `${lang} · ${(g.source as string).split('\n').length} L`;
    }
    return `obj · ${Object.keys(g).length} k`;
  }
  return typeof graph;
}
