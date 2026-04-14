#!/usr/bin/env node
/**
 * Parse data/session_2026_04_11.qlang and POST it to QLANG GraphStore.
 * Uses a minimal QLANG parser for the subset of syntax used in session files.
 */

import { readFileSync } from 'fs';
import http from 'http';

const QLANG_FILE = '/home/mirkulix/AI/neoqlang/qlang/data/session_2026_04_11.qlang';
const QO_HOST = 'localhost';
const QO_PORT = 4646;

// Parse .qlang file into nodes + edges
function parseQlang(text) {
  const nodes = [];
  const edges = [];
  let metadata = {};
  let title = 'session_2026_04_11';

  // Match `graph NAME { ... }`
  const graphMatch = text.match(/graph\s+(\w+)\s*\{/);
  if (graphMatch) title = graphMatch[1];

  // Match nodes: `node NAME : OP { ... }` or `node NAME : OP {}`
  const nodeRegex = /node\s+(\w+)\s*:\s*(\w+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*?)\}/g;
  let m;
  while ((m = nodeRegex.exec(text)) !== null) {
    const [_, name, op, body] = m;
    // Extract inner metadata if any
    const metaMatch = body.match(/metadata\s*\{([^}]+)\}/);
    let meta = {};
    if (metaMatch) {
      const lines = metaMatch[1].split('\n');
      for (const line of lines) {
        const kv = line.match(/(\w+):\s*"?([^",\n]+)"?/);
        if (kv) meta[kv[1].trim()] = kv[2].trim();
      }
    }

    nodes.push({
      id: name,
      op: op,
      node_type: op === 'detect' || op === 'extract' ? 'Input' :
                 op === 'subgraph' ? 'Deterministic' :
                 op === 'evolve' || op === 'scan' ? 'Llm' :
                 op === 'cond' ? 'Deterministic' : 'Deterministic',
      label: `${name} (${op})`,
      agent: meta.agent || null,
      status: 'Completed',
      duration_ms: null,
      input_type: null,
      output_type: null,
    });
  }

  // Match edges: `edge A -> B` or `edge A -> B` inside node bodies
  const edgeRegex = /edge\s+(\w+)\s*->\s*(\w+)/g;
  while ((m = edgeRegex.exec(text)) !== null) {
    edges.push({ from: m[1], to: m[2], data_type: 'Tensor' });
  }

  return {
    id: 0,
    timestamp: Math.floor(Date.now() / 1000),
    graph_type: 'AgentTask',
    title,
    nodes,
    edges,
    metadata: {
      total_duration_ms: null,
      llm_tier: 'session-tracking',
      tokens_estimated: null,
      cost_usd: null,
    }
  };
}

// Call health first
function req(method, path, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const opts = {
      host: QO_HOST, port: QO_PORT, path, method,
      headers: { 'Content-Type': 'application/json', ...(data ? { 'Content-Length': Buffer.byteLength(data) } : {}) }
    };
    const r = http.request(opts, res => {
      let chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString();
        try { resolve({ status: res.statusCode, body: JSON.parse(body) }); }
        catch { resolve({ status: res.statusCode, body }); }
      });
    });
    r.on('error', reject);
    if (data) r.write(data);
    r.end();
  });
}

async function main() {
  const text = readFileSync(QLANG_FILE, 'utf8');
  const graph = parseQlang(text);

  console.log(`Parsed: ${graph.nodes.length} nodes, ${graph.edges.length} edges`);
  console.log(`Title: ${graph.title}`);
  console.log(`Nodes: ${graph.nodes.slice(0, 5).map(n => n.id).join(', ')}...`);

  // Check server
  const health = await req('GET', '/api/health');
  if (health.status !== 200) { console.error('Server not healthy'); process.exit(1); }
  console.log(`Server: ${health.body.status}`);

  // List existing graphs to see the current count
  const before = await req('GET', '/api/graphs/stats');
  console.log(`Graphs before: ${before.body.total_graphs || 0}`);

  // POST to the GraphStore
  console.log('\nUploading to GraphStore...');
  const post = await req('POST', '/api/graphs', graph);
  if (post.status === 200 || post.status === 201) {
    console.log(`✓ Graph gespeichert! ID: ${post.body.id}`);
    console.log(`  ${post.body.message}`);
  } else {
    console.log(`✗ Upload failed (${post.status}):`, post.body);
  }

  // Verify
  const after = await req('GET', '/api/graphs/stats');
  console.log(`\nGraphs after: ${after.body.total_graphs}`);

  const list = await req('GET', '/api/graphs?limit=5');
  console.log(`\nLetzte Graphen:`);
  for (const g of (list.body || []).slice(0, 5)) {
    console.log(`  #${g.id}: ${g.title} [${g.graph_type}] — ${g.nodes.length} nodes, ${g.edges.length} edges`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
