#!/usr/bin/env node
/**
 * Publish the DeepSeek Harness session inventory into the qo graph store.
 *
 * Presence covers LIVE sessions and is deliberately ephemeral (60s TTL, wiped
 * by a qo restart). This importer covers the other half: the sessions that
 * already exist on disk become one durable graph in qo, listed by
 * `GET /api/graphs` and visible in the cockpit.
 *
 * The source is the harness session projection cache, which already holds the
 * title, workspace and turn counts — no session log has to be decompressed.
 *
 * Re-running stores a new snapshot rather than mutating the old one: the graph
 * store is append-only, so the history of "which sessions existed when" stays
 * readable.
 *
 * Config (env): QO_URL, QO_TOKEN, QO_API_KEYS, DSH_HOME.
 */

import { readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(HERE, '..', '..')
const QO_URL = (process.env.QO_URL ?? 'http://127.0.0.1:4646').replace(/\/+$/, '')
const DSH_HOME = process.env.DSH_HOME ?? join(homedir(), '.dsh')
const KEYS_PATH = process.env.QO_API_KEYS ?? join(REPO_ROOT, '.qlang', 'api_keys.json')

function token() {
  if (process.env.QO_TOKEN !== undefined) return process.env.QO_TOKEN
  try {
    return JSON.parse(readFileSync(KEYS_PATH, 'utf8')).keys?.[0]?.secret ?? ''
  } catch {
    // No key store is legal: an unauthenticated qo accepts the request.
    return ''
  }
}

/** Parse `@{a=1; b=2}`-free JSON projection rows into the fields we publish. */
function readSessions() {
  const path = join(DSH_HOME, 'storages', 'session_projcache.json')
  const cache = JSON.parse(readFileSync(path, 'utf8'))
  const table = cache.tables?.sessions ?? {}
  return Object.entries(table).map(([id, record]) => {
    const rows = record.rows ?? {}
    const stats = rows.sessionStats?.val ?? {}
    return {
      id,
      title: typeof rows.title?.val === 'string' ? rows.title.val : '(untitled)',
      cwd: record.identity?.cwd ?? '',
      createdAt: record.identity?.createdAt ?? 0,
      turns: typeof stats.turns === 'number' ? stats.turns : 0,
      steps: typeof stats.steps === 'number' ? stats.steps : 0,
      llmMs: typeof stats.llmMs === 'number' ? stats.llmMs : 0,
    }
  })
}

const sessions = readSessions()
if (sessions.length === 0) {
  console.error('no sessions found in the projection cache; nothing to publish')
  process.exit(1)
}

const now = Math.floor(Date.now() / 1000)
const graph = {
  id: 0,
  timestamp: now,
  graph_type: 'AgentTask',
  title: `DeepSeek Harness sessions (${String(sessions.length)})`,
  nodes: sessions.map(session => ({
    id: session.id,
    op: 'dsh_session',
    node_type: 'Memory',
    label: `${session.title} — ${session.cwd} (${String(session.turns)} turns, ${String(session.steps)} steps)`,
    agent: session.cwd === '' ? null : session.cwd,
    status: 'Completed',
    duration_ms: session.llmMs,
    input_type: 'session',
    output_type: new Date(session.createdAt).toISOString(),
  })),
  edges: [],
  metadata: {
    total_duration_ms: sessions.reduce((sum, session) => sum + session.llmMs, 0),
    llm_tier: 'deepseek-harness',
    tokens_estimated: null,
    cost_usd: null,
  },
}

const value = token()
const response = await fetch(`${QO_URL}/api/graphs`, {
  method: 'POST',
  headers: value === '' ? { 'content-type': 'application/json' } : { 'content-type': 'application/json', authorization: `Bearer ${value}` },
  body: JSON.stringify(graph),
})
if (!response.ok) {
  console.error(`qo answered ${String(response.status)}: ${await response.text().catch(() => '')}`)
  process.exit(1)
}
const stored = await response.json()
console.log(`published ${String(sessions.length)} session(s) to qo as graph #${String(stored.id)}: ${stored.message}`)
for (const session of sessions) console.log(`  - ${session.title} [${session.cwd}] ${String(session.turns)} turns`)
