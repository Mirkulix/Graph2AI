#!/usr/bin/env node
/**
 * qo session bridge — an MCP stdio server that lets DeepSeek Harness sessions
 * register themselves in the qo server and talk to each other over the qo
 * message bus.
 *
 * It owns no state that matters: presence lives in qo (in-memory, 60s TTL,
 * kept alive by this process's heartbeat), messages live in qo's bus ring
 * buffer. What this process keeps is a per-handle read cursor so an inbox
 * read returns only what arrived since the last one.
 *
 * Transport is newline-delimited JSON-RPC 2.0 on stdio — the MCP stdio
 * framing. Zero dependencies on purpose: the file must run from any
 * directory the harness spawns it in.
 *
 * Config (all optional, env):
 *   QO_URL       qo base URL              (default http://127.0.0.1:4646)
 *   QO_TOKEN     bearer token             (default: first secret in QO_API_KEYS)
 *   QO_API_KEYS  path to api_keys.json    (default <repo>/.qlang/api_keys.json)
 */

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(HERE, '..', '..')

const QO_URL = (process.env.QO_URL ?? 'http://127.0.0.1:4646').replace(/\/+$/, '')
const KEYS_PATH = process.env.QO_API_KEYS ?? join(REPO_ROOT, '.qlang', 'api_keys.json')

/** Presence TTL in qo is 60s; refresh well inside it. */
const HEARTBEAT_MS = 25_000
/** qo's bus ring buffer caps at 200; ask for the whole window when reading. */
const RECENT_WINDOW = 200

/** Bearer token, resolved once. Absent is legal: qo without auth accepts anything. */
let cachedToken = process.env.QO_TOKEN ?? null
function token() {
  if (cachedToken !== null) return cachedToken
  try {
    const parsed = JSON.parse(readFileSync(KEYS_PATH, 'utf8'))
    cachedToken = parsed.keys?.[0]?.secret ?? ''
  } catch {
    // No key store: an unauthenticated qo (no token, no seats) accepts the
    // request anyway, and an authenticated one answers 401, which the caller
    // sees as a tool error naming the endpoint.
    cachedToken = ''
  }
  return cachedToken
}

function headers() {
  const value = token()
  return value === '' ? { 'content-type': 'application/json' } : { 'content-type': 'application/json', authorization: `Bearer ${value}` }
}

/**
 * One qo HTTP call.
 *
 * Bus message ids are u64 snowflakes past `Number.MAX_SAFE_INTEGER`, so
 * `stringifyIds` re-quotes them before parsing: read as plain JSON numbers
 * they round to multiples of 16, and two messages produced in the same
 * millisecond then compare equal — which silently drops one from an inbox.
 * @param {string} path - path beginning with a slash.
 * @param {{method?: string, body?: unknown, stringifyIds?: boolean}} [options] - request options.
 * @returns {Promise<unknown>} the parsed JSON body.
 * @throws when qo is unreachable or answers a non-2xx status.
 */
async function qo(path, options = {}) {
  const method = options.method ?? 'GET'
  let response
  try {
    response = await fetch(`${QO_URL}${path}`, {
      method,
      headers: headers(),
      ...options.body === undefined ? {} : { body: JSON.stringify(options.body) },
    })
  } catch (cause) {
    throw new Error(`qo is unreachable at ${QO_URL} (${method} ${path}). Start it with target\\debug\\qo.exe --offline from the OrbitQLang repo. Cause: ${String(cause)}`)
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`qo answered ${String(response.status)} for ${method} ${path}${text === '' ? '' : `: ${text.slice(0, 400)}`}`)
  }
  const text = await response.text()
  if (text === '') return null
  return JSON.parse(options.stringifyIds === true ? text.replace(/"id":(\d+)/g, '"id":"$1"') : text)
}

/** Bus messages with their ids kept exact as decimal strings. */
async function recentMessages() {
  const recent = await qo(`/api/messages/recent?n=${String(RECENT_WINDOW)}`, { stringifyIds: true })
  return Array.isArray(recent) ? recent : []
}

/** Handles this process registered, with their heartbeat timers and read cursors. */
const registered = new Map()

/**
 * Register one handle in qo presence and keep it alive for this process's life.
 * @param {string} handle - the session's identity on the bus.
 * @param {string} note - free-text description shown in the directory.
 * @param {string} workspace - workspace path reported to qo.
 * @returns {Promise<void>} resolves once qo holds the registration.
 */
async function registerHandle(handle, note, workspace) {
  await qo('/api/presence/register', {
    method: 'POST',
    body: {
      identity: handle,
      ide_name: 'DeepSeek Harness',
      capabilities: ['dsh-session', 'chat', 'tools', ...note === '' ? [] : [`note:${note}`]],
      ...workspace === '' ? {} : { workspace_path: workspace },
    },
  })
  const existing = registered.get(handle)
  if (existing?.timer !== undefined) clearInterval(existing.timer)
  const timer = setInterval(() => {
    void qo(`/api/presence/heartbeat/${encodeURIComponent(handle)}`, { method: 'POST', body: {} })
      .catch(async () => {
        // Swept after a qo restart or a long stall: re-register rather than
        // letting the handle silently disappear from the directory.
        await qo('/api/presence/register', {
          method: 'POST',
          body: { identity: handle, ide_name: 'DeepSeek Harness', capabilities: ['dsh-session', 'chat', 'tools'] },
        }).catch(() => {})
      })
  }, HEARTBEAT_MS)
  timer.unref()
  registered.set(handle, { timer, note, cursor: existing?.cursor ?? await highestMessageId() })
}

/** The newest bus message id as a BigInt, so a fresh registration does not replay history. */
async function highestMessageId() {
  const list = await recentMessages().catch(() => [])
  return list.reduce((max, message) => {
    const id = BigInt(message.id)
    return id > max ? id : max
  }, 0n)
}

/** Format one presence entry for the model. */
function describePresence(entry) {
  const note = (entry.capabilities ?? []).find(capability => capability.startsWith('note:'))
  return `- ${entry.identity}${entry.ide_name === undefined ? '' : ` [${entry.ide_name}]`}`
    + `${note === undefined ? '' : ` — ${note.slice(5)}`}`
    + ` (last seen ${entry.last_seen_at})`
}

// ---- Tools ----

const TOOLS = [
  {
    name: 'session_register',
    description:
      'Register THIS session in the qo server so other sessions can see and message it. '
      + 'Call once at the start of a session, before session_send or session_inbox. '
      + 'Pick a short stable handle (e.g. "web-ui", "refactor-a"). The registration is kept alive automatically.',
    inputSchema: {
      type: 'object',
      properties: {
        handle: { type: 'string', description: 'Short stable identity for this session, e.g. "refactor-a".' },
        note: { type: 'string', description: 'One line on what this session is working on; shown to other sessions.' },
        workspace: { type: 'string', description: 'Absolute path of this session\'s workspace, when it has one.' },
      },
      required: ['handle'],
    },
  },
  {
    name: 'session_directory',
    description: 'List the sessions and agents currently registered in the qo server, with what each said it is working on.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'session_send',
    description:
      'Send a message from this session to another registered session over the qo message bus. '
      + 'Use the exact handle from session_directory as "to", or "*" to reach every other registered session. '
      + 'Delivery is asynchronous: the recipient reads it with session_inbox, so do not expect an immediate reply.',
    inputSchema: {
      type: 'object',
      properties: {
        from: { type: 'string', description: 'This session\'s handle, as registered with session_register.' },
        to: { type: 'string', description: 'Recipient handle from session_directory, or "*" for every other session.' },
        text: { type: 'string', description: 'The message body.' },
      },
      required: ['from', 'to', 'text'],
    },
  },
  {
    name: 'session_inbox',
    description:
      'Read messages other sessions sent to this session. Returns only what arrived since the last read; '
      + 'pass all=true to re-read the whole retained window (qo keeps the newest 200 bus messages).',
    inputSchema: {
      type: 'object',
      properties: {
        handle: { type: 'string', description: 'This session\'s handle, as registered with session_register.' },
        all: { type: 'boolean', description: 'Return the whole retained window instead of only new messages.' },
      },
      required: ['handle'],
    },
  },
]

/**
 * Execute one tool call.
 * @param {string} name - the tool name.
 * @param {Record<string, unknown>} args - the model-supplied arguments.
 * @returns {Promise<string>} the text the model sees.
 */
async function callTool(name, args) {
  switch (name) {
    case 'session_register': {
      const handle = String(args.handle ?? '').trim()
      if (handle === '') throw new Error('handle is required and must be non-empty')
      const note = String(args.note ?? '').trim()
      await registerHandle(handle, note, String(args.workspace ?? '').trim())
      const others = (await qo('/api/presence')).filter(entry => entry.identity !== handle)
      return `Registered "${handle}" in qo at ${QO_URL}.\n\n`
        + (others.length === 0
          ? 'No other sessions are registered right now. Others appear here once they call session_register.'
          : `Other sessions online (${String(others.length)}):\n${others.map(describePresence).join('\n')}`)
    }
    case 'session_directory': {
      const entries = await qo('/api/presence')
      return entries.length === 0
        ? 'No sessions are registered in qo right now.'
        : `Registered sessions (${String(entries.length)}):\n${entries.map(describePresence).join('\n')}`
    }
    case 'session_send': {
      const from = String(args.from ?? '').trim()
      const to = String(args.to ?? '').trim()
      const text = String(args.text ?? '')
      if (from === '' || to === '' || text.trim() === '') throw new Error('from, to and text are all required')
      const present = await qo('/api/presence')
      const targets = to === '*'
        ? present.map(entry => entry.identity).filter(identity => identity !== from)
        : [to]
      if (targets.length === 0) throw new Error('no other session is registered, so there is nobody to send to')
      const result = await qo('/api/broadcast', { method: 'POST', body: { from, targets, prompt: text } })
      const failures = result.failures ?? []
      const delivered = targets.filter(target => !failures.some(failure => failure.target === target))
      const lines = [`Sent to ${String(delivered.length)} session(s): ${delivered.join(', ') || '(none)'}`]
      if (failures.length > 0) {
        lines.push(`Not delivered: ${failures.map(failure => `${failure.target} (${failure.error})`).join(', ')}`)
        lines.push('A target must have called session_register and still be online; check session_directory.')
      }
      lines.push('The recipient sees it on its next session_inbox call.')
      return lines.join('\n')
    }
    case 'session_inbox': {
      const handle = String(args.handle ?? '').trim()
      if (handle === '') throw new Error('handle is required and must be non-empty')
      const all = args.all === true
      const recent = await recentMessages()
      const entry = registered.get(handle)
      const cursor = all ? 0n : entry?.cursor ?? 0n
      const mine = recent
        .filter(message => message.to === handle && message.from !== handle)
        .filter(message => BigInt(message.id) > cursor)
        .sort((left, right) => (BigInt(left.id) < BigInt(right.id) ? -1 : 1))
      const highest = recent.reduce((max, message) => {
        const id = BigInt(message.id)
        return id > max ? id : max
      }, cursor)
      if (entry !== undefined) entry.cursor = highest
      else registered.set(handle, { timer: undefined, note: '', cursor: highest })
      if (mine.length === 0) {
        return entry === undefined
          ? `No messages for "${handle}". Note: this handle is not registered here — call session_register so other sessions can reach it.`
          : `No new messages for "${handle}".`
      }
      return `${String(mine.length)} message(s) for "${handle}":\n\n`
        + mine.map(message => `From ${message.from} (id ${String(message.id)}):\n${message.content}`).join('\n\n')
    }
    default:
      throw new Error(`unknown tool: ${name}`)
  }
}

// ---- MCP stdio JSON-RPC ----

const SERVER_INFO = {
  protocolVersion: '2024-11-05',
  serverInfo: { name: 'qo-session-bridge', version: '0.1.0' },
  capabilities: { tools: {} },
}

function write(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`)
}

function ok(id, result) {
  write({ jsonrpc: '2.0', id, result })
}

function fail(id, code, message) {
  write({ jsonrpc: '2.0', id, error: { code, message } })
}

const rl = createInterface({ input: process.stdin })
rl.on('line', (line) => {
  const trimmed = line.trim()
  if (trimmed === '') return
  let request
  try {
    request = JSON.parse(trimmed)
  } catch {
    write({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } })
    return
  }
  // Notifications carry no id and expect no response.
  if (request.id === undefined || request.id === null) return
  void (async () => {
    try {
      switch (request.method) {
        case 'initialize':
          ok(request.id, SERVER_INFO)
          return
        case 'tools/list':
          ok(request.id, { tools: TOOLS })
          return
        case 'tools/call': {
          const name = request.params?.name
          if (typeof name !== 'string') {
            fail(request.id, -32602, 'Missing tool name')
            return
          }
          try {
            const text = await callTool(name, request.params?.arguments ?? {})
            ok(request.id, { content: [{ type: 'text', text }] })
          } catch (error) {
            // A tool-level failure is a result the model must see and can act
            // on, not a protocol error that would fail the whole call.
            ok(request.id, { content: [{ type: 'text', text: `Error: ${error instanceof Error ? error.message : String(error)}` }], isError: true })
          }
          return
        }
        case 'ping':
          ok(request.id, {})
          return
        default:
          fail(request.id, -32601, `Method not found: ${String(request.method)}`)
      }
    } catch (error) {
      fail(request.id, -32603, error instanceof Error ? error.message : String(error))
    }
  })()
})

rl.on('close', () => {
  for (const entry of registered.values()) if (entry.timer !== undefined) clearInterval(entry.timer)
  process.exit(0)
})
