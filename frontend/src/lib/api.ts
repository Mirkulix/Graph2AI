// Typed fetch wrapper for the QO backend. Auth-token interception happens in main.tsx.

export interface AgentInfo {
  name: string;
  role?: string;
  online?: boolean;
  mailbox_size?: number;
  capabilities?: string[];
  last_seen?: string;
}

export interface BusStats {
  total_messages?: number;
  msgs_per_minute?: number;
  active_agents?: number;
  uptime_seconds?: number;
}

export interface Conversation {
  id: string;
  participants: string[];
  message_count: number;
  last_at?: string;
}

export interface BusMessage {
  id?: number | string;
  from?: { name: string } | string;
  to?: { name: string } | string;
  intent?: string | { kind?: string };
  graph?: unknown;
  graph_hash?: number[] | string;
  signature?: number[] | null;
  signer_pubkey?: number[] | null;
  in_reply_to?: number | null;
  ts?: string;
  signed?: boolean;
  signature_verified?: boolean;
  content?: string;
  is_reply?: boolean;
}

export interface ProviderTemplate {
  id: string;
  name: string;                              // backend uses `name` not `display_name`
  provider_type?: string;
  base_url?: string;
  description?: string;
  free?: boolean;
  tier?: number;
  models: Array<{ id: string; name?: string; cost_per_1k?: number; recommended?: boolean }>;
}

export interface ProviderConfig {
  id: string;
  name: string;                              // backend uses `name` not `display_name`
  provider_type: string;
  model?: string;                            // backend uses `model` not `default_model`
  base_url?: string | null;
  enabled: boolean;
  tier?: number;
  cost_per_1k_tokens?: number;
  requests?: number;
  tokens?: number;
  cost_usd?: number;
  avg_latency_ms?: number;
  source?: string;                           // 'env' or 'config'
}

export interface PresenceEntry {
  identity: string;
  ide_name?: string;
  host?: string;
  capabilities?: string[];
  llm_provider?: string;
  llm_model?: string;
  registered_at?: string;
  last_seen_at?: string;
  expires_at?: string;
}

export interface GraphSummary {
  id: string;
  name?: string;
  node_count?: number;
  edge_count?: number;
  hash?: string;
  created_at?: string;
}

export interface GraphDetail extends GraphSummary {
  nodes?: Array<{ id: string; op?: string; label?: string }>;
  edges?: Array<{ from: string; to: string }>;
  raw?: unknown;
}

async function jsonRequest<T>(method: string, path: string, body?: unknown): Promise<T> {
  const resp = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${method} ${path} -> HTTP ${resp.status}: ${text || resp.statusText}`);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

const get  = <T>(p: string) => jsonRequest<T>('GET', p);
const post = <T>(p: string, b?: unknown) => jsonRequest<T>('POST', p, b);
const put  = <T>(p: string, b?: unknown) => jsonRequest<T>('PUT', p, b);
const del  = <T>(p: string) => jsonRequest<T>('DELETE', p);

// ─── Domain calls ────────────────────────────────────────────────────

export const api = {
  health: () => get<{ status: string; version?: string; system?: string }>('/api/health'),

  // Messages / bus
  busStats:        () => get<BusStats>('/api/messages/stats'),
  // Backend currently returns Vec<String> (just names). Normalize to AgentInfo[].
  busAgents:       async (): Promise<AgentInfo[]> => {
                     const raw = await get<unknown>('/api/messages/agents');
                     if (!Array.isArray(raw)) return [];
                     return raw.map((entry): AgentInfo => {
                       if (typeof entry === 'string') return { name: entry, online: true };
                       const obj = entry as Record<string, unknown>;
                       return {
                         name: typeof obj.name === 'string' ? obj.name : String(obj),
                         role: typeof obj.role === 'string' ? obj.role : undefined,
                         online: obj.online !== false,
                         mailbox_size: typeof obj.mailbox_size === 'number' ? obj.mailbox_size : undefined,
                         capabilities: Array.isArray(obj.capabilities) ? obj.capabilities as string[] : undefined,
                       };
                     });
                   },
  conversations:   () => get<Conversation[]>('/api/messages/conversations'),

  // QLMS bridge — sign + deliver
  qlmsReply:       (messages: unknown[], seedHex?: string) =>
                     post<{ encoding: string; frame: string; size_bytes: number }>(
                       '/qlms/v1.1/reply',
                       { messages, seed_hex: seedHex },
                     ),
  qlmsDeliver:     (frame: string) =>
                     post<{ version: number; flags: number; signed: boolean; signature_verified: boolean; msg_count: number; messages: BusMessage[] }>(
                       '/qlms/v1.1/deliver',
                       { encoding: 'base64', frame },
                     ),

  // Chat
  chatHistory:     () => get<{ messages: Array<{ role: string; content: string; ts?: string }> }>('/api/chat/history'),
  chatSend:        (prompt: string) =>
                     post<{ reply?: string; graph?: unknown }>('/api/chat', { prompt }),

  // Providers — /api/providers/configured returns the array; /api/providers wraps it
  providers:       () => get<ProviderConfig[]>('/api/providers/configured'),
  providerTemplates: () => get<ProviderTemplate[]>('/api/providers/templates'),
  providerCosts:   () => get<{ total_usd?: number; by_provider?: Record<string, number> }>('/api/providers/costs'),
  providerAdd:     (config: { template_id: string; api_key: string; model: string }) =>
                     post<ProviderConfig>('/api/providers/add', config),
  providerTest:    (id: string) => post<{ ok: boolean; latency_ms?: number; error?: string }>(`/api/providers/test`, { id }),
  providerToggle:  (id: string, enabled: boolean) => put<ProviderConfig>(`/api/providers/${id}/toggle`, { enabled }),
  providerEdit:    (id: string, patch: Partial<ProviderConfig>) => put<ProviderConfig>(`/api/providers/${id}/edit`, patch),
  providerDelete:  (id: string) => del<void>(`/api/providers/${id}`),

  // Graphs
  graphs:          () => get<GraphSummary[]>('/api/graphs'),
  graph:           (id: string) => get<GraphDetail>(`/api/graphs/${id}`),
  graphStats:      () => get<{ total: number; by_op?: Record<string, number> }>('/api/graphs/stats'),

  // Supervisor
  supervisorState: () => get<{ agents?: AgentInfo[]; tasks?: unknown[]; sessions?: unknown[] }>('/api/supervisor/state'),

  // Federation
  federationPeers: () => get<Array<{ id: string; address?: string; last_seen?: string; rounds?: number }>>('/api/federation/peers'),
  federationStats: () => get<{ peer_count?: number; rounds_completed?: number; convergence?: number }>('/api/federation/stats'),

  // Werte (5 values)
  values:          () => get<Record<string, number>>('/api/values'),
  valuesPost:      (patch: Record<string, number>) => post<Record<string, number>>('/api/values', patch),

  // Presence (online IDE clients)
  presence:        () => get<PresenceEntry[]>('/api/presence'),

  // Hardware (Neo legacy — useful endpoint)
  hardware:        () => get<{ cpu?: { model?: string; cores?: number; load?: number }; memory?: { total_mb?: number; used_mb?: number }; gpu?: Array<{ name?: string; temp_c?: number; util?: number; mem_mb?: number }> }>('/api/neo/hardware'),
};

// ─── Helpers ────────────────────────────────────────────────────────

export function nameOf(party: BusMessage['from'] | BusMessage['to']): string {
  if (!party) return '?';
  if (typeof party === 'string') return party;
  return party.name ?? '?';
}

export function intentOf(intent: BusMessage['intent']): string {
  if (!intent) return '?';
  if (typeof intent === 'string') return intent;
  return intent.kind ?? '?';
}

export function bytesToHex(bytes?: number[] | string | null, take = 8): string {
  if (!bytes) return '';
  if (typeof bytes === 'string') return bytes.slice(0, take);
  return bytes.slice(0, take / 2).map(b => b.toString(16).padStart(2, '0')).join('');
}
