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
  auto_triggered?: boolean;
  trigger_kind?: string;
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
  // Absolute path of the IDE's currently-open workspace.
  workspace_path?: string;
  // Whether the QO supervisor may dispatch swarm subtasks to this IDE.
  // Defaults to true when the field is missing.
  eligible_for_swarms?: boolean;
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

export interface KnowledgeStats {
  verified: number;
  observed: number;
  proposed: number;
  stale: number;
  refuted: number;
  load_bearing: number;
  total: number;
  entities: number;
}

export interface KnowledgeEntity {
  id: string;
  kind: string;
  name: string;
}

export interface KnowledgeEvidence {
  kind: string;
  locator: string;
  lines?: [number, number] | null;
  excerpt?: string | null;
  supports: boolean;
}

export interface KnowledgeClaim {
  id: string;
  statement: string;
  subject: string;
  relation?: string | null;
  object?: string | null;
  status: 'observed' | 'proposed' | 'verified' | 'stale' | 'refuted' | string;
  provenance: { producer: string; observed_at: number; git_revision?: string | null; run_id?: string | null };
  evidence: KnowledgeEvidence[];
  revision: number;
}

export interface KnowledgeSnapshot {
  entities: KnowledgeEntity[];
  claims: KnowledgeClaim[];
}

// ─── Consensus (multi-agent fan-out) ─────────────────────────────────

export interface ConsensusRequest {
  prompt: string;
  agents: string[];
  system_prompt?: string | null;
  timeout_ms?: number;
}

export interface ConsensusReply {
  agent: string;
  content: string;
  latency_ms: number;
  ok: boolean;
  error?: string | null;
}

export type ConsensusLabel =
  | 'strong-agreement'
  | 'majority-agrees'
  | 'mixed-signals'
  | 'diverse-opinions'
  | string;

export interface ConsensusSummary {
  total_replies: number;
  successful: number;
  failed: number;
  avg_latency_ms: number;
  consensus_score: number;
  consensus_label: ConsensusLabel;
  // Pseudo-semantic score (character-trigram cosine). Optional for
  // backward compat with older backends that only return Jaccard.
  consensus_score_semantic?: number;
  consensus_label_semantic?: ConsensusLabel;
}

export interface ConsensusResponse {
  prompt: string;
  agents_asked: string[];
  replies: ConsensusReply[];
  summary: ConsensusSummary;
}

// ─── Swarm (autonomous multi-agent runs) ────────────────────────────

export type SwarmStatus =
  | 'planning'
  | 'dispatching'
  | 'evaluating'
  | 'done'
  | 'stopped'
  | 'error';

export type SubtaskStatus = 'pending' | 'running' | 'done' | 'error';

export interface SwarmSubtask {
  id: string;
  assigned_to: string;
  prompt: string;
  response?: string;
  status: SubtaskStatus;
  started_at?: number;
  finished_at?: number;
}

export interface SwarmRound {
  round_number: number;
  plan: string;
  subtasks: SwarmSubtask[];
  eval?: string;
}

export interface SwarmState {
  id: number;
  goal: string;
  status: SwarmStatus;
  rounds: SwarmRound[];
  tokens_used: number;
  cost_usd: number;
  started_at: number;
  finished_at?: number;
  stop_requested: boolean;
}

// ─── Local Multi-Agent runs (Planner -> Worker -> Reviewer) ─────────

export interface MultiAgentRunRequest {
  goal: string;
  max_revisions?: number;
  write_artifacts?: boolean;
}

export interface MultiAgentPlan {
  goal_summary: string;
  deliverable: string;
  acceptance_criteria: string[];
  worker_instructions: string;
}

export interface MultiAgentAgentOutput {
  agent: string;
  tier: string;
  output: string;
}

export interface MultiAgentArtifact {
  path: string;
  content: string;
}

export interface MultiAgentArtifactWriteResult {
  path: string;
  written: boolean;
  resolved_path?: string | null;
  error?: string | null;
}

export interface MultiAgentWorkerRound {
  iteration: number;
  tier: string;
  output: string;
  artifacts: MultiAgentArtifact[];
  artifact_writes: MultiAgentArtifactWriteResult[];
}

export interface MultiAgentReviewerRound {
  iteration: number;
  tier: string;
  approved: boolean;
  feedback: string;
  final_answer: string;
  output: string;
}

export interface MultiAgentRunResponse {
  run_id: number;
  started_at: number;
  finished_at: number;
  mode: string;
  goal: string;
  status: string;
  plan: MultiAgentPlan;
  planner: MultiAgentAgentOutput;
  worker_rounds: MultiAgentWorkerRound[];
  reviewer_rounds: MultiAgentReviewerRound[];
  deliverable: string;
  final_answer: string;
}

export interface MultiAgentRunStartedResponse {
  run_id: number;
}

export interface StoredMultiAgentRun {
  run_id: number;
  started_at: number;
  finished_at?: number | null;
  request: MultiAgentRunRequest;
  goal: string;
  mode: string;
  status: string;
  phase: string;
  plan?: MultiAgentPlan | null;
  planner?: MultiAgentAgentOutput | null;
  worker_rounds: MultiAgentWorkerRound[];
  reviewer_rounds: MultiAgentReviewerRound[];
  deliverable?: string | null;
  final_answer?: string | null;
  error?: string | null;
}

export interface MultiAgentRunSummary {
  run_id: number;
  started_at: number;
  finished_at?: number | null;
  mode: string;
  goal: string;
  status: string;
  worker_rounds: number;
  reviewer_rounds: number;
  artifacts_detected: number;
  artifacts_written: number;
}

export interface MultiAgentRunEvent {
  kind: string;
  run: StoredMultiAgentRun;
}

// ─── Autonomous mode (always-on swarm scheduler) ────────────────────

export interface AutonomousConfig {
  interval_seconds: number;
  daily_budget_usd: number;
  max_swarms_per_hour: number;
  goals_queue: string[];
  meta_agent_enabled: boolean;
  swarm_max_rounds: number;
  swarm_max_tokens: number;
}

export interface AutonomousRun {
  swarm_id: number;
  goal: string;
  goal_source: 'queue' | 'meta-agent';
  started_at: number;
  finished_at?: number;
  cost_usd: number;
  status: string;
}

export interface AutonomousState {
  enabled: boolean;
  config: AutonomousConfig;
  started_at?: number;
  paused_reason?: string;
  swarms_today: number;
  spent_today_usd: number;
  last_swarm_at?: number;
  next_swarm_at?: number;
  current_swarm_id?: number;
  history: AutonomousRun[];
}

// ─── Git (auto-branches created by overnight swarms) ────────────────

export interface AutoBranchInfo {
  name: string;             // e.g. "auto/fix-correlation-1745234567"
  last_commit_sha: string;          // backend field name
  last_commit_message: string;
  last_commit_date: string;
  diff?: { files_changed: number; insertions: number; deletions: number } | null;
  // Optional: backend may report whether tests passed for this branch.
  // 'passed' | 'failed' | 'unknown' (or omit). Treated as unknown if missing.
  tests?: 'passed' | 'failed' | 'unknown';
  tests_detail?: string;    // short note, e.g. "cargo build: linker error"
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
  // Server-side ring buffer (cap 200) of the most recent bus messages.
  // Used by the cockpit to hydrate liveTail on a fresh machine where
  // localStorage is empty. Falls back to localStorage on network error.
  recentMessages:  (n: number = 50) => get<BusMessage[]>(`/api/messages/recent?n=${n}`),

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

  // Consensus — fan out one prompt to N agents and rank agreement
  consensus:       (prompt: string, agents: string[], opts?: { systemPrompt?: string; timeoutMs?: number }) =>
                     post<ConsensusResponse>('/api/consensus', {
                       prompt,
                       agents,
                       system_prompt: opts?.systemPrompt ?? null,
                       timeout_ms: opts?.timeoutMs,
                     } satisfies ConsensusRequest),

  // Broadcast — fire-and-forget mesh fan-out. Sends one prompt to every
  // selected IDE identity; replies (if any) trickle back through the
  // standard SSE stream. Use when you want every IDE to react in
  // parallel without the cockpit blocking on completion.
  broadcast:       (prompt: string, targets: string[], opts?: { from?: string }) =>
                     post<{
                       sent: number;
                       from: string;
                       targets: string[];
                       message_ids: number[];
                       failures: Array<{ target: string; error: string }>;
                     }>('/api/broadcast', {
                       prompt,
                       targets,
                       from: opts?.from,
                     }),

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

  // Knowledge graph — read-only cockpit projection of latest claim revisions.
  knowledgeStats:    () => get<KnowledgeStats>('/api/knowledge/stats'),
  knowledgeSnapshot: (limit: number = 100) => get<KnowledgeSnapshot>(`/api/knowledge/snapshot?limit=${Math.max(1, Math.min(500, limit))}`),

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
  presenceSetEligibility: (identity: string, eligible: boolean) =>
                     post<PresenceEntry>(
                       `/api/presence/${encodeURIComponent(identity)}/eligibility`,
                       { eligible_for_swarms: eligible },
                     ),

  // Hardware (Neo legacy — useful endpoint)
  hardware:        () => get<{ cpu?: { model?: string; cores?: number; load?: number }; memory?: { total_mb?: number; used_mb?: number }; gpu?: Array<{ name?: string; temp_c?: number; util?: number; mem_mb?: number }> }>('/api/neo/hardware'),

  // Swarm — autonomous multi-agent orchestration
  swarmStart:      (goal: string, opts?: { maxRounds?: number; maxTokens?: number }) =>
                     post<{ swarm_id: number }>('/api/swarm/start', {
                       goal,
                       max_rounds: opts?.maxRounds,
                       max_tokens: opts?.maxTokens,
                     }),
  swarmGet:        (id: number) => get<SwarmState>(`/api/swarm/${id}`),
  swarmStop:       (id: number) => post<{ stopped: boolean }>(`/api/swarm/${id}/stop`),
  swarmActive:     () => get<SwarmState[]>('/api/swarm/active'),

  // Local multi-agent product path
  multiAgentRun:   (req: MultiAgentRunRequest) =>
                     post<MultiAgentRunResponse>('/api/multi-agent/run', req),
  multiAgentStart: (req: MultiAgentRunRequest) =>
                     post<MultiAgentRunStartedResponse>('/api/multi-agent/runs/start', req),
  multiAgentRuns:  () => get<MultiAgentRunSummary[]>('/api/multi-agent/runs'),
  multiAgentRunGet:(id: number) => get<StoredMultiAgentRun>(`/api/multi-agent/runs/${id}`),

  // Autonomous mode — long-running scheduler that drives swarms automatically
  autonomousStatus:   () => get<AutonomousState>('/api/autonomous/status'),
  autonomousStart:    (config: Partial<AutonomousConfig>) =>
                        post<AutonomousState>('/api/autonomous/start', config),
  autonomousStop:     () => post<AutonomousState>('/api/autonomous/stop'),
  autonomousSetQueue: (goals: string[]) =>
                        put<AutonomousState>('/api/autonomous/queue', { goals }),

  // Git — branches created automatically by overnight swarms
  gitBranches: () => get<AutoBranchInfo[]>('/api/git/branches'),
  gitDiff:     (branch: string) =>
                 get<{ diff: string; truncated: boolean }>(`/api/git/diff/${encodeURIComponent(branch)}`),
  gitMerge:    (branch: string) =>
                 post<{ merged: boolean; conflict?: string }>('/api/git/merge', { branch }),
  gitDiscard:  (branch: string) =>
                 post<{ discarded: boolean }>('/api/git/discard', { branch }),
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
