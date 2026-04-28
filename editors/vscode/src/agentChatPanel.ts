// Agent chat sidebar — a VS Code WebviewView that lets the user chat with
// the OrbitQO mesh agents (developer, strategist, guardian, artisan,
// researcher, ceo) and any connected peer IDEs without leaving the editor.
//
// The webview is a vanilla HTML/CSS/JS panel (no framework, no extra deps)
// served inline with a strict CSP. The extension host acts as a proxy:
//   - single-mode → POST /api/llm/proxy via callLlm({providerType:'server'})
//   - consensus-mode → POST /api/consensus directly with fetch
//   - agent list → GET /api/messages/agents + GET /api/presence
//
// All requests carry the IDE's bus identity in `requester_identity` so the
// qo server can route claude-cli-agent calls to the correct workspace path.

import * as vscode from 'vscode';
import { randomBytes } from 'crypto';
import { callLlm, ProviderType } from './llm';

const QLMS_CONFIG_SECTION = 'qlang.qlms';
const DEFAULT_AGENTS = ['developer', 'strategist', 'guardian', 'artisan', 'researcher', 'ceo'];

interface PresenceEntry {
    identity: string;
    ide_name?: string;
    host?: string;
    capabilities?: string[];
}

interface AgentListMessage {
    type: 'agents';
    list: { server: string[]; ides: PresenceEntry[] };
}

interface ReplyMessage {
    type: 'reply';
    from: string;
    content: string;
    latencyMs: number;
    ok: boolean;
    error?: string;
}

interface ThinkingMessage {
    type: 'thinking';
    from: string;
}

interface ErrorMessage {
    type: 'error';
    message: string;
}

type ExtensionToWebview = AgentListMessage | ReplyMessage | ThinkingMessage | ErrorMessage;

interface WebviewSendRequest {
    type: 'send';
    mode: 'single' | 'consensus';
    agents: string[];
    prompt: string;
    includeSelection: boolean;
    model?: string;
    preferredProvider?: string;
}

interface WebviewGetAgentsRequest { type: 'getAgents'; }
interface WebviewClearHistoryRequest { type: 'clearHistory'; }

type WebviewToExtension =
    | WebviewSendRequest
    | WebviewGetAgentsRequest
    | WebviewClearHistoryRequest;

/**
 * WebviewViewProvider that renders the sidebar chat panel and bridges
 * webview<->extension messages.
 */
export class AgentChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'qlang.agentChat';

    private view: vscode.WebviewView | undefined;

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly identity: string,
    ) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ): void {
        this.view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [],
        };
        webviewView.webview.html = this.renderHtml(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(
            (msg: WebviewToExtension) => this.handleWebviewMessage(msg),
            undefined,
            this.context.subscriptions,
        );

        // Push the initial agent list as soon as the webview mounts.
        void this.pushAgents();
    }

    /** Posts a message to the webview if it is alive. */
    private post(msg: ExtensionToWebview): void {
        try {
            void this.view?.webview.postMessage(msg);
        } catch {
            // Webview may have been disposed; ignore.
        }
    }

    private async handleWebviewMessage(msg: WebviewToExtension): Promise<void> {
        switch (msg.type) {
            case 'getAgents':
                await this.pushAgents();
                return;
            case 'clearHistory':
                // History lives in the webview state; nothing extension-side to do.
                return;
            case 'send':
                await this.handleSend(msg);
                return;
        }
    }

    /** Refreshes the agent picker by querying qo for server agents + IDEs. */
    private async pushAgents(): Promise<void> {
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        const baseUrl = (cfg.get<string>('baseUrl') || 'http://localhost:4646').replace(/\/$/, '');
        const authToken = cfg.get<string>('authToken') || '';

        let server: string[] = DEFAULT_AGENTS;
        let ides: PresenceEntry[] = [];

        try {
            const headers: Record<string, string> = { Accept: 'application/json' };
            if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
            const r = await fetch(`${baseUrl}/api/messages/agents`, { method: 'GET', headers });
            if (r.ok) {
                const j = (await r.json()) as unknown;
                if (Array.isArray(j)) {
                    server = j.filter((x): x is string => typeof x === 'string' && x.length > 0);
                    if (server.length === 0) server = DEFAULT_AGENTS;
                }
            }
        } catch {
            // Fall back to the hardcoded mesh roster.
        }

        try {
            const headers: Record<string, string> = { Accept: 'application/json' };
            if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
            const r = await fetch(`${baseUrl}/api/presence`, { method: 'GET', headers });
            if (r.ok) {
                const j = (await r.json()) as unknown;
                if (Array.isArray(j)) {
                    ides = j
                        .filter((p): p is PresenceEntry =>
                            !!p && typeof p === 'object' && typeof (p as PresenceEntry).identity === 'string')
                        .filter((p) => p.identity !== this.identity);
                }
            }
        } catch {
            ides = [];
        }

        this.post({ type: 'agents', list: { server, ides } });
    }

    /** Reads the active editor's selection, formatted for inclusion in a prompt. */
    private readSelection(): string | undefined {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return undefined;
        const sel = editor.selection;
        if (!sel || sel.isEmpty) return undefined;
        const text = editor.document.getText(sel);
        if (!text || text.length === 0) return undefined;
        const filename = editor.document.fileName.split(/[\\/]/).pop() || editor.document.fileName;
        const startLine = sel.start.line + 1;
        const endLine = sel.end.line + 1;
        // NOTE: triple-backtick fences inside a template literal — keep these
        // readable by concatenating instead of nesting.
        const fence = '```';
        return `[Code from ${filename}:${startLine}-${endLine}]\n${fence}\n${text}\n${fence}\n`;
    }

    /** Dispatches a send request: single-agent (proxy) or consensus (fan-out). */
    private async handleSend(msg: WebviewSendRequest): Promise<void> {
        const prompt = (msg.prompt || '').trim();
        if (prompt.length === 0) {
            this.post({ type: 'error', message: 'Empty prompt — type something first.' });
            return;
        }
        if (msg.agents.length === 0) {
            this.post({ type: 'error', message: 'Pick at least one agent.' });
            return;
        }

        let userContent = prompt;
        if (msg.includeSelection) {
            const sel = this.readSelection();
            if (sel) userContent = `${sel}\n${prompt}`;
        }

        if (msg.mode === 'consensus' || msg.agents.length > 1) {
            await this.askConsensus(msg.agents, userContent);
        } else {
            await this.askSingle(msg.agents[0]!, userContent, msg.model, msg.preferredProvider);
        }
    }

    /** Single-agent path: server-proxy provider via callLlm. */
    private async askSingle(
        agent: string,
        userContent: string,
        modelOverride: string | undefined,
        preferredProviderOverride: string | undefined,
    ): Promise<void> {
        this.post({ type: 'thinking', from: agent });
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        const model = (modelOverride && modelOverride.trim()) ||
            cfg.get<string>('autoRespond.model', 'sonnet') ||
            'sonnet';
        // Per-agent persona system prompt. Mirrors qo's consensus.rs table so
        // single-mode replies match the same voice as fan-out.
        const sys = systemPromptFor(agent);

        const start = Date.now();
        try {
            // We thread the user's preferred provider override through a temporary
            // configuration tweak: callLlm reads it directly from the IDE config.
            // Since we cannot mutate config per-call, we route via a direct fetch
            // when the user picked a non-default provider.
            const content = preferredProviderOverride
                ? await this.callServerProxyDirect(agent, sys, userContent, preferredProviderOverride, model)
                : await callLlm({
                    providerType: 'server' as ProviderType,
                    apiKey: '',
                    model,
                    systemPrompt: sys,
                    userContent,
                    requesterIdentity: this.identity,
                });
            const latencyMs = Date.now() - start;
            this.post({ type: 'reply', from: agent, content, latencyMs, ok: true });
        } catch (err) {
            const latencyMs = Date.now() - start;
            this.post({
                type: 'reply',
                from: agent,
                content: '',
                latencyMs,
                ok: false,
                error: err instanceof Error ? err.message : String(err),
            });
        }
    }

    /** Direct /api/llm/proxy POST — used when the user picks a per-call provider. */
    private async callServerProxyDirect(
        agent: string,
        systemPrompt: string,
        userContent: string,
        preferredProvider: string,
        model: string,
    ): Promise<string> {
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        const baseUrl = (cfg.get<string>('baseUrl') || 'http://localhost:4646').replace(/\/$/, '');
        const authToken = cfg.get<string>('authToken') || '';

        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

        const finalSystem = `${systemPrompt}\n\n[Requested role: ${agent}]`;
        const messages = [
            { role: 'system', content: finalSystem },
            { role: 'user', content: userContent },
        ];
        const ac = new AbortController();
        const timeout = setTimeout(() => ac.abort(), 180_000);
        try {
            const r = await fetch(`${baseUrl}/api/llm/proxy`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    provider: preferredProvider,
                    model: model || undefined,
                    messages,
                    requester_identity: this.identity,
                }),
                signal: ac.signal,
            });
            if (!r.ok) {
                const body = await r.text().catch(() => '');
                throw new Error(`HTTP ${r.status}: ${body.slice(0, 200)}`);
            }
            const j = (await r.json()) as { content?: string; error?: string };
            if (j.error) throw new Error(j.error);
            if (typeof j.content !== 'string' || j.content.length === 0) {
                throw new Error('empty content');
            }
            return j.content;
        } finally {
            clearTimeout(timeout);
        }
    }

    /** Consensus path: POST /api/consensus, then post one reply per agent. */
    private async askConsensus(agents: string[], prompt: string): Promise<void> {
        for (const a of agents) {
            this.post({ type: 'thinking', from: a });
        }
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        const baseUrl = (cfg.get<string>('baseUrl') || 'http://localhost:4646').replace(/\/$/, '');
        const authToken = cfg.get<string>('authToken') || '';

        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

        const ac = new AbortController();
        const timeout = setTimeout(() => ac.abort(), 120_000);
        try {
            const r = await fetch(`${baseUrl}/api/consensus`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    prompt,
                    agents,
                    timeout_ms: 60_000,
                }),
                signal: ac.signal,
            });
            if (!r.ok) {
                const body = await r.text().catch(() => '');
                throw new Error(`consensus HTTP ${r.status}: ${body.slice(0, 200)}`);
            }
            interface Reply { agent: string; content: string; latency_ms: number; ok: boolean; error?: string; }
            const j = (await r.json()) as { replies?: Reply[] };
            const replies = Array.isArray(j.replies) ? j.replies : [];
            for (const reply of replies) {
                this.post({
                    type: 'reply',
                    from: reply.agent,
                    content: reply.content || '',
                    latencyMs: reply.latency_ms || 0,
                    ok: !!reply.ok,
                    error: reply.error,
                });
            }
            // If qo returned fewer entries than asked, fill the gap with errors.
            const got = new Set(replies.map((rp) => rp.agent));
            for (const a of agents) {
                if (!got.has(a)) {
                    this.post({
                        type: 'reply',
                        from: a,
                        content: '',
                        latencyMs: 0,
                        ok: false,
                        error: 'no reply received from /api/consensus',
                    });
                }
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            for (const a of agents) {
                this.post({
                    type: 'reply',
                    from: a,
                    content: '',
                    latencyMs: 0,
                    ok: false,
                    error: message,
                });
            }
        } finally {
            clearTimeout(timeout);
        }
    }

    /** Builds the webview HTML with a per-load nonce + locked-down CSP. */
    private renderHtml(webview: vscode.Webview): string {
        const nonce = randomBytes(16).toString('hex');
        const csp = [
            `default-src 'none'`,
            `style-src ${webview.cspSource} 'unsafe-inline'`,
            `script-src 'nonce-${nonce}'`,
            `font-src ${webview.cspSource}`,
            `img-src ${webview.cspSource} data:`,
        ].join('; ');

        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OrbitQO Agent Chat</title>
<style>
:root {
  color-scheme: var(--vscode-color-scheme, light dark);
}
* { box-sizing: border-box; }
html, body {
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: var(--vscode-font-family);
  font-size: var(--vscode-font-size);
  color: var(--vscode-foreground);
  background: var(--vscode-sideBar-background, var(--vscode-editor-background));
}
body { display: flex; flex-direction: column; height: 100vh; }
.header {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px;
  border-bottom: 1px solid var(--vscode-panel-border, transparent);
  background: var(--vscode-sideBarSectionHeader-background, transparent);
  align-items: center;
}
.header select, .header button {
  font-family: inherit;
  font-size: 12px;
  background: var(--vscode-dropdown-background);
  color: var(--vscode-dropdown-foreground);
  border: 1px solid var(--vscode-dropdown-border, var(--vscode-focusBorder));
  padding: 3px 6px;
  border-radius: 2px;
  cursor: pointer;
}
.header select:focus, .header button:focus {
  outline: 1px solid var(--vscode-focusBorder);
  outline-offset: -1px;
}
.icon-btn {
  background: transparent !important;
  border: none !important;
  padding: 3px 6px !important;
  font-size: 14px !important;
}
.settings {
  display: none;
  padding: 8px;
  border-bottom: 1px solid var(--vscode-panel-border, transparent);
  background: var(--vscode-editor-background);
  font-size: 12px;
}
.settings.open { display: block; }
.settings label {
  display: block;
  margin-top: 6px;
  margin-bottom: 2px;
  color: var(--vscode-descriptionForeground);
}
.settings input, .settings select {
  width: 100%;
  font-family: inherit;
  font-size: 12px;
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border, var(--vscode-focusBorder));
  padding: 3px 6px;
  border-radius: 2px;
}
.multi-agents {
  display: none;
  padding: 8px;
  border-bottom: 1px solid var(--vscode-panel-border, transparent);
  background: var(--vscode-editor-background);
  font-size: 12px;
  max-height: 180px;
  overflow-y: auto;
}
.multi-agents.open { display: block; }
.multi-agents .group-label {
  font-weight: bold;
  color: var(--vscode-descriptionForeground);
  margin: 6px 0 2px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.multi-agents label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 0;
  cursor: pointer;
  user-select: none;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.msg {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.msg-head {
  display: flex;
  gap: 6px;
  align-items: baseline;
  font-size: 11px;
  color: var(--vscode-descriptionForeground);
}
.msg-from {
  font-weight: bold;
  color: var(--vscode-textLink-foreground);
}
.msg-from.you { color: var(--vscode-charts-orange, var(--vscode-textLink-foreground)); }
.msg-from.error { color: var(--vscode-errorForeground); }
.msg-time, .msg-latency { opacity: 0.7; }
.msg-body {
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.45;
  background: var(--vscode-textBlockQuote-background, transparent);
  border-left: 2px solid var(--vscode-textLink-foreground);
  padding: 6px 8px;
  border-radius: 2px;
}
.msg.you .msg-body { border-left-color: var(--vscode-charts-orange, var(--vscode-textLink-foreground)); }
.msg.error .msg-body {
  border-left-color: var(--vscode-errorForeground);
  color: var(--vscode-errorForeground);
}
.msg-body pre {
  background: var(--vscode-textCodeBlock-background, var(--vscode-editor-background));
  padding: 6px 8px;
  border-radius: 2px;
  overflow-x: auto;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 12px;
  margin: 4px 0;
}
.thinking {
  display: flex;
  align-items: center;
  gap: 6px;
  font-style: italic;
  color: var(--vscode-descriptionForeground);
  font-size: 12px;
}
.thinking::before {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--vscode-progressBar-background, var(--vscode-textLink-foreground));
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
.composer {
  padding: 8px;
  border-top: 1px solid var(--vscode-panel-border, transparent);
  background: var(--vscode-sideBarSectionHeader-background, transparent);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.composer-row {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 12px;
  color: var(--vscode-descriptionForeground);
}
.composer textarea {
  width: 100%;
  min-height: 60px;
  resize: vertical;
  font-family: inherit;
  font-size: 13px;
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border, var(--vscode-focusBorder));
  padding: 6px 8px;
  border-radius: 2px;
}
.composer textarea:focus {
  outline: 1px solid var(--vscode-focusBorder);
  outline-offset: -1px;
}
.composer-actions {
  display: flex;
  gap: 6px;
}
.btn-primary {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
  border: none;
  padding: 5px 12px;
  border-radius: 2px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}
.btn-primary:hover { background: var(--vscode-button-hoverBackground); }
.btn-secondary {
  background: var(--vscode-button-secondaryBackground, transparent);
  color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
  border: 1px solid var(--vscode-button-border, var(--vscode-focusBorder));
  padding: 5px 12px;
  border-radius: 2px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}
.btn-secondary:hover {
  background: var(--vscode-button-secondaryHoverBackground, var(--vscode-list-hoverBackground));
}
.identity-tag {
  font-size: 10px;
  opacity: 0.6;
  padding: 0 4px;
}
.empty-state {
  padding: 20px;
  text-align: center;
  color: var(--vscode-descriptionForeground);
  font-style: italic;
}
</style>
</head>
<body>
  <div class="header">
    <select id="agent-picker" title="Pick the agent that will answer">
      <option value="developer">developer</option>
    </select>
    <select id="mode-picker" title="single = one agent, consensus = ask many in parallel">
      <option value="single">mode: single</option>
      <option value="consensus">mode: consensus</option>
    </select>
    <button class="icon-btn" id="settings-toggle" title="Settings">&#9776;</button>
    <button class="icon-btn" id="refresh-agents" title="Refresh agent list">&#x21bb;</button>
    <span class="identity-tag" id="identity-tag"></span>
  </div>

  <div class="settings" id="settings-panel">
    <label for="provider-input">Preferred provider (server-side backend)</label>
    <select id="provider-input">
      <option value="">(use IDE setting)</option>
      <option value="claude-cli">claude-cli (Max subscription, fast, no tools)</option>
      <option value="claude-cli-agent">claude-cli-agent (with Read/Edit/Bash tools)</option>
      <option value="anthropic">anthropic (direct API)</option>
      <option value="deepseek">deepseek</option>
      <option value="groq">groq</option>
      <option value="ollama">ollama (local)</option>
      <option value="openai">openai</option>
    </select>
    <label for="model-input">Model (free text — defaults to "sonnet")</label>
    <input id="model-input" type="text" placeholder="sonnet" />
  </div>

  <div class="multi-agents" id="multi-agents-panel">
    <div class="group-label">Pick agents (consensus mode)</div>
    <div id="multi-agents-list"></div>
  </div>

  <div class="messages" id="messages">
    <div class="empty-state">Pick an agent and ask anything. Ctrl+Enter sends.</div>
  </div>

  <div class="composer">
    <div class="composer-row">
      <label>
        <input type="checkbox" id="include-selection" />
        include selected code
      </label>
    </div>
    <textarea id="prompt-input" placeholder="Ask the mesh anything &mdash; Ctrl+Enter sends, Esc clears"></textarea>
    <div class="composer-actions">
      <button class="btn-primary" id="send-btn">Send (Ctrl+Enter)</button>
      <button class="btn-secondary" id="clear-btn">Clear chat</button>
    </div>
  </div>

<script nonce="${nonce}">
${WEBVIEW_JS.replace('__IDENTITY__', JSON.stringify(this.identity))}
</script>
</body>
</html>`;
    }
}

/**
 * Persona system prompts. Keep in sync with `system_prompt_for` in
 * `qo/qo-server/src/routes/consensus.rs` so single-mode and consensus-mode
 * speak with the same voice for each role.
 */
function systemPromptFor(role: string): string {
    switch (role) {
        case 'ceo':
            return 'You are CEO, a coordinator agent. Decompose the user\'s request into clear steps, suggest which specialist should handle each step (developer, researcher, guardian, strategist, artisan), and give a one-paragraph executive summary.';
        case 'developer':
            return 'You are Developer, a senior software engineer. Review code, suggest refactors, write functions, and explain trade-offs. Be precise. Use code blocks for any code you produce.';
        case 'researcher':
            return 'You are Researcher, a knowledge synthesizer. Find relevant information, cite sources when possible, summarize concisely, and flag uncertainty.';
        case 'guardian':
            return 'You are Guardian, a security and safety reviewer. Find vulnerabilities, unsafe patterns, missing validation, and compliance gaps. Suggest concrete mitigations.';
        case 'strategist':
            return 'You are Strategist, a planning advisor. Lay out multi-step strategies, trade-offs, and second-order effects. Prefer numbered plans.';
        case 'artisan':
            return 'You are Artisan, a creative implementer. Generate concrete artifacts (text, prose, examples, snippets) that match the user\'s intent.';
        default:
            return 'You are an AI assistant on the OrbitQO mesh. Help the user with their request. Be precise and concise.';
    }
}

/**
 * Webview-side script. Pure vanilla JS — no bundler, no framework, no extra
 * deps. The constant `__IDENTITY__` is replaced at render time with the
 * IDE's bus identity (stringified) so the panel can label outgoing turns.
 */
const WEBVIEW_JS = `(() => {
  'use strict';
  const vscode = acquireVsCodeApi();
  const IDENTITY = __IDENTITY__;
  const MAX_HISTORY = 30;

  const $ = (id) => document.getElementById(id);
  const messagesEl = $('messages');
  const promptEl = $('prompt-input');
  const agentPicker = $('agent-picker');
  const modePicker = $('mode-picker');
  const settingsToggle = $('settings-toggle');
  const settingsPanel = $('settings-panel');
  const multiPanel = $('multi-agents-panel');
  const multiList = $('multi-agents-list');
  const includeSel = $('include-selection');
  const sendBtn = $('send-btn');
  const clearBtn = $('clear-btn');
  const refreshBtn = $('refresh-agents');
  const providerInput = $('provider-input');
  const modelInput = $('model-input');
  const identityTag = $('identity-tag');

  identityTag.textContent = 'as ' + IDENTITY;

  // --- Persisted state -------------------------------------------------
  const persisted = vscode.getState() || {};
  let history = Array.isArray(persisted.history) ? persisted.history.slice(-MAX_HISTORY) : [];
  let lastAgent = typeof persisted.lastAgent === 'string' ? persisted.lastAgent : 'developer';
  let lastMode = persisted.lastMode === 'consensus' ? 'consensus' : 'single';
  let lastProvider = typeof persisted.lastProvider === 'string' ? persisted.lastProvider : '';
  let lastModel = typeof persisted.lastModel === 'string' ? persisted.lastModel : '';
  let lastSelected = Array.isArray(persisted.lastSelected) ? persisted.lastSelected : ['developer'];
  let knownServerAgents = ['developer', 'strategist', 'guardian', 'artisan', 'researcher', 'ceo'];
  let knownIdes = [];

  modePicker.value = lastMode;
  providerInput.value = lastProvider;
  modelInput.value = lastModel;
  applyMode();

  function persist() {
    vscode.setState({
      history: history.slice(-MAX_HISTORY),
      lastAgent,
      lastMode,
      lastProvider,
      lastModel,
      lastSelected,
    });
  }

  function applyMode() {
    if (modePicker.value === 'consensus') {
      multiPanel.classList.add('open');
      agentPicker.style.display = 'none';
    } else {
      multiPanel.classList.remove('open');
      agentPicker.style.display = '';
    }
  }

  // --- Rendering -------------------------------------------------------
  function renderAll() {
    messagesEl.innerHTML = '';
    if (history.length === 0) {
      const e = document.createElement('div');
      e.className = 'empty-state';
      e.textContent = 'Pick an agent and ask anything. Ctrl+Enter sends.';
      messagesEl.appendChild(e);
      return;
    }
    for (const m of history) renderMessage(m, false);
    scrollToBottom();
  }

  function renderMessage(m, scroll) {
    const empty = messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    const wrap = document.createElement('div');
    wrap.className = 'msg' + (m.kind === 'you' ? ' you' : '') + (m.ok === false ? ' error' : '');
    const head = document.createElement('div');
    head.className = 'msg-head';
    const from = document.createElement('span');
    from.className = 'msg-from' + (m.kind === 'you' ? ' you' : '') + (m.ok === false ? ' error' : '');
    from.textContent = (m.kind === 'you' ? 'you' : (m.from || '?'));
    head.appendChild(from);
    const time = document.createElement('span');
    time.className = 'msg-time';
    time.textContent = m.ts || '';
    head.appendChild(time);
    if (typeof m.latencyMs === 'number' && m.latencyMs > 0) {
      const lat = document.createElement('span');
      lat.className = 'msg-latency';
      lat.textContent = '(' + (m.latencyMs / 1000).toFixed(1) + 's)';
      head.appendChild(lat);
    }
    wrap.appendChild(head);
    const body = document.createElement('div');
    body.className = 'msg-body';
    renderBody(body, m.ok === false ? (m.error || '(unknown error)') : (m.content || ''));
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    if (scroll) scrollToBottom();
  }

  function renderBody(container, text) {
    // Minimal markdown: split out fenced code blocks. Everything else goes
    // through textContent so HTML in agent replies cannot escape the panel.
    const parts = [];
    const re = /\`\`\`([\\s\\S]*?)\`\`\`/g;
    let lastIdx = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > lastIdx) {
        parts.push({ kind: 'text', body: text.slice(lastIdx, m.index) });
      }
      parts.push({ kind: 'code', body: m[1].replace(/^[a-zA-Z0-9_+-]*\\n/, '') });
      lastIdx = re.lastIndex;
    }
    if (lastIdx < text.length) parts.push({ kind: 'text', body: text.slice(lastIdx) });
    if (parts.length === 0) parts.push({ kind: 'text', body: text });

    for (const p of parts) {
      if (p.kind === 'code') {
        const pre = document.createElement('pre');
        pre.textContent = p.body;
        container.appendChild(pre);
      } else {
        const span = document.createElement('span');
        span.textContent = p.body;
        container.appendChild(span);
      }
    }
  }

  function showThinking(from) {
    const empty = messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();
    const wrap = document.createElement('div');
    wrap.className = 'msg thinking-wrap';
    wrap.dataset.thinkingFrom = from;
    const head = document.createElement('div');
    head.className = 'msg-head';
    const fromEl = document.createElement('span');
    fromEl.className = 'msg-from';
    fromEl.textContent = from;
    head.appendChild(fromEl);
    wrap.appendChild(head);
    const body = document.createElement('div');
    body.className = 'thinking';
    body.textContent = 'thinking…';
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function clearThinking(from) {
    const sel = '[data-thinking-from="' + cssEscape(from) + '"]';
    const el = messagesEl.querySelector(sel);
    if (el) el.remove();
  }

  function cssEscape(s) {
    return String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => '\\\\' + c);
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function nowHHMM() {
    const d = new Date();
    return d.toTimeString().slice(0, 5);
  }

  // --- Agent picker ----------------------------------------------------
  function rebuildAgentPicker() {
    agentPicker.innerHTML = '';
    for (const a of knownServerAgents) {
      const o = document.createElement('option');
      o.value = a;
      o.textContent = a;
      agentPicker.appendChild(o);
    }
    if (knownIdes.length > 0) {
      const sep = document.createElement('option');
      sep.disabled = true;
      sep.textContent = '─── connected IDEs ───';
      agentPicker.appendChild(sep);
      for (const ide of knownIdes) {
        const o = document.createElement('option');
        o.value = ide.identity;
        const lbl = ide.ide_name ? (ide.identity + '  (' + ide.ide_name + ')') : ide.identity;
        o.textContent = lbl;
        agentPicker.appendChild(o);
      }
    }
    if (lastAgent && [...agentPicker.options].some((o) => o.value === lastAgent)) {
      agentPicker.value = lastAgent;
    }
  }

  function rebuildMultiList() {
    multiList.innerHTML = '';
    const groupServer = document.createElement('div');
    groupServer.className = 'group-label';
    groupServer.textContent = 'Server agents';
    multiList.appendChild(groupServer);
    for (const a of knownServerAgents) {
      multiList.appendChild(makeCheckbox(a, lastSelected.includes(a)));
    }
    if (knownIdes.length > 0) {
      const groupIdes = document.createElement('div');
      groupIdes.className = 'group-label';
      groupIdes.textContent = 'Connected IDEs';
      multiList.appendChild(groupIdes);
      for (const ide of knownIdes) {
        multiList.appendChild(makeCheckbox(ide.identity, lastSelected.includes(ide.identity), ide.ide_name));
      }
    }
  }

  function makeCheckbox(value, checked, label) {
    const wrap = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = value;
    cb.checked = !!checked;
    cb.dataset.agent = value;
    wrap.appendChild(cb);
    const span = document.createElement('span');
    span.textContent = label ? (value + '  (' + label + ')') : value;
    wrap.appendChild(span);
    return wrap;
  }

  function selectedConsensusAgents() {
    return [...multiList.querySelectorAll('input[type="checkbox"]:checked')]
      .map((cb) => cb.value);
  }

  // --- Send ------------------------------------------------------------
  function doSend() {
    const text = (promptEl.value || '').trim();
    if (text.length === 0) return;

    const mode = modePicker.value === 'consensus' ? 'consensus' : 'single';
    let agents;
    if (mode === 'single') {
      agents = [agentPicker.value || 'developer'];
      lastAgent = agents[0];
    } else {
      agents = selectedConsensusAgents();
      if (agents.length === 0) {
        alert('Pick at least one agent in the consensus list.');
        return;
      }
      lastSelected = agents.slice();
    }
    lastMode = mode;
    lastProvider = providerInput.value || '';
    lastModel = modelInput.value || '';

    // Echo the user turn into history.
    const youTurn = {
      kind: 'you',
      from: IDENTITY,
      content: text,
      ts: nowHHMM(),
      ok: true,
    };
    history.push(youTurn);
    history = history.slice(-MAX_HISTORY);
    renderMessage(youTurn, true);
    persist();

    promptEl.value = '';

    vscode.postMessage({
      type: 'send',
      mode,
      agents,
      prompt: text,
      includeSelection: !!includeSel.checked,
      model: lastModel || undefined,
      preferredProvider: lastProvider || undefined,
    });
  }

  // --- Wiring ----------------------------------------------------------
  sendBtn.addEventListener('click', doSend);
  clearBtn.addEventListener('click', () => {
    history = [];
    persist();
    renderAll();
    vscode.postMessage({ type: 'clearHistory' });
  });
  refreshBtn.addEventListener('click', () => vscode.postMessage({ type: 'getAgents' }));
  settingsToggle.addEventListener('click', () => settingsPanel.classList.toggle('open'));
  modePicker.addEventListener('change', () => {
    lastMode = modePicker.value === 'consensus' ? 'consensus' : 'single';
    applyMode();
    persist();
  });
  agentPicker.addEventListener('change', () => { lastAgent = agentPicker.value; persist(); });
  providerInput.addEventListener('change', () => { lastProvider = providerInput.value; persist(); });
  modelInput.addEventListener('input', () => { lastModel = modelInput.value; persist(); });

  promptEl.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      doSend();
    } else if (e.key === 'Escape') {
      promptEl.value = '';
    }
  });

  window.addEventListener('message', (event) => {
    const m = event.data;
    if (!m || typeof m !== 'object') return;
    if (m.type === 'agents') {
      knownServerAgents = (m.list && Array.isArray(m.list.server) && m.list.server.length > 0)
        ? m.list.server
        : knownServerAgents;
      knownIdes = (m.list && Array.isArray(m.list.ides)) ? m.list.ides : [];
      rebuildAgentPicker();
      rebuildMultiList();
    } else if (m.type === 'thinking') {
      showThinking(m.from);
    } else if (m.type === 'reply') {
      clearThinking(m.from);
      const turn = {
        kind: 'agent',
        from: m.from,
        content: m.content || '',
        latencyMs: m.latencyMs,
        ts: nowHHMM(),
        ok: m.ok !== false,
        error: m.error,
      };
      history.push(turn);
      history = history.slice(-MAX_HISTORY);
      renderMessage(turn, true);
      persist();
    } else if (m.type === 'error') {
      const turn = {
        kind: 'agent',
        from: 'system',
        content: '',
        latencyMs: 0,
        ts: nowHHMM(),
        ok: false,
        error: m.message || 'unknown error',
      };
      history.push(turn);
      history = history.slice(-MAX_HISTORY);
      renderMessage(turn, true);
      persist();
    }
  });

  rebuildAgentPicker();
  rebuildMultiList();
  renderAll();
  vscode.postMessage({ type: 'getAgents' });
})();`;
