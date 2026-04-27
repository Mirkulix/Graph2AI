// Auto-respond LLM client. Uses Node 18+ native fetch. No third-party deps.
//
// Wired providers:
//   - deepseek   (OpenAI-compatible chat-completions shape)
//   - openai     (OpenAI-compatible chat-completions shape)
//   - groq       (OpenAI-compatible chat-completions shape)
//   - anthropic  (/v1/messages — x-api-key + anthropic-version headers)
//   - ollama     (local /api/chat — no auth, stream:false)
//   - vscode-lm  (vscode.lm.selectChatModels — uses the IDE's bundled
//                 LLM access: Copilot in VS Code, etc. No API key.)
//   - server     (delegates to the qo server's /api/llm/proxy — qo holds
//                 the API keys, the IDE only needs baseUrl. The actual
//                 backend is picked via autoRespond.preferredProvider.)

import * as vscode from 'vscode';

export type ProviderType =
    | 'deepseek'
    | 'openai'
    | 'anthropic'
    | 'ollama'
    | 'groq'
    | 'vscode-lm'
    | 'server';

export interface LlmCallOptions {
    providerType: ProviderType;
    apiKey: string;
    model: string;
    baseUrl?: string;
    systemPrompt?: string;
    userContent: string;
    timeoutMs?: number;
    /** This IDE's bus identity. Sent to the qo server-proxy provider so qo
     *  can scope claude-cli-agent to this IDE's registered workspace_path
     *  instead of the default ORBITQ_REPO. Only used when providerType='server'. */
    requesterIdentity?: string;
}

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_SYSTEM_PROMPT =
    'You are a helpful AI coding assistant responding to a peer IDE on a graph message bus. Be concise and direct.';

/**
 * Calls the configured LLM provider with the supplied user content and
 * returns the assistant's reply text. Throws on transport / API errors.
 */
export async function callLlm(opts: LlmCallOptions): Promise<string> {
    switch (opts.providerType) {
        case 'deepseek':
            return callOpenAiCompat({
                ...opts,
                baseUrl: opts.baseUrl || 'https://api.deepseek.com/v1',
            });
        case 'openai':
            return callOpenAiCompat({
                ...opts,
                baseUrl: opts.baseUrl || 'https://api.openai.com/v1',
            });
        case 'groq':
            return callOpenAiCompat({
                ...opts,
                baseUrl: opts.baseUrl || 'https://api.groq.com/openai/v1',
            });
        case 'anthropic':
            return callAnthropic({
                ...opts,
                baseUrl: opts.baseUrl || 'https://api.anthropic.com',
            });
        case 'ollama':
            return callOllama({
                ...opts,
                baseUrl: opts.baseUrl || 'http://localhost:11434',
            });
        case 'vscode-lm':
            return callVscodeLm(opts);
        case 'server':
            return callServerProxy(opts);
        default:
            throw new Error(`Unknown provider type: ${String(opts.providerType)}`);
    }
}

/**
 * Routes the auto-respond call through the qo server's /api/llm/proxy
 * endpoint. The server holds the actual API keys; the IDE only needs the
 * qo URL (qlang.qlms.baseUrl) and an optional bearer token
 * (qlang.qlms.authToken). The "preferred backend" — which real LLM the
 * server should hit — is taken from qlang.qlms.autoRespond.preferredProvider
 * (defaults to 'claude-cli').
 */
async function callServerProxy(opts: LlmCallOptions): Promise<string> {
    const cfg = vscode.workspace.getConfiguration('qlang.qlms');
    const baseUrl = (cfg.get<string>('baseUrl') || 'http://localhost:4646').replace(/\/$/, '');
    const authToken = cfg.get<string>('authToken') || '';
    const preferred = cfg.get<string>('autoRespond.preferredProvider') || 'claude-cli';

    const url = `${baseUrl}/api/llm/proxy`;
    const sys = opts.systemPrompt && opts.systemPrompt.trim().length > 0
        ? opts.systemPrompt
        : DEFAULT_SYSTEM_PROMPT;
    const messages = [
        { role: 'system', content: sys },
        { role: 'user', content: opts.userContent },
    ];

    const ac = new AbortController();
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeout = setTimeout(() => ac.abort(), timeoutMs);
    try {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
        const resp = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                provider: preferred,
                model: opts.model || undefined,
                messages,
                requester_identity: opts.requesterIdentity || undefined,
            }),
            signal: ac.signal,
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(`server proxy HTTP ${resp.status}: ${text.slice(0, 300)}`);
        }
        const json = (await resp.json()) as { content?: string; error?: string };
        if (json.error) throw new Error(`server proxy: ${json.error}`);
        if (typeof json.content !== 'string' || json.content.length === 0) {
            throw new Error('server proxy returned empty content');
        }
        return json.content;
    } finally {
        clearTimeout(timeout);
    }
}

/**
 * Use the IDE's bundled language-model access. The `model` field is matched
 * against vscode.lm chat-model identifiers — empty matches the first
 * available, otherwise we look for a substring/family/id match. Cursor /
 * Antigravity / Trae / Kiro do NOT expose this API in practice — only
 * stock VS Code with Copilot installed does. Verified via the
 * `qlang.qlms.probeLmModels` command.
 */
async function callVscodeLm(opts: LlmCallOptions): Promise<string> {
    if (!('lm' in vscode) || typeof vscode.lm?.selectChatModels !== 'function') {
        throw new Error(
            'vscode-lm provider: this IDE does not expose vscode.lm — ' +
            'requires VS Code 1.90+ or a fork that bundled the LM API.',
        );
    }
    const wanted = (opts.model || '').trim().toLowerCase();
    const all = await vscode.lm.selectChatModels({});
    if (!all || all.length === 0) {
        throw new Error(
            'vscode-lm: no chat models available. ' +
            'In VS Code: install GitHub Copilot. In Cursor/Antigravity: ' +
            'no extension-facing LM API is exposed — use a different provider.',
        );
    }
    const picked = wanted.length === 0
        ? all[0]
        : all.find((m) => {
            const id = (m.id || '').toLowerCase();
            const family = (m.family || '').toLowerCase();
            const name = (m.name || '').toLowerCase();
            return id === wanted ||
                family === wanted ||
                id.includes(wanted) ||
                family.includes(wanted) ||
                name.includes(wanted);
        }) || all[0];

    const sys = opts.systemPrompt && opts.systemPrompt.trim().length > 0
        ? opts.systemPrompt
        : DEFAULT_SYSTEM_PROMPT;
    const messages = [
        vscode.LanguageModelChatMessage.User(`${sys}\n\n${opts.userContent}`),
    ];

    const ac = new AbortController();
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeout = setTimeout(() => ac.abort(), timeoutMs);
    try {
        const response = await picked.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);
        let buffer = '';
        for await (const fragment of response.text) {
            buffer += fragment;
        }
        if (buffer.length === 0) {
            throw new Error(`vscode-lm '${picked.id}' returned empty response`);
        }
        return buffer;
    } finally {
        clearTimeout(timeout);
    }
}

/** Probe what models the current IDE exposes via vscode.lm. */
export async function listVscodeLmModels(): Promise<Array<{ id: string; family: string; name: string; vendor: string }>> {
    if (!('lm' in vscode) || typeof vscode.lm?.selectChatModels !== 'function') {
        return [];
    }
    try {
        const all = await vscode.lm.selectChatModels({});
        return (all ?? []).map((m) => ({
            id: m.id,
            family: m.family,
            name: m.name,
            vendor: m.vendor,
        }));
    } catch {
        return [];
    }
}

interface OpenAiCompatOpts extends LlmCallOptions {
    baseUrl: string;
}

async function callOpenAiCompat(opts: OpenAiCompatOpts): Promise<string> {
    if (!opts.apiKey) {
        throw new Error(
            `Missing API key for provider '${opts.providerType}'. Set qlang.qlms.autoRespond.apiKey.`,
        );
    }
    const url = `${opts.baseUrl.replace(/\/$/, '')}/chat/completions`;
    const messages: { role: 'system' | 'user'; content: string }[] = [];
    const sys = opts.systemPrompt && opts.systemPrompt.trim().length > 0
        ? opts.systemPrompt
        : DEFAULT_SYSTEM_PROMPT;
    messages.push({ role: 'system', content: sys });
    messages.push({ role: 'user', content: opts.userContent });

    const ac = new AbortController();
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeout = setTimeout(() => ac.abort(), timeoutMs);
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${opts.apiKey}`,
            },
            body: JSON.stringify({
                model: opts.model,
                messages,
                stream: false,
            }),
            signal: ac.signal,
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(
                `LLM ${opts.providerType} HTTP ${resp.status}: ${text.slice(0, 300)}`,
            );
        }
        const json = (await resp.json()) as {
            choices?: { message?: { content?: string } }[];
        };
        const content = json?.choices?.[0]?.message?.content;
        if (typeof content !== 'string' || content.length === 0) {
            throw new Error(`LLM ${opts.providerType} returned empty response`);
        }
        return content;
    } finally {
        clearTimeout(timeout);
    }
}

interface ProviderCallOpts extends LlmCallOptions {
    baseUrl: string;
}

const ANTHROPIC_VERSION = '2023-06-01';
const ANTHROPIC_DEFAULT_MAX_TOKENS = 1024;

async function callAnthropic(opts: ProviderCallOpts): Promise<string> {
    if (!opts.apiKey) {
        throw new Error('Anthropic provider needs apiKey');
    }
    const url = `${opts.baseUrl.replace(/\/$/, '')}/v1/messages`;

    // Anthropic does NOT accept role="system" in messages[] — extract it.
    const sys = opts.systemPrompt && opts.systemPrompt.trim().length > 0
        ? opts.systemPrompt
        : DEFAULT_SYSTEM_PROMPT;
    const messages: { role: 'user' | 'assistant'; content: string }[] = [
        { role: 'user', content: opts.userContent },
    ];
    if (messages.length === 0) {
        throw new Error('Anthropic needs at least one user message');
    }

    const body: Record<string, unknown> = {
        model: opts.model,
        max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
        messages,
    };
    if (sys && sys.length > 0) {
        body.system = sys;
    }

    const ac = new AbortController();
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeout = setTimeout(() => ac.abort(), timeoutMs);
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': opts.apiKey,
                'anthropic-version': ANTHROPIC_VERSION,
            },
            body: JSON.stringify(body),
            signal: ac.signal,
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(
                `anthropic -> HTTP ${resp.status}: ${text.slice(0, 300)}`,
            );
        }
        let json: { content?: { type?: string; text?: string }[] };
        try {
            json = (await resp.json()) as typeof json;
        } catch (e) {
            throw new Error(
                `anthropic -> HTTP ${resp.status}: failed to parse JSON response`,
            );
        }
        const block = json?.content?.find(
            (b) => b && b.type === 'text' && typeof b.text === 'string',
        );
        const text = block?.text;
        if (typeof text !== 'string' || text.length === 0) {
            throw new Error(
                `anthropic -> HTTP ${resp.status}: unexpected response shape (no content[].text)`,
            );
        }
        return text;
    } finally {
        clearTimeout(timeout);
    }
}

async function callOllama(opts: ProviderCallOpts): Promise<string> {
    // Local Ollama needs no auth — apiKey is intentionally ignored.
    const url = `${opts.baseUrl.replace(/\/$/, '')}/api/chat`;
    const messages: { role: 'system' | 'user' | 'assistant'; content: string }[] = [];
    const sys = opts.systemPrompt && opts.systemPrompt.trim().length > 0
        ? opts.systemPrompt
        : DEFAULT_SYSTEM_PROMPT;
    messages.push({ role: 'system', content: sys });
    messages.push({ role: 'user', content: opts.userContent });

    const ac = new AbortController();
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeout = setTimeout(() => ac.abort(), timeoutMs);
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: opts.model,
                messages,
                stream: false,
            }),
            signal: ac.signal,
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(
                `ollama -> HTTP ${resp.status}: ${text.slice(0, 300)}`,
            );
        }
        let json: { message?: { role?: string; content?: string } };
        try {
            json = (await resp.json()) as typeof json;
        } catch (e) {
            throw new Error(
                `ollama -> HTTP ${resp.status}: failed to parse JSON response`,
            );
        }
        const content = json?.message?.content;
        if (typeof content !== 'string' || content.length === 0) {
            throw new Error(
                `ollama -> HTTP ${resp.status}: unexpected response shape (no message.content)`,
            );
        }
        return content;
    } finally {
        clearTimeout(timeout);
    }
}
