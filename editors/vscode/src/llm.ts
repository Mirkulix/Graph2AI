// Auto-respond LLM client. Uses Node 18+ native fetch. No third-party deps.
//
// Wired providers:
//   - deepseek  (OpenAI-compatible chat-completions shape)
//   - openai    (OpenAI-compatible chat-completions shape)
//   - groq      (OpenAI-compatible chat-completions shape)
//   - anthropic (/v1/messages — x-api-key + anthropic-version headers)
//   - ollama    (local /api/chat — no auth, stream:false)

export type ProviderType = 'deepseek' | 'openai' | 'anthropic' | 'ollama' | 'groq';

export interface LlmCallOptions {
    providerType: ProviderType;
    apiKey: string;
    model: string;
    baseUrl?: string;
    systemPrompt?: string;
    userContent: string;
    timeoutMs?: number;
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
        default:
            throw new Error(`Unknown provider type: ${String(opts.providerType)}`);
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
