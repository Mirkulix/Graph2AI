// Auto-respond LLM client. Uses Node 18+ native fetch. No third-party deps.
//
// Wired providers:
//   - deepseek (OpenAI-compatible chat-completions shape)
//   - openai   (OpenAI-compatible chat-completions shape)
//   - groq     (OpenAI-compatible chat-completions shape)
// Stubbed providers (return a friendly error):
//   - anthropic (TODO: implement /v1/messages shape)
//   - ollama    (TODO: implement /api/chat local-server flow)

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
            // TODO: implement /v1/messages — different request/response shape
            throw new Error(
                "Anthropic provider not yet implemented in the auto-respond client. Pick 'deepseek', 'openai', or 'groq'.",
            );
        case 'ollama':
            // TODO: implement local /api/chat flow
            throw new Error(
                "Ollama provider not yet implemented in the auto-respond client. Pick 'deepseek', 'openai', or 'groq'.",
            );
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
