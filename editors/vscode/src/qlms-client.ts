// QLMS client for the QLANG VSCode extension (PRD Task 5.1).
//
// Wraps the QO server's `/qlms/v1.1/deliver` + `/qlms/v1.1/reply` routes
// so future extension features (multi-agent coder, inline graph replay,
// status-bar trust badges) can exchange signed QLMS frames without
// re-implementing base64-over-JSON in every call site.
//
// No third-party deps — the extension sticks to the Node 18 runtime that
// VSCode ships. `fetch` is ambient in Node 18+, `Buffer` handles base64.
// Strict mode compiles cleanly against tsconfig.json in this folder.

export interface GraphMessage {
    id: number;
    from: { name: string; capabilities: unknown[] };
    to: { name: string; capabilities: unknown[] };
    graph: unknown;
    inputs: Record<string, unknown>;
    intent: unknown;
    in_reply_to: number | null;
    signature: number[] | null;
    signer_pubkey: number[] | null;
    graph_hash: number[] | null;
}

export interface DeliverResponse {
    version: number;
    flags: number;
    signed: boolean;
    signature_verified: boolean;
    msg_count: number;
    messages: GraphMessage[];
}

export interface ReplyResponse {
    encoding: string;
    frame: string;
    size_bytes: number;
}

export interface QlmsClientOptions {
    /** e.g. "http://localhost:4646" — no trailing slash. */
    baseUrl: string;
    /** Optional bearer token. Falls back to QO_AUTH_TOKEN env var. */
    authToken?: string;
    /** Per-request timeout (ms). Default 15 000. */
    timeoutMs?: number;
}

export class QlmsClientError extends Error {
    constructor(
        message: string,
        public readonly status?: number,
        public readonly body?: string,
    ) {
        super(message);
        this.name = 'QlmsClientError';
    }
}

/**
 * Thin HTTP client for QLMS v1.1 bridge routes. Methods return the raw
 * JSON shape the server sends; signature-verification failures surface
 * as HTTP 401 and are therefore thrown as `QlmsClientError` with
 * `status=401` — callers can distinguish them from network errors via
 * the `status` property.
 */
export class QlmsClient {
    private readonly baseUrl: string;
    private readonly authToken: string | undefined;
    private readonly timeoutMs: number;

    constructor(options: QlmsClientOptions) {
        this.baseUrl = options.baseUrl.replace(/\/$/, '');
        this.authToken =
            options.authToken ??
            (typeof process !== 'undefined' ? process.env?.QO_AUTH_TOKEN : undefined);
        this.timeoutMs = options.timeoutMs ?? 15_000;
    }

    /** Decode + verify a base64-wrapped QLMS envelope. */
    async deliver(frameBase64: string): Promise<DeliverResponse> {
        return this.postJson<DeliverResponse>('/qlms/v1.1/deliver', {
            encoding: 'base64',
            frame: frameBase64,
        });
    }

    /**
     * Build a QLMS envelope from a message array. When `seedHex` is a
     * 64-char hex string, the frame is signed (v2 SIGNED); otherwise
     * it falls back to an unsigned v1 envelope.
     */
    async reply(
        messages: GraphMessage[],
        seedHex?: string,
    ): Promise<ReplyResponse> {
        return this.postJson<ReplyResponse>('/qlms/v1.1/reply', {
            messages,
            seed_hex: seedHex,
        });
    }

    /**
     * Ergonomic helper: takes raw envelope bytes, base64-encodes them,
     * and ships them through /deliver. Useful when the extension reads
     * a `.qlms` fixture off disk.
     */
    async deliverBytes(frame: Uint8Array): Promise<DeliverResponse> {
        const b64 = Buffer.from(frame).toString('base64');
        return this.deliver(b64);
    }

    /**
     * Base64-decode a reply frame into raw bytes. Mirrors
     * `deliverBytes` on the outbound path so the extension can hand a
     * `.qlms` dump to another tool without re-parsing.
     */
    static frameToBytes(frameBase64: string): Uint8Array {
        return Buffer.from(frameBase64, 'base64');
    }

    /**
     * Light "is the server alive?" probe used by the status bar. We
     * send a tiny unsigned envelope through /reply — the server echoes
     * it back through /deliver, so success proves both routes respond.
     */
    async checkConnection(): Promise<QlmsConnectionReport> {
        try {
            const reply = await this.reply([], undefined);
            const deliver = await this.deliver(reply.frame);
            return {
                ok: true,
                signed: deliver.signed,
                signatureVerified: deliver.signature_verified,
                sizeBytes: reply.size_bytes,
            };
        } catch (err) {
            return {
                ok: false,
                signed: false,
                signatureVerified: false,
                sizeBytes: 0,
                error: err instanceof Error ? err.message : String(err),
            };
        }
    }

    private async postJson<T>(path: string, body: unknown): Promise<T> {
        const url = `${this.baseUrl}${path}`;
        const ac = new AbortController();
        const timeout = setTimeout(() => ac.abort(), this.timeoutMs);
        try {
            const headers: Record<string, string> = {
                'Content-Type': 'application/json',
            };
            if (this.authToken) {
                headers['Authorization'] = `Bearer ${this.authToken}`;
            }
            const resp = await fetch(url, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
                signal: ac.signal,
            });
            if (!resp.ok) {
                const text = await resp.text().catch(() => '');
                throw new QlmsClientError(
                    `QLMS ${path} -> HTTP ${resp.status}`,
                    resp.status,
                    text,
                );
            }
            return (await resp.json()) as T;
        } finally {
            clearTimeout(timeout);
        }
    }
}

export interface QlmsConnectionReport {
    ok: boolean;
    signed: boolean;
    signatureVerified: boolean;
    sizeBytes: number;
    error?: string;
}
