// QLMS inbox: subscribes to /api/messages/stream (SSE) and surfaces
// incoming bus messages addressed to this IDE's identity. No third-party
// deps — uses Node 18+ native `fetch` with a streaming body reader.

import * as vscode from 'vscode';
import { QlmsClient } from './qlms-client';
import { callLlm, ProviderType } from './llm';

const QLMS_CONFIG_SECTION = 'qlang.qlms';

export interface InboxOptions {
    baseUrl: string;
    identity: string;
    authToken?: string;
    /** Signing seed used for auto-respond replies. Optional. */
    seedHex?: string;
    /** Max entries kept in memory for the showInbox quick-pick. */
    capacity?: number;
}

interface InboxMessage {
    id?: number;
    from?: unknown;
    to?: unknown;
    intent?: unknown;
    content?: unknown;
    is_reply?: unknown;
    graph?: unknown;
    [k: string]: unknown;
}

/**
 * Live inbox for replies addressed to `identity`. Auto-reconnects on
 * stream errors with a 5-second backoff until `dispose()` is called.
 */
export class QlmsInbox {
    private readonly messages: InboxMessage[] = [];
    private readonly capacity: number;
    private stopped = false;
    private controller: AbortController | undefined;

    constructor(private readonly opts: InboxOptions) {
        this.capacity = opts.capacity ?? 50;
    }

    /** Returns a snapshot of the in-memory inbox (newest first). */
    snapshot(): InboxMessage[] {
        return this.messages.slice().reverse();
    }

    /** Stops the SSE loop; safe to call multiple times. */
    dispose(): void {
        this.stopped = true;
        try {
            this.controller?.abort();
        } catch {
            // ignore
        }
    }

    /** Starts the auto-reconnecting SSE loop in the background. */
    start(): void {
        void this.runLoop();
    }

    private async runLoop(): Promise<void> {
        const url = `${this.opts.baseUrl.replace(/\/$/, '')}/api/messages/stream`;
        while (!this.stopped) {
            try {
                await this.consumeStream(url);
            } catch (err) {
                if (this.stopped) return;
                console.warn(
                    `[qlang.inbox] stream error: ${err instanceof Error ? err.message : String(err)}; reconnecting in 5s`,
                );
            }
            if (this.stopped) return;
            await delay(5_000);
        }
    }

    private async consumeStream(url: string): Promise<void> {
        this.controller = new AbortController();
        const headers: Record<string, string> = { Accept: 'text/event-stream' };
        if (this.opts.authToken) {
            headers['Authorization'] = `Bearer ${this.opts.authToken}`;
        }
        const resp = await fetch(url, {
            method: 'GET',
            headers,
            signal: this.controller.signal,
        });
        if (!resp.ok || !resp.body) {
            throw new Error(`HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (!this.stopped) {
            const { value, done } = await reader.read();
            if (done) return;
            buffer += decoder.decode(value, { stream: true });
            // SSE events are separated by blank lines.
            let sepIdx: number;
            while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
                const rawEvent = buffer.slice(0, sepIdx);
                buffer = buffer.slice(sepIdx + 2);
                this.handleEvent(rawEvent);
            }
        }
    }

    private handleEvent(rawEvent: string): void {
        const dataLines: string[] = [];
        for (const line of rawEvent.split('\n')) {
            if (line.startsWith('data: ')) {
                dataLines.push(line.slice(6));
            } else if (line.startsWith('data:')) {
                dataLines.push(line.slice(5));
            }
        }
        if (dataLines.length === 0) return;
        const payload = dataLines.join('\n').trim();
        if (!payload) return;
        let msg: InboxMessage;
        try {
            msg = JSON.parse(payload) as InboxMessage;
        } catch {
            return;
        }
        if (!this.isAddressedToMe(msg)) return;
        this.record(msg);
        this.notify(msg);
        // Auto-respond runs out-of-band so it does not block the SSE reader.
        void this.maybeAutoRespond(msg);
    }

    private isAddressedToMe(msg: InboxMessage): boolean {
        const to = msg.to;
        if (typeof to === 'string') return to === this.opts.identity;
        if (to && typeof to === 'object' && 'name' in to) {
            const name = (to as { name?: unknown }).name;
            return typeof name === 'string' && name === this.opts.identity;
        }
        return false;
    }

    private record(msg: InboxMessage): void {
        this.messages.push(msg);
        while (this.messages.length > this.capacity) {
            this.messages.shift();
        }
    }

    private notify(msg: InboxMessage): void {
        const fromLabel = labelOf(msg.from) ?? '?';
        const intent = typeof msg.intent === 'string' ? msg.intent : (msg.intent ? JSON.stringify(msg.intent) : '?');
        void vscode.window
            .showInformationMessage(
                `Reply from ${fromLabel} (intent: ${intent})`,
                'Open in editor',
                'Dismiss',
            )
            .then((choice) => {
                if (choice === 'Open in editor') {
                    openAsJson(msg);
                }
            });
    }

    /**
     * If auto-respond is enabled and this message is NOT itself a reply,
     * call the configured LLM and ship the answer back as a Result frame.
     */
    private async maybeAutoRespond(msg: InboxMessage): Promise<void> {
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        if (!cfg.get<boolean>('autoRespond.enabled', false)) return;
        if (msg.is_reply === true) return;

        const fromName = labelOf(msg.from);
        if (!fromName) return;

        const userContent = typeof msg.content === 'string' && msg.content.length > 0
            ? msg.content
            : JSON.stringify(msg).slice(0, 4096);

        const providerType = (cfg.get<string>('autoRespond.providerType', 'deepseek') ?? 'deepseek') as ProviderType;
        const apiKey = cfg.get<string>('autoRespond.apiKey', '') ?? '';
        const model = cfg.get<string>('autoRespond.model', 'deepseek-chat') ?? 'deepseek-chat';
        const baseUrlOverride = cfg.get<string>('autoRespond.baseUrl', '') ?? '';
        const systemPrompt = cfg.get<string>('autoRespond.systemPrompt', '') ?? '';

        let answer: string;
        try {
            answer = await callLlm({
                providerType,
                apiKey,
                model,
                baseUrl: baseUrlOverride || undefined,
                systemPrompt: systemPrompt || undefined,
                userContent,
            });
        } catch (err) {
            void vscode.window.showWarningMessage(
                `Auto-respond failed (${providerType}): ${err instanceof Error ? err.message : String(err)}`,
            );
            return;
        }

        try {
            await this.sendReply(fromName, msg, answer);
            void vscode.window.showInformationMessage(
                `Auto-responded as ${this.opts.identity} → ${fromName}`,
            );
        } catch (err) {
            void vscode.window.showWarningMessage(
                `Auto-respond send failed: ${err instanceof Error ? err.message : String(err)}`,
            );
        }
    }

    private async sendReply(toName: string, original: InboxMessage, answer: string): Promise<void> {
        const client = new QlmsClient({
            baseUrl: this.opts.baseUrl,
            authToken: this.opts.authToken,
        });
        const replyId = Math.floor(Math.random() * 1_000_000);
        const originalId = typeof original.id === 'number' ? original.id : null;
        const graph: Record<string, unknown> = {
            id: `vscode-autorespond-${Date.now()}`,
            version: '1.0',
            nodes: [],
            edges: [],
            constraints: [],
            metadata: {
                source: 'vscode-extension-autorespond',
                identity: this.opts.identity,
                content: answer,
            },
        };
        const reply = QlmsClient.createMessage({
            id: replyId,
            from: this.opts.identity,
            to: toName,
            graph,
        });
        // Rust enum MessageIntent::Result is a struct variant — must be the
        // {Result: {original_message_id}} object, NOT the bare string 'Result'.
        // The string form gets a 422 from /qlms/v1.1/reply.
        (reply as unknown as { intent: unknown }).intent = {
            Result: { original_message_id: originalId ?? 0 },
        };
        if (originalId !== null) {
            (reply as unknown as { in_reply_to: number | null }).in_reply_to = originalId;
        }
        const built = await client.reply([reply], this.opts.seedHex);
        await client.deliver(built.frame);
    }
}

/** Opens an inbox message as a new untitled JSON document. */
export function openAsJson(msg: unknown): void {
    void vscode.workspace
        .openTextDocument({ language: 'json', content: JSON.stringify(msg, null, 2) })
        .then((doc) => vscode.window.showTextDocument(doc));
}

/** Best-effort string label for the from/to fields (which may be string or {name}). */
export function labelOf(field: unknown): string | undefined {
    if (typeof field === 'string') return field;
    if (field && typeof field === 'object' && 'name' in field) {
        const name = (field as { name?: unknown }).name;
        if (typeof name === 'string') return name;
    }
    return undefined;
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
