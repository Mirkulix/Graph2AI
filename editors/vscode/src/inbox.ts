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
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);

        // Replies (Result envelopes) are what the user actually wants to
        // SEE — they sent a handover, they want the answer right now, no
        // extra click needed. Auto-open as side-by-side markdown unless
        // the user explicitly turned auto-action off via the new setting
        // qlang.qlms.inbox.replyAutoAction.
        if (msg.is_reply === true) {
            const autoAction = cfg.get<string>('inbox.replyAutoAction', 'side-by-side');
            if (autoAction === 'side-by-side') {
                void openSideBySide(msg);
                return;
            }
            if (autoAction === 'silent') {
                // record but don't surface
                return;
            }
            // 'prompt' falls through to the legacy 4-action popup
        }

        const autoRespondActive = cfg.get<boolean>('autoRespond.enabled', false) && msg.is_reply !== true;
        const label = autoRespondActive
            ? `Reply from ${fromLabel} (intent: ${intent}, auto-responded as ${this.opts.identity})`
            : `Reply from ${fromLabel} (intent: ${intent})`;
        void vscode.window
            .showInformationMessage(
                label,
                'Insert as Comment',
                'Open Side-by-Side',
                'Copy',
                'Open as JSON',
            )
            .then((action) => handleInboxAction(action, msg));
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
                requesterIdentity: this.opts.identity,
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

type InboxAction = 'Insert as Comment' | 'Open Side-by-Side' | 'Copy' | 'Open as JSON';

interface CommentStyle {
    kind: 'line' | 'block';
    /** Line-comment prefix (kind === 'line') OR opening delimiter (kind === 'block'). */
    prefix: string;
    /** Closing delimiter for block comments. */
    suffix?: string;
}

const LINE_LANGS: Record<string, string> = {
    rust: '// ',
    c: '// ',
    cpp: '// ',
    java: '// ',
    javascript: '// ',
    typescript: '// ',
    typescriptreact: '// ',
    javascriptreact: '// ',
    go: '// ',
    swift: '// ',
    kotlin: '// ',
    php: '// ',
    csharp: '// ',
    python: '# ',
    ruby: '# ',
    bash: '# ',
    shell: '# ',
    shellscript: '# ',
    yaml: '# ',
    toml: '# ',
    dockerfile: '# ',
    sql: '-- ',
};

const BLOCK_LANGS: Record<string, { open: string; close: string }> = {
    html: { open: '<!--', close: '-->' },
    xml: { open: '<!--', close: '-->' },
    vue: { open: '<!--', close: '-->' },
    svg: { open: '<!--', close: '-->' },
    css: { open: '/*', close: '*/' },
    scss: { open: '/*', close: '*/' },
    less: { open: '/*', close: '*/' },
};

function commentStyleFor(languageId: string): CommentStyle {
    const blk = BLOCK_LANGS[languageId];
    if (blk) return { kind: 'block', prefix: blk.open, suffix: blk.close };
    const line = LINE_LANGS[languageId];
    if (line) return { kind: 'line', prefix: line };
    return { kind: 'line', prefix: '// ' };
}

function intentLabel(intent: unknown): string {
    if (typeof intent === 'string') return intent;
    if (intent && typeof intent === 'object') {
        const keys = Object.keys(intent as object);
        if (keys.length > 0) return keys[0];
    }
    return '?';
}

function formatAsComment(style: CommentStyle, header: string, body: string): string {
    const lines = body.split(/\r?\n/);
    if (style.kind === 'line') {
        const prefixed = [header, ...lines].map((l) => `${style.prefix}${l}`);
        return prefixed.join('\n');
    }
    // Block comment: wrap header + body in a single block.
    const inner = [header, '', ...lines].join('\n');
    return `${style.prefix}\n${inner}\n${style.suffix}`;
}

async function insertAsComment(msg: InboxMessage): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    // No editor open → fall back to opening the reply in a side-by-side
    // markdown view instead of erroring out. This is what the user
    // actually meant when they clicked the action; "no active editor"
    // popups are useless and the inbox should never silently lose data.
    if (!editor) {
        void vscode.window.showInformationMessage(
            'No active editor — opening reply as markdown side-by-side instead.'
        );
        return openSideBySide(msg);
    }
    // Refuse to insert into source files of common code languages —
    // a QLMS reply as a comment in the middle of qo-server/lib.rs is
    // pollution. Fall back to side-by-side markdown for those.
    const languageId = editor.document.languageId;
    const NON_CODE_LANGS = new Set([
        'plaintext', 'markdown', 'log', 'restructuredtext', 'asciidoc',
        'tex', 'latex', 'bibtex', 'org',
    ]);
    if (!NON_CODE_LANGS.has(languageId)) {
        void vscode.window.showInformationMessage(
            `Active editor is a ${languageId} file — opening reply as markdown side-by-side to avoid polluting source code.`
        );
        return openSideBySide(msg);
    }
    const style = commentStyleFor(languageId);
    const fromName = labelOf(msg.from) ?? '?';
    const intent = intentLabel(msg.intent);
    const body = typeof msg.content === 'string' && msg.content.length > 0
        ? msg.content
        : JSON.stringify(msg, null, 2);
    const header = `🤖 reply from ${fromName} (${intent}):`;
    const text = formatAsComment(style, header, body);
    await editor.edit((eb) => eb.insert(editor.selection.start, text + '\n'));
}

async function openSideBySide(msg: InboxMessage): Promise<void> {
    const fromName = labelOf(msg.from) ?? '?';
    const intent = intentLabel(msg.intent);
    const ts = typeof msg['timestamp'] === 'string' || typeof msg['timestamp'] === 'number'
        ? String(msg['timestamp'])
        : new Date().toISOString();
    const id = typeof msg.id === 'number' ? String(msg.id) : '?';
    const body = typeof msg.content === 'string' && msg.content.length > 0
        ? msg.content
        : '```json\n' + JSON.stringify(msg, null, 2) + '\n```';
    const md = `# Reply from ${fromName}\n\n**intent**: ${intent}\n**timestamp**: ${ts}\n**id**: ${id}\n\n---\n\n${body}\n`;
    const doc = await vscode.workspace.openTextDocument({ language: 'markdown', content: md });
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
}

async function copyToClipboard(msg: InboxMessage): Promise<void> {
    const text = typeof msg.content === 'string' && msg.content.length > 0
        ? msg.content
        : JSON.stringify(msg, null, 2);
    await vscode.env.clipboard.writeText(text);
    vscode.window.setStatusBarMessage('Reply copied to clipboard', 2000);
}

async function handleInboxAction(action: string | undefined, msg: InboxMessage): Promise<void> {
    if (!action) return;
    switch (action as InboxAction) {
        case 'Insert as Comment':
            await insertAsComment(msg);
            break;
        case 'Open Side-by-Side':
            await openSideBySide(msg);
            break;
        case 'Copy':
            await copyToClipboard(msg);
            break;
        case 'Open as JSON':
            openAsJson(msg);
            break;
    }
}
