import * as vscode from 'vscode';
import * as os from 'os';
import { randomBytes } from 'crypto';
import { QlmsClient, QlmsConnectionReport } from './qlms-client';
import { QlmsInbox, openAsJson, labelOf } from './inbox';
import { startLanguageServer, QlangLspHandle } from './lsp';

const QLMS_CONFIG_SECTION = 'qlang.qlms';
const QLANG_CONFIG_SECTION = 'qlang';
const QLMS_STATUS_REFRESH_MS = 30_000;
const PRESENCE_HEARTBEAT_MS = 25_000;
const LOCAL_SEED_KEY = 'qlang.qlms.localSeedHex';
const LOCAL_IDENTITY_KEY = 'qlang.qlms.identity';

let lspHandle: QlangLspHandle | undefined;
let inboxHandle: QlmsInbox | undefined;
let presenceCleanup: (() => Promise<void>) | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('QLANG extension activated');

    const identity = resolveIdentity(context);
    console.log(`[qlang] bus identity: ${identity}`);

    registerRunCommands(context);
    registerHandover(context, identity);
    registerDashboard(context);
    registerStartServer(context);
    registerStatusBars(context);
    registerQlmsCheck(context);
    startLspIfEnabled(context);
    startInboxIfEnabled(context, identity);
    startPresence(context, identity);

    vscode.window.showInformationMessage(
        `QLANG extension ready as '${identity}'. Use Cmd+Shift+R to run; QLMS trust badge in the status bar.`,
    );
}

export async function deactivate() {
    try {
        inboxHandle?.dispose();
    } catch {
        // ignore
    }
    try {
        lspHandle?.dispose();
    } catch {
        // ignore
    }
    try {
        if (presenceCleanup) {
            await presenceCleanup();
        }
    } catch {
        // ignore
    }
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/**
 * Resolves this IDE instance's bus identity. Resolution order:
 *   1. `qlang.qlms.inbox.identity` setting (user override)
 *   2. Persisted globalState value (one stable identity per install)
 *   3. Freshly generated `<ide>-<host>-<6-char-uuid>` (then persisted)
 */
export function resolveIdentity(context: vscode.ExtensionContext): string {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    const fromSetting = cfg.get<string>('inbox.identity', '').trim();
    if (fromSetting) return fromSetting;

    const stored = context.globalState.get<string>(LOCAL_IDENTITY_KEY);
    if (stored && stored.length > 0) return stored;

    const ide = sanitizeSlug((vscode.env.appName ?? 'code').toLowerCase()) || 'code';
    const host = sanitizeSlug(os.hostname() || 'host').slice(0, 12) || 'host';
    const uuid = randomBytes(3).toString('hex');
    const identity = `${ide}-${host}-${uuid}`;
    void context.globalState.update(LOCAL_IDENTITY_KEY, identity);
    return identity;
}

function sanitizeSlug(input: string): string {
    // Keep ide names like "visual studio code" -> "visual-studio-code", drop edge dashes.
    return input
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

// ---------------------------------------------------------------------------
// Presence (register / heartbeat / deregister)
// ---------------------------------------------------------------------------

interface PresenceEntry {
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

async function registerPresence(
    baseUrl: string,
    identity: string,
    extras: {
        ideName: string;
        host: string;
        capabilities: string[];
        llmProvider?: string;
        llmModel?: string;
    },
    authToken?: string,
): Promise<void> {
    const url = `${baseUrl.replace(/\/$/, '')}/api/presence/register`;
    const body = {
        identity,
        ide_name: extras.ideName,
        host: extras.host,
        capabilities: extras.capabilities,
        llm_provider: extras.llmProvider,
        llm_model: extras.llmModel,
    };
    await presencePost(url, body, authToken);
}

async function heartbeatPresence(
    baseUrl: string,
    identity: string,
    authToken?: string,
): Promise<void> {
    const url = `${baseUrl.replace(/\/$/, '')}/api/presence/heartbeat/${encodeURIComponent(identity)}`;
    await presencePost(url, undefined, authToken);
}

async function deregisterPresence(
    baseUrl: string,
    identity: string,
    authToken?: string,
): Promise<void> {
    const url = `${baseUrl.replace(/\/$/, '')}/api/presence/${encodeURIComponent(identity)}`;
    const headers: Record<string, string> = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const resp = await fetch(url, { method: 'DELETE', headers });
    if (!resp.ok) {
        throw new Error(`presence DELETE -> HTTP ${resp.status}`);
    }
}

async function listPresence(
    baseUrl: string,
    authToken?: string,
): Promise<PresenceEntry[]> {
    const url = `${baseUrl.replace(/\/$/, '')}/api/presence`;
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const resp = await fetch(url, { method: 'GET', headers });
    if (!resp.ok) {
        throw new Error(`presence GET -> HTTP ${resp.status}`);
    }
    const json = (await resp.json()) as PresenceEntry[];
    return Array.isArray(json) ? json : [];
}

async function presencePost(
    url: string,
    body: unknown,
    authToken?: string,
): Promise<void> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const resp = await fetch(url, {
        method: 'POST',
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!resp.ok) {
        throw new Error(`presence POST ${url} -> HTTP ${resp.status}`);
    }
}

/** Fires off a single registration + recurring heartbeat; tolerates a missing endpoint. */
function startPresence(context: vscode.ExtensionContext, identity: string): void {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
    const authToken = cfg.get<string>('authToken') || undefined;
    const autoRespondEnabled = cfg.get<boolean>('autoRespond.enabled', false);
    const llmProvider = autoRespondEnabled
        ? cfg.get<string>('autoRespond.providerType', 'deepseek')
        : undefined;
    const llmModel = autoRespondEnabled
        ? cfg.get<string>('autoRespond.model', '')
        : undefined;
    const capabilities: string[] = ['execute'];
    if (autoRespondEnabled) capabilities.push('auto-respond');

    const ideName = vscode.env.appName ?? 'Code';
    const host = os.hostname();

    let presenceAvailable = true;

    const doRegister = async () => {
        try {
            await registerPresence(
                baseUrl,
                identity,
                { ideName, host, capabilities, llmProvider, llmModel },
                authToken,
            );
        } catch (err) {
            presenceAvailable = false;
            console.warn(
                `[qlang.presence] register failed (older qo without /api/presence?): ${
                    err instanceof Error ? err.message : String(err)
                }`,
            );
        }
    };

    void doRegister();

    const handle = setInterval(async () => {
        if (!presenceAvailable) return;
        try {
            await heartbeatPresence(baseUrl, identity, authToken);
        } catch (err) {
            // Heartbeat failed: try one re-register, then back off.
            console.warn(
                `[qlang.presence] heartbeat failed: ${
                    err instanceof Error ? err.message : String(err)
                }`,
            );
            try {
                await registerPresence(
                    baseUrl,
                    identity,
                    { ideName, host, capabilities, llmProvider, llmModel },
                    authToken,
                );
            } catch {
                presenceAvailable = false;
            }
        }
    }, PRESENCE_HEARTBEAT_MS);

    context.subscriptions.push({ dispose: () => clearInterval(handle) });

    presenceCleanup = async () => {
        clearInterval(handle);
        if (!presenceAvailable) return;
        try {
            await deregisterPresence(baseUrl, identity, authToken);
        } catch (err) {
            console.warn(
                `[qlang.presence] deregister failed: ${
                    err instanceof Error ? err.message : String(err)
                }`,
            );
        }
    };
}

// ---------------------------------------------------------------------------
// Run / REPL
// ---------------------------------------------------------------------------

/** Registers `qlang.run` and `qlang.repl` terminal commands. */
function registerRunCommands(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.run', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }
            const terminal = vscode.window.createTerminal('QLANG Run');
            terminal.show();
            terminal.sendText(`qlang-cli exec "${editor.document.fileName}"`);
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.repl', () => {
            const terminal = vscode.window.createTerminal('QLANG REPL');
            terminal.show();
            terminal.sendText('qlang-cli repl');
        }),
    );
}

/** Registers the `qlang.qlms.startServer` terminal command. */
function registerStartServer(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.qlms.startServer', () => {
            const terminal = vscode.window.createTerminal('OrbitQLang Backend');
            terminal.show();
            terminal.sendText('cargo run --bin qo -- --offline');
            vscode.window.showInformationMessage('Starting OrbitQLang Backend...');
        }),
    );
}

/** Opens the QO supervisor cockpit in the system browser. */
function registerDashboard(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.dashboard', async () => {
            const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
            const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646').replace(/\/$/, '');
            await vscode.env.openExternal(vscode.Uri.parse(`${baseUrl}/supervisor`));
        }),
    );
}

// ---------------------------------------------------------------------------
// Status bars
// ---------------------------------------------------------------------------

/** Adds the run, handover, and QLMS trust badge entries to the status bar. */
function registerStatusBars(context: vscode.ExtensionContext): void {
    const runStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    runStatus.text = '$(play) QLANG';
    runStatus.command = 'qlang.run';
    runStatus.tooltip = 'Run QLANG program (Cmd+Shift+R)';
    runStatus.show();
    context.subscriptions.push(runStatus);

    const bridgeStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 98);
    bridgeStatus.text = '$(export) Handover';
    bridgeStatus.command = 'qlang.qlms.handover';
    bridgeStatus.tooltip = 'Handover current graph to OrbitQLang Agent Bus';
    bridgeStatus.show();
    context.subscriptions.push(bridgeStatus);
}

// ---------------------------------------------------------------------------
// Handover (signed + delivered)
// ---------------------------------------------------------------------------

interface HandoverPick extends vscode.QuickPickItem {
    target: string;
}

/** Registers `qlang.qlms.handover` — signs the active doc and dispatches it. */
function registerHandover(context: vscode.ExtensionContext, identity: string): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.qlms.handover', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor to handover');
                return;
            }

            const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
            const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
            const authToken = cfg.get<string>('authToken') || undefined;

            const target = await pickHandoverTarget(baseUrl, authToken, identity);
            if (!target) return;

            const client = new QlmsClient({ baseUrl, authToken });

            try {
                const content = editor.document.getText();
                // The server's `Graph` struct demands {id, version, nodes, edges, constraints, metadata}.
                // We build a valid empty-graph envelope and stuff the source into metadata.
                // If the file IS a valid serialized graph, parse it and use it directly.
                let graph: Record<string, unknown>;
                let parsed: Record<string, unknown> | null = null;
                try {
                    const maybe = JSON.parse(content);
                    if (maybe && typeof maybe === 'object' && Array.isArray(maybe.nodes)) {
                        parsed = maybe as Record<string, unknown>;
                    }
                } catch {
                    // not JSON, fall through to wrapper
                }
                if (parsed) {
                    graph = parsed;
                } else {
                    graph = {
                        id: `vscode-handover-${Date.now()}`,
                        version: '1.0',
                        nodes: [],
                        edges: [],
                        constraints: [],
                        metadata: {
                            source: 'vscode-extension',
                            filename: editor.document.fileName,
                            language: editor.document.languageId,
                            content,
                        },
                    };
                }

                const seedHex = resolveSeedHex(context);
                const msg = QlmsClient.createMessage({
                    id: Math.floor(Math.random() * 1_000_000),
                    from: identity,
                    to: target,
                    graph,
                    intent: 'Execute',
                });

                const reply = await client.reply([msg], seedHex);
                const delivered = await client.deliver(reply.frame);

                if (delivered.signature_verified) {
                    vscode.window.showInformationMessage(
                        `Signed handover delivered to '${target}' (${reply.size_bytes} B, ${delivered.msg_count} msg). Verified.`,
                    );
                } else if (delivered.signed) {
                    vscode.window.showErrorMessage(
                        `Handover to '${target}' delivered but signature NOT verified (signed=${delivered.signed}, verified=${delivered.signature_verified}).`,
                    );
                } else {
                    vscode.window.showWarningMessage(
                        `Handover to '${target}' delivered unsigned (${delivered.msg_count} msg). Set qlang.qlms.seedHex or QO_SEED_HEX to sign.`,
                    );
                }
            } catch (err) {
                vscode.window.showErrorMessage(
                    `Handover failed: ${err instanceof Error ? err.message : String(err)}`,
                );
            }
        }),
    );
}

/** Builds a QuickPick of server agents + connected IDEs (excluding self) and returns the chosen target identity. */
async function pickHandoverTarget(
    baseUrl: string,
    authToken: string | undefined,
    selfIdentity: string,
): Promise<string | undefined> {
    const agents = ['ceo', 'developer', 'researcher', 'guardian', 'strategist', 'artisan'];
    const items: HandoverPick[] = [];

    items.push({
        label: 'Server agents',
        kind: vscode.QuickPickItemKind.Separator,
        target: '',
    });
    for (const a of agents) {
        items.push({ label: a, description: 'server agent', target: a });
    }

    let presence: PresenceEntry[] = [];
    try {
        presence = await listPresence(baseUrl, authToken);
    } catch (err) {
        console.warn(
            `[qlang.handover] presence list failed: ${
                err instanceof Error ? err.message : String(err)
            }`,
        );
    }

    const others = presence.filter((p) => p.identity && p.identity !== selfIdentity);
    if (others.length > 0) {
        items.push({
            label: 'Connected IDEs',
            kind: vscode.QuickPickItemKind.Separator,
            target: '',
        });
        for (const p of others) {
            const ide = p.ide_name ?? 'IDE';
            const host = p.host ?? '?';
            const caps = (p.capabilities ?? []).join(',');
            const desc = `${ide} · ${host}${caps ? ` · ${caps}` : ''}`;
            items.push({
                label: p.identity,
                description: desc,
                target: p.identity,
            });
        }
    }

    const pick = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select target (server agent or connected IDE) for handover',
    });
    if (!pick || !pick.target) return undefined;
    return pick.target;
}

/**
 * Resolves the signing seed: setting -> env -> persisted workspace seed
 * (auto-generated and stored on first use).
 */
function resolveSeedHex(context: vscode.ExtensionContext): string {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    const fromSetting = cfg.get<string>('seedHex', '').trim();
    if (fromSetting) return fromSetting;
    const fromEnv = process.env?.QO_SEED_HEX?.trim();
    if (fromEnv) return fromEnv;
    const stored = context.workspaceState.get<string>(LOCAL_SEED_KEY);
    if (stored) return stored;
    const generated = randomBytes(32).toString('hex');
    void context.workspaceState.update(LOCAL_SEED_KEY, generated);
    return generated;
}

// ---------------------------------------------------------------------------
// QLMS connectivity badge
// ---------------------------------------------------------------------------

/** Registers `qlang.qlms.check` plus the periodic 30 s ping that drives the badge. */
function registerQlmsCheck(context: vscode.ExtensionContext): void {
    const qlmsStatus = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        99,
    );
    qlmsStatus.command = 'qlang.qlms.check';
    qlmsStatus.text = '$(sync~spin) QLMS …';
    qlmsStatus.tooltip = 'Checking QLMS connection …';
    qlmsStatus.show();
    context.subscriptions.push(qlmsStatus);

    const runCheck = async () => {
        const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
        const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
        const authToken = cfg.get<string>('authToken') || undefined;
        const client = new QlmsClient({ baseUrl, authToken });
        const report = await client.checkConnection();
        applyStatus(qlmsStatus, report, baseUrl);
    };

    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.qlms.check', async () => {
            await runCheck();
            const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
            const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
            const pick = await vscode.window.showQuickPick(
                [
                    { label: '$(refresh) Re-check connection', action: 'recheck' },
                    { label: '$(settings) Change QO base URL', action: 'settings' },
                ],
                { placeHolder: `QLMS — ${baseUrl}` },
            );
            if (pick?.action === 'recheck') {
                await runCheck();
            } else if (pick?.action === 'settings') {
                await vscode.commands.executeCommand(
                    'workbench.action.openSettings',
                    QLMS_CONFIG_SECTION,
                );
            }
        }),
    );

    void runCheck();
    const handle = setInterval(() => void runCheck(), QLMS_STATUS_REFRESH_MS);
    context.subscriptions.push({ dispose: () => clearInterval(handle) });
}

function applyStatus(
    item: vscode.StatusBarItem,
    report: QlmsConnectionReport,
    baseUrl: string,
): void {
    if (!report.ok) {
        item.text = '$(warning) QLMS offline';
        item.tooltip = `Cannot reach ${baseUrl}\n${report.error ?? 'unknown error'}`;
        item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        return;
    }
    if (report.signatureVerified) {
        item.text = '$(shield) QLMS signed';
        item.tooltip = `Connected to ${baseUrl}\nSigned frame round-trip verified (${report.sizeBytes} B)`;
        item.backgroundColor = undefined;
        return;
    }
    item.text = '$(unverified) QLMS unverified';
    item.tooltip = `Connected to ${baseUrl}\nReply frame round-trip succeeded but signature was NOT verified`;
    item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
}

// ---------------------------------------------------------------------------
// LSP
// ---------------------------------------------------------------------------

/** Spawns the QLANG language server when `qlang.lsp.enabled` is true. */
function startLspIfEnabled(context: vscode.ExtensionContext): void {
    const cfg = vscode.workspace.getConfiguration(QLANG_CONFIG_SECTION);
    if (!cfg.get<boolean>('lsp.enabled', true)) return;
    const lspPath = cfg.get<string>('lsp.path', 'qlang-cli');
    lspHandle = startLanguageServer(lspPath);
    if (lspHandle) {
        context.subscriptions.push({ dispose: () => lspHandle?.dispose() });
    }
}

// ---------------------------------------------------------------------------
// Inbox
// ---------------------------------------------------------------------------

/** Subscribes to /api/messages/stream and registers `qlang.qlms.showInbox`. */
function startInboxIfEnabled(context: vscode.ExtensionContext, identity: string): void {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    const enabled = cfg.get<boolean>('inbox.enabled', true);
    const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
    const authToken = cfg.get<string>('authToken') || undefined;
    const seedHex = resolveSeedHex(context);

    if (enabled) {
        inboxHandle = new QlmsInbox({ baseUrl, identity, authToken, seedHex });
        inboxHandle.start();
        context.subscriptions.push({ dispose: () => inboxHandle?.dispose() });
    }

    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.qlms.showInbox', async () => {
            if (!inboxHandle) {
                vscode.window.showInformationMessage(
                    'QLMS inbox is disabled. Enable qlang.qlms.inbox.enabled to receive replies.',
                );
                return;
            }
            const items = inboxHandle.snapshot();
            if (items.length === 0) {
                vscode.window.showInformationMessage(
                    `Inbox is empty for '${identity}'. Replies addressed to this IDE will appear here.`,
                );
                return;
            }
            const picks = items.map((msg, idx) => {
                const from = labelOf(msg.from) ?? '?';
                const to = labelOf(msg.to) ?? '?';
                const intent =
                    typeof msg.intent === 'string'
                        ? msg.intent
                        : msg.intent
                            ? JSON.stringify(msg.intent)
                            : '?';
                return {
                    label: `${from} → ${to} · ${intent}`,
                    description: `#${items.length - idx}`,
                    msg,
                };
            });
            const pick = await vscode.window.showQuickPick(picks, {
                placeHolder: `QLMS inbox (${items.length} message${items.length === 1 ? '' : 's'})`,
            });
            if (pick) openAsJson(pick.msg);
        }),
    );
}
