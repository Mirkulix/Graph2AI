import * as vscode from 'vscode';
import { randomBytes } from 'crypto';
import { QlmsClient, QlmsConnectionReport } from './qlms-client';
import { QlmsInbox, openAsJson, labelOf } from './inbox';
import { startLanguageServer, QlangLspHandle } from './lsp';

const QLMS_CONFIG_SECTION = 'qlang.qlms';
const QLANG_CONFIG_SECTION = 'qlang';
const QLMS_STATUS_REFRESH_MS = 30_000;
const LOCAL_SEED_KEY = 'qlang.qlms.localSeedHex';

let lspHandle: QlangLspHandle | undefined;
let inboxHandle: QlmsInbox | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('QLANG extension activated');

    registerRunCommands(context);
    registerHandover(context);
    registerDashboard(context);
    registerStartServer(context);
    registerStatusBars(context);
    registerQlmsCheck(context);
    startLspIfEnabled(context);
    startInboxIfEnabled(context);

    vscode.window.showInformationMessage(
        'QLANG extension ready. Use Cmd+Shift+R to run; QLMS trust badge in the status bar.',
    );
}

export function deactivate() {
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

/** Registers `qlang.qlms.handover` — signs the active doc and dispatches it. */
function registerHandover(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.qlms.handover', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor to handover');
                return;
            }

            const agents = ['ceo', 'developer', 'researcher', 'guardian', 'strategist', 'artisan'];
            const target = await vscode.window.showQuickPick(agents, {
                placeHolder: 'Select target agent for handover',
            });
            if (!target) return;

            const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
            const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
            const authToken = cfg.get<string>('authToken') || undefined;
            const identity = cfg.get<string>('inbox.identity', 'vscode-assistant');
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
function startInboxIfEnabled(context: vscode.ExtensionContext): void {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    const enabled = cfg.get<boolean>('inbox.enabled', true);
    const identity = cfg.get<string>('inbox.identity', 'vscode-assistant');
    const baseUrl = cfg.get<string>('baseUrl', 'http://localhost:4646');
    const authToken = cfg.get<string>('authToken') || undefined;

    if (enabled) {
        inboxHandle = new QlmsInbox({ baseUrl, identity, authToken });
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
