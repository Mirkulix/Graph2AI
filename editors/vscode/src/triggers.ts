// Auto-trigger system: dispatches signed QLMS handovers in response to
// VS Code editor events (save / change / open) based on a workspace-local
// `.qlang/routing.json`. Sender side of the auto-respond pair.
//
// All side-effects are gated behind `qlang.qlms.triggers.enabled` (off by
// default) so installing the extension never silently fans out traffic.

import * as vscode from 'vscode';
import {
    loadRoutingConfig,
    matches,
    Rule,
    RoutingConfig,
    TriggerKind,
} from './routing-config';
import { GraphMessage, QlmsClient } from './qlms-client';

const QLMS_CONFIG_SECTION = 'qlang.qlms';
const ROUTING_GLOB = '**/.qlang/routing.json';

export interface TriggersHandle {
    dispose(): void;
}

interface InternalState {
    config: RoutingConfig | null;
    debounceMs: number;
    maxConcurrent: number;
    inflight: number;
    debounceTimers: Map<string, NodeJS.Timeout>;
}

/**
 * Wires VS Code editor events to QLMS handovers per `.qlang/routing.json`.
 * Returns `undefined` when triggers are disabled, the workspace has no
 * routing config, or the config fails to validate.
 */
export function startTriggers(
    context: vscode.ExtensionContext,
    identity: string,
    baseUrl: string,
    authToken: string | undefined,
    seedHex: string,
): TriggersHandle | undefined {
    const cfg = vscode.workspace.getConfiguration(QLMS_CONFIG_SECTION);
    if (!cfg.get<boolean>('triggers.enabled', false)) {
        console.log('[qlang.triggers] triggers disabled (qlang.qlms.triggers.enabled=false)');
        return undefined;
    }

    const state: InternalState = {
        config: null,
        debounceMs: cfg.get<number>('triggers.debounceMs', 2000),
        maxConcurrent: cfg.get<number>('triggers.maxConcurrent', 3),
        inflight: 0,
        debounceTimers: new Map(),
    };

    const disposables: vscode.Disposable[] = [];
    let cancelled = false;

    void initialLoad(state).then((found) => {
        if (cancelled) return;
        if (!found) {
            console.log(
                '[qlang.triggers] no .qlang/routing.json in workspace — triggers idle. Copy editors/vscode/src/example-routing.json to enable.',
            );
        }
    });

    // Hot-reload on config edits so users don't have to restart VS Code.
    const watcher = vscode.workspace.createFileSystemWatcher(ROUTING_GLOB);
    disposables.push(watcher);
    const reload = (uri: vscode.Uri | undefined) => {
        void (async () => {
            if (uri) {
                state.config = await loadRoutingConfig(uri);
                console.log(
                    `[qlang.triggers] reloaded routing config from ${uri.fsPath} (${state.config?.rules.length ?? 0} rules)`,
                );
            } else {
                state.config = null;
                console.log('[qlang.triggers] routing config removed; triggers idle');
            }
        })();
    };
    disposables.push(watcher.onDidCreate(reload));
    disposables.push(watcher.onDidChange(reload));
    disposables.push(watcher.onDidDelete(() => reload(undefined)));

    const client = new QlmsClient({ baseUrl, authToken });

    disposables.push(
        vscode.workspace.onDidSaveTextDocument((doc) => {
            void dispatchForEvent(state, client, identity, seedHex, doc, 'on_save');
        }),
    );

    disposables.push(
        vscode.workspace.onDidOpenTextDocument((doc) => {
            void dispatchForEvent(state, client, identity, seedHex, doc, 'on_open');
        }),
    );

    disposables.push(
        vscode.workspace.onDidChangeTextDocument((evt) => {
            const key = evt.document.uri.toString();
            const existing = state.debounceTimers.get(key);
            if (existing) clearTimeout(existing);
            const timer = setTimeout(() => {
                state.debounceTimers.delete(key);
                void dispatchForEvent(
                    state,
                    client,
                    identity,
                    seedHex,
                    evt.document,
                    'on_change',
                );
            }, state.debounceMs);
            state.debounceTimers.set(key, timer);
        }),
    );

    return {
        dispose: () => {
            cancelled = true;
            for (const t of state.debounceTimers.values()) clearTimeout(t);
            state.debounceTimers.clear();
            for (const d of disposables) {
                try {
                    d.dispose();
                } catch {
                    // ignore
                }
            }
        },
    };
}

async function initialLoad(state: InternalState): Promise<boolean> {
    const found = await vscode.workspace.findFiles('.qlang/routing.json', null, 1);
    if (found.length === 0) return false;
    state.config = await loadRoutingConfig(found[0]);
    if (state.config) {
        console.log(
            `[qlang.triggers] loaded routing config from ${found[0].fsPath} (${state.config.rules.length} rules)`,
        );
    }
    return true;
}

async function dispatchForEvent(
    state: InternalState,
    client: QlmsClient,
    identity: string,
    seedHex: string,
    doc: vscode.TextDocument,
    kind: TriggerKind,
): Promise<void> {
    const config = state.config;
    if (!config) return;
    // Skip the routing config itself + scratch documents to prevent feedback loops.
    if (doc.uri.scheme !== 'file') return;
    if (doc.fileName.replace(/\\/g, '/').endsWith('/.qlang/routing.json')) return;

    for (const rule of config.rules) {
        if (rule.enabled === false) continue;
        if (rule.trigger !== kind) continue;
        if (!matches(rule.when, doc)) continue;

        const targets = Array.isArray(rule.send_to) ? rule.send_to : [rule.send_to];
        for (const target of targets) {
            if (state.inflight >= state.maxConcurrent) {
                console.warn(
                    `[qlang.triggers] dropped ${rule.id} -> ${target}: maxConcurrent=${state.maxConcurrent} reached`,
                );
                continue;
            }
            state.inflight += 1;
            void dispatchOne(client, identity, seedHex, rule, target, doc, kind)
                .catch((err) => {
                    console.warn(
                        `[qlang.triggers] dispatch ${rule.id} -> ${target} failed: ${
                            err instanceof Error ? err.message : String(err)
                        }`,
                    );
                })
                .finally(() => {
                    state.inflight = Math.max(0, state.inflight - 1);
                });
        }
    }
}

async function dispatchOne(
    client: QlmsClient,
    identity: string,
    seedHex: string,
    rule: Rule,
    target: string,
    doc: vscode.TextDocument,
    kind: TriggerKind,
): Promise<void> {
    const text = doc.getText();
    const intent = rule.intent ?? 'Execute';
    const graph: Record<string, unknown> = {
        id: `auto-${rule.id}-${Date.now()}`,
        version: '1.0',
        nodes: [],
        edges: [],
        constraints: [],
        metadata: {
            source: 'auto-trigger',
            auto_triggered: 'true',
            trigger_kind: kind,
            rule_id: rule.id,
            filename: doc.fileName,
            language: doc.languageId,
            content: `${rule.prompt}\n\n---\n${text}`,
        },
    };

    const msg: GraphMessage = QlmsClient.createMessage({
        id: Math.floor(Math.random() * 1_000_000),
        from: identity,
        to: target,
        graph,
        intent,
    });

    const built = await client.reply([msg], seedHex);
    await client.deliver(built.frame);

    // Statusbar only — popups would be obnoxious for a passive system.
    vscode.window.setStatusBarMessage(
        `auto-trigger: ${rule.id} → ${target}`,
        4000,
    );
}
