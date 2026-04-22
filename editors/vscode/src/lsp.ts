// QLANG Language Server bootstrap. Spawns the configured `qlang-cli lsp`
// binary over stdio and wires it into vscode-languageclient.

import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

/** Started LSP client wrapper. `dispose()` stops the underlying process. */
export interface QlangLspHandle {
    client: LanguageClient;
    dispose(): void;
}

/**
 * Starts the QLANG language server. Returns `undefined` (and logs a
 * warning) when the configured binary cannot be launched, so a missing
 * `qlang-cli` never crashes the extension.
 */
export function startLanguageServer(lspPath: string): QlangLspHandle | undefined {
    const serverOptions: ServerOptions = {
        run: { command: lspPath, args: ['lsp'], transport: TransportKind.stdio },
        debug: { command: lspPath, args: ['lsp'], transport: TransportKind.stdio },
    };
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'qlang' }],
    };
    try {
        const client = new LanguageClient(
            'qlangLsp',
            'QLANG Language Server',
            serverOptions,
            clientOptions,
        );
        void client.start();
        return {
            client,
            dispose: () => {
                void client.stop().catch(() => undefined);
            },
        };
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`[qlang.lsp] failed to start '${lspPath} lsp': ${msg}`);
        void vscode.window.showWarningMessage(
            `QLANG LSP could not start ('${lspPath}' not on PATH?). Editor features will be limited.`,
        );
        return undefined;
    }
}
