import * as vscode from 'vscode';
import { QlmsClient, QlmsConnectionReport } from './qlms-client';

const QLMS_CONFIG_SECTION = 'qlang.qlms';
const QLMS_STATUS_REFRESH_MS = 30_000;

export function activate(context: vscode.ExtensionContext) {
    console.log('QLANG extension activated');

    // ---------------------------------------------------------------------
    // Existing runner commands
    // ---------------------------------------------------------------------
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

    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.aiTrain', () => {
            const terminal = vscode.window.createTerminal('QLANG AI');
            terminal.show();
            terminal.sendText('qlang-cli ai-train --quick');
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.dashboard', () => {
            const terminal = vscode.window.createTerminal('QLANG Dashboard');
            terminal.show();
            terminal.sendText('qlang-cli web --port 8081');
        }),
    );

    // Run status-bar entry (existing behaviour).
    const runStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    runStatus.text = '$(play) QLANG';
    runStatus.command = 'qlang.run';
    runStatus.tooltip = 'Run QLANG program (Cmd+Shift+R)';
    runStatus.show();
    context.subscriptions.push(runStatus);

    // ---------------------------------------------------------------------
    // QLMS connectivity — PRD Task 5.1
    // ---------------------------------------------------------------------
    //
    // Status-bar badge reflecting the trust state of the configured QO
    // server. Clicking it opens quick-pick with reconnect / diagnose
    // actions. All HTTP traffic goes through the `/qlms/v1.1/*` routes
    // landed in Phase 2, so the status tracks signature verification
    // live rather than faking a green tick.

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
        qlmsStatus.text = '$(sync~spin) QLMS …';
        qlmsStatus.tooltip = `Probing ${baseUrl} …`;
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
                {
                    placeHolder: `QLMS — ${baseUrl}`,
                },
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

    // Fire once at startup, then every 30s.
    void runCheck();
    const handle = setInterval(() => {
        void runCheck();
    }, QLMS_STATUS_REFRESH_MS);
    context.subscriptions.push({ dispose: () => clearInterval(handle) });

    vscode.window.showInformationMessage(
        'QLANG extension ready. Use Cmd+Shift+R to run; QLMS trust badge in the status bar.',
    );
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
        item.text = '$(shield) QLMS ✓ signed';
        item.tooltip = `Connected to ${baseUrl}\nSigned frame round-trip verified (${report.sizeBytes} B)`;
        item.backgroundColor = undefined;
        return;
    }
    item.text = '$(unverified) QLMS unverified';
    item.tooltip = `Connected to ${baseUrl}\nReply frame round-trip succeeded but signature was NOT verified`;
    item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
}

export function deactivate() {}
