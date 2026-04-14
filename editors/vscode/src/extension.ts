import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('QLANG extension activated');

    // Register Run command (Cmd+Shift+R)
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
        })
    );

    // Register REPL command
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.repl', () => {
            const terminal = vscode.window.createTerminal('QLANG REPL');
            terminal.show();
            terminal.sendText('qlang-cli repl');
        })
    );

    // Register AI Train command
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.aiTrain', () => {
            const terminal = vscode.window.createTerminal('QLANG AI');
            terminal.show();
            terminal.sendText('qlang-cli ai-train --quick');
        })
    );

    // Register Dashboard command
    context.subscriptions.push(
        vscode.commands.registerCommand('qlang.dashboard', () => {
            const terminal = vscode.window.createTerminal('QLANG Dashboard');
            terminal.show();
            terminal.sendText('qlang-cli web --port 8081');
        })
    );

    // Status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = '$(play) QLANG';
    statusBar.command = 'qlang.run';
    statusBar.tooltip = 'Run QLANG program (Cmd+Shift+R)';
    statusBar.show();
    context.subscriptions.push(statusBar);

    vscode.window.showInformationMessage('QLANG extension ready. Use Cmd+Shift+R to run.');
}

export function deactivate() {}
