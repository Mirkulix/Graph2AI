"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
function activate(context) {
    console.log('QLANG extension activated');
    // Register Run command (Cmd+Shift+R)
    context.subscriptions.push(vscode.commands.registerCommand('qlang.run', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }
        const terminal = vscode.window.createTerminal('QLANG Run');
        terminal.show();
        terminal.sendText(`qlang-cli exec "${editor.document.fileName}"`);
    }));
    // Register REPL command
    context.subscriptions.push(vscode.commands.registerCommand('qlang.repl', () => {
        const terminal = vscode.window.createTerminal('QLANG REPL');
        terminal.show();
        terminal.sendText('qlang-cli repl');
    }));
    // Register AI Train command
    context.subscriptions.push(vscode.commands.registerCommand('qlang.aiTrain', () => {
        const terminal = vscode.window.createTerminal('QLANG AI');
        terminal.show();
        terminal.sendText('qlang-cli ai-train --quick');
    }));
    // Register Dashboard command
    context.subscriptions.push(vscode.commands.registerCommand('qlang.dashboard', () => {
        const terminal = vscode.window.createTerminal('QLANG Dashboard');
        terminal.show();
        terminal.sendText('qlang-cli web --port 8081');
    }));
    // Status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = '$(play) QLANG';
    statusBar.command = 'qlang.run';
    statusBar.tooltip = 'Run QLANG program (Cmd+Shift+R)';
    statusBar.show();
    context.subscriptions.push(statusBar);
    vscode.window.showInformationMessage('QLANG extension ready. Use Cmd+Shift+R to run.');
}
function deactivate() { }
//# sourceMappingURL=extension.js.map