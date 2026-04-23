// Routing config loader + matcher for the auto-trigger system.
//
// Reads `.qlang/routing.json` from the workspace and validates it against
// the Rule shape declared below. Pure functions only — no vscode.window
// side-effects so the loader can be unit-tested in isolation.

import * as vscode from 'vscode';

export type TriggerKind = 'on_save' | 'on_change' | 'on_open';

export interface RuleWhen {
    languageId?: string;
    pathMatches?: string;
    contentMatches?: string;
    minLines?: number;
    maxLines?: number;
}

export interface Rule {
    id: string;
    trigger: TriggerKind;
    when?: RuleWhen;
    send_to: string | string[];
    prompt: string;
    intent?: string;
    enabled?: boolean;
}

export interface RoutingConfig {
    version: number;
    rules: Rule[];
}

export type ValidateResult =
    | { ok: true; config: RoutingConfig }
    | { ok: false; errors: string[] };

const TRIGGER_KINDS: ReadonlySet<TriggerKind> = new Set([
    'on_save',
    'on_change',
    'on_open',
]);

/**
 * Reads + parses the routing config at `uri`. Returns `null` on any
 * filesystem error or validation failure (with reasons logged to the
 * extension console — never throws).
 */
export async function loadRoutingConfig(
    uri: vscode.Uri,
): Promise<RoutingConfig | null> {
    let bytes: Uint8Array;
    try {
        bytes = await vscode.workspace.fs.readFile(uri);
    } catch (err) {
        console.warn(
            `[qlang.triggers] cannot read routing config at ${uri.fsPath}: ${
                err instanceof Error ? err.message : String(err)
            }`,
        );
        return null;
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(new TextDecoder('utf-8').decode(bytes));
    } catch (err) {
        console.warn(
            `[qlang.triggers] routing config is not valid JSON: ${
                err instanceof Error ? err.message : String(err)
            }`,
        );
        return null;
    }
    const result = validateConfig(parsed);
    if (!result.ok) {
        console.warn(
            `[qlang.triggers] routing config rejected:\n  - ${result.errors.join('\n  - ')}`,
        );
        return null;
    }
    return result.config;
}

/**
 * Validates a parsed JSON value against the {@link RoutingConfig} shape.
 * Each rule is checked independently so a single bad rule still yields
 * targeted error messages.
 */
export function validateConfig(parsed: unknown): ValidateResult {
    const errors: string[] = [];
    if (!isRecord(parsed)) {
        return { ok: false, errors: ['top-level value must be a JSON object'] };
    }
    const version = parsed.version;
    if (typeof version !== 'number' || !Number.isFinite(version)) {
        errors.push('version: missing or not a finite number');
    }
    const rulesRaw = parsed.rules;
    if (!Array.isArray(rulesRaw)) {
        errors.push('rules: missing or not an array');
        return { ok: false, errors };
    }
    const rules: Rule[] = [];
    rulesRaw.forEach((raw, idx) => {
        const ruleErrors = validateRule(raw, idx);
        if (ruleErrors.length > 0) {
            errors.push(...ruleErrors);
        } else {
            rules.push(raw as Rule);
        }
    });
    if (errors.length > 0) {
        return { ok: false, errors };
    }
    return {
        ok: true,
        config: { version: version as number, rules },
    };
}

function validateRule(raw: unknown, idx: number): string[] {
    const errs: string[] = [];
    const at = `rules[${idx}]`;
    if (!isRecord(raw)) {
        return [`${at}: must be an object`];
    }
    if (typeof raw.id !== 'string' || raw.id.length === 0) {
        errs.push(`${at}.id: must be a non-empty string`);
    }
    if (typeof raw.trigger !== 'string' || !TRIGGER_KINDS.has(raw.trigger as TriggerKind)) {
        errs.push(`${at}.trigger: must be one of on_save | on_change | on_open`);
    }
    if (raw.when !== undefined && !isRecord(raw.when)) {
        errs.push(`${at}.when: must be an object when present`);
    } else if (isRecord(raw.when)) {
        const w = raw.when;
        if (w.languageId !== undefined && typeof w.languageId !== 'string') {
            errs.push(`${at}.when.languageId: must be a string`);
        }
        if (w.pathMatches !== undefined && typeof w.pathMatches !== 'string') {
            errs.push(`${at}.when.pathMatches: must be a string`);
        }
        if (w.contentMatches !== undefined && typeof w.contentMatches !== 'string') {
            errs.push(`${at}.when.contentMatches: must be a string`);
        }
        if (w.minLines !== undefined && typeof w.minLines !== 'number') {
            errs.push(`${at}.when.minLines: must be a number`);
        }
        if (w.maxLines !== undefined && typeof w.maxLines !== 'number') {
            errs.push(`${at}.when.maxLines: must be a number`);
        }
    }
    if (typeof raw.send_to !== 'string') {
        if (
            !Array.isArray(raw.send_to) ||
            !raw.send_to.every((t) => typeof t === 'string' && t.length > 0)
        ) {
            errs.push(`${at}.send_to: must be a non-empty string or array of non-empty strings`);
        }
    } else if ((raw.send_to as string).length === 0) {
        errs.push(`${at}.send_to: must not be empty`);
    }
    if (typeof raw.prompt !== 'string' || raw.prompt.length === 0) {
        errs.push(`${at}.prompt: must be a non-empty string`);
    }
    if (raw.intent !== undefined && typeof raw.intent !== 'string') {
        errs.push(`${at}.intent: must be a string when present`);
    }
    if (raw.enabled !== undefined && typeof raw.enabled !== 'boolean') {
        errs.push(`${at}.enabled: must be a boolean when present`);
    }
    return errs;
}

/**
 * AND-combine all configured `when` predicates. Returns true when no
 * `when` block is supplied (i.e. the rule fires unconditionally).
 *
 * Regex compilation errors are treated as "no match" so a malformed
 * pattern can't crash the trigger loop.
 */
export function matches(when: Rule['when'], doc: vscode.TextDocument): boolean {
    if (!when) return true;
    if (when.languageId !== undefined && doc.languageId !== when.languageId) {
        return false;
    }
    if (when.pathMatches !== undefined) {
        const re = safeRegex(when.pathMatches);
        if (!re || !re.test(doc.fileName)) return false;
    }
    if (when.contentMatches !== undefined) {
        const re = safeRegex(when.contentMatches);
        if (!re || !re.test(doc.getText())) return false;
    }
    if (when.minLines !== undefined && doc.lineCount < when.minLines) {
        return false;
    }
    if (when.maxLines !== undefined && doc.lineCount > when.maxLines) {
        return false;
    }
    return true;
}

function safeRegex(pattern: string): RegExp | undefined {
    try {
        return new RegExp(pattern);
    } catch (err) {
        console.warn(
            `[qlang.triggers] invalid regex /${pattern}/: ${
                err instanceof Error ? err.message : String(err)
            }`,
        );
        return undefined;
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
