# Coding Handover over QLMS

`coding-handover` is a minimal QLANG/QLMS bridge for two coding CLIs.

It is meant for flows like:

1. Agent A analyzes a repository and writes a `.qlms` handover.
2. Agent B reads the file, adds a structured reply, and writes a new `.qlms`.
3. Either side can inspect the conversation without replaying the full chat.

## Why this exists

Two coding CLIs usually exchange context through copied text or repeated prompts.

This tool keeps the handover structured:

- summary
- request
- next action
- relevant files
- evidence
- risks
- proposed changes
- tests
- notes

The payload is stored in `GraphMessage.inputs` as UTF-8 tensors, and the message itself is wrapped in a QLMS envelope.

## Example

Create a handover:

```powershell
cargo run --bin coding-handover --no-default-features --offline -- create `
  --from claude `
  --to codex `
  --phase analyze `
  --summary "Parser panic on empty token stream" `
  --request "Inspect the parser crash and decide the first patch" `
  --file crates/qlang-compile/src/parser.rs `
  --evidence "stacktrace points to parser.rs:118" `
  --risk "production panic in compile path" `
  --next-action "Codex should prepare the first patch plan" `
  --output handoff.qlms
```

Reply:

```powershell
cargo run --bin coding-handover --no-default-features --offline -- reply `
  --input handoff.qlms `
  --from codex `
  --to claude `
  --phase patch `
  --summary "Prepared patch plan" `
  --request "Parser crash narrowed down" `
  --change "Guard empty token stream before indexing" `
  --test "cargo test -p qlang-compile parser --offline" `
  --next-action "Claude should review and message the user" `
  --output handoff-reply.qlms
```

Inspect:

```powershell
cargo run --bin coding-handover --no-default-features --offline -- show --input handoff-reply.qlms
```

## Current scope

This is an MVP for structured context exchange, not a full autonomous multi-agent runtime.

It does not yet:

- stream live updates over the message bus
- merge conflicting edits
- diff repositories automatically
- execute patches directly from the handover

It does provide a stable place to move coding context out of free text and into a QLMS conversation file.
