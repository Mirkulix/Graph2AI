# OrbitQO — Pitch

> This replaces an older pitch that sold an ML compiler (IGQK compression,
> LLVM JIT, edge deployment). Those features were removed from the codebase
> — see `QLANG-STATUS.md`. This pitch sells what actually runs and is tested.

## The problem, in one sentence

When several AI coding agents work on the same codebase, they share what they
know by copying conversation text — which is lossy, unverifiable, and gets
stale the moment someone changes a file.

Concretely, on any team using Claude Code, Cursor, or similar across more than
one session:

- **Duplicated work.** Session B re-discovers what session A already learned,
  because A's findings live in a transcript B never sees.
- **Confident wrong answers.** An agent asserts "auth uses bcrypt" because it
  read it in a prompt three steps ago — nobody checked, and it's now argon2.
- **No shared, durable memory.** Close the tab, lose the context. The next
  session starts cold.

## The product

**OrbitQO is a shared, verifiable memory for AI coding agents.**

Instead of copying text, agents exchange **signed, evidence-backed graph
deltas**. Each fact carries who observed it, when, and the exact file and line
that proves it. A later agent can *check* a claim instead of trusting it.

Three properties make it real rather than a wiki with extra steps:

1. **A proposal is never a fact.** An agent can suggest "auth uses bcrypt", but
   it stays a proposal — invisible to the next agent's context — until real
   evidence promotes it. This is enforced in the type system, not by
   convention.
2. **Signed, so provenance is proof.** Every delta is Ed25519-signed by a key
   the operator trusts. "Who said this" is cryptographic, not a string anyone
   can claim. Rotation and revocation are built in.
3. **Conflicts surface, never overwrite.** Two agents disagree? Both sides keep
   their evidence, and the conflict shows up in the cockpit for a human to
   settle. Nothing is silently lost.

## What's real today (verified, not claimed)

Every number here is reproducible from the repo:

- **1013 passing tests, 0 failures** across the workspace.
- The signature path is covered by **33 tests written as attacks** — foreign
  keys, producer impersonation, tampering, replay, backdating past a
  revocation. All rejected.
- **~4x smaller than JSON** for the same graph delta
  (`cargo run -p qo-knowledge --example orbitql_demo` prints the exact bytes).
- Verified end-to-end against a running server: unsigned submission → 401,
  replay → 409, tampered → 401, and the graph kept only the legitimate write.
- Works today with **DeepSeek, Groq, and local Ollama**; OpenAI/Anthropic/etc.
  through an OpenAI-compatible gateway. (We do not overstate this: there is one
  shared "cloud" slot, documented honestly in the code.)

## What it is not, yet

Selling this honestly means saying where the edges are:

- **Not multi-tenant.** Today it's one team, one instance. The SaaS path (below)
  is where that changes.
- **Agents still write the delta format by hand** (or are prompted to). The
  automatic "extract findings from a task" step is on the roadmap, not done.
- **The trust store says *who* may write, not *what about*.** Per-entity write
  policy is a refinement, not shipped.

We'd rather a customer hear this from us than discover it.

## Who pays, and why

**Engineering teams running AI coding agents at scale** — the ones who already
feel the pain of agents re-doing each other's work and asserting stale facts.

The value is not "another AI tool." It's:

- **Less duplicated agent work** → fewer tokens burned re-discovering the known.
- **Fewer confident-wrong changes** → the evidence rule blocks unverified
  claims from becoming "context."
- **Auditable AI decisions** → every change is signed and provenance-tracked,
  which the compliance-minded will pay for on its own.

## Business model: hosted SaaS

Core stays open (MIT). Money comes from **hosted OrbitQO for teams** — they
connect their agents, we run the instance, sync the graph, keep it durable.

| Tier | Who | Price (indicative) |
|------|-----|--------------------|
| **Free** | Solo, self-hosted | $0 — open source |
| **Team** | Up to ~10 seats, hosted | ~$49/seat/month |
| **Enterprise** | SSO, audit export, trust-store management, SLA | Custom |

Pricing is a hypothesis to validate, not a commitment — see the go-to-market
note.

## Why now

Every serious engineering org is adopting AI coding agents *this year*. The
multi-agent coordination problem is brand new and unowned. The teams feeling it
first are exactly the ones who buy infrastructure to fix it.

## The ask / next step

Not "build all of SaaS." First: **put the honest product in front of ten
teams** with a landing page and a live demo, and find out what they'd pay for.
Build the billing and multi-tenancy once someone has said yes.
