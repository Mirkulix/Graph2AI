# OrbitQO — from working code to first revenue

> The goal is money, and the fastest honest path to it is not "finish the SaaS
> platform." It's "get ten teams to say yes," then build only what those yeses
> require. This file is the plan for that, and what already exists to support
> it.

## The product being sold

Hosted, verifiable shared memory for AI coding agents. Teams connect their
agents; findings are exchanged as signed, evidence-backed graph deltas instead
of copied conversation text. See `docs/PITCH.md` for the full story and
`QLANG-STATUS.md` for exactly what runs.

## What already exists for selling it

Built and verified, not planned:

- **A landing page** (`commercial/landing.html`, published as an Artifact) that
  sells the *real* product honestly — including a "what it isn't yet" section,
  which is a trust asset, not a weakness.
- **Per-seat access.** `qlang keys issue|list|revoke` gives an admin real
  seats: one key per person, individually revocable, three roles
  (member/admin/viewer). Verified end to end — no key → 401, valid seat → 200,
  and an instance with issued seats binds to the network instead of loopback.
  This is the single feature that turns "one shared token" into "something you
  can charge per seat for."
- **The security story a buyer's reviewer will ask about.** Ed25519-signed
  deltas, a managed trust store, replay protection, fail-closed auth, an
  append-only audit trail. 210 passing tests across the two core crates; the
  signature path alone has 33 tests written as attacks.

## What is deliberately NOT built yet

Because no customer has asked for it. Building these before a yes is how
products die with a full feature list and no users.

- **Billing.** No Stripe, no invoicing. The first ten teams are onboarded by
  hand and likely free or founder-priced. `active_seats()` already returns the
  number an invoice would count — that's the only billing primitive worth
  having before there's someone to bill.
- **Multi-tenancy.** Today it's one instance per team. That's fine for hand
  onboarding: give each early team its own instance. A shared control plane is
  a real project, and it's the *second* thing to build, once recurring revenue
  justifies it.
- **Self-serve signup.** Early access is a mailto and a conversation. Automate
  it after the manual version has taught you what to automate.

## The sequence

1. **Put the landing page in front of ten teams.** Not "launch" — ten
   specific teams whose engineers run more than one AI coding session against
   the same repo. That's the qualifying question, and it's on the page.
2. **Onboard the interested ones by hand.** Their own instance, a few issued
   keys, a walkthrough of the conflict cockpit. Watch what confuses them and
   what they reach for.
3. **Find the price by asking.** The $49/seat on the page is a hypothesis. The
   early teams set it. Lock it in for them as the reward for being early.
4. **Build billing when a team says "yes, take my money."** Not before. Stripe
   plus the existing `active_seats()` count is a day, not a quarter.
5. **Build the shared control plane when running per-team instances by hand
   stops scaling.** That's a good problem, and it means there's revenue paying
   for the work.

## What to build next in the product (once there's a reason)

In rough priority, each tied to a customer signal:

- **Automatic finding extraction** — so agents don't hand-write deltas. This is
  the biggest single lift to "it just works," and the roadmap's next item.
- **A Stripe hook on `active_seats()`** — the moment someone wants to pay.
- **SSO + signed audit export** — the moment an enterprise buyer's security
  review asks for it. Both are Enterprise-tier revenue, not table stakes.
- **Shared multi-tenant control plane** — when hand-run instances stop scaling.

## The honest risk

This is infrastructure for a workflow (multiple coordinated AI coding agents)
that is real but still early. The bet is that it becomes common fast. If it
does, being the memory layer underneath it is valuable. The way to de-risk the
bet is step 1 — ten conversations — not more code.
