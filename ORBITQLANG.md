# OrbitQlang — Cleanup Pass

> Status: `orbitqlang` branch.  This is the curated, honest cut of the
> A-2A-qlang repo prepared for outside readers (AAIF submission, code review,
> first-time contributors).  When this file disagrees with `README.md`, this
> file wins.  When this file disagrees with `QLANG-STATUS.md`, the status
> file wins.

## What changed in this branch

This branch took the working repo and made it ship-ready by separating
**what works on real data** from **research code that doesn't ship yet**.

### 1. Single Source of Truth for capabilities
- `QLANG-STATUS.md` is the canonical "what works" table.
- `README.md` and `docs/vault/*` cross-reference it.
- See `QLANG-STATUS.md` for current MNIST / CIFAR-10 numbers.

### 2. Experimental ML behind a feature gate
The following modules trained badly on real data (≤25 % on CIFAR-10,
~10 % on MNIST) and are now disabled by default:

`spiking`, `hebbian`, `hybrid_spiking`, `mamba`, `mamba_train`, `cifar10`,
`cifar10_features`, `vision_transformer`, `qlang_lm`, `gpu_mamba`,
`gpu_train`, `candle_train`, `lm_export`, `organism`.

Enable with:
```bash
cargo build --features experimental-ml
```
The corresponding `qo-server` routes (`/api/organism/*`,
`/api/training/gpu/*`) and the `gpu_train` binary are gated the same way.

### 3. Security hardening
- `qo-server` auth comparison is now constant-time
  (`qlang_core::crypto::ct_eq`) — no timing leak on bearer tokens.
- HMAC verification in `qlang-core::crypto` was already constant-time.
- New `replay_guard.rs` in `qlang-core` (nonce + timestamp window)
  for QLMS v1.1 opt-in.

### 4. Build / CI / deploy cleanup
- `install.sh` / `setup.sh` URLs point at the right repo
  (`Mirkulix/A-2A-qlang`), plus ASCII glyphs for portability.
- `Makefile` has `qo`, `docker-qo`, and `build-no-llvm` targets.
- `docker-compose.yml` marked legacy; `docker-compose.swarm.yml` is
  the supported multi-node path.
- `.dockerignore` updated for the new compose files.

### 5. CLAUDE.md
- Rewrote the project-wide CLAUDE.md from a stale template into
  project-specific guidance: file layout, build commands, security
  rules, honesty rules.

## Default build (what you get without any features)

```bash
# Stable runtime + supervisor + cockpit
cargo build --release --no-default-features
cargo build --release   # adds LLVM JIT (needs LLVM 18, see docs/BUILD.md)

# QO cockpit
QO_PORT=4646 ./target/release/qo
# → http://localhost:4646/supervisor

# QLANG CLI
./target/release/qlang train --data data/mnist --epochs 10 --output model.qlbg
```

The whole workspace must pass:
```bash
cargo check --workspace --no-default-features
```

## What still needs work (post-cleanup punch list)

- `docs/vault/` consolidation — many files predate the gating pass.
- `frontend/design-system/qo/MASTER.md` — Clarity light-first sync.
- CI workflow — `.github/workflows/ci.yml` doesn't exist yet; the
  `cargo check --no-default-features` line above is the gate that
  should run on every PR.
- AAIF submission packet — extract from `spec/` and the QLMS
  conformance suite.

## Where to look first

| You want to                       | Read                                |
|-----------------------------------|-------------------------------------|
| Understand what works today       | `QLANG-STATUS.md`                   |
| Understand the project intent     | `README.md`, `docs/vault/Vision.md` |
| Build & run locally               | `docs/BUILD.md`, `docs/QUICKSTART.md` |
| Wire format                       | `spec/QLMS.md`                      |
| Project-specific Claude Code rules| `CLAUDE.md`                         |
