# Contributing

Thanks for your interest in `burn-automl`. This is a short guide; for how the
codebase fits together, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first —
it explains the crate seams, the study loop, and how to add a domain adapter.

## Setup

Requires Rust **1.98+** (the pinned MSRV). Clone and build:

```bash
cargo build --workspace
```

## The gates (must stay green)

CI enforces all of these on every push and pull request; run them locally before
opening a PR:

```bash
cargo test --workspace                                  # fast (model-training tests skipped)
cargo test --workspace --features sqlite,slow-tests      # full suite, as CI runs it
cargo clippy --all-targets --features sqlite -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features sqlite
```

Clippy and rustfmt are release gates, not advisory. Model-training tests are
behind the `slow-tests` feature so the default `cargo test` stays fast.

## Conventions

- **Determinism is load-bearing.** Anything that samples derives a `ChaCha8Rng`
  from an explicit seed; keep replays bit-identical. `ParamSet`/`NamedMetrics`
  stay on `BTreeMap` for stable ordering.
- **Traits over enums** for extension points (samplers, pruners, storage,
  budgets, task adapters). New trait capabilities ship as default-implemented
  methods so existing implementors keep compiling (see
  [docs/STABILITY.md](docs/STABILITY.md)).
- **Errors are actionable** (`error::Error`): name the parameter or trial involved.
- Every public item has a doc comment; the core is `#![forbid(unsafe_code)]`.
- **ML threshold tests**: assert a margin above chance/baseline, not an exact
  value — Burn's ndarray backend reduces floats with rayon, so results vary
  slightly by platform. If one flakes, widen the margin; never chase an exact
  number.

## Public API and semver

The public trait surface and the `Study` API are under strict semver. A breaking
change needs a major version bump and a CHANGELOG entry under a **Breaking**
heading with a before/after snippet. When in doubt, add rather than change.

## Pull requests

Keep PRs focused. Update `CHANGELOG.md` under `[Unreleased]` (or the current
release section) describing the change. Make sure all gates above pass.
