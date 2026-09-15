# CLAUDE.md — burn-automl

Guidance for working in this repository.

## What this is

A Rust-native AutoML / hyperparameter-optimization platform being built from a
detailed implementation plan (v0.1 → v1.0). The design invariant: the core
**optimizes experiments, not tasks** — `automl-core` knows about parameter
spaces, trials, objectives, budgets, pruning, storage and reproducibility, and
nothing about specific ML models. Domain layers translate ML problems into
these primitives via the `TaskAdapter` trait.

## Layout

Cargo workspace. `crates/automl-core` is the framework-agnostic engine and must
never depend on `burn` or any `automl-*` crate. New crates are extracted only on
the explicit triggers in the plan (§21.1) — default to fewer crates.

Core modules (all in `automl-core`): `distribution`, `space`, `param`,
`metrics`, `trial`, `budget`, `sampler` (+ `tpe`, `evolution`), `pruner`,
`storage` (+ feature-gated `sqlite`), `executor`, `objective`, `pareto`,
`importance`, `study`. The public `prelude` re-exports the common types.
Other crates: `automl-burn` (Burn adapter), `automl-cli` (`automl` binary +
dashboard).

## Conventions

- **Determinism is load-bearing.** Anything that samples derives a
  `ChaCha8Rng` from an explicit seed; replay with the same seed + history must
  be bit-identical. Keep `ParamSet`/`NamedMetrics` on `BTreeMap` for stable
  ordering.
- **Traits over enums** for extension points (samplers, pruners, storage,
  budgets, task adapters) — the plan requires replaceability via public traits.
- **Errors are actionable** (`error::Error`): name the parameter/trial involved.
- Every public item has a doc comment (`#![warn(missing_docs)]`).
- `#![forbid(unsafe_code)]` in the core.

## Gates (must stay green)

```bash
cargo test
cargo clippy --all-targets    # clippy is a gate, not advisory
cargo fmt --check
```

## Implementation order (from the plan §29)

Done: core domain model → Random/Grid samplers → in-memory storage + replay →
median pruning → budgets → sequential `Study` loop.

Next: TPE (flat) → conditional TPE + random fallback → SQLite + resume +
migrations → thread/process executors → `automl-burn` adapter + MNIST.

## Codebase search

A `codebase-memory-mcp` knowledge-graph server is wired via `.mcp.json`. Prefer
its graph tools (`search_graph`, `trace_path`, `query_graph`, `get_code_snippet`)
over grep for structural queries; re-index after large changes.
