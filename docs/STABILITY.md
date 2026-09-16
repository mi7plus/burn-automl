# API stability & semver policy

This document is the stability contract for `burn-automl`, per PRD §22. It states
what may change, what may not, and how any change is communicated.

## The 1.0 line is under strict semver

As of **v1.0.0**, `burn-automl` follows [Semantic Versioning](https://semver.org)
strictly:

- **No breaking change to any public API ships without a major version bump.**
  A breaking change is anything that can make previously-compiling downstream
  code fail to compile, or change its runtime contract: removing or renaming a
  public item, changing a function signature or trait method, removing an enum
  variant, or narrowing behavior a caller could rely on.
- **Minor releases (1.x)** add functionality in a backward-compatible way — new
  modules, new `Auto*` adapters, new samplers/pruners/backends, new optional
  methods with default implementations on existing traits.
- **Patch releases (1.x.y)** are backward-compatible bug fixes.

Before v1.0 the line was pre-1.0: minor bumps (0.x → 0.(x+1)) were permitted to
break public traits, as is normal for a 0.x line. That period is over.

## The public trait contract

These traits are the engine's extension points and the core of the stability
promise. Implementors and callers can rely on their signatures across a major
version:

| Trait | Role |
|-------|------|
| `Sampler` | proposes parameter sets from a search space + history |
| `Pruner` | early-stops running trials |
| `Executor` | runs trial batches (sequential / threaded / …) |
| `Storage` | persists studies, trials, metrics, artifacts, leases |
| `Budget` | bounds a study (trials, wall time, epochs, steps, …) |
| `TaskAdapter` / `Objective` / `ReportSink` | the workload seam |

New capabilities are added to these traits **only as methods with default
implementations**, so existing implementors keep compiling. The distributed
lease methods (`claim_trial`, `renew_lease`, `recover_orphans`) and the artifact
methods (`save_artifact`, …) were added exactly this way and are non-breaking.

## The `Study` API is the slowest-moving surface

The low-level `Study` / `StudyBuilder` API is the most stable surface in the
crate. Once stabilized it changes rarely and deliberately; a breaking change to
`Study` itself requires a design record, not just a code review (PRD §22, §436).

High-level `Auto*` builders (`AutoClassifier`, `AutoVision`, `AutoNas`,
`AutoPipeline`, …) are conveniences layered on top of `Study`; they may evolve
faster within the backward-compatibility rules above, and a custom `Objective`
over the `Study` API is always the stable fallback.

## Storage schema compatibility

Persisted studies remain loadable across versions. Schema migrations are
versioned integers applied at `Storage::open()` time and are **forward-compatible
within a major version**: a study created under an earlier `burn-automl` opens
under a later one (read-only if necessary). No destructive migration ships
without a documented downgrade path. See [MIGRATION.md](MIGRATION.md).

## How changes are communicated

- Every change is recorded in [`CHANGELOG.md`](../CHANGELOG.md).
- Any breaking change to a public trait is documented under a **Breaking**
  heading **with a before/after code snippet** — not merely described.
- Any deprecation ships with a documented migration path and a deprecation
  period before removal in the next major version.

## Determinism is part of the contract

Reproducibility is a supported guarantee, not an accident: given the same seed
and the same trial history, a study's proposals replay bit-identically. Anything
that samples derives a `ChaCha8Rng` from an explicit seed, and stable-ordered
`BTreeMap`s back `ParamSet` / `NamedMetrics`. A release will not silently break
replay of an existing seeded study.
