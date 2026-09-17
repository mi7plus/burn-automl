# Architecture

An orientation for contributors. For the stability contract see
[STABILITY.md](STABILITY.md); for the release scope, [v1.0-checklist.md](v1.0-checklist.md).

## The one invariant

**The core optimizes experiments, not tasks.** `automl-core` understands
parameter spaces, trials, observations, objectives, budgets, pruning decisions,
persistence and reproducibility — and *nothing* about regression, a Transformer,
or a U-Net. Anything that can produce named metrics can be optimized. Domain
layers translate an ML problem into those generic primitives. Keep this line
sharp: if a change would teach `automl-core` about a specific model or dataset,
it belongs in an adapter crate instead.

## Crates

```
automl-core     framework-agnostic engine — no `burn`, no `automl-*` deps
  ├─ automl-tasks   classical adapters (no deep learning): clustering, anomaly, RL, pipeline
  ├─ automl-burn    the Burn deep-learning adapter: the Auto* APIs, CPU/GPU backends
  └─ automl-cli     the `automl` binary: inspect studies, dashboard, worker, importance
```

`automl-tasks`, `automl-burn` and `automl-cli` depend on `automl-core`; they do
not depend on each other. New crates are extracted only on the plan's explicit
triggers (§21.1) — default to fewer crates.

## The trait seams

Extension points are **traits**, so each is replaceable without touching the loop:

| Trait | Responsibility | Built-ins |
|-------|----------------|-----------|
| `Sampler` | propose the next `ParamSet` from the space + history | Random, Grid, TPE (+ multivariate), Evolutionary, QMC, NSGA-II |
| `Pruner` | early-stop a running trial by its curve | None, Median (+ robust window), ASHA, MultiObjective |
| `Executor` | run a batch of trial jobs | Sequential, Thread; `ProcessPool` for crash isolation |
| `Storage` | persist studies/trials/metrics/artifacts/leases | InMemory, SQLite; PostgreSQL (separate `automl-postgres` crate) |
| `Budget` | bound a study | Trial, WallTime, Epoch, Step, Unbounded |
| `Objective` / `ReportSink` | the workload seam | user closures, `TaskAdapter`s |

New capabilities are added to these traits **only as default-implemented
methods** (see [STABILITY.md](STABILITY.md)), so existing implementors keep
compiling.

## How a study runs

`Study::optimize` is the loop (`study.rs`):

1. **Ask** — call `Sampler::suggest(space, history)` for a batch of proposals.
   Each proposal is enqueued as a *running* trial before the next `suggest`, so
   successive calls see a growing history and stay diverse under parallelism.
2. **Run** — the `Executor` evaluates each proposal's `Objective`. During
   evaluation the objective reports intermediate metrics to a `ReportSink`, which
   persists them and consults the `Pruner`; a pruned trial stops early.
3. **Tell** — completed trials are observed (`Sampler::on_trial_complete`), their
   consumed resources tallied against the `Budget`, and the loop repeats until the
   budget is exhausted.

`Storage` is the single source of truth. The distributed protocol (`distributed.rs`)
builds on that: workers claim trials by compare-and-swap, heartbeat a lease, and
complete a trial only while they still hold it — so a worker declared dead and
reassigned never double-completes.

## Determinism

Reproducibility is load-bearing. Anything that samples derives a `ChaCha8Rng`
from an explicit seed; `ParamSet`/`NamedMetrics` are `BTreeMap`-backed for stable
ordering. A replay with the same seed and history is bit-identical, and this is
covered by tests, not left to chance.

## Encoding structure as search spaces

The recurring trick that keeps the core simple: NAS (`nas.rs`), pipelines
(`pipeline.rs`) and multimodal fusion (`fusion.rs`) all **encode their structure
as ordinary conditional `SearchSpace`s** and decode a sampled `ParamSet` back
into a concrete plan. So the sampler, pruner and storage never learn that
architecture or pipeline search is happening — they optimize parameters as
always. Follow this pattern when adding a new kind of structured search.

## Adding a domain adapter

A high-level `Auto*` API is a thin builder over `Study` plus a pre-built
`SearchSpace` and an `Objective`:

1. Build a `SearchSpace` describing the hyperparameters.
2. Write an objective closure `|params, sink| -> Result<NamedMetrics>` that
   materializes the config, trains/evaluates it, reports intermediate metrics to
   `sink`, and returns the final metrics.
3. `Study::builder(space).maximize("metric").sampler(..).pruner(..).build()?`
   then `optimize_n`.

See `automl-burn/src/vision.rs` (a self-contained CNN adapter) or
`automl-tasks/src/pipeline.rs` (a framework-agnostic one) as templates. Shared
Burn helpers live in `automl-burn/src/common.rs`; the MNIST example wiring is in
`automl-burn/src/mnist.rs`.

## Gates

```bash
cargo test --workspace --features sqlite,slow-tests   # model-training tests behind the feature
cargo clippy --all-targets --features sqlite -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features sqlite
```

Clippy and rustfmt are release gates, not advisory. CI enforces all of the above
plus an MSRV (1.98) build and a wgpu compile-check.
