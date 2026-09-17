# burn-automl

Rust-native AutoML and hyperparameter optimization, with [Burn](https://burn.dev)
as the flagship deep-learning execution adapter.

`burn-automl` optimizes **experiments, not tasks**. The core engine understands
parameter spaces, trials, observations, objectives, budgets, pruning decisions,
persistence and reproducibility — and nothing about regression, a Transformer,
or a U-Net. Domain-specific task layers translate an ML problem into those
generic primitives. Any workload that can produce named metrics can be
optimized.

> Status: **v1.0** — the full HPO + NAS + multi-objective + distributed + pipeline
> story, under [strict semver](docs/STABILITY.md). See the
> [v1.0 definition-of-done](docs/v1.0-checklist.md) and the
> [migration guide](docs/MIGRATION.md).

## Quickstart

```rust
use automl_core::prelude::*;

// Minimize the 2-D Sphere function.
let space = SearchSpace::new()
    .add("x", Distribution::float(-5.0, 5.0))
    .add("y", Distribution::float(-5.0, 5.0));

let mut study = Study::builder(space)
    .minimize("loss")
    .seed(42)
    .build()?;

study.optimize_n(&|p: &ParamSet, _r: &mut dyn ReportSink| {
    let (x, y) = (p.float("x")?, p.float("y")?);
    Ok(NamedMetrics::single("loss", x * x + y * y))
}, 500)?;

let best = study.best_trial()?.unwrap();
println!("best loss = {:?}", best.final_value("loss"));
# Ok::<(), automl_core::Error>(())
```

Run the bundled examples:

```bash
cargo run -p automl-core --example sphere
```

```bash
cargo run -p automl-burn --release --example mnist_search
```

The `mnist_search` example searches MLP hyperparameters to maximize MNIST
validation accuracy (TPE sampling + median pruning). It downloads MNIST on first
run and trains on the CPU.

## Workspace

| Crate | Role |
|-------|------|
| `automl-core` | Framework-agnostic engine: distributions, search spaces, samplers, pruners, storage, budgets, executors, studies. No `burn` or `automl-*` dependencies. |
| `automl-burn` | Burn deep-learning execution adapter with high-level `Auto*` APIs: tabular (`AutoClassifier`/`AutoRegressor`), time series (`AutoForecaster`), sequences (`AutoSequence`, `AutoTransformer`, `AutoSeq2Seq`), text (`AutoText`), vision (`AutoVision`, `AutoSegmentation`, `AutoDetection`), audio (`AutoAudio`), video (`AutoVideo`), neural architecture search (`AutoNas`), and generative (`AutoAutoencoder`). CPU (`ndarray`) backend by default; GPU via the `wgpu`, `cuda`, or `metal` feature. |
| `automl-tasks` | Framework-agnostic classical task adapters (no Burn): `AutoCluster` (K-means + silhouette), `AutoAnomaly` (k-NN), `AutoRl` (tabular Q-learning with robust noisy-return aggregation), and `AutoPipeline` (preprocessing + model pipeline search). |
| `automl-cli` | The `automl` binary: inspect studies (`automl list`) and render a read-only HTML dashboard (`automl dashboard`, with learning-curve overlays and utilization) from a SQLite store. |

Further crates are introduced release-by-release under the extraction criteria in
the plan (§21); the core stays free of `burn` and any `automl-*` dependency.

## What burn-automl provides

- **Search spaces** — uniform/log/stepped floats and ints, categorical,
  boolean, and conditional/hierarchical branches (a Transformer branch's
  parameters activate only once `model = "transformer"` is chosen).
- **Samplers** — `Random`, `Grid`, `TPE` (per-branch KDE + random fallback for
  under-observed conditional branches), `Evolutionary` (a real-coded genetic
  algorithm), and `QMC` (Halton), all correct over every conditional space.
- **Pruning & multi-fidelity** — median (with warmup, min-observation, and a
  robust noisy-curve mode), ASHA successive-halving, and multi-objective pruners,
  plus `Hyperband`/BOHB, which allocate budget itself as a search dimension.
- **Warm-starting** — `WarmStartSampler` seeds a new study with the best configs
  of a prior one, transferring known-good regions.
- **Storage** — thread-safe in-memory backend, a persistent `SqliteStorage`
  (feature `sqlite`), and a shared `PostgresStorage` (feature `postgres`), all with
  versioned migrations, study resume, and artifacts; reporting is idempotent by
  `(trial, step)`.
- **Budgets** — `Budget` as a trait: trials, wall time, epochs, steps, unbounded.
- **Executors & distribution** — `SequentialExecutor`, `ThreadExecutor`, a
  crash-isolated `ProcessPool` (subprocess workers), and a lease-backed
  distributed worker protocol (claim / heartbeat / orphan recovery) with
  exactly-once completion.
- **Studies** — the `Study` handle and a deterministic, replayable optimization
  loop with pluggable executors, samplers and pruners.
- **Multi-objective** — `ParetoFront` with direction-aware dominance and
  hypervolume via `Study::pareto_front()`, plus the `Nsga2Sampler` (non-dominated
  sorting + crowding distance) to evolve the whole trade-off front.
- **Robust aggregation** — replicated evaluation with median/trimmed-mean
  aggregates for noisy objectives (RL, GANs).
- **NAS & pipelines** — architecture graphs (`MacroSpace`) and full pipelines
  (`PipelineSpace`) encoded as ordinary conditional search spaces, plus
  multimodal fusion with pre-scheduling constraint validation.
- **Observability** — dependency-free hyperparameter importance and a read-only
  HTML dashboard with learning-curve overlays and utilization.

## Stability

`burn-automl` 1.0 is under **strict semver**: no breaking change to a public API
without a major version bump. See [docs/STABILITY.md](docs/STABILITY.md) for the
trait contract and [docs/MIGRATION.md](docs/MIGRATION.md) for schema
compatibility.

## Development

```bash
cargo test                              # fast: unit tests, model-training tests skipped
cargo test --features sqlite             # includes the persistent SQLite backend
cargo test --features slow-tests         # also runs the model-training tests (CI does)
cargo clippy --all-targets
cargo fmt --check
```

Model-training tests are gated behind the `slow-tests` feature so the default
`cargo test` stays fast; CI runs `--features sqlite,slow-tests`.

```bash
cargo bench -p automl-core          # sampler throughput (criterion)
cargo run  -p automl-core --example nsga2   # and hyperband, warm_start, benchmarks
```

Clippy and rustfmt are release gates, not advisory; CI enforces them, an MSRV
(1.98) build, and a warning-free doc build on every push.

## Documentation

- [Architecture overview](docs/ARCHITECTURE.md) — crate seams, the study loop, how to add an adapter
- [Stability & semver policy](docs/STABILITY.md) and the [migration guide](docs/MIGRATION.md)
- [v1.0 definition-of-done map](docs/v1.0-checklist.md)
- [Sampler benchmark report](docs/BENCHMARKS.md) — Random vs TPE vs multivariate TPE vs Evolutionary
- [Publishing checklist](docs/PUBLISHING.md)

## License

MIT — see [LICENSE](LICENSE).
