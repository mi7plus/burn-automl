# burn-automl

Rust-native AutoML and hyperparameter optimization, with [Burn](https://burn.dev)
as the flagship deep-learning execution adapter.

`burn-automl` optimizes **experiments, not tasks**. The core engine understands
parameter spaces, trials, observations, objectives, budgets, pruning decisions,
persistence and reproducibility — and nothing about regression, a Transformer,
or a U-Net. Domain-specific task layers translate an ML problem into those
generic primitives. Any workload that can produce named metrics can be
optimized.

> Status: **v0.1 in progress** — building the optimization foundation. See the
> [full implementation plan](docs/) (v0.1 → v1.0).

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
| `automl-burn` | Burn deep-learning execution adapter with high-level `Auto*` APIs: tabular (`AutoClassifier`/`AutoRegressor`), time series (`AutoForecaster`), sequences (`AutoSequence`, `AutoTransformer`), vision (`AutoVision`, `AutoSegmentation`, `AutoDetection`), audio (`AutoAudio`), video (`AutoVideo`), and neural architecture search (`AutoNas`). CPU (`ndarray`) backend by default. |
| `automl-tasks` | Framework-agnostic classical task adapters (no Burn): `AutoCluster` (K-means + silhouette) and `AutoAnomaly` (k-NN). |
| `automl-cli` | The `automl` binary: inspect studies (`automl list`) and render a read-only HTML dashboard (`automl dashboard`) from a SQLite store. |

Additional crates (`automl-tasks`, `automl-vision`, …) are introduced
release-by-release under the extraction criteria in the plan (§21).

## What v0.1 provides today

- **Search spaces** — uniform/log/stepped floats and ints, categorical,
  boolean, and conditional/hierarchical branches (a Transformer branch's
  parameters activate only once `model = "transformer"` is chosen).
- **Samplers** — `Random`, `Grid`, `TPE` (per-branch KDE + random fallback for
  under-observed conditional branches), and `Evolutionary` (a real-coded genetic
  algorithm), all correct over every conditional space.
- **Pruning** — median pruner with warmup and minimum-observation guards.
- **Storage** — thread-safe in-memory backend plus a persistent `SqliteStorage`
  (feature `sqlite`) with versioned migrations and study resume; reporting is
  idempotent by `(trial, step)`.
- **Budgets** — `Budget` as a trait: trials, wall time, epochs, unbounded.
- **Executors** — `SequentialExecutor` (deterministic default) and
  `ThreadExecutor` (scoped thread pool) tiers behind an `Executor` trait.
- **Studies** — the `Study` handle and a deterministic, replayable
  optimization loop with pluggable executors.
- **Importance analysis** — dependency-free main-effect (fANOVA-style)
  hyperparameter importance over a study's history, as a plain data structure.
- **Multi-objective** — `ParetoFront` with direction-aware dominance and
  hypervolume, via `Study::pareto_front()`.

## Development

```bash
cargo test                       # unit + doc tests (in-memory)
cargo test --features sqlite      # includes the persistent SQLite backend
cargo clippy --all-targets
cargo fmt --check
```

Clippy and rustfmt are release gates, not advisory.

## License

MIT — see [LICENSE](LICENSE).
