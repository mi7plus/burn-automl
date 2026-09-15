# Changelog

All notable changes to this project are documented here. Per the plan's semver
policy (§22), breaking changes to public traits (`Sampler`, `Pruner`,
`Executor`, `Storage`, `TaskAdapter`, `Budget`) are called out under a
**Breaking** heading with a before/after snippet.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This line is pre-1.0: minor bumps (0.x → 0.(x+1)) may break public traits.

## [Unreleased]

### Added — v0.1 optimization foundation (in progress)

- `automl-core` crate: framework-agnostic optimization engine.
- Parameter distributions (`Distribution`): uniform/log/stepped floats and
  ints, categorical, boolean, with bound validation.
- Search spaces (`SearchSpace`) with conditional/hierarchical branches and
  declaration-order validation.
- Concrete parameter values (`ParamValue`, `ParamSet`) with typed accessors.
- Named metrics and directions (`NamedMetrics`, `Direction`, `Objective`).
- Trial model (`TrialRecord`, `TrialState`, `TrialHistory`, `TrialProgress`)
  matching the minimum persisted record (§19.2).
- `Sampler` trait with `RandomSampler` and `GridSampler`, both correct over
  conditional spaces (§5.1 correctness floor).
- `TpeSampler`: Tree-structured Parzen Estimator with per-branch KDE and a
  random fallback for under-observed conditional branches — the §5.1 mitigation
  built in from the start, so TPE degrades gracefully rather than erroring on
  complex conditional spaces.
- `EvolutionarySampler`: a real-coded genetic algorithm (tournament selection,
  uniform crossover, Gaussian mutation) — the evolutionary/CMA-style method the
  v1.0 DoD requires (§28). Handles conditional spaces, and is the strongest
  sampler on the benchmark suite (best mean on 4 of 5 functions).
- `QmcSampler`: a quasi-Monte-Carlo (Halton) sampler giving deterministic
  low-discrepancy coverage — evener space filling than random on
  low-dimensional continuous problems. Pure function of the trial index (no
  RNG), and conditional-space aware like the other samplers.
- `Pruner` trait with `NoPruner`, `MedianPruner`, `AshaPruner`
  (Asynchronous Successive Halving: geometric rungs, top-`1/eta` promotion), and
  `MultiObjectivePruner` (drops a trial dominated by a majority of peers, §17).
- `benchmarks` example: the §24.1 standard functions (Sphere, Rosenbrock,
  Ackley, Rastrigin, Himmelblau) with a seed-averaged Random-vs-TPE comparison
  reporting mean best and solve-rate.
- `Storage` trait with a thread-safe `InMemoryStorage` backend; reporting is
  idempotent by `(trial, step)` (§18.2).
- `SqliteStorage` (feature `sqlite`): first persistent backend, with a
  versioned `_schema_version` migrations table applied at open time and study
  resume across process restarts. `INSERT OR REPLACE` keeps reporting
  idempotent by `(trial, step)`. Bundled SQLite is compiled in, so it is gated
  behind the `sqlite` feature (extraction trigger #1, §21.1).
- `Study::resume`: reconnect to a persisted study id and continue optimizing
  without losing completed trials (§26 recovery gate).
- `Budget` trait with `TrialBudget`, `WallTimeBudget`, `EpochBudget`,
  `Unbounded` built-ins.
- `Objective` / `TaskAdapter` / `ReportSink` seam between the engine and
  concrete workloads.
- `Executor` trait with `SequentialExecutor` (deterministic, default) and
  `ThreadExecutor` (scoped thread pool) tiers, plus `ResourceSpec`. The study
  loop is now batch-oriented: it enqueues each proposal as a running trial
  before the next `suggest`, keeping suggestions diverse under parallelism while
  the sequential path stays byte-identical.
- `Study` and `StudyBuilder`: the deterministic, replayable optimization loop
  with pluggable executor tiers.
- Importance analysis v1 (`importance::importance`, `Study::importance`): a
  dependency-free main-effect (fANOVA-style) variance decomposition over a
  study's history, returned as a plain data structure with no UI dependency
  (§25, pulled forward to v0.2).
- `ParetoFront` (§17.1): the first-class multi-objective core object —
  direction-aware dominance, an archive `insert` that keeps the non-dominated
  set, and `hypervolume` (exact for 1-2 objectives, deterministic Monte-Carlo
  for 3+). Exposed via `Study::pareto_front()`; multi-objective studies return
  `None` from `best_trial()` and use the front instead.
- `sphere` example (Sphere benchmark from §24.1).

### Added — CLI and dashboard v0

- `automl-cli` crate with an `automl` binary: `automl list <db>` summarizes
  persisted studies; `automl dashboard <db>` renders a **self-contained HTML
  dashboard** (trial table, objective-over-time with best-so-far, per-parameter
  scatter) read-only against the SQLite store — no external assets, no new
  persisted state (§25, dashboard v0 pulled forward). `render_dashboard` is a
  pure, unit-tested function.

### Added — High-level tabular API

- `AutoClassifier` and `AutoRegressor` (in `automl-burn`): the one-call `Auto*`
  builders from §20/§6. Given in-memory tabular data they split off a validation
  set, search MLP hyperparameters (`lr`/`hidden_size`/`num_layers`/`dropout`),
  train with per-epoch reporting + median pruning, and return the best config
  plus the `Study` for inspection. They share one MLP training core (§6).
  `Mlp::forward_flat` generalizes the model to flat feature vectors.

### Added — Burn integration

- `automl-burn` crate: the Burn deep-learning execution adapter. A configurable
  MLP, a manual training loop that reports validation accuracy per epoch to the
  core `ReportSink` (feeding the pruner) and honors `should_stop`, and a
  `MnistMlpObjective` that turns MNIST training into an `automl_core::Objective`.
  Uses the CPU-only `ndarray` backend (no GPU/system deps). Includes a
  download-free synthetic-task test and an end-to-end `mnist_search` example.

### Not yet implemented (planned for v0.1 completion)

- Process executor tier (crash isolation) and the distributed tier (§18).
