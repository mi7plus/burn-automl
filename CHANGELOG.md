# Changelog

All notable changes to this project are documented here. Per the plan's semver
policy (§22), breaking changes to public traits (`Sampler`, `Pruner`,
`Executor`, `Storage`, `TaskAdapter`, `Budget`) are called out under a
**Breaking** heading with a before/after snippet.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This line is pre-1.0: minor bumps (0.x → 0.(x+1)) may break public traits.

## [Unreleased]

### Added — examples for every use case

- A runnable, self-contained example per use case (indexed in
  [docs/EXAMPLES.md](docs/EXAMPLES.md)): engine capabilities (`multimodal`,
  `robust_aggregation`, `distributed`), classical adapters (`clustering`,
  `anomaly`, `reinforcement_learning`, `pipeline`), and the deep-learning
  adapters (`tabular`, `time_series`, `sequence`, `transformer`, `seq2seq`,
  `text`, `vision`, `segmentation`, `detection`, `audio`, `video`, `generative`),
  alongside the existing `sphere`/`benchmarks`/`hyperband`/`nsga2`/`warm_start`
  and the real-MNIST `mnist_search`/`mnist_nas`. The synthetic ones need no
  download and run in seconds.

### Added — supply chain & project hygiene

- **`cargo audit` CI job** scanning `Cargo.lock` against the RustSec advisory
  database (informational; currently only two transitive *unmaintained* notices,
  no vulnerabilities), plus **Dependabot** for cargo and GitHub Actions updates.
- **`SECURITY.md`** (private vulnerability reporting) and **`CONTRIBUTING.md`**
  (setup, gates, conventions, pointing at the architecture doc).
- **`#![forbid(unsafe_code)]`** added to `automl-cli`, so all four crates now
  forbid unsafe.

## [1.4.0] — 2026-09-17

Additive release: a second audit-pass — hardening, tooling, and dependency
currency. No breaking changes.

### Added

- **Property-based sampler tests** (`proptest`): every sampler (Random / Grid /
  TPE / Evolutionary / QMC) is checked to return a *valid assignment* over
  arbitrary conditional search spaces, hardening the §5.1 correctness floor.
- **`automl importance <db>`** CLI subcommand: prints per-parameter hyperparameter
  importance with a text bar, reusing the tested importance analysis.
- **`docs/ARCHITECTURE.md`**: contributor orientation — crate seams, the study
  loop, determinism, and how to add a domain adapter.

### Changed

- **NaN-safe float sorts**: every `partial_cmp().unwrap()` in library code is now
  `unwrap_or(Equal)`, so a NaN objective value (e.g. a diverged training run) can
  never panic a sort.
- **Dependencies**: `thiserror` 1 → 2, `rusqlite` 0.31 → 0.40 (no code changes).

### Deliberately held

- **`rand` stays at 0.8** (not 0.9/0.10). Burn 0.21 pulls rand 0.8 regardless, so
  an upgrade would not shrink the dependency tree — and rand's post-0.8
  range-sampling change would alter our seeded RNG output and break the
  determinism guarantee. **Burn stays at 0.21** (the latest *stable*; 0.22 is a
  pre-release). Both are documented decisions, revisited when Burn moves.

### Breaking

None.

## [1.3.0] — 2026-09-17

Additive release: a repo-audit pass — one new capability, quality-of-life, and
refactoring. No breaking changes.

### Added

- **NSGA-II multi-objective sampler** (`automl-core::nsga2`): fills the gap where
  multi-objective studies sampled effectively at random. Non-dominated sorting +
  crowding distance + elite survival selection + crowded tournament + local
  Gaussian mutation; beats random search on ZDT1. Fully worked `nsga2` example.
- **Runnable examples** for the recent optimization features: `hyperband`
  (multi-fidelity / BOHB), `warm_start` (transfer), `nsga2` (multi-objective).
- **Serialization round-trip tests** pinning the persistence contract for
  `ParamSet` / `SearchSpace` / `NamedMetrics` / `TrialRecord`.
- **Criterion sampler benchmark** (`cargo bench -p automl-core`), CI compile-checked.
- **CUDA/Metal parity** noted, and a **GPU (wgpu) compile-check CI job**.

### Changed

- Model-training tests are gated behind a `slow-tests` feature, so `cargo test`
  runs in seconds (CI runs `--features sqlite,slow-tests`).
- De-duplicated the `split`/`image_tensor` helpers into an `automl-burn::common`
  module, and moved the MNIST machinery out of the crate root into `mnist` (public
  API unchanged via re-exports). The crate root dropped from 433 to ~150 lines.

### Breaking

None.

## [1.2.0] — 2026-09-17

Additive release (no breaking changes): post-1.0 optimization enhancements.

### Added

- **Multi-fidelity Hyperband / BOHB** (`automl-core::multifidelity`): `Hyperband`
  allocates budget as a search dimension via successive-halving brackets —
  evaluate many configs cheaply, keep the top `1/eta`, multiply resource by `eta`,
  spend full-fidelity training only on survivors. The objective is fidelity-aware;
  `optimize()` takes any `Sampler`, so pairing it with `TpeSampler` gives BOHB
  (model-based multi-fidelity).
- **Study warm-starting & transfer** (`automl-core::warmstart`): `WarmStartSampler`
  wraps any sampler and replays a queue of seed configurations before delegating;
  `best_configs` extracts the top-N configs from a prior study's history to
  transfer. A study seeded with the optimum finds it on the first trial.
- **Learned text embeddings** (`AutoText::learned_embedding`): trains a token
  embedding table end-to-end (masked mean-pooling over the sequence) instead of
  feature hashing, capturing token similarity a fixed front-end cannot.
- **CUDA and Metal GPU backends** (`automl-burn` features `cuda`, `metal`): swap
  `TrainBackend` to the corresponding autodiff backend, alongside the existing
  `wgpu` feature. Off by default; enable at most one.
- **Benchmark**: the sampler benchmark example and report now include multivariate
  TPE, which is the strongest sampler overall (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

### Breaking

None. New modules, a new `AutoText` option, and new feature flags, all over
unchanged public traits and the `Study` API.

## [1.1.0] — 2026-09-17

Additive release (no breaking changes): completes the previously-deferred
functionality and the remaining coverage/hardening items, plus CI.

### Added

- **CI** — a GitHub Actions workflow enforcing the release gates on push/PR:
  `fmt --check`, `clippy --all-targets -D warnings`, workspace tests with
  `sqlite`, an MSRV (1.98.0) build, and a warning-free doc build.
- **Process executor tier** (`automl-core::process`, §18): `ProcessPool` runs
  trials in crash-isolated OS subprocesses over the shared lease-backed store —
  a worker crash only loses that process; its trial's lease expires and is
  recovered and retried. Adds an `automl worker` subcommand. Integration test
  drives real subprocesses through a crash-and-recover, exactly-once drain.
- **PostgreSQL backend** (`automl-core::postgres`, feature `postgres`, §18.4): a
  shared-server `Storage` mirroring the SQLite backend — same versioned
  migrations, idempotent reporting, and the lease protocol as a `FOR UPDATE SKIP
  LOCKED` compare-and-swap. Integration test runs against `TEST_POSTGRES_URL`.
- **NLP text classification** (`automl-burn::text`, §12): `AutoText` tokenizes and
  feature-hashes documents into sequences the recurrent classifier consumes.
- **Sequence-to-sequence** (`automl-burn::seq2seq`, §9): `AutoSeq2Seq`, an LSTM
  encoder–decoder with teacher-forced training and free-running greedy decode.
- **Multivariate TPE** (`TpeSampler::multivariate`, §5): joint, good-trial-anchored
  candidate sampling that follows inter-parameter correlations — fixes the
  classic independent-per-dim stall on coupled objectives (verified on Rosenbrock).
- **GPU execution** (`automl-burn` feature `wgpu`, §18): swaps `TrainBackend` to
  autodiff-over-WGPU so every `Auto*` adapter runs on GPU with no code change.
- **Larger-dataset example** (`mnist_nas`): architecture search on real MNIST.
- **Publish metadata**: shared keywords/categories/homepage across crates.

### Breaking

None. Every change is additive: new modules, a new sampler option, new feature
flags (`postgres`, `wgpu`), and new `Auto*` adapters, all layered over unchanged
public traits and the `Study` API.

## [1.0.0] — 2026-09-17

First release under **strict semver**. From here, no breaking change to a public
API ships without a major version bump; see [docs/STABILITY.md](docs/STABILITY.md)
and the [migration guide](docs/MIGRATION.md). This release wires the full
HPO + NAS + multi-objective + distributed + pipeline story end to end (proved by
`crates/automl-tasks/tests/end_to_end.rs`, one strand per §28 DoD item) and
completes the documentation pass.

### v1.0 stabilization

- **Strict-semver stability contract** ([docs/STABILITY.md](docs/STABILITY.md)):
  the public trait surface (`Sampler`, `Pruner`, `Executor`, `Storage`, `Budget`,
  `TaskAdapter`), the slow-moving `Study` API, storage-schema compatibility, and
  determinism are all documented guarantees; new trait capabilities ship only as
  default-implemented methods.
- **Migration guarantees** ([docs/MIGRATION.md](docs/MIGRATION.md)): the SQLite
  schema history (v1–v4) is documented with additive-only, forward-compatible
  upgrades and a downgrade path; a study from any earlier release stays loadable.
- **End-to-end story suite**: an integration test exercising HPO + resume,
  multi-objective + Pareto/hypervolume, NAS macro-space search, executable
  pipeline search, multimodal pre-scheduling validation, distributed
  exactly-once draining, and deterministic replay — against the real public API.
- **Documentation pass**: [docs/v1.0-checklist.md](docs/v1.0-checklist.md) maps
  every §28 DoD item to its implementation and proof; README updated to 1.0.

**Breaking:** none. This release only adds the stability contract and docs; no
public trait or `Study` signature changed from the 0.x line.

### Added — v0.1 optimization foundation

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
- Provenance & timing (§19): every `TrialRecord` now carries an `EnvSnapshot`
  (OS/arch/core version) and `TrialTiming` (queued/started/completed millis,
  with `wall_time_ms`). Storage stamps start/complete times; the SQLite backend
  gains a v2 migration (nullable `env`/`queued_at`/`started_at`/`completed_at`
  columns) that upgrades a v1 database in place. The dashboard shows per-trial
  wall time. Old persisted records read back with default env and empty timing.
- `Budget` trait with `TrialBudget`, `WallTimeBudget`, `EpochBudget`,
  `StepBudget`, `Unbounded` built-ins. The study loop now tallies epochs and
  steps consumed (a trial's last reported step, and its report count) into
  `Consumption`, so `EpochBudget`/`StepBudget` actually fire end-to-end.
- `Objective` / `TaskAdapter` / `ReportSink` seam between the engine and
  concrete workloads.
- `DeviceScheduler` and memory-aware admission control (§18.3): reserves
  per-device capacity via RAII `DeviceLease`s, distinguishes an impossible
  configuration (`AdmissionError::Impossible`) from transient contention
  (`NoCapacityAvailable`), and models shared (MPS-style) devices through
  fractional GPU/CPU requests. Pure `ResourceSpec` accounting, ready to wire
  into the distributed executor.
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
- Cross-validation (§6): an `Evaluation` scheme — `Holdout`, `KFold`, and
  `StratifiedKFold` — chosen via `.evaluation(..)` on the `Auto*` builders. The
  objective runs each fold and reports one aggregated metric (a per-fold
  running mean under K-fold), so the sampler and pruner stay oblivious to the
  scheme. Holdout keeps the per-epoch learning curve.

### Added — Pipeline & multimodal (v0.9)

- Pipeline-AutoML composition (`automl-core::pipeline`, §16): a `PipelineSpace`
  of ordered `Stage`s, each offering `Component`s that carry their own conditional
  hyperparameter subspace. `to_search_space` lowers a whole pipeline to a flat
  conditional `SearchSpace` and `decode` lifts a sampled `ParamSet` back to a
  `PipelinePlan` — **no separate pipeline DSL**, so the sampler, pruner and
  storage layers stay unaware pipeline search is happening (§16), the same
  encode/decode pattern as `nas`.
- Executable tabular pipeline search (`automl-tasks::pipeline`, §16/§20):
  `AutoPipeline` jointly searches a preprocessing transform (standardize /
  normalize / none) and a classifier (nearest-centroid or k-NN with searched `k`)
  over the core `PipelineSpace`, maximizing validation accuracy.
- Multimodal fusion primitives (`automl-core::fusion`, §15): a `MultimodalSpace`
  searches each modality's encoder and projection dimension plus a `Fusion`
  strategy (`Concat`/`Mean`/`Sum`); `MultimodalPlan::validate` checks dimension
  compatibility **before scheduling** so an incompatible configuration never
  reaches the executor (§15), and `fuse` combines per-modality features.
- Dashboard v1 (`automl-cli`, §25): per-trial **learning-curve overlays colored by
  fate** (surviving vs pruned vs failed) and a **run-summary / utilization** panel
  (state breakdown, wall-time totals), still a self-contained, read-only pure
  function of `Storage` history.
- Distributed hardening (`automl-core::distributed`, §18, §24): a
  failure-injection suite — mid-trial worker crash with orphan recovery,
  orphan-recovery under load losing no trials, and concurrent-claim mutual
  exclusion — all asserting exactly-once completion, plus documented lease-TTL /
  heartbeat tuning guidance.

### Added — RL & generative (v0.8)

- Robust aggregation for noisy objectives (`automl-core::robust`, §13/§14, §27):
  the `Aggregator` (`Mean`/`Median`/`TrimmedMean`) reduces replicate scores to an
  outlier-resistant point estimate with a `spread` confidence proxy, and
  `replicate` wraps any objective to evaluate it several times under derived seeds
  and optimize the robust aggregate transparently — the sampler and pruner never
  learn the objective was noisy. This is the §27 "replicates, confidence
  estimates, robust aggregation mode" mitigation, locked in for RL/GAN/diffusion.
- Median pruner robust noisy-curve mode (`MedianPruner::with_robust_window`, §18):
  compares the median of a trial's last `k` reported values against peers instead
  of its single latest value, so a lucky/unlucky spike on a noisy learning curve
  does not decide pruning.
- Reinforcement-learning search (`automl-tasks::rl`, §14/§20): a stochastic
  `GridWorld` (slippery corridor) and a tabular `QLearner`, wrapped by `AutoRl`,
  which searches learning rate / discount / exploration to maximize episode return
  — evaluated with **replicated robust aggregation** of the noisy return. The
  adapter owns the environment, rollouts and the episode/step budgets (§14).
- Generative helpers (`automl-burn::generative`, §13/§20): `AutoAutoencoder`, a
  one-call reconstruction-autoencoder search (latent size, width, depth, lr,
  minimizing validation MSE) — the "autoencoders arrive earlier" deliverable;
  `DiffusionSchedule` (linear/cosine noise schedules with derived `alpha_bar`s);
  and `gan_space`, a coordinated generator/discriminator search space that is
  dimension-compatible by construction. GAN/diffusion ship as *helpers*, their
  noisy quality metrics intended for the robust-aggregation mode above.

### Added — NAS & video (v0.7)

- Architecture-graph search primitives (`automl-core::nas`, §4.1/§21): a
  framework-agnostic `MacroSpace` describes a variable-depth stack of cells, each
  choosing one op from a shared, **reusable** `OpPalette` plus a channel width and
  an optional residual skip. `MacroSpace::to_search_space` *encodes the
  architecture as an ordinary conditional `SearchSpace`*, so every existing
  sampler searches architectures with no NAS-specific machinery — and the
  `EvolutionarySampler` in particular gives **evolutionary NAS** for free.
  `MacroSpace::mutate` is the explicit graph operator (op swap, width change, skip
  toggle, depth ±1), and `decode` turns a sampled `ParamSet` into a concrete
  `Architecture`. The core knows only typed op *names*; adapters materialize them.
- `AutoNas` (`automl-burn::nas`, §10/§20): the Burn materialization — a `NasCnn`
  builds a decoded `Architecture` into same-resolution conv cells (with residual
  projections for skips) plus adaptive pooling and a linear head, and `AutoNas`
  runs a one-call image-classification NAS driven by the evolutionary sampler.
- Video classification (`automl-burn::video`, §11/§20): a `frame_pool` spatial
  front-end turns each clip into a sequence of pooled per-frame features, and
  `AutoVideo` classifies clips with the recurrent `AutoSequence` model (the
  CNN+RNN family from §11) — video reusing the sequence engine the way audio does.
- Checkpoint promotion (`automl-core::checkpoint`, §10/§11 "checkpoint reuse",
  §18): a `CheckpointPromoter` couples ASHA-style geometric rungs to the v0.5
  artifact store — it records a trial's checkpoint at each rung, selects the top
  `1/eta` promoted set, and `warm_start`s a promoted continuation from the best
  predecessor's weights instead of retraining. Framework-agnostic: it moves opaque
  checkpoint bytes through `Storage` artifacts and ranks by reported score.

### Added — Distributed alpha (v0.6)

- Single-object detection (`automl-burn::detection`, §10/§20): a `Detector` CNN
  that regresses a normalized bounding box, a `box_iou` metric, and
  `AutoDetection`, a one-call localization search scored by mean box IoU.
- Speech/audio classification (`automl-burn::audio`, §12/§20): a `spectrogram`
  front-end (windowed Hann DFT) and `AutoAudio`, which turns waveforms into
  spectrogram sequences and classifies them with the recurrent `AutoSequence`
  model.

- Distributed lease protocol (§18.2, §23): the `Storage` trait gains
  `claim_trial` (compare-and-swap: claim a `Waiting` trial, or a `Running` one
  whose lease expired — an orphan), `renew_lease` (heartbeat + exactly-once
  guard), and `recover_orphans` (requeue expired leases). Implemented in the
  in-memory backend and the SQLite backend (schema migration v4, lease columns).
- `automl-core::distributed`: `enqueue_pending` (the coordinator enqueues
  `Waiting` trials) and `Worker`, which claims a trial, runs it while renewing
  its lease at each report (the heartbeat), and completes it **only if it still
  holds the lease** — so a worker declared dead and reassigned never
  double-completes. `Storage` stays the single source of truth (stateless
  coordinator). Tested with interleaved and multi-threaded workers plus orphan
  recovery.

### Added — Vision & multi-objective (v0.5)

- Image classification (`automl-burn::vision`, §10/§20): a `CnnClassifier`
  (two conv blocks + adaptive pooling + linear head, channel widths searchable)
  and `AutoVision`, a one-call image-classification search. Adaptive pooling
  makes the head independent of input resolution.
- Semantic segmentation (`automl-burn::segmentation`, §10/§20): a same-resolution
  `FcnSegmenter` producing per-pixel logits, a mean-IoU metric, and
  `AutoSegmentation`, a one-call segmentation search over images and per-pixel
  masks.

- Hypervolume reporting (§17.1, §23): `Study::hypervolume(reference)` and
  `Study::hypervolume_history(reference)` — the latter the monotone
  hypervolume-over-time curve for multi-objective progress.
- Artifact storage (§19 "Artifacts", §23): `Storage::{save,load,list}_artifact`
  persist named binary artifacts (checkpoints, exported models, configs, logs)
  per trial, idempotent by `(trial, name)`. Implemented in the in-memory backend
  and the SQLite backend (schema migration v3, a `BLOB` `artifacts` table);
  default trait methods keep custom backends non-breaking.

### Added — Transformers & unsupervised (v0.4)

- Transformer sequence classifier (`automl-burn::transformer`, §9/§20):
  `AutoTransformer` searches heads, per-head width, depth, FFN ratio, lr and
  dropout over a Burn `TransformerEncoder` (embedding + sinusoidal positions +
  mean-pool + head). The `d_model % heads == 0` compatibility constraint (§5)
  holds by construction (`d_model = heads * head_dim`).
- `automl-tasks` crate (§30): framework-agnostic classical task adapters that
  need no deep-learning backend.
  - Clustering (§7/§20): K-means (k-means++) with silhouette scoring, and
    `AutoCluster`, which grid-searches the number of clusters to maximize
    silhouette.
  - Anomaly detection (§7/§20): a k-NN distance scorer and `AutoAnomaly`, which
    jointly searches the neighbour count and threshold to maximize F1.

### Added — Time series & sequences (v0.3)

- Time-series forecasting (`automl-burn::timeseries`, §8/§20): time-aware
  backtesting (`TimeSplit::Expanding`/`Sliding`) that always validates *after*
  training and asserts loudly on temporal leakage; forecasting metrics (MAE,
  RMSE, sMAPE, MASE, pinball); and `AutoForecaster`, a one-call search over
  lookback window + MLP hyperparameters evaluated by backtest.
- Recurrent sequence models (`automl-burn::sequence`, §8/§20): an LSTM/GRU
  `RnnClassifier` whose cell type/width/dropout are searchable, and
  `AutoSequence`, a one-call sequence-classification search over fixed-length
  multivariate sequences.

### Added — Burn integration

- `automl-burn` crate: the Burn deep-learning execution adapter. A configurable
  MLP, a manual training loop that reports validation accuracy per epoch to the
  core `ReportSink` (feeding the pruner) and honors `should_stop`, and a
  `MnistMlpObjective` that turns MNIST training into an `automl_core::Objective`.
  Uses the CPU-only `ndarray` backend (no GPU/system deps). Includes a
  download-free synthetic-task test and an end-to-end `mnist_search` example.

### Deferred

- Nothing outstanding from the v0.1→v1.0 plan. The two items deferred through
  v1.0 — the process executor tier and the PostgreSQL backend — both shipped in
  1.1.0 (see above). Post-1.0 extension work (a learned text embedding layer,
  multi-fidelity BOHB, additional accelerator backends) is tracked as new work,
  not carried-over debt.
