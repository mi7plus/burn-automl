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

- Process executor tier (crash isolation), §18.
- PostgreSQL storage backend (§18.4): the distributed lease protocol is
  backend-agnostic and the SQLite backend already implements the persistent
  lease table, so a Postgres backend is a mechanical translation behind a
  feature flag — deferred until a database is available to test against rather
  than shipping unverified DB code.
