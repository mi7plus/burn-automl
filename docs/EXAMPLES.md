# Examples

Every example is runnable and self-contained; the ones marked *synthetic* need no
download and finish in seconds. Run any with `cargo run -p <crate> --example
<name>` (add `--release` for the model-training ones).

## Engine capabilities — `automl-core` (framework-agnostic, synthetic)

| Example | What it shows |
|---------|---------------|
| `sphere` | The one-call quickstart: minimize a 2-D function |
| `benchmarks` | Random vs TPE vs multivariate-TPE vs Evolutionary on the standard test functions |
| `hyperband` | Multi-fidelity Hyperband, and BOHB (Hyperband + TPE) |
| `nsga2` | Multi-objective search over ZDT1 with the NSGA-II sampler |
| `warm_start` | Transferring a prior study's best configs into a new one |
| `multimodal` | Searchable modality encoders + fusion, with pre-scheduling dimension validation |
| `robust_aggregation` | Replicated evaluation of a noisy objective (Mean vs Median vs TrimmedMean) |
| `distributed` | Several `Worker`s draining a shared study queue exactly once |

```bash
cargo run -p automl-core --release --example benchmarks
```

## Classical adapters — `automl-tasks` (no deep learning, synthetic)

| Example | Use case |
|---------|----------|
| `clustering` | Unsupervised K-means, searching the number of clusters (`AutoCluster`) |
| `anomaly` | k-NN anomaly detection, searching neighbours + threshold (`AutoAnomaly`) |
| `reinforcement_learning` | Tabular Q-learning on a slippery corridor with robust returns (`AutoRl`) |
| `pipeline` | Joint preprocessing + model pipeline search (`AutoPipeline`) |

```bash
cargo run -p automl-tasks --release --example pipeline
```

## Deep-learning adapters — `automl-burn`

Synthetic (no download, run in seconds — CNNs/RNNs still train, so use `--release`):

| Example | Use case (`Auto*` API) |
|---------|------------------------|
| `tabular` | Supervised classification & regression (`AutoClassifier`, `AutoRegressor`) |
| `time_series` | Forecasting with time-aware backtesting (`AutoForecaster`) |
| `sequence` | Recurrent sequence classification (`AutoSequence`) |
| `transformer` | Transformer-encoder sequence classification (`AutoTransformer`) |
| `seq2seq` | Encoder–decoder sequence transduction (`AutoSeq2Seq`) |
| `text` | NLP text classification, hashing & learned embedding (`AutoText`) |
| `vision` | Image classification (`AutoVision`) |
| `segmentation` | Semantic segmentation (`AutoSegmentation`) |
| `detection` | Single-object detection (`AutoDetection`) |
| `audio` | Speech/audio classification (`AutoAudio`) |
| `video` | Video classification (`AutoVideo`) |
| `generative` | Reconstruction autoencoder + diffusion schedules (`AutoAutoencoder`, `DiffusionSchedule`) |

Real MNIST (downloads on first run):

| Example | Use case |
|---------|----------|
| `mnist_search` | MLP hyperparameter search on real MNIST |
| `mnist_nas` | Neural architecture search on real MNIST (`AutoNas`) |

```bash
cargo run -p automl-burn --release --example vision
```

## Notes

- The synthetic examples use tiny data and few trials so they run quickly; scale
  `.trials(..)`, `.epochs(..)` and the dataset up for real work.
- Every example is deterministic given its seed.
- Enable a GPU backend with `--features wgpu` (or `cuda` / `metal`) on the
  `automl-burn` examples; it is a device change, not a code change.
