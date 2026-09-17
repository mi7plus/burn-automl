//! MNIST training machinery and the `MnistMlpObjective` example objective.
//!
//! This is the concrete end-to-end example wiring: a batcher that normalizes
//! MNIST items, a manual training loop that reports per-epoch validation
//! accuracy to the core [`ReportSink`] (feeding the pruner) and honors
//! `should_stop`, and an [`Objective`] that turns MNIST hyperparameter search
//! into an [`automl_core::objective::Objective`]. Kept out of the crate root so
//! the dataset-specific dependencies stay contained.

use crate::{Mlp, MlpConfig, TrainBackend};
use automl_core::error::Result as CoreResult;
use automl_core::metrics::NamedMetrics;
use automl_core::objective::{Objective, ReportSink};
use automl_core::param::ParamSet;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::{MnistDataset, MnistItem};
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Batches [`MnistItem`]s into normalized image and integer-label tensors.
/// Mirrors the normalization from Burn's own MNIST guide.
#[derive(Clone, Default)]
pub struct MnistBatcher {}

/// A batch of images and their target labels.
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    /// Normalized images, shape `[batch, 28, 28]`.
    pub images: Tensor<B, 3>,
    /// Target class indices, shape `[batch]`.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Scale to [0,1] then standardize with MNIST's mean/std.
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

/// Hyperparameters and schedule for one MLP training run.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Number of epochs to train.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Optimizer learning rate.
    pub lr: f64,
    /// Seed for train-set shuffling.
    pub seed: u64,
    /// Model architecture.
    pub model: MlpConfig,
}

/// Train an [`Mlp`] on the given items and return validation accuracy (percent).
///
/// After every epoch the current validation accuracy is reported to `sink`
/// under the metric name `"accuracy"`, and [`ReportSink::should_stop`] is
/// polled so a pruned trial exits early. The most recent validation accuracy is
/// returned.
pub fn train_and_eval<B: AutodiffBackend>(
    config: &TrainConfig,
    mut train_items: Vec<MnistItem>,
    test_items: Vec<MnistItem>,
    device: &B::Device,
    sink: &mut dyn ReportSink,
) -> f32 {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.seed);
    let mut model = config.model.init::<B>(device);
    let mut optim = AdamConfig::new().init();
    let batcher = MnistBatcher::default();

    // `AutodiffBackend` guarantees the inner backend shares the device type, so
    // the same device handle validates the non-autodiff model.
    let mut last_accuracy = 0.0;

    for epoch in 1..=config.epochs {
        train_items.shuffle(&mut rng);

        for chunk in train_items.chunks(config.batch_size) {
            let batch = batcher.batch(chunk.to_vec(), device);
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device()).forward(output, batch.targets);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(config.lr, model, grads);
        }

        // Validate on the non-autodiff model.
        let valid = model.valid();
        last_accuracy =
            accuracy::<B::InnerBackend>(&valid, &batcher, &test_items, config.batch_size, device);

        let _ = sink.report(
            epoch as u64,
            NamedMetrics::single("accuracy", last_accuracy as f64),
        );
        if sink.should_stop() {
            break;
        }
    }
    last_accuracy
}

/// Compute classification accuracy (percent) of `model` over `items`.
fn accuracy<B: Backend>(
    model: &Mlp<B>,
    batcher: &MnistBatcher,
    items: &[MnistItem],
    batch_size: usize,
    device: &B::Device,
) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    let mut correct: i64 = 0;
    let mut total: usize = 0;
    for chunk in items.chunks(batch_size) {
        let batch = batcher.batch(chunk.to_vec(), device);
        let output = model.forward(batch.images);
        let predictions = output.argmax(1).squeeze_dim(1);
        let corrects = predictions.equal(batch.targets).int().sum().into_scalar();
        correct += corrects.elem::<i64>();
        total += chunk.len();
    }
    correct as f32 / total as f32 * 100.0
}

/// An [`Objective`] that trains an MLP on MNIST for a sampled set of
/// hyperparameters and returns validation accuracy.
///
/// The search space it expects (built by [`MnistMlpObjective::search_space`]):
/// `lr` (log-float), `hidden_size` (int), `num_layers` (int), `dropout`
/// (float). Loading the dataset downloads MNIST on first use.
#[derive(Clone, Debug)]
pub struct MnistMlpObjective {
    /// Epochs per trial.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Number of training examples to use (a subset keeps CPU trials fast).
    pub train_size: usize,
    /// Number of test examples to evaluate on.
    pub test_size: usize,
}

impl Default for MnistMlpObjective {
    fn default() -> Self {
        MnistMlpObjective {
            epochs: 3,
            batch_size: 64,
            train_size: 4000,
            test_size: 1000,
        }
    }
}

impl MnistMlpObjective {
    /// The search space this objective optimizes over. Maximize `"accuracy"`.
    pub fn search_space() -> automl_core::space::SearchSpace {
        use automl_core::distribution::Distribution;
        automl_core::space::SearchSpace::new()
            .add("lr", Distribution::log_float(1e-4, 1e-2))
            .add("hidden_size", Distribution::int(32, 256))
            .add("num_layers", Distribution::int(1, 3))
            .add("dropout", Distribution::float(0.0, 0.5))
    }

    fn config_from(&self, params: &ParamSet) -> CoreResult<TrainConfig> {
        Ok(TrainConfig {
            epochs: self.epochs,
            batch_size: self.batch_size,
            lr: params.float("lr")?,
            seed: 42,
            model: MlpConfig::new(params.int("hidden_size")? as usize)
                .with_num_hidden_layers(params.int("num_layers")? as usize)
                .with_dropout(params.float("dropout")?),
        })
    }
}

impl Objective for MnistMlpObjective {
    fn evaluate(&self, params: &ParamSet, report: &mut dyn ReportSink) -> CoreResult<NamedMetrics> {
        let config = self.config_from(params)?;
        let device = Default::default();

        let train = subset(&MnistDataset::train(), self.train_size);
        let test = subset(&MnistDataset::test(), self.test_size);

        let acc = train_and_eval::<TrainBackend>(&config, train, test, &device, report);
        Ok(NamedMetrics::single("accuracy", acc as f64))
    }
}

/// Take the first `n` items of a dataset into a `Vec`.
fn subset<D: Dataset<MnistItem>>(dataset: &D, n: usize) -> Vec<MnistItem> {
    let count = dataset.len().min(n);
    (0..count).filter_map(|i| dataset.get(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use automl_core::trial::TrialId;

    struct NullSink(TrialId);
    impl ReportSink for NullSink {
        fn trial_id(&self) -> TrialId {
            self.0
        }
        fn report(&mut self, _step: u64, _metrics: NamedMetrics) -> CoreResult<()> {
            Ok(())
        }
        fn should_stop(&self) -> bool {
            false
        }
    }

    /// A trivially learnable synthetic task: the label is encoded as a single
    /// hot pixel. No network access, so this runs in CI. The MLP should learn it
    /// well above chance in a few epochs, exercising the whole train/eval path.
    fn synthetic(n: usize) -> Vec<MnistItem> {
        (0..n)
            .map(|i| {
                let label = (i % 10) as u8;
                let mut image = [[0.0f32; 28]; 28];
                // A full class-specific row at full intensity (0-255 scale) is a
                // strong, linearly separable signal that survives the batcher's
                // /255 normalization.
                image[label as usize] = [255.0; 28];
                MnistItem { image, label }
            })
            .collect()
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn trains_on_synthetic_task_above_chance() {
        let config = TrainConfig {
            epochs: 10,
            batch_size: 32,
            lr: 1e-2,
            seed: 1,
            model: MlpConfig::new(64).with_dropout(0.0),
        };
        let device = Default::default();
        let mut sink = NullSink(TrialId(0));
        let acc = train_and_eval::<TrainBackend>(
            &config,
            synthetic(500),
            synthetic(100),
            &device,
            &mut sink,
        );
        // Chance is 10%; the task is trivially separable, so expect well above.
        assert!(
            acc > 50.0,
            "accuracy was {acc}, expected the MLP to learn the task"
        );
    }

    #[test]
    fn search_space_is_valid() {
        assert!(MnistMlpObjective::search_space().validate().is_ok());
    }
}
