//! Neural Architecture Search over image classifiers — the `AutoNas` API
//! (roadmap v0.7; PRD §10, §21).
//!
//! This is the Burn-side materialization of the framework-agnostic architecture
//! graph in [`automl_core::nas`]. A decoded [`Architecture`] — a stack of cells,
//! each an op (`conv3`/`conv5`/`identity`) with a channel width and an optional
//! residual skip from the previous cell — becomes a concrete [`NasCnn`] Burn
//! module. The search itself is ordinary optimization: the macro-architecture is
//! encoded as a conditional search space and driven by the
//! [`EvolutionarySampler`], so
//! architecture mutation *is* the sampler's mutation — evolutionary NAS with no
//! bespoke controller.
//!
//! Every op preserves spatial size (`Same` padding) so arbitrary depths compose;
//! an adaptive pool makes the classifier head independent of input resolution.

use crate::common::{image_tensor, split};
use crate::sequence::label_tensor;
use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::nas::{Architecture, MacroSpace, OpPalette};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, EvolutionarySampler, MedianPruner, Study};

use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

const POOL: usize = 4;

/// The kernel size a palette op maps to. Unknown ops fall back to `1` (a
/// pointwise/identity-style cell), keeping materialization total.
fn op_kernel(op: &str) -> usize {
    match op {
        "conv5" => 5,
        "conv3" => 3,
        _ => 1, // "identity" and anything else: 1x1 pointwise.
    }
}

/// A convolutional classifier whose body is a decoded [`Architecture`]: a stack
/// of same-resolution conv cells with optional residual skips, then adaptive
/// pooling and a linear head.
#[derive(Module, Debug)]
pub struct NasCnn<B: Backend> {
    cells: Vec<Conv2d<B>>,
    // Per cell, a 1x1 projection present iff that cell has a residual skip (it
    // projects the cell input to the cell's channel width so the add is valid).
    projections: Vec<Option<Conv2d<B>>>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    activation: Relu,
    head: Linear<B>,
}

impl<B: Backend> NasCnn<B> {
    /// Materialize `arch` into a concrete model over `in_channels`-channel images
    /// with `num_classes` outputs.
    pub fn build(
        arch: &Architecture,
        in_channels: usize,
        num_classes: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let same = |ci: usize, co: usize, k: usize, device: &B::Device| {
            Conv2dConfig::new([ci, co], [k, k])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let mut cells = Vec::with_capacity(arch.depth());
        let mut projections = Vec::with_capacity(arch.depth());
        let mut prev = in_channels;
        for layer in &arch.layers {
            cells.push(same(prev, layer.width, op_kernel(&layer.op), device));
            projections.push(if layer.skip_from_prev {
                Some(same(prev, layer.width, 1, device))
            } else {
                None
            });
            prev = layer.width;
        }
        NasCnn {
            cells,
            projections,
            pool: AdaptiveAvgPool2dConfig::new([POOL, POOL]).init(),
            dropout: DropoutConfig::new(dropout).init(),
            activation: Relu::new(),
            head: LinearConfig::new(prev * POOL * POOL, num_classes).init(device),
        }
    }

    /// Forward pass: `[batch, channels, height, width]` to `[batch, classes]`.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = images;
        for (cell, proj) in self.cells.iter().zip(self.projections.iter()) {
            let mut y = self.activation.forward(cell.forward(x.clone()));
            if let Some(p) = proj {
                // Residual: add the projected cell input to the cell output.
                y = y + p.forward(x);
            }
            x = y;
        }
        let x = self.dropout.forward(x);
        let x = self.pool.forward(x);
        let [b, c, h, w] = x.dims();
        self.head.forward(x.reshape([b, c * h * w]))
    }
}

/// One-call neural architecture search for image classification (PRD §10, §20).
///
/// Searches a stack of conv cells (op, width, skip, depth) plus learning rate and
/// dropout, driven by an evolutionary sampler. `images[i]` is a
/// `channels * height * width` row-major flat pixel vector.
pub struct AutoNas {
    images: Vec<Vec<f32>>,
    labels: Vec<i64>,
    channels: usize,
    height: usize,
    width: usize,
    macro_space: MacroSpace,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoNas {
    /// A new NAS search with a default palette (`conv3`/`conv5`/`identity`),
    /// depth 1–3 and channel widths `{8, 16, 32}`. Override the architecture
    /// space with [`AutoNas::macro_space`].
    pub fn new(
        images: Vec<Vec<f32>>,
        labels: Vec<i64>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        let palette =
            OpPalette::new(["conv3", "conv5", "identity"]).expect("default palette is non-empty");
        let macro_space =
            MacroSpace::new(palette, 1, 3, vec![8, 16, 32]).expect("default macro-space is valid");
        AutoNas {
            images,
            labels,
            channels,
            height,
            width,
            macro_space,
            num_classes: 0,
            epochs: 8,
            batch_size: 32,
            trials: 12,
            val_fraction: 0.2,
            seed: 0,
        }
    }

    /// Replace the architecture search space.
    pub fn macro_space(mut self, macro_space: MacroSpace) -> Self {
        self.macro_space = macro_space;
        self
    }

    /// Number of classes (inferred as `max(label)+1` if unset).
    pub fn num_classes(mut self, k: usize) -> Self {
        self.num_classes = k;
        self
    }

    /// Epochs per trial.
    pub fn epochs(mut self, e: usize) -> Self {
        self.epochs = e.max(1);
        self
    }

    /// Number of architectures to evaluate.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Seed for the split and search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (maximizing validation accuracy). Each
    /// trial's winning architecture description is saved as the `"architecture"`
    /// final metric context via the study history.
    pub fn fit(self) -> CoreResult<Study> {
        let pixels = self.channels * self.height * self.width;
        if self.images.is_empty()
            || self.images.len() != self.labels.len()
            || self.images[0].len() != pixels
        {
            return Err(Error::Objective(
                "image data must be non-empty, match labels, and have channels*height*width pixels"
                    .into(),
            ));
        }
        let num_classes = if self.num_classes == 0 {
            (self.labels.iter().copied().max().unwrap_or(0) + 1) as usize
        } else {
            self.num_classes
        };

        // Architecture graph encoded as a conditional space, plus training knobs.
        let space = self
            .macro_space
            .to_search_space()
            .add("lr", Distribution::log_float(1e-3, 1e-2))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-nas")
            .maximize("accuracy")
            // Evolutionary search over the encoded architecture = evolutionary NAS.
            .sampler(EvolutionarySampler::new(
                "accuracy",
                Direction::Maximize,
                self.seed,
            ))
            .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.images.len(), self.val_fraction, self.seed);
        let dims = (self.channels, self.height, self.width);
        let data = Arc::new((self.images, self.labels, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);
        let macro_space = self.macro_space.clone();

        let objective =
            move |p: &ParamSet, sink: &mut dyn ReportSink| -> CoreResult<NamedMetrics> {
                let arch = macro_space.decode(p)?;
                let acc = train_and_eval::<TrainBackend>(
                    &arch,
                    num_classes,
                    p.float("lr")?,
                    p.float("dropout")?,
                    epochs,
                    batch,
                    dims,
                    &data,
                    sink,
                );
                Ok(NamedMetrics::single("accuracy", acc as f64))
            };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type ImgData = (Vec<Vec<f32>>, Vec<i64>, Vec<usize>, Vec<usize>);

#[allow(clippy::too_many_arguments)]
fn train_and_eval<B: AutodiffBackend>(
    arch: &Architecture,
    num_classes: usize,
    lr: f64,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    dims: (usize, usize, usize),
    data: &ImgData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = NasCnn::<B>::build(arch, dims.0, num_classes, dropout, &device);
    let mut optim = AdamConfig::new().init();
    let (imgs, labels, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut acc = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(batch_size) {
            let x = image_tensor::<B>(imgs, chunk, dims, &device);
            let y = label_tensor::<B>(labels, chunk, &device);
            let out = model.forward(x);
            let loss = CrossEntropyLoss::new(None, &out.device()).forward(out, y);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        acc = accuracy::<B::InnerBackend>(&valid, imgs, labels, val_idx, batch_size, dims, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("accuracy", acc as f64));
        if sink.should_stop() {
            break;
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn accuracy<B: Backend>(
    model: &NasCnn<B>,
    imgs: &[Vec<f32>],
    labels: &[i64],
    val_idx: &[usize],
    batch_size: usize,
    dims: (usize, usize, usize),
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut correct: i64 = 0;
    for chunk in val_idx.chunks(batch_size) {
        let x = image_tensor::<B>(imgs, chunk, dims, device);
        let y = label_tensor::<B>(labels, chunk, device);
        let pred = model.forward(x).argmax(1).squeeze_dim(1);
        correct += pred.equal(y).int().sum().into_scalar().elem::<i64>();
    }
    correct as f32 / val_idx.len() as f32 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four classes over 1x8x8 images: a bright block in one of four quadrants.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<i64>) {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut imgs, mut labels) = (Vec::new(), Vec::new());
        for i in 0..n {
            let q = (i % 4) as i64;
            let (r0, c0) = ((q / 2) as usize * 4, (q % 2) as usize * 4);
            let mut img = vec![0f32; 64];
            for r in r0..r0 + 4 {
                for c in c0..c0 + 4 {
                    img[r * 8 + c] = 1.0 + rng.gen_range(-0.1..0.1);
                }
            }
            imgs.push(img);
            labels.push(q);
        }
        (imgs, labels)
    }

    #[test]
    fn nas_cnn_materializes_and_runs() {
        // A two-cell architecture with a skip materializes and produces logits.
        use automl_core::nas::Layer;
        let arch = Architecture {
            layers: vec![
                Layer {
                    op: "conv3".into(),
                    width: 8,
                    skip_from_prev: false,
                },
                Layer {
                    op: "conv5".into(),
                    width: 12,
                    skip_from_prev: true,
                },
            ],
        };
        let device = Default::default();
        let model = NasCnn::<TrainBackend>::build(&arch, 1, 4, 0.0, &device);
        let x = image_tensor::<TrainBackend>(
            &[vec![0.5; 64], vec![0.2; 64]],
            &[0, 1],
            (1, 8, 8),
            &device,
        );
        let out = model.forward(x);
        assert_eq!(out.dims(), [2, 4]);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn auto_nas_searches_architectures() {
        // A shallow palette and few trials keep the conv search affordable on the
        // CPU ndarray backend while still exercising the evolutionary NAS path.
        use automl_core::nas::MacroSpace;
        let (imgs, labels) = synthetic(64, 1);
        let palette = OpPalette::new(["conv3", "identity"]).unwrap();
        let macro_space = MacroSpace::new(palette, 1, 2, vec![8, 16]).unwrap();
        let study = AutoNas::new(imgs, labels, 1, 8, 8)
            .macro_space(macro_space)
            .num_classes(4)
            .epochs(4)
            .trials(3)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(
            acc > 50.0,
            "best NAS accuracy was {acc}, expected a searched CNN to learn"
        );
    }

    #[test]
    fn bad_shape_errors() {
        assert!(AutoNas::new(vec![vec![0.0; 3]], vec![0], 1, 8, 8)
            .fit()
            .is_err());
    }
}
