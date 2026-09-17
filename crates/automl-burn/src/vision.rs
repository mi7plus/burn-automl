//! Convolutional image classification and the `AutoVision` API (PRD §10, §20).
//!
//! A small CNN (two conv blocks + adaptive pooling + linear head) whose channel
//! widths and regularization are searchable, plus `AutoVision`, a one-call image
//! classification search. Images are fixed-shape `[channels, height, width]`
//! (flattened row-major); adaptive pooling makes the head independent of the
//! input resolution.

use crate::common::{image_tensor, split};
use crate::sequence::label_tensor;
use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

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

/// A small convolutional image classifier.
#[derive(Module, Debug)]
pub struct CnnClassifier<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    activation: Relu,
    head: Linear<B>,
}

/// Configuration for [`CnnClassifier`].
#[derive(Config, Debug)]
pub struct CnnConfig {
    /// Input channel count.
    pub in_channels: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// First convolution's output channels.
    #[config(default = 16)]
    pub ch1: usize,
    /// Second convolution's output channels.
    #[config(default = 32)]
    pub ch2: usize,
    /// Dropout rate.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl CnnConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnClassifier<B> {
        // "Same" padding preserves spatial size so adaptive pooling always has a
        // full-resolution feature map regardless of the input image size.
        let conv = |ci: usize, co: usize, device: &B::Device| {
            Conv2dConfig::new([ci, co], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        CnnClassifier {
            conv1: conv(self.in_channels, self.ch1, device),
            conv2: conv(self.ch1, self.ch2, device),
            pool: AdaptiveAvgPool2dConfig::new([POOL, POOL]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            head: LinearConfig::new(self.ch2 * POOL * POOL, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> CnnClassifier<B> {
    /// Forward pass: `[batch, channels, height, width]` to `[batch, classes]`.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.conv1.forward(images));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.dropout.forward(x);
        let x = self.pool.forward(x); // [b, ch2, POOL, POOL]
        let [b, c, h, w] = x.dims();
        self.head.forward(x.reshape([b, c * h * w]))
    }
}

/// One-call image classification search (PRD §10, §20).
///
/// `images[i]` is a `channels * height * width` row-major flat pixel vector.
pub struct AutoVision {
    images: Vec<Vec<f32>>,
    labels: Vec<i64>,
    channels: usize,
    height: usize,
    width: usize,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoVision {
    /// A new image classification search. `channels*height*width` must equal
    /// the length of each flat image.
    pub fn new(
        images: Vec<Vec<f32>>,
        labels: Vec<i64>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        AutoVision {
            images,
            labels,
            channels,
            height,
            width,
            num_classes: 0,
            epochs: 8,
            batch_size: 32,
            trials: 10,
            val_fraction: 0.2,
            seed: 0,
        }
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

    /// Number of trials to search.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Seed for the split and search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (maximizing validation accuracy).
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

        let space = SearchSpace::new()
            .add("ch1", Distribution::int(4, 16))
            .add("ch2", Distribution::int(8, 32))
            .add("lr", Distribution::log_float(1e-4, 1e-2))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-vision")
            .maximize("accuracy")
            .sampler(TpeSampler::new("accuracy", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.images.len(), self.val_fraction, self.seed);
        let dims = (self.channels, self.height, self.width);
        let data = Arc::new((self.images, self.labels, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective =
            move |p: &ParamSet, sink: &mut dyn ReportSink| -> CoreResult<NamedMetrics> {
                let cfg = CnnConfig::new(dims.0, num_classes)
                    .with_ch1(p.int("ch1")? as usize)
                    .with_ch2(p.int("ch2")? as usize)
                    .with_dropout(p.float("dropout")?);
                let acc = train_and_eval::<TrainBackend>(
                    &cfg,
                    p.float("lr")?,
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
    cfg: &CnnConfig,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    dims: (usize, usize, usize),
    data: &ImgData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
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
    model: &CnnClassifier<B>,
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

    /// Four classes over 1x8x8 images: a bright block occupies one of the four
    /// quadrants. A CNN with spatial pooling classifies the location.
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
    fn auto_vision_classifies_quadrants() {
        let (imgs, labels) = synthetic(120, 1);
        let study = AutoVision::new(imgs, labels, 1, 8, 8)
            .num_classes(4)
            .epochs(4)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(
            acc > 60.0,
            "best vision accuracy was {acc}, expected the CNN to learn"
        );
    }

    #[test]
    fn bad_shape_errors() {
        // 3 pixels declared as 1x8x8 (=64) mismatch.
        assert!(AutoVision::new(vec![vec![0.0; 3]], vec![0], 1, 8, 8)
            .fit()
            .is_err());
    }
}
