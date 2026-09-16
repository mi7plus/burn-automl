//! Semantic segmentation: a fully-convolutional segmenter and the
//! `AutoSegmentation` API (PRD §10, §20).
//!
//! A same-resolution fully-convolutional network predicts a class per pixel;
//! configurations are scored by mean Intersection-over-Union. `AutoSegmentation`
//! searches the channel width, loss composition and training over fixed-shape
//! images and per-pixel masks.

use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{PaddingConfig2d, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// A fully-convolutional segmenter: same-resolution conv blocks then a `1x1`
/// classification convolution producing per-pixel logits.
#[derive(Module, Debug)]
pub struct FcnSegmenter<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    head: Conv2d<B>,
    activation: Relu,
}

/// Configuration for [`FcnSegmenter`].
#[derive(Config, Debug)]
pub struct FcnConfig {
    /// Input channel count.
    pub in_channels: usize,
    /// Number of segmentation classes.
    pub num_classes: usize,
    /// Hidden convolution channels.
    #[config(default = 16)]
    pub channels: usize,
}

impl FcnConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FcnSegmenter<B> {
        let conv = |ci, co, device: &B::Device| {
            Conv2dConfig::new([ci, co], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        FcnSegmenter {
            conv1: conv(self.in_channels, self.channels, device),
            conv2: conv(self.channels, self.channels, device),
            head: Conv2dConfig::new([self.channels, self.num_classes], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> FcnSegmenter<B> {
    /// Forward pass: `[batch, channels, h, w]` to per-pixel logits
    /// `[batch, num_classes, h, w]`.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.activation.forward(self.conv1.forward(images));
        let x = self.activation.forward(self.conv2.forward(x));
        self.head.forward(x)
    }
}

/// Mean Intersection-over-Union over `num_classes`, from flat pixel labels.
pub fn mean_iou(predicted: &[i64], actual: &[i64], num_classes: usize) -> f32 {
    let mut ious = Vec::new();
    for c in 0..num_classes as i64 {
        let mut inter = 0u64;
        let mut union = 0u64;
        for (&p, &a) in predicted.iter().zip(actual) {
            let pp = p == c;
            let aa = a == c;
            if pp && aa {
                inter += 1;
            }
            if pp || aa {
                union += 1;
            }
        }
        if union > 0 {
            ious.push(inter as f32 / union as f32);
        }
    }
    if ious.is_empty() {
        0.0
    } else {
        ious.iter().sum::<f32>() / ious.len() as f32
    }
}

/// One-call semantic-segmentation search (PRD §10, §20).
///
/// `images[i]` is a `channels*height*width` flat pixel vector; `masks[i]` is a
/// `height*width` flat per-pixel class-label vector.
pub struct AutoSegmentation {
    images: Vec<Vec<f32>>,
    masks: Vec<Vec<i64>>,
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

impl AutoSegmentation {
    /// A new segmentation search over images and per-pixel masks.
    pub fn new(
        images: Vec<Vec<f32>>,
        masks: Vec<Vec<i64>>,
        channels: usize,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Self {
        AutoSegmentation {
            images,
            masks,
            channels,
            height,
            width,
            num_classes,
            epochs: 8,
            batch_size: 16,
            trials: 8,
            val_fraction: 0.2,
            seed: 0,
        }
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

    /// Run the search, returning the study (maximizing mean IoU).
    pub fn fit(self) -> CoreResult<Study> {
        let pixels = self.channels * self.height * self.width;
        let mask_pixels = self.height * self.width;
        if self.images.is_empty()
            || self.images.len() != self.masks.len()
            || self.images[0].len() != pixels
            || self.masks[0].len() != mask_pixels
        {
            return Err(Error::Objective(
                "segmentation images/masks must be non-empty and correctly shaped".into(),
            ));
        }

        let space = SearchSpace::new()
            .add("channels", Distribution::int(4, 24))
            .add("lr", Distribution::log_float(1e-3, 1e-2));

        let mut study = Study::builder(space)
            .name("auto-segmentation")
            .maximize("iou")
            .sampler(TpeSampler::new("iou", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("iou", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.images.len(), self.val_fraction, self.seed);
        let dims = (self.channels, self.height, self.width);
        let num_classes = self.num_classes;
        let data = Arc::new((self.images, self.masks, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective =
            move |p: &ParamSet, sink: &mut dyn ReportSink| -> CoreResult<NamedMetrics> {
                let cfg =
                    FcnConfig::new(dims.0, num_classes).with_channels(p.int("channels")? as usize);
                let iou = train_and_eval::<TrainBackend>(
                    &cfg,
                    p.float("lr")?,
                    epochs,
                    batch,
                    dims,
                    num_classes,
                    &data,
                    sink,
                );
                Ok(NamedMetrics::single("iou", iou as f64))
            };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type SegData = (Vec<Vec<f32>>, Vec<Vec<i64>>, Vec<usize>, Vec<usize>);

fn split(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

#[allow(clippy::too_many_arguments)]
fn train_and_eval<B: AutodiffBackend>(
    cfg: &FcnConfig,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    dims: (usize, usize, usize),
    num_classes: usize,
    data: &SegData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (imgs, masks, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut iou = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(batch_size) {
            let x = image_tensor::<B>(imgs, chunk, dims, &device);
            let logits = model.forward(x); // [b, C, h, w]
            let loss = pixel_ce(logits, masks, chunk, dims, num_classes, &device);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        iou = eval_iou::<B::InnerBackend>(&valid, imgs, masks, val_idx, dims, num_classes, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("iou", iou as f64));
        if sink.should_stop() {
            break;
        }
    }
    iou
}

/// Per-pixel cross-entropy: flatten `[b, C, h, w]` logits and `[b, h, w]`
/// targets to `[b*h*w, C]` and `[b*h*w]`.
fn pixel_ce<B: Backend>(
    logits: Tensor<B, 4>,
    masks: &[Vec<i64>],
    idx: &[usize],
    dims: (usize, usize, usize),
    num_classes: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let (_c, h, w) = dims;
    let [b, cc, _h, _w] = logits.dims();
    let flat = logits.permute([0, 2, 3, 1]).reshape([b * h * w, cc]);
    let targets: Vec<i64> = idx.iter().flat_map(|&i| masks[i].iter().copied()).collect();
    let y = Tensor::<B, 1, Int>::from_data(
        TensorData::new(targets, [b * h * w]).convert::<B::IntElem>(),
        device,
    );
    let _ = num_classes;
    CrossEntropyLoss::new(None, &flat.device()).forward(flat, y)
}

#[allow(clippy::too_many_arguments)]
fn eval_iou<B: Backend>(
    model: &FcnSegmenter<B>,
    imgs: &[Vec<f32>],
    masks: &[Vec<i64>],
    val_idx: &[usize],
    dims: (usize, usize, usize),
    num_classes: usize,
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let x = image_tensor::<B>(imgs, val_idx, dims, device);
    // argmax over the class dim -> [b, 1, h, w] -> flatten to pixel labels.
    let preds: Vec<i64> = model
        .forward(x)
        .argmax(1)
        .flatten::<1>(0, 3)
        .into_data()
        .to_vec()
        .expect("i64 predictions");
    let actual: Vec<i64> = val_idx
        .iter()
        .flat_map(|&i| masks[i].iter().copied())
        .collect();
    mean_iou(&preds, &actual, num_classes) * 100.0
}

fn image_tensor<B: Backend>(
    imgs: &[Vec<f32>],
    idx: &[usize],
    dims: (usize, usize, usize),
    device: &B::Device,
) -> Tensor<B, 4> {
    let (c, h, w) = dims;
    let n = idx.len();
    let flat: Vec<f32> = idx.iter().flat_map(|&i| imgs[i].iter().copied()).collect();
    Tensor::<B, 4>::from_data(
        TensorData::new(flat, [n, c, h, w]).convert::<B::FloatElem>(),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_iou_is_correct() {
        // Perfect prediction -> IoU 1.0.
        let a = [0i64, 1, 1, 0];
        assert!((mean_iou(&a, &a, 2) - 1.0).abs() < 1e-6);
        // One pixel wrong.
        let p = [0i64, 1, 0, 0];
        // class 0: inter {0,3}, union {0,2,3} -> 2/3; class 1: inter {1}, union {1,2} -> 1/2.
        assert!((mean_iou(&p, &a, 2) - (2.0 / 3.0 + 0.5) / 2.0).abs() < 1e-6);
    }

    /// Foreground segmentation: the mask marks bright pixels. A conv net learns
    /// to segment them (a value-based, translation-equivariant task).
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<i64>>) {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut imgs, mut masks) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let mut img = vec![0f32; 64];
            let mut mask = vec![0i64; 64];
            for j in 0..64 {
                let bright = rng.gen_bool(0.5);
                img[j] = if bright { 1.0 } else { 0.0 } + rng.gen_range(-0.05..0.05);
                mask[j] = bright as i64;
            }
            imgs.push(img);
            masks.push(mask);
        }
        (imgs, masks)
    }

    #[test]
    fn auto_segmentation_learns_foreground() {
        let (imgs, masks) = synthetic(60, 1);
        let study = AutoSegmentation::new(imgs, masks, 1, 8, 8, 2)
            .epochs(8)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let iou = best.final_value("iou").unwrap();
        assert!(
            iou > 70.0,
            "best mean IoU was {iou}, expected the segmenter to learn"
        );
    }

    #[test]
    fn bad_shape_errors() {
        assert!(
            AutoSegmentation::new(vec![vec![0.0; 3]], vec![vec![0; 64]], 1, 8, 8, 2)
                .fit()
                .is_err()
        );
    }
}
