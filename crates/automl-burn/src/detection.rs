//! Single-object detection (localization) and the `AutoDetection` API
//! (PRD §10, §20).
//!
//! A CNN backbone regresses a normalized bounding box `[x0, y0, x1, y1]` in
//! `[0, 1]` for each image; configurations are scored by mean box IoU.
//! `AutoDetection` searches the backbone channels and training. This is the
//! single-object case; multi-object detection (anchors, NMS) builds on the same
//! backbone in a later release.

use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

const POOL: usize = 4;

/// A CNN that regresses a single normalized bounding box per image.
#[derive(Module, Debug)]
pub struct Detector<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    activation: Relu,
    head: Linear<B>,
}

/// Configuration for [`Detector`].
#[derive(Config, Debug)]
pub struct DetectorConfig {
    /// Input channel count.
    pub in_channels: usize,
    /// First convolution channels.
    #[config(default = 16)]
    pub ch1: usize,
    /// Second convolution channels.
    #[config(default = 32)]
    pub ch2: usize,
}

impl DetectorConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Detector<B> {
        let conv = |ci, co, device: &B::Device| {
            Conv2dConfig::new([ci, co], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        Detector {
            conv1: conv(self.in_channels, self.ch1, device),
            conv2: conv(self.ch1, self.ch2, device),
            pool: AdaptiveAvgPool2dConfig::new([POOL, POOL]).init(),
            activation: Relu::new(),
            head: LinearConfig::new(self.ch2 * POOL * POOL, 4).init(device),
        }
    }
}

impl<B: Backend> Detector<B> {
    /// Forward pass: `[batch, channels, h, w]` to normalized boxes
    /// `[batch, 4]` in `[0, 1]` (x0, y0, x1, y1).
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.conv1.forward(images));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.pool.forward(x);
        let [b, c, h, w] = x.dims();
        sigmoid(self.head.forward(x.reshape([b, c * h * w])))
    }
}

/// IoU between two axis-aligned boxes `[x0, y0, x1, y1]` (corners unordered).
pub fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let norm = |q: &[f32; 4]| {
        (
            q[0].min(q[2]),
            q[1].min(q[3]),
            q[0].max(q[2]),
            q[1].max(q[3]),
        )
    };
    let (ax0, ay0, ax1, ay1) = norm(a);
    let (bx0, by0, bx1, by1) = norm(b);
    let iw = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let ih = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = iw * ih;
    let union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// One-call single-object detection search (PRD §10, §20).
///
/// `images[i]` is a `channels*height*width` flat pixel vector; `boxes[i]` is a
/// normalized `[x0, y0, x1, y1]` in `[0, 1]`.
pub struct AutoDetection {
    images: Vec<Vec<f32>>,
    boxes: Vec<[f32; 4]>,
    channels: usize,
    height: usize,
    width: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoDetection {
    /// A new detection search over images and their bounding boxes.
    pub fn new(
        images: Vec<Vec<f32>>,
        boxes: Vec<[f32; 4]>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        AutoDetection {
            images,
            boxes,
            channels,
            height,
            width,
            epochs: 10,
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

    /// Run the search, returning the study (maximizing mean box IoU).
    pub fn fit(self) -> CoreResult<Study> {
        let pixels = self.channels * self.height * self.width;
        if self.images.is_empty()
            || self.images.len() != self.boxes.len()
            || self.images[0].len() != pixels
        {
            return Err(Error::Objective(
                "detection images/boxes must be non-empty and correctly shaped".into(),
            ));
        }

        let space = SearchSpace::new()
            .add("ch1", Distribution::int(4, 16))
            .add("ch2", Distribution::int(8, 32))
            .add("lr", Distribution::log_float(1e-3, 1e-2));

        let mut study = Study::builder(space)
            .name("auto-detection")
            .maximize("iou")
            .sampler(TpeSampler::new("iou", Direction::Maximize, self.seed))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.images.len(), self.val_fraction, self.seed);
        let dims = (self.channels, self.height, self.width);
        let data = Arc::new((self.images, self.boxes, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective =
            move |p: &ParamSet, sink: &mut dyn ReportSink| -> CoreResult<NamedMetrics> {
                let cfg = DetectorConfig::new(dims.0)
                    .with_ch1(p.int("ch1")? as usize)
                    .with_ch2(p.int("ch2")? as usize);
                let iou = train_and_eval::<TrainBackend>(
                    &cfg,
                    p.float("lr")?,
                    epochs,
                    batch,
                    dims,
                    &data,
                    sink,
                );
                Ok(NamedMetrics::single("iou", iou as f64))
            };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type DetData = (Vec<Vec<f32>>, Vec<[f32; 4]>, Vec<usize>, Vec<usize>);

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
    cfg: &DetectorConfig,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    dims: (usize, usize, usize),
    data: &DetData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (imgs, boxes, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut iou = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(batch_size) {
            let x = image_tensor::<B>(imgs, chunk, dims, &device);
            let y = box_tensor::<B>(boxes, chunk, &device);
            let out = model.forward(x);
            let loss = MseLoss::new().forward(out, y, Reduction::Mean);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        iou = eval_iou::<B::InnerBackend>(&valid, imgs, boxes, val_idx, dims, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("iou", iou as f64));
        if sink.should_stop() {
            break;
        }
    }
    iou
}

fn eval_iou<B: Backend>(
    model: &Detector<B>,
    imgs: &[Vec<f32>],
    boxes: &[[f32; 4]],
    val_idx: &[usize],
    dims: (usize, usize, usize),
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let x = image_tensor::<B>(imgs, val_idx, dims, device);
    let preds: Vec<f32> = model
        .forward(x)
        .flatten::<1>(0, 1)
        .into_data()
        .to_vec()
        .expect("f32");
    let mut sum = 0.0;
    for (k, &i) in val_idx.iter().enumerate() {
        let pb = [
            preds[k * 4],
            preds[k * 4 + 1],
            preds[k * 4 + 2],
            preds[k * 4 + 3],
        ];
        sum += box_iou(&pb, &boxes[i]);
    }
    sum / val_idx.len() as f32 * 100.0
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

fn box_tensor<B: Backend>(boxes: &[[f32; 4]], idx: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let n = idx.len();
    let flat: Vec<f32> = idx.iter().flat_map(|&i| boxes[i]).collect();
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [n, 4]).convert::<B::FloatElem>(),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_iou_is_correct() {
        assert!((box_iou(&[0.0, 0.0, 1.0, 1.0], &[0.0, 0.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!(box_iou(&[0.0, 0.0, 0.5, 0.5], &[0.5, 0.5, 1.0, 1.0]).abs() < 1e-6);
        // Half-overlap: two unit-height boxes sharing half their width.
        let iou = box_iou(&[0.0, 0.0, 0.5, 1.0], &[0.25, 0.0, 0.75, 1.0]);
        assert!((iou - (0.25 / 0.75)).abs() < 1e-6, "iou={iou}");
    }

    /// A bright 3x3 square at a random location; the box is its bounds. The CNN
    /// regresses the location.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<[f32; 4]>) {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut imgs, mut boxes) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let r0 = rng.gen_range(0..6);
            let c0 = rng.gen_range(0..6);
            let mut img = vec![0f32; 64];
            for r in r0..r0 + 3 {
                for c in c0..c0 + 3 {
                    img[r * 8 + c] = 1.0;
                }
            }
            imgs.push(img);
            // Normalized corners in [0,1].
            boxes.push([
                c0 as f32 / 8.0,
                r0 as f32 / 8.0,
                (c0 + 3) as f32 / 8.0,
                (r0 + 3) as f32 / 8.0,
            ]);
        }
        (imgs, boxes)
    }

    #[test]
    fn auto_detection_localizes_a_square() {
        let (imgs, boxes) = synthetic(64, 1);
        let study = AutoDetection::new(imgs, boxes, 1, 8, 8)
            .epochs(8)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let iou = best.final_value("iou").unwrap();
        // Random boxes score ~10% IoU here; clearing 22% shows real localization,
        // with cushion for the noisy, platform-varying IoU value.
        assert!(
            iou > 22.0,
            "best detection IoU was {iou}, expected the detector to localize"
        );
    }
}
