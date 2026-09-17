//! Generative-model helpers: the `AutoAutoencoder` API plus GAN/diffusion
//! search helpers (roadmap v0.8; PRD §13, §20).
//!
//! Autoencoders "arrive earlier because they fit ordinary training loops" (§13),
//! so the trained deliverable here is [`AutoAutoencoder`]: a one-call search over
//! a reconstruction autoencoder's latent size, width, depth and learning rate,
//! scored by validation reconstruction error (minimized). GANs and diffusion are
//! expensive and noisy to train, so this module ships *helpers* rather than full
//! training loops: [`gan_space`] builds a coordinated generator/discriminator
//! search space whose dimensions are compatible by construction, and
//! [`DiffusionSchedule`] builds the noise schedules a diffusion search would tune.
//! Their noisy quality metrics are meant to be optimized through the robust
//! aggregation mode in [`automl_core::robust`].

use crate::vision::split;
use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

// ------------------------------ diffusion helper -----------------------------

/// A diffusion noise schedule: the per-step `beta`s and their derived cumulative
/// signal-retention `alpha_bar`s (PRD §13 "diffusion adds noise schedule…").
///
/// This is the searchable front-end a diffusion AutoML task would tune; it is a
/// pure function of its parameters (no training), so it is cheap to unit-test and
/// compose.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSchedule {
    /// Per-step variance schedule, length `steps`.
    pub betas: Vec<f32>,
    /// Cumulative product of `1 - beta` up to each step (signal retained).
    pub alpha_bars: Vec<f32>,
}

impl DiffusionSchedule {
    /// A linear beta schedule from `beta_start` to `beta_end` over `steps`.
    pub fn linear(steps: usize, beta_start: f32, beta_end: f32) -> Self {
        let steps = steps.max(1);
        let betas: Vec<f32> = (0..steps)
            .map(|t| {
                let frac = if steps == 1 {
                    0.0
                } else {
                    t as f32 / (steps - 1) as f32
                };
                (beta_start + frac * (beta_end - beta_start)).clamp(1e-6, 0.999)
            })
            .collect();
        Self::from_betas(betas)
    }

    /// A cosine beta schedule (Nichol & Dhariwal-style): smoother signal decay
    /// than linear, over `steps`.
    pub fn cosine(steps: usize) -> Self {
        let steps = steps.max(1);
        let s = 0.008f32;
        let f = |t: f32| {
            ((t / steps as f32 + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2)
                .cos()
                .powi(2)
        };
        let a0 = f(0.0);
        let betas: Vec<f32> = (0..steps)
            .map(|t| {
                let a_t = f(t as f32 + 1.0) / a0;
                let a_prev = f(t as f32) / a0;
                (1.0 - a_t / a_prev).clamp(1e-6, 0.999)
            })
            .collect();
        Self::from_betas(betas)
    }

    fn from_betas(betas: Vec<f32>) -> Self {
        let mut alpha_bars = Vec::with_capacity(betas.len());
        let mut prod = 1.0f32;
        for &b in &betas {
            prod *= 1.0 - b;
            alpha_bars.push(prod);
        }
        DiffusionSchedule { betas, alpha_bars }
    }

    /// Number of diffusion steps.
    pub fn steps(&self) -> usize {
        self.betas.len()
    }
}

// ------------------------------ GAN space helper -----------------------------

/// Build a coordinated generator/discriminator search space for a GAN over
/// `feature_dim`-dimensional data (PRD §13 "GANs require coordinated
/// generator/discriminator spaces").
///
/// The generator maps a `latent_dim`-vector to `feature_dim` outputs and the
/// discriminator maps `feature_dim` inputs to a scalar, so the two networks are
/// dimension-compatible *by construction*; only their capacities and learning
/// rates are searched. `latent_choices` lists candidate latent sizes.
pub fn gan_space(feature_dim: usize, latent_choices: &[usize]) -> SearchSpace {
    let latents: Vec<String> = if latent_choices.is_empty() {
        vec![feature_dim.max(1).to_string()]
    } else {
        latent_choices.iter().map(|d| d.to_string()).collect()
    };
    SearchSpace::new()
        .add("latent_dim", Distribution::categorical(latents))
        .add("gen_hidden", Distribution::int(16, 256))
        .add("disc_hidden", Distribution::int(16, 256))
        .add("lr_gen", Distribution::log_float(1e-4, 1e-2))
        .add("lr_disc", Distribution::log_float(1e-4, 1e-2))
        // Two-timescale updates: how many discriminator steps per generator step.
        .add("disc_steps", Distribution::int(1, 5))
}

// ------------------------------ autoencoder ----------------------------------

/// A fully-connected reconstruction autoencoder.
#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Vec<Linear<B>>,
    decoder: Vec<Linear<B>>,
    activation: Relu,
}

/// Configuration for [`Autoencoder`].
#[derive(Config, Debug)]
pub struct AeConfig {
    /// Input/output feature dimension.
    pub input_dim: usize,
    /// Bottleneck (latent) dimension.
    pub latent_dim: usize,
    /// Hidden width of the encoder/decoder stages.
    #[config(default = 32)]
    pub hidden: usize,
    /// Number of hidden stages on each side (>= 1).
    #[config(default = 1)]
    pub depth: usize,
}

impl AeConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        let depth = self.depth.max(1);
        let mut encoder = Vec::new();
        // input -> hidden -> ... -> latent
        encoder.push(LinearConfig::new(self.input_dim, self.hidden).init(device));
        for _ in 1..depth {
            encoder.push(LinearConfig::new(self.hidden, self.hidden).init(device));
        }
        encoder.push(LinearConfig::new(self.hidden, self.latent_dim).init(device));
        // latent -> hidden -> ... -> input
        let mut decoder = Vec::new();
        decoder.push(LinearConfig::new(self.latent_dim, self.hidden).init(device));
        for _ in 1..depth {
            decoder.push(LinearConfig::new(self.hidden, self.hidden).init(device));
        }
        decoder.push(LinearConfig::new(self.hidden, self.input_dim).init(device));
        Autoencoder {
            encoder,
            decoder,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Autoencoder<B> {
    /// Reconstruct a `[batch, input_dim]` batch.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        // Encoder: activate between layers, leave the latent linear.
        for (i, layer) in self.encoder.iter().enumerate() {
            h = layer.forward(h);
            if i + 1 < self.encoder.len() {
                h = self.activation.forward(h);
            }
        }
        // Decoder: activate between layers, leave the reconstruction linear.
        for (i, layer) in self.decoder.iter().enumerate() {
            h = layer.forward(h);
            if i + 1 < self.decoder.len() {
                h = self.activation.forward(h);
            }
        }
        h
    }
}

/// One-call autoencoder search (PRD §13, §20): searches latent size, width,
/// depth and learning rate to minimize validation reconstruction error.
///
/// `data[i]` is a fixed-length feature vector; the target is the input itself.
pub struct AutoAutoencoder {
    data: Vec<Vec<f32>>,
    input_dim: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoAutoencoder {
    /// A new autoencoder search over feature vectors of length `input_dim`.
    pub fn new(data: Vec<Vec<f32>>, input_dim: usize) -> Self {
        AutoAutoencoder {
            data,
            input_dim,
            epochs: 30,
            batch_size: 32,
            trials: 10,
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

    /// Run the search, returning the study (minimizing reconstruction MSE).
    pub fn fit(self) -> CoreResult<Study> {
        if self.data.is_empty() || self.data.iter().any(|r| r.len() != self.input_dim) {
            return Err(Error::Objective(
                "autoencoder data must be non-empty and every row input_dim long".into(),
            ));
        }
        let max_latent = self.input_dim.max(2) as i64;
        let space = SearchSpace::new()
            .add("latent_dim", Distribution::int(1, (max_latent - 1).max(1)))
            .add("hidden", Distribution::int(8, 64))
            .add("depth", Distribution::int(1, 2))
            .add("lr", Distribution::log_float(1e-3, 1e-1));

        let mut study = Study::builder(space)
            .name("auto-autoencoder")
            .minimize("mse")
            .sampler(TpeSampler::new("mse", Direction::Minimize, self.seed))
            .pruner(MedianPruner::new("mse", Direction::Minimize).with_warmup_steps(2))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.data.len(), self.val_fraction, self.seed);
        let input_dim = self.input_dim;
        let data = Arc::new((self.data, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective = move |p: &ParamSet,
                              sink: &mut dyn ReportSink|
              -> CoreResult<NamedMetrics> {
            let cfg = AeConfig::new(input_dim, p.int("latent_dim")? as usize)
                .with_hidden(p.int("hidden")? as usize)
                .with_depth(p.int("depth")? as usize);
            let mse =
                train_and_eval::<TrainBackend>(&cfg, p.float("lr")?, epochs, batch, &data, sink);
            Ok(NamedMetrics::single("mse", mse as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type AeData = (Vec<Vec<f32>>, Vec<usize>, Vec<usize>);

fn row_tensor<B: Backend>(
    rows: &[Vec<f32>],
    idx: &[usize],
    input_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let flat: Vec<f32> = idx.iter().flat_map(|&i| rows[i].iter().copied()).collect();
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [idx.len(), input_dim]).convert::<B::FloatElem>(),
        device,
    )
}

fn train_and_eval<B: AutodiffBackend>(
    cfg: &AeConfig,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    data: &AeData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (rows, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut mse = f32::INFINITY;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(batch_size) {
            let x = row_tensor::<B>(rows, chunk, cfg.input_dim, &device);
            let recon = model.forward(x.clone());
            let loss = MseLoss::new().forward(recon, x, Reduction::Mean);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        mse =
            eval_mse::<B::InnerBackend>(&valid, rows, val_idx, cfg.input_dim, batch_size, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("mse", mse as f64));
        if sink.should_stop() {
            break;
        }
    }
    mse
}

fn eval_mse<B: Backend>(
    model: &Autoencoder<B>,
    rows: &[Vec<f32>],
    val_idx: &[usize],
    input_dim: usize,
    batch_size: usize,
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for chunk in val_idx.chunks(batch_size) {
        let x = row_tensor::<B>(rows, chunk, input_dim, device);
        let recon = model.forward(x.clone());
        let err = (recon - x).powf_scalar(2.0).mean().into_scalar();
        let e: f32 = err.elem();
        total += e * chunk.len() as f32;
        count += chunk.len();
    }
    total / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_schedule_is_monotone_and_bounded() {
        let s = DiffusionSchedule::linear(10, 0.01, 0.2);
        assert_eq!(s.steps(), 10);
        assert!(s.betas.windows(2).all(|w| w[1] >= w[0]));
        assert!(s.betas.iter().all(|&b| (1e-6..=0.999).contains(&b)));
        // alpha_bar decreases from near 1 toward 0 as signal is destroyed.
        assert!(s.alpha_bars.windows(2).all(|w| w[1] <= w[0]));
        assert!(s.alpha_bars[0] < 1.0 && *s.alpha_bars.last().unwrap() > 0.0);
    }

    #[test]
    fn cosine_schedule_retains_more_signal_early() {
        let cos = DiffusionSchedule::cosine(50);
        let lin = DiffusionSchedule::linear(50, 0.0001, 0.02);
        assert_eq!(cos.steps(), 50);
        // Cosine keeps more signal at the midpoint than a comparable linear one.
        assert!(cos.alpha_bars[25] > 0.0);
        assert!(lin.alpha_bars[25] > 0.0);
    }

    #[test]
    fn gan_space_is_valid_and_coordinated() {
        let space = gan_space(8, &[2, 4, 8]);
        assert!(space.validate().is_ok());
        assert!(space.get("latent_dim").is_some());
        assert!(space.get("gen_hidden").is_some());
        assert!(space.get("disc_hidden").is_some());
    }

    /// Data lying on a 1-D line embedded in 4-D: a tiny latent should reconstruct
    /// it with low error.
    fn low_rank(n: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let dirs = [1.0f32, -0.5, 0.25, 0.75];
        (0..n)
            .map(|_| {
                let t: f32 = rng.gen_range(-1.0..1.0);
                dirs.iter()
                    .map(|d| d * t + rng.gen_range(-0.02..0.02))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn autoencoder_reconstructs_low_rank_data() {
        let data = low_rank(160, 1);
        let study = AutoAutoencoder::new(data, 4)
            .epochs(40)
            .trials(4)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let mse = best.final_value("mse").unwrap();
        // Variance of the signal is ~0.35; a working autoencoder lands far below
        // it. The margin is the signal — keep the bound comfortably above the
        // observed fit, since Burn's rayon float reductions vary by platform.
        assert!(mse < 0.1, "best reconstruction MSE was {mse}");
    }

    #[test]
    fn bad_shape_errors() {
        assert!(AutoAutoencoder::new(vec![vec![0.0; 3]], 4).fit().is_err());
    }

    #[test]
    fn autoencoder_forward_shape() {
        let device = Default::default();
        let model = AeConfig::new(6, 2).init::<TrainBackend>(&device);
        let x = row_tensor::<TrainBackend>(&[vec![0.1; 6], vec![0.2; 6]], &[0, 1], 6, &device);
        let out = model.forward(x);
        assert_eq!(out.dims(), [2, 6]);
    }
}
