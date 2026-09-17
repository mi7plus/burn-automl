//! # automl-burn
//!
//! The Burn deep-learning execution adapter for `burn-automl` (PRD §29 item 9,
//! §27 "Burn API evolution"). Burn is the flagship execution adapter, but it
//! stays a *thin* layer: this crate depends on `automl-core` and `burn`, and
//! turns a Burn training run into an [`automl_core::objective::Objective`] so
//! the generic engine can optimize it like any other workload.
//!
//! The bridge is deliberately small — the load-bearing invariant from the core
//! (§4.2) is that anything producing named metrics is optimizable. A training
//! loop here:
//!
//! 1. reports validation accuracy to the [`ReportSink`](automl_core::objective::ReportSink)
//!    at each epoch, feeding the pruner's learning-curve view, and
//! 2. checks [`ReportSink::should_stop`](automl_core::objective::ReportSink::should_stop)
//!    after each report, so a pruned trial stops early instead of training to completion.
//!
//! The CPU-only `ndarray` backend is used so the build needs no GPU or system
//! libraries; accelerator backends are a feature-flag change, not a code change.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio;
pub(crate) mod common;
pub mod detection;
pub mod evaluation;
pub mod generative;
pub mod mnist;
pub mod nas;
pub mod segmentation;
pub mod seq2seq;
pub mod sequence;
pub mod tabular;
pub mod text;
pub mod timeseries;
pub mod transformer;
pub mod video;
pub mod vision;
pub use audio::AutoAudio;
pub use detection::AutoDetection;
pub use evaluation::Evaluation;
pub use generative::{gan_space, AutoAutoencoder, Autoencoder, DiffusionSchedule};
pub use mnist::{train_and_eval, MnistBatch, MnistBatcher, MnistMlpObjective, TrainConfig};
pub use nas::{AutoNas, NasCnn};
pub use segmentation::AutoSegmentation;
pub use seq2seq::{AutoSeq2Seq, Seq2Seq};
pub use sequence::AutoSequence;
pub use tabular::{AutoClassifier, AutoRegressor, AutoResult};
pub use text::AutoText;
pub use timeseries::{AutoForecaster, TimeSplit};
pub use transformer::AutoTransformer;
pub use video::AutoVideo;
pub use vision::AutoVision;

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;

/// The training backend: autodiff over the pure-Rust ndarray backend by default,
/// or over an accelerator backend when a GPU feature is enabled — a device
/// change, not a code change (§18 device-aware execution). The whole crate is
/// written against this alias, so every `Auto*` adapter runs on whichever backend
/// is selected. Enable at most one GPU feature.
#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "metal")))]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;

/// Autodiff over Burn's cross-platform WGPU backend (`wgpu` feature).
#[cfg(feature = "wgpu")]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

/// Autodiff over Burn's CUDA backend for NVIDIA GPUs (`cuda` feature; needs the
/// CUDA toolkit).
#[cfg(feature = "cuda")]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Cuda>;

/// Autodiff over Burn's Metal backend for Apple GPUs (`metal` feature; needs
/// macOS).
#[cfg(feature = "metal")]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Metal>;

// ------------------------------- model ---------------------------------------

/// A configurable multi-layer perceptron whose width, depth and dropout are the
/// searchable hyperparameters.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    input: Linear<B>,
    hidden: Vec<Linear<B>>,
    output: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

/// Architecture configuration for [`Mlp`].
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Flattened input dimension (784 for 28x28 MNIST).
    #[config(default = 784)]
    pub input_dim: usize,
    /// Number of output classes.
    #[config(default = 10)]
    pub num_classes: usize,
    /// Width of every hidden layer.
    pub hidden_size: usize,
    /// Number of hidden layers between the input and output projections.
    #[config(default = 1)]
    pub num_hidden_layers: usize,
    /// Dropout probability applied after each activation.
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl MlpConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let input = LinearConfig::new(self.input_dim, self.hidden_size).init(device);
        let hidden = (0..self.num_hidden_layers)
            .map(|_| LinearConfig::new(self.hidden_size, self.hidden_size).init(device))
            .collect();
        let output = LinearConfig::new(self.hidden_size, self.num_classes).init(device);
        Mlp {
            input,
            hidden,
            output,
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    /// Forward pass: `[batch, height, width]` images to `[batch, num_classes]`
    /// logits (flattens the image and delegates to [`Mlp::forward_flat`]).
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, height, width] = images.dims();
        self.forward_flat(images.reshape([batch, height * width]))
    }

    /// Forward pass over already-flat features `[batch, features]` to
    /// `[batch, outputs]` logits. Used by the tabular `Auto*` helpers.
    pub fn forward_flat(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.activation.forward(self.input.forward(x));
        x = self.dropout.forward(x);
        for layer in &self.hidden {
            x = self.activation.forward(layer.forward(x));
            x = self.dropout.forward(x);
        }
        self.output.forward(x)
    }
}
