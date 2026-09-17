//! Transformer encoder sequence classifier and the `AutoTransformer` API
//! (PRD §9, §20).
//!
//! Searches a Transformer encoder over sequences: the number of attention
//! heads, per-head width, depth, and feed-forward ratio. The plan calls out the
//! `d_model % heads == 0` compatibility constraint (§5); rather than reject
//! invalid combinations after sampling, this parameterizes the model by
//! `heads` and `head_dim` and sets `d_model = heads * head_dim`, so every
//! sampled configuration is valid by construction.
//!
//! Input features are linearly embedded to `d_model`, given sinusoidal
//! positional encodings, passed through the encoder, mean-pooled over time, and
//! classified with a linear head.

use crate::common::split;
use crate::sequence::{label_tensor, seq_tensor};
use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// A Transformer-encoder sequence classifier.
#[derive(Module, Debug)]
pub struct TransformerClassifier<B: Backend> {
    embed: Linear<B>,
    encoder: TransformerEncoder<B>,
    dropout: Dropout,
    head: Linear<B>,
}

/// Configuration for [`TransformerClassifier`]. `d_model` is `n_heads *
/// head_dim`, keeping it divisible by the head count by construction (§5).
#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// Number of input features per timestep.
    pub input_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Number of attention heads.
    #[config(default = 2)]
    pub n_heads: usize,
    /// Per-head width; `d_model = n_heads * head_dim`.
    #[config(default = 16)]
    pub head_dim: usize,
    /// Number of encoder layers.
    #[config(default = 2)]
    pub n_layers: usize,
    /// Feed-forward width as a multiple of `d_model`.
    #[config(default = 2)]
    pub ff_ratio: usize,
    /// Dropout rate.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl TransformerConfig {
    /// The model dimension, `n_heads * head_dim`.
    pub fn d_model(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerClassifier<B> {
        let d_model = self.d_model();
        let d_ff = d_model * self.ff_ratio;
        TransformerClassifier {
            embed: LinearConfig::new(self.input_dim, d_model).init(device),
            encoder: TransformerEncoderConfig::new(d_model, d_ff, self.n_heads, self.n_layers)
                .with_dropout(self.dropout)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            head: LinearConfig::new(d_model, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> TransformerClassifier<B> {
    /// Forward pass: `[batch, seq_len, features]` to `[batch, num_classes]`.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.embed.forward(input); // [b, s, d_model]
        let [b, s, d] = x.dims();
        let pos = sinusoidal::<B>(s, d, &x.device()).unsqueeze::<3>(); // [1, s, d]
        let x = x + pos;
        let encoded = self.encoder.forward(TransformerEncoderInput::new(x)); // [b, s, d]
        let pooled = encoded.mean_dim(1).reshape([b, d]); // mean over time
        self.head.forward(self.dropout.forward(pooled))
    }
}

/// Sinusoidal positional encoding `[seq_len, d_model]`.
fn sinusoidal<B: Backend>(seq: usize, d: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0f32; seq * d];
    for (pos, row) in data.chunks_mut(d).enumerate() {
        for (i, cell) in row.iter_mut().enumerate() {
            let angle = pos as f32 / 10000f32.powf((2 * (i / 2)) as f32 / d as f32);
            *cell = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::<B, 2>::from_data(
        TensorData::new(data, [seq, d]).convert::<B::FloatElem>(),
        device,
    )
}

/// One-call Transformer architecture/training search for sequence
/// classification (PRD §9, §20).
pub struct AutoTransformer {
    sequences: Vec<Vec<Vec<f32>>>,
    labels: Vec<i64>,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoTransformer {
    /// A new Transformer search over the given fixed-length sequences.
    pub fn new(sequences: Vec<Vec<Vec<f32>>>, labels: Vec<i64>) -> Self {
        AutoTransformer {
            sequences,
            labels,
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
        if self.sequences.is_empty() || self.sequences.len() != self.labels.len() {
            return Err(Error::Objective(
                "sequence data and labels must match and be non-empty".into(),
            ));
        }
        let feat_dim = self.sequences[0].first().map_or(0, |s| s.len());
        if feat_dim == 0 {
            return Err(Error::Objective(
                "sequences must have at least one feature".into(),
            ));
        }
        let num_classes = if self.num_classes == 0 {
            (self.labels.iter().copied().max().unwrap_or(0) + 1) as usize
        } else {
            self.num_classes
        };

        // heads and head_dim are searched independently; d_model = heads*head_dim
        // is divisible by heads by construction (§5 compatibility constraint).
        let space = SearchSpace::new()
            .add("n_heads", Distribution::categorical(["1", "2", "4"]))
            .add("head_dim", Distribution::int(8, 32))
            .add("n_layers", Distribution::int(1, 3))
            .add("ff_ratio", Distribution::int(2, 4))
            .add("lr", Distribution::log_float(1e-4, 1e-2))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-transformer")
            .maximize("accuracy")
            .sampler(TpeSampler::new("accuracy", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.sequences.len(), self.val_fraction, self.seed);
        let data = Arc::new((self.sequences, self.labels, train, val));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective = move |p: &ParamSet,
                              sink: &mut dyn ReportSink|
              -> CoreResult<NamedMetrics> {
            let cfg = TransformerConfig::new(feat_dim, num_classes)
                .with_n_heads(p.categorical("n_heads")?.parse().unwrap_or(2))
                .with_head_dim(p.int("head_dim")? as usize)
                .with_n_layers(p.int("n_layers")? as usize)
                .with_ff_ratio(p.int("ff_ratio")? as usize)
                .with_dropout(p.float("dropout")?);
            let acc =
                train_and_eval::<TrainBackend>(&cfg, p.float("lr")?, epochs, batch, &data, sink);
            Ok(NamedMetrics::single("accuracy", acc as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type SeqData = (Vec<Vec<Vec<f32>>>, Vec<i64>, Vec<usize>, Vec<usize>);

fn train_and_eval<B: AutodiffBackend>(
    cfg: &TransformerConfig,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    data: &SeqData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (seqs, labels, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut acc = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(batch_size) {
            let x = seq_tensor::<B>(seqs, chunk, &device);
            let y = label_tensor::<B>(labels, chunk, &device);
            let out = model.forward(x);
            let loss = CrossEntropyLoss::new(None, &out.device()).forward(out, y);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        acc = accuracy::<B::InnerBackend>(&valid, seqs, labels, val_idx, batch_size, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("accuracy", acc as f64));
        if sink.should_stop() {
            break;
        }
    }
    acc
}

fn accuracy<B: Backend>(
    model: &TransformerClassifier<B>,
    seqs: &[Vec<Vec<f32>>],
    labels: &[i64],
    val_idx: &[usize],
    batch_size: usize,
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut correct: i64 = 0;
    for chunk in val_idx.chunks(batch_size) {
        let x = seq_tensor::<B>(seqs, chunk, device);
        let y = label_tensor::<B>(labels, chunk, device);
        let pred = model.forward(x).argmax(1).squeeze_dim(1);
        correct += pred.equal(y).int().sum().into_scalar().elem::<i64>();
    }
    correct as f32 / val_idx.len() as f32 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_model_is_divisible_by_heads_by_construction() {
        let cfg = TransformerConfig::new(4, 2)
            .with_n_heads(4)
            .with_head_dim(16);
        assert_eq!(cfg.d_model(), 64);
        assert_eq!(cfg.d_model() % cfg.n_heads, 0);
    }

    /// Two classes over length-12 univariate sequences distinguished by where a
    /// spike occurs (early vs late) — an order-dependent task positional
    /// encoding + attention can solve.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<Vec<f32>>>, Vec<i64>) {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut seqs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let early = i % 2 == 0;
            let spike = if early { 2 } else { 9 };
            let seq: Vec<Vec<f32>> = (0..12)
                .map(|t| {
                    vec![if t == spike {
                        5.0
                    } else {
                        rng.gen_range(-0.3..0.3)
                    }]
                })
                .collect();
            seqs.push(seq);
            labels.push(if early { 0 } else { 1 });
        }
        (seqs, labels)
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn auto_transformer_classifies_positional_task() {
        let (seqs, labels) = synthetic(200, 2);
        let study = AutoTransformer::new(seqs, labels)
            .num_classes(2)
            .epochs(6)
            .trials(2)
            .seed(2)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(acc > 63.0, "best transformer accuracy was {acc}");
    }
}
