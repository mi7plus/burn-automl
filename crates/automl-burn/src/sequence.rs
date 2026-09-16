//! Recurrent sequence models (LSTM / GRU) and the `AutoSequence` high-level API
//! (PRD §8, §20).
//!
//! Sequence classification/regression needs a recurrent (or attention) model
//! over variable content; this module provides an LSTM/GRU classifier whose
//! cell type, width and regularization are searchable hyperparameters, and an
//! `AutoSequence` builder that searches them over fixed-length multivariate
//! sequences. The last recurrent hidden state feeds a linear classification
//! head.

use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use crate::TrainBackend;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Dropout, DropoutConfig, GruConfig, Linear, LinearConfig, Lstm, LstmConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// A recurrent sequence classifier: an LSTM *or* GRU encoder followed by a
/// linear head over the final hidden state. Exactly one cell is populated,
/// selected by [`RnnConfig::cell`].
#[derive(Module, Debug)]
pub struct RnnClassifier<B: Backend> {
    lstm: Option<Lstm<B>>,
    gru: Option<burn::nn::Gru<B>>,
    dropout: Dropout,
    head: Linear<B>,
}

/// Configuration for [`RnnClassifier`].
#[derive(Config, Debug)]
pub struct RnnConfig {
    /// Number of input features per timestep.
    pub input_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Recurrent hidden-state width.
    #[config(default = 32)]
    pub hidden_size: usize,
    /// Cell type: `"lstm"` or `"gru"`.
    #[config(default = "String::from(\"lstm\")")]
    pub cell: String,
    /// Dropout applied to the final hidden state.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl RnnConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RnnClassifier<B> {
        let (lstm, gru) = if self.cell == "gru" {
            (
                None,
                Some(GruConfig::new(self.input_dim, self.hidden_size, true).init(device)),
            )
        } else {
            (
                Some(LstmConfig::new(self.input_dim, self.hidden_size, true).init(device)),
                None,
            )
        };
        RnnClassifier {
            lstm,
            gru,
            dropout: DropoutConfig::new(self.dropout).init(),
            head: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> RnnClassifier<B> {
    /// Forward pass: `[batch, seq_len, features]` to `[batch, num_classes]`.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let last_hidden = if let Some(lstm) = &self.lstm {
            // LstmState.hidden is the final hidden state `[batch, hidden]`.
            let (_outputs, state) = lstm.forward(input, None);
            state.hidden
        } else if let Some(gru) = &self.gru {
            // GRU returns `[batch, seq, hidden]`; take the last timestep.
            let outputs = gru.forward(input, None);
            let [b, s, h] = outputs.dims();
            outputs.narrow(1, s - 1, 1).reshape([b, h])
        } else {
            unreachable!("RnnClassifier always has exactly one cell")
        };
        self.head.forward(self.dropout.forward(last_hidden))
    }
}

/// One-call hyperparameter search for sequence classification (PRD §20).
///
/// Sequences are fixed-length multivariate: `sequences[i]` is
/// `[seq_len][features]` and `labels[i]` its class.
pub struct AutoSequence {
    sequences: Vec<Vec<Vec<f32>>>,
    labels: Vec<i64>,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoSequence {
    /// A new sequence-classification search over the given data.
    pub fn new(sequences: Vec<Vec<Vec<f32>>>, labels: Vec<i64>) -> Self {
        AutoSequence {
            sequences,
            labels,
            num_classes: 0,
            epochs: 8,
            batch_size: 32,
            trials: 12,
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

        let space = SearchSpace::new()
            .add("cell", Distribution::categorical(["lstm", "gru"]))
            .add("hidden_size", Distribution::int(8, 64))
            .add("lr", Distribution::log_float(1e-3, 1e-1))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-sequence")
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
            let cfg = RnnConfig::new(feat_dim, num_classes)
                .with_hidden_size(p.int("hidden_size")? as usize)
                .with_cell(p.categorical("cell")?.to_string())
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

fn split(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

fn train_and_eval<B: AutodiffBackend>(
    cfg: &RnnConfig,
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
    model: &RnnClassifier<B>,
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

pub(crate) fn seq_tensor<B: Backend>(
    seqs: &[Vec<Vec<f32>>],
    idx: &[usize],
    device: &B::Device,
) -> Tensor<B, 3> {
    let n = idx.len();
    let s = seqs[idx[0]].len();
    let f = seqs[idx[0]].first().map_or(0, |v| v.len());
    let flat: Vec<f32> = idx
        .iter()
        .flat_map(|&i| seqs[i].iter().flat_map(|step| step.iter().copied()))
        .collect();
    Tensor::<B, 3>::from_data(
        TensorData::new(flat, [n, s, f]).convert::<B::FloatElem>(),
        device,
    )
}

pub(crate) fn label_tensor<B: Backend>(
    labels: &[i64],
    idx: &[usize],
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let vals: Vec<i64> = idx.iter().map(|&i| labels[i]).collect();
    Tensor::<B, 1, Int>::from_data(
        TensorData::new(vals, [idx.len()]).convert::<B::IntElem>(),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two classes over length-12 univariate sequences: class 0 rises, class 1
    /// falls. A recurrent model separates them easily.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<Vec<f32>>>, Vec<i64>) {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut seqs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let rising = i % 2 == 0;
            let seq: Vec<Vec<f32>> = (0..12)
                .map(|t| {
                    let base = if rising { t as f32 } else { 12.0 - t as f32 };
                    vec![base + rng.gen_range(-0.5..0.5)]
                })
                .collect();
            seqs.push(seq);
            labels.push(if rising { 0 } else { 1 });
        }
        (seqs, labels)
    }

    #[test]
    fn auto_sequence_classifies_rising_vs_falling() {
        let (seqs, labels) = synthetic(200, 1);
        let study = AutoSequence::new(seqs, labels)
            .num_classes(2)
            .epochs(6)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(
            acc > 66.0,
            "best sequence accuracy was {acc}, expected the RNN to learn"
        );
    }

    #[test]
    fn empty_data_errors() {
        assert!(AutoSequence::new(Vec::new(), Vec::new()).fit().is_err());
    }
}
