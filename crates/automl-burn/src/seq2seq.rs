//! Sequence-to-sequence (encoder–decoder) search — the `AutoSeq2Seq` API
//! (PRD §9 "encoder-decoder systems", §20).
//!
//! An LSTM **encoder** compresses the source sequence into a final hidden state
//! that initializes an LSTM **decoder**; a linear head projects each decoder step
//! to vocabulary logits. Training is teacher-forced in a single parallel pass;
//! evaluation is free-running greedy decoding (the decoder consumes its own
//! predictions), so the reported token accuracy reflects real generation, not a
//! gold-prefix shortcut. The encoder and decoder hidden width, learning rate and
//! dropout are searched. Token ids are one-hot encoded, so no embedding table is
//! needed; a learned embedding is a natural later addition (§9).

use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// The reserved beginning-of-sequence / padding token id (`0`); real symbols use
/// ids `1..vocab`.
pub const BOS: usize = 0;

/// An LSTM encoder–decoder with a linear vocabulary head.
#[derive(Module, Debug)]
pub struct Seq2Seq<B: Backend> {
    encoder: Lstm<B>,
    decoder: Lstm<B>,
    head: Linear<B>,
}

/// Configuration for [`Seq2Seq`].
#[derive(Config, Debug)]
pub struct Seq2SeqConfig {
    /// Vocabulary size (including [`BOS`]).
    pub vocab: usize,
    /// Recurrent hidden width shared by encoder and decoder.
    #[config(default = 32)]
    pub hidden: usize,
}

impl Seq2SeqConfig {
    /// Initialize the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Seq2Seq<B> {
        Seq2Seq {
            encoder: LstmConfig::new(self.vocab, self.hidden, true).init(device),
            decoder: LstmConfig::new(self.vocab, self.hidden, true).init(device),
            head: LinearConfig::new(self.hidden, self.vocab).init(device),
        }
    }
}

impl<B: Backend> Seq2Seq<B> {
    /// Teacher-forced forward pass: encode `src` `[batch, s_len, vocab]`, decode
    /// `tgt_in` `[batch, t_len, vocab]` from the encoder state, and return
    /// per-step logits `[batch, t_len, vocab]`.
    pub fn forward(&self, src: Tensor<B, 3>, tgt_in: Tensor<B, 3>) -> Tensor<B, 3> {
        let (_, enc_state) = self.encoder.forward(src, None);
        let (dec_out, _) = self.decoder.forward(tgt_in, Some(enc_state));
        let [b, t, h] = dec_out.dims();
        let flat = self.head.forward(dec_out.reshape([b * t, h]));
        let [_, v] = flat.dims();
        flat.reshape([b, t, v])
    }

    /// Free-running greedy decode: encode `src`, then autoregressively emit
    /// `out_len` tokens starting from [`BOS`]. Returns predicted ids
    /// `[batch, out_len]`.
    pub fn generate(&self, src: Tensor<B, 3>, out_len: usize, vocab: usize) -> Vec<Vec<usize>> {
        let device = src.device();
        let batch = src.dims()[0];
        let (_, mut state) = self.encoder.forward(src, None);
        // First decoder input: BOS for every sequence.
        let mut token = vec![BOS; batch];
        let mut preds = vec![Vec::with_capacity(out_len); batch];
        for _ in 0..out_len {
            let step = onehot_step::<B>(&token, vocab, &device);
            let (out, new_state) = self.decoder.forward(step, Some(state));
            state = new_state;
            let [b, _t, h] = out.dims();
            let logits = self.head.forward(out.reshape([b, h])); // [b, vocab]
            let next = logits.argmax(1).squeeze_dim::<1>(1);
            let data: Vec<i64> = next.into_data().iter::<i64>().collect();
            for (i, id) in data.into_iter().enumerate() {
                token[i] = id as usize;
                preds[i].push(id as usize);
            }
        }
        preds
    }
}

/// One-call sequence-to-sequence search (PRD §9, §20). Each `sources[i]` maps to
/// `targets[i]`; token ids are in `1..vocab` with `0` reserved for [`BOS`].
pub struct AutoSeq2Seq {
    sources: Vec<Vec<usize>>,
    targets: Vec<Vec<usize>>,
    vocab: usize,
    epochs: usize,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoSeq2Seq {
    /// A new seq2seq search over `(source, target)` id sequences with the given
    /// `vocab` size (including [`BOS`]).
    pub fn new(sources: Vec<Vec<usize>>, targets: Vec<Vec<usize>>, vocab: usize) -> Self {
        AutoSeq2Seq {
            sources,
            targets,
            vocab: vocab.max(2),
            epochs: 15,
            trials: 6,
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

    /// Run the search, returning the study (maximizing validation token accuracy).
    pub fn fit(self) -> CoreResult<Study> {
        if self.sources.is_empty() || self.sources.len() != self.targets.len() {
            return Err(Error::Objective(
                "seq2seq sources and targets must match and be non-empty".into(),
            ));
        }
        if self
            .sources
            .iter()
            .chain(&self.targets)
            .flatten()
            .any(|&t| t >= self.vocab)
        {
            return Err(Error::Objective("a token id is >= vocab".into()));
        }

        let space = SearchSpace::new()
            .add("hidden", Distribution::int(16, 48))
            .add("lr", Distribution::log_float(1e-3, 2e-2));

        let mut study = Study::builder(space)
            .name("auto-seq2seq")
            .maximize("token_acc")
            .sampler(TpeSampler::new("token_acc", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("token_acc", Direction::Maximize).with_warmup_steps(2))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.sources.len(), self.val_fraction, self.seed);
        let vocab = self.vocab;
        let data = Arc::new((self.sources, self.targets, train, val));
        let epochs = self.epochs;

        let objective =
            move |p: &ParamSet, sink: &mut dyn ReportSink| -> CoreResult<NamedMetrics> {
                let cfg = Seq2SeqConfig::new(vocab).with_hidden(p.int("hidden")? as usize);
                let acc = train_and_eval::<TrainBackend>(&cfg, p.float("lr")?, epochs, &data, sink);
                Ok(NamedMetrics::single("token_acc", acc as f64))
            };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

type SeqData = (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>);

/// One-hot `[batch, 1, vocab]` tensor for a single decoding step.
fn onehot_step<B: Backend>(tokens: &[usize], vocab: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut flat = vec![0f32; tokens.len() * vocab];
    for (i, &t) in tokens.iter().enumerate() {
        flat[i * vocab + t.min(vocab - 1)] = 1.0;
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(flat, [tokens.len(), 1, vocab]).convert::<B::FloatElem>(),
        device,
    )
}

/// One-hot `[n, len, vocab]` tensor for a batch of id sequences.
fn onehot_seq<B: Backend>(
    seqs: &[Vec<usize>],
    idx: &[usize],
    len: usize,
    vocab: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut flat = vec![0f32; idx.len() * len * vocab];
    for (row, &i) in idx.iter().enumerate() {
        for (t, &tok) in seqs[i].iter().take(len).enumerate() {
            flat[(row * len + t) * vocab + tok.min(vocab - 1)] = 1.0;
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(flat, [idx.len(), len, vocab]).convert::<B::FloatElem>(),
        device,
    )
}

fn train_and_eval<B: AutodiffBackend>(
    cfg: &Seq2SeqConfig,
    lr: f64,
    epochs: usize,
    data: &SeqData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (src, tgt, train_idx, val_idx) = data;
    let vocab = cfg.vocab;
    let t_len = tgt.first().map(|t| t.len()).unwrap_or(0);
    let mut order = train_idx.clone();
    let mut acc = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(32) {
            let s_len = chunk.iter().map(|&i| src[i].len()).max().unwrap_or(1);
            let src_t = onehot_seq::<B>(src, chunk, s_len, vocab, &device);
            // Decoder input: BOS followed by the target shifted right.
            let dec_in: Vec<Vec<usize>> = chunk
                .iter()
                .map(|&i| {
                    let mut v = vec![BOS];
                    v.extend(tgt[i].iter().take(t_len.saturating_sub(1)).copied());
                    v
                })
                .collect();
            let dec_idx: Vec<usize> = (0..chunk.len()).collect();
            let dec_t = onehot_seq::<B>(&dec_in, &dec_idx, t_len, vocab, &device);

            let logits = model.forward(src_t, dec_t); // [b, t_len, vocab]
            let [b, t, v] = logits.dims();
            let targets = target_tensor::<B>(tgt, chunk, t_len, &device); // [b*t]
            let loss =
                CrossEntropyLoss::new(None, &device).forward(logits.reshape([b * t, v]), targets);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        acc = token_accuracy::<B::InnerBackend>(&valid, src, tgt, val_idx, t_len, vocab, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("token_acc", acc as f64));
        if sink.should_stop() {
            break;
        }
    }
    acc
}

fn target_tensor<B: Backend>(
    tgt: &[Vec<usize>],
    idx: &[usize],
    len: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let mut flat = Vec::with_capacity(idx.len() * len);
    for &i in idx {
        for t in 0..len {
            flat.push(*tgt[i].get(t).unwrap_or(&BOS) as i64);
        }
    }
    Tensor::<B, 1, Int>::from_data(
        TensorData::new(flat, [idx.len() * len]).convert::<B::IntElem>(),
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn token_accuracy<B: Backend>(
    model: &Seq2Seq<B>,
    src: &[Vec<usize>],
    tgt: &[Vec<usize>],
    val_idx: &[usize],
    t_len: usize,
    vocab: usize,
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    let mut total = 0usize;
    for chunk in val_idx.chunks(32) {
        let s_len = chunk.iter().map(|&i| src[i].len()).max().unwrap_or(1);
        let src_t = onehot_seq::<B>(src, chunk, s_len, vocab, device);
        let preds = model.generate(src_t, t_len, vocab);
        for (row, &i) in chunk.iter().enumerate() {
            for t in 0..t_len {
                let gold = *tgt[i].get(t).unwrap_or(&BOS);
                if preds[row].get(t) == Some(&gold) {
                    correct += 1;
                }
                total += 1;
            }
        }
    }
    correct as f32 / total.max(1) as f32 * 100.0
}

fn split(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reverse task: target is the source reversed. Symbols are `1..=S`.
    fn reverse_task(
        n: usize,
        s_len: usize,
        symbols: usize,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut src, mut tgt) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let s: Vec<usize> = (0..s_len).map(|_| rng.gen_range(1..=symbols)).collect();
            let mut r = s.clone();
            r.reverse();
            src.push(s);
            tgt.push(r);
        }
        (src, tgt)
    }

    #[test]
    fn auto_seq2seq_learns_to_reverse() {
        // Symbols 1..=5 (vocab 6 incl. BOS), length-4 sequences.
        let (src, tgt) = reverse_task(320, 4, 5, 1);
        let study = AutoSeq2Seq::new(src, tgt, 6)
            .epochs(24)
            .trials(3)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("token_acc").unwrap();
        // Chance token accuracy is 20% (1/5); the encoder-decoder learns to
        // reverse well above it. The margin, not the exact figure, is the signal.
        assert!(acc > 45.0, "best seq2seq token accuracy was {acc}");
    }

    #[test]
    fn rejects_out_of_vocab_and_empty() {
        assert!(AutoSeq2Seq::new(vec![vec![9]], vec![vec![1]], 3)
            .fit()
            .is_err());
        assert!(AutoSeq2Seq::new(Vec::new(), Vec::new(), 3).fit().is_err());
    }
}
