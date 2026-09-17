//! Text classification and the `AutoText` API (PRD §12 "NLP: classification …",
//! §20).
//!
//! Documents are tokenized and each token is projected to a fixed-width vector by
//! **feature hashing** (the hashing trick): a deterministic, vocabulary-free
//! front-end that needs no learned embedding table. The resulting per-token
//! sequence is exactly what the recurrent [`AutoSequence`]
//! classifier consumes — so text classification reuses the sequence model over a
//! lexical front-end, the same way audio reuses it over a spectral one. Learned
//! embeddings and attention encoders can layer on later (§12).

use crate::common::split;
use crate::sequence::label_tensor;
use crate::AutoSequence;
use crate::TrainBackend;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;

/// Split text into lowercased alphanumeric tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Hash a token to a bucket in `[0, dim)` (FNV-1a, deterministic across runs).
fn hash_bucket(token: &str, dim: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in token.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % dim as u64) as usize
}

/// Turn a document into a sequence of one-hot hashed-token vectors, truncated or
/// zero-padded to `max_len` frames of width `dim`.
pub fn hashed_sequence(text: &str, dim: usize, max_len: usize) -> Vec<Vec<f32>> {
    let dim = dim.max(1);
    let max_len = max_len.max(1);
    let mut frames: Vec<Vec<f32>> = tokenize(text)
        .into_iter()
        .take(max_len)
        .map(|tok| {
            let mut v = vec![0f32; dim];
            v[hash_bucket(&tok, dim)] = 1.0;
            v
        })
        .collect();
    // Pad short documents so every sequence is the same length.
    while frames.len() < max_len {
        frames.push(vec![0f32; dim]);
    }
    frames
}

/// One-call text classification search (PRD §12, §20).
///
/// Each `texts[i]` is a raw document string; a hashed-token front-end turns it
/// into sequence frames that a recurrent model classifies.
pub struct AutoText {
    texts: Vec<String>,
    labels: Vec<i64>,
    dim: usize,
    max_len: usize,
    num_classes: usize,
    epochs: usize,
    trials: u64,
    seed: u64,
    learned_embedding: bool,
    vocab_size: usize,
}

impl AutoText {
    /// A new text classification search with a default hashing front-end
    /// (256 buckets, 32-token window).
    pub fn new(texts: Vec<String>, labels: Vec<i64>) -> Self {
        AutoText {
            texts,
            labels,
            dim: 256,
            max_len: 32,
            num_classes: 0,
            epochs: 8,
            trials: 10,
            seed: 0,
            learned_embedding: false,
            vocab_size: 2000,
        }
    }

    /// Set the hashing front-end: `dim` buckets and a `max_len`-token window.
    pub fn frontend(mut self, dim: usize, max_len: usize) -> Self {
        self.dim = dim.max(1);
        self.max_len = max_len.max(1);
        self
    }

    /// Use a **learned embedding** table instead of feature hashing: a
    /// vocabulary of the `vocab_size` most frequent tokens is embedded into a
    /// searched-dimension dense space, trained end-to-end with the classifier.
    /// This captures token similarity a fixed hashing front-end cannot.
    pub fn learned_embedding(mut self, vocab_size: usize) -> Self {
        self.learned_embedding = true;
        self.vocab_size = vocab_size.max(2);
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
        if self.texts.is_empty() || self.texts.len() != self.labels.len() {
            return Err(Error::Objective(
                "text documents and labels must match and be non-empty".into(),
            ));
        }
        if self.learned_embedding {
            return self.fit_embedding();
        }

        let sequences: Vec<Vec<Vec<f32>>> = self
            .texts
            .iter()
            .map(|t| hashed_sequence(t, self.dim, self.max_len))
            .collect();

        let mut search = AutoSequence::new(sequences, self.labels)
            .epochs(self.epochs)
            .trials(self.trials)
            .seed(self.seed);
        if self.num_classes > 0 {
            search = search.num_classes(self.num_classes);
        }
        search.fit()
    }

    /// The learned-embedding path: build a vocabulary, encode documents to token
    /// ids, and search an embedding+recurrent classifier trained end-to-end.
    fn fit_embedding(self) -> CoreResult<Study> {
        let num_classes = if self.num_classes == 0 {
            (self.labels.iter().copied().max().unwrap_or(0) + 1) as usize
        } else {
            self.num_classes
        };
        let vocab = build_vocab(&self.texts, self.vocab_size);
        // vocab id 0 is reserved for padding / out-of-vocabulary.
        let vocab_n = vocab.len() + 1;
        let max_len = self.max_len;
        let ids: Vec<Vec<i64>> = self
            .texts
            .iter()
            .map(|t| encode(t, &vocab, max_len))
            .collect();

        let space = SearchSpace::new()
            .add("embed_dim", Distribution::int(8, 48))
            .add("lr", Distribution::log_float(1e-3, 2e-2))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-text-embed")
            .maximize("accuracy")
            .sampler(TpeSampler::new("accuracy", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(ids.len(), 0.2, self.seed);
        let data = Arc::new((ids, self.labels, train, val));
        let epochs = self.epochs;

        let objective = move |p: &ParamSet,
                              sink: &mut dyn ReportSink|
              -> CoreResult<NamedMetrics> {
            let cfg = EmbTextConfig::new(vocab_n, num_classes)
                .with_embed_dim(p.int("embed_dim")? as usize)
                .with_dropout(p.float("dropout")?);
            let acc =
                train_and_eval::<TrainBackend>(&cfg, p.float("lr")?, epochs, max_len, &data, sink);
            Ok(NamedMetrics::single("accuracy", acc as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

// ---------------------------- learned embedding ------------------------------

/// Build a vocabulary of the `max_vocab` most frequent tokens, mapped to ids
/// `1..=n`; id `0` is reserved for padding and out-of-vocabulary tokens. Ties
/// break by token text for determinism.
fn build_vocab(texts: &[String], max_vocab: usize) -> HashMap<String, i64> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in texts {
        for tok in tokenize(t) {
            *counts.entry(tok).or_default() += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(max_vocab.saturating_sub(1))
        .enumerate()
        .map(|(i, (tok, _))| (tok, i as i64 + 1))
        .collect()
}

/// Encode a document to a fixed-length vector of token ids (`0` = pad/OOV).
fn encode(text: &str, vocab: &HashMap<String, i64>, max_len: usize) -> Vec<i64> {
    let mut ids: Vec<i64> = tokenize(text)
        .into_iter()
        .take(max_len)
        .map(|tok| *vocab.get(&tok).unwrap_or(&0))
        .collect();
    ids.resize(max_len, 0);
    ids
}

/// A learned-embedding text classifier: token embeddings, **masked mean-pooling**
/// over the non-padding positions (fastText-style, so trailing padding does not
/// dilute the representation), then a linear head.
#[derive(Module, Debug)]
pub struct EmbText<B: Backend> {
    embedding: Embedding<B>,
    dropout: burn::nn::Dropout,
    head: Linear<B>,
}

/// Configuration for [`EmbText`].
#[derive(Config, Debug)]
pub struct EmbTextConfig {
    /// Vocabulary size (including the reserved pad/OOV id 0).
    pub vocab: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Embedding dimension.
    #[config(default = 16)]
    pub embed_dim: usize,
    /// Dropout on the pooled representation.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl EmbTextConfig {
    /// Initialize the model on `device`. The `hidden` field is retained for API
    /// stability but the pooled classifier projects the embedding directly.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbText<B> {
        EmbText {
            embedding: EmbeddingConfig::new(self.vocab, self.embed_dim).init(device),
            dropout: burn::nn::DropoutConfig::new(self.dropout).init(),
            head: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> EmbText<B> {
    /// Forward pass: token ids `[batch, seq]` to class logits `[batch, classes]`.
    /// Padding tokens (id 0) are masked out of the mean pool.
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let embedded = self.embedding.forward(ids.clone()); // [b, t, embed]
                                                            // Mask: 1.0 for real tokens, 0.0 for padding (id 0).
        let mask = ids.greater_elem(0).float().unsqueeze_dim::<3>(2); // [b, t, 1]
        let summed = (embedded * mask.clone()).sum_dim(1); // [b, 1, embed]
        let count = mask.sum_dim(1).clamp_min(1.0); // [b, 1, 1]
        let [b, _, e] = summed.dims();
        let pooled = (summed / count).reshape([b, e]); // [b, embed]
        self.head.forward(self.dropout.forward(pooled))
    }
}

type IdData = (Vec<Vec<i64>>, Vec<i64>, Vec<usize>, Vec<usize>);

fn id_tensor<B: Backend>(
    ids: &[Vec<i64>],
    idx: &[usize],
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let flat: Vec<i64> = idx.iter().flat_map(|&i| ids[i].iter().copied()).collect();
    Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat, [idx.len(), max_len]).convert::<B::IntElem>(),
        device,
    )
}

fn train_and_eval<B: AutodiffBackend>(
    cfg: &EmbTextConfig,
    lr: f64,
    epochs: usize,
    max_len: usize,
    data: &IdData,
    sink: &mut dyn ReportSink,
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut model = cfg.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let (ids, labels, train_idx, val_idx) = data;
    let mut order = train_idx.clone();
    let mut acc = 0.0;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(32) {
            let x = id_tensor::<B>(ids, chunk, max_len, &device);
            let y = label_tensor::<B>(labels, chunk, &device);
            let out = model.forward(x);
            let loss = CrossEntropyLoss::new(None, &out.device()).forward(out, y);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let valid = model.valid();
        acc = accuracy::<B::InnerBackend>(&valid, ids, labels, val_idx, max_len, &device);
        let _ = sink.report(epoch as u64, NamedMetrics::single("accuracy", acc as f64));
        if sink.should_stop() {
            break;
        }
    }
    acc
}

fn accuracy<B: Backend>(
    model: &EmbText<B>,
    ids: &[Vec<i64>],
    labels: &[i64],
    val_idx: &[usize],
    max_len: usize,
    device: &B::Device,
) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut correct: i64 = 0;
    for chunk in val_idx.chunks(32) {
        let x = id_tensor::<B>(ids, chunk, max_len, device);
        let y = label_tensor::<B>(labels, chunk, device);
        let pred = model.forward(x).argmax(1).squeeze_dim::<1>(1);
        correct += pred.equal(y).int().sum().into_scalar().elem::<i64>();
    }
    correct as f32 / val_idx.len() as f32 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_and_hashes_deterministically() {
        assert_eq!(tokenize("Hello, world! 42"), vec!["hello", "world", "42"]);
        // Hashing is stable across calls.
        assert_eq!(hash_bucket("cat", 128), hash_bucket("cat", 128));
        let seq = hashed_sequence("good movie", 64, 5);
        assert_eq!(seq.len(), 5); // padded to max_len
        assert_eq!(seq[0].len(), 64);
        assert_eq!(seq[0].iter().filter(|&&v| v == 1.0).count(), 1); // one-hot
        assert!(seq[4].iter().all(|&v| v == 0.0)); // padding frame
    }

    /// Two topics with disjoint vocabularies plus shared filler words.
    fn synthetic(n: usize, seed: u64) -> (Vec<String>, Vec<i64>) {
        use rand::{seq::SliceRandom, Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let pos = ["great", "excellent", "love", "wonderful", "brilliant"];
        let neg = ["terrible", "awful", "hate", "boring", "worst"];
        let filler = ["the", "a", "this", "was", "it", "and"];
        let (mut texts, mut labels) = (Vec::new(), Vec::new());
        for i in 0..n {
            let positive = i % 2 == 0;
            let topic = if positive { &pos } else { &neg };
            let mut words: Vec<&str> = Vec::new();
            for _ in 0..rng.gen_range(6..12) {
                if rng.gen_bool(0.5) {
                    words.push(topic[rng.gen_range(0..topic.len())]);
                } else {
                    words.push(filler[rng.gen_range(0..filler.len())]);
                }
            }
            words.shuffle(&mut rng);
            texts.push(words.join(" "));
            labels.push(if positive { 0 } else { 1 });
        }
        (texts, labels)
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn auto_text_classifies_sentiment_like_topics() {
        let (texts, labels) = synthetic(140, 1);
        let study = AutoText::new(texts, labels)
            .frontend(128, 16)
            .num_classes(2)
            .epochs(6)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(acc > 66.0, "best text accuracy was {acc}");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn auto_text_learned_embedding_classifies() {
        let (texts, labels) = synthetic(140, 2);
        let study = AutoText::new(texts, labels)
            .learned_embedding(64)
            .num_classes(2)
            .epochs(6)
            .trials(2)
            .seed(2)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(acc > 66.0, "best learned-embedding text accuracy was {acc}");
    }

    #[test]
    fn vocab_builds_and_encodes() {
        let texts = vec!["cat dog cat".to_string(), "dog bird".to_string()];
        let vocab = build_vocab(&texts, 10);
        // "cat" and "dog" are the frequent tokens; ids are >= 1, 0 is reserved.
        assert!(vocab.values().all(|&v| v >= 1));
        let enc = encode("cat unknownword", &vocab, 4);
        assert_eq!(enc.len(), 4); // padded
        assert!(enc[0] >= 1); // "cat" is in vocab
        assert_eq!(enc[1], 0); // OOV -> 0
        assert_eq!(enc[3], 0); // padding
    }

    #[test]
    fn empty_input_errors() {
        assert!(AutoText::new(Vec::new(), Vec::new()).fit().is_err());
    }
}
