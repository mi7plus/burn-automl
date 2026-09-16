//! Text classification and the `AutoText` API (PRD §12 "NLP: classification …",
//! §20).
//!
//! Documents are tokenized and each token is projected to a fixed-width vector by
//! **feature hashing** (the hashing trick): a deterministic, vocabulary-free
//! front-end that needs no learned embedding table. The resulting per-token
//! sequence is exactly what the recurrent [`AutoSequence`](crate::AutoSequence)
//! classifier consumes — so text classification reuses the sequence model over a
//! lexical front-end, the same way audio reuses it over a spectral one. Learned
//! embeddings and attention encoders can layer on later (§12).

use crate::AutoSequence;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::prelude::Study;

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
        }
    }

    /// Set the hashing front-end: `dim` buckets and a `max_len`-token window.
    pub fn frontend(mut self, dim: usize, max_len: usize) -> Self {
        self.dim = dim.max(1);
        self.max_len = max_len.max(1);
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
    fn empty_input_errors() {
        assert!(AutoText::new(Vec::new(), Vec::new()).fit().is_err());
    }
}
