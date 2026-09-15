//! Anomaly detection task adapter (PRD §7, §20).
//!
//! A distance-based (k-NN) anomaly detector: each point's score is the mean
//! distance to its `k` nearest neighbours among the (mostly normal) reference
//! set, and a point is flagged when its score exceeds a threshold. Following the
//! plan (§7), the scoring model (`k`) and the operating threshold are optimized
//! *jointly* — here against a labelled evaluation set, maximizing F1. Like
//! clustering, this is a framework-agnostic `automl_core::Study` objective.

use automl_core::error::{Error, Result};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, SearchSpace, Study, TpeSampler};
use std::sync::Arc;

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// The mean distance from each query point to its `k` nearest neighbours in
/// `reference` — a simple, effective anomaly score (higher is more anomalous).
pub fn knn_anomaly_scores(reference: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<f32> {
    queries
        .iter()
        .map(|q| {
            let mut ds: Vec<f32> = reference.iter().map(|r| dist(q, r)).collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = k.clamp(1, ds.len().max(1));
            if ds.is_empty() {
                0.0
            } else {
                ds[..k].iter().sum::<f32>() / k as f32
            }
        })
        .collect()
}

/// The `q`-quantile of `values` (`q` in `[0, 1]`).
fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((q.clamp(0.0, 1.0)) * (v.len() - 1) as f32).round() as usize;
    v[idx.min(v.len() - 1)]
}

/// F1 score for the positive (anomaly = 1) class.
fn f1(labels: &[i64], predictions: &[i64]) -> f32 {
    let mut tp = 0i32;
    let mut fp = 0i32;
    let mut fn_ = 0i32;
    for (&l, &p) in labels.iter().zip(predictions) {
        match (l, p) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }
    let denom = 2 * tp + fp + fn_;
    if denom == 0 {
        0.0
    } else {
        2.0 * tp as f32 / denom as f32
    }
}

/// One-call anomaly-detector search (PRD §20). Jointly searches the neighbour
/// count `k` and the score threshold (as a quantile), maximizing F1 on a
/// labelled evaluation set.
pub struct AutoAnomaly {
    reference: Vec<Vec<f32>>,
    eval: Vec<Vec<f32>>,
    eval_labels: Vec<i64>,
    max_k: usize,
    trials: u64,
    seed: u64,
}

impl AutoAnomaly {
    /// Build a search from a reference set (assumed mostly normal) and a
    /// labelled evaluation set (`0` normal, `1` anomaly).
    pub fn new(reference: Vec<Vec<f32>>, eval: Vec<Vec<f32>>, eval_labels: Vec<i64>) -> Self {
        AutoAnomaly {
            reference,
            eval,
            eval_labels,
            max_k: 20,
            trials: 20,
            seed: 0,
        }
    }

    /// Largest neighbour count the search may use.
    pub fn max_k(mut self, k: usize) -> Self {
        self.max_k = k.max(1);
        self
    }

    /// Number of search trials.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Seed for the search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (maximizing F1). The best trial's
    /// `k` and `threshold_quantile` parameters define the detector.
    pub fn fit(self) -> Result<Study> {
        if self.reference.is_empty()
            || self.eval.len() != self.eval_labels.len()
            || self.eval.is_empty()
        {
            return Err(Error::Objective(
                "anomaly detector needs a reference set and a labelled eval set".into(),
            ));
        }
        let max_k = self.max_k.min(self.reference.len()).max(1);
        let space = SearchSpace::new()
            .add("k", Distribution::int(1, max_k as i64))
            .add("threshold_quantile", Distribution::float(0.5, 0.999));

        let mut study = Study::builder(space)
            .name("auto-anomaly")
            .maximize("f1")
            .sampler(TpeSampler::new("f1", Direction::Maximize, self.seed))
            .seed(self.seed)
            .build()?;

        let reference = Arc::new(self.reference);
        let eval = Arc::new(self.eval);
        let labels = Arc::new(self.eval_labels);

        let objective = move |p: &ParamSet, _sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let k = p.int("k")? as usize;
            let q = p.float("threshold_quantile")? as f32;
            let scores = knn_anomaly_scores(&reference, &eval, k);
            let threshold = quantile(&scores, q);
            let predictions: Vec<i64> = scores
                .iter()
                .map(|&s| if s > threshold { 1 } else { 0 })
                .collect();
            Ok(NamedMetrics::single("f1", f1(&labels, &predictions) as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn detects_injected_outliers() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        // Reference: a tight normal cluster near the origin.
        let reference: Vec<Vec<f32>> = (0..200)
            .map(|_| vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)])
            .collect();
        // Eval: normals near origin (label 0) + far outliers (label 1).
        let mut eval = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..60 {
            eval.push(vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
            labels.push(0);
        }
        for _ in 0..20 {
            eval.push(vec![rng.gen_range(8.0..12.0), rng.gen_range(8.0..12.0)]);
            labels.push(1);
        }
        let study = AutoAnomaly::new(reference, eval, labels)
            .trials(10)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        // Well-separated outliers should be detected near-perfectly.
        assert!(
            best.final_value("f1").unwrap() > 0.9,
            "F1 {:?}",
            best.final_value("f1")
        );
    }

    #[test]
    fn empty_inputs_error() {
        assert!(AutoAnomaly::new(Vec::new(), Vec::new(), Vec::new())
            .fit()
            .is_err());
    }
}
