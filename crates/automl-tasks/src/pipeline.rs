//! Executable tabular pipeline search — the `AutoPipeline` API (roadmap v0.9;
//! PRD §16, §20).
//!
//! This is the framework-agnostic materialization of [`automl_core::pipeline`]:
//! it searches a real preprocessing → model pipeline for tabular classification.
//! The preprocessing stage chooses standardize / normalize / none and the model
//! stage chooses a nearest-centroid or k-NN classifier (with a searched `k`),
//! all expressed through the core [`PipelineSpace`] — so the sampler and pruner
//! never know a pipeline is being searched, exactly as §16 requires.

use automl_core::error::{Error, Result};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::pipeline::{Component, PipelineSpace, Stage};
use automl_core::prelude::{Distribution, Study, TpeSampler};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

/// One-call tabular pipeline search (PRD §16, §20): jointly searches a
/// preprocessing transform and a classifier, maximizing validation accuracy.
pub struct AutoPipeline {
    features: Vec<Vec<f32>>,
    labels: Vec<i64>,
    trials: u64,
    val_fraction: f64,
    seed: u64,
}

impl AutoPipeline {
    /// A new pipeline search over row-major `features` and integer `labels`.
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<i64>) -> Self {
        AutoPipeline {
            features,
            labels,
            trials: 16,
            val_fraction: 0.25,
            seed: 0,
        }
    }

    /// Number of pipeline configurations to try.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Seed for the split and search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// The pipeline search space: a preprocessing stage and a model stage.
    pub fn space() -> PipelineSpace {
        PipelineSpace::new()
            .stage(
                Stage::new("preprocess")
                    .component(Component::new("standardize"))
                    .component(Component::new("normalize"))
                    .component(Component::new("none")),
            )
            .stage(
                Stage::new("model")
                    .component(Component::new("centroid"))
                    .component(Component::new("knn").param("k", Distribution::int(1, 15))),
            )
    }

    /// Run the search, returning the study (maximizing validation accuracy).
    pub fn fit(self) -> Result<Study> {
        if self.features.is_empty()
            || self.features.len() != self.labels.len()
            || self
                .features
                .iter()
                .any(|r| r.len() != self.features[0].len())
        {
            return Err(Error::Objective(
                "pipeline data must be non-empty, rectangular, and match labels".into(),
            ));
        }
        let pipeline = Self::space();
        let space = pipeline.to_search_space();

        let mut study = Study::builder(space)
            .name("auto-pipeline")
            .maximize("accuracy")
            .sampler(TpeSampler::new("accuracy", Direction::Maximize, self.seed))
            .seed(self.seed)
            .build()?;

        let (train, val) = split(self.features.len(), self.val_fraction, self.seed);
        let data = Arc::new((self.features, self.labels, train, val));

        let objective = move |p: &ParamSet, _sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let plan = pipeline.decode(p)?;
            let (rows, labels, train_idx, val_idx) = &*data;

            // Fit the chosen preprocessing on the training rows, apply to both.
            let pre = plan.component("preprocess").unwrap_or("none");
            let transform = Preprocess::fit(pre, rows, train_idx);
            let train_x: Vec<Vec<f32>> = train_idx
                .iter()
                .map(|&i| transform.apply(&rows[i]))
                .collect();
            let val_x: Vec<Vec<f32>> = val_idx.iter().map(|&i| transform.apply(&rows[i])).collect();
            let train_y: Vec<i64> = train_idx.iter().map(|&i| labels[i]).collect();
            let val_y: Vec<i64> = val_idx.iter().map(|&i| labels[i]).collect();

            // Fit + score the chosen model.
            let model = plan.choice("model");
            let acc = match model.map(|c| c.component.as_str()) {
                Some("knn") => {
                    let k = model.and_then(|c| c.params.int("k").ok()).unwrap_or(3) as usize;
                    knn_accuracy(&train_x, &train_y, &val_x, &val_y, k)
                }
                _ => centroid_accuracy(&train_x, &train_y, &val_x, &val_y),
            };
            Ok(NamedMetrics::single("accuracy", acc))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

/// A fitted preprocessing transform over per-feature statistics.
enum Preprocess {
    Standardize { mean: Vec<f32>, std: Vec<f32> },
    Normalize { min: Vec<f32>, span: Vec<f32> },
    None,
}

impl Preprocess {
    fn fit(kind: &str, rows: &[Vec<f32>], idx: &[usize]) -> Preprocess {
        let d = rows[0].len();
        match kind {
            "standardize" => {
                let n = idx.len().max(1) as f32;
                let mut mean = vec![0f32; d];
                for &i in idx {
                    for (m, v) in mean.iter_mut().zip(&rows[i]) {
                        *m += v;
                    }
                }
                mean.iter_mut().for_each(|m| *m /= n);
                let mut var = vec![0f32; d];
                for &i in idx {
                    for (s, (v, m)) in var.iter_mut().zip(rows[i].iter().zip(&mean)) {
                        *s += (v - m).powi(2);
                    }
                }
                let std: Vec<f32> = var.iter().map(|s| (s / n).sqrt().max(1e-6)).collect();
                Preprocess::Standardize { mean, std }
            }
            "normalize" => {
                let mut min = vec![f32::INFINITY; d];
                let mut max = vec![f32::NEG_INFINITY; d];
                for &i in idx {
                    for (j, v) in rows[i].iter().enumerate() {
                        min[j] = min[j].min(*v);
                        max[j] = max[j].max(*v);
                    }
                }
                let span: Vec<f32> = min
                    .iter()
                    .zip(&max)
                    .map(|(lo, hi)| (hi - lo).max(1e-6))
                    .collect();
                Preprocess::Normalize { min, span }
            }
            _ => Preprocess::None,
        }
    }

    fn apply(&self, row: &[f32]) -> Vec<f32> {
        match self {
            Preprocess::Standardize { mean, std } => row
                .iter()
                .zip(mean.iter().zip(std))
                .map(|(v, (m, s))| (v - m) / s)
                .collect(),
            Preprocess::Normalize { min, span } => row
                .iter()
                .zip(min.iter().zip(span))
                .map(|(v, (lo, sp))| (v - lo) / sp)
                .collect(),
            Preprocess::None => row.to_vec(),
        }
    }
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Nearest-class-centroid classification accuracy.
fn centroid_accuracy(
    train_x: &[Vec<f32>],
    train_y: &[i64],
    val_x: &[Vec<f32>],
    val_y: &[i64],
) -> f64 {
    use std::collections::BTreeMap;
    let d = train_x.first().map(|r| r.len()).unwrap_or(0);
    let mut sums: BTreeMap<i64, (Vec<f32>, usize)> = BTreeMap::new();
    for (x, y) in train_x.iter().zip(train_y) {
        let e = sums.entry(*y).or_insert_with(|| (vec![0f32; d], 0));
        for (s, v) in e.0.iter_mut().zip(x) {
            *s += v;
        }
        e.1 += 1;
    }
    let centroids: Vec<(i64, Vec<f32>)> = sums
        .into_iter()
        .map(|(y, (mut s, n))| {
            s.iter_mut().for_each(|v| *v /= n.max(1) as f32);
            (y, s)
        })
        .collect();
    if centroids.is_empty() || val_x.is_empty() {
        return 0.0;
    }
    let correct = val_x
        .iter()
        .zip(val_y)
        .filter(|(x, y)| {
            let pred = centroids
                .iter()
                .min_by(|a, b| dist2(x, &a.1).partial_cmp(&dist2(x, &b.1)).unwrap())
                .map(|(lbl, _)| *lbl);
            pred == Some(**y)
        })
        .count();
    correct as f64 / val_x.len() as f64 * 100.0
}

/// k-nearest-neighbour classification accuracy.
fn knn_accuracy(
    train_x: &[Vec<f32>],
    train_y: &[i64],
    val_x: &[Vec<f32>],
    val_y: &[i64],
    k: usize,
) -> f64 {
    use std::collections::BTreeMap;
    if train_x.is_empty() || val_x.is_empty() {
        return 0.0;
    }
    let k = k.clamp(1, train_x.len());
    let correct = val_x
        .iter()
        .zip(val_y)
        .filter(|(x, y)| {
            let mut dists: Vec<(f32, i64)> = train_x
                .iter()
                .zip(train_y)
                .map(|(t, ty)| (dist2(x, t), *ty))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut votes: BTreeMap<i64, usize> = BTreeMap::new();
            for (_, lbl) in dists.iter().take(k) {
                *votes.entry(*lbl).or_default() += 1;
            }
            let pred = votes
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(lbl, _)| lbl);
            pred == Some(**y)
        })
        .count();
    correct as f64 / val_x.len() as f64 * 100.0
}

fn split(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Two classes separated along a *small-scale* feature, buried next to a
    /// large-scale noise feature. Raw distances are dominated by the noise, so
    /// only a pipeline that standardizes/normalizes recovers the signal — the
    /// search must choose preprocessing, not just a model.
    fn scale_trap(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<i64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let label = rng.gen_range(0..2);
            let signal = if label == 0 {
                rng.gen_range(-1.0..0.0)
            } else {
                rng.gen_range(0.0..1.0)
            };
            let noise = rng.gen_range(-500.0..500.0); // huge scale, label-independent
            x.push(vec![noise, signal]);
            y.push(label);
        }
        (x, y)
    }

    #[test]
    fn auto_pipeline_finds_a_preprocessing_that_works() {
        let (x, y) = scale_trap(200, 1);
        let study = AutoPipeline::new(x, y).trials(16).seed(1).fit().unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        // Only a scaling preprocessing recovers the small-scale signal.
        assert!(acc > 80.0, "best pipeline accuracy was {acc}");
    }

    #[test]
    fn bad_shape_errors() {
        assert!(
            AutoPipeline::new(vec![vec![0.0; 2], vec![0.0; 3]], vec![0, 1])
                .fit()
                .is_err()
        );
    }
}
