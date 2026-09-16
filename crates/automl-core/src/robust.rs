//! Robust aggregation for noisy objectives (roadmap v0.8; PRD §13/§14, §18,
//! §27 "Noisy objectives → replicates, confidence estimates, robust aggregation
//! mode").
//!
//! GAN/diffusion training and RL returns are the canonical *noisy* objectives:
//! the same configuration scored twice gives two different numbers. Optimizing
//! the raw score then chases noise. The mitigation the PRD locks in is
//! **replicated evaluation with a robust aggregate** — evaluate a configuration
//! several times under different seeds and reduce the replicates to a single,
//! outlier-resistant point estimate plus a spread that quantifies confidence.
//!
//! This module is framework-agnostic: [`Aggregator`] reduces a slice of scores,
//! and [`replicate`] wraps any objective closure so a [`crate::study::Study`]
//! optimizes the robust aggregate transparently — the sampler and pruner never
//! know the objective was noisy.

use crate::error::Result;
use crate::metrics::NamedMetrics;
use crate::objective::ReportSink;
use serde::{Deserialize, Serialize};

/// How to reduce a set of replicate scores to one robust point estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Aggregator {
    /// Arithmetic mean — lowest variance when the noise is well-behaved, but
    /// sensitive to outliers (a single diverged GAN run drags it down).
    Mean,
    /// Median — the robust default: unaffected by a minority of outlier runs.
    Median,
    /// Mean after trimming the given fraction from each tail (e.g. `0.2` drops
    /// the top and bottom 20%). A tunable midpoint between mean and median.
    TrimmedMean(f64),
}

impl Aggregator {
    /// Reduce `scores` to a single robust estimate. NaNs are ignored. Returns
    /// `None` only when every score is NaN or the slice is empty.
    pub fn aggregate(&self, scores: &[f64]) -> Option<f64> {
        let mut v: Vec<f64> = scores.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(match *self {
            Aggregator::Mean => v.iter().sum::<f64>() / v.len() as f64,
            Aggregator::Median => median_sorted(&v),
            Aggregator::TrimmedMean(frac) => {
                let frac = frac.clamp(0.0, 0.49);
                let cut = (v.len() as f64 * frac).floor() as usize;
                let kept = &v[cut..v.len() - cut];
                // Trimming can empty a tiny sample; fall back to the median.
                if kept.is_empty() {
                    median_sorted(&v)
                } else {
                    kept.iter().sum::<f64>() / kept.len() as f64
                }
            }
        })
    }

    /// A spread estimate for `scores` — the standard error of the main estimate,
    /// a confidence proxy (smaller is more trustworthy). Uses the standard error
    /// of the mean for [`Aggregator::Mean`] and a robust MAD-based standard error
    /// otherwise. Returns `0.0` for fewer than two finite scores.
    pub fn spread(&self, scores: &[f64]) -> f64 {
        let mut v: Vec<f64> = scores.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.len() < 2 {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len() as f64;
        match self {
            Aggregator::Mean => {
                let mean = v.iter().sum::<f64>() / n;
                let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                (var / n).sqrt()
            }
            _ => {
                // MAD scaled to a standard-deviation estimate, then standard error.
                let med = median_sorted(&v);
                let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
                dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mad = median_sorted(&dev);
                (1.4826 * mad) / n.sqrt()
            }
        }
    }
}

/// Median of an already-sorted, NaN-free, non-empty slice.
fn median_sorted(v: &[f64]) -> f64 {
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Evaluate a noisy objective `n_replicates` times and return the robust
/// aggregate of each metric.
///
/// Replicate `i` receives seed `base_seed + i` — so the caller's objective can
/// derive its own randomness deterministically and each replicate differs. Every
/// replicate's raw metrics are reported to `sink` at steps `1..=n_replicates`
/// (so the pruner still sees a curve and can stop a clearly-bad configuration
/// early), and the returned [`NamedMetrics`] holds the per-metric aggregate plus,
/// for the primary `objective`, a `"{objective}.spread"` confidence estimate.
///
/// `objective` names the metric whose spread is reported; pass the study's
/// optimized metric name.
pub fn replicate<F>(
    aggregator: Aggregator,
    n_replicates: usize,
    base_seed: u64,
    objective: &str,
    sink: &mut dyn ReportSink,
    mut eval: F,
) -> Result<NamedMetrics>
where
    F: FnMut(u64, &mut dyn ReportSink) -> Result<NamedMetrics>,
{
    let n = n_replicates.max(1);
    // Collect each metric's replicate scores in stable (BTree) order.
    let mut collected: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let metrics = eval(base_seed + i as u64, sink)?;
        for (name, value) in metrics.iter() {
            collected.entry(name.to_string()).or_default().push(*value);
        }
        // Report the primary objective's running aggregate as the curve, so a
        // pruner comparing at each step sees a stabilizing estimate.
        if let Some(scores) = collected.get(objective) {
            if let Some(agg) = aggregator.aggregate(scores) {
                let _ = sink.report((i + 1) as u64, NamedMetrics::single(objective, agg));
            }
        }
        if sink.should_stop() {
            break;
        }
    }

    let mut out = NamedMetrics::new();
    for (name, scores) in &collected {
        if let Some(agg) = aggregator.aggregate(scores) {
            out.insert(name.clone(), agg);
        }
    }
    if let Some(scores) = collected.get(objective) {
        out.insert(format!("{objective}.spread"), aggregator.spread(scores));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialId;

    #[test]
    fn median_resists_outliers_mean_does_not() {
        let scores = [1.0, 1.1, 0.9, 1.0, 10.0]; // one diverged run.
        let mean = Aggregator::Mean.aggregate(&scores).unwrap();
        let median = Aggregator::Median.aggregate(&scores).unwrap();
        assert!(mean > 2.0, "mean dragged by the outlier: {mean}");
        assert!(
            (median - 1.0).abs() < 1e-9,
            "median stayed robust: {median}"
        );
    }

    #[test]
    fn trimmed_mean_between_mean_and_median() {
        let scores = [1.0, 2.0, 3.0, 4.0, 100.0];
        let trimmed = Aggregator::TrimmedMean(0.2).aggregate(&scores).unwrap();
        // Drops one from each tail -> mean of [2,3,4] = 3.
        assert!((trimmed - 3.0).abs() < 1e-9, "trimmed mean was {trimmed}");
    }

    #[test]
    fn aggregate_ignores_nan_and_empty() {
        assert_eq!(Aggregator::Median.aggregate(&[]), None);
        assert_eq!(Aggregator::Median.aggregate(&[f64::NAN]), None);
        let m = Aggregator::Mean.aggregate(&[2.0, f64::NAN, 4.0]).unwrap();
        assert!((m - 3.0).abs() < 1e-9);
    }

    #[test]
    fn spread_shrinks_with_agreement() {
        let tight = Aggregator::Mean.spread(&[1.0, 1.01, 0.99, 1.0]);
        let loose = Aggregator::Mean.spread(&[1.0, 5.0, -3.0, 2.0]);
        assert!(tight < loose, "tight {tight} should be < loose {loose}");
        assert_eq!(Aggregator::Median.spread(&[1.0]), 0.0);
    }

    struct VecSink {
        id: TrialId,
        steps: Vec<(u64, f64)>,
    }
    impl ReportSink for VecSink {
        fn trial_id(&self) -> TrialId {
            self.id
        }
        fn report(&mut self, step: u64, metrics: NamedMetrics) -> Result<()> {
            if let Some(v) = metrics.get("reward") {
                self.steps.push((step, v));
            }
            Ok(())
        }
        fn should_stop(&self) -> bool {
            false
        }
    }

    #[test]
    fn replicate_aggregates_and_reports_spread() {
        let mut sink = VecSink {
            id: TrialId(0),
            steps: Vec::new(),
        };
        // A "noisy" objective: reward = seed parity gives 1.0 or 3.0, plus a fixed
        // secondary metric. Median over 4 seeds (1,2,3,4) -> {1,3,1,3} -> 2.0.
        let out = replicate(Aggregator::Median, 4, 1, "reward", &mut sink, |seed, _| {
            let reward = if seed % 2 == 1 { 1.0 } else { 3.0 };
            Ok(NamedMetrics::single("reward", reward).with("cost", 5.0))
        })
        .unwrap();
        assert_eq!(out.get("reward"), Some(2.0));
        assert_eq!(out.get("cost"), Some(5.0));
        assert!(out.get("reward.spread").unwrap() > 0.0);
        // A curve was reported at each replicate step.
        assert_eq!(sink.steps.len(), 4);
        assert_eq!(sink.steps.last().unwrap().0, 4);
    }
}
