//! Cross-validation schemes for the tabular helpers (PRD §6).
//!
//! The plan calls for holdout, K-fold, stratified and grouped evaluation, "all
//! routed through a shared EvaluationAdapter trait so the sampler never has to
//! know which scheme produced a metric." Here that shared seam is
//! [`Evaluation`] plus [`Evaluation::folds`]: it turns a dataset size (and, for
//! stratification, the labels) into a list of `(train, validation)` index
//! partitions. The `Auto*` objective runs each fold and reports a single
//! aggregated metric, so pruning and sampling stay oblivious to the scheme.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeMap;

/// How a trial's configuration is evaluated on the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Evaluation {
    /// A single train/validation split holding out `val_fraction` of the data.
    Holdout {
        /// Fraction of rows used for validation.
        val_fraction: f64,
    },
    /// `k`-fold cross-validation: each fold validates on one of `k` disjoint
    /// partitions and trains on the rest.
    KFold {
        /// Number of folds.
        k: usize,
    },
    /// Stratified `k`-fold: like [`Evaluation::KFold`], but each class is spread
    /// as evenly as possible across folds (classification only; falls back to
    /// plain K-fold when no labels are available).
    StratifiedKFold {
        /// Number of folds.
        k: usize,
    },
}

impl Default for Evaluation {
    fn default() -> Self {
        Evaluation::Holdout { val_fraction: 0.2 }
    }
}

impl Evaluation {
    /// Whether this scheme produces a single fold (holdout).
    pub fn is_holdout(&self) -> bool {
        matches!(self, Evaluation::Holdout { .. })
    }

    /// Generate `(train_indices, validation_indices)` partitions for `n` rows.
    ///
    /// `labels` is used only by [`Evaluation::StratifiedKFold`]; pass `None` for
    /// regression (stratified then behaves as plain K-fold). The split is
    /// deterministic given `seed`.
    pub fn folds(
        &self,
        n: usize,
        labels: Option<&[i64]>,
        seed: u64,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        match *self {
            Evaluation::Holdout { val_fraction } => vec![holdout(n, val_fraction, seed)],
            Evaluation::KFold { k } => kfold(n, k, seed),
            Evaluation::StratifiedKFold { k } => match labels {
                Some(labels) => stratified_kfold(labels, k, seed),
                None => kfold(n, k, seed),
            },
        }
    }
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    idx
}

/// One holdout split: the last `val_fraction` of the shuffled order validates.
fn holdout(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx = shuffled(n, seed);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

/// Assign `k` validation buckets over the shuffled indices, each fold training
/// on the complement of its bucket.
fn kfold(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let k = k.clamp(2, n.max(2));
    let order = shuffled(n, seed);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &row) in order.iter().enumerate() {
        buckets[i % k].push(row);
    }
    folds_from_buckets(buckets, n)
}

/// Stratified `k`-fold: distribute each label's rows round-robin across buckets
/// so class proportions are preserved in every fold.
fn stratified_kfold(labels: &[i64], k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = labels.len();
    let k = k.clamp(2, n.max(2));

    // Group row indices by label, shuffling within each group for fairness.
    let mut by_label: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_label.entry(l).or_default().push(i);
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); k];
    for group in by_label.values_mut() {
        group.shuffle(&mut rng);
        for (i, &row) in group.iter().enumerate() {
            buckets[i % k].push(row);
        }
    }
    folds_from_buckets(buckets, n)
}

/// Turn validation buckets into `(train, val)` folds, dropping empty buckets.
fn folds_from_buckets(buckets: Vec<Vec<usize>>, n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    buckets
        .into_iter()
        .filter(|b| !b.is_empty())
        .map(|val| {
            let val_set: std::collections::HashSet<usize> = val.iter().copied().collect();
            let train: Vec<usize> = (0..n).filter(|i| !val_set.contains(i)).collect();
            (train, val)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_is_a_single_fold() {
        let folds = Evaluation::Holdout { val_fraction: 0.25 }.folds(100, None, 1);
        assert_eq!(folds.len(), 1);
        let (train, val) = &folds[0];
        assert_eq!(train.len() + val.len(), 100);
        assert_eq!(val.len(), 25);
    }

    #[test]
    fn kfold_partitions_cover_all_rows_disjointly() {
        let k = 5;
        let folds = Evaluation::KFold { k }.folds(50, None, 2);
        assert_eq!(folds.len(), k);
        // Every row appears in exactly one validation set.
        let mut seen = [0u32; 50];
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 50);
            for &v in val {
                seen[v] += 1;
            }
            // train and val are disjoint.
            let vs: std::collections::HashSet<_> = val.iter().collect();
            assert!(train.iter().all(|t| !vs.contains(t)));
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each row validated exactly once"
        );
    }

    #[test]
    fn stratified_kfold_balances_classes() {
        // 40 rows: 20 class 0, 20 class 1.
        let labels: Vec<i64> = (0..40).map(|i| (i % 2) as i64).collect();
        let folds = Evaluation::StratifiedKFold { k: 4 }.folds(40, Some(&labels), 3);
        assert_eq!(folds.len(), 4);
        for (_, val) in &folds {
            let ones = val.iter().filter(|&&i| labels[i] == 1).count();
            let zeros = val.iter().filter(|&&i| labels[i] == 0).count();
            // Each fold's validation set should be class-balanced (5 vs 5).
            assert_eq!(
                ones, zeros,
                "fold not balanced: {ones} ones vs {zeros} zeros"
            );
        }
    }

    #[test]
    fn stratified_without_labels_falls_back_to_kfold() {
        let folds = Evaluation::StratifiedKFold { k: 3 }.folds(30, None, 4);
        assert_eq!(folds.len(), 3);
    }
}
