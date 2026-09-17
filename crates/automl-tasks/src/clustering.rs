//! Clustering task adapter: K-means with internal-metric optimization
//! (PRD §7, §20 `AutoCluster`).
//!
//! Clustering is optimized against an internal quality metric — here the mean
//! silhouette — with the number of clusters (and the random restart) as the
//! searched hyperparameters. This is a *framework-agnostic* task: it drives an
//! `automl_core::Study` over a pure-Rust objective with no deep-learning
//! dependency, demonstrating the load-bearing invariant that anything producing
//! a named metric is optimizable (§4.2).

use automl_core::error::{Error, Result};
use automl_core::metrics::NamedMetrics;
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, GridSampler, SearchSpace, Study};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

/// The result of clustering a dataset.
#[derive(Debug, Clone)]
pub struct Clustering {
    /// Cluster index per input point.
    pub assignments: Vec<usize>,
    /// Cluster centroids.
    pub centroids: Vec<Vec<f32>>,
    /// Sum of squared distances to the nearest centroid (inertia).
    pub inertia: f32,
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// K-means clustering with k-means++ initialization. Deterministic given `seed`.
pub fn kmeans(points: &[Vec<f32>], k: usize, seed: u64, max_iter: usize) -> Clustering {
    let n = points.len();
    let dim = points.first().map_or(0, |p| p.len());
    let k = k.clamp(1, n.max(1));
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // k-means++ seeding.
    let mut centroids: Vec<Vec<f32>> = vec![points[rng.gen_range(0..n)].clone()];
    while centroids.len() < k {
        let d2: Vec<f32> = points
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| dist2(p, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let sum: f32 = d2.iter().sum();
        let pick = if sum <= 0.0 {
            rng.gen_range(0..n)
        } else {
            let mut r = rng.gen::<f32>() * sum;
            let mut idx = n - 1;
            for (i, &d) in d2.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centroids.push(points[pick].clone());
    }

    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let best = (0..centroids.len())
                .min_by(|&a, &b| {
                    dist2(p, &centroids[a])
                        .partial_cmp(&dist2(p, &centroids[b]))
                        .unwrap()
                })
                .unwrap();
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c][j] += p[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let inertia = points
        .iter()
        .enumerate()
        .map(|(i, p)| dist2(p, &centroids[assignments[i]]))
        .sum();
    Clustering {
        assignments,
        centroids,
        inertia,
    }
}

/// Mean silhouette coefficient in `[-1, 1]` (higher is better). O(n^2).
pub fn silhouette(points: &[Vec<f32>], assignments: &[usize]) -> f32 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let k = assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c].push(i);
    }

    let dist = |a: usize, b: usize| dist2(&points[a], &points[b]).sqrt();
    let mut total = 0.0;
    let mut count = 0u32;
    for (i, &ci) in assignments.iter().enumerate() {
        if clusters[ci].len() <= 1 {
            continue; // singleton clusters contribute silhouette 0.
        }
        // a(i): mean distance to other members of its cluster.
        let a: f32 = clusters[ci]
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| dist(i, j))
            .sum::<f32>()
            / (clusters[ci].len() - 1) as f32;
        // b(i): smallest mean distance to any other cluster.
        let b = (0..k)
            .filter(|&c| c != ci && !clusters[c].is_empty())
            .map(|c| {
                clusters[c].iter().map(|&j| dist(i, j)).sum::<f32>() / clusters[c].len() as f32
            })
            .fold(f32::INFINITY, f32::min);
        if b.is_finite() {
            total += (b - a) / a.max(b);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// One-call clustering search over the number of clusters (PRD §20).
///
/// For each candidate `k`, K-means is run `n_init` times (keeping the lowest
/// inertia) and scored by the mean silhouette, which the study maximizes.
pub struct AutoCluster {
    data: Vec<Vec<f32>>,
    min_k: usize,
    max_k: usize,
    n_init: usize,
    trials: u64,
    seed: u64,
}

impl AutoCluster {
    /// A new clustering search over `data`, considering `k` in `[2, max_k]`.
    pub fn new(data: Vec<Vec<f32>>, max_k: usize) -> Self {
        AutoCluster {
            data,
            min_k: 2,
            max_k: max_k.max(2),
            n_init: 4,
            trials: 12,
            seed: 0,
        }
    }

    /// Number of K-means restarts per candidate `k`.
    pub fn n_init(mut self, n: usize) -> Self {
        self.n_init = n.max(1);
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

    /// Run the search. Returns the study (maximizing silhouette); the best
    /// trial's `k` parameter is the chosen number of clusters.
    pub fn fit(self) -> Result<Study> {
        if self.data.len() < self.min_k {
            return Err(Error::Objective(
                "clustering needs at least as many points as clusters".into(),
            ));
        }
        let max_k = self.max_k.min(self.data.len() - 1).max(self.min_k);
        let space = SearchSpace::new().add("k", Distribution::int(self.min_k as i64, max_k as i64));

        // The number of clusters is a single small integer dimension, so grid
        // search evaluates each candidate exactly rather than sampling blindly.
        let mut study = Study::builder(space)
            .name("auto-cluster")
            .maximize("silhouette")
            .sampler(GridSampler::new(1))
            .seed(self.seed)
            .build()?;

        let data = Arc::new(self.data);
        let n_init = self.n_init;
        let objective = move |p: &ParamSet, _sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let k = p.int("k")? as usize;
            // Best of n_init restarts by inertia.
            let best = (0..n_init)
                .map(|r| kmeans(&data, k, (k as u64) * 1000 + r as u64, 50))
                .min_by(|a, b| {
                    a.inertia
                        .partial_cmp(&b.inertia)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let score = silhouette(&data, &best.assignments);
            Ok(NamedMetrics::single("silhouette", score as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate `k` well-separated Gaussian blobs in 2-D.
    fn blobs(k: usize, per: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pts = Vec::new();
        for c in 0..k {
            let (cx, cy) = ((c as f32) * 10.0, (c as f32 % 2.0) * 10.0);
            for _ in 0..per {
                pts.push(vec![
                    cx + rng.gen_range(-0.6..0.6),
                    cy + rng.gen_range(-0.6..0.6),
                ]);
            }
        }
        pts
    }

    #[test]
    fn kmeans_recovers_separated_blobs() {
        let pts = blobs(3, 30, 1);
        let c = kmeans(&pts, 3, 7, 50);
        // Well-separated blobs => high silhouette.
        let s = silhouette(&pts, &c.assignments);
        assert!(s > 0.7, "silhouette {s} too low for separated blobs");
    }

    #[test]
    fn auto_cluster_finds_the_true_k() {
        let pts = blobs(3, 40, 2);
        let study = AutoCluster::new(pts, 6).trials(5).seed(2).fit().unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let k = best.params.int("k").unwrap();
        assert_eq!(k, 3, "expected 3 clusters, chose {k}");
        assert!(best.final_value("silhouette").unwrap() > 0.7);
    }

    #[test]
    fn too_few_points_errors() {
        assert!(AutoCluster::new(vec![vec![0.0]], 3).fit().is_err());
    }
}
