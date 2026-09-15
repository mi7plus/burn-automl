//! `ParetoFront` — the core object for multi-objective optimization (PRD §17.1,
//! §29 item 15).
//!
//! Multi-objective studies return a *set* of non-dominated trials rather than a
//! single best. This matters especially for Rust deployments, where latency,
//! memory and model size can weigh as much as predictive quality (§17). The
//! plan adds `ParetoFront` as a first-class core object owned by the study
//! alongside the sampler and pruner.
//!
//! A point `a` **dominates** `b` when `a` is at least as good as `b` on every
//! objective and strictly better on at least one, judged per that objective's
//! [`Direction`]. The front keeps only the mutually non-dominated members and
//! exposes [`ParetoFront::hypervolume`] as a scalar quality measure of the whole
//! set.

use crate::metrics::{Direction, NamedMetrics, Objective};
use crate::trial::TrialId;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// One member of the front: a trial and its objective values (in objective
/// order).
#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    /// The trial this point came from.
    pub trial: TrialId,
    /// Objective values aligned to the front's objective order.
    pub values: Vec<f64>,
}

/// The non-dominated set under a multi-objective study.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    objectives: Vec<Objective>,
    members: Vec<Member>,
}

impl ParetoFront {
    /// A new, empty front over the given objectives (name + direction each).
    pub fn new(objectives: Vec<Objective>) -> Self {
        ParetoFront {
            objectives,
            members: Vec::new(),
        }
    }

    /// The objectives this front ranks by.
    pub fn objectives(&self) -> &[Objective] {
        &self.objectives
    }

    /// The current non-dominated members.
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// The trial ids currently on the front.
    pub fn trial_ids(&self) -> Vec<TrialId> {
        self.members.iter().map(|m| m.trial).collect()
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the front is empty.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Extract this front's objective values from a metrics bag, or `None` if
    /// any objective is missing.
    fn extract(&self, metrics: &NamedMetrics) -> Option<Vec<f64>> {
        self.objectives
            .iter()
            .map(|o| metrics.get(&o.name))
            .collect()
    }

    /// Whether `a` dominates `b` under the front's directions.
    fn dominates_values(&self, a: &[f64], b: &[f64]) -> bool {
        let mut strictly_better_somewhere = false;
        for (i, dir) in self.objectives.iter().map(|o| o.direction).enumerate() {
            let (av, bv) = (a[i], b[i]);
            if dir.is_better(bv, av) {
                // b is strictly better on this objective => a does not dominate.
                return false;
            }
            if dir.is_better(av, bv) {
                strictly_better_somewhere = true;
            }
        }
        strictly_better_somewhere
    }

    /// Whether one metrics bag dominates another (public convenience matching
    /// the PRD signature).
    pub fn dominates(&self, a: &NamedMetrics, b: &NamedMetrics) -> bool {
        match (self.extract(a), self.extract(b)) {
            (Some(av), Some(bv)) => self.dominates_values(&av, &bv),
            _ => false,
        }
    }

    /// Offer a trial's metrics to the front. Returns `true` if it was added to
    /// the non-dominated set (removing any members it dominates); `false` if it
    /// was dominated by an existing member, a duplicate, or missing objectives.
    pub fn insert(&mut self, trial: TrialId, metrics: &NamedMetrics) -> bool {
        let Some(values) = self.extract(metrics) else {
            return false;
        };
        // Rejected if an existing member dominates it or exactly equals it.
        if self
            .members
            .iter()
            .any(|m| m.values == values || self.dominates_values(&m.values, &values))
        {
            return false;
        }
        // Remove members the newcomer dominates.
        let dominated: Vec<usize> = self
            .members
            .iter()
            .enumerate()
            .filter(|(_, m)| self.dominates_values(&values, &m.values))
            .map(|(i, _)| i)
            .collect();
        for i in dominated.into_iter().rev() {
            self.members.remove(i);
        }
        self.members.push(Member { trial, values });
        true
    }

    /// The dominated hypervolume of the front relative to a `reference` point
    /// (a point each front member should dominate — e.g. worst acceptable
    /// values). Larger is better. Objectives are internally transformed to a
    /// minimization problem so the measure is direction-agnostic.
    ///
    /// Exact for one or two objectives; for three or more it is a deterministic
    /// Monte-Carlo estimate (documented as approximate).
    pub fn hypervolume(&self, reference: &NamedMetrics) -> f64 {
        let Some(reference) = self.extract(reference) else {
            return 0.0;
        };
        if self.members.is_empty() {
            return 0.0;
        }

        // Transform everything to minimization "loss" space; the reference must
        // be an upper bound (worse than every point) there.
        let to_loss = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&self.objectives)
                .map(|(x, o)| match o.direction {
                    Direction::Minimize => *x,
                    Direction::Maximize => -*x,
                })
                .collect()
        };
        let ref_loss = to_loss(&reference);
        let points: Vec<Vec<f64>> = self.members.iter().map(|m| to_loss(&m.values)).collect();

        match self.objectives.len() {
            0 => 0.0,
            1 => points
                .iter()
                .map(|p| (ref_loss[0] - p[0]).max(0.0))
                .fold(0.0, f64::max),
            2 => hypervolume_2d(&points, &ref_loss),
            _ => hypervolume_monte_carlo(&points, &ref_loss),
        }
    }
}

/// Exact 2-D hypervolume for a non-dominated set in minimization space.
fn hypervolume_2d(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    // Keep only points strictly inside the reference box.
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (p[0], p[1]))
        .filter(|&(x, y)| x < reference[0] && y < reference[1])
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    // Sort by x ascending; for a non-dominated set y is then descending.
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut area = 0.0;
    let mut prev_y = reference[1];
    for (x, y) in pts {
        if y < prev_y {
            area += (reference[0] - x) * (prev_y - y);
            prev_y = y;
        }
    }
    area
}

/// Deterministic Monte-Carlo hypervolume estimate for 3+ objectives.
fn hypervolume_monte_carlo(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let dims = reference.len();
    // Bounding box: from the per-dimension minimum (ideal) to the reference.
    let ideal: Vec<f64> = (0..dims)
        .map(|d| points.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min))
        .collect();
    let box_volume: f64 = (0..dims)
        .map(|d| (reference[d] - ideal[d]).max(0.0))
        .product();
    if box_volume <= 0.0 {
        return 0.0;
    }

    let samples = 20_000;
    // Fixed seed keeps the estimate deterministic across runs.
    let mut rng = ChaCha8Rng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
    let mut inside = 0usize;
    let mut sample = vec![0.0; dims];
    for _ in 0..samples {
        for d in 0..dims {
            sample[d] = rng.gen_range(ideal[d]..reference[d]);
        }
        // Dominated by the front iff some point is <= sample in every dim.
        let dominated = points.iter().any(|p| (0..dims).all(|d| p[d] <= sample[d]));
        if dominated {
            inside += 1;
        }
    }
    box_volume * inside as f64 / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objs() -> Vec<Objective> {
        vec![
            Objective::minimize("latency"),
            Objective::maximize("accuracy"),
        ]
    }

    fn m(latency: f64, accuracy: f64) -> NamedMetrics {
        NamedMetrics::new()
            .with("latency", latency)
            .with("accuracy", accuracy)
    }

    #[test]
    fn insert_keeps_non_dominated() {
        let mut front = ParetoFront::new(objs());
        // Lower latency & higher accuracy both better.
        assert!(front.insert(TrialId(0), &m(10.0, 0.90)));
        assert!(front.insert(TrialId(1), &m(20.0, 0.95))); // trades latency for accuracy
        assert_eq!(front.len(), 2);
        // A point worse on both is dominated -> rejected.
        assert!(!front.insert(TrialId(2), &m(25.0, 0.85)));
        assert_eq!(front.len(), 2);
        // A point better on both dominates trial 0 -> replaces it.
        assert!(front.insert(TrialId(3), &m(8.0, 0.92)));
        assert!(!front.trial_ids().contains(&TrialId(0)));
        assert!(front.trial_ids().contains(&TrialId(3)));
    }

    #[test]
    fn dominance_respects_direction() {
        let front = ParetoFront::new(objs());
        // Lower latency, higher accuracy => (5, 0.9) dominates (10, 0.8).
        assert!(front.dominates(&m(5.0, 0.9), &m(10.0, 0.8)));
        assert!(!front.dominates(&m(10.0, 0.8), &m(5.0, 0.9)));
        // Trade-off: neither dominates.
        assert!(!front.dominates(&m(5.0, 0.8), &m(10.0, 0.9)));
    }

    #[test]
    fn hypervolume_2d_matches_hand_calculation() {
        // Minimize both objectives here for a clean hand-check.
        let mut front = ParetoFront::new(vec![Objective::minimize("a"), Objective::minimize("b")]);
        front.insert(
            TrialId(0),
            &NamedMetrics::new().with("a", 1.0).with("b", 3.0),
        );
        front.insert(
            TrialId(1),
            &NamedMetrics::new().with("a", 2.0).with("b", 2.0),
        );
        // Reference (4, 4): union area of [1,4]x[3,4] and [2,4]x[2,4] = 5.
        let hv = front.hypervolume(&NamedMetrics::new().with("a", 4.0).with("b", 4.0));
        assert!((hv - 5.0).abs() < 1e-9, "hv = {hv}");
    }

    #[test]
    fn hypervolume_grows_with_better_front() {
        let mk = |pts: &[(f64, f64)]| {
            let mut f = ParetoFront::new(vec![Objective::minimize("a"), Objective::minimize("b")]);
            for (i, (a, b)) in pts.iter().enumerate() {
                f.insert(
                    TrialId(i as u64),
                    &NamedMetrics::new().with("a", *a).with("b", *b),
                );
            }
            f.hypervolume(&NamedMetrics::new().with("a", 10.0).with("b", 10.0))
        };
        let worse = mk(&[(5.0, 5.0)]);
        let better = mk(&[(2.0, 2.0)]);
        assert!(better > worse, "better={better} worse={worse}");
    }
}
