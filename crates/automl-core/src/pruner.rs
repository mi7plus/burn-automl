//! Pruners: early-stopping decisions for running trials (PRD §4.1, §29 items 6,
//! 14).
//!
//! v0.1 ships four pruners: a no-op pruner, the median pruner, the ASHA
//! (Asynchronous Successive Halving) pruner, and the multi-objective pruner.
//! Median pruning compares a running trial's latest intermediate value against
//! the median of completed trials' values *at the same step*; ASHA promotes
//! only the top `1/eta` of trials that reach each rung; the multi-objective
//! pruner (§17) drops a trial dominated by a majority of its peers.

use crate::metrics::Direction;
use crate::trial::{TrialHistory, TrialProgress};

/// Decides whether a running trial should be terminated early (PRD §4.1).
pub trait Pruner: Send + Sync {
    /// Return true to prune (stop) the trial described by `trial`.
    fn should_prune(&self, trial: &TrialProgress, history: &TrialHistory) -> bool;

    /// A short, stable name persisted in study metadata.
    fn name(&self) -> &'static str;
}

/// Never prunes. The default when a study opts out of early stopping.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPruner;

impl Pruner for NoPruner {
    fn should_prune(&self, _trial: &TrialProgress, _history: &TrialHistory) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "none"
    }
}

/// Median pruner: prune a trial whose latest value is worse than the median of
/// completed trials at the same step.
///
/// - `objective`/`direction` select which reported metric to compare and which
///   way is "better".
/// - `warmup_steps`: never prune before this step (let trials establish a curve).
/// - `min_trials`: require at least this many prior observations at the step
///   before pruning, so early trials aren't judged against thin evidence.
pub struct MedianPruner {
    objective: String,
    direction: Direction,
    warmup_steps: u64,
    min_trials: usize,
    robust_window: usize,
}

impl MedianPruner {
    /// Construct a median pruner for a named objective and direction.
    pub fn new(objective: impl Into<String>, direction: Direction) -> Self {
        MedianPruner {
            objective: objective.into(),
            direction,
            warmup_steps: 1,
            min_trials: 1,
            robust_window: 1,
        }
    }

    /// Set the number of warmup steps before pruning may occur.
    pub fn with_warmup_steps(mut self, steps: u64) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set the minimum number of prior observations required to prune.
    pub fn with_min_trials(mut self, n: usize) -> Self {
        self.min_trials = n;
        self
    }

    /// Enable the robust noisy-curve mode (PRD §18): compare the *median of the
    /// trial's last `window` reported values* against peers, instead of its
    /// single latest value. This smooths the short-term noise of GAN/RL-style
    /// learning curves so a lucky or unlucky spike does not decide pruning. A
    /// window of 1 (the default) is the ordinary latest-value behavior.
    pub fn with_robust_window(mut self, window: usize) -> Self {
        self.robust_window = window.max(1);
        self
    }

    /// Median of a slice (ignoring NaNs). Returns `None` for an empty slice.
    fn median(values: &[f64]) -> Option<f64> {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = v.len() / 2;
        if v.len().is_multiple_of(2) {
            Some((v[mid - 1] + v[mid]) / 2.0)
        } else {
            Some(v[mid])
        }
    }
}

impl Pruner for MedianPruner {
    fn should_prune(&self, trial: &TrialProgress, history: &TrialHistory) -> bool {
        let Some(step) = trial.last_step() else {
            return false;
        };
        if step < self.warmup_steps {
            return false;
        }
        // In robust mode, judge the trial by the median of its recent curve
        // rather than its single latest (possibly noisy) value.
        let current = if self.robust_window > 1 {
            let recent: Vec<f64> = trial
                .intermediate
                .iter()
                .rev()
                .filter_map(|r| r.metrics.get(&self.objective))
                .take(self.robust_window)
                .collect();
            match Self::median(&recent) {
                Some(m) => m,
                None => return false,
            }
        } else {
            let Some(current) = trial.last_value(&self.objective) else {
                return false;
            };
            current
        };
        let peers = history.intermediate_values_at(step, &self.objective);
        if peers.len() < self.min_trials {
            return false;
        }
        match Self::median(&peers) {
            // Prune when the current value is strictly worse than the median.
            Some(m) => self.direction.is_better(m, current),
            None => false,
        }
    }

    fn name(&self) -> &'static str {
        "median"
    }
}

/// Asynchronous Successive Halving (ASHA) pruner.
///
/// ASHA defines geometric *rungs* of resource (step/epoch): `min_resource *
/// eta^k` for `k = 0, 1, 2, ...`. When a trial reaches a rung, it is *promoted*
/// (kept) only if it ranks in the top `1/eta` of the trials that have reached
/// that same rung; otherwise it is pruned. Unlike synchronous Hyperband, the
/// decision is made asynchronously as each trial reports, with no waiting for a
/// bracket to fill — the practical variant used at scale. Between rungs the
/// pruner never stops a trial, letting it run to the next rung.
///
/// This is the allocation half of PRD §29 item 14; full Hyperband bracketing
/// (multiple `min_resource` values run together) can layer on top later.
pub struct AshaPruner {
    objective: String,
    direction: Direction,
    min_resource: u64,
    reduction_factor: u64,
}

impl AshaPruner {
    /// Construct an ASHA pruner. `min_resource` is the first rung's resource
    /// level (e.g. 1 epoch); `reduction_factor` (eta, >= 2) is both the rung
    /// spacing and the inverse promotion fraction (top `1/eta` survive).
    pub fn new(objective: impl Into<String>, direction: Direction) -> Self {
        AshaPruner {
            objective: objective.into(),
            direction,
            min_resource: 1,
            reduction_factor: 3,
        }
    }

    /// Set the first rung's resource level (default 1).
    pub fn with_min_resource(mut self, min_resource: u64) -> Self {
        self.min_resource = min_resource.max(1);
        self
    }

    /// Set the reduction factor eta (default 3; clamped to >= 2).
    pub fn with_reduction_factor(mut self, eta: u64) -> Self {
        self.reduction_factor = eta.max(2);
        self
    }

    /// Whether `step` is exactly a rung level `min_resource * eta^k`.
    fn is_rung(&self, step: u64) -> bool {
        if step < self.min_resource {
            return false;
        }
        let mut level = self.min_resource;
        loop {
            if level == step {
                return true;
            }
            if level > step / self.reduction_factor {
                // Next multiply would exceed `step`; no exact match.
                return false;
            }
            level *= self.reduction_factor;
        }
    }
}

impl Pruner for AshaPruner {
    fn should_prune(&self, trial: &TrialProgress, history: &TrialHistory) -> bool {
        let Some(step) = trial.last_step() else {
            return false;
        };
        // ASHA only decides at rung boundaries.
        if !self.is_rung(step) {
            return false;
        }
        let Some(current) = trial.last_value(&self.objective) else {
            return false;
        };
        // Peer values are other trials that reached this same rung.
        let peers = history.intermediate_values_at(step, &self.objective);
        let n = peers.len() + 1; // include this trial
                                 // Need at least `eta` trials at the rung before halving is meaningful.
        if (n as u64) < self.reduction_factor {
            return false;
        }
        let keep = (n as u64 / self.reduction_factor).max(1) as usize;
        // Rank: how many peers are strictly better than the current value.
        let better = peers
            .iter()
            .filter(|&&v| self.direction.is_better(v, current))
            .count();
        // Promoted iff within the top `keep`; otherwise prune.
        better >= keep
    }

    fn name(&self) -> &'static str {
        "asha"
    }
}

/// Multi-objective pruner using non-domination against peers (PRD §17).
///
/// Single-objective pruners (median, ASHA) compare one metric; a multi-objective
/// study has no single scalar to threshold. This pruner instead prunes a running
/// trial when, at the current step, a majority of the peers that reached the
/// same step **dominate** it under the study's directions — a first-order,
/// NSGA-II-flavored front-rank rule. It dispatches only when a study declares
/// more than one objective, so single-objective studies pay nothing for it.
pub struct MultiObjectivePruner {
    front: crate::pareto::ParetoFront,
    warmup_steps: u64,
    min_trials: usize,
}

impl MultiObjectivePruner {
    /// Construct a multi-objective pruner over the study's objectives.
    pub fn new(objectives: Vec<crate::metrics::Objective>) -> Self {
        MultiObjectivePruner {
            front: crate::pareto::ParetoFront::new(objectives),
            warmup_steps: 1,
            min_trials: 1,
        }
    }

    /// Set the number of warmup steps before pruning may occur.
    pub fn with_warmup_steps(mut self, steps: u64) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set the minimum number of peer observations required before pruning.
    pub fn with_min_trials(mut self, n: usize) -> Self {
        self.min_trials = n;
        self
    }
}

impl Pruner for MultiObjectivePruner {
    fn should_prune(&self, trial: &TrialProgress, history: &TrialHistory) -> bool {
        let Some(step) = trial.last_step() else {
            return false;
        };
        if step < self.warmup_steps {
            return false;
        }
        let Some(current) = trial.last_metrics() else {
            return false;
        };
        let peers = history.intermediate_metrics_at(step);
        if peers.len() < self.min_trials {
            return false;
        }
        let dominated_by = peers
            .iter()
            .filter(|p| self.front.dominates(p, current))
            .count();
        // Prune when strictly more than half of peers dominate the current point.
        dominated_by * 2 > peers.len()
    }

    fn name(&self) -> &'static str {
        "multi-objective"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NamedMetrics;
    use crate::param::ParamSet;
    use crate::trial::{IntermediateReport, StudyId, TrialId, TrialRecord, TrialState};

    fn completed_with_curve(id: u64, curve: &[(u64, f64)]) -> TrialRecord {
        let mut r = TrialRecord::new(TrialId(id), StudyId(0), ParamSet::new(), id);
        r.state = TrialState::Complete;
        r.intermediate = curve
            .iter()
            .map(|(s, v)| IntermediateReport {
                step: *s,
                metrics: NamedMetrics::single("loss", *v),
            })
            .collect();
        r
    }

    #[test]
    fn prunes_worse_than_median() {
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(1, 0.1)]),
            completed_with_curve(1, &[(1, 0.2)]),
            completed_with_curve(2, &[(1, 0.3)]),
        ]);
        let pruner = MedianPruner::new("loss", Direction::Minimize);
        // Current trial reports 0.5 at step 1; median of peers is 0.2 -> prune.
        let reports = vec![IntermediateReport {
            step: 1,
            metrics: NamedMetrics::single("loss", 0.5),
        }];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(pruner.should_prune(&progress, &history));
    }

    #[test]
    fn keeps_better_than_median() {
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(1, 0.4)]),
            completed_with_curve(1, &[(1, 0.5)]),
        ]);
        let pruner = MedianPruner::new("loss", Direction::Minimize);
        let reports = vec![IntermediateReport {
            step: 1,
            metrics: NamedMetrics::single("loss", 0.1),
        }];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!pruner.should_prune(&progress, &history));
    }

    #[test]
    fn respects_warmup() {
        let history = TrialHistory::new(vec![completed_with_curve(0, &[(1, 0.1)])]);
        let pruner = MedianPruner::new("loss", Direction::Minimize).with_warmup_steps(5);
        let reports = vec![IntermediateReport {
            step: 1,
            metrics: NamedMetrics::single("loss", 9.9),
        }];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!pruner.should_prune(&progress, &history));
    }

    fn progress_at(step: u64, value: f64) -> Vec<IntermediateReport> {
        vec![IntermediateReport {
            step,
            metrics: NamedMetrics::single("loss", value),
        }]
    }

    #[test]
    fn robust_window_smooths_a_noisy_spike() {
        // Peers sit at loss 0.2; the current trial's curve is good (0.1) but its
        // latest point spiked to 0.9 from noise.
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(3, 0.2)]),
            completed_with_curve(1, &[(3, 0.2)]),
            completed_with_curve(2, &[(3, 0.2)]),
        ]);
        let curve = vec![
            IntermediateReport {
                step: 1,
                metrics: NamedMetrics::single("loss", 0.10),
            },
            IntermediateReport {
                step: 2,
                metrics: NamedMetrics::single("loss", 0.12),
            },
            IntermediateReport {
                step: 3,
                metrics: NamedMetrics::single("loss", 0.90), // noisy spike
            },
        ];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &curve,
        };
        // Latest-value mode prunes on the spike (0.9 worse than 0.2)...
        let plain = MedianPruner::new("loss", Direction::Minimize);
        assert!(plain.should_prune(&progress, &history));
        // ...but a robust window of 3 compares median{0.10,0.12,0.90}=0.12 < 0.2,
        // so the trial survives its unlucky spike.
        let robust = MedianPruner::new("loss", Direction::Minimize).with_robust_window(3);
        assert!(!robust.should_prune(&progress, &history));
    }

    #[test]
    fn asha_rung_detection() {
        let asha = AshaPruner::new("loss", Direction::Minimize)
            .with_min_resource(1)
            .with_reduction_factor(3);
        assert!(asha.is_rung(1));
        assert!(asha.is_rung(3));
        assert!(asha.is_rung(9));
        assert!(!asha.is_rung(2));
        assert!(!asha.is_rung(4));
        assert!(!asha.is_rung(0));
    }

    #[test]
    fn asha_prunes_bottom_fraction_at_rung() {
        // Three peers reached rung step=1 with good values; a fourth (current)
        // is worst. eta=3, n=4 -> keep floor(4/3)=1, so all but the best prune.
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(1, 0.1)]),
            completed_with_curve(1, &[(1, 0.2)]),
            completed_with_curve(2, &[(1, 0.3)]),
        ]);
        let asha = AshaPruner::new("loss", Direction::Minimize).with_reduction_factor(3);
        let reports = progress_at(1, 0.9);
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(asha.should_prune(&progress, &history));
    }

    #[test]
    fn asha_promotes_top_trial() {
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(1, 0.4)]),
            completed_with_curve(1, &[(1, 0.5)]),
            completed_with_curve(2, &[(1, 0.6)]),
        ]);
        let asha = AshaPruner::new("loss", Direction::Minimize).with_reduction_factor(3);
        // Current is the best -> promoted (not pruned).
        let reports = progress_at(1, 0.1);
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!asha.should_prune(&progress, &history));
    }

    #[test]
    fn asha_never_prunes_between_rungs_or_with_too_few_peers() {
        let asha = AshaPruner::new("loss", Direction::Minimize).with_reduction_factor(3);
        // Non-rung step: never prune even if worst.
        let history = TrialHistory::new(vec![
            completed_with_curve(0, &[(2, 0.1)]),
            completed_with_curve(1, &[(2, 0.2)]),
            completed_with_curve(2, &[(2, 0.3)]),
        ]);
        let reports = progress_at(2, 9.9);
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!asha.should_prune(&progress, &history));

        // Rung step but fewer than eta trials present -> no decision yet.
        let history = TrialHistory::new(vec![completed_with_curve(0, &[(1, 0.1)])]);
        let reports = progress_at(1, 9.9);
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!asha.should_prune(&progress, &history));
    }

    fn mo_completed(id: u64, step: u64, err: f64, size: f64) -> TrialRecord {
        let mut r = TrialRecord::new(TrialId(id), StudyId(0), ParamSet::new(), id);
        r.state = TrialState::Complete;
        r.intermediate = vec![IntermediateReport {
            step,
            metrics: NamedMetrics::new().with("err", err).with("size", size),
        }];
        r
    }

    #[test]
    fn multi_objective_prunes_dominated_point() {
        use crate::metrics::Objective;
        // Minimize both err and size. Peers are all good; current is worse on
        // both, so a majority dominate it -> prune.
        let history = TrialHistory::new(vec![
            mo_completed(0, 1, 0.1, 0.1),
            mo_completed(1, 1, 0.2, 0.2),
            mo_completed(2, 1, 0.3, 0.3),
        ]);
        let pruner = MultiObjectivePruner::new(vec![
            Objective::minimize("err"),
            Objective::minimize("size"),
        ]);
        let reports = vec![IntermediateReport {
            step: 1,
            metrics: NamedMetrics::new().with("err", 0.9).with("size", 0.9),
        }];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(pruner.should_prune(&progress, &history));
    }

    #[test]
    fn multi_objective_keeps_non_dominated_tradeoff() {
        use crate::metrics::Objective;
        // Current trades err for size: best on err, worst on size. No peer
        // dominates it, so it must be kept.
        let history = TrialHistory::new(vec![
            mo_completed(0, 1, 0.5, 0.1),
            mo_completed(1, 1, 0.4, 0.2),
            mo_completed(2, 1, 0.3, 0.3),
        ]);
        let pruner = MultiObjectivePruner::new(vec![
            Objective::minimize("err"),
            Objective::minimize("size"),
        ]);
        let reports = vec![IntermediateReport {
            step: 1,
            metrics: NamedMetrics::new().with("err", 0.05).with("size", 0.9),
        }];
        let progress = TrialProgress {
            id: TrialId(9),
            intermediate: &reports,
        };
        assert!(!pruner.should_prune(&progress, &history));
    }
}
