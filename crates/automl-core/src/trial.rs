//! Trials: identity, lifecycle state, intermediate reports, and history.
//!
//! A [`TrialRecord`] is the minimum persisted record from PRD §19.2. The
//! [`TrialHistory`] handed to samplers and pruners is a read-only view over
//! completed and running trials for a study.

use crate::metrics::NamedMetrics;
use crate::param::ParamSet;
use serde::{Deserialize, Serialize};

/// Opaque identifier for a study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StudyId(pub u64);

/// Opaque identifier for a trial, unique within a study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TrialId(pub u64);

impl std::fmt::Display for TrialId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trial#{}", self.0)
    }
}

impl std::fmt::Display for StudyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "study#{}", self.0)
    }
}

/// Lifecycle state of a trial (PRD §4.2 `TrialState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialState {
    /// Enqueued, not yet started.
    Waiting,
    /// Currently executing.
    Running,
    /// Finished successfully with final metrics.
    Complete,
    /// Terminated early by a pruner.
    Pruned,
    /// Terminated by an error in the objective.
    Failed,
    /// Cancelled by the study or a lease expiry.
    Cancelled,
}

impl TrialState {
    /// Whether this is a terminal state (no further reports expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TrialState::Complete | TrialState::Pruned | TrialState::Failed | TrialState::Cancelled
        )
    }
}

/// A single intermediate report: metrics at a training step.
///
/// Writes are keyed by `(trial_id, step)` in storage so a retried report cannot
/// double-count a metric (PRD §18.2 idempotent writes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntermediateReport {
    /// The step (epoch, iteration, environment step) this report is for.
    pub step: u64,
    /// Metrics observed at this step.
    pub metrics: NamedMetrics,
}

/// The persisted record of a trial (PRD §19.2 minimum record).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialRecord {
    /// Trial identity.
    pub id: TrialId,
    /// Owning study.
    pub study_id: StudyId,
    /// The sampled parameters.
    pub params: ParamSet,
    /// Current lifecycle state.
    pub state: TrialState,
    /// Intermediate reports in step order.
    pub intermediate: Vec<IntermediateReport>,
    /// Final metrics, present once the trial completes.
    pub final_metrics: Option<NamedMetrics>,
    /// The seed used to make this trial reproducible.
    pub seed: u64,
}

impl TrialRecord {
    /// Create a fresh waiting trial record.
    pub fn new(id: TrialId, study_id: StudyId, params: ParamSet, seed: u64) -> Self {
        TrialRecord {
            id,
            study_id,
            params,
            state: TrialState::Waiting,
            intermediate: Vec::new(),
            final_metrics: None,
            seed,
        }
    }

    /// The most recent intermediate metrics, if any.
    pub fn last_intermediate(&self) -> Option<&IntermediateReport> {
        self.intermediate.last()
    }

    /// Value of a named objective at the final metrics, if present.
    pub fn final_value(&self, objective: &str) -> Option<f64> {
        self.final_metrics.as_ref().and_then(|m| m.get(objective))
    }
}

/// A running trial's progress, handed to a [`crate::pruner::Pruner`].
#[derive(Debug, Clone)]
pub struct TrialProgress<'a> {
    /// Which trial is being evaluated.
    pub id: TrialId,
    /// The intermediate reports observed so far.
    pub intermediate: &'a [IntermediateReport],
}

impl<'a> TrialProgress<'a> {
    /// The latest step number reported, if any.
    pub fn last_step(&self) -> Option<u64> {
        self.intermediate.last().map(|r| r.step)
    }

    /// The latest value for a named objective.
    pub fn last_value(&self, objective: &str) -> Option<f64> {
        self.intermediate
            .last()
            .and_then(|r| r.metrics.get(objective))
    }

    /// The full metrics bag of the most recent report (all objectives at once),
    /// used by the multi-objective pruner.
    pub fn last_metrics(&self) -> Option<&NamedMetrics> {
        self.intermediate.last().map(|r| &r.metrics)
    }

    /// The value for a named objective at a specific step.
    pub fn value_at(&self, step: u64, objective: &str) -> Option<f64> {
        self.intermediate
            .iter()
            .find(|r| r.step == step)
            .and_then(|r| r.metrics.get(objective))
    }
}

/// Read-only view of a study's trials, passed to samplers and pruners.
///
/// Samplers use it to model the response surface (PRD §4.1 `Sampler::suggest`);
/// pruners compare a running trial against the distribution of completed ones.
#[derive(Debug, Clone, Default)]
pub struct TrialHistory {
    records: Vec<TrialRecord>,
}

impl TrialHistory {
    /// Build a history from a set of records.
    pub fn new(records: Vec<TrialRecord>) -> Self {
        TrialHistory { records }
    }

    /// All records, in insertion order.
    pub fn records(&self) -> &[TrialRecord] {
        &self.records
    }

    /// Number of trials recorded.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether there are no trials yet.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate only completed trials.
    pub fn completed(&self) -> impl Iterator<Item = &TrialRecord> {
        self.records
            .iter()
            .filter(|r| r.state == TrialState::Complete)
    }

    /// The completed trials' final values for a named objective, in order.
    pub fn completed_values(&self, objective: &str) -> Vec<f64> {
        self.completed()
            .filter_map(|r| r.final_value(objective))
            .collect()
    }

    /// Intermediate values reported by completed trials at a given step,
    /// used by the median pruner to build a per-step comparison distribution.
    pub fn intermediate_values_at(&self, step: u64, objective: &str) -> Vec<f64> {
        self.records
            .iter()
            .filter(|r| r.state == TrialState::Complete || r.state == TrialState::Pruned)
            .filter_map(|r| {
                r.intermediate
                    .iter()
                    .find(|ir| ir.step == step)
                    .and_then(|ir| ir.metrics.get(objective))
            })
            .collect()
    }

    /// Full intermediate metrics bags reported by settled trials at a given
    /// step, used by the multi-objective pruner for dominance comparison.
    pub fn intermediate_metrics_at(&self, step: u64) -> Vec<NamedMetrics> {
        self.records
            .iter()
            .filter(|r| r.state == TrialState::Complete || r.state == TrialState::Pruned)
            .filter_map(|r| {
                r.intermediate
                    .iter()
                    .find(|ir| ir.step == step)
                    .map(|ir| ir.metrics.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_terminality() {
        assert!(TrialState::Complete.is_terminal());
        assert!(!TrialState::Running.is_terminal());
        assert!(!TrialState::Waiting.is_terminal());
    }

    #[test]
    fn history_filters_completed() {
        let mut r1 = TrialRecord::new(TrialId(0), StudyId(0), ParamSet::new(), 1);
        r1.state = TrialState::Complete;
        r1.final_metrics = Some(NamedMetrics::single("loss", 0.2));
        let mut r2 = TrialRecord::new(TrialId(1), StudyId(0), ParamSet::new(), 2);
        r2.state = TrialState::Failed;
        let h = TrialHistory::new(vec![r1, r2]);
        assert_eq!(h.completed().count(), 1);
        assert_eq!(h.completed_values("loss"), vec![0.2]);
    }
}
