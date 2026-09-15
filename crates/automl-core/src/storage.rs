//! Storage: the `Storage` trait plus the in-memory backend (PRD §4.1, §29 item 3).
//!
//! In-memory storage is the first backend so debugging and replay are
//! trustworthy from day one (§29 item 3). It is authoritative for trial state,
//! which is the same invariant the distributed protocol later relies on
//! (§18.2 "Storage is the single source of truth"). The SQLite backend (§19.1)
//! layers on later behind the same trait.
//!
//! Idempotent reporting (§18.2): [`Storage::report`] is keyed by `(trial, step)`
//! — a retried report at an existing step overwrites rather than appends, so a
//! network retry cannot double-count a metric.

use crate::error::{Error, Result};
use crate::metrics::{Direction, NamedMetrics};
use crate::param::ParamSet;
use crate::trial::{IntermediateReport, StudyId, TrialHistory, TrialId, TrialRecord, TrialState};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Metadata describing a study, persisted at creation (PRD §19.2 `StudyMeta`).
#[derive(Debug, Clone, PartialEq)]
pub struct StudyMeta {
    /// Human-readable study name.
    pub name: String,
    /// One direction per named objective. A single entry is a single-objective
    /// study; multiple entries feed the multi-objective machinery (§17).
    pub directions: Vec<(String, Direction)>,
    /// Name of the sampler used (provenance).
    pub sampler_name: String,
    /// Name of the pruner used (provenance).
    pub pruner_name: String,
}

impl StudyMeta {
    /// Whether this is a multi-objective study.
    pub fn is_multi_objective(&self) -> bool {
        self.directions.len() > 1
    }

    /// The single objective's `(name, direction)` if this is single-objective.
    pub fn single_objective(&self) -> Option<&(String, Direction)> {
        if self.directions.len() == 1 {
            self.directions.first()
        } else {
            None
        }
    }
}

/// Persists studies, trials, metrics and provenance (PRD §4.1).
///
/// Implementations must be safe to share across threads; `&self` methods take
/// interior mutability so a single `Storage` handle can back concurrent
/// executors.
pub trait Storage: Send + Sync {
    /// Create a study and return its id.
    fn create_study(&self, meta: StudyMeta) -> Result<StudyId>;

    /// Enqueue a trial with sampled parameters, returning its id.
    fn enqueue_trial(&self, study: StudyId, params: ParamSet, seed: u64) -> Result<TrialId>;

    /// Transition a trial to running.
    fn start_trial(&self, trial: TrialId) -> Result<()>;

    /// Record an intermediate metric report, keyed idempotently by `(trial, step)`.
    fn report(&self, trial: TrialId, step: u64, metrics: NamedMetrics) -> Result<()>;

    /// Finalize a trial with a terminal state and optional final metrics.
    fn complete(
        &self,
        trial: TrialId,
        state: TrialState,
        final_metrics: Option<NamedMetrics>,
    ) -> Result<()>;

    /// Load one trial record.
    fn load_trial(&self, trial: TrialId) -> Result<TrialRecord>;

    /// Load the full trial history for a study.
    fn load_history(&self, study: StudyId) -> Result<TrialHistory>;

    /// Load a study's metadata.
    fn load_meta(&self, study: StudyId) -> Result<StudyMeta>;
}

/// Internal per-study bookkeeping for the in-memory backend.
#[derive(Default)]
struct StudyEntry {
    meta: Option<StudyMeta>,
    trials: Vec<TrialId>,
}

#[derive(Default)]
struct Inner {
    next_study: u64,
    next_trial: u64,
    studies: BTreeMap<StudyId, StudyEntry>,
    trials: BTreeMap<TrialId, TrialRecord>,
}

/// A thread-safe, non-persistent [`Storage`] backend.
///
/// Everything lives behind one [`Mutex`]; contention is irrelevant at v0.1
/// scale and keeps the single-source-of-truth semantics obvious.
#[derive(Default)]
pub struct InMemoryStorage {
    inner: Mutex<Inner>,
}

impl InMemoryStorage {
    /// A fresh, empty in-memory store.
    pub fn new() -> Self {
        InMemoryStorage::default()
    }
}

impl Storage for InMemoryStorage {
    fn create_study(&self, meta: StudyMeta) -> Result<StudyId> {
        let mut inner = self.inner.lock().unwrap();
        let id = StudyId(inner.next_study);
        inner.next_study += 1;
        inner.studies.insert(
            id,
            StudyEntry {
                meta: Some(meta),
                trials: Vec::new(),
            },
        );
        Ok(id)
    }

    fn enqueue_trial(&self, study: StudyId, params: ParamSet, seed: u64) -> Result<TrialId> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.studies.contains_key(&study) {
            return Err(Error::NotFound {
                kind: "study",
                id: study.to_string(),
            });
        }
        let id = TrialId(inner.next_trial);
        inner.next_trial += 1;
        let record = TrialRecord::new(id, study, params, seed);
        inner.trials.insert(id, record);
        inner.studies.get_mut(&study).unwrap().trials.push(id);
        Ok(id)
    }

    fn start_trial(&self, trial: TrialId) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .trials
            .get_mut(&trial)
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })?;
        rec.state = TrialState::Running;
        rec.timing.started_at_ms = Some(crate::provenance::now_ms());
        Ok(())
    }

    fn report(&self, trial: TrialId, step: u64, metrics: NamedMetrics) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .trials
            .get_mut(&trial)
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })?;
        // Idempotent by (trial, step): overwrite an existing step's report.
        match rec.intermediate.iter_mut().find(|r| r.step == step) {
            Some(existing) => existing.metrics = metrics,
            None => {
                rec.intermediate.push(IntermediateReport { step, metrics });
                rec.intermediate.sort_by_key(|r| r.step);
            }
        }
        Ok(())
    }

    fn complete(
        &self,
        trial: TrialId,
        state: TrialState,
        final_metrics: Option<NamedMetrics>,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .trials
            .get_mut(&trial)
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })?;
        rec.state = state;
        rec.final_metrics = final_metrics;
        rec.timing.completed_at_ms = Some(crate::provenance::now_ms());
        Ok(())
    }

    fn load_trial(&self, trial: TrialId) -> Result<TrialRecord> {
        let inner = self.inner.lock().unwrap();
        inner
            .trials
            .get(&trial)
            .cloned()
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })
    }

    fn load_history(&self, study: StudyId) -> Result<TrialHistory> {
        let inner = self.inner.lock().unwrap();
        let entry = inner.studies.get(&study).ok_or_else(|| Error::NotFound {
            kind: "study",
            id: study.to_string(),
        })?;
        let records = entry
            .trials
            .iter()
            .filter_map(|id| inner.trials.get(id).cloned())
            .collect();
        Ok(TrialHistory::new(records))
    }

    fn load_meta(&self, study: StudyId) -> Result<StudyMeta> {
        let inner = self.inner.lock().unwrap();
        inner
            .studies
            .get(&study)
            .and_then(|e| e.meta.clone())
            .ok_or_else(|| Error::NotFound {
                kind: "study",
                id: study.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> StudyMeta {
        StudyMeta {
            name: "t".into(),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "random".into(),
            pruner_name: "none".into(),
        }
    }

    #[test]
    fn lifecycle_roundtrip() {
        let s = InMemoryStorage::new();
        let study = s.create_study(meta()).unwrap();
        let t = s.enqueue_trial(study, ParamSet::new(), 42).unwrap();
        s.start_trial(t).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.5)).unwrap();
        s.complete(
            t,
            TrialState::Complete,
            Some(NamedMetrics::single("loss", 0.3)),
        )
        .unwrap();

        let rec = s.load_trial(t).unwrap();
        assert_eq!(rec.state, TrialState::Complete);
        assert_eq!(rec.final_value("loss"), Some(0.3));
        assert_eq!(s.load_history(study).unwrap().len(), 1);
    }

    #[test]
    fn report_is_idempotent_by_step() {
        let s = InMemoryStorage::new();
        let study = s.create_study(meta()).unwrap();
        let t = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.9)).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.4)).unwrap(); // retry same step
        let rec = s.load_trial(t).unwrap();
        assert_eq!(rec.intermediate.len(), 1);
        assert_eq!(rec.intermediate[0].metrics.get("loss"), Some(0.4));
    }

    #[test]
    fn enqueue_unknown_study_errors() {
        let s = InMemoryStorage::new();
        assert!(matches!(
            s.enqueue_trial(StudyId(99), ParamSet::new(), 1),
            Err(Error::NotFound { kind: "study", .. })
        ));
    }

    #[test]
    fn provenance_and_timing_are_stamped() {
        let s = InMemoryStorage::new();
        let study = s.create_study(meta()).unwrap();
        let t = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();

        let queued = s.load_trial(t).unwrap();
        assert!(queued.timing.queued_at_ms > 0);
        assert!(queued.timing.started_at_ms.is_none());
        assert!(!queued.env.os.is_empty());

        s.start_trial(t).unwrap();
        s.complete(t, TrialState::Complete, None).unwrap();
        let done = s.load_trial(t).unwrap();
        assert!(done.timing.started_at_ms.is_some());
        assert!(done.timing.completed_at_ms.is_some());
        // wall time is defined once both ends are present.
        assert!(done.wall_time_ms().is_some());
    }
}
