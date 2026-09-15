//! `Study`: the top-level handle that drives the universal optimization loop.
//!
//! This is the low-level surface the PRD calls the slowest-moving API (§22): a
//! study owns directions, budget, sampler, pruner and storage, and runs the
//! loop from §4:
//!
//! ```text
//! Sampler -> params -> Storage(enqueue) -> Objective -> Reports -> Pruner
//!         -> Storage(complete) -> Sampler.observe -> next trial
//! ```
//!
//! v0.1 executes trials **sequentially in-process** (the deterministic
//! debugging tier from §18.1); the thread/process/distributed executors layer
//! in behind the same loop in later sprints (§23, §29 item 8).

use crate::budget::{Budget, Consumption, TrialBudget};
use crate::error::Result;
use crate::executor::{Executor, SequentialExecutor};
use crate::metrics::{Direction, NamedMetrics};
use crate::objective::{Objective, ReportSink};
use crate::pruner::{NoPruner, Pruner};
use crate::sampler::Sampler;
use crate::space::SearchSpace;
use crate::storage::{InMemoryStorage, Storage, StudyMeta};
use crate::trial::{StudyId, TrialHistory, TrialId, TrialRecord, TrialState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A configured optimization study, ready to run an objective.
pub struct Study {
    id: StudyId,
    space: SearchSpace,
    storage: Arc<dyn Storage>,
    sampler: Box<dyn Sampler>,
    pruner: Arc<dyn Pruner>,
    executor: Arc<dyn Executor>,
    directions: Vec<(String, Direction)>,
    base_seed: u64,
}

impl Study {
    /// Start building a study over the given search space.
    pub fn builder(space: SearchSpace) -> StudyBuilder {
        StudyBuilder::new(space)
    }

    /// Reconnect to an existing persisted study and continue optimizing it.
    ///
    /// The study's directions are loaded from storage (its provenance), while
    /// the search space, sampler and pruner are supplied by the caller — these
    /// live in code, not the database. Prior trials remain in `history()` and
    /// are observed by the sampler, so a resumed run picks up where it left off
    /// without losing completed trials (PRD §26 recovery gate). This is the
    /// counterpart to a persistent backend such as
    /// [`crate::sqlite::SqliteStorage`].
    pub fn resume(
        space: SearchSpace,
        storage: Arc<dyn Storage>,
        study_id: StudyId,
        sampler: impl Sampler + 'static,
        pruner: Arc<dyn Pruner>,
        base_seed: u64,
    ) -> Result<Study> {
        space.validate()?;
        let meta = storage.load_meta(study_id)?;
        Ok(Study {
            id: study_id,
            space,
            storage,
            sampler: Box::new(sampler),
            pruner,
            executor: Arc::new(SequentialExecutor),
            directions: meta.directions,
            base_seed,
        })
    }

    /// The study's storage id.
    pub fn id(&self) -> StudyId {
        self.id
    }

    /// The objectives and their directions.
    pub fn directions(&self) -> &[(String, Direction)] {
        &self.directions
    }

    /// The full trial history recorded so far.
    pub fn history(&self) -> Result<TrialHistory> {
        self.storage.load_history(self.id)
    }

    /// Run the optimization loop until `budget` is exhausted.
    ///
    /// Returns the number of trials completed (including pruned/failed). The
    /// loop is deterministic given the sampler seed and objective.
    pub fn optimize<O, B>(&mut self, objective: &O, budget: &B) -> Result<usize>
    where
        O: Objective,
        B: Budget + ?Sized,
    {
        let start = Instant::now();
        let mut consumption = Consumption::new();

        // Trials already recorded (non-zero when resuming a persisted study);
        // used to offset per-trial seeds so a resumed run does not replay the
        // seeds of the original run.
        let prior_trials = self.storage.load_history(self.id)?.len() as u64;
        let batch_size = self.executor.preferred_batch_size().max(1);

        'outer: loop {
            if budget.is_exhausted(&consumption) {
                break;
            }

            // Ask the sampler for a batch. Each proposal is enqueued as a
            // running trial *before* the next `suggest`, so successive calls see
            // a growing history and stay diverse even under parallel execution.
            let mut jobs: Vec<crate::executor::Job> = Vec::with_capacity(batch_size);
            let mut batch_ids: Vec<TrialId> = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let mut projected = consumption.clone();
                projected.trials += batch_ids.len() as u64;
                if budget.is_exhausted(&projected) {
                    break;
                }

                let history = self.storage.load_history(self.id)?;
                let params = self.sampler.suggest(&self.space, &history);

                // Per-trial seed from the study seed and a globally unique trial
                // index keeps each trial independently reproducible (§19).
                let index = prior_trials + consumption.trials + batch_ids.len() as u64;
                let seed = self
                    .base_seed
                    .wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));

                let trial_id = self.storage.enqueue_trial(self.id, params.clone(), seed)?;
                self.storage.start_trial(trial_id)?;
                batch_ids.push(trial_id);

                // A self-contained job: evaluate the objective and persist the
                // outcome. It touches only storage (thread-safe) and the shared
                // objective, never the sampler, so it is safe to run in parallel.
                let storage = self.storage.clone();
                let pruner = self.pruner.clone();
                let study_id = self.id;
                jobs.push(Box::new(move || {
                    let mut sink =
                        StudyReportSink::new(trial_id, study_id, storage.clone(), pruner);
                    let outcome = objective.evaluate(&params, &mut sink);
                    let state = match &outcome {
                        Ok(_) if sink.was_pruned() => TrialState::Pruned,
                        Ok(_) => TrialState::Complete,
                        Err(_) => TrialState::Failed,
                    };
                    let _ = storage.complete(trial_id, state, outcome.ok());
                }));
            }

            if batch_ids.is_empty() {
                break 'outer;
            }

            self.executor.run_all(jobs);

            // Observe completed trials in deterministic (enqueue) order, and
            // tally the training resources they consumed so epoch/step budgets
            // (§4) actually fire. A trial's last reported step is its epoch/step
            // count; the number of reports is its intermediate-step count.
            for trial_id in &batch_ids {
                if let Ok(record) = self.storage.load_trial(*trial_id) {
                    consumption.epochs += record.intermediate.last().map(|r| r.step).unwrap_or(0);
                    consumption.steps += record.intermediate.len() as u64;
                    self.sampler.on_trial_complete(&record);
                }
            }

            consumption.trials += batch_ids.len() as u64;
            consumption.wall_time = start.elapsed();
        }
        Ok(consumption.trials as usize)
    }

    /// Convenience: run for a fixed number of trials.
    pub fn optimize_n<O: Objective>(&mut self, objective: &O, n: u64) -> Result<usize> {
        self.optimize(objective, &TrialBudget { max_trials: n })
    }

    /// The best completed trial for a single-objective study.
    ///
    /// Returns `None` if the study is multi-objective (use the Pareto front,
    /// arriving in a later release) or has no completed trials.
    pub fn best_trial(&self) -> Result<Option<TrialRecord>> {
        let Some((name, direction)) = self.single_objective() else {
            return Ok(None);
        };
        let history = self.storage.load_history(self.id)?;
        let best = history
            .completed()
            .filter_map(|r| r.final_value(&name).map(|v| (v, r)))
            .fold(None::<(f64, &TrialRecord)>, |acc, (v, r)| match acc {
                Some((bv, _)) if !direction.is_better(v, bv) => acc,
                _ => Some((v, r)),
            })
            .map(|(_, r)| r.clone());
        Ok(best)
    }

    /// The Pareto front of completed trials for a multi-objective study
    /// (PRD §17.1). Completed trials are inserted in id order; the returned
    /// [`crate::pareto::ParetoFront`] holds the non-dominated set. For a
    /// single-objective study this still works (the front collapses to the best
    /// trial), but [`Study::best_trial`] is the idiomatic accessor there.
    pub fn pareto_front(&self) -> Result<crate::pareto::ParetoFront> {
        let objectives = self
            .directions
            .iter()
            .map(|(name, direction)| crate::metrics::Objective {
                name: name.clone(),
                direction: *direction,
            })
            .collect();
        let mut front = crate::pareto::ParetoFront::new(objectives);
        for rec in self.storage.load_history(self.id)?.completed() {
            if let Some(metrics) = &rec.final_metrics {
                front.insert(rec.id, metrics);
            }
        }
        Ok(front)
    }

    /// Per-parameter importance for the study's first objective, most-important
    /// first (PRD §25). Returns a plain data structure with no UI dependency.
    pub fn importance(&self) -> Result<Vec<crate::importance::ParamImportance>> {
        let objective = self
            .directions
            .first()
            .map(|(n, _)| n.clone())
            .unwrap_or_default();
        let history = self.storage.load_history(self.id)?;
        Ok(crate::importance::importance(&history, &objective))
    }

    fn single_objective(&self) -> Option<(String, Direction)> {
        if self.directions.len() == 1 {
            self.directions.first().cloned()
        } else {
            None
        }
    }
}

/// A [`ReportSink`] that persists intermediate metrics and evaluates pruning at
/// each step. It records whether the pruner asked to stop so the study can mark
/// the trial `Pruned` when the objective returns.
struct StudyReportSink {
    trial_id: TrialId,
    study_id: StudyId,
    storage: Arc<dyn Storage>,
    pruner: Arc<dyn Pruner>,
    stop: AtomicBool,
}

impl StudyReportSink {
    fn new(
        trial_id: TrialId,
        study_id: StudyId,
        storage: Arc<dyn Storage>,
        pruner: Arc<dyn Pruner>,
    ) -> Self {
        StudyReportSink {
            trial_id,
            study_id,
            storage,
            pruner,
            stop: AtomicBool::new(false),
        }
    }

    fn was_pruned(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

impl ReportSink for StudyReportSink {
    fn trial_id(&self) -> TrialId {
        self.trial_id
    }

    fn report(&mut self, step: u64, metrics: NamedMetrics) -> Result<()> {
        self.storage.report(self.trial_id, step, metrics)?;
        // Evaluate pruning against the current history and this trial's curve.
        let history = self.storage.load_history(self.study_id)?;
        let record = self.storage.load_trial(self.trial_id)?;
        let progress = crate::trial::TrialProgress {
            id: self.trial_id,
            intermediate: &record.intermediate,
        };
        if self.pruner.should_prune(&progress, &history) {
            self.stop.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

/// Builder for [`Study`], mirroring the ergonomics sketched in PRD §20.
pub struct StudyBuilder {
    name: String,
    space: SearchSpace,
    directions: Vec<(String, Direction)>,
    storage: Option<Arc<dyn Storage>>,
    sampler: Option<Box<dyn Sampler>>,
    pruner: Option<Arc<dyn Pruner>>,
    executor: Option<Arc<dyn Executor>>,
    base_seed: u64,
}

impl StudyBuilder {
    /// Start a builder over the given space.
    pub fn new(space: SearchSpace) -> Self {
        StudyBuilder {
            name: "study".to_string(),
            space,
            directions: Vec::new(),
            storage: None,
            sampler: None,
            pruner: None,
            executor: None,
            base_seed: 0,
        }
    }

    /// Set a human-readable study name (provenance).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add an objective to minimize.
    pub fn minimize(mut self, objective: impl Into<String>) -> Self {
        self.directions
            .push((objective.into(), Direction::Minimize));
        self
    }

    /// Add an objective to maximize.
    pub fn maximize(mut self, objective: impl Into<String>) -> Self {
        self.directions
            .push((objective.into(), Direction::Maximize));
        self
    }

    /// Provide a sampler (defaults to random if unset).
    pub fn sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(Box::new(sampler));
        self
    }

    /// Provide a pruner (defaults to [`NoPruner`] if unset).
    pub fn pruner(mut self, pruner: impl Pruner + 'static) -> Self {
        self.pruner = Some(Arc::new(pruner));
        self
    }

    /// Provide a storage backend (defaults to in-memory if unset).
    pub fn storage(mut self, storage: Arc<dyn Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Provide an executor tier (defaults to
    /// [`SequentialExecutor`](crate::executor::SequentialExecutor) if unset).
    /// A parallel executor's `preferred_batch_size` becomes the ask-batch size.
    pub fn executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Set the base seed for deterministic replay.
    pub fn seed(mut self, seed: u64) -> Self {
        self.base_seed = seed;
        self
    }

    /// Validate the configuration and create the backing study record.
    pub fn build(self) -> Result<Study> {
        self.space.validate()?;
        let directions = if self.directions.is_empty() {
            // Default to a single minimized "objective" metric.
            vec![("objective".to_string(), Direction::Minimize)]
        } else {
            self.directions
        };
        let storage = self
            .storage
            .unwrap_or_else(|| Arc::new(InMemoryStorage::new()));
        let sampler = self
            .sampler
            .unwrap_or_else(|| Box::new(crate::sampler::RandomSampler::new(self.base_seed)));
        let pruner: Arc<dyn Pruner> = self.pruner.unwrap_or_else(|| Arc::new(NoPruner));
        let executor: Arc<dyn Executor> = self
            .executor
            .unwrap_or_else(|| Arc::new(SequentialExecutor));

        let meta = StudyMeta {
            name: self.name,
            directions: directions.clone(),
            sampler_name: sampler.name().to_string(),
            pruner_name: pruner.name().to_string(),
        };
        let id = storage.create_study(meta)?;

        Ok(Study {
            id,
            space: self.space,
            storage,
            sampler,
            pruner,
            executor,
            directions,
            base_seed: self.base_seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::objective::ReportSink;
    use crate::param::ParamSet;
    use crate::sampler::RandomSampler;

    // A convex 1-D objective: minimize (x - 2)^2 over x in [-5, 5].
    fn quadratic(params: &ParamSet, _sink: &mut dyn ReportSink) -> Result<NamedMetrics> {
        let x = params.float("x")?;
        Ok(NamedMetrics::single("loss", (x - 2.0).powi(2)))
    }

    fn space() -> SearchSpace {
        SearchSpace::new().add("x", Distribution::float(-5.0, 5.0))
    }

    #[test]
    fn optimize_runs_and_finds_reasonable_min() {
        let mut study = Study::builder(space())
            .minimize("loss")
            .sampler(RandomSampler::new(123))
            .seed(123)
            .build()
            .unwrap();
        let n = study.optimize_n(&quadratic, 200).unwrap();
        assert_eq!(n, 200);
        let best = study.best_trial().unwrap().unwrap();
        let x = best.params.float("x").unwrap();
        // With 200 random samples the best x should be near 2.
        assert!((x - 2.0).abs() < 0.5, "best x = {x}");
    }

    #[test]
    fn epoch_budget_stops_after_total_epochs() {
        use crate::budget::EpochBudget;
        // Each trial reports 5 epochs, so consumption grows by 5 per trial.
        let curve_obj = |_p: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            for step in 1..=5u64 {
                sink.report(step, NamedMetrics::single("loss", 0.1))?;
            }
            Ok(NamedMetrics::single("loss", 0.1))
        };
        let mut study = Study::builder(space())
            .minimize("loss")
            .sampler(RandomSampler::new(1))
            .seed(1)
            .build()
            .unwrap();
        // Budget of 12 epochs: trial1 -> 5, trial2 -> 10, trial3 -> 15 (>=12 stops
        // the loop before trial4). Sequential executor checks the budget per trial.
        let n = study
            .optimize(&curve_obj, &EpochBudget { max_epochs: 12 })
            .unwrap();
        assert_eq!(n, 3, "expected to stop after 3 trials (15 epochs) got {n}");
    }

    #[test]
    fn thread_executor_runs_all_trials_concurrently() {
        use crate::executor::ThreadExecutor;

        let mut study = Study::builder(space())
            .minimize("loss")
            .sampler(RandomSampler::new(9))
            .executor(Arc::new(ThreadExecutor::new(4)))
            .seed(9)
            .build()
            .unwrap();
        let n = study.optimize_n(&quadratic, 200).unwrap();
        assert_eq!(n, 200);
        // Every trial must have been evaluated and completed.
        let history = study.history().unwrap();
        assert_eq!(history.len(), 200);
        assert_eq!(history.completed().count(), 200);
        // The search still works under parallelism.
        let best = study.best_trial().unwrap().unwrap();
        assert!((best.params.float("x").unwrap() - 2.0).abs() < 0.6);
    }

    #[test]
    fn multi_objective_study_builds_a_pareto_front() {
        // Two conflicting objectives over x in [0, 10]:
        //   minimize `err`  = x
        //   maximize `size` = x   (they trade off perfectly)
        // Every distinct x is non-dominated, so the front has many members.
        let space = SearchSpace::new().add("x", Distribution::float(0.0, 10.0));
        let mut study = Study::builder(space)
            .minimize("err")
            .maximize("size")
            .sampler(RandomSampler::new(3))
            .seed(3)
            .build()
            .unwrap();
        study
            .optimize_n(
                &|p: &ParamSet, _s: &mut dyn ReportSink| {
                    let x = p.float("x")?;
                    Ok(NamedMetrics::new().with("err", x).with("size", x))
                },
                40,
            )
            .unwrap();

        // best_trial is None for multi-objective studies.
        assert!(study.best_trial().unwrap().is_none());

        let front = study.pareto_front().unwrap();
        assert!(
            front.len() > 1,
            "expected a multi-point front, got {}",
            front.len()
        );
        // Hypervolume relative to a worst-case reference is positive.
        let hv = front.hypervolume(&NamedMetrics::new().with("err", 10.0).with("size", 0.0));
        assert!(hv > 0.0, "hv = {hv}");
    }

    #[test]
    fn tpe_drives_full_loop_and_converges() {
        use crate::metrics::Direction;
        use crate::tpe::TpeSampler;

        let mut study = Study::builder(space())
            .minimize("loss")
            .sampler(TpeSampler::new("loss", Direction::Minimize, 42))
            .seed(42)
            .build()
            .unwrap();
        study.optimize_n(&quadratic, 80).unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let x = best.params.float("x").unwrap();
        // TPE should reliably pin a smooth 1-D convex objective near its optimum.
        assert!((x - 2.0).abs() < 0.4, "best x = {x}");
    }

    #[test]
    fn replay_is_deterministic() {
        let run = || {
            let mut s = Study::builder(space())
                .minimize("loss")
                .sampler(RandomSampler::new(7))
                .seed(7)
                .build()
                .unwrap();
            s.optimize_n(&quadratic, 50).unwrap();
            s.best_trial().unwrap().unwrap().params.float("x").unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn pruning_marks_trials_pruned() {
        use crate::metrics::Direction;
        use crate::pruner::MedianPruner;

        // Objective reports a curve and honors should_stop.
        let curve_obj = |params: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let offset = params.float("x")?; // higher x => worse curve
            let mut last = 0.0;
            for step in 1..=10u64 {
                last = offset + step as f64 * 0.0; // flat-ish, offset dominates
                sink.report(step, NamedMetrics::single("loss", offset))?;
                if sink.should_stop() {
                    break;
                }
            }
            Ok(NamedMetrics::single("loss", last.max(offset)))
        };

        let mut study = Study::builder(SearchSpace::new().add("x", Distribution::float(0.0, 10.0)))
            .minimize("loss")
            .sampler(RandomSampler::new(5))
            .pruner(MedianPruner::new("loss", Direction::Minimize).with_min_trials(1))
            .seed(5)
            .build()
            .unwrap();
        study.optimize_n(&curve_obj, 30).unwrap();
        let history = study.history().unwrap();
        let pruned = history
            .records()
            .iter()
            .filter(|r| r.state == TrialState::Pruned)
            .count();
        // With a median pruner and varied offsets, some trials must be pruned.
        assert!(pruned > 0, "expected some pruned trials, got {pruned}");
    }
}
