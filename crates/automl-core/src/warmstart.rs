//! Study warm-starting and transfer (post-1.0).
//!
//! A new study rarely starts from nothing: a related study has usually already
//! found promising regions, and re-exploring them from scratch wastes budget.
//! Warm-starting seeds the search with known-good configurations — the best
//! trials of a prior study — which are evaluated first, before the sampler takes
//! over. Because those seed trials then enter the history, a model-based sampler
//! (TPE) immediately conditions on them, transferring the prior's structure into
//! the new run.
//!
//! This is a *composable* mechanism: [`WarmStartSampler`] wraps any [`Sampler`]
//! and replays a queue of seed [`ParamSet`]s, so it needs no change to the study
//! loop. [`best_configs`] extracts the seed set from a prior study's history.

use crate::metrics::Direction;
use crate::param::ParamSet;
use crate::sampler::Sampler;
use crate::space::SearchSpace;
use crate::trial::{TrialHistory, TrialRecord};
use std::collections::VecDeque;
use std::sync::Mutex;

/// The `n` best configurations from a prior study's history for `objective`,
/// best-first. Only completed trials that reported the objective are considered;
/// ties break by ascending trial id for determinism. Transfer these into a new
/// study with [`WarmStartSampler`].
pub fn best_configs(
    history: &TrialHistory,
    objective: &str,
    direction: Direction,
    n: usize,
) -> Vec<ParamSet> {
    let mut scored: Vec<(f64, u64, &TrialRecord)> = history
        .completed()
        .filter_map(|r| r.final_value(objective).map(|v| (v, r.id.0, r)))
        .filter(|(v, _, _)| v.is_finite())
        .collect();
    scored.sort_by(|a, b| {
        let ord = match direction {
            Direction::Minimize => a.0.partial_cmp(&b.0),
            Direction::Maximize => b.0.partial_cmp(&a.0),
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        ord.then(a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .take(n)
        .map(|(_, _, r)| r.params.clone())
        .collect()
}

/// A sampler that replays a queue of seed configurations before delegating to an
/// inner sampler.
///
/// Each `suggest` first pops a seed (filtered to the parameters the space
/// actually defines, so a seed from a slightly different space still applies its
/// overlapping dimensions); once the seeds are exhausted it defers entirely to
/// the inner sampler. The seeds' results enter the study history, so a
/// model-based inner sampler benefits from them immediately.
pub struct WarmStartSampler<S: Sampler> {
    inner: S,
    seeds: Mutex<VecDeque<ParamSet>>,
}

impl<S: Sampler> WarmStartSampler<S> {
    /// Wrap `inner`, replaying `seeds` (in order) before the inner sampler runs.
    pub fn new(inner: S, seeds: impl IntoIterator<Item = ParamSet>) -> Self {
        WarmStartSampler {
            inner,
            seeds: Mutex::new(seeds.into_iter().collect()),
        }
    }

    /// How many seed configurations remain unplayed.
    pub fn remaining(&self) -> usize {
        self.seeds.lock().unwrap().len()
    }

    /// Keep only the seed parameters this space defines, so a transferred config
    /// from a related-but-different space contributes its overlapping dimensions
    /// rather than being rejected.
    fn project(space: &SearchSpace, seed: ParamSet) -> ParamSet {
        let mut out = ParamSet::new();
        for def in space.params() {
            if let Some(v) = seed.get(&def.name) {
                out.insert(def.name.clone(), v.clone());
            }
        }
        out
    }
}

impl<S: Sampler> Sampler for WarmStartSampler<S> {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        if let Some(seed) = self.seeds.lock().unwrap().pop_front() {
            let projected = Self::project(space, seed);
            // A projected seed missing space parameters would be an invalid trial;
            // fall back to the inner sampler only if it is empty.
            if !projected.is_empty() {
                return projected;
            }
        }
        self.inner.suggest(space, history)
    }

    fn on_trial_complete(&mut self, trial: &TrialRecord) {
        self.inner.on_trial_complete(trial);
    }

    fn name(&self) -> &'static str {
        "warm-start"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::param::ParamValue;
    use crate::sampler::RandomSampler;
    use crate::trial::{StudyId, TrialId, TrialRecord, TrialState};

    fn record(id: u64, x: f64, loss: f64) -> TrialRecord {
        let mut r = TrialRecord::new(
            TrialId(id),
            StudyId(0),
            ParamSet::new().with("x", ParamValue::Float(x)),
            id,
        );
        r.state = TrialState::Complete;
        r.final_metrics = Some(crate::metrics::NamedMetrics::single("loss", loss));
        r
    }

    #[test]
    fn best_configs_returns_top_n_in_order() {
        let history = TrialHistory::new(vec![
            record(0, 1.0, 0.5),
            record(1, 2.0, 0.1),
            record(2, 3.0, 0.9),
            record(3, 4.0, 0.2),
        ]);
        let top = best_configs(&history, "loss", Direction::Minimize, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].float("x").unwrap(), 2.0); // loss 0.1
        assert_eq!(top[1].float("x").unwrap(), 4.0); // loss 0.2
    }

    #[test]
    fn warm_start_replays_seeds_then_delegates() {
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let seeds = vec![
            ParamSet::new().with("x", ParamValue::Float(2.0)),
            ParamSet::new().with("x", ParamValue::Float(-2.0)),
        ];
        let mut sampler = WarmStartSampler::new(RandomSampler::new(1), seeds);
        let history = TrialHistory::default();

        assert_eq!(sampler.remaining(), 2);
        assert_eq!(sampler.suggest(&space, &history).float("x").unwrap(), 2.0);
        assert_eq!(sampler.suggest(&space, &history).float("x").unwrap(), -2.0);
        assert_eq!(sampler.remaining(), 0);
        // Seeds exhausted: now the inner random sampler drives, in range.
        let v = sampler.suggest(&space, &history).float("x").unwrap();
        assert!((-5.0..=5.0).contains(&v));
    }

    #[test]
    fn warm_start_projects_onto_the_space() {
        // A seed carrying an extra parameter keeps only what the space defines.
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let seed = ParamSet::new()
            .with("x", ParamValue::Float(1.0))
            .with("obsolete", ParamValue::Int(9));
        let mut sampler = WarmStartSampler::new(RandomSampler::new(1), [seed]);
        let s = sampler.suggest(&space, &TrialHistory::default());
        assert_eq!(s.float("x").unwrap(), 1.0);
        assert!(s.get("obsolete").is_none());
    }

    #[test]
    fn warm_started_study_finds_the_optimum_immediately() {
        use crate::objective::ReportSink;
        use crate::study::Study;

        // Seed the exact optimum of (x-2)^2 + (y+1)^2; the first trial nails it.
        let space = SearchSpace::new()
            .add("x", Distribution::float(-5.0, 5.0))
            .add("y", Distribution::float(-5.0, 5.0));
        let seed = ParamSet::new()
            .with("x", ParamValue::Float(2.0))
            .with("y", ParamValue::Float(-1.0));
        let sampler = WarmStartSampler::new(
            crate::tpe::TpeSampler::new("loss", Direction::Minimize, 1),
            [seed],
        );
        let mut study = Study::builder(space)
            .minimize("loss")
            .sampler(sampler)
            .seed(1)
            .build()
            .unwrap();
        study
            .optimize_n(
                &|p: &ParamSet, _s: &mut dyn ReportSink| {
                    let (x, y) = (p.float("x")?, p.float("y")?);
                    Ok(crate::metrics::NamedMetrics::single(
                        "loss",
                        (x - 2.0).powi(2) + (y + 1.0).powi(2),
                    ))
                },
                5,
            )
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        assert!(
            best.final_value("loss").unwrap() < 1e-9,
            "warm start should evaluate the seeded optimum first"
        );
    }
}
