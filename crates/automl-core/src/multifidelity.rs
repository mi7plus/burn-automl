//! Multi-fidelity optimization: Hyperband and BOHB (post-1.0; PRD §4 budgets,
//! §17 pruning/allocation).
//!
//! Median/ASHA pruning stops weak trials mid-curve; **Hyperband** goes further
//! and allocates *budget itself* as a first-class search dimension. It runs
//! several brackets of successive halving: a bracket starts many configurations
//! at a small resource, keeps the top `1/eta`, multiplies their resource by
//! `eta`, and repeats — so most configurations are only ever evaluated cheaply,
//! and full-resource training is spent on survivors. Different brackets trade off
//! *many-cheap* against *few-expensive*, hedging against the risk that cheap
//! evaluations mis-rank.
//!
//! **BOHB** is Hyperband with model-based sampling: pass a
//! [`crate::tpe::TpeSampler`] as the sampler and each new configuration is drawn
//! from a model fit to everything evaluated so far, rather than at random.
//! [`Hyperband::optimize`] takes any [`Sampler`], so BOHB is
//! `Hyperband::optimize(space, &mut TpeSampler::new(..), eval)`.
//!
//! The objective is *fidelity-aware*: it receives the resource level (e.g. epoch
//! count) to evaluate at and returns named metrics. The engine stays framework-
//! agnostic — resource is just a `u64` the objective interprets.

use crate::metrics::{Direction, NamedMetrics};
use crate::param::ParamSet;
use crate::sampler::Sampler;
use crate::space::SearchSpace;
use crate::trial::{StudyId, TrialHistory, TrialId, TrialRecord, TrialState};

/// The result of a Hyperband/BOHB run.
#[derive(Debug, Clone)]
pub struct HyperbandOutcome {
    /// The best configuration found (by score at its highest evaluated resource).
    pub best_params: ParamSet,
    /// That configuration's best score.
    pub best_score: f64,
    /// The resource level at which the best score was observed.
    pub best_resource: u64,
    /// Total resource consumed across all evaluations (the cost metric).
    pub total_resource: u64,
    /// Number of individual (config, resource) evaluations performed.
    pub evaluations: usize,
}

/// A Hyperband multi-fidelity optimizer. Pair with a [`crate::tpe::TpeSampler`]
/// for BOHB.
pub struct Hyperband {
    max_resource: u64,
    eta: usize,
    direction: Direction,
    objective: String,
}

impl Hyperband {
    /// A new optimizer with maximum resource `max_resource` per configuration and
    /// reduction factor `eta` (>= 2), optimizing the named `objective`.
    pub fn new(
        objective: impl Into<String>,
        direction: Direction,
        max_resource: u64,
        eta: usize,
    ) -> Self {
        Hyperband {
            max_resource: max_resource.max(1),
            eta: eta.max(2),
            direction,
            objective: objective.into(),
        }
    }

    /// Whether score `a` is better than `b` under the configured direction.
    fn better(&self, a: f64, b: f64) -> bool {
        match self.direction {
            Direction::Minimize => a < b,
            Direction::Maximize => a > b,
        }
    }

    /// Run all Hyperband brackets. `eval(params, resource)` evaluates a
    /// configuration at a resource level and returns its metrics (which must
    /// include the optimized objective). Configurations are drawn from `sampler`,
    /// which sees the history of everything evaluated so far — so a model-based
    /// sampler yields BOHB.
    pub fn optimize<F>(
        &self,
        space: &SearchSpace,
        sampler: &mut dyn Sampler,
        mut eval: F,
    ) -> HyperbandOutcome
    where
        F: FnMut(&ParamSet, u64) -> NamedMetrics,
    {
        let eta = self.eta as f64;
        // s_max: the number of successive-halving rungs in the most aggressive
        // bracket; each bracket s runs s+1 rungs.
        let s_max = (self.max_resource as f64).ln().div_euclid(eta.ln()) as i64;
        let s_max = s_max.max(0);
        let budget_per_bracket = (s_max as f64 + 1.0) * self.max_resource as f64;

        let mut history: Vec<TrialRecord> = Vec::new();
        let mut next_id: u64 = 0;
        let mut best: Option<(ParamSet, f64, u64)> = None;
        let mut total_resource: u64 = 0;
        let mut evaluations: usize = 0;

        // A helper to evaluate a config at a resource, record it, and update best.
        // Returns the objective score.
        let mut evaluate = |params: &ParamSet,
                            resource: u64,
                            history: &mut Vec<TrialRecord>,
                            next_id: &mut u64,
                            best: &mut Option<(ParamSet, f64, u64)>,
                            total_resource: &mut u64,
                            evaluations: &mut usize|
         -> f64 {
            let metrics = eval(params, resource);
            let score = metrics.get(&self.objective).unwrap_or(f64::NAN);
            *total_resource += resource;
            *evaluations += 1;

            let mut rec = TrialRecord::new(TrialId(*next_id), StudyId(0), params.clone(), *next_id);
            *next_id += 1;
            rec.state = TrialState::Complete;
            rec.final_metrics = Some(metrics);
            history.push(rec);

            // Choose the winner only among full-resource evaluations: cheap
            // rungs rank configs but their optimistic noise must not decide
            // the result. Every bracket's final rung runs at `max_resource`.
            if score.is_finite()
                && resource >= self.max_resource
                && best.as_ref().is_none_or(|(_, b, _)| self.better(score, *b))
            {
                *best = Some((params.clone(), score, resource));
            }
            score
        };

        for s in (0..=s_max).rev() {
            // Initial configuration count and resource for bracket s.
            let n = (budget_per_bracket / self.max_resource as f64 * eta.powi(s as i32)
                / (s as f64 + 1.0))
                .ceil() as usize;
            let n = n.max(1);
            let r0 = self.max_resource as f64 * eta.powi(-(s as i32));

            // Successive halving: s+1 rungs, resource growing by eta each rung.
            let mut configs: Vec<ParamSet> = Vec::new();
            for i in 0..=s {
                let n_i = (n as f64 * eta.powi(-(i as i32))).floor() as usize;
                let r_i = (r0 * eta.powi(i as i32)).round().max(1.0) as u64;
                let r_i = r_i.min(self.max_resource);

                let mut scored: Vec<(f64, ParamSet)> = if i == 0 {
                    // Rung 0: sample and evaluate one config at a time, so the
                    // sampler sees a growing history and returns *distinct*
                    // configs (and a model-based sampler conditions on prior
                    // evaluations — the BOHB behaviour).
                    (0..n)
                        .map(|_| {
                            let h = TrialHistory::new(history.clone());
                            let c = sampler.suggest(space, &h);
                            let score = evaluate(
                                &c,
                                r_i,
                                &mut history,
                                &mut next_id,
                                &mut best,
                                &mut total_resource,
                                &mut evaluations,
                            );
                            (score, c)
                        })
                        .collect()
                } else {
                    // Later rungs: re-evaluate the surviving configs at higher
                    // resource.
                    configs
                        .iter()
                        .map(|c| {
                            let score = evaluate(
                                c,
                                r_i,
                                &mut history,
                                &mut next_id,
                                &mut best,
                                &mut total_resource,
                                &mut evaluations,
                            );
                            (score, c.clone())
                        })
                        .collect()
                };

                // Keep the top 1/eta for the next rung.
                let keep = (n_i as f64 / eta).floor() as usize;
                if keep == 0 || i == s {
                    break;
                }
                scored.sort_by(|a, b| match self.direction {
                    Direction::Minimize => {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    Direction::Maximize => {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
                configs = scored.into_iter().take(keep).map(|(_, c)| c).collect();
                if configs.is_empty() {
                    break;
                }
            }
        }

        let (best_params, best_score, best_resource) =
            best.unwrap_or_else(|| (ParamSet::new(), f64::NAN, 0));
        HyperbandOutcome {
            best_params,
            best_score,
            best_resource,
            total_resource,
            evaluations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::sampler::RandomSampler;
    use crate::tpe::TpeSampler;

    /// A multi-fidelity objective: the true loss is (x-2)^2, and a low-resource
    /// evaluation adds noise that shrinks as resource grows — cheap evaluations
    /// rank configs approximately, full-resource ones exactly.
    fn objective(seed_base: u64) -> impl FnMut(&ParamSet, u64) -> NamedMetrics {
        use rand::{Rng, SeedableRng};
        move |p: &ParamSet, resource: u64| {
            let x = p.float("x").unwrap();
            let true_loss = (x - 2.0).powi(2);
            // Deterministic pseudo-noise keyed by the config and resource.
            let key = (x * 1000.0) as i64 as u64 ^ resource.wrapping_mul(0x9E37) ^ seed_base;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(key);
            let noise = rng.gen_range(-1.0..1.0) / resource as f64;
            NamedMetrics::single("loss", true_loss + noise)
        }
    }

    #[test]
    fn hyperband_finds_the_optimum_with_multi_fidelity() {
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let hb = Hyperband::new("loss", Direction::Minimize, 27, 3);
        let mut sampler = RandomSampler::new(1);
        let out = hb.optimize(&space, &mut sampler, objective(1));

        assert!(out.evaluations > 0);
        assert!(out.total_resource > 0);
        // The best config is near x = 2 (the true optimum).
        assert!(
            (out.best_params.float("x").unwrap() - 2.0).abs() < 0.6,
            "best x = {:?}, score {}",
            out.best_params.float("x"),
            out.best_score
        );
    }

    #[test]
    fn successive_halving_saves_resource_vs_all_at_max() {
        // Hyperband evaluates most configs cheaply, so its total resource is far
        // below evaluating every sampled config at the maximum resource.
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let hb = Hyperband::new("loss", Direction::Minimize, 27, 3);
        let mut sampler = RandomSampler::new(2);
        let out = hb.optimize(&space, &mut sampler, objective(2));
        // If every one of the `evaluations` runs had used the max resource:
        let naive = out.evaluations as u64 * 27;
        assert!(
            out.total_resource < naive,
            "total {} should beat naive {}",
            out.total_resource,
            naive
        );
    }

    #[test]
    fn bohb_pairs_hyperband_with_tpe() {
        // BOHB: the same runner driven by a model-based sampler.
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let hb = Hyperband::new("loss", Direction::Minimize, 27, 3);
        let mut sampler = TpeSampler::new("loss", Direction::Minimize, 3);
        let out = hb.optimize(&space, &mut sampler, objective(3));
        assert!((out.best_params.float("x").unwrap() - 2.0).abs() < 0.6);
    }

    #[test]
    fn deterministic_for_a_fixed_seed() {
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let run = || {
            let hb = Hyperband::new("loss", Direction::Minimize, 27, 3);
            let mut sampler = RandomSampler::new(7);
            hb.optimize(&space, &mut sampler, objective(7))
                .best_params
                .float("x")
                .unwrap()
        };
        assert_eq!(run(), run());
    }
}
