//! Evolutionary sampler — the evolutionary/CMA-style method the v1.0 DoD
//! requires (PRD §28; complements Random, Grid and TPE).
//!
//! This is a real-coded genetic algorithm over the search space: it selects
//! parents from the best completed trials by tournament, combines them with
//! uniform crossover, and perturbs the child with Gaussian mutation. It is a
//! strong optimizer on rugged, multimodal and coupled landscapes where TPE's
//! independent per-parameter model struggles (e.g. Rosenbrock's curved valley),
//! and — like every sampler here — it handles conditional/hierarchical spaces
//! (§5.1) by building the child in declaration order and only sampling a
//! parameter when its branch is active.
//!
//! Determinism (§2): the per-suggestion RNG is derived from the seed and the
//! trial count, so replay with the same seed and history is bit-identical.

use crate::distribution::Distribution;
use crate::metrics::Direction;
use crate::param::{ParamSet, ParamValue};
use crate::sampler::{RandomSampler, Sampler};
use crate::space::SearchSpace;
use crate::trial::TrialHistory;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// A genetic-algorithm sampler for a single named objective.
pub struct EvolutionarySampler {
    seed: u64,
    objective: String,
    direction: Direction,
    population_size: usize,
    tournament_size: usize,
    mutation_prob: f64,
    mutation_scale: f64,
}

impl EvolutionarySampler {
    /// A sampler optimizing `objective` in `direction`.
    pub fn new(objective: impl Into<String>, direction: Direction, seed: u64) -> Self {
        EvolutionarySampler {
            seed,
            objective: objective.into(),
            direction,
            population_size: 20,
            tournament_size: 3,
            mutation_prob: 0.2,
            mutation_scale: 0.1,
        }
    }

    /// Number of random startup trials before evolution begins (default 20).
    pub fn with_population_size(mut self, n: usize) -> Self {
        self.population_size = n.max(2);
        self
    }

    /// Tournament size for parent selection (default 3).
    pub fn with_tournament_size(mut self, n: usize) -> Self {
        self.tournament_size = n.max(1);
        self
    }

    /// Per-parameter mutation probability (default 0.2).
    pub fn with_mutation_prob(mut self, p: f64) -> Self {
        self.mutation_prob = p.clamp(0.0, 1.0);
        self
    }

    /// Mutation standard deviation as a fraction of each parameter's range
    /// (default 0.1).
    pub fn with_mutation_scale(mut self, s: f64) -> Self {
        self.mutation_scale = s.max(0.0);
        self
    }

    fn loss(&self, value: f64) -> f64 {
        match self.direction {
            Direction::Minimize => value,
            Direction::Maximize => -value,
        }
    }

    /// Tournament selection: sample `tournament_size` trials, return the best.
    fn select<'a, R: Rng + ?Sized>(
        &self,
        pool: &[(&'a ParamSet, f64)],
        rng: &mut R,
    ) -> &'a ParamSet {
        let mut best: Option<(&ParamSet, f64)> = None;
        for _ in 0..self.tournament_size {
            let (p, l) = pool[rng.gen_range(0..pool.len())];
            if best.is_none_or(|(_, bl)| l < bl) {
                best = Some((p, l));
            }
        }
        best.unwrap().0
    }
}

impl Sampler for EvolutionarySampler {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ (history.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );

        let pool: Vec<(&ParamSet, f64)> = history
            .completed()
            .filter_map(|r| {
                r.final_value(&self.objective)
                    .map(|v| (&r.params, self.loss(v)))
            })
            .filter(|(_, l)| l.is_finite())
            .collect();

        // Random startup until there is a population to evolve.
        if pool.len() < self.population_size {
            return RandomSampler::sample_space(space, &mut rng);
        }

        let parent_a = self.select(&pool, &mut rng);
        let parent_b = self.select(&pool, &mut rng);

        let mut child = ParamSet::new();
        for def in space.params() {
            if !SearchSpace::is_active(def, &child) {
                continue;
            }
            // Uniform crossover, falling back across parents and finally to the
            // prior when a conditional parameter is absent from a parent.
            let take_a = rng.gen_bool(0.5);
            let chosen = if take_a {
                parent_a.get(&def.name).or_else(|| parent_b.get(&def.name))
            } else {
                parent_b.get(&def.name).or_else(|| parent_a.get(&def.name))
            };
            let mut value = match chosen {
                Some(v) => v.clone(),
                None => def.distribution.sample(&mut rng),
            };
            if rng.gen_bool(self.mutation_prob) {
                value = mutate(&def.distribution, &value, self.mutation_scale, &mut rng);
            }
            child.insert(def.name.clone(), value);
        }
        child
    }

    fn name(&self) -> &'static str {
        "evolutionary"
    }
}

/// Perturb a value within its distribution. Numeric parameters get a Gaussian
/// step (in log space when log-scaled, snapped to any step grid); categorical
/// and boolean parameters are re-drawn uniformly.
pub(crate) fn mutate<R: Rng + ?Sized>(
    dist: &Distribution,
    value: &ParamValue,
    scale: f64,
    rng: &mut R,
) -> ParamValue {
    match (dist, value) {
        (
            Distribution::Float {
                low,
                high,
                log,
                step,
            },
            ParamValue::Float(x),
        ) => {
            let (lo, hi) = if *log {
                (low.ln(), high.ln())
            } else {
                (*low, *high)
            };
            let cur = if *log { x.max(1e-300).ln() } else { *x };
            let perturbed = (cur + gaussian(rng) * scale * (hi - lo)).clamp(lo, hi);
            let mut base = if *log { perturbed.exp() } else { perturbed };
            if let Some(s) = step {
                let k = ((base - *low) / *s).round();
                base = low + k * s;
            }
            ParamValue::Float(base.clamp(*low, *high))
        }
        (
            Distribution::Int {
                low,
                high,
                log,
                step,
            },
            ParamValue::Int(x),
        ) => {
            let (lo, hi) = if *log {
                ((*low as f64).ln(), (*high as f64).ln())
            } else {
                (*low as f64, *high as f64)
            };
            let cur = if *log {
                (*x as f64).max(1.0).ln()
            } else {
                *x as f64
            };
            let perturbed = (cur + gaussian(rng) * scale * (hi - lo)).clamp(lo, hi);
            let base = if *log { perturbed.exp() } else { perturbed };
            let k = ((base - *low as f64) / *step as f64).round() as i64;
            ParamValue::Int((low + k * step).clamp(*low, *high))
        }
        // Categorical / boolean (or any type mismatch): re-draw from the prior.
        _ => dist.sample(rng),
    }
}

/// A standard-normal sample via the Box-Muller transform.
pub(crate) fn gaussian<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Condition;

    fn space() -> SearchSpace {
        SearchSpace::new().add("x", Distribution::float(-5.0, 5.0))
    }

    #[test]
    fn startup_returns_valid_random_params() {
        let mut s = EvolutionarySampler::new("loss", Direction::Minimize, 1);
        let p = s.suggest(&space(), &TrialHistory::default());
        assert!((-5.0..=5.0).contains(&p.float("x").unwrap()));
    }

    #[test]
    fn deterministic_for_same_history() {
        use crate::metrics::NamedMetrics;
        use crate::param::ParamValue;
        use crate::trial::{StudyId, TrialId, TrialRecord, TrialState};

        let mut records = Vec::new();
        for i in 0..25u64 {
            let x = -5.0 + i as f64 * 0.4;
            let mut r = TrialRecord::new(
                TrialId(i),
                StudyId(0),
                ParamSet::new().with("x", ParamValue::Float(x)),
                i,
            );
            r.state = TrialState::Complete;
            r.final_metrics = Some(NamedMetrics::single("loss", (x - 2.0).powi(2)));
            records.push(r);
        }
        let hist = TrialHistory::new(records);
        let mut s1 = EvolutionarySampler::new("loss", Direction::Minimize, 9);
        let mut s2 = EvolutionarySampler::new("loss", Direction::Minimize, 9);
        assert_eq!(s1.suggest(&space(), &hist), s2.suggest(&space(), &hist));
    }

    #[test]
    fn respects_conditional_branches() {
        let space = SearchSpace::new()
            .add("model", Distribution::categorical(["mlp", "transformer"]))
            .add_conditional(
                "d_model",
                Distribution::int(64, 128),
                Condition::when_eq("model", "transformer"),
            )
            .add_conditional(
                "hidden",
                Distribution::int(16, 32),
                Condition::when_eq("model", "mlp"),
            );
        let mut s = EvolutionarySampler::new("loss", Direction::Minimize, 2);
        for _ in 0..30 {
            let p = s.suggest(&space, &TrialHistory::default());
            match p.categorical("model").unwrap() {
                "transformer" => assert!(p.contains("d_model") && !p.contains("hidden")),
                "mlp" => assert!(p.contains("hidden") && !p.contains("d_model")),
                _ => unreachable!(),
            }
        }
    }
}
