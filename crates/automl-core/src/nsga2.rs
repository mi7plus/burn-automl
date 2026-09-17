//! NSGA-II multi-objective sampler (post-1.0; PRD §17).
//!
//! The `TpeSampler` and `EvolutionarySampler` optimize a single objective; a
//! multi-objective study using them samples effectively at random with respect to
//! the trade-off surface. NSGA-II is the standard evolutionary answer: it evolves
//! a population toward the whole Pareto front at once, selecting parents by
//! **non-dominated rank** (which front a solution lies on) and, within a front,
//! **crowding distance** (how isolated it is), which spreads solutions evenly
//! along the trade-off rather than clumping them.
//!
//! Implemented as an ordinary [`Sampler`]: each `suggest` ranks the archive,
//! keeps the elite population (survival selection), binary-tournament-selects two
//! parents by crowded comparison, and produces one child by uniform crossover and
//! local (Gaussian) mutation — correct over conditional spaces like the other
//! samplers.

use crate::metrics::Direction;
use crate::param::ParamSet;
use crate::sampler::{RandomSampler, Sampler};
use crate::space::SearchSpace;
use crate::trial::TrialHistory;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A multi-objective sampler implementing NSGA-II selection.
pub struct Nsga2Sampler {
    objectives: Vec<(String, Direction)>,
    seed: u64,
    n_startup: usize,
    population_size: usize,
    mutation_prob: f64,
    mutation_scale: f64,
}

impl Nsga2Sampler {
    /// A sampler for the given named objectives and directions.
    pub fn new(objectives: Vec<(String, Direction)>, seed: u64) -> Self {
        Nsga2Sampler {
            objectives,
            seed,
            n_startup: 12,
            population_size: 24,
            mutation_prob: 0.2,
            mutation_scale: 0.15,
        }
    }

    /// Gaussian mutation step size, as a fraction of each parameter's range
    /// (default 0.15).
    pub fn with_mutation_scale(mut self, s: f64) -> Self {
        self.mutation_scale = s.max(0.0);
        self
    }

    /// Number of random trials before evolutionary selection begins (default 12).
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.n_startup = n;
        self
    }

    /// The elite mating-pool size retained across the archive (default 24).
    pub fn with_population_size(mut self, n: usize) -> Self {
        self.population_size = n.max(2);
        self
    }

    /// Per-parameter probability of resampling during mutation (default 0.1).
    pub fn with_mutation_prob(mut self, p: f64) -> Self {
        self.mutation_prob = p.clamp(0.0, 1.0);
        self
    }

    /// The objective vector of a record, normalized so **lower is better** on
    /// every axis (maximize objectives are negated), or `None` if any objective
    /// is missing.
    fn objective_vec(&self, params_value: impl Fn(&str) -> Option<f64>) -> Option<Vec<f64>> {
        self.objectives
            .iter()
            .map(|(name, dir)| {
                params_value(name).map(|v| match dir {
                    Direction::Minimize => v,
                    Direction::Maximize => -v,
                })
            })
            .collect()
    }
}

/// Whether `a` dominates `b` (both minimization-normalized): no worse on all
/// objectives and strictly better on at least one.
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }
        if x < y {
            strictly = true;
        }
    }
    strictly
}

/// Fast non-dominated sort: returns the front rank (0 = best) of each individual.
fn non_dominated_ranks(pop: &[Vec<f64>]) -> Vec<usize> {
    let n = pop.len();
    let mut dominated_by = vec![0usize; n]; // how many dominate i
    let mut dominates_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rank = vec![0usize; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&pop[i], &pop[j]) {
                dominates_set[i].push(j);
            } else if dominates(&pop[j], &pop[i]) {
                dominated_by[i] += 1;
            }
        }
    }

    let mut current: Vec<usize> = (0..n).filter(|&i| dominated_by[i] == 0).collect();
    let mut front = 0;
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            rank[i] = front;
            for &j in &dominates_set[i] {
                dominated_by[j] -= 1;
                if dominated_by[j] == 0 {
                    next.push(j);
                }
            }
        }
        front += 1;
        current = next;
    }
    rank
}

/// Crowding distance per individual: large for isolated points, infinite at the
/// per-objective boundaries. Computed within each rank so it only compares peers.
fn crowding_distances(pop: &[Vec<f64>], ranks: &[usize]) -> Vec<f64> {
    let n = pop.len();
    let mut dist = vec![0.0f64; n];
    if n == 0 {
        return dist;
    }
    let n_obj = pop[0].len();
    let max_rank = *ranks.iter().max().unwrap_or(&0);
    for r in 0..=max_rank {
        let front: Vec<usize> = (0..n).filter(|&i| ranks[i] == r).collect();
        if front.len() <= 2 {
            for &i in &front {
                dist[i] = f64::INFINITY;
            }
            continue;
        }
        for m in 0..n_obj {
            let mut order = front.clone();
            order.sort_by(|&a, &b| {
                pop[a][m]
                    .partial_cmp(&pop[b][m])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let lo = pop[order[0]][m];
            let hi = pop[order[order.len() - 1]][m];
            let span = (hi - lo).max(1e-12);
            dist[order[0]] = f64::INFINITY;
            dist[order[order.len() - 1]] = f64::INFINITY;
            for k in 1..order.len() - 1 {
                dist[order[k]] += (pop[order[k + 1]][m] - pop[order[k - 1]][m]) / span;
            }
        }
    }
    dist
}

impl Sampler for Nsga2Sampler {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ (history.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );

        // Population: completed trials with every objective present.
        let mut pop_params: Vec<&ParamSet> = Vec::new();
        let mut pop_obj: Vec<Vec<f64>> = Vec::new();
        for r in history.completed() {
            if let Some(v) = self.objective_vec(|name| r.final_value(name)) {
                if v.iter().all(|x| x.is_finite()) {
                    pop_params.push(&r.params);
                    pop_obj.push(v);
                }
            }
        }

        if pop_obj.len() < self.n_startup {
            return RandomSampler::sample_space(space, &mut rng);
        }

        // Rank the whole archive, then keep the elite population (best by rank,
        // then crowding) as the mating pool. This is NSGA-II survival selection:
        // it retains the best-ever non-dominated set (elitism) while bounding the
        // pool and preserving spread, which is what applies selection pressure.
        let ranks = non_dominated_ranks(&pop_obj);
        let crowd = crowding_distances(&pop_obj, &ranks);
        let mut elite: Vec<usize> = (0..pop_obj.len()).collect();
        elite.sort_by(|&a, &b| {
            ranks[a].cmp(&ranks[b]).then(
                crowd[b]
                    .partial_cmp(&crowd[a])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        elite.truncate(self.population_size.max(2));

        // Binary tournament within the elite: lower rank wins; ties break toward
        // higher crowding.
        let mut tournament = |rng: &mut ChaCha8Rng| -> usize {
            let a = elite[rng.gen_range(0..elite.len())];
            let b = elite[rng.gen_range(0..elite.len())];
            if ranks[a] < ranks[b] || (ranks[a] == ranks[b] && crowd[a] >= crowd[b]) {
                a
            } else {
                b
            }
        };
        let p1 = pop_params[tournament(&mut rng)];
        let p2 = pop_params[tournament(&mut rng)];

        // Uniform crossover then local (Gaussian) mutation, in declaration order
        // so conditional gates see the values chosen so far. Local mutation lets
        // a continuous dimension *refine* toward the front rather than only
        // interpolating between elites (uniform resampling cannot fine-tune).
        let mut child = ParamSet::new();
        for def in space.params() {
            if !SearchSpace::is_active(def, &child) {
                continue;
            }
            let src = if rng.gen_bool(0.5) { p1 } else { p2 };
            let mut value = match src.get(&def.name) {
                Some(v) => v.clone(),
                None => def.distribution.sample(&mut rng),
            };
            if rng.gen_bool(self.mutation_prob) {
                value = crate::evolution::mutate(
                    &def.distribution,
                    &value,
                    self.mutation_scale,
                    &mut rng,
                );
            }
            child.insert(def.name.clone(), value);
        }
        child
    }

    fn name(&self) -> &'static str {
        "nsga2"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_and_ranks() {
        // Minimization-normalized points.
        let pop = vec![
            vec![1.0, 1.0], // front 0
            vec![2.0, 2.0], // dominated by [1,1] -> front 1
            vec![1.0, 3.0], // non-dominated vs [1,1]? [1,1] <= [1,3], strict on 2nd -> dominated
            vec![0.5, 4.0], // trades off with [1,1] -> front 0
        ];
        assert!(dominates(&pop[0], &pop[1]));
        assert!(!dominates(&pop[0], &pop[3]));
        let ranks = non_dominated_ranks(&pop);
        assert_eq!(ranks[0], 0);
        assert_eq!(ranks[3], 0);
        assert!(ranks[1] >= 1);
        assert!(ranks[2] >= 1);
    }

    #[test]
    fn crowding_marks_boundaries_infinite() {
        let pop = vec![
            vec![0.0, 3.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 0.0],
        ];
        let ranks = vec![0, 0, 0, 0];
        let d = crowding_distances(&pop, &ranks);
        assert!(d[0].is_infinite() && d[3].is_infinite());
        assert!(d[1].is_finite() && d[2].is_finite());
    }

    #[test]
    fn nsga2_beats_random_on_zdt1() {
        use crate::distribution::Distribution;
        use crate::metrics::NamedMetrics;
        use crate::objective::ReportSink;
        use crate::sampler::RandomSampler;
        use crate::space::SearchSpace;
        use crate::study::Study;

        // ZDT1 (2 variables): the Pareto front lies at y = 0, so a sampler must
        // apply selection pressure to drive y down — random cannot, NSGA-II can.
        //   f1 = x ; g = 1 + 9y ; f2 = g*(1 - sqrt(x/g))
        let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
            let x = p.float("x")?;
            let y = p.float("y")?;
            let g = 1.0 + 9.0 * y;
            let f2 = g * (1.0 - (x / g).sqrt());
            Ok(NamedMetrics::new().with("f1", x).with("f2", f2))
        };
        // Reference dominated by every achievable point.
        let reference = NamedMetrics::new().with("f1", 1.1).with("f2", 11.0);

        let run_hv = |nsga: bool| -> f64 {
            let space = SearchSpace::new()
                .add("x", Distribution::float(0.0, 1.0))
                .add("y", Distribution::float(0.0, 1.0));
            let mut b = Study::builder(space).minimize("f1").minimize("f2").seed(3);
            b = if nsga {
                b.sampler(Nsga2Sampler::new(
                    vec![
                        ("f1".into(), Direction::Minimize),
                        ("f2".into(), Direction::Minimize),
                    ],
                    3,
                ))
            } else {
                b.sampler(RandomSampler::new(3))
            };
            let mut study = b.build().unwrap();
            study.optimize_n(&objective, 120).unwrap();
            study.hypervolume(&reference).unwrap()
        };

        let nsga_hv = run_hv(true);
        let rand_hv = run_hv(false);
        assert!(nsga_hv > 0.0, "NSGA-II produced an empty/zero front");
        // Driving y -> 0 gives a strictly better front than random scattering.
        assert!(
            nsga_hv > rand_hv,
            "NSGA-II hypervolume {nsga_hv} should beat random {rand_hv} on ZDT1"
        );
    }
}
