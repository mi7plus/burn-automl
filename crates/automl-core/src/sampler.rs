//! Samplers: the `Sampler` trait plus the Random and Grid baselines.
//!
//! Per PRD §29 (recommended sequence, items 1-2) and §5.1, Random and Grid must
//! handle *every* conditional space correctly from v0.1 so that later, harder
//! samplers (TPE) never block a release — TPE will degrade to random within an
//! under-observed branch. These two are the correctness baseline the benchmark
//! suite (§24.1) measures against.
//!
//! Sampling is define-and-run: [`Sampler::suggest`] receives the whole
//! [`SearchSpace`] and returns a [`ParamSet`]. Conditional parameters are only
//! sampled when their gate is satisfied by values chosen earlier in the same
//! call, honoring the declaration-order invariant enforced by
//! [`SearchSpace::validate`].

use crate::param::ParamSet;
use crate::space::SearchSpace;
use crate::trial::TrialHistory;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Suggests the next [`ParamSet`] to evaluate from a study's history (PRD §4.1).
pub trait Sampler: Send + Sync {
    /// Propose parameters for the next trial.
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet;

    /// Observe a completed trial. Stateless samplers may ignore this.
    fn on_trial_complete(&mut self, _trial: &crate::trial::TrialRecord) {}

    /// A short, stable name persisted in study metadata (PRD §19.2).
    fn name(&self) -> &'static str;
}

/// Independent, uniform random sampling over the (conditional) search space.
///
/// Deterministic given its seed: the RNG is derived per-call from the base seed
/// and the number of trials already seen, so a replayed study with the same
/// seed and history produces identical suggestions (PRD §2 deterministic replay).
pub struct RandomSampler {
    seed: u64,
}

impl RandomSampler {
    /// A random sampler seeded for reproducibility.
    pub fn new(seed: u64) -> Self {
        RandomSampler { seed }
    }

    /// Draw one active-parameter assignment using the given RNG.
    ///
    /// Exposed to the crate so the TPE sampler can reuse the exact same
    /// prior-sampling logic for its startup phase and per-branch fallback (§5.1).
    pub(crate) fn sample_space<R: Rng + ?Sized>(space: &SearchSpace, rng: &mut R) -> ParamSet {
        let mut params = ParamSet::new();
        for def in space.params() {
            if SearchSpace::is_active(def, &params) {
                params.insert(def.name.clone(), def.distribution.sample(rng));
            }
        }
        params
    }
}

impl Sampler for RandomSampler {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        // Derive a per-trial RNG from base seed + trial index so replay is exact.
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ (history.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );
        Self::sample_space(space, &mut rng)
    }

    fn name(&self) -> &'static str {
        "random"
    }
}

/// Exhaustive grid sampling over the Cartesian product of per-parameter grids.
///
/// Conditional parameters are handled by expanding child grids only under the
/// parent values that activate them, so the enumeration never emits an inactive
/// parameter. Continuous floats are discretized into `float_samples` points.
/// Once the grid is exhausted it wraps around (re-emitting from the start),
/// which keeps the study loop simple; callers bound the run with a [`crate::budget::Budget`].
pub struct GridSampler {
    grid: Vec<ParamSet>,
    cursor: usize,
    float_samples: usize,
    built: bool,
}

impl GridSampler {
    /// A grid sampler discretizing continuous floats into `float_samples` points each.
    pub fn new(float_samples: usize) -> Self {
        GridSampler {
            grid: Vec::new(),
            cursor: 0,
            float_samples: float_samples.max(1),
            built: false,
        }
    }

    /// Total number of grid points (0 until the first `suggest` builds it).
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Whether the built grid is empty.
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }

    /// Recursively enumerate the conditional Cartesian product.
    fn build(space: &SearchSpace, float_samples: usize) -> Vec<ParamSet> {
        let mut acc = vec![ParamSet::new()];
        for def in space.params() {
            let mut next: Vec<ParamSet> = Vec::new();
            for partial in &acc {
                if !SearchSpace::is_active(def, partial) {
                    // Inactive under this partial assignment: carry it unchanged.
                    next.push(partial.clone());
                    continue;
                }
                for value in def.distribution.grid_values(float_samples) {
                    let mut extended = partial.clone();
                    extended.insert(def.name.clone(), value);
                    next.push(extended);
                }
            }
            acc = next;
        }
        acc
    }
}

impl Sampler for GridSampler {
    fn suggest(&mut self, space: &SearchSpace, _history: &TrialHistory) -> ParamSet {
        if !self.built {
            self.grid = Self::build(space, self.float_samples);
            self.built = true;
        }
        if self.grid.is_empty() {
            return ParamSet::new();
        }
        let point = self.grid[self.cursor % self.grid.len()].clone();
        self.cursor += 1;
        point
    }

    fn name(&self) -> &'static str {
        "grid"
    }
}

/// A helper for objectives that need a deterministic per-trial RNG derived from
/// the trial's own seed (e.g. weight init). Kept here so samplers and objectives
/// derive RNGs the same way.
pub fn trial_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::space::Condition;

    fn flat_space() -> SearchSpace {
        SearchSpace::new()
            .add("lr", Distribution::log_float(1e-4, 1e-1))
            .add("depth", Distribution::int(1, 4))
            .add("model", Distribution::categorical(["a", "b"]))
    }

    fn conditional_space() -> SearchSpace {
        SearchSpace::new()
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
            )
    }

    #[test]
    fn random_is_deterministic_for_same_history_len() {
        let space = flat_space();
        let hist = TrialHistory::default();
        let mut s1 = RandomSampler::new(7);
        let mut s2 = RandomSampler::new(7);
        assert_eq!(s1.suggest(&space, &hist), s2.suggest(&space, &hist));
    }

    #[test]
    fn random_respects_conditions() {
        let space = conditional_space();
        let mut s = RandomSampler::new(1);
        for _ in 0..50 {
            let h = TrialHistory::new(vec![]);
            let p = s.suggest(&space, &h);
            match p.categorical("model").unwrap() {
                "transformer" => {
                    assert!(p.contains("d_model"));
                    assert!(!p.contains("hidden"));
                }
                "mlp" => {
                    assert!(p.contains("hidden"));
                    assert!(!p.contains("d_model"));
                }
                other => panic!("unexpected model {other}"),
            }
        }
    }

    #[test]
    fn grid_enumerates_conditional_product() {
        let space = conditional_space();
        let hist = TrialHistory::default();
        let mut g = GridSampler::new(1);
        let first = g.suggest(&space, &hist);
        // model has 2 choices; transformer expands d_model(64..=128) and mlp
        // expands hidden(16..=32). Grid size = 65 + 17 = 82.
        assert_eq!(g.len(), 65 + 17);
        // Every emitted point respects the active-parameter rule.
        assert!(first.contains("model"));
    }

    #[test]
    fn grid_never_emits_inactive_param() {
        let space = conditional_space();
        let hist = TrialHistory::default();
        let mut g = GridSampler::new(1);
        for _ in 0..(65 + 17) {
            let p = g.suggest(&space, &hist);
            match p.categorical("model").unwrap() {
                "transformer" => assert!(p.contains("d_model") && !p.contains("hidden")),
                "mlp" => assert!(p.contains("hidden") && !p.contains("d_model")),
                _ => unreachable!(),
            }
        }
    }
}
