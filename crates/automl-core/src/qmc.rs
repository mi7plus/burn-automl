//! Quasi-Monte-Carlo (Halton) sampler — deterministic low-discrepancy search.
//!
//! Random sampling clumps and leaves gaps; a quasi-random *low-discrepancy*
//! sequence covers the space far more evenly, which typically finds good
//! regions faster on low-dimensional continuous problems. This sampler uses the
//! Halton sequence: the trial index's radical inverse in a distinct prime base
//! per parameter, mapped through each parameter's distribution. It is fully
//! deterministic (a pure function of the trial index — no RNG), so replay is
//! exact (§2), and it handles conditional spaces like the other samplers by
//! only sampling a parameter's dimension when its branch is active (§5.1).

use crate::distribution::Distribution;
use crate::param::{ParamSet, ParamValue};
use crate::sampler::Sampler;
use crate::space::SearchSpace;
use crate::trial::TrialHistory;

/// A quasi-Monte-Carlo sampler over the Halton sequence.
pub struct QmcSampler {
    /// How many leading sequence points to skip (Halton's first few points in
    /// low bases are mildly correlated; skipping improves uniformity).
    skip: u64,
}

impl QmcSampler {
    /// A QMC sampler skipping the first few (default 5) sequence points.
    pub fn new() -> Self {
        QmcSampler { skip: 5 }
    }

    /// Set how many leading sequence points to skip.
    pub fn with_skip(mut self, skip: u64) -> Self {
        self.skip = skip;
        self
    }
}

impl Default for QmcSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for QmcSampler {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        // 1-based sequence index for this trial; skip leading points.
        let index = history.len() as u64 + self.skip + 1;
        let mut params = ParamSet::new();
        for (dim, def) in space.params().iter().enumerate() {
            if !SearchSpace::is_active(def, &params) {
                continue;
            }
            let u = radical_inverse(index, prime(dim));
            params.insert(def.name.clone(), unit_to_value(&def.distribution, u));
        }
        params
    }

    fn name(&self) -> &'static str {
        "qmc"
    }
}

/// The van der Corput / Halton radical inverse of `i` in the given `base`,
/// returning a point in `[0, 1)`.
fn radical_inverse(mut i: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    while i > 0 {
        result += (i % base) as f64 * f;
        i /= base;
        f /= base as f64;
    }
    result
}

/// The `n`-th prime (0-indexed), used as the Halton base for dimension `n`.
/// Beyond the table, cycles through it — acceptable for the modest dimensions
/// v0.1 search spaces reach.
fn prime(n: usize) -> u64 {
    const PRIMES: [u64; 64] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311,
    ];
    PRIMES[n % PRIMES.len()]
}

/// Map a unit-interval value `u` in `[0, 1)` to a concrete value in a
/// distribution (an inverse-CDF-style transform).
fn unit_to_value(dist: &Distribution, u: f64) -> ParamValue {
    let u = u.clamp(0.0, 1.0 - f64::EPSILON);
    match dist {
        Distribution::Float {
            low,
            high,
            log,
            step,
        } => {
            let mut v = if *log {
                (low.ln() + u * (high.ln() - low.ln())).exp()
            } else {
                low + u * (high - low)
            };
            if let Some(s) = step {
                let k = ((v - *low) / *s).round();
                v = low + k * s;
            }
            ParamValue::Float(v.clamp(*low, *high))
        }
        Distribution::Int {
            low,
            high,
            log,
            step,
        } => {
            let v = if *log {
                ((*low as f64).ln() + u * ((*high as f64).ln() - (*low as f64).ln())).exp()
            } else {
                *low as f64 + u * (*high as f64 - *low as f64 + 1.0)
            };
            // Snap to the step grid and clamp.
            let base = v.floor();
            let k = (((base - *low as f64) / *step as f64).round() as i64).max(0);
            ParamValue::Int((low + k * step).clamp(*low, *high))
        }
        Distribution::Categorical { choices } => {
            let idx = ((u * choices.len() as f64).floor() as usize).min(choices.len() - 1);
            ParamValue::Categorical(choices[idx].clone())
        }
        Distribution::Bool => ParamValue::Bool(u >= 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Condition;

    #[test]
    fn radical_inverse_base2_matches_known_sequence() {
        // van der Corput base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8.
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];
        for (i, e) in expected.iter().enumerate() {
            let got = radical_inverse(i as u64 + 1, 2);
            assert!((got - e).abs() < 1e-12, "i={i}: got {got}, want {e}");
        }
    }

    #[test]
    fn deterministic_and_within_bounds() {
        let space = SearchSpace::new()
            .add("x", Distribution::float(-5.0, 5.0))
            .add("k", Distribution::int(1, 10))
            .add("c", Distribution::categorical(["a", "b", "c"]));
        let hist = TrialHistory::default();
        let mut s1 = QmcSampler::new();
        let mut s2 = QmcSampler::new();
        let p1 = s1.suggest(&space, &hist);
        assert_eq!(p1, s2.suggest(&space, &hist));
        assert!((-5.0..=5.0).contains(&p1.float("x").unwrap()));
        assert!((1..=10).contains(&p1.int("k").unwrap()));
        assert!(["a", "b", "c"].contains(&p1.categorical("c").unwrap()));
    }

    #[test]
    fn low_discrepancy_covers_the_range() {
        // Over N points on [0,1], the largest gap between sorted samples should
        // be far smaller than random sampling would typically give.
        let space = SearchSpace::new().add("x", Distribution::float(0.0, 1.0));
        let mut xs = Vec::new();
        for i in 0..64u64 {
            let mut s = QmcSampler::new();
            // History length varies the sequence index, so successive trials
            // draw successive Halton points.
            let hist = TrialHistory::new((0..i).map(dummy).collect());
            xs.push(s.suggest(&space, &hist).float("x").unwrap());
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut max_gap = xs[0]; // gap from 0
        for w in xs.windows(2) {
            max_gap = max_gap.max(w[1] - w[0]);
        }
        max_gap = max_gap.max(1.0 - xs[xs.len() - 1]);
        // 64 Halton base-2 points cover [0,1] with gaps ~1/64; allow slack.
        assert!(
            max_gap < 0.06,
            "max gap {max_gap} too large — poor coverage"
        );
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
        for i in 0..40u64 {
            let mut s = QmcSampler::new();
            let hist = TrialHistory::new((0..i).map(dummy).collect());
            let p = s.suggest(&space, &hist);
            match p.categorical("model").unwrap() {
                "transformer" => assert!(p.contains("d_model") && !p.contains("hidden")),
                "mlp" => assert!(p.contains("hidden") && !p.contains("d_model")),
                _ => unreachable!(),
            }
        }
    }

    fn dummy(id: u64) -> crate::trial::TrialRecord {
        crate::trial::TrialRecord::new(
            crate::trial::TrialId(id),
            crate::trial::StudyId(0),
            ParamSet::new(),
            id,
        )
    }
}
