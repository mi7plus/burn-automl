//! Tree-structured Parzen Estimator (TPE) sampler — the first strong real-world
//! optimizer (PRD §29 items 4-5).
//!
//! TPE models the density of parameter values among "good" trials (`l(x)`) and
//! "bad" trials (`g(x)`), splitting the observed trials at a quantile of the
//! objective, then proposes the candidate that maximizes `l(x)/g(x)`.
//!
//! ## Conditional spaces and the §5.1 risk
//!
//! The plan names conditional-space TPE as a first-class research risk: a
//! branch's density is only well-defined conditional on that branch being
//! chosen, and pooling observations across branches biases the estimator. Two
//! mitigations are built in here from the start:
//!
//! 1. **Independent per-branch estimation** — when estimating a parameter, only
//!    observations from trials where that parameter was *active* are used. Since
//!    parameters are sampled in declaration order with parents first, filtering
//!    each parameter's observations to trials that contain it yields exactly the
//!    per-branch conditioning the plan calls for.
//! 2. **Random fallback for thin branches** — if an active branch has fewer than
//!    `min_branch_obs` good observations (default 10), that parameter is drawn
//!    from its prior instead of a poorly-fit KDE. TPE therefore degrades
//!    gracefully to random sampling within an under-observed branch rather than
//!    erroring, exactly as the plan requires.
//!
//! Truncation renormalization of the Gaussian mixtures is intentionally omitted
//! for v0.1: both `l` and `g` share the same bounds, so the omission cancels to
//! first order in the ratio that drives selection.

use crate::distribution::Distribution;
use crate::metrics::Direction;
use crate::param::{ParamSet, ParamValue};
use crate::sampler::{RandomSampler, Sampler};
use crate::space::SearchSpace;
use crate::trial::TrialHistory;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// A TPE sampler for a single named objective.
///
/// TPE is inherently single-objective; multi-objective studies use the Pareto
/// machinery (§17), which lands in a later release. Construct with the metric
/// name and direction the study optimizes.
pub struct TpeSampler {
    seed: u64,
    objective: String,
    direction: Direction,
    n_startup_trials: usize,
    n_candidates: usize,
    gamma: f64,
    min_branch_obs: usize,
}

impl TpeSampler {
    /// A TPE sampler for `objective` optimized in `direction`.
    pub fn new(objective: impl Into<String>, direction: Direction, seed: u64) -> Self {
        TpeSampler {
            seed,
            objective: objective.into(),
            direction,
            n_startup_trials: 10,
            n_candidates: 24,
            gamma: 0.25,
            min_branch_obs: 10,
        }
    }

    /// Number of random trials before TPE modeling begins (default 10).
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Number of candidates drawn from `l(x)` per parameter (default 24).
    pub fn with_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n.max(1);
        self
    }

    /// Quantile splitting good from bad trials (default 0.25).
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma.clamp(0.01, 0.99);
        self
    }

    /// Minimum good observations before a branch is modeled rather than sampled
    /// from its prior (default 10). This is the §5.1 fallback threshold.
    pub fn with_min_branch_obs(mut self, n: usize) -> Self {
        self.min_branch_obs = n;
        self
    }

    /// Convert an objective value to a loss (lower is better) per the direction.
    fn loss(&self, value: f64) -> f64 {
        match self.direction {
            Direction::Minimize => value,
            Direction::Maximize => -value,
        }
    }

    /// Number of trials placed in the "good" group.
    fn n_below(&self, n: usize) -> usize {
        (((self.gamma * n as f64).ceil()) as usize)
            .clamp(1, n.saturating_sub(1))
            .min(25)
    }

    /// Suggest a single parameter value, given the good/bad observations for it.
    fn suggest_param<R: Rng + ?Sized>(
        &self,
        dist: &Distribution,
        good: &[&ParamValue],
        bad: &[&ParamValue],
        rng: &mut R,
    ) -> ParamValue {
        match dist {
            Distribution::Float { .. } | Distribution::Int { .. } => {
                let (lo, hi) = axis_bounds(dist);
                let g_obs: Vec<f64> = good.iter().map(|v| to_axis(dist, v)).collect();
                let b_obs: Vec<f64> = bad.iter().map(|v| to_axis(dist, v)).collect();
                let l = Parzen1D::new(&g_obs, lo, hi);
                let g = Parzen1D::new(&b_obs, lo, hi);

                let mut best = l.sample(rng);
                let mut best_score = l.log_pdf(best) - g.log_pdf(best);
                for _ in 1..self.n_candidates {
                    let cand = l.sample(rng);
                    let score = l.log_pdf(cand) - g.log_pdf(cand);
                    if score > best_score {
                        best_score = score;
                        best = cand;
                    }
                }
                from_axis(dist, best)
            }
            Distribution::Categorical { choices } => {
                let l = cat_probs(good, choices, |c, v| c.iter().position(|x| x == v));
                let g = cat_probs(bad, choices, |c, v| c.iter().position(|x| x == v));
                let idx = argmax_ratio(&l, &g);
                ParamValue::Categorical(choices[idx].clone())
            }
            Distribution::Bool => {
                let choices = [false, true];
                let l = bool_probs(good);
                let g = bool_probs(bad);
                let idx = argmax_ratio(&l, &g);
                ParamValue::Bool(choices[idx])
            }
        }
    }
}

impl Sampler for TpeSampler {
    fn suggest(&mut self, space: &SearchSpace, history: &TrialHistory) -> ParamSet {
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ (history.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );

        // Completed trials that actually reported the objective, as (loss, params).
        let mut scored: Vec<(f64, &ParamSet)> = history
            .completed()
            .filter_map(|r| {
                r.final_value(&self.objective)
                    .map(|v| (self.loss(v), &r.params))
            })
            .filter(|(l, _)| l.is_finite())
            .collect();

        // Startup phase: pure random until enough evidence exists to model.
        if scored.len() < self.n_startup_trials {
            return RandomSampler::sample_space(space, &mut rng);
        }

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n_below = self.n_below(scored.len());
        let good: Vec<&ParamSet> = scored[..n_below].iter().map(|(_, p)| *p).collect();
        let bad: Vec<&ParamSet> = scored[n_below..].iter().map(|(_, p)| *p).collect();

        let mut params = ParamSet::new();
        for def in space.params() {
            if !SearchSpace::is_active(def, &params) {
                continue;
            }
            // Per-branch conditioning: only trials where this parameter was
            // active (i.e. present) contribute observations.
            let good_vals: Vec<&ParamValue> =
                good.iter().filter_map(|p| p.get(&def.name)).collect();
            let bad_vals: Vec<&ParamValue> = bad.iter().filter_map(|p| p.get(&def.name)).collect();

            let value = if good_vals.len() < self.min_branch_obs {
                // §5.1 fallback: thin branch → draw from the prior.
                def.distribution.sample(&mut rng)
            } else {
                self.suggest_param(&def.distribution, &good_vals, &bad_vals, &mut rng)
            };
            params.insert(def.name.clone(), value);
        }
        params
    }

    fn name(&self) -> &'static str {
        "tpe"
    }
}

// ----- 1-D Parzen (Gaussian-mixture) density over a bounded continuous axis ---

/// A weighted Gaussian mixture over `[lo, hi]` with a broad prior component,
/// used to model `l(x)` and `g(x)` for continuous parameters.
struct Parzen1D {
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    log_weights: Vec<f64>,
    lo: f64,
    hi: f64,
}

impl Parzen1D {
    fn new(obs: &[f64], lo: f64, hi: f64) -> Self {
        let prior_mu = (lo + hi) / 2.0;
        let prior_sigma = (hi - lo).max(1e-12);

        // Mixture components: one per observation plus a broad prior at the end.
        let mut mus: Vec<f64> = obs.to_vec();
        mus.push(prior_mu);
        let n = mus.len();

        // Assign each component a bandwidth from its neighbor spacing (sorted),
        // clipped to [prior_sigma/100, prior_sigma]; the prior keeps prior_sigma.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| mus[a].partial_cmp(&mus[b]).unwrap());
        let sigma_min = prior_sigma / 100.0;
        let mut sigmas = vec![prior_sigma; n];
        for rank in 0..n {
            let i = order[rank];
            if i == n - 1 {
                continue; // prior component
            }
            let cur = mus[i];
            let left = if rank > 0 {
                cur - mus[order[rank - 1]]
            } else {
                f64::INFINITY
            };
            let right = if rank < n - 1 {
                mus[order[rank + 1]] - cur
            } else {
                f64::INFINITY
            };
            let spacing = left.min(right);
            sigmas[i] = if spacing.is_finite() {
                spacing.clamp(sigma_min, prior_sigma)
            } else {
                prior_sigma
            };
        }

        // Equal weights across all components (observations + prior).
        let w = 1.0 / n as f64;
        let log_weights = vec![w.ln(); n];

        Parzen1D {
            mus,
            sigmas,
            log_weights,
            lo,
            hi,
        }
    }

    /// Sample from the mixture, truncated to `[lo, hi]`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let k = rng.gen_range(0..self.mus.len());
        for _ in 0..50 {
            let x = sample_normal(rng, self.mus[k], self.sigmas[k]);
            if x >= self.lo && x <= self.hi {
                return x;
            }
        }
        // Give up truncating after a few tries; clamp into range.
        sample_normal(rng, self.mus[k], self.sigmas[k]).clamp(self.lo, self.hi)
    }

    /// Log density at `x` (ignoring truncation renormalization; see module docs).
    fn log_pdf(&self, x: f64) -> f64 {
        let terms: Vec<f64> = (0..self.mus.len())
            .map(|i| {
                let z = (x - self.mus[i]) / self.sigmas[i];
                self.log_weights[i] - 0.5 * z * z - self.sigmas[i].ln() - 0.5 * (2.0 * PI).ln()
            })
            .collect();
        logsumexp(&terms)
    }
}

// ----- axis transforms between ParamValue and the continuous modeling axis -----

fn axis_bounds(dist: &Distribution) -> (f64, f64) {
    match dist {
        Distribution::Float { low, high, log, .. } => {
            if *log {
                (low.ln(), high.ln())
            } else {
                (*low, *high)
            }
        }
        Distribution::Int { low, high, log, .. } => {
            if *log {
                ((*low as f64).ln(), (*high as f64).ln())
            } else {
                (*low as f64, *high as f64)
            }
        }
        _ => (0.0, 1.0),
    }
}

fn to_axis(dist: &Distribution, v: &ParamValue) -> f64 {
    match (dist, v) {
        (Distribution::Float { log, .. }, ParamValue::Float(x)) => {
            if *log {
                x.max(1e-300).ln()
            } else {
                *x
            }
        }
        (Distribution::Int { log, .. }, ParamValue::Int(x)) => {
            if *log {
                (*x as f64).max(1e-300).ln()
            } else {
                *x as f64
            }
        }
        _ => 0.0,
    }
}

fn from_axis(dist: &Distribution, t: f64) -> ParamValue {
    match dist {
        Distribution::Float {
            low,
            high,
            log,
            step,
        } => {
            let mut x = if *log { t.exp() } else { t };
            if let Some(s) = step {
                let k = ((x - *low) / *s).round();
                x = low + k * s;
            }
            ParamValue::Float(x.clamp(*low, *high))
        }
        Distribution::Int {
            low,
            high,
            log,
            step,
        } => {
            let base = if *log { t.exp() } else { t };
            // Snap to the step grid low + k*step, then clamp.
            let k = ((base - *low as f64) / *step as f64).round() as i64;
            let v = (low + k * step).clamp(*low, *high);
            ParamValue::Int(v)
        }
        _ => unreachable!("from_axis only handles continuous distributions"),
    }
}

// ----- categorical / boolean estimators -----

fn cat_probs<F>(obs: &[&ParamValue], choices: &[String], index_of: F) -> Vec<f64>
where
    F: Fn(&[String], &str) -> Option<usize>,
{
    let mut counts = vec![1.0f64; choices.len()]; // Laplace prior of 1 per choice
    for v in obs {
        if let ParamValue::Categorical(s) = v {
            if let Some(i) = index_of(choices, s) {
                counts[i] += 1.0;
            }
        }
    }
    normalize(counts)
}

fn bool_probs(obs: &[&ParamValue]) -> Vec<f64> {
    let mut counts = vec![1.0f64; 2];
    for v in obs {
        if let ParamValue::Bool(b) = v {
            counts[*b as usize] += 1.0;
        }
    }
    normalize(counts)
}

fn normalize(counts: Vec<f64>) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    counts.into_iter().map(|c| c / total).collect()
}

fn argmax_ratio(l: &[f64], g: &[f64]) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..l.len() {
        let score = l[i].ln() - g[i].ln();
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

// ----- numeric helpers -----

fn sample_normal<R: Rng + ?Sized>(rng: &mut R, mu: f64, sigma: f64) -> f64 {
    // Box-Muller transform.
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mu + sigma * z
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NamedMetrics;
    use crate::param::ParamSet;
    use crate::space::Condition;
    use crate::trial::{StudyId, TrialId, TrialRecord, TrialState};

    fn completed(id: u64, params: ParamSet, loss: f64) -> TrialRecord {
        let mut r = TrialRecord::new(TrialId(id), StudyId(0), params, id);
        r.state = TrialState::Complete;
        r.final_metrics = Some(NamedMetrics::single("loss", loss));
        r
    }

    #[test]
    fn falls_back_to_random_before_startup() {
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let mut s = TpeSampler::new("loss", Direction::Minimize, 1);
        let p = s.suggest(&space, &TrialHistory::default());
        assert!(p.contains("x"));
        let x = p.float("x").unwrap();
        assert!((-5.0..=5.0).contains(&x));
    }

    #[test]
    fn deterministic_for_same_history() {
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let mut history = Vec::new();
        for i in 0..15u64 {
            let x = -5.0 + (i as f64) * 0.5;
            let p = ParamSet::new().with("x", ParamValue::Float(x));
            history.push(completed(i, p, (x - 2.0).powi(2)));
        }
        let hist = TrialHistory::new(history);
        let mut s1 = TpeSampler::new("loss", Direction::Minimize, 99);
        let mut s2 = TpeSampler::new("loss", Direction::Minimize, 99);
        assert_eq!(s1.suggest(&space, &hist), s2.suggest(&space, &hist));
    }

    #[test]
    fn concentrates_near_optimum() {
        // Feed TPE a set of observations of (x-2)^2 and check that the values it
        // proposes concentrate near x=2 (better than uniform expectation).
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let mut history = Vec::new();
        let mut seed_rng = ChaCha8Rng::seed_from_u64(7);
        for i in 0..40u64 {
            let x: f64 = seed_rng.gen_range(-5.0..=5.0);
            let p = ParamSet::new().with("x", ParamValue::Float(x));
            history.push(completed(i, p, (x - 2.0).powi(2)));
        }
        let hist = TrialHistory::new(history);
        // suggest() derives its RNG from (seed, history.len()); history is fixed
        // here, so vary the seed per draw to sample the proposal distribution.
        let mut close = 0;
        for seed in 0..40u64 {
            let mut s = TpeSampler::new("loss", Direction::Minimize, seed);
            let p = s.suggest(&space, &hist);
            if (p.float("x").unwrap() - 2.0).abs() < 2.0 {
                close += 1;
            }
        }
        // A uniform sampler would land within |x-2|<2 about 40% of the time
        // (range 4 of 10). TPE should do clearly better.
        assert!(
            close > 24,
            "only {close}/40 proposals were near the optimum"
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
        // Enough startup evidence so TPE modeling engages.
        let mut history = Vec::new();
        for i in 0..20u64 {
            let p = ParamSet::new()
                .with("model", ParamValue::Categorical("mlp".into()))
                .with("hidden", ParamValue::Int(24));
            history.push(completed(i, p, 1.0));
        }
        let hist = TrialHistory::new(history);
        let mut s = TpeSampler::new("loss", Direction::Minimize, 5);
        for _ in 0..20 {
            let p = s.suggest(&space, &hist);
            match p.categorical("model").unwrap() {
                "transformer" => assert!(p.contains("d_model") && !p.contains("hidden")),
                "mlp" => assert!(p.contains("hidden") && !p.contains("d_model")),
                _ => unreachable!(),
            }
        }
    }

    /// Conditional-space TPE hardening (§5.1, §24.1): a mixed conditional
    /// optimization where the *branch itself* and its within-branch parameter
    /// must both be found. Branch "a" bottoms out at loss 5, branch "b" reaches
    /// loss 0 at y = -3. TPE must learn to prefer branch "b" (steered by the
    /// categorical density ratio) and locate its optimum (per-branch KDE), while
    /// under-observed branches fall back to the prior rather than erroring.
    #[test]
    fn tpe_optimizes_a_conditional_space_end_to_end() {
        use crate::study::Study;

        let space = SearchSpace::new()
            .add("model", Distribution::categorical(["a", "b"]))
            .add_conditional(
                "x",
                Distribution::float(-5.0, 5.0),
                Condition::when_eq("model", "a"),
            )
            .add_conditional(
                "y",
                Distribution::float(-5.0, 5.0),
                Condition::when_eq("model", "b"),
            );

        let objective = |p: &ParamSet, _s: &mut dyn crate::objective::ReportSink| {
            let loss = match p.categorical("model")? {
                "a" => 5.0 + (p.float("x")? - 2.0).powi(2), // best 5.0
                "b" => (p.float("y")? + 3.0).powi(2),       // best 0.0 at y=-3
                _ => unreachable!(),
            };
            Ok(NamedMetrics::single("loss", loss))
        };

        let mut study = Study::builder(space)
            .minimize("loss")
            .sampler(TpeSampler::new("loss", Direction::Minimize, 11))
            .seed(11)
            .build()
            .unwrap();
        study.optimize_n(&objective, 120).unwrap();

        let best = study.best_trial().unwrap().unwrap();
        // TPE should discover branch "b" is superior and hone in on y ≈ -3.
        assert_eq!(
            best.params.categorical("model").unwrap(),
            "b",
            "should prefer branch b"
        );
        assert!(
            best.final_value("loss").unwrap() < 0.5,
            "best loss {:?} — did not find the conditional optimum",
            best.final_value("loss")
        );
    }
}
