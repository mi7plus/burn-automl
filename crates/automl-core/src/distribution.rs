//! Parameter distributions — the value spaces a single hyperparameter can take.
//!
//! Distributions are the leaves of a [`crate::space::SearchSpace`]. They cover
//! the flat forms called out in PRD §5: uniform/log floats and integers,
//! stepped values, categorical and boolean. Conditional/hierarchical structure
//! lives one level up, in [`crate::space`].

use crate::error::{Error, Result};
use crate::param::ParamValue;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A value space for a single parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    /// A floating-point range `[low, high]`, optionally log-scaled or stepped.
    Float {
        /// Inclusive lower bound.
        low: f64,
        /// Inclusive upper bound.
        high: f64,
        /// Sample in log space (both bounds must be strictly positive).
        log: bool,
        /// Optional discretization step (in linear space).
        step: Option<f64>,
    },
    /// An integer range `[low, high]` inclusive, optionally log-scaled/stepped.
    Int {
        /// Inclusive lower bound.
        low: i64,
        /// Inclusive upper bound.
        high: i64,
        /// Sample in log space (both bounds must be strictly positive).
        log: bool,
        /// Step between successive candidate values (>= 1).
        step: i64,
    },
    /// A finite set of string choices.
    Categorical {
        /// The candidate values; must be non-empty.
        choices: Vec<String>,
    },
    /// A boolean.
    Bool,
}

impl Distribution {
    /// Uniform float in `[low, high]`.
    pub fn float(low: f64, high: f64) -> Self {
        Distribution::Float {
            low,
            high,
            log: false,
            step: None,
        }
    }

    /// Log-uniform float in `[low, high]` (both bounds must be positive).
    pub fn log_float(low: f64, high: f64) -> Self {
        Distribution::Float {
            low,
            high,
            log: true,
            step: None,
        }
    }

    /// Stepped float: candidate values are `low, low+step, low+2*step, ...`.
    pub fn stepped_float(low: f64, high: f64, step: f64) -> Self {
        Distribution::Float {
            low,
            high,
            log: false,
            step: Some(step),
        }
    }

    /// Uniform integer in `[low, high]` inclusive.
    pub fn int(low: i64, high: i64) -> Self {
        Distribution::Int {
            low,
            high,
            log: false,
            step: 1,
        }
    }

    /// Log-uniform integer in `[low, high]` inclusive.
    pub fn log_int(low: i64, high: i64) -> Self {
        Distribution::Int {
            low,
            high,
            log: true,
            step: 1,
        }
    }

    /// Categorical over the given string choices.
    pub fn categorical<I, S>(choices: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Distribution::Categorical {
            choices: choices.into_iter().map(Into::into).collect(),
        }
    }

    /// Boolean distribution.
    pub fn boolean() -> Self {
        Distribution::Bool
    }

    /// A short, stable label used in error messages and mismatch checks.
    pub fn kind(&self) -> &'static str {
        match self {
            Distribution::Float { .. } => "float",
            Distribution::Int { .. } => "int",
            Distribution::Categorical { .. } => "categorical",
            Distribution::Bool => "bool",
        }
    }

    /// Validate the distribution's bounds. Called when registering a parameter
    /// so that impossible spaces fail loudly at definition time (PRD §26).
    pub fn validate(&self, name: &str) -> Result<()> {
        let invalid = |reason: String| {
            Err(Error::InvalidDistribution {
                name: name.to_string(),
                reason,
            })
        };
        match self {
            Distribution::Float {
                low,
                high,
                log,
                step,
            } => {
                if !(low.is_finite() && high.is_finite()) {
                    return invalid("bounds must be finite".into());
                }
                if low > high {
                    return invalid(format!("low ({low}) > high ({high})"));
                }
                if *log && *low <= 0.0 {
                    return invalid("log-scale requires low > 0".into());
                }
                if let Some(s) = step {
                    if *s <= 0.0 {
                        return invalid("step must be positive".into());
                    }
                    if *log {
                        return invalid("stepped and log are mutually exclusive".into());
                    }
                }
                Ok(())
            }
            Distribution::Int {
                low,
                high,
                log,
                step,
            } => {
                if low > high {
                    return invalid(format!("low ({low}) > high ({high})"));
                }
                if *step < 1 {
                    return invalid("step must be >= 1".into());
                }
                if *log && *low <= 0 {
                    return invalid("log-scale requires low > 0".into());
                }
                Ok(())
            }
            Distribution::Categorical { choices } => {
                if choices.is_empty() {
                    return Err(Error::EmptyCategorical(name.to_string()));
                }
                Ok(())
            }
            Distribution::Bool => Ok(()),
        }
    }

    /// Draw a single value from this distribution using the given RNG.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ParamValue {
        match self {
            Distribution::Float {
                low,
                high,
                log,
                step,
            } => {
                let raw = if *log {
                    let (ll, lh) = (low.ln(), high.ln());
                    (rng.gen_range(ll..=lh)).exp()
                } else {
                    if (high - low).abs() < f64::EPSILON {
                        *low
                    } else {
                        rng.gen_range(*low..=*high)
                    }
                };
                let v = match step {
                    Some(s) => snap_to_step(raw, *low, *high, *s),
                    None => raw,
                };
                ParamValue::Float(v)
            }
            Distribution::Int {
                low,
                high,
                log,
                step,
            } => {
                let v = if *log {
                    let (ll, lh) = ((*low as f64).ln(), (*high as f64).ln());
                    let raw = (rng.gen_range(ll..=lh)).exp().round() as i64;
                    raw.clamp(*low, *high)
                } else {
                    // Number of steps available in [low, high].
                    let span = (high - low) / step;
                    let k = rng.gen_range(0..=span);
                    (low + k * step).min(*high)
                };
                ParamValue::Int(v)
            }
            Distribution::Categorical { choices } => {
                let idx = rng.gen_range(0..choices.len());
                ParamValue::Categorical(choices[idx].clone())
            }
            Distribution::Bool => ParamValue::Bool(rng.gen_bool(0.5)),
        }
    }

    /// Enumerate the discrete grid of candidate values for this distribution.
    ///
    /// Continuous floats have no natural finite grid; `float_samples` controls
    /// how many evenly spaced points to place across a non-stepped float range.
    /// Used by the grid sampler.
    pub fn grid_values(&self, float_samples: usize) -> Vec<ParamValue> {
        match self {
            Distribution::Float {
                low,
                high,
                log,
                step,
            } => match step {
                Some(s) => {
                    let mut v = *low;
                    let mut out = Vec::new();
                    while v <= *high + f64::EPSILON {
                        out.push(ParamValue::Float(v.min(*high)));
                        v += s;
                    }
                    out
                }
                None => {
                    let n = float_samples.max(1);
                    if n == 1 {
                        return vec![ParamValue::Float(*low)];
                    }
                    (0..n)
                        .map(|i| {
                            let t = i as f64 / (n - 1) as f64;
                            let v = if *log {
                                (low.ln() + t * (high.ln() - low.ln())).exp()
                            } else {
                                low + t * (high - low)
                            };
                            ParamValue::Float(v)
                        })
                        .collect()
                }
            },
            Distribution::Int {
                low, high, step, ..
            } => {
                let mut out = Vec::new();
                let mut v = *low;
                while v <= *high {
                    out.push(ParamValue::Int(v));
                    v += step;
                }
                out
            }
            Distribution::Categorical { choices } => choices
                .iter()
                .cloned()
                .map(ParamValue::Categorical)
                .collect(),
            Distribution::Bool => {
                vec![ParamValue::Bool(false), ParamValue::Bool(true)]
            }
        }
    }

    /// Whether a concrete value is a legal member of this distribution.
    pub fn contains(&self, value: &ParamValue) -> bool {
        match (self, value) {
            (Distribution::Float { low, high, .. }, ParamValue::Float(v)) => {
                *v >= *low - 1e-9 && *v <= *high + 1e-9
            }
            (Distribution::Int { low, high, .. }, ParamValue::Int(v)) => *v >= *low && *v <= *high,
            (Distribution::Categorical { choices }, ParamValue::Categorical(v)) => {
                choices.contains(v)
            }
            (Distribution::Bool, ParamValue::Bool(_)) => true,
            _ => false,
        }
    }
}

/// Round `raw` to the nearest `low + k*step`, clamped to `[low, high]`.
fn snap_to_step(raw: f64, low: f64, high: f64, step: f64) -> f64 {
    let k = ((raw - low) / step).round();
    (low + k * step).clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn float_sample_within_bounds() {
        let d = Distribution::float(-1.0, 1.0);
        let mut r = rng();
        for _ in 0..1000 {
            if let ParamValue::Float(v) = d.sample(&mut r) {
                assert!((-1.0..=1.0).contains(&v));
            } else {
                panic!("expected float");
            }
        }
    }

    #[test]
    fn log_float_within_bounds() {
        let d = Distribution::log_float(1e-4, 1e-1);
        let mut r = rng();
        for _ in 0..1000 {
            if let ParamValue::Float(v) = d.sample(&mut r) {
                assert!((1e-4 - 1e-9..=1e-1 + 1e-9).contains(&v), "v={v}");
            }
        }
    }

    #[test]
    fn int_sample_respects_step() {
        let d = Distribution::Int {
            low: 0,
            high: 10,
            log: false,
            step: 2,
        };
        let mut r = rng();
        for _ in 0..1000 {
            if let ParamValue::Int(v) = d.sample(&mut r) {
                assert!((0..=10).contains(&v));
                assert_eq!(v % 2, 0, "v={v} not on step grid");
            }
        }
    }

    #[test]
    fn categorical_sample_is_a_choice() {
        let d = Distribution::categorical(["a", "b", "c"]);
        let mut r = rng();
        for _ in 0..100 {
            if let ParamValue::Categorical(s) = d.sample(&mut r) {
                assert!(["a", "b", "c"].contains(&s.as_str()));
            }
        }
    }

    #[test]
    fn validate_rejects_bad_bounds() {
        assert!(Distribution::float(2.0, 1.0).validate("x").is_err());
        assert!(Distribution::log_float(-1.0, 1.0).validate("x").is_err());
        assert!(Distribution::categorical(Vec::<String>::new())
            .validate("x")
            .is_err());
        assert!(Distribution::float(0.0, 1.0).validate("x").is_ok());
    }

    #[test]
    fn grid_int_enumerates_step() {
        let d = Distribution::Int {
            low: 0,
            high: 6,
            log: false,
            step: 3,
        };
        let g = d.grid_values(5);
        assert_eq!(
            g,
            vec![ParamValue::Int(0), ParamValue::Int(3), ParamValue::Int(6)]
        );
    }

    #[test]
    fn grid_float_endpoints() {
        let d = Distribution::float(0.0, 10.0);
        let g = d.grid_values(3);
        assert_eq!(
            g,
            vec![
                ParamValue::Float(0.0),
                ParamValue::Float(5.0),
                ParamValue::Float(10.0)
            ]
        );
    }
}
