//! Budget as a first-class trait (PRD §4, §2).
//!
//! The PRD is explicit (executive summary gap #6): budget is a *trait* with a
//! small closed set of built-in implementations, not an open enum. This
//! resolves the ambiguity for `Executor`/`Pruner` interfaces — anything that
//! can decide "are we done?" from a [`Consumption`] tally is a budget.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Running tally of resources a study has consumed, compared against a budget.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Consumption {
    /// Number of trials started.
    pub trials: u64,
    /// Wall-clock time elapsed since the study began.
    pub wall_time: Duration,
    /// Total training epochs consumed across trials.
    pub epochs: u64,
    /// Total environment/optimizer steps consumed.
    pub steps: u64,
    /// GPU-hours consumed.
    pub gpu_hours: f64,
}

impl Consumption {
    /// A zeroed tally.
    pub fn new() -> Self {
        Consumption::default()
    }
}

/// How much of a budget remains. Reported for observability; the authoritative
/// decision is [`Budget::is_exhausted`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetRemaining {
    /// A finite, unitless fraction in `[0.0, 1.0]` of the budget remaining.
    Fraction(f64),
    /// The budget cannot be expressed as a simple fraction (e.g. wall time with
    /// an unknown per-trial cost); only exhaustion is meaningful.
    Unknown,
    /// The budget is unbounded.
    Unbounded,
}

/// The exhaustion contract every budget implements (PRD §4.1).
pub trait Budget: Send + Sync {
    /// Whether the study should stop launching new trials.
    fn is_exhausted(&self, consumed: &Consumption) -> bool;

    /// A best-effort estimate of remaining budget for reporting.
    fn remaining(&self, consumed: &Consumption) -> BudgetRemaining;

    /// A short label for logging and provenance.
    fn describe(&self) -> String;
}

/// Stop after a fixed number of trials.
#[derive(Debug, Clone, Copy)]
pub struct TrialBudget {
    /// Maximum number of trials to run.
    pub max_trials: u64,
}

impl Budget for TrialBudget {
    fn is_exhausted(&self, consumed: &Consumption) -> bool {
        consumed.trials >= self.max_trials
    }
    fn remaining(&self, consumed: &Consumption) -> BudgetRemaining {
        if self.max_trials == 0 {
            return BudgetRemaining::Fraction(0.0);
        }
        let used = consumed.trials.min(self.max_trials) as f64 / self.max_trials as f64;
        BudgetRemaining::Fraction((1.0 - used).max(0.0))
    }
    fn describe(&self) -> String {
        format!("{} trials", self.max_trials)
    }
}

/// Stop after a wall-clock duration.
#[derive(Debug, Clone, Copy)]
pub struct WallTimeBudget {
    /// Maximum wall-clock duration.
    pub limit: Duration,
}

impl Budget for WallTimeBudget {
    fn is_exhausted(&self, consumed: &Consumption) -> bool {
        consumed.wall_time >= self.limit
    }
    fn remaining(&self, consumed: &Consumption) -> BudgetRemaining {
        if self.limit.is_zero() {
            return BudgetRemaining::Fraction(0.0);
        }
        let frac = 1.0 - consumed.wall_time.as_secs_f64() / self.limit.as_secs_f64();
        BudgetRemaining::Fraction(frac.clamp(0.0, 1.0))
    }
    fn describe(&self) -> String {
        format!("{:?} wall time", self.limit)
    }
}

/// Stop after a total number of training epochs across all trials.
#[derive(Debug, Clone, Copy)]
pub struct EpochBudget {
    /// Maximum total epochs.
    pub max_epochs: u64,
}

impl Budget for EpochBudget {
    fn is_exhausted(&self, consumed: &Consumption) -> bool {
        consumed.epochs >= self.max_epochs
    }
    fn remaining(&self, consumed: &Consumption) -> BudgetRemaining {
        if self.max_epochs == 0 {
            return BudgetRemaining::Fraction(0.0);
        }
        let used = consumed.epochs.min(self.max_epochs) as f64 / self.max_epochs as f64;
        BudgetRemaining::Fraction((1.0 - used).max(0.0))
    }
    fn describe(&self) -> String {
        format!("{} epochs", self.max_epochs)
    }
}

/// A budget that never exhausts. Useful for interactive exploration where the
/// caller stops the study manually.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unbounded;

impl Budget for Unbounded {
    fn is_exhausted(&self, _consumed: &Consumption) -> bool {
        false
    }
    fn remaining(&self, _consumed: &Consumption) -> BudgetRemaining {
        BudgetRemaining::Unbounded
    }
    fn describe(&self) -> String {
        "unbounded".to_string()
    }
}

/// Convenience constructors mirroring the high-level API sketch in PRD §20
/// (`Budget::trials(100)`).
impl dyn Budget {
    /// A [`TrialBudget`].
    pub fn trials(max_trials: u64) -> TrialBudget {
        TrialBudget { max_trials }
    }
    /// A [`WallTimeBudget`].
    pub fn wall_time(limit: Duration) -> WallTimeBudget {
        WallTimeBudget { limit }
    }
    /// An [`EpochBudget`].
    pub fn epochs(max_epochs: u64) -> EpochBudget {
        EpochBudget { max_epochs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trial_budget_exhausts() {
        let b = TrialBudget { max_trials: 3 };
        let mut c = Consumption::new();
        assert!(!b.is_exhausted(&c));
        c.trials = 3;
        assert!(b.is_exhausted(&c));
        assert_eq!(
            b.remaining(&Consumption {
                trials: 0,
                ..Default::default()
            }),
            BudgetRemaining::Fraction(1.0)
        );
    }

    #[test]
    fn wall_time_budget() {
        let b = WallTimeBudget {
            limit: Duration::from_secs(10),
        };
        let c = Consumption {
            wall_time: Duration::from_secs(5),
            ..Default::default()
        };
        assert!(!b.is_exhausted(&c));
        assert_eq!(b.remaining(&c), BudgetRemaining::Fraction(0.5));
    }

    #[test]
    fn unbounded_never_exhausts() {
        let b = Unbounded;
        assert!(!b.is_exhausted(&Consumption {
            trials: u64::MAX,
            ..Default::default()
        }));
    }
}
