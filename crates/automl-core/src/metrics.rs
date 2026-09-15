//! Named metrics and optimization directions.
//!
//! Per PRD §5, objective results are *named* rather than positional so quality,
//! latency, memory, model size and domain metrics coexist in one objective
//! without positional-index bugs. A study declares one [`Direction`] per named
//! objective; single-direction studies are single-objective, multi-direction
//! studies feed the `ParetoFront` machinery (§17, arriving in a later release).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Whether an objective should be minimized or maximized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Lower is better (e.g. loss, latency).
    Minimize,
    /// Higher is better (e.g. accuracy, F1).
    Maximize,
}

impl Direction {
    /// Returns true if `a` is a better value than `b` under this direction.
    pub fn is_better(&self, a: f64, b: f64) -> bool {
        match self {
            Direction::Minimize => a < b,
            Direction::Maximize => a > b,
        }
    }

    /// The worst possible value under this direction, useful as a fold seed.
    pub fn worst(&self) -> f64 {
        match self {
            Direction::Minimize => f64::INFINITY,
            Direction::Maximize => f64::NEG_INFINITY,
        }
    }
}

/// A named objective paired with its optimization direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    /// The metric name reported by the objective (must match a key in `NamedMetrics`).
    pub name: String,
    /// Whether to minimize or maximize this metric.
    pub direction: Direction,
}

impl Objective {
    /// A metric to be minimized.
    pub fn minimize(name: impl Into<String>) -> Self {
        Objective {
            name: name.into(),
            direction: Direction::Minimize,
        }
    }

    /// A metric to be maximized.
    pub fn maximize(name: impl Into<String>) -> Self {
        Objective {
            name: name.into(),
            direction: Direction::Maximize,
        }
    }
}

/// A bag of named scalar metrics reported by a trial (intermediate or final).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NamedMetrics {
    values: BTreeMap<String, f64>,
}

impl NamedMetrics {
    /// An empty metrics bag.
    pub fn new() -> Self {
        NamedMetrics {
            values: BTreeMap::new(),
        }
    }

    /// Construct from a single name/value pair.
    pub fn single(name: impl Into<String>, value: f64) -> Self {
        let mut m = NamedMetrics::new();
        m.insert(name, value);
        m
    }

    /// Insert or overwrite a metric.
    pub fn insert(&mut self, name: impl Into<String>, value: f64) {
        self.values.insert(name.into(), value);
    }

    /// Builder-style insert.
    pub fn with(mut self, name: impl Into<String>, value: f64) -> Self {
        self.insert(name, value);
        self
    }

    /// Look up a metric by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }

    /// Iterate metrics in deterministic (sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.values.iter()
    }

    /// Number of metrics.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no metrics.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_is_better() {
        assert!(Direction::Minimize.is_better(1.0, 2.0));
        assert!(!Direction::Minimize.is_better(2.0, 1.0));
        assert!(Direction::Maximize.is_better(2.0, 1.0));
    }

    #[test]
    fn metrics_named_access() {
        let m = NamedMetrics::new()
            .with("loss", 0.5)
            .with("latency_ms", 12.0);
        assert_eq!(m.get("loss"), Some(0.5));
        assert_eq!(m.get("latency_ms"), Some(12.0));
        assert_eq!(m.get("nope"), None);
    }
}
