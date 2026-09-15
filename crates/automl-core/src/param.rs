//! Concrete parameter values and the [`ParamSet`] that groups them for a trial.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single concrete parameter value sampled from a [`crate::distribution::Distribution`].
///
/// Ordering of the [`BTreeMap`] in [`ParamSet`] plus the total ordering here keeps
/// serialized parameter sets deterministic, which the PRD requires for replay
/// (§2 "deterministic replay").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParamValue {
    /// A floating-point value.
    Float(f64),
    /// An integer value.
    Int(i64),
    /// One choice from a categorical distribution.
    Categorical(String),
    /// A boolean.
    Bool(bool),
}

impl ParamValue {
    /// A short, stable label used in type-mismatch error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            ParamValue::Float(_) => "float",
            ParamValue::Int(_) => "int",
            ParamValue::Categorical(_) => "categorical",
            ParamValue::Bool(_) => "bool",
        }
    }

    /// Interpret this value as `f64` if it is a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Interpret this value as `i64` if it is an int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Interpret this value as a categorical `&str` if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Categorical(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Interpret this value as a `bool` if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// An ordered, named collection of concrete parameter values — the output of a
/// [`crate::sampler::Sampler`] and the input to an objective evaluation.
///
/// A [`BTreeMap`] backs it so iteration order is deterministic regardless of
/// insertion order, which keeps replay stable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParamSet {
    values: BTreeMap<String, ParamValue>,
}

impl ParamSet {
    /// An empty parameter set.
    pub fn new() -> Self {
        ParamSet {
            values: BTreeMap::new(),
        }
    }

    /// Insert or overwrite a named value.
    pub fn insert(&mut self, name: impl Into<String>, value: ParamValue) {
        self.values.insert(name.into(), value);
    }

    /// Builder-style insert.
    pub fn with(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.insert(name, value);
        self
    }

    /// Look up a raw value by name.
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.values.get(name)
    }

    /// Whether a parameter is present.
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate name/value pairs in deterministic (sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParamValue)> {
        self.values.iter()
    }

    /// Typed accessor for a float parameter.
    pub fn float(&self, name: &str) -> Result<f64> {
        self.typed(name, ParamValue::as_float, "float")
    }

    /// Typed accessor for an int parameter.
    pub fn int(&self, name: &str) -> Result<i64> {
        self.typed(name, ParamValue::as_int, "int")
    }

    /// Typed accessor for a categorical parameter.
    pub fn categorical(&self, name: &str) -> Result<&str> {
        let v = self
            .values
            .get(name)
            .ok_or_else(|| Error::MissingParam(name.to_string()))?;
        v.as_str().ok_or_else(|| Error::ParamTypeMismatch {
            name: name.to_string(),
            actual: v.kind().to_string(),
            expected: "categorical".to_string(),
        })
    }

    /// Typed accessor for a boolean parameter.
    pub fn boolean(&self, name: &str) -> Result<bool> {
        self.typed(name, ParamValue::as_bool, "bool")
    }

    fn typed<T>(
        &self,
        name: &str,
        f: impl Fn(&ParamValue) -> Option<T>,
        expected: &'static str,
    ) -> Result<T> {
        let v = self
            .values
            .get(name)
            .ok_or_else(|| Error::MissingParam(name.to_string()))?;
        f(v).ok_or_else(|| Error::ParamTypeMismatch {
            name: name.to_string(),
            actual: v.kind().to_string(),
            expected: expected.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_accessors_roundtrip() {
        let mut p = ParamSet::new();
        p.insert("lr", ParamValue::Float(0.01));
        p.insert("depth", ParamValue::Int(4));
        p.insert("model", ParamValue::Categorical("mlp".into()));
        p.insert("bn", ParamValue::Bool(true));

        assert_eq!(p.float("lr").unwrap(), 0.01);
        assert_eq!(p.int("depth").unwrap(), 4);
        assert_eq!(p.categorical("model").unwrap(), "mlp");
        assert!(p.boolean("bn").unwrap());
    }

    #[test]
    fn missing_and_mismatch_errors() {
        let mut p = ParamSet::new();
        p.insert("lr", ParamValue::Float(0.01));
        assert!(matches!(p.float("nope"), Err(Error::MissingParam(_))));
        assert!(matches!(p.int("lr"), Err(Error::ParamTypeMismatch { .. })));
    }

    #[test]
    fn iteration_is_sorted() {
        let mut p = ParamSet::new();
        p.insert("z", ParamValue::Int(1));
        p.insert("a", ParamValue::Int(2));
        let keys: Vec<_> = p.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec!["a".to_string(), "z".to_string()]);
    }
}
