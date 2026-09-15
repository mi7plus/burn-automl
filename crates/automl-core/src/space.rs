//! Search spaces: ordered parameter definitions with optional conditional gating.
//!
//! This is the define-and-run structure a [`crate::sampler::Sampler`] receives.
//! It supports the flat forms and the conditional/hierarchical branches from
//! PRD §5 — a Transformer branch's attention parameters are only *active* once
//! the `model` categorical selects `"transformer"`.
//!
//! Conditions form a DAG rooted at unconditional parameters. A parameter is
//! *active* for a given [`ParamSet`] iff its condition (if any) is satisfied by
//! already-sampled parent values. Parameters are stored in declaration order,
//! and callers must declare a parent before any child that gates on it; this is
//! validated by [`SearchSpace::validate`].

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::{ParamSet, ParamValue};
use serde::{Deserialize, Serialize};

/// A predicate over an already-sampled parent parameter that gates whether a
/// child parameter is active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    /// The parent parameter this condition inspects.
    pub parent: String,
    /// The parent values for which the gated parameter becomes active.
    pub equals: Vec<ParamValue>,
}

impl Condition {
    /// Active when `parent` equals any of `values`.
    pub fn when_in(
        parent: impl Into<String>,
        values: impl IntoIterator<Item = ParamValue>,
    ) -> Self {
        Condition {
            parent: parent.into(),
            equals: values.into_iter().collect(),
        }
    }

    /// Active when `parent` equals the given categorical string.
    pub fn when_eq(parent: impl Into<String>, value: impl Into<String>) -> Self {
        Condition {
            parent: parent.into(),
            equals: vec![ParamValue::Categorical(value.into())],
        }
    }

    /// Evaluate the condition against a (partial) parameter set. A condition
    /// whose parent has not been sampled yet is treated as unsatisfied.
    pub fn is_satisfied(&self, params: &ParamSet) -> bool {
        match params.get(&self.parent) {
            Some(v) => self.equals.contains(v),
            None => false,
        }
    }
}

/// A single named parameter definition within a search space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDef {
    /// Unique parameter name within the space.
    pub name: String,
    /// The value space to sample from.
    pub distribution: Distribution,
    /// Optional gate; `None` means always active.
    pub condition: Option<Condition>,
}

/// An ordered collection of parameter definitions describing a search space.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchSpace {
    params: Vec<ParamDef>,
}

impl SearchSpace {
    /// An empty search space.
    pub fn new() -> Self {
        SearchSpace { params: Vec::new() }
    }

    /// Add an always-active parameter. Panics on duplicate names in debug via
    /// [`SearchSpace::validate`]; prefer [`SearchSpace::try_add`] to surface an error.
    pub fn add(mut self, name: impl Into<String>, distribution: Distribution) -> Self {
        self.params.push(ParamDef {
            name: name.into(),
            distribution,
            condition: None,
        });
        self
    }

    /// Add a conditionally-active parameter gated by `condition`.
    pub fn add_conditional(
        mut self,
        name: impl Into<String>,
        distribution: Distribution,
        condition: Condition,
    ) -> Self {
        self.params.push(ParamDef {
            name: name.into(),
            distribution,
            condition: Some(condition),
        });
        self
    }

    /// Fallible push that rejects duplicate names immediately.
    pub fn try_add(&mut self, def: ParamDef) -> Result<()> {
        if self.params.iter().any(|p| p.name == def.name) {
            return Err(Error::InvalidDistribution {
                name: def.name,
                reason: "duplicate parameter name in search space".into(),
            });
        }
        self.params.push(def);
        Ok(())
    }

    /// All parameter definitions in declaration order.
    pub fn params(&self) -> &[ParamDef] {
        &self.params
    }

    /// Look up a parameter definition by name.
    pub fn get(&self, name: &str) -> Option<&ParamDef> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Number of parameter definitions (active or not).
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the space has no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Whether `def` is active given the values sampled so far.
    pub fn is_active(def: &ParamDef, params: &ParamSet) -> bool {
        match &def.condition {
            None => true,
            Some(c) => c.is_satisfied(params),
        }
    }

    /// Validate structural integrity of the space:
    /// - every distribution has legal bounds,
    /// - names are unique,
    /// - each condition references a parent declared *earlier* (no cycles, and
    ///   parents are always sampled before their children).
    pub fn validate(&self) -> Result<()> {
        let mut seen: Vec<&str> = Vec::new();
        for def in &self.params {
            def.distribution.validate(&def.name)?;
            if seen.contains(&def.name.as_str()) {
                return Err(Error::InvalidDistribution {
                    name: def.name.clone(),
                    reason: "duplicate parameter name in search space".into(),
                });
            }
            if let Some(cond) = &def.condition {
                if !seen.contains(&cond.parent.as_str()) {
                    return Err(Error::InvalidDistribution {
                        name: def.name.clone(),
                        reason: format!(
                            "condition references parent `{}` which is not declared before it",
                            cond.parent
                        ),
                    });
                }
                // The parent must be able to take the gate values.
                if let Some(parent_def) = self.get(&cond.parent) {
                    for v in &cond.equals {
                        if !parent_def.distribution.contains(v) {
                            return Err(Error::InvalidDistribution {
                                name: def.name.clone(),
                                reason: format!(
                                    "condition value {:?} is not in parent `{}`'s distribution",
                                    v, cond.parent
                                ),
                            });
                        }
                    }
                }
            }
            seen.push(&def.name);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conditional_space() -> SearchSpace {
        SearchSpace::new()
            .add("model", Distribution::categorical(["mlp", "transformer"]))
            .add_conditional(
                "d_model",
                Distribution::int(64, 512),
                Condition::when_eq("model", "transformer"),
            )
            .add_conditional(
                "hidden",
                Distribution::int(16, 256),
                Condition::when_eq("model", "mlp"),
            )
    }

    #[test]
    fn validate_accepts_well_ordered_conditions() {
        assert!(conditional_space().validate().is_ok());
    }

    #[test]
    fn validate_rejects_forward_reference() {
        let space = SearchSpace::new().add_conditional(
            "d_model",
            Distribution::int(64, 512),
            Condition::when_eq("model", "transformer"),
        );
        assert!(space.validate().is_err());
    }

    #[test]
    fn activity_depends_on_parent() {
        let space = conditional_space();
        let mut p = ParamSet::new();
        p.insert("model", ParamValue::Categorical("transformer".into()));
        let d_model = space.get("d_model").unwrap();
        let hidden = space.get("hidden").unwrap();
        assert!(SearchSpace::is_active(d_model, &p));
        assert!(!SearchSpace::is_active(hidden, &p));
    }

    #[test]
    fn validate_rejects_bad_condition_value() {
        let space = SearchSpace::new()
            .add("model", Distribution::categorical(["mlp"]))
            .add_conditional(
                "x",
                Distribution::int(1, 2),
                Condition::when_eq("model", "not-a-choice"),
            );
        assert!(space.validate().is_err());
    }
}
