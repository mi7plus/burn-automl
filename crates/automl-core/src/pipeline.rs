//! Pipeline-AutoML composition primitives (roadmap v0.9; PRD §16).
//!
//! The long-term differentiator is searching *complete pipelines* — data →
//! preprocessing → model → postprocessing — not just a model's hyperparameters.
//! The PRD is emphatic that this introduces **no separate pipeline DSL**:
//! pipeline nodes are conditional subspaces composed via the same [`SearchSpace`]
//! primitives used everywhere else, so the sampler, pruner and storage layers
//! stay unaware that pipeline search is even happening (§16).
//!
//! A [`PipelineSpace`] is an ordered list of [`Stage`]s (e.g. `preprocess`,
//! `model`, `postprocess`); each stage offers candidate [`Component`]s, and each
//! component carries its own hyperparameter subspace that is *active only when
//! that component is selected*. [`PipelineSpace::to_search_space`] lowers this to
//! a flat conditional [`SearchSpace`]; [`PipelineSpace::decode`] lifts a sampled
//! [`ParamSet`] back to a concrete [`PipelinePlan`]. This is the same
//! encode/decode pattern as [`crate::nas`], applied to whole pipelines.

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::{ParamSet, ParamValue};
use crate::space::{Condition, SearchSpace};
use serde::{Deserialize, Serialize};

/// The separator between a stage, a component and a parameter in the encoded
/// search-space key (`"model.knn.k"`). Chosen not to collide with typical names.
const SEP: char = '.';

/// One candidate for a stage, with its own conditional hyperparameter subspace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Component {
    name: String,
    params: Vec<(String, Distribution)>,
}

impl Component {
    /// A component with no hyperparameters (e.g. a `"none"` / passthrough option).
    pub fn new(name: impl Into<String>) -> Self {
        Component {
            name: name.into(),
            params: Vec::new(),
        }
    }

    /// Add a hyperparameter to this component's subspace. It becomes active only
    /// when this component is selected for its stage.
    pub fn param(mut self, name: impl Into<String>, distribution: Distribution) -> Self {
        self.params.push((name.into(), distribution));
        self
    }

    /// The component name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A pipeline stage: a named choice point offering one of several components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    name: String,
    components: Vec<Component>,
}

impl Stage {
    /// A new, empty stage (e.g. `"preprocess"`).
    pub fn new(name: impl Into<String>) -> Self {
        Stage {
            name: name.into(),
            components: Vec::new(),
        }
    }

    /// Add a candidate component to this stage.
    pub fn component(mut self, component: Component) -> Self {
        self.components.push(component);
        self
    }

    /// The stage name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The candidate components.
    pub fn components(&self) -> &[Component] {
        &self.components
    }
}

/// A searchable pipeline: an ordered list of stages, each a conditional choice of
/// component plus that component's hyperparameters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineSpace {
    stages: Vec<Stage>,
}

impl PipelineSpace {
    /// An empty pipeline space.
    pub fn new() -> Self {
        PipelineSpace { stages: Vec::new() }
    }

    /// Append a stage.
    pub fn stage(mut self, stage: Stage) -> Self {
        self.stages = {
            let mut s = self.stages;
            s.push(stage);
            s
        };
        self
    }

    /// The stages, in order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Lower the pipeline to a flat conditional [`SearchSpace`]: one categorical
    /// per stage (the component choice), then each component's parameters gated on
    /// that choice. Parameter keys are `"{stage}.{component}.{param}"`.
    pub fn to_search_space(&self) -> SearchSpace {
        let mut space = SearchSpace::new();
        for stage in &self.stages {
            let names: Vec<String> = stage.components.iter().map(|c| c.name.clone()).collect();
            space = space.add(stage.name.clone(), Distribution::categorical(names));
            for component in &stage.components {
                let gate = Condition {
                    parent: stage.name.clone(),
                    equals: vec![ParamValue::Categorical(component.name.clone())],
                };
                for (pname, dist) in &component.params {
                    let key = format!("{}{SEP}{}{SEP}{}", stage.name, component.name, pname);
                    space = space.add_conditional(key, dist.clone(), gate.clone());
                }
            }
        }
        space
    }

    /// Lift a sampled [`ParamSet`] into a concrete [`PipelinePlan`].
    pub fn decode(&self, params: &ParamSet) -> Result<PipelinePlan> {
        let mut chosen = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let component = params.categorical(&stage.name)?.to_string();
            let Some(comp) = stage.components.iter().find(|c| c.name == component) else {
                return Err(Error::Objective(format!(
                    "stage '{}' selected unknown component '{}'",
                    stage.name, component
                )));
            };
            let mut sub = ParamSet::new();
            for (pname, _) in &comp.params {
                let key = format!("{}{SEP}{}{SEP}{}", stage.name, component, pname);
                if let Some(v) = params.get(&key) {
                    sub.insert(pname.clone(), v.clone());
                }
            }
            chosen.push(StageChoice {
                stage: stage.name.clone(),
                component,
                params: sub,
            });
        }
        Ok(PipelinePlan { stages: chosen })
    }
}

/// The chosen component (and its hyperparameters) for one stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageChoice {
    /// The stage name.
    pub stage: String,
    /// The selected component's name.
    pub component: String,
    /// The selected component's hyperparameters, keyed by their bare names.
    pub params: ParamSet,
}

/// A decoded pipeline: the component chosen at each stage, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelinePlan {
    /// One choice per stage, in declaration order.
    pub stages: Vec<StageChoice>,
}

impl PipelinePlan {
    /// The component chosen for a named stage, if present.
    pub fn component(&self, stage: &str) -> Option<&str> {
        self.stages
            .iter()
            .find(|c| c.stage == stage)
            .map(|c| c.component.as_str())
    }

    /// The chosen component and its parameters for a named stage.
    pub fn choice(&self, stage: &str) -> Option<&StageChoice> {
        self.stages.iter().find(|c| c.stage == stage)
    }

    /// A short, stable description for logs/provenance, e.g.
    /// `preprocess=standardize · model=knn · postprocess=none`.
    pub fn describe(&self) -> String {
        self.stages
            .iter()
            .map(|c| format!("{}={}", c.stage, c.component))
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{RandomSampler, Sampler};
    use crate::trial::TrialHistory;

    fn space() -> PipelineSpace {
        PipelineSpace::new()
            .stage(
                Stage::new("preprocess")
                    .component(Component::new("standardize"))
                    .component(Component::new("normalize"))
                    .component(Component::new("none")),
            )
            .stage(
                Stage::new("model")
                    .component(Component::new("knn").param("k", Distribution::int(1, 15)))
                    .component(Component::new("centroid")),
            )
    }

    #[test]
    fn encodes_stages_and_gated_component_params() {
        let s = space().to_search_space();
        assert!(s.validate().is_ok());
        // 2 stage categoricals + 1 gated param (model.knn.k) = 3.
        assert_eq!(s.len(), 3);
        assert_eq!(s.get("preprocess").unwrap().condition, None);
        assert_eq!(s.get("model").unwrap().condition, None);
        assert!(s.get("model.knn.k").unwrap().condition.is_some());
    }

    #[test]
    fn decodes_choices_and_component_params() {
        let ps = space();
        let sp = ps.to_search_space();
        let mut sampler = RandomSampler::new(3);
        let history = TrialHistory::default();
        for _ in 0..300 {
            let params = sampler.suggest(&sp, &history);
            let plan = ps.decode(&params).unwrap();
            assert_eq!(plan.stages.len(), 2);
            let pre = plan.component("preprocess").unwrap();
            assert!(["standardize", "normalize", "none"].contains(&pre));
            let model = plan.choice("model").unwrap();
            if model.component == "knn" {
                // The gated k is present and within range only for the knn branch.
                let k = model.params.int("k").unwrap();
                assert!((1..=15).contains(&k));
            } else {
                assert!(model.params.int("k").is_err());
            }
        }
    }

    #[test]
    fn describe_is_readable() {
        let ps = space();
        let sp = ps.to_search_space();
        let mut sampler = RandomSampler::new(1);
        let plan = ps
            .decode(&sampler.suggest(&sp, &TrialHistory::default()))
            .unwrap();
        let d = plan.describe();
        assert!(d.starts_with("preprocess="));
        assert!(d.contains("· model="));
    }
}
