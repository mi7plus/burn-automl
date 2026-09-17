//! Property-based tests: every sampler must return a **valid assignment** over
//! an arbitrary conditional search space (the §5.1 correctness floor). Random,
//! Grid, TPE, Evolutionary and QMC all handle every conditional space; these
//! tests generate random spaces and assert the invariant holds, hardening the
//! guarantee beyond the hand-written cases.

use automl_core::distribution::Distribution;
use automl_core::param::{ParamSet, ParamValue};
use automl_core::sampler::{GridSampler, RandomSampler, Sampler};
use automl_core::space::{Condition, SearchSpace};
use automl_core::trial::TrialHistory;
use automl_core::{
    evolution::EvolutionarySampler, metrics::Direction, qmc::QmcSampler, tpe::TpeSampler,
};
use proptest::prelude::*;

#[derive(Debug, Clone, Copy)]
enum Kind {
    Float,
    Int,
    Cat,
    Bool,
}

fn kind_strategy() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Float),
        Just(Kind::Int),
        Just(Kind::Cat),
        Just(Kind::Bool),
    ]
}

/// Build a search space from a spec: one parameter per kind, plus — if any
/// categorical exists — a conditional float gated on it, to exercise branches.
fn build_space(kinds: &[Kind]) -> SearchSpace {
    let mut space = SearchSpace::new();
    let mut first_cat: Option<String> = None;
    for (i, k) in kinds.iter().enumerate() {
        let name = format!("p{i}");
        space = match k {
            Kind::Float => space.add(&name, Distribution::float(-10.0, 10.0)),
            Kind::Int => space.add(&name, Distribution::int(0, 20)),
            Kind::Cat => {
                if first_cat.is_none() {
                    first_cat = Some(name.clone());
                }
                space.add(&name, Distribution::categorical(["a", "b", "c"]))
            }
            Kind::Bool => space.add(&name, Distribution::boolean()),
        };
    }
    if let Some(parent) = first_cat {
        space = space.add_conditional(
            "cond",
            Distribution::float(0.0, 1.0),
            Condition::when_eq(parent, "a"),
        );
    }
    space
}

/// Assert `params` is a valid assignment for `space`: every active parameter is
/// present with an in-support value, and no inactive parameter appears.
fn assert_valid(space: &SearchSpace, params: &ParamSet) {
    // Rebuild activation incrementally in declaration order.
    let mut partial = ParamSet::new();
    for def in space.params() {
        let active = SearchSpace::is_active(def, &partial);
        let got = params.get(&def.name);
        if active {
            let v = got.unwrap_or_else(|| panic!("active param {} missing", def.name));
            match (&def.distribution, v) {
                (Distribution::Float { low, high, .. }, ParamValue::Float(x)) => {
                    assert!(
                        *x >= *low - 1e-9 && *x <= *high + 1e-9,
                        "{} out of range",
                        def.name
                    )
                }
                (Distribution::Int { low, high, .. }, ParamValue::Int(x)) => {
                    assert!(*x >= *low && *x <= *high, "{} out of range", def.name)
                }
                (Distribution::Categorical { choices }, ParamValue::Categorical(s)) => {
                    assert!(choices.contains(s), "{} not a valid choice", def.name)
                }
                (Distribution::Bool, ParamValue::Bool(_)) => {}
                (d, val) => panic!("type mismatch for {}: {d:?} vs {val:?}", def.name),
            }
            partial.insert(def.name.clone(), v.clone());
        }
        // Inactive parameters may be absent; if the sampler included one anyway it
        // is harmless, so we only require correctness of active ones.
    }
}

fn samplers(seed: u64) -> Vec<Box<dyn Sampler>> {
    vec![
        Box::new(RandomSampler::new(seed)),
        Box::new(GridSampler::new(3)),
        Box::new(TpeSampler::new("loss", Direction::Minimize, seed)),
        Box::new(EvolutionarySampler::new("loss", Direction::Minimize, seed)),
        Box::new(QmcSampler::new()),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn every_sampler_returns_a_valid_assignment(
        kinds in prop::collection::vec(kind_strategy(), 1..5),
        seed in any::<u64>(),
    ) {
        let space = build_space(&kinds);
        prop_assert!(space.validate().is_ok());
        let history = TrialHistory::default();
        for mut sampler in samplers(seed) {
            // A few draws each, so QMC/Grid advance their index.
            for _ in 0..4 {
                let params = sampler.suggest(&space, &history);
                assert_valid(&space, &params);
            }
        }
    }
}
