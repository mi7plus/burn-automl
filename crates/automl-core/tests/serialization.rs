//! Serialization round-trip tests for the persisted types (post-1.0).
//!
//! Studies are stored as JSON-serialized `ParamSet` / `NamedMetrics` / provenance
//! and can be reloaded across versions (the migration contract). These tests pin
//! that the core value types survive a serde round-trip unchanged, so a schema or
//! type change that would silently break persistence fails a test instead.

use automl_core::distribution::Distribution;
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::param::{ParamSet, ParamValue};
use automl_core::space::{Condition, SearchSpace};
use automl_core::trial::{IntermediateReport, StudyId, TrialId, TrialRecord, TrialState};

fn roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json = serde_json::to_string(value).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn param_set_roundtrips_all_value_types() {
    let mut p = ParamSet::new();
    p.insert("f", ParamValue::Float(1.5));
    p.insert("i", ParamValue::Int(-3));
    p.insert("c", ParamValue::Categorical("relu".into()));
    p.insert("b", ParamValue::Bool(true));
    assert_eq!(roundtrip(&p), p);
}

#[test]
fn conditional_search_space_roundtrips() {
    let space = SearchSpace::new()
        .add("model", Distribution::categorical(["mlp", "cnn"]))
        .add("lr", Distribution::log_float(1e-4, 1e-1))
        .add("depth", Distribution::int(1, 8))
        .add_conditional(
            "kernel",
            Distribution::int(3, 7),
            Condition::when_eq("model", "cnn"),
        )
        .add("use_bn", Distribution::boolean());
    let back = roundtrip(&space);
    assert_eq!(back, space);
    // The restored space is still valid and preserves the conditional gate.
    assert!(back.validate().is_ok());
    assert!(back.get("kernel").unwrap().condition.is_some());
}

#[test]
fn named_metrics_roundtrip_stable_order() {
    let m = NamedMetrics::new()
        .with("accuracy", 0.93)
        .with("loss", 0.21)
        .with("latency_ms", 4.5);
    let back = roundtrip(&m);
    assert_eq!(back, m);
    // BTreeMap backing keeps a stable, sorted iteration order.
    let keys: Vec<&String> = back.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, vec!["accuracy", "latency_ms", "loss"]);
}

#[test]
fn trial_record_roundtrips_with_history_and_provenance() {
    let mut rec = TrialRecord::new(
        TrialId(7),
        StudyId(2),
        ParamSet::new().with("x", ParamValue::Float(0.5)),
        7,
    );
    rec.state = TrialState::Complete;
    rec.intermediate = vec![
        IntermediateReport {
            step: 1,
            metrics: NamedMetrics::single("loss", 0.9),
        },
        IntermediateReport {
            step: 2,
            metrics: NamedMetrics::single("loss", 0.4),
        },
    ];
    rec.final_metrics = Some(NamedMetrics::single("loss", 0.3));

    let back = roundtrip(&rec);
    assert_eq!(back, rec);
    assert_eq!(back.state, TrialState::Complete);
    assert_eq!(back.intermediate.len(), 2);
    assert_eq!(back.final_value("loss"), Some(0.3));
    assert_eq!(back.params.float("x").unwrap(), 0.5);
}

#[test]
fn direction_roundtrips() {
    assert_eq!(roundtrip(&Direction::Minimize), Direction::Minimize);
    assert_eq!(roundtrip(&Direction::Maximize), Direction::Maximize);
}
