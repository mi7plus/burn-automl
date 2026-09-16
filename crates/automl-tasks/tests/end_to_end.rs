//! v1.0 end-to-end story test (PRD §28 definition of done).
//!
//! Each test exercises one strand of the "full HPO + NAS + multi-objective +
//! distributed + pipeline story" the v1.0 DoD requires, using the real public
//! API — proving the pieces compose, not just that each unit passes in isolation.

use automl_core::prelude::*;
use std::sync::Arc;

/// HPO: TPE minimizes a function, the study persists to SQLite, and a *resumed*
/// study continues without losing completed trials (§26/§28 recovery).
#[test]
fn hpo_persists_and_resumes() {
    let dir = std::env::temp_dir().join(format!("burn_automl_e2e_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&dir);
    let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open(&dir).unwrap());

    let space = SearchSpace::new()
        .add("x", Distribution::float(-5.0, 5.0))
        .add("y", Distribution::float(-5.0, 5.0));
    let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
        Ok(NamedMetrics::single(
            "loss",
            p.float("x")?.powi(2) + p.float("y")?.powi(2),
        ))
    };

    let study_id;
    {
        let mut study = Study::builder(space.clone())
            .name("e2e-hpo")
            .minimize("loss")
            .sampler(TpeSampler::new("loss", Direction::Minimize, 1))
            .storage(storage.clone())
            .seed(1)
            .build()
            .unwrap();
        study.optimize_n(&objective, 15).unwrap();
        study_id = study.id();
        assert_eq!(study.history().unwrap().completed().count(), 15);
    }

    // Resume against the same store and add more trials; earlier ones survive.
    let mut resumed = Study::resume(
        space,
        storage.clone(),
        study_id,
        TpeSampler::new("loss", Direction::Minimize, 1),
        Arc::new(NoPruner),
        1,
    )
    .unwrap();
    resumed.optimize_n(&objective, 10).unwrap();
    assert_eq!(resumed.history().unwrap().completed().count(), 25);
    let best = resumed.best_trial().unwrap().unwrap();
    // Optimization clearly beats the ~17 average loss of uniform sampling over
    // this box; the exact optimum is luck-dependent, the improvement is not.
    assert!(
        best.final_value("loss").unwrap() < 3.0,
        "search should improve on the mean"
    );

    let _ = std::fs::remove_file(&dir);
}

/// Multi-objective: a two-objective study yields a non-trivial Pareto front with
/// a positive hypervolume (§17/§28).
#[test]
fn multi_objective_builds_a_pareto_front() {
    // Minimize x^2 while maximizing x (a genuine trade-off over x in [0, 3]).
    let space = SearchSpace::new().add("x", Distribution::float(0.0, 3.0));
    let mut study = Study::builder(space)
        .name("e2e-mo")
        .minimize("sq")
        .maximize("lin")
        .sampler(RandomSampler::new(2))
        .seed(2)
        .build()
        .unwrap();
    study
        .optimize_n(
            &|p: &ParamSet, _s: &mut dyn ReportSink| {
                let x = p.float("x")?;
                Ok(NamedMetrics::new().with("sq", x * x).with("lin", x))
            },
            40,
        )
        .unwrap();

    let front = study.pareto_front().unwrap();
    assert!(
        front.len() >= 2,
        "a trade-off should yield multiple non-dominated points"
    );
    // Single-objective best_trial is None for a multi-objective study.
    assert!(study.best_trial().unwrap().is_none());
    // Reference worse than any point on both objectives -> positive hypervolume.
    let reference = NamedMetrics::new().with("sq", 10.0).with("lin", -1.0);
    assert!(study.hypervolume(&reference).unwrap() > 0.0);
}

/// NAS: a macro-architecture is encoded as a conditional space, searched by an
/// ordinary sampler, and every proposal decodes to a valid architecture (§21/§28
/// "NAS searches architecture graphs").
#[test]
fn nas_architecture_search_via_ordinary_optimization() {
    let palette = OpPalette::new(["conv3", "conv5", "identity"]).unwrap();
    let macro_space = MacroSpace::new(palette, 1, 4, vec![8, 16, 32]).unwrap();
    let space = macro_space.to_search_space();

    // A cheap proxy objective scoring decoded architectures — no training needed
    // to prove the search/encode/decode loop works end to end.
    let ms = macro_space.clone();
    let mut study = Study::builder(space)
        .name("e2e-nas")
        .maximize("score")
        .sampler(EvolutionarySampler::new("score", Direction::Maximize, 3))
        .seed(3)
        .build()
        .unwrap();
    study
        .optimize_n(
            &move |p: &ParamSet, _s: &mut dyn ReportSink| {
                let arch = ms.decode(p)?;
                // Prefer moderately deep nets with skips (a stand-in for accuracy).
                let score = arch.depth() as f64
                    + arch.layers.iter().filter(|l| l.skip_from_prev).count() as f64;
                Ok(NamedMetrics::single("score", score))
            },
            30,
        )
        .unwrap();
    let best = study.best_trial().unwrap().unwrap();
    // The decoded winner is a real, valid architecture.
    let arch = macro_space.decode(&best.params).unwrap();
    assert!((1..=4).contains(&arch.depth()));
}

/// Pipeline: the executable tabular pipeline search finds a working
/// preprocessing+model combination (§16/§28).
#[test]
fn pipeline_search_end_to_end() {
    use automl_tasks::AutoPipeline;
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..200 {
        let label = rng.gen_range(0..2);
        let signal = if label == 0 {
            rng.gen_range(-1.0..0.0)
        } else {
            rng.gen_range(0.0..1.0)
        };
        x.push(vec![rng.gen_range(-400.0f32..400.0), signal]);
        y.push(label);
    }
    let study = AutoPipeline::new(x, y).trials(14).seed(4).fit().unwrap();
    let best = study.best_trial().unwrap().unwrap();
    assert!(best.final_value("accuracy").unwrap() > 80.0);
}

/// Multimodal: an incompatible fusion configuration is rejected *before*
/// scheduling, and a compatible one fuses (§15/§28).
#[test]
fn multimodal_validates_before_scheduling() {
    let bad = MultimodalPlan {
        encoders: vec![
            automl_core::fusion::ModalityChoice {
                name: "img".into(),
                encoder: "cnn".into(),
                dim: 8,
            },
            automl_core::fusion::ModalityChoice {
                name: "txt".into(),
                encoder: "rnn".into(),
                dim: 4,
            },
        ],
        fusion: Fusion::Mean,
    };
    assert!(
        bad.validate().is_err(),
        "mismatched dims must be rejected up front"
    );

    let fused = fuse(&[vec![1.0, 2.0], vec![3.0, 4.0]], Fusion::Concat).unwrap();
    assert_eq!(fused, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Distributed: a coordinator enqueues trials and several workers drain the queue
/// exactly once, all completing — the §18.2 protocol end to end.
#[test]
fn distributed_workers_drain_exactly_once() {
    let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
    let study = storage
        .create_study(StudyMeta {
            name: "e2e-dist".into(),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "random".into(),
            pruner_name: "none".into(),
        })
        .unwrap();
    let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
    let mut sampler = RandomSampler::new(5);
    enqueue_pending(&storage, study, &space, &mut sampler, 24, 5).unwrap();

    let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
        Ok(NamedMetrics::single("loss", (p.float("x")? - 1.0).powi(2)))
    };
    let total: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|w| {
                let storage = storage.clone();
                scope.spawn(move || {
                    Worker::new(format!("w{w}"), 60_000)
                        .run_until_idle(&storage, study, &objective)
                        .unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(total, 24);
    assert_eq!(storage.load_history(study).unwrap().completed().count(), 24);
}

/// Reproducibility: two studies with the same seed and history produce
/// bit-identical trial parameters (§28 "winning trials are reproducible").
#[test]
fn replay_is_deterministic() {
    fn run() -> Vec<f64> {
        let space = SearchSpace::new()
            .add("x", Distribution::float(-5.0, 5.0))
            .add("y", Distribution::log_float(1e-3, 1.0));
        let mut study = Study::builder(space)
            .name("e2e-determinism")
            .minimize("loss")
            .sampler(TpeSampler::new("loss", Direction::Minimize, 42))
            .seed(42)
            .build()
            .unwrap();
        study
            .optimize_n(
                &|p: &ParamSet, _s: &mut dyn ReportSink| {
                    Ok(NamedMetrics::single(
                        "loss",
                        p.float("x")?.powi(2) + p.float("y")?,
                    ))
                },
                20,
            )
            .unwrap();
        let mut xs: Vec<f64> = study
            .history()
            .unwrap()
            .records()
            .iter()
            .map(|r| r.params.float("x").unwrap())
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs
    }
    assert_eq!(run(), run(), "same seed must replay bit-identically");
}
