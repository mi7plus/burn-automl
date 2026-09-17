//! Distributed execution: a coordinator enqueues trials into a shared in-memory
//! store and several `Worker`s drain the queue concurrently. Each worker claims a
//! trial by compare-and-swap, heartbeats its lease, and completes it only while
//! it still holds the lease — so the queue drains exactly once.
//!
//! ```bash
//! cargo run -p automl-core --release --example distributed
//! ```

use automl_core::prelude::*;
use std::sync::Arc;

fn main() {
    let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
    let study = storage
        .create_study(StudyMeta {
            name: "distributed-demo".into(),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "random".into(),
            pruner_name: "none".into(),
        })
        .unwrap();

    let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
    let mut sampler = RandomSampler::new(1);
    enqueue_pending(&storage, study, &space, &mut sampler, 24, 1).unwrap();

    let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
        Ok(NamedMetrics::single("loss", (p.float("x")? - 2.0).powi(2)))
    };

    // Four workers race to drain the shared queue.
    let ran: usize = std::thread::scope(|scope| {
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

    let completed = storage.load_history(study).unwrap().completed().count();
    println!("4 workers ran {ran} trials; {completed} completed (each exactly once)");
    let best = Study::resume(
        space,
        storage,
        study,
        RandomSampler::new(1),
        std::sync::Arc::new(NoPruner),
        1,
    )
    .unwrap()
    .best_trial()
    .unwrap()
    .unwrap();
    println!(
        "best loss {:.4} at x={:.3}",
        best.final_value("loss").unwrap_or(f64::NAN),
        best.params.float("x").unwrap_or(f64::NAN)
    );
}
