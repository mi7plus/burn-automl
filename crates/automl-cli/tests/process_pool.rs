//! Process-executor tier integration test (PRD §18): the `ProcessPool` spawns
//! the real `automl worker` binary as crash-isolated subprocesses and drains a
//! persisted study, recovering orphans when workers crash.

use automl_core::distribution::Distribution;
use automl_core::distributed::enqueue_pending;
use automl_core::metrics::Direction;
use automl_core::process::ProcessPool;
use automl_core::sampler::RandomSampler;
use automl_core::space::SearchSpace;
use automl_core::sqlite::SqliteStorage;
use automl_core::storage::{Storage, StudyMeta};
use automl_core::trial::StudyId;
use std::sync::Arc;

const WORKER_BIN: &str = env!("CARGO_BIN_EXE_automl");

fn setup(tag: &str, n: usize) -> (std::path::PathBuf, Arc<dyn Storage>, StudyId) {
    let db = std::env::temp_dir().join(format!(
        "burn_automl_procpool_{tag}_{}.db",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&db);
    let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open(&db).unwrap());
    let study = storage
        .create_study(StudyMeta {
            name: format!("procpool-{tag}"),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "random".into(),
            pruner_name: "none".into(),
        })
        .unwrap();
    let space = SearchSpace::new()
        .add("x", Distribution::float(-5.0, 5.0))
        .add("y", Distribution::float(-5.0, 5.0));
    let mut sampler = RandomSampler::new(1);
    enqueue_pending(&storage, study, &space, &mut sampler, n, 1).unwrap();
    (db, storage, study)
}

#[test]
fn process_pool_drains_a_study_with_subprocess_workers() {
    let (db, storage, study) = setup("healthy", 8);
    let report = ProcessPool::new(WORKER_BIN)
        .arg("worker")
        .arg(db.to_string_lossy().to_string())
        .arg(study.0.to_string())
        .arg("30000") // lease ttl ms
        .workers(2)
        .lease_ttl_ms(30_000)
        .max_restarts(4)
        .run(&storage, study)
        .unwrap();

    assert!(report.drained, "the queue should drain: {report:?}");
    assert_eq!(storage.load_history(study).unwrap().completed().count(), 8);
    let _ = std::fs::remove_file(&db);
}

#[test]
fn process_pool_recovers_from_worker_crashes() {
    // Each worker aborts after completing 2 trials, so no single round finishes
    // the 8-trial queue; the pool must recover orphaned trials and restart until
    // everything completes — crash isolation end to end.
    let (db, storage, study) = setup("crashy", 8);
    let report = ProcessPool::new(WORKER_BIN)
        .arg("worker")
        .arg(db.to_string_lossy().to_string())
        .arg(study.0.to_string())
        .arg("1000")
        .env("AUTOML_WORKER_CRASH_AFTER", "2")
        .workers(2)
        .lease_ttl_ms(1000)
        .max_restarts(12)
        .run(&storage, study)
        .unwrap();

    assert!(report.crashed > 0, "workers were supposed to crash: {report:?}");
    assert!(report.drained, "recovery should still drain the queue: {report:?}");
    assert_eq!(storage.load_history(study).unwrap().completed().count(), 8);
    let _ = std::fs::remove_file(&db);
}
