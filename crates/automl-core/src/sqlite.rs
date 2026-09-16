//! SQLite storage backend with versioned migrations and study resume
//! (PRD §19.1, §29 item 7).
//!
//! This is the first *persistent* [`Storage`](crate::storage::Storage) backend: it turns experiments
//! into durable studies that survive process exit. Per §19.1 the schema is
//! versioned from this first persistent release, using a `_schema_version`
//! table with monotonically increasing integer versions applied at open time;
//! no destructive migration ships without a documented downgrade path.
//!
//! Reporting stays idempotent by `(trial, step)` (§18.2) via `INSERT OR
//! REPLACE` into a `reports` table keyed on that pair.
//!
//! Gated behind the `sqlite` feature so plain in-memory HPO users don't compile
//! the bundled C library. A future `automl-storage` crate can lift this module
//! wholesale once a second extraction trigger (§21.1) applies.

use crate::error::{Error, Result};
use crate::metrics::{Direction, NamedMetrics};
use crate::param::ParamSet;
use crate::storage::{Storage, StudyMeta};
use crate::trial::{IntermediateReport, StudyId, TrialHistory, TrialId, TrialRecord, TrialState};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;

/// Ordered schema migrations. The index (1-based) is the version each statement
/// upgrades *to*; append new migrations, never edit shipped ones.
const MIGRATIONS: &[&str] = &[
    // v1: initial schema.
    r#"
    CREATE TABLE studies (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        directions   TEXT NOT NULL,   -- JSON: [[name, "Minimize"|"Maximize"], ...]
        sampler_name TEXT NOT NULL,
        pruner_name  TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );
    CREATE TABLE trials (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id      INTEGER NOT NULL REFERENCES studies(id),
        params        TEXT NOT NULL,  -- JSON ParamSet
        state         TEXT NOT NULL,  -- JSON TrialState
        final_metrics TEXT,           -- JSON NamedMetrics or NULL
        seed          INTEGER NOT NULL
    );
    CREATE INDEX idx_trials_study ON trials(study_id);
    CREATE TABLE reports (
        trial_id INTEGER NOT NULL REFERENCES trials(id),
        step     INTEGER NOT NULL,
        metrics  TEXT NOT NULL,       -- JSON NamedMetrics
        PRIMARY KEY (trial_id, step)
    );
    "#,
    // v2: per-trial provenance and timing (§19). Added as nullable columns so a
    // v1 database upgrades in place; older rows read back with default env and
    // empty timing.
    r#"
    ALTER TABLE trials ADD COLUMN env          TEXT;    -- JSON EnvSnapshot
    ALTER TABLE trials ADD COLUMN queued_at    INTEGER; -- unix millis
    ALTER TABLE trials ADD COLUMN started_at   INTEGER; -- unix millis or NULL
    ALTER TABLE trials ADD COLUMN completed_at INTEGER; -- unix millis or NULL
    "#,
    // v3: per-trial binary artifacts (§19 "Artifacts": checkpoints, exported
    // models, configs, logs).
    r#"
    CREATE TABLE artifacts (
        trial_id INTEGER NOT NULL REFERENCES trials(id),
        name     TEXT NOT NULL,
        bytes    BLOB NOT NULL,
        PRIMARY KEY (trial_id, name)
    );
    "#,
    // v4: distributed lease columns (§18.2). The lease owner and expiry back the
    // claim/renew/orphan-recovery compare-and-swaps.
    r#"
    ALTER TABLE trials ADD COLUMN lease_owner  TEXT;    -- worker id or NULL
    ALTER TABLE trials ADD COLUMN lease_expiry INTEGER; -- unix millis or NULL
    "#,
];

/// A persistent [`Storage`] backend over a SQLite database file.
///
/// The [`Connection`] is wrapped in a [`Mutex`] so the backend is `Send + Sync`
/// as the trait requires; SQLite serializes writers anyway.
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Open (creating if absent) a database at `path`, applying any pending
    /// migrations. Reopening the same path resumes all persisted studies.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).map_err(sql)?;
        Self::from_connection(conn)
    }

    /// An in-memory database (does not persist across connections). Useful for
    /// tests of the schema and query paths without touching the filesystem.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sql)?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(sql)?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(sql)?;
        Self::migrate(&conn)?;
        Ok(SqliteStorage {
            conn: Mutex::new(conn),
        })
    }

    /// Apply migrations whose version exceeds the recorded schema version.
    fn migrate(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER NOT NULL);",
        )
        .map_err(sql)?;
        let current: i64 = conn
            .query_row("SELECT MAX(version) FROM _schema_version", [], |r| r.get(0))
            .optional()
            .map_err(sql)?
            .flatten()
            .unwrap_or(0);

        for (i, stmt) in MIGRATIONS.iter().enumerate() {
            let version = (i + 1) as i64;
            if version > current {
                conn.execute_batch(stmt).map_err(sql)?;
                conn.execute(
                    "INSERT INTO _schema_version (version) VALUES (?1)",
                    params![version],
                )
                .map_err(sql)?;
            }
        }
        Ok(())
    }

    /// The current schema version recorded in the database.
    pub fn schema_version(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT MAX(version) FROM _schema_version", [], |r| r.get(0))
            .optional()
            .map_err(sql)?
            .flatten()
            .map(Ok)
            .unwrap_or(Ok(0))
    }

    /// Enumerate the ids of all persisted studies, for resume/inspection.
    pub fn study_ids(&self) -> Result<Vec<StudyId>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id FROM studies ORDER BY id")
            .map_err(sql)?;
        let ids = stmt
            .query_map([], |r| r.get::<_, i64>(0))
            .map_err(sql)?
            .map(|r| r.map(|v| StudyId(v as u64)).map_err(sql))
            .collect::<Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// Read one trial record plus its intermediate reports from an open lock.
    fn read_trial(conn: &Connection, trial: TrialId) -> Result<TrialRecord> {
        #[allow(clippy::type_complexity)]
        let (study_id, params_json, state_json, final_json, seed, env_json, queued, started, completed): (
            i64,
            String,
            String,
            Option<String>,
            i64,
            Option<String>,
            Option<i64>,
            Option<i64>,
            Option<i64>,
        ) = conn
            .query_row(
                "SELECT study_id, params, state, final_metrics, seed, env, queued_at, started_at, completed_at \
                 FROM trials WHERE id = ?1",
                params![trial.0 as i64],
                |r| {
                    Ok((
                        r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?,
                        r.get(7)?, r.get(8)?,
                    ))
                },
            )
            .optional()
            .map_err(sql)?
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })?;

        let params: ParamSet = serde_json::from_str(&params_json)?;
        let state: TrialState = serde_json::from_str(&state_json)?;
        let final_metrics: Option<NamedMetrics> =
            final_json.map(|s| serde_json::from_str(&s)).transpose()?;
        // Provenance columns are absent (NULL) on rows written under v1.
        let env: crate::provenance::EnvSnapshot = match env_json {
            Some(s) => serde_json::from_str(&s)?,
            None => crate::provenance::EnvSnapshot::default(),
        };
        let timing = crate::provenance::TrialTiming {
            queued_at_ms: queued.unwrap_or(0) as u64,
            started_at_ms: started.map(|v| v as u64),
            completed_at_ms: completed.map(|v| v as u64),
        };

        let mut stmt = conn
            .prepare("SELECT step, metrics FROM reports WHERE trial_id = ?1 ORDER BY step")
            .map_err(sql)?;
        let intermediate = stmt
            .query_map(params![trial.0 as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(sql)?
            .map(|row| {
                let (step, metrics_json) = row.map_err(sql)?;
                let metrics: NamedMetrics = serde_json::from_str(&metrics_json)?;
                Ok(IntermediateReport {
                    step: step as u64,
                    metrics,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(TrialRecord {
            id: trial,
            study_id: StudyId(study_id as u64),
            params,
            state,
            intermediate,
            final_metrics,
            seed: seed as u64,
            env,
            timing,
        })
    }
}

impl Storage for SqliteStorage {
    fn create_study(&self, meta: StudyMeta) -> Result<StudyId> {
        let conn = self.conn.lock().unwrap();
        let directions = serde_json::to_string(&meta.directions)?;
        let created_at = now_rfc3339();
        conn.execute(
            "INSERT INTO studies (name, directions, sampler_name, pruner_name, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                meta.name,
                directions,
                meta.sampler_name,
                meta.pruner_name,
                created_at
            ],
        )
        .map_err(sql)?;
        Ok(StudyId(conn.last_insert_rowid() as u64))
    }

    fn enqueue_trial(&self, study: StudyId, params: ParamSet, seed: u64) -> Result<TrialId> {
        let conn = self.conn.lock().unwrap();
        let params_json = serde_json::to_string(&params)?;
        let state_json = serde_json::to_string(&TrialState::Waiting)?;
        let env_json = serde_json::to_string(&crate::provenance::EnvSnapshot::capture())?;
        let queued = crate::provenance::now_ms() as i64;
        conn.execute(
            "INSERT INTO trials (study_id, params, state, final_metrics, seed, env, queued_at)
             VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6)",
            params![
                study.0 as i64,
                params_json,
                state_json,
                seed as i64,
                env_json,
                queued
            ],
        )
        .map_err(|e| match e {
            rusqlite::Error::SqliteFailure(_, _) => Error::NotFound {
                kind: "study",
                id: study.to_string(),
            },
            other => sql(other),
        })?;
        Ok(TrialId(conn.last_insert_rowid() as u64))
    }

    fn start_trial(&self, trial: TrialId) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let affected = conn
            .execute(
                "UPDATE trials SET state = ?1, started_at = ?2 WHERE id = ?3",
                params![
                    serde_json::to_string(&TrialState::Running)?,
                    crate::provenance::now_ms() as i64,
                    trial.0 as i64
                ],
            )
            .map_err(sql)?;
        if affected == 0 {
            return Err(Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            });
        }
        Ok(())
    }

    fn report(&self, trial: TrialId, step: u64, metrics: NamedMetrics) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let metrics_json = serde_json::to_string(&metrics)?;
        // Idempotent by (trial, step): a retried report overwrites in place.
        conn.execute(
            "INSERT OR REPLACE INTO reports (trial_id, step, metrics) VALUES (?1, ?2, ?3)",
            params![trial.0 as i64, step as i64, metrics_json],
        )
        .map_err(sql)?;
        Ok(())
    }

    fn complete(
        &self,
        trial: TrialId,
        state: TrialState,
        final_metrics: Option<NamedMetrics>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let state_json = serde_json::to_string(&state)?;
        let metrics_json = final_metrics
            .map(|m| serde_json::to_string(&m))
            .transpose()?;
        let affected = conn
            .execute(
                "UPDATE trials SET state = ?1, final_metrics = ?2, completed_at = ?3 WHERE id = ?4",
                params![
                    state_json,
                    metrics_json,
                    crate::provenance::now_ms() as i64,
                    trial.0 as i64
                ],
            )
            .map_err(sql)?;
        if affected == 0 {
            return Err(Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            });
        }
        Ok(())
    }

    fn load_trial(&self, trial: TrialId) -> Result<TrialRecord> {
        let conn = self.conn.lock().unwrap();
        Self::read_trial(&conn, trial)
    }

    fn load_history(&self, study: StudyId) -> Result<TrialHistory> {
        let conn = self.conn.lock().unwrap();
        // Confirm the study exists so a missing study is distinguishable from
        // one that simply has no trials yet.
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM studies WHERE id = ?1",
                params![study.0 as i64],
                |_| Ok(true),
            )
            .optional()
            .map_err(sql)?
            .unwrap_or(false);
        if !exists {
            return Err(Error::NotFound {
                kind: "study",
                id: study.to_string(),
            });
        }
        let mut stmt = conn
            .prepare("SELECT id FROM trials WHERE study_id = ?1 ORDER BY id")
            .map_err(sql)?;
        let ids = stmt
            .query_map(params![study.0 as i64], |r| r.get::<_, i64>(0))
            .map_err(sql)?
            .map(|r| r.map_err(sql))
            .collect::<Result<Vec<_>>>()?;
        let records = ids
            .into_iter()
            .map(|id| Self::read_trial(&conn, TrialId(id as u64)))
            .collect::<Result<Vec<_>>>()?;
        Ok(TrialHistory::new(records))
    }

    fn load_meta(&self, study: StudyId) -> Result<StudyMeta> {
        let conn = self.conn.lock().unwrap();
        let (name, directions_json, sampler_name, pruner_name): (String, String, String, String) =
            conn.query_row(
                "SELECT name, directions, sampler_name, pruner_name FROM studies WHERE id = ?1",
                params![study.0 as i64],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
            .map_err(sql)?
            .ok_or_else(|| Error::NotFound {
                kind: "study",
                id: study.to_string(),
            })?;
        let directions: Vec<(String, Direction)> = serde_json::from_str(&directions_json)?;
        Ok(StudyMeta {
            name,
            directions,
            sampler_name,
            pruner_name,
        })
    }

    fn save_artifact(&self, trial: TrialId, name: &str, bytes: &[u8]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO artifacts (trial_id, name, bytes) VALUES (?1, ?2, ?3)",
            params![trial.0 as i64, name, bytes],
        )
        .map_err(sql)?;
        Ok(())
    }

    fn load_artifact(&self, trial: TrialId, name: &str) -> Result<Option<Vec<u8>>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT bytes FROM artifacts WHERE trial_id = ?1 AND name = ?2",
            params![trial.0 as i64, name],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .optional()
        .map_err(sql)
    }

    fn list_artifacts(&self, trial: TrialId) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT name FROM artifacts WHERE trial_id = ?1 ORDER BY name")
            .map_err(sql)?;
        let names = stmt
            .query_map(params![trial.0 as i64], |r| r.get::<_, String>(0))
            .map_err(sql)?
            .map(|r| r.map_err(sql))
            .collect::<Result<Vec<_>>>()?;
        Ok(names)
    }

    fn claim_trial(
        &self,
        study: StudyId,
        owner: &str,
        now_ms: u64,
        lease_expiry_ms: u64,
    ) -> Result<Option<TrialId>> {
        let conn = self.conn.lock().unwrap();
        let waiting = serde_json::to_string(&TrialState::Waiting)?;
        let running = serde_json::to_string(&TrialState::Running)?;
        // The mutex serializes access, so SELECT-then-UPDATE is an atomic CAS:
        // a Waiting trial, or a Running one whose lease has expired (orphan).
        let candidate: Option<i64> = conn
            .query_row(
                "SELECT id FROM trials \
                 WHERE study_id = ?1 AND (state = ?2 \
                    OR (state = ?3 AND (lease_expiry IS NULL OR lease_expiry <= ?4))) \
                 ORDER BY id LIMIT 1",
                params![study.0 as i64, waiting, running, now_ms as i64],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        match candidate {
            Some(id) => {
                conn.execute(
                    "UPDATE trials SET state = ?1, lease_owner = ?2, lease_expiry = ?3, \
                     started_at = COALESCE(started_at, ?4) WHERE id = ?5",
                    params![running, owner, lease_expiry_ms as i64, now_ms as i64, id],
                )
                .map_err(sql)?;
                Ok(Some(TrialId(id as u64)))
            }
            None => Ok(None),
        }
    }

    fn renew_lease(&self, trial: TrialId, owner: &str, new_expiry_ms: u64) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let affected = conn
            .execute(
                "UPDATE trials SET lease_expiry = ?1 WHERE id = ?2 AND lease_owner = ?3",
                params![new_expiry_ms as i64, trial.0 as i64, owner],
            )
            .map_err(sql)?;
        Ok(affected > 0)
    }

    fn recover_orphans(&self, study: StudyId, now_ms: u64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let waiting = serde_json::to_string(&TrialState::Waiting)?;
        let running = serde_json::to_string(&TrialState::Running)?;
        let affected = conn
            .execute(
                "UPDATE trials SET state = ?1, lease_owner = NULL \
                 WHERE study_id = ?2 AND state = ?3 \
                   AND lease_expiry IS NOT NULL AND lease_expiry <= ?4",
                params![waiting, study.0 as i64, running, now_ms as i64],
            )
            .map_err(sql)?;
        Ok(affected)
    }
}

/// Map a rusqlite error into our storage error.
fn sql(e: rusqlite::Error) -> Error {
    Error::Storage(e.to_string())
}

/// A minimal RFC3339-ish timestamp without pulling in a time crate: seconds
/// since the Unix epoch, which is monotone and sortable for provenance.
fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::ParamValue;

    fn meta() -> StudyMeta {
        StudyMeta {
            name: "persisted".into(),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "tpe".into(),
            pruner_name: "median".into(),
        }
    }

    fn temp_db_path(tag: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "automl_sqlite_{tag}_{}_{nanos}.db",
            std::process::id()
        ))
    }

    #[test]
    fn migrations_set_version() {
        let s = SqliteStorage::open_in_memory().unwrap();
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);
    }

    #[test]
    fn lifecycle_roundtrip_in_memory() {
        let s = SqliteStorage::open_in_memory().unwrap();
        let study = s.create_study(meta()).unwrap();
        let p = ParamSet::new().with("x", ParamValue::Float(1.5));
        let t = s.enqueue_trial(study, p, 7).unwrap();
        s.start_trial(t).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.9)).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.4)).unwrap(); // idempotent
        s.complete(
            t,
            TrialState::Complete,
            Some(NamedMetrics::single("loss", 0.3)),
        )
        .unwrap();

        let rec = s.load_trial(t).unwrap();
        assert_eq!(rec.state, TrialState::Complete);
        assert_eq!(rec.intermediate.len(), 1);
        assert_eq!(rec.intermediate[0].metrics.get("loss"), Some(0.4));
        assert_eq!(rec.final_value("loss"), Some(0.3));
        assert_eq!(rec.params.float("x").unwrap(), 1.5);
    }

    #[test]
    fn resume_after_reopen() {
        let path = temp_db_path("resume");
        let study;
        {
            let s = SqliteStorage::open(&path).unwrap();
            study = s.create_study(meta()).unwrap();
            let t = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();
            s.complete(
                t,
                TrialState::Complete,
                Some(NamedMetrics::single("loss", 0.25)),
            )
            .unwrap();
            // `s` dropped here — simulates process exit.
        }
        let reopened = SqliteStorage::open(&path).unwrap();
        assert_eq!(reopened.study_ids().unwrap(), vec![study]);
        let history = reopened.load_history(study).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history.completed().count(), 1);
        assert_eq!(reopened.load_meta(study).unwrap().name, "persisted");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_study_errors() {
        let s = SqliteStorage::open_in_memory().unwrap();
        assert!(matches!(
            s.load_history(StudyId(123)),
            Err(Error::NotFound { kind: "study", .. })
        ));
    }

    #[test]
    fn artifacts_persist_and_survive_reopen() {
        let path = temp_db_path("artifacts");
        let (study, trial);
        {
            let s = SqliteStorage::open(&path).unwrap();
            study = s.create_study(meta()).unwrap();
            trial = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();
            s.save_artifact(trial, "ckpt", &[0u8, 1, 2, 3, 255])
                .unwrap();
        }
        // Reopen: v3 migration applied, artifact still present.
        let s = SqliteStorage::open(&path).unwrap();
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);
        assert_eq!(
            s.load_artifact(trial, "ckpt").unwrap(),
            Some(vec![0, 1, 2, 3, 255])
        );
        assert_eq!(s.list_artifacts(trial).unwrap(), vec!["ckpt".to_string()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lease_claim_renew_and_orphan_recovery() {
        let s = SqliteStorage::open_in_memory().unwrap();
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);
        let study = s.create_study(meta()).unwrap();
        let t = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();

        // Worker "a" claims the waiting trial (lease to t=100).
        assert_eq!(s.claim_trial(study, "a", 0, 100).unwrap(), Some(t));
        // Nothing else claimable while the lease holds.
        assert_eq!(s.claim_trial(study, "b", 50, 150).unwrap(), None);
        // "a" can renew, a stranger cannot.
        assert!(s.renew_lease(t, "a", 200).unwrap());
        assert!(!s.renew_lease(t, "b", 200).unwrap());

        // Once the (renewed) lease expires, the orphan is reclaimable.
        assert_eq!(s.claim_trial(study, "b", 250, 350).unwrap(), Some(t));
        // And an explicit sweep would requeue an expired one.
        assert_eq!(s.recover_orphans(study, 400).unwrap(), 1);
        assert_eq!(s.load_trial(t).unwrap().state, TrialState::Waiting);
    }

    #[test]
    fn provenance_and_timing_roundtrip() {
        let s = SqliteStorage::open_in_memory().unwrap();
        // The v2 provenance migration is applied.
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);

        let study = s.create_study(meta()).unwrap();
        let t = s.enqueue_trial(study, ParamSet::new(), 1).unwrap();
        s.start_trial(t).unwrap();
        s.complete(
            t,
            TrialState::Complete,
            Some(NamedMetrics::single("loss", 0.1)),
        )
        .unwrap();

        let rec = s.load_trial(t).unwrap();
        assert!(!rec.env.os.is_empty());
        assert!(!rec.env.core_version.is_empty());
        assert!(rec.timing.queued_at_ms > 0);
        assert!(rec.timing.started_at_ms.is_some());
        assert!(rec.timing.completed_at_ms.is_some());
    }

    #[test]
    fn study_resumes_across_reopen_without_losing_trials() {
        use crate::distribution::Distribution;
        use crate::objective::ReportSink;
        use crate::pruner::NoPruner;
        use crate::sampler::RandomSampler;
        use crate::space::SearchSpace;
        use crate::study::Study;
        use std::sync::Arc;

        let path = temp_db_path("resume_study");
        let space = || SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
            let x = p.float("x")?;
            Ok(NamedMetrics::single("loss", (x - 2.0).powi(2)))
        };

        // First run: 30 trials, then "process exit".
        let study_id = {
            let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open(&path).unwrap());
            let mut study = Study::builder(space())
                .minimize("loss")
                .sampler(RandomSampler::new(1))
                .storage(storage)
                .seed(1)
                .build()
                .unwrap();
            study.optimize_n(&objective, 30).unwrap();
            study.id()
        };

        // Reopen the database and resume the same study for 30 more trials.
        let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open(&path).unwrap());
        let mut resumed = Study::resume(
            space(),
            storage,
            study_id,
            RandomSampler::new(2),
            Arc::new(NoPruner),
            2,
        )
        .unwrap();
        assert_eq!(
            resumed.history().unwrap().len(),
            30,
            "prior trials must survive"
        );
        resumed.optimize_n(&objective, 30).unwrap();
        assert_eq!(
            resumed.history().unwrap().len(),
            60,
            "resumed run appends, not replaces"
        );
        assert!(resumed.best_trial().unwrap().is_some());

        let _ = std::fs::remove_file(&path);
    }
}
