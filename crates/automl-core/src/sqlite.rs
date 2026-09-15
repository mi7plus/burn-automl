//! SQLite storage backend with versioned migrations and study resume
//! (PRD §19.1, §29 item 7).
//!
//! This is the first *persistent* [`Storage`] backend: it turns experiments
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
        let (study_id, params_json, state_json, final_json, seed): (
            i64,
            String,
            String,
            Option<String>,
            i64,
        ) = conn
            .query_row(
                "SELECT study_id, params, state, final_metrics, seed FROM trials WHERE id = ?1",
                params![trial.0 as i64],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
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
        conn.execute(
            "INSERT INTO trials (study_id, params, state, final_metrics, seed)
             VALUES (?1, ?2, ?3, NULL, ?4)",
            params![study.0 as i64, params_json, state_json, seed as i64],
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
        self.set_state(trial, TrialState::Running, None)
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
        self.set_state(trial, state, Some(final_metrics))
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
}

impl SqliteStorage {
    /// Shared UPDATE for state transitions; `final_metrics = Some(v)` also
    /// writes the (possibly NULL) final metrics column.
    fn set_state(
        &self,
        trial: TrialId,
        state: TrialState,
        final_metrics: Option<Option<NamedMetrics>>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let state_json = serde_json::to_string(&state)?;
        let affected = match final_metrics {
            Some(metrics) => {
                let metrics_json = metrics.map(|m| serde_json::to_string(&m)).transpose()?;
                conn.execute(
                    "UPDATE trials SET state = ?1, final_metrics = ?2 WHERE id = ?3",
                    params![state_json, metrics_json, trial.0 as i64],
                )
                .map_err(sql)?
            }
            None => conn
                .execute(
                    "UPDATE trials SET state = ?1 WHERE id = ?2",
                    params![state_json, trial.0 as i64],
                )
                .map_err(sql)?,
        };
        if affected == 0 {
            return Err(Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            });
        }
        Ok(())
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
