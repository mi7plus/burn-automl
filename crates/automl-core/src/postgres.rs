//! PostgreSQL storage backend (PRD §18.4).
//!
//! The distributed lease protocol (§18.2) is backend-agnostic, so a shared
//! Postgres store is a mechanical translation of the SQLite backend behind the
//! same [`Storage`] trait: the same versioned migrations, the same idempotent
//! `(trial, step)` reporting, and the same claim / renew / recover compare-and-
//! swaps — now over a server many machines can share, rather than a local file.
//!
//! Gated behind the `postgres` feature. The client is pure Rust (no C library),
//! so it compiles anywhere; integration tests require a live server (set
//! `TEST_POSTGRES_URL`) and are skipped otherwise.
//!
//! ## Concurrency
//!
//! The synchronous `postgres` client is not `Sync`, so it is wrapped in a
//! `Mutex<Client>`: all database calls from one `PostgresStorage` handle
//! serialize through that one connection. This is correct — the lease
//! compare-and-swaps still race safely *across processes* on the server via
//! `FOR UPDATE SKIP LOCKED` — but it caps *in-process* parallelism. For many
//! concurrent workers in a single process, give each its own `PostgresStorage`
//! (its own connection), or put a connection pool behind the `Storage` trait.
//! Across machines, each process has its own handle, so this is a non-issue.

use crate::error::{Error, Result};
use crate::metrics::{Direction, NamedMetrics};
use crate::param::ParamSet;
use crate::storage::{Storage, StudyMeta};
use crate::trial::{IntermediateReport, StudyId, TrialHistory, TrialId, TrialRecord, TrialState};
use postgres::{Client, NoTls};
use std::sync::Mutex;

/// Ordered schema migrations — the Postgres translation of the SQLite schema.
/// The 1-based index is the version each statement upgrades *to*; append new
/// migrations, never edit shipped ones.
const MIGRATIONS: &[&str] = &[
    // v1: initial schema.
    r#"
    CREATE TABLE studies (
        id           BIGSERIAL PRIMARY KEY,
        name         TEXT NOT NULL,
        directions   TEXT NOT NULL,
        sampler_name TEXT NOT NULL,
        pruner_name  TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );
    CREATE TABLE trials (
        id            BIGSERIAL PRIMARY KEY,
        study_id      BIGINT NOT NULL REFERENCES studies(id),
        params        TEXT NOT NULL,
        state         TEXT NOT NULL,
        final_metrics TEXT,
        seed          BIGINT NOT NULL
    );
    CREATE INDEX idx_trials_study ON trials(study_id);
    CREATE TABLE reports (
        trial_id BIGINT NOT NULL REFERENCES trials(id),
        step     BIGINT NOT NULL,
        metrics  TEXT NOT NULL,
        PRIMARY KEY (trial_id, step)
    );
    "#,
    // v2: per-trial provenance and timing (§19), nullable so v1 upgrades in place.
    r#"
    ALTER TABLE trials ADD COLUMN env          TEXT;
    ALTER TABLE trials ADD COLUMN queued_at    BIGINT;
    ALTER TABLE trials ADD COLUMN started_at   BIGINT;
    ALTER TABLE trials ADD COLUMN completed_at BIGINT;
    "#,
    // v3: per-trial binary artifacts (§19 "Artifacts").
    r#"
    CREATE TABLE artifacts (
        trial_id BIGINT NOT NULL REFERENCES trials(id),
        name     TEXT NOT NULL,
        bytes    BYTEA NOT NULL,
        PRIMARY KEY (trial_id, name)
    );
    "#,
    // v4: distributed lease columns (§18.2).
    r#"
    ALTER TABLE trials ADD COLUMN lease_owner  TEXT;
    ALTER TABLE trials ADD COLUMN lease_expiry BIGINT;
    "#,
];

/// A persistent, shareable [`Storage`] backend over a PostgreSQL server.
///
/// The [`Client`] is wrapped in a [`Mutex`] so the backend is `Send + Sync`; the
/// server itself serializes the compare-and-swaps that back the lease protocol.
pub struct PostgresStorage {
    client: Mutex<Client>,
}

impl PostgresStorage {
    /// Connect to `url` (e.g. `host=localhost user=postgres dbname=automl`),
    /// applying any pending migrations. Reconnecting resumes persisted studies.
    pub fn connect(url: &str) -> Result<Self> {
        let mut client = Client::connect(url, NoTls).map_err(pg)?;
        Self::migrate(&mut client)?;
        Ok(PostgresStorage {
            client: Mutex::new(client),
        })
    }

    /// Apply migrations whose version exceeds the recorded schema version.
    fn migrate(client: &mut Client) -> Result<()> {
        client
            .batch_execute("CREATE TABLE IF NOT EXISTS _schema_version (version BIGINT NOT NULL);")
            .map_err(pg)?;
        let current: i64 = client
            .query_one("SELECT COALESCE(MAX(version), 0) FROM _schema_version", &[])
            .map_err(pg)?
            .get(0);

        for (i, stmt) in MIGRATIONS.iter().enumerate() {
            let version = (i + 1) as i64;
            if version > current {
                client.batch_execute(stmt).map_err(pg)?;
                client
                    .execute(
                        "INSERT INTO _schema_version (version) VALUES ($1)",
                        &[&version],
                    )
                    .map_err(pg)?;
            }
        }
        Ok(())
    }

    /// The current schema version recorded in the database.
    pub fn schema_version(&self) -> Result<i64> {
        let mut client = self.client.lock().unwrap();
        Ok(client
            .query_one("SELECT COALESCE(MAX(version), 0) FROM _schema_version", &[])
            .map_err(pg)?
            .get(0))
    }

    /// Enumerate the ids of all persisted studies, for resume/inspection.
    pub fn study_ids(&self) -> Result<Vec<StudyId>> {
        let mut client = self.client.lock().unwrap();
        let rows = client
            .query("SELECT id FROM studies ORDER BY id", &[])
            .map_err(pg)?;
        Ok(rows
            .iter()
            .map(|r| StudyId(r.get::<_, i64>(0) as u64))
            .collect())
    }

    /// Read one trial record plus its intermediate reports.
    fn read_trial(client: &mut Client, trial: TrialId) -> Result<TrialRecord> {
        let row = client
            .query_opt(
                "SELECT study_id, params, state, final_metrics, seed, env, queued_at, started_at, completed_at \
                 FROM trials WHERE id = $1",
                &[&(trial.0 as i64)],
            )
            .map_err(pg)?
            .ok_or_else(|| Error::NotFound {
                kind: "trial",
                id: trial.to_string(),
            })?;

        let params: ParamSet = serde_json::from_str(&row.get::<_, String>(1))?;
        let state: TrialState = serde_json::from_str(&row.get::<_, String>(2))?;
        let final_metrics: Option<NamedMetrics> = row
            .get::<_, Option<String>>(3)
            .map(|s| serde_json::from_str(&s))
            .transpose()?;
        let env: crate::provenance::EnvSnapshot = match row.get::<_, Option<String>>(5) {
            Some(s) => serde_json::from_str(&s)?,
            None => crate::provenance::EnvSnapshot::default(),
        };
        let timing = crate::provenance::TrialTiming {
            queued_at_ms: row.get::<_, Option<i64>>(6).unwrap_or(0) as u64,
            started_at_ms: row.get::<_, Option<i64>>(7).map(|v| v as u64),
            completed_at_ms: row.get::<_, Option<i64>>(8).map(|v| v as u64),
        };

        let report_rows = client
            .query(
                "SELECT step, metrics FROM reports WHERE trial_id = $1 ORDER BY step",
                &[&(trial.0 as i64)],
            )
            .map_err(pg)?;
        let intermediate = report_rows
            .iter()
            .map(|r| {
                let metrics: NamedMetrics = serde_json::from_str(&r.get::<_, String>(1))?;
                Ok(IntermediateReport {
                    step: r.get::<_, i64>(0) as u64,
                    metrics,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(TrialRecord {
            id: trial,
            study_id: StudyId(row.get::<_, i64>(0) as u64),
            params,
            state,
            intermediate,
            final_metrics,
            seed: row.get::<_, i64>(4) as u64,
            env,
            timing,
        })
    }
}

impl Storage for PostgresStorage {
    fn create_study(&self, meta: StudyMeta) -> Result<StudyId> {
        let mut client = self.client.lock().unwrap();
        let directions = serde_json::to_string(&meta.directions)?;
        let created_at = now_epoch_secs();
        let row = client
            .query_one(
                "INSERT INTO studies (name, directions, sampler_name, pruner_name, created_at) \
                 VALUES ($1, $2, $3, $4, $5) RETURNING id",
                &[
                    &meta.name,
                    &directions,
                    &meta.sampler_name,
                    &meta.pruner_name,
                    &created_at,
                ],
            )
            .map_err(pg)?;
        Ok(StudyId(row.get::<_, i64>(0) as u64))
    }

    fn enqueue_trial(&self, study: StudyId, params: ParamSet, seed: u64) -> Result<TrialId> {
        let mut client = self.client.lock().unwrap();
        let params_json = serde_json::to_string(&params)?;
        let state_json = serde_json::to_string(&TrialState::Waiting)?;
        let env_json = serde_json::to_string(&crate::provenance::EnvSnapshot::capture())?;
        let queued = crate::provenance::now_ms() as i64;
        let row = client
            .query_one(
                "INSERT INTO trials (study_id, params, state, final_metrics, seed, env, queued_at) \
                 VALUES ($1, $2, $3, NULL, $4, $5, $6) RETURNING id",
                &[
                    &(study.0 as i64),
                    &params_json,
                    &state_json,
                    &(seed as i64),
                    &env_json,
                    &queued,
                ],
            )
            .map_err(|_| Error::NotFound {
                kind: "study",
                id: study.to_string(),
            })?;
        Ok(TrialId(row.get::<_, i64>(0) as u64))
    }

    fn start_trial(&self, trial: TrialId) -> Result<()> {
        let mut client = self.client.lock().unwrap();
        let affected = client
            .execute(
                "UPDATE trials SET state = $1, started_at = $2 WHERE id = $3",
                &[
                    &serde_json::to_string(&TrialState::Running)?,
                    &(crate::provenance::now_ms() as i64),
                    &(trial.0 as i64),
                ],
            )
            .map_err(pg)?;
        not_found_if_zero(affected, trial)
    }

    fn report(&self, trial: TrialId, step: u64, metrics: NamedMetrics) -> Result<()> {
        let mut client = self.client.lock().unwrap();
        let metrics_json = serde_json::to_string(&metrics)?;
        // Idempotent by (trial, step).
        client
            .execute(
                "INSERT INTO reports (trial_id, step, metrics) VALUES ($1, $2, $3) \
                 ON CONFLICT (trial_id, step) DO UPDATE SET metrics = EXCLUDED.metrics",
                &[&(trial.0 as i64), &(step as i64), &metrics_json],
            )
            .map_err(pg)?;
        Ok(())
    }

    fn complete(
        &self,
        trial: TrialId,
        state: TrialState,
        final_metrics: Option<NamedMetrics>,
    ) -> Result<()> {
        let mut client = self.client.lock().unwrap();
        let state_json = serde_json::to_string(&state)?;
        let metrics_json = final_metrics
            .map(|m| serde_json::to_string(&m))
            .transpose()?;
        let affected = client
            .execute(
                "UPDATE trials SET state = $1, final_metrics = $2, completed_at = $3 WHERE id = $4",
                &[
                    &state_json,
                    &metrics_json,
                    &(crate::provenance::now_ms() as i64),
                    &(trial.0 as i64),
                ],
            )
            .map_err(pg)?;
        not_found_if_zero(affected, trial)
    }

    fn load_trial(&self, trial: TrialId) -> Result<TrialRecord> {
        let mut client = self.client.lock().unwrap();
        Self::read_trial(&mut client, trial)
    }

    fn load_history(&self, study: StudyId) -> Result<TrialHistory> {
        let mut client = self.client.lock().unwrap();
        let exists = client
            .query_opt("SELECT 1 FROM studies WHERE id = $1", &[&(study.0 as i64)])
            .map_err(pg)?
            .is_some();
        if !exists {
            return Err(Error::NotFound {
                kind: "study",
                id: study.to_string(),
            });
        }
        let ids: Vec<i64> = client
            .query(
                "SELECT id FROM trials WHERE study_id = $1 ORDER BY id",
                &[&(study.0 as i64)],
            )
            .map_err(pg)?
            .iter()
            .map(|r| r.get::<_, i64>(0))
            .collect();
        let records = ids
            .into_iter()
            .map(|id| Self::read_trial(&mut client, TrialId(id as u64)))
            .collect::<Result<Vec<_>>>()?;
        Ok(TrialHistory::new(records))
    }

    fn load_meta(&self, study: StudyId) -> Result<StudyMeta> {
        let mut client = self.client.lock().unwrap();
        let row = client
            .query_opt(
                "SELECT name, directions, sampler_name, pruner_name FROM studies WHERE id = $1",
                &[&(study.0 as i64)],
            )
            .map_err(pg)?
            .ok_or_else(|| Error::NotFound {
                kind: "study",
                id: study.to_string(),
            })?;
        let directions: Vec<(String, Direction)> = serde_json::from_str(&row.get::<_, String>(1))?;
        Ok(StudyMeta {
            name: row.get(0),
            directions,
            sampler_name: row.get(2),
            pruner_name: row.get(3),
        })
    }

    fn save_artifact(&self, trial: TrialId, name: &str, bytes: &[u8]) -> Result<()> {
        let mut client = self.client.lock().unwrap();
        client
            .execute(
                "INSERT INTO artifacts (trial_id, name, bytes) VALUES ($1, $2, $3) \
                 ON CONFLICT (trial_id, name) DO UPDATE SET bytes = EXCLUDED.bytes",
                &[&(trial.0 as i64), &name, &bytes],
            )
            .map_err(pg)?;
        Ok(())
    }

    fn load_artifact(&self, trial: TrialId, name: &str) -> Result<Option<Vec<u8>>> {
        let mut client = self.client.lock().unwrap();
        Ok(client
            .query_opt(
                "SELECT bytes FROM artifacts WHERE trial_id = $1 AND name = $2",
                &[&(trial.0 as i64), &name],
            )
            .map_err(pg)?
            .map(|r| r.get::<_, Vec<u8>>(0)))
    }

    fn list_artifacts(&self, trial: TrialId) -> Result<Vec<String>> {
        let mut client = self.client.lock().unwrap();
        Ok(client
            .query(
                "SELECT name FROM artifacts WHERE trial_id = $1 ORDER BY name",
                &[&(trial.0 as i64)],
            )
            .map_err(pg)?
            .iter()
            .map(|r| r.get::<_, String>(0))
            .collect())
    }

    fn claim_trial(
        &self,
        study: StudyId,
        owner: &str,
        now_ms: u64,
        lease_expiry_ms: u64,
    ) -> Result<Option<TrialId>> {
        let mut client = self.client.lock().unwrap();
        let waiting = serde_json::to_string(&TrialState::Waiting)?;
        let running = serde_json::to_string(&TrialState::Running)?;
        // A single atomic UPDATE ... WHERE id IN (SELECT ... FOR UPDATE SKIP
        // LOCKED LIMIT 1) is the server-side compare-and-swap: it both selects a
        // claimable trial and marks it running, so concurrent claimers cannot
        // grab the same trial.
        let row = client
            .query_opt(
                "UPDATE trials SET state = $5, lease_owner = $6, lease_expiry = $7, \
                     started_at = COALESCE(started_at, $4) \
                 WHERE id = ( \
                     SELECT id FROM trials \
                     WHERE study_id = $1 AND (state = $2 \
                        OR (state = $3 AND (lease_expiry IS NULL OR lease_expiry <= $4))) \
                     ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) \
                 RETURNING id",
                &[
                    &(study.0 as i64),
                    &waiting,
                    &running,
                    &(now_ms as i64),
                    &running,
                    &owner,
                    &(lease_expiry_ms as i64),
                ],
            )
            .map_err(pg)?;
        Ok(row.map(|r| TrialId(r.get::<_, i64>(0) as u64)))
    }

    fn renew_lease(&self, trial: TrialId, owner: &str, new_expiry_ms: u64) -> Result<bool> {
        let mut client = self.client.lock().unwrap();
        let affected = client
            .execute(
                "UPDATE trials SET lease_expiry = $1 WHERE id = $2 AND lease_owner = $3",
                &[&(new_expiry_ms as i64), &(trial.0 as i64), &owner],
            )
            .map_err(pg)?;
        Ok(affected > 0)
    }

    fn recover_orphans(&self, study: StudyId, now_ms: u64) -> Result<usize> {
        let mut client = self.client.lock().unwrap();
        let waiting = serde_json::to_string(&TrialState::Waiting)?;
        let running = serde_json::to_string(&TrialState::Running)?;
        let affected = client
            .execute(
                "UPDATE trials SET state = $1, lease_owner = NULL \
                 WHERE study_id = $2 AND state = $3 \
                   AND lease_expiry IS NOT NULL AND lease_expiry <= $4",
                &[&waiting, &(study.0 as i64), &running, &(now_ms as i64)],
            )
            .map_err(pg)?;
        Ok(affected as usize)
    }
}

/// Map a postgres error into our storage error.
fn pg(e: postgres::Error) -> Error {
    Error::Storage(e.to_string())
}

fn not_found_if_zero(affected: u64, trial: TrialId) -> Result<()> {
    if affected == 0 {
        Err(Error::NotFound {
            kind: "trial",
            id: trial.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Seconds since the Unix epoch (provenance timestamp), avoiding a time crate.
fn now_epoch_secs() -> String {
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

    /// Run the full storage lifecycle and lease protocol against a live server.
    /// Requires `TEST_POSTGRES_URL` (e.g. `host=localhost user=postgres
    /// dbname=automl_test`); the test skips cleanly when it is unset, so CI and
    /// offline builds pass while the code path is exercised wherever a DB exists.
    #[test]
    fn lifecycle_and_leases_against_live_server() {
        let Ok(url) = std::env::var("TEST_POSTGRES_URL") else {
            eprintln!("skipping: set TEST_POSTGRES_URL to run the Postgres backend test");
            return;
        };
        // A fresh schema per run keeps the test self-contained.
        {
            let mut admin = Client::connect(&url, NoTls).unwrap();
            admin
                .batch_execute(
                    "DROP TABLE IF EXISTS artifacts, reports, trials, studies, _schema_version CASCADE;",
                )
                .unwrap();
        }

        let s = PostgresStorage::connect(&url).unwrap();
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);

        let study = s
            .create_study(StudyMeta {
                name: "pg".into(),
                directions: vec![("loss".into(), Direction::Minimize)],
                sampler_name: "tpe".into(),
                pruner_name: "median".into(),
            })
            .unwrap();
        let t = s
            .enqueue_trial(study, ParamSet::new().with("x", ParamValue::Float(1.5)), 7)
            .unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.9)).unwrap();
        s.report(t, 1, NamedMetrics::single("loss", 0.4)).unwrap(); // idempotent
        s.save_artifact(t, "ckpt", &[1u8, 2, 3, 255]).unwrap();

        // Lease protocol: claim, exclusive hold, renew, orphan recovery.
        assert_eq!(s.claim_trial(study, "a", 0, 100).unwrap(), Some(t));
        assert_eq!(s.claim_trial(study, "b", 50, 150).unwrap(), None);
        assert!(s.renew_lease(t, "a", 200).unwrap());
        assert!(!s.renew_lease(t, "b", 200).unwrap());
        assert_eq!(s.recover_orphans(study, 300).unwrap(), 1);

        s.complete(
            t,
            TrialState::Complete,
            Some(NamedMetrics::single("loss", 0.3)),
        )
        .unwrap();
        let rec = s.load_trial(t).unwrap();
        assert_eq!(rec.state, TrialState::Complete);
        assert_eq!(rec.intermediate[0].metrics.get("loss"), Some(0.4));
        assert_eq!(rec.final_value("loss"), Some(0.3));
        assert_eq!(rec.params.float("x").unwrap(), 1.5);
        assert_eq!(
            s.load_artifact(t, "ckpt").unwrap(),
            Some(vec![1, 2, 3, 255])
        );
        assert_eq!(s.study_ids().unwrap(), vec![study]);
    }
}
