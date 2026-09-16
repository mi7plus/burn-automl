//! Distributed execution: the worker protocol over a lease-backed store
//! (PRD §18.2, §23 distributed-alpha).
//!
//! The plan specifies distributed coordination to protocol level: trial leasing
//! by compare-and-swap, heartbeats, orphan recovery, idempotent writes, and
//! exactly-once trial semantics — with `Storage` as the single source of truth,
//! so a coordinator holds no unrecoverable state (§18.2 coordinator
//! statelessness). This module implements the *worker* side of that protocol
//! against the [`Storage`] lease API:
//!
//! - A coordinator enqueues pending (`Waiting`) trials with
//!   [`enqueue_pending`]; sampling stays in one place and stays ask-tell aware.
//! - A [`Worker`] claims a trial ([`Storage::claim_trial`]), runs the objective
//!   while renewing its lease at each report (the heartbeat), and completes it
//!   *only if it still holds the lease* — so a worker that was declared dead and
//!   had its trial reassigned never double-completes it.
//! - Expired leases are recovered to the queue by [`Storage::recover_orphans`]
//!   (or lazily by the next [`Storage::claim_trial`]).
//!
//! The same store backs a single-process pool or many machines; only the
//! `Storage` implementation changes.
//!
//! ## Tuning the lease TTL (v0.9 hardening)
//!
//! [`Worker::new`] takes a `lease_ttl_ms` — how long a claim is valid before a
//! peer may reclaim it as an orphan. It trades failure-detection latency against
//! tolerance for slow steps:
//!
//! - Set the TTL to a comfortable multiple (≈3–5×) of the longest expected gap
//!   between reports (the heartbeat interval). Each [`ReportSink::report`] renews
//!   the lease, so a trial that reports steadily never expires mid-run.
//! - Too short and a healthy-but-slow trial is stolen and wastefully re-run; too
//!   long and a genuinely dead worker's trial sits idle until the TTL elapses.
//! - Objectives that report rarely (few, long epochs) want a larger TTL or an
//!   extra keep-alive report; objectives that report every step tolerate a small
//!   one. Recovery is idempotent either way — a stolen trial completes exactly
//!   once — so mistuning costs throughput, never correctness.
//!
//! The failure-injection tests below exercise mid-trial crashes, orphan-recovery
//! under load, and concurrent claim races to keep these guarantees honest (§24).

use crate::error::Result;
use crate::metrics::NamedMetrics;
use crate::objective::{Objective, ReportSink};
use crate::provenance::now_ms;
use crate::sampler::Sampler;
use crate::space::SearchSpace;
use crate::storage::Storage;
use crate::trial::{StudyId, TrialId};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Sample and enqueue `n` pending trials for a study (the coordinator's role).
/// Each proposal is enqueued as `Waiting` for workers to claim.
pub fn enqueue_pending(
    storage: &Arc<dyn Storage>,
    study: StudyId,
    space: &SearchSpace,
    sampler: &mut dyn Sampler,
    n: usize,
    base_seed: u64,
) -> Result<Vec<TrialId>> {
    let mut ids = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let history = storage.load_history(study)?;
        let params = sampler.suggest(space, &history);
        let seed = base_seed.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
        ids.push(storage.enqueue_trial(study, params, seed)?);
    }
    Ok(ids)
}

/// The outcome of one [`Worker::poll`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poll {
    /// No claimable trial was available.
    Idle,
    /// The worker ran and completed the given trial.
    Ran(TrialId),
    /// The worker ran the trial but lost its lease before completing (its work
    /// is discarded; another worker owns the trial now). Exactly-once is
    /// preserved.
    LostLease(TrialId),
}

/// A distributed worker that claims, runs and completes trials under a lease.
pub struct Worker {
    id: String,
    lease_ttl_ms: u64,
}

impl Worker {
    /// A worker identified by `id`, holding leases for `lease_ttl_ms` and
    /// renewing them at every report.
    pub fn new(id: impl Into<String>, lease_ttl_ms: u64) -> Self {
        Worker {
            id: id.into(),
            lease_ttl_ms: lease_ttl_ms.max(1),
        }
    }

    /// The worker's identity.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Claim and run a single trial, if one is available.
    pub fn poll<O: Objective>(
        &self,
        storage: &Arc<dyn Storage>,
        study: StudyId,
        objective: &O,
    ) -> Result<Poll> {
        let now = now_ms();
        let Some(trial) = storage.claim_trial(study, &self.id, now, now + self.lease_ttl_ms)?
        else {
            return Ok(Poll::Idle);
        };

        let params = storage.load_trial(trial)?.params;
        let mut sink = LeaseSink::new(storage.clone(), trial, self.id.clone(), self.lease_ttl_ms);
        let outcome = objective.evaluate(&params, &mut sink);
        let state = match &outcome {
            Ok(_) => crate::trial::TrialState::Complete,
            Err(_) => crate::trial::TrialState::Failed,
        };

        // Exactly-once: only write the terminal result if we still own the
        // lease. If the lease was lost mid-run (we were declared dead and the
        // trial reassigned), drop our result so we do not double-complete.
        if storage.renew_lease(trial, &self.id, now_ms() + self.lease_ttl_ms)? {
            storage.complete(trial, state, outcome.ok())?;
            Ok(Poll::Ran(trial))
        } else {
            Ok(Poll::LostLease(trial))
        }
    }

    /// Repeatedly [`poll`](Worker::poll) until no trial is available, returning
    /// how many trials this worker ran to completion.
    pub fn run_until_idle<O: Objective>(
        &self,
        storage: &Arc<dyn Storage>,
        study: StudyId,
        objective: &O,
    ) -> Result<usize> {
        let mut ran = 0;
        loop {
            match self.poll(storage, study, objective)? {
                Poll::Idle => break,
                Poll::Ran(_) => ran += 1,
                Poll::LostLease(_) => {}
            }
        }
        Ok(ran)
    }
}

/// A [`ReportSink`] that persists metrics and renews the worker's lease at each
/// report — the heartbeat. If the lease is lost, `should_stop` returns true so
/// the objective can bail out of an orphaned run.
struct LeaseSink {
    storage: Arc<dyn Storage>,
    trial: TrialId,
    owner: String,
    lease_ttl_ms: u64,
    lost: AtomicBool,
}

impl LeaseSink {
    fn new(storage: Arc<dyn Storage>, trial: TrialId, owner: String, lease_ttl_ms: u64) -> Self {
        LeaseSink {
            storage,
            trial,
            owner,
            lease_ttl_ms,
            lost: AtomicBool::new(false),
        }
    }
}

impl ReportSink for LeaseSink {
    fn trial_id(&self) -> TrialId {
        self.trial
    }

    fn report(&mut self, step: u64, metrics: NamedMetrics) -> Result<()> {
        self.storage.report(self.trial, step, metrics)?;
        // Heartbeat: renew the lease; losing it signals we were orphaned.
        let held =
            self.storage
                .renew_lease(self.trial, &self.owner, now_ms() + self.lease_ttl_ms)?;
        if !held {
            self.lost.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::metrics::Direction;
    use crate::param::ParamSet;
    use crate::sampler::RandomSampler;
    use crate::space::SearchSpace;
    use crate::storage::{InMemoryStorage, StudyMeta};
    use crate::trial::TrialState;

    fn study_with(storage: &Arc<dyn Storage>) -> StudyId {
        storage
            .create_study(StudyMeta {
                name: "dist".into(),
                directions: vec![("loss".into(), Direction::Minimize)],
                sampler_name: "random".into(),
                pruner_name: "none".into(),
            })
            .unwrap()
    }

    fn quadratic(p: &ParamSet, _s: &mut dyn ReportSink) -> Result<NamedMetrics> {
        let x = p.float("x")?;
        Ok(NamedMetrics::single("loss", (x - 2.0).powi(2)))
    }

    #[test]
    fn two_workers_share_the_queue_without_double_running() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let mut sampler = RandomSampler::new(1);
        let ids = enqueue_pending(&storage, study, &space, &mut sampler, 8, 1).unwrap();

        let a = Worker::new("a", 60_000);
        let b = Worker::new("b", 60_000);
        // Interleave the two workers until the queue drains.
        let mut ran_by_a = Vec::new();
        let mut ran_by_b = Vec::new();
        loop {
            let pa = a.poll(&storage, study, &quadratic).unwrap();
            let pb = b.poll(&storage, study, &quadratic).unwrap();
            if let Poll::Ran(t) = pa {
                ran_by_a.push(t);
            }
            if let Poll::Ran(t) = pb {
                ran_by_b.push(t);
            }
            if pa == Poll::Idle && pb == Poll::Idle {
                break;
            }
        }

        // Every trial ran exactly once across the two workers.
        let mut all: Vec<TrialId> = ran_by_a.iter().chain(&ran_by_b).copied().collect();
        all.sort();
        let mut expected = ids.clone();
        expected.sort();
        assert_eq!(all, expected, "each trial must run exactly once");

        let history = storage.load_history(study).unwrap();
        assert_eq!(history.completed().count(), 8);
    }

    #[test]
    fn many_threads_drain_the_queue_exactly_once() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let mut sampler = RandomSampler::new(4);
        enqueue_pending(&storage, study, &space, &mut sampler, 30, 4).unwrap();

        // Four worker threads race to drain the shared queue.
        let total: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|w| {
                    let storage = storage.clone();
                    scope.spawn(move || {
                        Worker::new(format!("w{w}"), 60_000)
                            .run_until_idle(&storage, study, &quadratic)
                            .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        // The 30 trials were run exactly once in aggregate, all completed.
        assert_eq!(total, 30);
        let history = storage.load_history(study).unwrap();
        assert_eq!(history.completed().count(), 30);
    }

    #[test]
    fn expired_lease_is_reclaimed_and_completes_exactly_once() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let mut sampler = RandomSampler::new(2);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let ids = enqueue_pending(&storage, study, &space, &mut sampler, 1, 2).unwrap();
        let t = ids[0];

        // Worker A claims with a lease expiring at t=100, then "dies".
        assert_eq!(storage.claim_trial(study, "a", 0, 100).unwrap(), Some(t));

        // Later, worker B claims: A's lease has expired, so the orphan is reassigned.
        assert_eq!(storage.claim_trial(study, "b", 200, 300).unwrap(), Some(t));

        // Exactly-once: A cannot renew (B owns it now), B can.
        assert!(!storage.renew_lease(t, "a", 400).unwrap());
        assert!(storage.renew_lease(t, "b", 400).unwrap());

        // B completes; the trial is terminal and complete once.
        storage
            .complete(
                t,
                TrialState::Complete,
                Some(NamedMetrics::single("loss", 0.0)),
            )
            .unwrap();
        assert_eq!(storage.load_trial(t).unwrap().state, TrialState::Complete);

        // A claimable scan finds nothing (the only trial is complete).
        assert_eq!(storage.claim_trial(study, "c", 500, 600).unwrap(), None);
    }

    #[test]
    fn recover_orphans_requeues_expired_leases() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let mut sampler = RandomSampler::new(3);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let ids = enqueue_pending(&storage, study, &space, &mut sampler, 2, 3).unwrap();

        // Claim both with short leases.
        assert!(storage.claim_trial(study, "a", 0, 100).unwrap().is_some());
        assert!(storage.claim_trial(study, "a", 0, 100).unwrap().is_some());

        // Nothing claimable while leases are valid.
        assert_eq!(storage.claim_trial(study, "b", 50, 150).unwrap(), None);

        // After expiry, an explicit sweep requeues both orphans.
        assert_eq!(storage.recover_orphans(study, 200).unwrap(), 2);
        for t in &ids {
            assert_eq!(storage.load_trial(*t).unwrap().state, TrialState::Waiting);
        }
    }

    // ---- failure-injection suite (roadmap v0.9, §24 "Failure injection") ------
    //
    // These simulate crashes at the storage-protocol level with injected clock
    // values, so the exactly-once and no-lost-trial guarantees are checked
    // deterministically rather than by racing wall-clock leases.

    #[test]
    fn worker_crash_mid_trial_is_recovered_and_completes_once() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let mut sampler = RandomSampler::new(5);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let ids = enqueue_pending(&storage, study, &space, &mut sampler, 1, 5).unwrap();
        let t = ids[0];

        // Worker A claims and reports partial progress, then crashes (never
        // completes, never renews).
        assert_eq!(storage.claim_trial(study, "a", 0, 100).unwrap(), Some(t));
        storage
            .report(t, 1, NamedMetrics::single("loss", 9.0))
            .unwrap();

        // A sweep after the lease expiry requeues the orphan; B picks it up.
        assert_eq!(storage.recover_orphans(study, 150).unwrap(), 1);
        assert_eq!(storage.claim_trial(study, "b", 150, 250).unwrap(), Some(t));

        // The crashed worker A can no longer influence the trial (exactly-once).
        assert!(!storage.renew_lease(t, "a", 300).unwrap());
        assert!(storage.renew_lease(t, "b", 300).unwrap());
        storage
            .complete(
                t,
                TrialState::Complete,
                Some(NamedMetrics::single("loss", 0.0)),
            )
            .unwrap();

        // Completed exactly once, with B's result.
        let history = storage.load_history(study).unwrap();
        assert_eq!(history.completed().count(), 1);
        assert_eq!(
            storage.load_trial(t).unwrap().final_value("loss"),
            Some(0.0)
        );
    }

    #[test]
    fn orphan_recovery_load_test_loses_no_trials() {
        // A larger queue processed with a deterministic crash pattern: every third
        // claim "crashes" (is left un-completed). Repeated recover-and-retry must
        // eventually complete every trial exactly once, with none lost.
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let mut sampler = RandomSampler::new(7);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let n = 40;
        enqueue_pending(&storage, study, &space, &mut sampler, n, 7).unwrap();

        let mut clock: u64 = 0;
        let mut claims = 0usize;
        let mut completed = 0usize;
        // Bound the loop generously; it should finish far sooner.
        for _ in 0..1000 {
            clock += 10;
            // Requeue anything whose (short) lease has expired.
            storage.recover_orphans(study, clock).unwrap();
            let Some(t) = storage.claim_trial(study, "w", clock, clock + 5).unwrap() else {
                // Nothing claimable now; if all are terminal we are done.
                if storage.load_history(study).unwrap().completed().count() == n {
                    break;
                }
                continue;
            };
            claims += 1;
            // Every third claim crashes: leave the lease to expire un-completed.
            if claims.is_multiple_of(3) {
                continue;
            }
            // Otherwise renew (still own it) and complete.
            assert!(storage.renew_lease(t, "w", clock + 100).unwrap());
            storage
                .complete(
                    t,
                    TrialState::Complete,
                    Some(NamedMetrics::single("loss", 1.0)),
                )
                .unwrap();
            completed += 1;
        }

        assert_eq!(completed, n, "every trial completed exactly once");
        let history = storage.load_history(study).unwrap();
        assert_eq!(history.completed().count(), n);
        // No trial left stranded in Waiting/Running.
        assert!(history
            .records()
            .iter()
            .all(|r| r.state == TrialState::Complete));
    }

    #[test]
    fn concurrent_claims_on_one_trial_are_mutually_exclusive() {
        // Many threads race to claim a single available trial at the same instant;
        // the compare-and-swap must grant it to exactly one.
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = study_with(&storage);
        let mut sampler = RandomSampler::new(11);
        let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
        let ids = enqueue_pending(&storage, study, &space, &mut sampler, 1, 11).unwrap();
        let t = ids[0];

        let winners: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|w| {
                    let storage = storage.clone();
                    scope.spawn(move || {
                        let owner = format!("w{w}");
                        matches!(
                            storage.claim_trial(study, &owner, 0, 1000),
                            Ok(Some(id)) if id == t
                        ) as usize
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        assert_eq!(winners, 1, "exactly one worker may claim the trial");
    }
}
