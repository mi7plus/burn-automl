//! Checkpoint promotion for successive-halving searches (roadmap v0.7;
//! PRD §10/§11 "checkpoint reuse", §18, §19 "Artifacts").
//!
//! ASHA (see [`crate::pruner::AshaPruner`]) defines geometric *rungs* of
//! resource — `min_resource * eta^k` epochs — and, at each rung, promotes only
//! the top `1/eta` trials that reached it. *Checkpoint promotion* couples that
//! ranking to persisted weights: when a trial reaches a rung, its checkpoint is
//! stored as a trial artifact (v0.5); a promoted continuation then **warm-starts
//! from the best predecessor's checkpoint** at the rung below instead of
//! retraining from scratch. This is what makes deep vision/video search (§10/§11)
//! affordable — the expensive early epochs are paid once and inherited.
//!
//! The primitive is framework-agnostic: it moves opaque checkpoint *bytes*
//! through the [`Storage`] artifact API and ranks trials by their reported
//! score. A framework adapter decides what those bytes are (serialized model
//! state) and how to resume from them.

use crate::error::Result;
use crate::metrics::Direction;
use crate::storage::Storage;
use crate::trial::TrialId;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Records checkpoints at successive-halving rungs and selects which trials are
/// promoted, so a continuation can inherit the best predecessor's weights.
///
/// Rung `k` corresponds to resource level `min_resource * eta^k` (matching
/// [`crate::pruner::AshaPruner`]). Checkpoints are stored as trial artifacts
/// named `ckpt@rung{k}`; the promotion ledger (which trials reached each rung,
/// with what score) is kept in memory and can be rebuilt from history.
pub struct CheckpointPromoter {
    storage: Arc<dyn Storage>,
    min_resource: u64,
    eta: usize,
    direction: Direction,
    // rung -> [(trial, score)] in insertion order; selection re-sorts by score.
    ledger: Mutex<BTreeMap<usize, Vec<(TrialId, f64)>>>,
}

impl CheckpointPromoter {
    /// A new promoter over `storage`. `min_resource` is the first rung's resource
    /// level (e.g. 1 epoch), `eta` (>= 2) is both the rung spacing and the inverse
    /// promotion fraction, and `direction` orders scores (which trials are "best").
    pub fn new(
        storage: Arc<dyn Storage>,
        min_resource: u64,
        eta: usize,
        direction: Direction,
    ) -> Self {
        CheckpointPromoter {
            storage,
            min_resource: min_resource.max(1),
            eta: eta.max(2),
            direction,
            ledger: Mutex::new(BTreeMap::new()),
        }
    }

    /// The rung index a resource level maps to, if it lands exactly on a rung
    /// (`min_resource * eta^k`). Returns `None` for levels between rungs.
    pub fn rung_of(&self, resource: u64) -> Option<usize> {
        if resource < self.min_resource || !resource.is_multiple_of(self.min_resource) {
            return None;
        }
        let mut level = self.min_resource;
        let mut k = 0;
        while level < resource {
            level = level.saturating_mul(self.eta as u64);
            k += 1;
        }
        (level == resource).then_some(k)
    }

    /// The artifact name used for a rung's checkpoint.
    pub fn checkpoint_name(rung: usize) -> String {
        format!("ckpt@rung{rung}")
    }

    /// Record that `trial` reached `rung` with `score`, persisting its checkpoint
    /// `bytes` as a trial artifact. Idempotent by `(trial, rung)`: re-recording a
    /// trial at the same rung overwrites its checkpoint and score.
    pub fn record(&self, rung: usize, trial: TrialId, score: f64, bytes: &[u8]) -> Result<()> {
        self.storage
            .save_artifact(trial, &Self::checkpoint_name(rung), bytes)?;
        let mut ledger = self.ledger.lock().unwrap();
        let entries = ledger.entry(rung).or_default();
        if let Some(e) = entries.iter_mut().find(|(t, _)| *t == trial) {
            e.1 = score;
        } else {
            entries.push((trial, score));
        }
        Ok(())
    }

    /// The trials that reached `rung`, best-first by the configured direction.
    /// Ties break by ascending trial id for determinism.
    pub fn ranked(&self, rung: usize) -> Vec<(TrialId, f64)> {
        let ledger = self.ledger.lock().unwrap();
        let mut entries = ledger.get(&rung).cloned().unwrap_or_default();
        entries.sort_by(|a, b| match self.direction {
            Direction::Maximize => {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0 .0.cmp(&b.0 .0))
            }
            Direction::Minimize => {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0 .0.cmp(&b.0 .0))
            }
        });
        entries
    }

    /// The promoted set at `rung`: the top `ceil(n / eta)` trials by score. At
    /// least one trial is promoted once any has reached the rung, matching ASHA's
    /// asynchronous top-`1/eta` rule.
    pub fn promoted(&self, rung: usize) -> Vec<TrialId> {
        let ranked = self.ranked(rung);
        if ranked.is_empty() {
            return Vec::new();
        }
        let keep = ranked.len().div_ceil(self.eta).max(1);
        ranked.into_iter().take(keep).map(|(t, _)| t).collect()
    }

    /// Whether `trial` is currently promoted at `rung`.
    pub fn is_promoted(&self, rung: usize, trial: TrialId) -> bool {
        self.promoted(rung).contains(&trial)
    }

    /// The checkpoint a promoted continuation should warm-start from: the bytes of
    /// the single best trial that reached `rung`, or `None` if no trial has yet
    /// reached it (the continuation then trains from scratch). This is the
    /// promotion step — weights flow from the best predecessor into its successor.
    pub fn warm_start(&self, rung: usize) -> Result<Option<Vec<u8>>> {
        let ranked = self.ranked(rung);
        match ranked.first() {
            Some((trial, _)) => self
                .storage
                .load_artifact(*trial, &Self::checkpoint_name(rung)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

    fn promoter(dir: Direction) -> CheckpointPromoter {
        CheckpointPromoter::new(Arc::new(InMemoryStorage::new()), 1, 3, dir)
    }

    #[test]
    fn rungs_are_geometric() {
        let p = promoter(Direction::Maximize); // min=1, eta=3 -> 1, 3, 9, 27
        assert_eq!(p.rung_of(1), Some(0));
        assert_eq!(p.rung_of(3), Some(1));
        assert_eq!(p.rung_of(9), Some(2));
        assert_eq!(p.rung_of(27), Some(3));
        // Between rungs: not a rung.
        assert_eq!(p.rung_of(2), None);
        assert_eq!(p.rung_of(4), None);
        assert_eq!(p.rung_of(0), None);
    }

    #[test]
    fn promotes_top_fraction_and_warm_starts_from_best() {
        let p = promoter(Direction::Maximize);
        // Six trials reach rung 0 with distinct scores; top ceil(6/3)=2 promoted.
        for (i, score) in [0.1, 0.9, 0.5, 0.8, 0.2, 0.3].into_iter().enumerate() {
            let payload = format!("weights-{i}");
            p.record(0, TrialId(i as u64), score, payload.as_bytes())
                .unwrap();
        }
        let promoted = p.promoted(0);
        assert_eq!(promoted.len(), 2);
        assert!(promoted.contains(&TrialId(1))); // score 0.9
        assert!(promoted.contains(&TrialId(3))); // score 0.8
        assert!(p.is_promoted(0, TrialId(1)));
        assert!(!p.is_promoted(0, TrialId(0)));
        // The warm-start checkpoint is the single best trial's bytes.
        let bytes = p.warm_start(0).unwrap().unwrap();
        assert_eq!(bytes, b"weights-1");
    }

    #[test]
    fn minimize_selects_lowest_scores() {
        let p = promoter(Direction::Minimize);
        for (i, loss) in [0.5, 0.1, 0.9].into_iter().enumerate() {
            p.record(0, TrialId(i as u64), loss, b"w").unwrap();
        }
        // ceil(3/3)=1 promoted: the lowest loss (trial 1, 0.1).
        assert_eq!(p.promoted(0), vec![TrialId(1)]);
    }

    #[test]
    fn record_is_idempotent_by_trial_and_rung() {
        let p = promoter(Direction::Maximize);
        p.record(0, TrialId(0), 0.2, b"old").unwrap();
        p.record(0, TrialId(0), 0.7, b"new").unwrap();
        // One entry, updated score and checkpoint.
        let ranked = p.ranked(0);
        assert_eq!(ranked, vec![(TrialId(0), 0.7)]);
        assert_eq!(p.warm_start(0).unwrap().unwrap(), b"new");
    }

    #[test]
    fn empty_rung_has_no_warm_start() {
        let p = promoter(Direction::Maximize);
        assert!(p.warm_start(0).unwrap().is_none());
        assert!(p.promoted(0).is_empty());
    }
}
