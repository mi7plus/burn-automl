//! Hyperparameter importance analysis (PRD §25, pulled forward to v0.2; §29
//! item 13).
//!
//! The plan moves importance analysis forward from the v1.0 checklist so the
//! tool is inspectable while it is being dogfooded. This is "importance v1": a
//! main-effect variance decomposition (a discretized, dependency-free fANOVA
//! first-order effect) over a completed study's [`TrialHistory`], returned as a
//! **plain data structure** with no UI dependency, so it can be consumed by a
//! CLI, a dashboard, or user code alike.
//!
//! For each parameter `p`, trials are grouped by `p`'s value (categorical: by
//! category; numeric: by quantile bins) and the between-group variance of the
//! objective is measured as a fraction of the objective's total variance within
//! the region where `p` is active — its main-effect share. Shares are then
//! normalized across parameters to sum to 1, matching the convention users
//! expect from tools like Optuna.
//!
//! Conditional parameters are handled per-branch: a parameter only present in
//! some trials is scored over exactly those trials, consistent with the
//! per-branch treatment elsewhere in the engine (§5.1).

use crate::param::ParamValue;
use crate::trial::TrialHistory;

/// The estimated importance of one parameter for a study's objective.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamImportance {
    /// The parameter name.
    pub param: String,
    /// Normalized main-effect importance in `[0, 1]`; importances across all
    /// parameters sum to 1 (or all are 0 when the objective has no variance).
    pub importance: f64,
}

/// Maximum number of quantile bins used to discretize a numeric parameter.
const MAX_BINS: usize = 10;

/// Compute per-parameter importances for `objective` over the completed trials
/// in `history`, sorted most-important first.
///
/// Returns an empty vector when there are too few completed trials (< 2) or the
/// objective has no variance (nothing to attribute).
pub fn importance(history: &TrialHistory, objective: &str) -> Vec<ParamImportance> {
    // Gather completed (params, y) pairs.
    let data: Vec<(&crate::param::ParamSet, f64)> = history
        .completed()
        .filter_map(|r| r.final_value(objective).map(|y| (&r.params, y)))
        .filter(|(_, y)| y.is_finite())
        .collect();

    if data.len() < 2 {
        return Vec::new();
    }

    // Union of parameter names appearing in any trial, in first-seen order.
    let mut names: Vec<String> = Vec::new();
    for (params, _) in &data {
        for (name, _) in params.iter() {
            if !names.iter().any(|n| n == name) {
                names.push(name.clone());
            }
        }
    }

    // Raw main-effect share (R^2 within active region) per parameter.
    let mut raw: Vec<(String, f64)> = Vec::new();
    for name in &names {
        let subset: Vec<(&ParamValue, f64)> = data
            .iter()
            .filter_map(|(p, y)| p.get(name).map(|v| (v, *y)))
            .collect();
        if subset.len() < 2 {
            raw.push((name.clone(), 0.0));
            continue;
        }
        let ys: Vec<f64> = subset.iter().map(|(_, y)| *y).collect();
        let total = variance(&ys);
        if total <= f64::EPSILON {
            raw.push((name.clone(), 0.0));
            continue;
        }
        let groups = group_objective(&subset);
        let between = between_group_variance(&groups);
        raw.push((name.clone(), (between / total).clamp(0.0, 1.0)));
    }

    // Normalize to sum to 1 for a conventional importance profile.
    let total_raw: f64 = raw.iter().map(|(_, v)| v).sum();
    let mut out: Vec<ParamImportance> = raw
        .into_iter()
        .map(|(param, v)| ParamImportance {
            param,
            importance: if total_raw > 0.0 { v / total_raw } else { 0.0 },
        })
        .collect();

    out.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.param.cmp(&b.param))
    });
    out
}

/// Group the objective values by the parameter value. Categorical and boolean
/// values group by exact value; numeric values are split into quantile bins.
fn group_objective(subset: &[(&ParamValue, f64)]) -> Vec<Vec<f64>> {
    match subset.first().map(|(v, _)| *v) {
        Some(ParamValue::Categorical(_)) => group_by_key(subset, |v| match v {
            ParamValue::Categorical(s) => s.clone(),
            _ => String::new(),
        }),
        Some(ParamValue::Bool(_)) => group_by_key(subset, |v| match v {
            ParamValue::Bool(b) => b.to_string(),
            _ => String::new(),
        }),
        Some(ParamValue::Float(_)) | Some(ParamValue::Int(_)) => quantile_bins(subset),
        None => Vec::new(),
    }
}

/// Group objective values by a string key derived from the parameter value.
fn group_by_key<F>(subset: &[(&ParamValue, f64)], key: F) -> Vec<Vec<f64>>
where
    F: Fn(&ParamValue) -> String,
{
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (v, y) in subset {
        groups.entry(key(v)).or_default().push(*y);
    }
    groups.into_values().collect()
}

/// Split numeric-valued observations into up to [`MAX_BINS`] contiguous
/// quantile bins by the parameter value, returning the objective values in each.
fn quantile_bins(subset: &[(&ParamValue, f64)]) -> Vec<Vec<f64>> {
    let mut pairs: Vec<(f64, f64)> = subset
        .iter()
        .filter_map(|(v, y)| numeric(v).map(|x| (x, *y)))
        .collect();
    if pairs.len() < 2 {
        return vec![pairs.into_iter().map(|(_, y)| y).collect()];
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Number of distinct x values caps the useful bin count.
    let distinct = {
        let mut d = 1usize;
        for w in pairs.windows(2) {
            if w[1].0 > w[0].0 {
                d += 1;
            }
        }
        d
    };
    let k = MAX_BINS.min(distinct).max(1);
    if k == 1 {
        return vec![pairs.into_iter().map(|(_, y)| y).collect()];
    }

    let n = pairs.len();
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); k];
    for (i, (_, y)) in pairs.iter().enumerate() {
        // Contiguous, near-equal-size bins by rank.
        let bin = (i * k) / n;
        bins[bin.min(k - 1)].push(*y);
    }
    bins.retain(|b| !b.is_empty());
    bins
}

/// Numeric axis value for a parameter, if it is float/int/bool.
fn numeric(v: &ParamValue) -> Option<f64> {
    match v {
        ParamValue::Float(x) => Some(*x),
        ParamValue::Int(x) => Some(*x as f64),
        ParamValue::Bool(b) => Some(*b as u8 as f64),
        ParamValue::Categorical(_) => None,
    }
}

/// Population variance of a slice.
fn variance(ys: &[f64]) -> f64 {
    let n = ys.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = ys.iter().sum::<f64>() / n;
    ys.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n
}

/// Between-group variance: the variance of group means weighted by group size,
/// measured against the grand mean over all grouped observations.
fn between_group_variance(groups: &[Vec<f64>]) -> f64 {
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total == 0 {
        return 0.0;
    }
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;
    groups
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| {
            let w = g.len() as f64 / n_total as f64;
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            w * (mean - grand_mean).powi(2)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;
    use crate::metrics::NamedMetrics;
    use crate::param::ParamSet;
    use crate::sampler::RandomSampler;
    use crate::space::SearchSpace;
    use crate::trial::{StudyId, TrialId, TrialRecord, TrialState};

    fn completed(id: u64, params: ParamSet, y: f64) -> TrialRecord {
        let mut r = TrialRecord::new(TrialId(id), StudyId(0), params, id);
        r.state = TrialState::Complete;
        r.final_metrics = Some(NamedMetrics::single("loss", y));
        r
    }

    #[test]
    fn empty_when_too_few_trials() {
        let h = TrialHistory::new(vec![completed(0, ParamSet::new(), 1.0)]);
        assert!(importance(&h, "loss").is_empty());
    }

    #[test]
    fn influential_parameter_ranks_higher() {
        // y depends strongly on `a`, negligibly on `b`.
        let space = SearchSpace::new()
            .add("a", Distribution::float(-5.0, 5.0))
            .add("b", Distribution::float(-5.0, 5.0));
        use crate::sampler::Sampler;
        let mut sampler = RandomSampler::new(1);
        let mut records = Vec::new();
        for i in 0..200u64 {
            let p = sampler.suggest(&space, &TrialHistory::new(records.clone()));
            let a = p.float("a").unwrap();
            let b = p.float("b").unwrap();
            let y = (a - 2.0).powi(2) + 1e-4 * b;
            records.push(completed(i, p, y));
        }
        let imp = importance(&TrialHistory::new(records), "loss");
        assert_eq!(imp.len(), 2);
        assert_eq!(imp[0].param, "a", "a should dominate: {imp:?}");
        assert!(
            imp[0].importance > 0.8,
            "a importance = {}",
            imp[0].importance
        );
        // Importances sum to ~1.
        let sum: f64 = imp.iter().map(|i| i.importance).sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
    }

    #[test]
    fn single_parameter_gets_all_importance() {
        let mut records = Vec::new();
        for i in 0..10u64 {
            let p = ParamSet::new().with("x", ParamValue::Float(i as f64));
            records.push(completed(i, p, (i as f64 - 3.0).powi(2)));
        }
        let imp = importance(&TrialHistory::new(records), "loss");
        assert_eq!(imp.len(), 1);
        assert!((imp[0].importance - 1.0).abs() < 1e-9);
    }
}
