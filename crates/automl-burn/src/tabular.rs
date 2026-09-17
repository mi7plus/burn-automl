//! High-level tabular AutoML: `AutoClassifier` and `AutoRegressor` (PRD §20,
//! §6).
//!
//! These are the one-call `Auto*` builders the plan sketches — thin wrappers
//! over a [`Study`] plus a pre-populated
//! [`SearchSpace`] (§4.2: "every downstream
//! `Auto*` API is just a `TaskAdapter` plus a pre-built `SearchSpace`"). The
//! plan (§6) notes classifier and regressor share ~80% of their
//! evaluation-adapter code, so they ship together and route through the same
//! MLP training core here.
//!
//! Both take in-memory tabular data (feature rows + targets), split off a
//! validation set, and search MLP hyperparameters (`lr`, `hidden_size`,
//! `num_layers`, `dropout`) to maximize validation accuracy (classification) or
//! minimize validation RMSE (regression).

use crate::evaluation::Evaluation;
use crate::{Mlp, MlpConfig, TrainBackend, TrainConfig};
use automl_core::error::{Error, Result};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};
use automl_core::trial::{TrialId, TrialRecord};

use burn::module::AutodiffModule;
use burn::nn::loss::{CrossEntropyLoss, MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// The outcome of an `Auto*` search.
pub struct AutoResult {
    /// The best trial's parameters, if any trial completed.
    pub best_params: Option<ParamSet>,
    /// The best validation score (accuracy % for classification, RMSE for
    /// regression).
    pub best_score: Option<f64>,
    /// The completed study, for history/importance/dashboard inspection.
    pub study: Study,
}

impl AutoResult {
    fn from_study(study: Study) -> Result<Self> {
        let best: Option<TrialRecord> = study.best_trial()?;
        let objective = study
            .directions()
            .first()
            .map(|(n, _)| n.clone())
            .unwrap_or_default();
        let best_score = best.as_ref().and_then(|t| t.final_value(&objective));
        let best_params = best.map(|t| t.params);
        Ok(AutoResult {
            best_params,
            best_score,
            study,
        })
    }
}

/// Search space shared by the tabular helpers.
fn tabular_space() -> SearchSpace {
    SearchSpace::new()
        .add("lr", Distribution::log_float(1e-4, 1e-1))
        .add("hidden_size", Distribution::int(16, 256))
        .add("num_layers", Distribution::int(1, 3))
        .add("dropout", Distribution::float(0.0, 0.5))
}

fn config_from(
    params: &ParamSet,
    input_dim: usize,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
) -> Result<TrainConfig> {
    Ok(TrainConfig {
        epochs,
        batch_size,
        lr: params.float("lr")?,
        seed: 42,
        model: MlpConfig::new(params.int("hidden_size")? as usize)
            .with_input_dim(input_dim)
            .with_num_classes(num_classes)
            .with_num_hidden_layers(params.int("num_layers")? as usize)
            .with_dropout(params.float("dropout")?),
    })
}

// ------------------------------ classifier -----------------------------------

/// One-call hyperparameter search for tabular classification (PRD §20).
pub struct AutoClassifier {
    features: Vec<Vec<f32>>,
    labels: Vec<i64>,
    num_classes: usize,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    evaluation: Evaluation,
    seed: u64,
}

impl AutoClassifier {
    /// A new builder with default schedule (5 epochs, batch 32, 20 trials) and
    /// holdout evaluation.
    pub fn new() -> Self {
        AutoClassifier {
            features: Vec::new(),
            labels: Vec::new(),
            num_classes: 0,
            epochs: 5,
            batch_size: 32,
            trials: 20,
            evaluation: Evaluation::default(),
            seed: 0,
        }
    }

    /// Provide the feature rows and integer class labels.
    pub fn dataset(mut self, features: Vec<Vec<f32>>, labels: Vec<i64>) -> Self {
        self.features = features;
        self.labels = labels;
        self
    }

    /// Number of classes. If unset, inferred as `max(label) + 1`.
    pub fn num_classes(mut self, k: usize) -> Self {
        self.num_classes = k;
        self
    }

    /// Epochs per trial.
    pub fn epochs(mut self, e: usize) -> Self {
        self.epochs = e.max(1);
        self
    }

    /// Number of trials to search.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Choose the evaluation scheme (default [`Evaluation::Holdout`] at 0.2).
    pub fn evaluation(mut self, evaluation: Evaluation) -> Self {
        self.evaluation = evaluation;
        self
    }

    /// Seed for the split and the search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search and return the best configuration.
    pub fn fit(self) -> Result<AutoResult> {
        if self.features.is_empty() || self.features.len() != self.labels.len() {
            return Err(Error::Objective(
                "classifier needs matching non-empty features and labels".into(),
            ));
        }
        let feat_dim = self.features[0].len();
        let num_classes = if self.num_classes == 0 {
            (self.labels.iter().copied().max().unwrap_or(0) + 1) as usize
        } else {
            self.num_classes
        };

        let folds = Arc::new(self.evaluation.folds(
            self.features.len(),
            Some(&self.labels),
            self.seed,
        ));
        let full = Arc::new((self.features, self.labels));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let mut study = Study::builder(tabular_space())
            .name("auto-classifier")
            .maximize("accuracy")
            .sampler(TpeSampler::new("accuracy", Direction::Maximize, self.seed))
            .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let objective = move |p: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let cfg = config_from(p, feat_dim, num_classes, epochs, batch)?;
            let acc = evaluate_classifier::<TrainBackend>(&cfg, &full, &folds, sink);
            Ok(NamedMetrics::single("accuracy", acc as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        AutoResult::from_study(study)
    }
}

impl Default for AutoClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------ regressor ------------------------------------

/// One-call hyperparameter search for tabular regression (PRD §20).
pub struct AutoRegressor {
    features: Vec<Vec<f32>>,
    targets: Vec<f32>,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    evaluation: Evaluation,
    seed: u64,
}

impl AutoRegressor {
    /// A new builder with default schedule (5 epochs, batch 32, 20 trials) and
    /// holdout evaluation.
    pub fn new() -> Self {
        AutoRegressor {
            features: Vec::new(),
            targets: Vec::new(),
            epochs: 5,
            batch_size: 32,
            trials: 20,
            evaluation: Evaluation::default(),
            seed: 0,
        }
    }

    /// Provide the feature rows and continuous targets.
    pub fn dataset(mut self, features: Vec<Vec<f32>>, targets: Vec<f32>) -> Self {
        self.features = features;
        self.targets = targets;
        self
    }

    /// Epochs per trial.
    pub fn epochs(mut self, e: usize) -> Self {
        self.epochs = e.max(1);
        self
    }

    /// Number of trials to search.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Choose the evaluation scheme (default [`Evaluation::Holdout`] at 0.2).
    /// Stratification is not meaningful for regression, so `StratifiedKFold`
    /// behaves as plain `KFold` here.
    pub fn evaluation(mut self, evaluation: Evaluation) -> Self {
        self.evaluation = evaluation;
        self
    }

    /// Seed for the split and the search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search and return the best configuration (minimizing RMSE).
    pub fn fit(self) -> Result<AutoResult> {
        if self.features.is_empty() || self.features.len() != self.targets.len() {
            return Err(Error::Objective(
                "regressor needs matching non-empty features and targets".into(),
            ));
        }
        let feat_dim = self.features[0].len();
        let folds = Arc::new(self.evaluation.folds(self.features.len(), None, self.seed));
        let full = Arc::new((self.features, self.targets));
        let (epochs, batch) = (self.epochs, self.batch_size);

        let mut study = Study::builder(tabular_space())
            .name("auto-regressor")
            .minimize("rmse")
            .sampler(TpeSampler::new("rmse", Direction::Minimize, self.seed))
            .pruner(MedianPruner::new("rmse", Direction::Minimize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let objective = move |p: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let cfg = config_from(p, feat_dim, 1, epochs, batch)?;
            let rmse = evaluate_regressor::<TrainBackend>(&cfg, &full, &folds, sink);
            Ok(NamedMetrics::single("rmse", rmse as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        AutoResult::from_study(study)
    }
}

impl Default for AutoRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------ training core --------------------------------

/// Train/validation split materialized as owned rows.
struct Dataset {
    train_x: Vec<Vec<f32>>,
    train_y: Vec<i64>,
    val_x: Vec<Vec<f32>>,
    val_y: Vec<i64>,
    // Regression targets parallel to the feature rows (empty for classification).
    train_t: Vec<f32>,
    val_t: Vec<f32>,
}

fn gather(features: &[Vec<f32>], labels: &[i64], train: &[usize], val: &[usize]) -> Dataset {
    Dataset {
        train_x: train.iter().map(|&i| features[i].clone()).collect(),
        train_y: train.iter().map(|&i| labels[i]).collect(),
        val_x: val.iter().map(|&i| features[i].clone()).collect(),
        val_y: val.iter().map(|&i| labels[i]).collect(),
        train_t: Vec::new(),
        val_t: Vec::new(),
    }
}

fn gather_reg(features: &[Vec<f32>], targets: &[f32], train: &[usize], val: &[usize]) -> Dataset {
    Dataset {
        train_x: train.iter().map(|&i| features[i].clone()).collect(),
        train_y: Vec::new(),
        val_x: val.iter().map(|&i| features[i].clone()).collect(),
        val_y: Vec::new(),
        train_t: train.iter().map(|&i| targets[i]).collect(),
        val_t: val.iter().map(|&i| targets[i]).collect(),
    }
}

/// A report sink that discards everything — used inside a cross-validation fold
/// so per-epoch reporting from individual folds does not reach the study (the
/// aggregated per-fold metric is reported to the real sink instead).
struct NullSink;
impl ReportSink for NullSink {
    fn trial_id(&self) -> TrialId {
        TrialId(0)
    }
    fn report(&mut self, _step: u64, _metrics: NamedMetrics) -> Result<()> {
        Ok(())
    }
    fn should_stop(&self) -> bool {
        false
    }
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

/// Evaluate a classifier config under the given folds, returning the mean
/// validation accuracy. A single (holdout) fold reports its learning curve to
/// `sink` per epoch; multiple folds report the running-mean accuracy per fold so
/// the median pruner can still act on a partial cross-validation.
fn evaluate_classifier<B: AutodiffBackend>(
    cfg: &TrainConfig,
    full: &(Vec<Vec<f32>>, Vec<i64>),
    folds: &[(Vec<usize>, Vec<usize>)],
    sink: &mut dyn ReportSink,
) -> f32 {
    if folds.len() == 1 {
        let ds = gather(&full.0, &full.1, &folds[0].0, &folds[0].1);
        return train_classifier::<B>(cfg, &ds, sink);
    }
    let mut scores = Vec::with_capacity(folds.len());
    for (fold, (train, val)) in folds.iter().enumerate() {
        let ds = gather(&full.0, &full.1, train, val);
        scores.push(train_classifier::<B>(cfg, &ds, &mut NullSink));
        let _ = sink.report(
            fold as u64 + 1,
            NamedMetrics::single("accuracy", mean(&scores) as f64),
        );
        if sink.should_stop() {
            break;
        }
    }
    mean(&scores)
}

/// Evaluate a regressor config under the given folds, returning the mean
/// validation RMSE (see [`evaluate_classifier`] for the reporting scheme).
fn evaluate_regressor<B: AutodiffBackend>(
    cfg: &TrainConfig,
    full: &(Vec<Vec<f32>>, Vec<f32>),
    folds: &[(Vec<usize>, Vec<usize>)],
    sink: &mut dyn ReportSink,
) -> f32 {
    if folds.len() == 1 {
        let ds = gather_reg(&full.0, &full.1, &folds[0].0, &folds[0].1);
        return train_regressor::<B>(cfg, &ds, sink);
    }
    let mut scores = Vec::with_capacity(folds.len());
    for (fold, (train, val)) in folds.iter().enumerate() {
        let ds = gather_reg(&full.0, &full.1, train, val);
        scores.push(train_regressor::<B>(cfg, &ds, &mut NullSink));
        let _ = sink.report(
            fold as u64 + 1,
            NamedMetrics::single("rmse", mean(&scores) as f64),
        );
        if sink.should_stop() {
            break;
        }
    }
    mean(&scores)
}

fn feature_tensor<B: Backend>(rows: &[Vec<f32>], device: &B::Device) -> Tensor<B, 2> {
    let n = rows.len();
    let d = rows.first().map_or(0, |r| r.len());
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [n, d]).convert::<B::FloatElem>(),
        device,
    )
}

fn int_tensor<B: Backend>(labels: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let data = TensorData::new(labels.to_vec(), [labels.len()]).convert::<B::IntElem>();
    Tensor::<B, 1, Int>::from_data(data, device)
}

fn target_tensor<B: Backend>(targets: &[f32], device: &B::Device) -> Tensor<B, 2> {
    let n = targets.len();
    Tensor::<B, 2>::from_data(
        TensorData::new(targets.to_vec(), [n, 1]).convert::<B::FloatElem>(),
        device,
    )
}

fn train_classifier<B: AutodiffBackend>(
    cfg: &TrainConfig,
    data: &Dataset,
    sink: &mut dyn ReportSink,
) -> f32 {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut model = cfg.model.init::<B>(&Default::default());
    let mut optim = AdamConfig::new().init();
    let device = Default::default();
    let mut order: Vec<usize> = (0..data.train_x.len()).collect();
    let mut acc = 0.0;

    for epoch in 1..=cfg.epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let feats: Vec<Vec<f32>> = chunk.iter().map(|&i| data.train_x[i].clone()).collect();
            let labels: Vec<i64> = chunk.iter().map(|&i| data.train_y[i]).collect();
            let x = feature_tensor::<B>(&feats, &device);
            let y = int_tensor::<B>(&labels, &device);
            let out = model.forward_flat(x);
            let loss = CrossEntropyLoss::new(None, &out.device()).forward(out, y);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
        let valid = model.valid();
        acc = classification_accuracy::<B::InnerBackend>(
            &valid,
            &data.val_x,
            &data.val_y,
            cfg.batch_size,
        );
        let _ = sink.report(epoch as u64, NamedMetrics::single("accuracy", acc as f64));
        if sink.should_stop() {
            break;
        }
    }
    acc
}

fn train_regressor<B: AutodiffBackend>(
    cfg: &TrainConfig,
    data: &Dataset,
    sink: &mut dyn ReportSink,
) -> f32 {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut model = cfg.model.init::<B>(&Default::default());
    let mut optim = AdamConfig::new().init();
    let device = Default::default();
    let mut order: Vec<usize> = (0..data.train_x.len()).collect();
    let mut rmse = f32::INFINITY;

    for epoch in 1..=cfg.epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let feats: Vec<Vec<f32>> = chunk.iter().map(|&i| data.train_x[i].clone()).collect();
            let targets: Vec<f32> = chunk.iter().map(|&i| data.train_t[i]).collect();
            let x = feature_tensor::<B>(&feats, &device);
            let y = target_tensor::<B>(&targets, &device);
            let out = model.forward_flat(x);
            let loss = MseLoss::new().forward(out, y, Reduction::Mean);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
        let valid = model.valid();
        rmse = regression_rmse::<B::InnerBackend>(&valid, &data.val_x, &data.val_t, cfg.batch_size);
        let _ = sink.report(epoch as u64, NamedMetrics::single("rmse", rmse as f64));
        if sink.should_stop() {
            break;
        }
    }
    rmse
}

fn classification_accuracy<B: Backend>(
    model: &Mlp<B>,
    val_x: &[Vec<f32>],
    val_y: &[i64],
    batch_size: usize,
) -> f32 {
    if val_x.is_empty() {
        return 0.0;
    }
    let device = Default::default();
    let mut correct: i64 = 0;
    for (fx, fy) in val_x.chunks(batch_size).zip(val_y.chunks(batch_size)) {
        let x = feature_tensor::<B>(fx, &device);
        let y = int_tensor::<B>(fy, &device);
        let out = model.forward_flat(x);
        let pred = out.argmax(1).squeeze_dim(1);
        correct += pred.equal(y).int().sum().into_scalar().elem::<i64>();
    }
    correct as f32 / val_x.len() as f32 * 100.0
}

fn regression_rmse<B: Backend>(
    model: &Mlp<B>,
    val_x: &[Vec<f32>],
    val_t: &[f32],
    batch_size: usize,
) -> f32 {
    if val_x.is_empty() {
        return f32::INFINITY;
    }
    let device = Default::default();
    let mut sse = 0.0f32;
    for (fx, ft) in val_x.chunks(batch_size).zip(val_t.chunks(batch_size)) {
        let x = feature_tensor::<B>(fx, &device);
        let out = model.forward_flat(x).squeeze_dim::<1>(1); // [batch]
        let y = Tensor::<B, 1>::from_data(
            TensorData::new(ft.to_vec(), [ft.len()]).convert::<B::FloatElem>(),
            &device,
        );
        let diff = out - y;
        sse += diff.clone().mul(diff).sum().into_scalar().elem::<f32>();
    }
    (sse / val_x.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A linearly separable 2-class problem: class 1 when x0 + x1 > 0.
    fn classification_data(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<i64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        use rand::Rng;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..n {
            let a: f32 = rng.gen_range(-1.0..1.0);
            let b: f32 = rng.gen_range(-1.0..1.0);
            xs.push(vec![a, b]);
            ys.push(if a + b > 0.0 { 1 } else { 0 });
        }
        (xs, ys)
    }

    #[test]
    fn auto_classifier_learns_separable_problem() {
        let (x, y) = classification_data(400, 1);
        let result = AutoClassifier::new()
            .dataset(x, y)
            .num_classes(2)
            .epochs(8)
            .trials(4)
            .seed(1)
            .fit()
            .unwrap();
        let acc = result.best_score.unwrap();
        assert!(
            acc > 80.0,
            "best accuracy was {acc}, expected the search to find a good model"
        );
    }

    #[test]
    fn auto_regressor_fits_linear_function() {
        // y = 2*x0 - x1 + 0.5
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        use rand::Rng;
        let (mut xs, mut ts) = (Vec::new(), Vec::new());
        for _ in 0..400 {
            let a: f32 = rng.gen_range(-1.0..1.0);
            let b: f32 = rng.gen_range(-1.0..1.0);
            xs.push(vec![a, b]);
            ts.push(2.0 * a - b + 0.5);
        }
        let result = AutoRegressor::new()
            .dataset(xs, ts)
            .epochs(12)
            .trials(4)
            .seed(2)
            .fit()
            .unwrap();
        let rmse = result.best_score.unwrap();
        // Target std is ~1.4; a fitted model should get RMSE well below that. The
        // margin (not the exact figure) is the signal — Burn's ndarray backend
        // reduces floats with rayon, so the value varies slightly across
        // platforms, so keep the bound comfortably above the observed fit.
        assert!(
            rmse < 0.7,
            "best rmse was {rmse}, expected a good linear fit"
        );
    }

    #[test]
    fn empty_dataset_errors() {
        assert!(AutoClassifier::new().fit().is_err());
        assert!(AutoRegressor::new().fit().is_err());
    }

    #[test]
    fn auto_classifier_with_stratified_kfold() {
        let (x, y) = classification_data(300, 4);
        let result = AutoClassifier::new()
            .dataset(x, y)
            .num_classes(2)
            .evaluation(Evaluation::StratifiedKFold { k: 3 })
            .epochs(6)
            .trials(3)
            .seed(4)
            .fit()
            .unwrap();
        // Cross-validated accuracy should still be high on a separable problem.
        assert!(
            result.best_score.unwrap() > 80.0,
            "cv accuracy {:?}",
            result.best_score
        );
        // The best trial reported one metric per fold (3 folds).
        let best = result.study.best_trial().unwrap().unwrap();
        assert_eq!(best.intermediate.len(), 3);
    }
}
