//! Time-series forecasting: temporal backtesting, forecasting metrics, and the
//! `AutoForecaster` high-level API (PRD §8, §20).
//!
//! Two invariants the plan is emphatic about:
//!
//! - **Backtesting must be time-aware.** Random cross-validation is rejected for
//!   forecasting at the adapter level, not left to user discipline (§8). Splits
//!   here always place the validation window *after* the training window.
//! - **Temporal leakage fails loudly.** A fold that leaks future data into
//!   training is an assertion failure, not a silent optimistic metric (§8) — see
//!   [`TimeSplit::folds`], which panics if any train index is not strictly
//!   before every test index.
//!
//! [`AutoForecaster`] turns a univariate series into a supervised
//! lagged-window regression problem and searches the window size and MLP
//! hyperparameters, evaluating each configuration by backtesting.

use crate::{MlpConfig, TrainBackend, TrainConfig};
use automl_core::error::{Error, Result};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{Distribution, MedianPruner, SearchSpace, Study, TpeSampler};

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

// ------------------------------ backtesting ----------------------------------

/// A time-aware backtesting scheme producing `(train, test)` index ranges where
/// the test window always follows the training window (§8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSplit {
    /// Expanding window: training grows from the start each fold; the test
    /// window slides forward by `horizon` each time.
    Expanding {
        /// Initial training length before the first test window.
        initial: usize,
        /// Length of each test window (also the step between folds).
        horizon: usize,
    },
    /// Sliding window: a fixed-length training window slides forward.
    Sliding {
        /// Training window length.
        window: usize,
        /// Length of each test window (also the step between folds).
        horizon: usize,
    },
}

impl TimeSplit {
    /// An expanding-window backtest.
    pub fn expanding(initial: usize, horizon: usize) -> Self {
        TimeSplit::Expanding {
            initial: initial.max(1),
            horizon: horizon.max(1),
        }
    }

    /// A sliding-window backtest.
    pub fn sliding(window: usize, horizon: usize) -> Self {
        TimeSplit::Sliding {
            window: window.max(1),
            horizon: horizon.max(1),
        }
    }

    /// Generate `(train_range, test_range)` folds over `n` time-ordered points.
    ///
    /// Each range is a half-open `start..end`. Panics if any fold would place a
    /// training index at or after a test index — the loud temporal-leakage
    /// assertion the plan requires (§8).
    pub fn folds(&self, n: usize) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let mut folds = Vec::new();
        match *self {
            TimeSplit::Expanding { initial, horizon } => {
                let mut test_start = initial;
                while test_start + horizon <= n {
                    folds.push((0..test_start, test_start..test_start + horizon));
                    test_start += horizon;
                }
            }
            TimeSplit::Sliding { window, horizon } => {
                let mut test_start = window;
                while test_start + horizon <= n {
                    folds.push((
                        test_start - window..test_start,
                        test_start..test_start + horizon,
                    ));
                    test_start += horizon;
                }
            }
        }
        // Loud leakage assertion: every training index precedes every test index.
        for (train, test) in &folds {
            assert!(
                train.end <= test.start,
                "temporal leakage: train {train:?} overlaps or follows test {test:?}"
            );
        }
        folds
    }
}

// ------------------------------ metrics --------------------------------------

/// Mean absolute error.
pub fn mae(actual: &[f32], predicted: &[f32]) -> f32 {
    mean(actual.iter().zip(predicted).map(|(a, p)| (a - p).abs()))
}

/// Root mean squared error.
pub fn rmse(actual: &[f32], predicted: &[f32]) -> f32 {
    mean(actual.iter().zip(predicted).map(|(a, p)| (a - p).powi(2))).sqrt()
}

/// Symmetric mean absolute percentage error, in `[0, 200]` (percent).
pub fn smape(actual: &[f32], predicted: &[f32]) -> f32 {
    mean(actual.iter().zip(predicted).map(|(a, p)| {
        let denom = a.abs() + p.abs();
        if denom == 0.0 {
            0.0
        } else {
            200.0 * (a - p).abs() / denom
        }
    }))
}

/// Mean absolute scaled error: MAE scaled by the in-sample MAE of a naive
/// one-step forecast over `train`. `< 1` beats the naive forecast.
pub fn mase(train: &[f32], actual: &[f32], predicted: &[f32]) -> f32 {
    let naive = mean(train.windows(2).map(|w| (w[1] - w[0]).abs()));
    if naive == 0.0 {
        return f32::INFINITY;
    }
    mae(actual, predicted) / naive
}

/// Pinball (quantile) loss at quantile `q` in `(0, 1)`.
pub fn pinball(actual: &[f32], predicted: &[f32], q: f32) -> f32 {
    mean(actual.iter().zip(predicted).map(|(a, p)| {
        let e = a - p;
        if e >= 0.0 {
            q * e
        } else {
            (q - 1.0) * e
        }
    }))
}

fn mean<I: Iterator<Item = f32>>(it: I) -> f32 {
    let mut sum = 0.0;
    let mut n = 0u32;
    for x in it {
        sum += x;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

// ---------------------------- windowing --------------------------------------

/// Build supervised `(lagged_window -> next_value)` samples from a slice of a
/// series: sample `i` has features `series[i..i+window]` and target
/// `series[i+window]`.
fn windows(series: &[f32], window: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    if series.len() > window {
        for i in 0..series.len() - window {
            x.push(series[i..i + window].to_vec());
            y.push(series[i + window]);
        }
    }
    (x, y)
}

// ---------------------------- AutoForecaster ---------------------------------

/// One-call forecasting search over a univariate series (PRD §20).
///
/// The series is turned into lagged-window regression samples and evaluated by
/// time-aware backtesting; the search covers the lookback `window` and the MLP
/// hyperparameters. Lower RMSE is better.
pub struct AutoForecaster {
    series: Vec<f32>,
    split: TimeSplit,
    epochs: usize,
    batch_size: usize,
    trials: u64,
    max_window: usize,
    seed: u64,
}

impl AutoForecaster {
    /// A new forecaster over `series` with the given backtesting scheme.
    pub fn new(series: Vec<f32>, split: TimeSplit) -> Self {
        AutoForecaster {
            series,
            split,
            epochs: 40,
            batch_size: 32,
            trials: 20,
            max_window: 24,
            seed: 0,
        }
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

    /// Largest lookback window the search may use.
    pub fn max_window(mut self, w: usize) -> Self {
        self.max_window = w.max(1);
        self
    }

    /// Seed for the search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (with the best RMSE configuration).
    pub fn fit(self) -> Result<Study> {
        if self.series.len() < 8 {
            return Err(Error::Objective("forecaster needs a longer series".into()));
        }
        let space = SearchSpace::new()
            .add("window", Distribution::int(2, self.max_window as i64))
            .add("lr", Distribution::log_float(1e-4, 1e-1))
            .add("hidden_size", Distribution::int(16, 128))
            .add("num_layers", Distribution::int(1, 3))
            .add("dropout", Distribution::float(0.0, 0.3));

        let mut study = Study::builder(space)
            .name("auto-forecaster")
            .minimize("rmse")
            .sampler(TpeSampler::new("rmse", Direction::Minimize, self.seed))
            .pruner(MedianPruner::new("rmse", Direction::Minimize).with_warmup_steps(1))
            .seed(self.seed)
            .build()?;

        let series = Arc::new(self.series);
        let split = self.split;
        let (epochs, batch) = (self.epochs, self.batch_size);

        let objective = move |p: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let window = p.int("window")? as usize;
            let cfg = TrainConfig {
                epochs,
                batch_size: batch,
                lr: p.float("lr")?,
                seed: 42,
                model: MlpConfig::new(p.int("hidden_size")? as usize)
                    .with_input_dim(window)
                    .with_num_classes(1)
                    .with_num_hidden_layers(p.int("num_layers")? as usize)
                    .with_dropout(p.float("dropout")?),
            };
            let rmse = backtest::<TrainBackend>(&series, split, window, &cfg, sink);
            Ok(NamedMetrics::single("rmse", rmse as f64))
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

/// Backtest a windowed-MLP forecaster: for each fold, train on the training
/// window and evaluate one-step forecasts on the test window, reporting the
/// running-mean RMSE per fold and returning the mean.
fn backtest<B: AutodiffBackend>(
    series: &[f32],
    split: TimeSplit,
    window: usize,
    cfg: &TrainConfig,
    sink: &mut dyn ReportSink,
) -> f32 {
    let folds = split.folds(series.len());
    if folds.is_empty() {
        return f32::INFINITY;
    }
    let mut scores = Vec::new();
    for (fold, (train_range, test_range)) in folds.iter().enumerate() {
        // Build windowed samples strictly within each range (no cross-range
        // leakage: test features come from the test range only).
        let (train_x, train_y) = windows(&series[train_range.clone()], window);
        let (test_x, test_y) = windows(&series[test_range.clone()], window);
        if train_x.is_empty() || test_x.is_empty() {
            continue;
        }
        let rmse_fold = train_and_score::<B>(cfg, &train_x, &train_y, &test_x, &test_y);
        scores.push(rmse_fold);
        let running = mean(scores.iter().copied());
        let _ = sink.report(
            fold as u64 + 1,
            NamedMetrics::single("rmse", running as f64),
        );
        if sink.should_stop() {
            break;
        }
    }
    if scores.is_empty() {
        f32::INFINITY
    } else {
        mean(scores.iter().copied())
    }
}

fn train_and_score<B: AutodiffBackend>(
    cfg: &TrainConfig,
    train_x: &[Vec<f32>],
    train_y: &[f32],
    test_x: &[Vec<f32>],
    test_y: &[f32],
) -> f32 {
    let device = Default::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut model = cfg.model.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let mut order: Vec<usize> = (0..train_x.len()).collect();

    for _ in 0..cfg.epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let feats: Vec<Vec<f32>> = chunk.iter().map(|&i| train_x[i].clone()).collect();
            let targets: Vec<f32> = chunk.iter().map(|&i| train_y[i]).collect();
            let x = feature_tensor::<B>(&feats, &device);
            let y = target_tensor::<B>(&targets, &device);
            let out = model.forward_flat(x);
            let loss = MseLoss::new().forward(out, y, Reduction::Mean);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
    }

    // One-step forecasts on the test windows. The inner backend shares the
    // device type with the autodiff backend, so `device` validates directly.
    let valid = model.valid();
    let x = feature_tensor::<B::InnerBackend>(test_x, &device);
    let preds: Vec<f32> = valid
        .forward_flat(x)
        .squeeze_dim::<1>(1)
        .into_data()
        .to_vec()
        .expect("f32 predictions");
    rmse(test_y, &preds)
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

fn target_tensor<B: Backend>(targets: &[f32], device: &B::Device) -> Tensor<B, 2> {
    let n = targets.len();
    Tensor::<B, 2>::from_data(
        TensorData::new(targets.to_vec(), [n, 1]).convert::<B::FloatElem>(),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expanding_split_is_time_ordered_and_leak_free() {
        let folds = TimeSplit::expanding(10, 5).folds(30);
        assert!(!folds.is_empty());
        for (train, test) in &folds {
            assert_eq!(train.start, 0, "expanding window starts at 0");
            assert!(train.end <= test.start, "no leakage");
            assert_eq!(test.end - test.start, 5, "horizon length");
        }
        // Training windows grow.
        assert!(folds.last().unwrap().0.end > folds[0].0.end);
    }

    #[test]
    fn sliding_split_has_fixed_window() {
        let folds = TimeSplit::sliding(8, 4).folds(30);
        for (train, test) in &folds {
            assert_eq!(train.end - train.start, 8, "fixed window");
            assert!(train.end <= test.start);
        }
    }

    #[test]
    fn forecasting_metrics_are_sane() {
        let actual = [1.0f32, 2.0, 3.0, 4.0];
        let perfect = actual;
        assert!(mae(&actual, &perfect).abs() < 1e-6);
        assert!(rmse(&actual, &perfect).abs() < 1e-6);
        assert!(smape(&actual, &perfect).abs() < 1e-6);

        let off_by_one = [2.0f32, 3.0, 4.0, 5.0];
        assert!((mae(&actual, &off_by_one) - 1.0).abs() < 1e-6);
        // Naive one-step MAE over a linear ramp train is 1.0, so MASE == MAE.
        let train = [0.0f32, 1.0, 2.0, 3.0];
        assert!((mase(&train, &actual, &off_by_one) - 1.0).abs() < 1e-6);
        // Pinball at q=0.5 is half the MAE.
        assert!((pinball(&actual, &off_by_one, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn windows_builds_lagged_samples() {
        let (x, y) = windows(&[1.0, 2.0, 3.0, 4.0, 5.0], 2);
        assert_eq!(x, vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]]);
        assert_eq!(y, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains a model; run with --features slow-tests"
    )]
    fn auto_forecaster_learns_a_trend() {
        // A smooth seasonal-ish series: sine plus slow linear trend.
        let series: Vec<f32> = (0..160)
            .map(|t| (t as f32 * 0.3).sin() * 3.0 + t as f32 * 0.05)
            .collect();
        let study = AutoForecaster::new(series, TimeSplit::expanding(80, 20))
            .epochs(30)
            .trials(3)
            .max_window(12)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let rmse = best.final_value("rmse").unwrap();
        // A naive last-value forecast on this series has RMSE well above 1;
        // the learned model should do clearly better.
        assert!(
            rmse < 1.5,
            "best backtest RMSE was {rmse}, expected a good fit"
        );
    }
}
