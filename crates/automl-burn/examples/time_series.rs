//! Time-series forecasting with `AutoForecaster`: a noisy trend + seasonality,
//! evaluated by time-aware backtesting (no future leakage). Scored by RMSE.
//!
//! ```bash
//! cargo run -p automl-burn --release --example time_series
//! ```

use automl_burn::{AutoForecaster, TimeSplit};
use std::f32::consts::PI;

fn main() {
    // 200 points: linear trend + a weekly cycle + mild noise.
    let series: Vec<f32> = (0..200)
        .map(|t| {
            let trend = t as f32 * 0.05;
            let season = (2.0 * PI * t as f32 / 7.0).sin();
            let noise = ((t * 31 % 13) as f32 / 13.0 - 0.5) * 0.2;
            trend + season + noise
        })
        .collect();
    let study = AutoForecaster::new(series, TimeSplit::expanding(120, 10))
        .epochs(12)
        .trials(4)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best backtest RMSE: {:.4}",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("rmse")
            .unwrap_or(f64::NAN)
    );
}
