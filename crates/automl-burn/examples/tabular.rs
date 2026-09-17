//! Supervised tabular AutoML: `AutoClassifier` and `AutoRegressor` on synthetic
//! in-memory data (no download). Searches MLP hyperparameters with TPE + median
//! pruning.
//!
//! ```bash
//! cargo run -p automl-burn --release --example tabular
//! ```

use automl_burn::{AutoClassifier, AutoRegressor};
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

    // Classification: two Gaussian blobs, label = which blob.
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        let c = rng.gen_range(0..2);
        let mu = if c == 0 { -1.5 } else { 1.5 };
        xs.push(vec![
            mu + rng.gen_range(-1.0..1.0),
            mu + rng.gen_range(-1.0..1.0),
        ]);
        ys.push(c);
    }
    let result = AutoClassifier::new()
        .dataset(xs, ys)
        .num_classes(2)
        .epochs(8)
        .trials(5)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "classification: best accuracy {:.1}%  (lr={:.4}, hidden={})",
        result.best_score.unwrap_or(0.0),
        result
            .best_params
            .as_ref()
            .and_then(|p| p.float("lr").ok())
            .unwrap_or(0.0),
        result
            .best_params
            .as_ref()
            .and_then(|p| p.int("hidden_size").ok())
            .unwrap_or(0),
    );

    // Regression: y = 2*a - b + 0.5.
    let (mut xr, mut yr) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        let (a, b): (f32, f32) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        xr.push(vec![a, b]);
        yr.push(2.0 * a - b + 0.5);
    }
    let reg = AutoRegressor::new()
        .dataset(xr, yr)
        .epochs(12)
        .trials(5)
        .seed(2)
        .fit()
        .unwrap();
    println!(
        "regression:     best RMSE {:.4}",
        reg.best_score.unwrap_or(f64::NAN)
    );
}
