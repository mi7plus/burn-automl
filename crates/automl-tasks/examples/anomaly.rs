//! Anomaly detection with `AutoAnomaly`: a k-NN distance scorer whose neighbour
//! count and threshold are searched to maximize F1 on labelled evaluation data.
//!
//! ```bash
//! cargo run -p automl-tasks --release --example anomaly
//! ```

use automl_tasks::AutoAnomaly;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    // Reference: normal points clustered near the origin.
    let reference: Vec<Vec<f32>> = (0..120)
        .map(|_| vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)])
        .collect();
    // Evaluation: mostly normal, plus far-away anomalies (label 1).
    let (mut eval, mut labels) = (Vec::new(), Vec::new());
    for i in 0..60 {
        if i % 4 == 0 {
            eval.push(vec![rng.gen_range(4.0..6.0), rng.gen_range(4.0..6.0)]);
            labels.push(1);
        } else {
            eval.push(vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
            labels.push(0);
        }
    }
    let study = AutoAnomaly::new(reference, eval, labels)
        .trials(8)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best F1: {:.3}",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("f1")
            .unwrap_or(f64::NAN)
    );
}
