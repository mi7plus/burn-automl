//! Unsupervised clustering with `AutoCluster`: K-means over well-separated 2-D
//! blobs, searching the number of clusters to maximize silhouette.
//!
//! ```bash
//! cargo run -p automl-tasks --release --example clustering
//! ```

use automl_tasks::AutoCluster;
use rand::{Rng, SeedableRng};

fn main() {
    // Three Gaussian blobs.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let centers = [[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]];
    let data: Vec<Vec<f32>> = (0..180)
        .map(|i| {
            let c = centers[i % 3];
            vec![
                c[0] + rng.gen_range(-0.6..0.6),
                c[1] + rng.gen_range(-0.6..0.6),
            ]
        })
        .collect();
    let study = AutoCluster::new(data, 6).trials(6).seed(1).fit().unwrap();
    let best = study.best_trial().unwrap().unwrap();
    println!(
        "best k = {} with silhouette {:.3}",
        best.params.int("k").unwrap_or(0),
        best.final_value("silhouette").unwrap_or(f64::NAN),
    );
}
