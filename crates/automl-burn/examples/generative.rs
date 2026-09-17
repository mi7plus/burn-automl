//! Generative helpers: `AutoAutoencoder` searches a reconstruction autoencoder
//! over low-rank data, and `DiffusionSchedule` builds noise schedules.
//!
//! ```bash
//! cargo run -p automl-burn --release --example generative
//! ```

use automl_burn::{AutoAutoencoder, DiffusionSchedule};
use rand::{Rng, SeedableRng};

fn main() {
    // Data on a 1-D line embedded in 4-D: a tiny latent reconstructs it well.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let dirs = [1.0f32, -0.5, 0.25, 0.75];
    let data: Vec<Vec<f32>> = (0..160)
        .map(|_| {
            let t: f32 = rng.gen_range(-1.0..1.0);
            dirs.iter()
                .map(|d| d * t + rng.gen_range(-0.02..0.02))
                .collect()
        })
        .collect();
    let study = AutoAutoencoder::new(data, 4)
        .epochs(40)
        .trials(4)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "autoencoder: best reconstruction MSE {:.4}",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("mse")
            .unwrap_or(f64::NAN)
    );

    let sched = DiffusionSchedule::cosine(10);
    println!(
        "diffusion cosine schedule ({} steps): alpha_bar[0]={:.3} .. alpha_bar[last]={:.3}",
        sched.steps(),
        sched.alpha_bars[0],
        sched.alpha_bars.last().unwrap()
    );
}
