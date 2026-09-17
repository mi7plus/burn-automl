//! Video classification with `AutoVideo`: 1x6x6 clips where a bright block slides
//! left-to-right (class 0) or right-to-left (class 1). A spatial-pool front-end
//! feeds the recurrent classifier, which reads the direction of travel.
//!
//! ```bash
//! cargo run -p automl-burn --release --example video
//! ```

use automl_burn::AutoVideo;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut clips, mut labels) = (Vec::new(), Vec::new());
    let (h, w, frames) = (6usize, 6usize, 6usize);
    for i in 0..120 {
        let rightward = i % 2 == 0;
        let mut clip = Vec::new();
        for t in 0..frames {
            let col = if rightward { t } else { frames - 1 - t }.min(w - 1);
            let mut frame = vec![0f32; h * w];
            for r in 0..h {
                frame[r * w + col] = 1.0 + rng.gen_range(-0.05..0.05);
            }
            clip.push(frame);
        }
        clips.push(clip);
        labels.push(if rightward { 0 } else { 1 });
    }
    let study = AutoVideo::new(clips, labels, 1, 6, 6)
        .num_classes(2)
        .grid(3)
        .epochs(6)
        .trials(2)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best validation accuracy: {:.1}%",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("accuracy")
            .unwrap_or(0.0)
    );
}
