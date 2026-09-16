//! Video classification and the `AutoVideo` API (PRD §11, §20; roadmap v0.7).
//!
//! A clip is a sequence of frames. A fixed spatial front-end average-pools each
//! frame's channels onto a small `grid × grid` map — a cheap, deterministic
//! feature extractor — turning the clip into exactly the fixed-length
//! multivariate sequence the recurrent [`AutoSequence`]
//! classifier consumes. So video reuses the CNN+RNN pattern (§11 "CNN+RNN"
//! candidate family) over a pooled spatial front-end, the same way audio reuses
//! the sequence model over a spectral front-end.
//!
//! Video is gated behind mature pruning, checkpoint reuse and scheduling (§11);
//! those primitives landed in v0.1–v0.7 (median/ASHA pruning, artifact storage,
//! [`automl_core::checkpoint`]), so a high-level API is now appropriate. The
//! frame stride and pooling grid are the searchable front-end knobs; searching
//! them jointly with the model can layer on later.

use crate::AutoSequence;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::prelude::Study;

/// Average-pool one flat `channels × height × width` frame onto a
/// `grid × grid` map per channel, returning a `channels * grid * grid` feature
/// vector. Cells split the frame as evenly as possible; a `grid` larger than a
/// spatial dimension collapses to that dimension.
pub fn frame_pool(
    frame: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    grid: usize,
) -> Vec<f32> {
    let grid = grid.max(1);
    let gh = grid.min(height.max(1));
    let gw = grid.min(width.max(1));
    let mut out = Vec::with_capacity(channels * gh * gw);
    for c in 0..channels {
        let base = c * height * width;
        for cell_r in 0..gh {
            // Even split of [0,height) into gh cells.
            let r0 = cell_r * height / gh;
            let r1 = ((cell_r + 1) * height / gh).max(r0 + 1).min(height);
            for cell_c in 0..gw {
                let c0 = cell_c * width / gw;
                let c1 = ((cell_c + 1) * width / gw).max(c0 + 1).min(width);
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for r in r0..r1 {
                    for col in c0..c1 {
                        sum += frame[base + r * width + col];
                        count += 1;
                    }
                }
                out.push(if count > 0 { sum / count as f32 } else { 0.0 });
            }
        }
    }
    out
}

/// One-call video classification search (PRD §11, §20).
///
/// `clips[i]` is a clip: a vector of frames, each frame a flat
/// `channels * height * width` row-major pixel vector. Every frame in every clip
/// must have the same shape.
pub struct AutoVideo {
    clips: Vec<Vec<Vec<f32>>>,
    labels: Vec<i64>,
    channels: usize,
    height: usize,
    width: usize,
    grid: usize,
    frame_stride: usize,
    num_classes: usize,
    epochs: usize,
    trials: u64,
    seed: u64,
}

impl AutoVideo {
    /// A new video classification search with a default `4 × 4` spatial pool and
    /// no frame subsampling.
    pub fn new(
        clips: Vec<Vec<Vec<f32>>>,
        labels: Vec<i64>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        AutoVideo {
            clips,
            labels,
            channels,
            height,
            width,
            grid: 4,
            frame_stride: 1,
            num_classes: 0,
            epochs: 8,
            trials: 10,
            seed: 0,
        }
    }

    /// Set the spatial pooling grid (`grid × grid` cells per channel per frame).
    pub fn grid(mut self, grid: usize) -> Self {
        self.grid = grid.max(1);
        self
    }

    /// Keep every `stride`-th frame (temporal subsampling / frame rate).
    pub fn frame_stride(mut self, stride: usize) -> Self {
        self.frame_stride = stride.max(1);
        self
    }

    /// Number of classes (inferred as `max(label)+1` if unset).
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

    /// Seed for the split and search.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (maximizing validation accuracy).
    pub fn fit(self) -> CoreResult<Study> {
        let pixels = self.channels * self.height * self.width;
        if self.clips.is_empty() || self.clips.len() != self.labels.len() {
            return Err(Error::Objective(
                "video clips and labels must match and be non-empty".into(),
            ));
        }
        if self
            .clips
            .iter()
            .any(|clip| clip.is_empty() || clip.iter().any(|f| f.len() != pixels))
        {
            return Err(Error::Objective(
                "every clip must be non-empty and its frames channels*height*width pixels".into(),
            ));
        }

        // Front-end: each clip becomes a sequence of pooled frame features.
        let sequences: Vec<Vec<Vec<f32>>> = self
            .clips
            .iter()
            .map(|clip| {
                clip.iter()
                    .step_by(self.frame_stride)
                    .map(|frame| {
                        frame_pool(frame, self.channels, self.height, self.width, self.grid)
                    })
                    .collect()
            })
            .collect();

        let mut search = AutoSequence::new(sequences, self.labels)
            .epochs(self.epochs)
            .trials(self.trials)
            .seed(self.seed);
        if self.num_classes > 0 {
            search = search.num_classes(self.num_classes);
        }
        search.fit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_pool_averages_quadrants() {
        // A 1x2x2 frame: pooling to 2x2 is the identity; to 1x1 is the mean.
        let frame = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(frame_pool(&frame, 1, 2, 2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(frame_pool(&frame, 1, 2, 2, 1), vec![2.5]);
    }

    /// Two classes of 1x6x6 clips: a bright block slides left-to-right (class 0)
    /// or right-to-left (class 1). Spatial pooling locates the block per frame;
    /// the recurrent model reads the direction of travel.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<Vec<f32>>>, Vec<i64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut clips, mut labels) = (Vec::new(), Vec::new());
        let (h, w, frames) = (6usize, 6usize, 6usize);
        for i in 0..n {
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
        (clips, labels)
    }

    #[test]
    fn auto_video_classifies_motion_direction() {
        let (clips, labels) = synthetic(120, 1);
        let study = AutoVideo::new(clips, labels, 1, 6, 6)
            .num_classes(2)
            .grid(3)
            .epochs(6)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(acc > 66.0, "best video accuracy was {acc}");
    }

    #[test]
    fn empty_input_errors() {
        assert!(AutoVideo::new(Vec::new(), Vec::new(), 1, 4, 4)
            .fit()
            .is_err());
    }
}
