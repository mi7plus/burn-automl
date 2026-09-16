//! Speech / audio classification and the `AutoAudio` API (PRD §12, §20).
//!
//! Waveforms are turned into log-magnitude spectrogram frames (a windowed DFT
//! with a Hann window), which are exactly the fixed-length multivariate
//! sequences the recurrent [`AutoSequence`](crate::AutoSequence) classifier
//! consumes — so audio classification reuses the sequence model over a
//! spectral front-end. The window/hop/bin front-end parameters are configurable
//! here; searching them jointly with the model can layer on later (§12).

use crate::AutoSequence;
use automl_core::error::{Error, Result as CoreResult};
use automl_core::prelude::Study;
use std::f32::consts::PI;

/// Compute log-magnitude spectrogram frames from a waveform: for each window of
/// `window` samples (stepped by `hop`), the magnitude at the first `n_bins`
/// DFT frequencies, Hann-windowed. Returns `frames × n_bins`.
pub fn spectrogram(wave: &[f32], window: usize, hop: usize, n_bins: usize) -> Vec<Vec<f32>> {
    let window = window.max(2);
    let hop = hop.max(1);
    let n_bins = n_bins.clamp(1, window / 2);
    // Precompute the Hann window.
    let hann: Vec<f32> = (0..window)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (window - 1) as f32).cos())
        .collect();

    let mut frames = Vec::new();
    let mut start = 0;
    while start + window <= wave.len() {
        let seg = &wave[start..start + window];
        let mut bins = vec![0f32; n_bins];
        for (k, bin) in bins.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, &x) in seg.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * n as f32 / window as f32;
                let xw = x * hann[n];
                re += xw * angle.cos();
                im += xw * angle.sin();
            }
            *bin = (re * re + im * im).sqrt().ln_1p();
        }
        frames.push(bins);
        start += hop;
    }
    frames
}

/// One-call audio classification search (PRD §12, §20).
///
/// Each `waveforms[i]` is a raw mono sample vector; a spectral front-end turns
/// it into sequence frames that a recurrent model classifies.
pub struct AutoAudio {
    waveforms: Vec<Vec<f32>>,
    labels: Vec<i64>,
    window: usize,
    hop: usize,
    n_bins: usize,
    num_classes: usize,
    epochs: usize,
    trials: u64,
    seed: u64,
}

impl AutoAudio {
    /// A new audio classification search with default front-end (window 32,
    /// hop 16, 16 bins).
    pub fn new(waveforms: Vec<Vec<f32>>, labels: Vec<i64>) -> Self {
        AutoAudio {
            waveforms,
            labels,
            window: 32,
            hop: 16,
            n_bins: 16,
            num_classes: 0,
            epochs: 8,
            trials: 10,
            seed: 0,
        }
    }

    /// Set the spectral front-end: DFT `window`, `hop`, and number of frequency
    /// `bins`.
    pub fn frontend(mut self, window: usize, hop: usize, n_bins: usize) -> Self {
        self.window = window;
        self.hop = hop;
        self.n_bins = n_bins;
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
        if self.waveforms.is_empty() || self.waveforms.len() != self.labels.len() {
            return Err(Error::Objective(
                "audio data and labels must match and be non-empty".into(),
            ));
        }
        // Front-end: each waveform becomes a spectrogram sequence.
        let sequences: Vec<Vec<Vec<f32>>> = self
            .waveforms
            .iter()
            .map(|w| spectrogram(w, self.window, self.hop, self.n_bins))
            .collect();
        if sequences.iter().any(|s| s.is_empty()) {
            return Err(Error::Objective(
                "waveforms are too short for the configured window".into(),
            ));
        }

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
    fn spectrogram_peaks_at_the_tone_frequency() {
        // A pure tone should concentrate energy in one bin.
        let n = 256;
        let freq_bin = 4.0;
        let wave: Vec<f32> = (0..n)
            .map(|t| (2.0 * PI * freq_bin * t as f32 / 32.0).sin())
            .collect();
        let frames = spectrogram(&wave, 32, 16, 16);
        assert!(!frames.is_empty());
        // The dominant bin should be near freq_bin in every frame.
        let peak = frames[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak as i32 - freq_bin as i32).abs() <= 1,
            "peak bin {peak}"
        );
    }

    /// Two classes of tones at distinct frequencies; the spectral front-end plus
    /// a recurrent classifier separate them.
    fn synthetic(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<i64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut waves, mut labels) = (Vec::new(), Vec::new());
        for i in 0..n {
            let low = i % 2 == 0;
            let f = if low { 3.0 } else { 9.0 };
            let phase: f32 = rng.gen_range(0.0..PI);
            let wave: Vec<f32> = (0..160)
                .map(|t| {
                    (2.0 * PI * f * t as f32 / 32.0 + phase).sin() + rng.gen_range(-0.05..0.05)
                })
                .collect();
            waves.push(wave);
            labels.push(if low { 0 } else { 1 });
        }
        (waves, labels)
    }

    #[test]
    fn auto_audio_classifies_tones() {
        let (waves, labels) = synthetic(120, 1);
        let study = AutoAudio::new(waves, labels)
            .num_classes(2)
            .epochs(6)
            .trials(2)
            .seed(1)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let acc = best.final_value("accuracy").unwrap();
        assert!(acc > 80.0, "best audio accuracy was {acc}");
    }

    #[test]
    fn empty_input_errors() {
        assert!(AutoAudio::new(Vec::new(), Vec::new()).fit().is_err());
    }
}
