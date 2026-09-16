//! Multimodal fusion search primitives (roadmap v0.9; PRD §15).
//!
//! Each modality encoder is a searchable component; a multimodal study searches
//! the per-modality encoder and projection dimension plus a fusion strategy and
//! whether encoders are shared or independent (§15). The load-bearing rule is
//! that **dimension/memory constraints are validated before scheduling** — a
//! rejected configuration must never reach the executor (§15). This module builds
//! the fusion search space, decodes a sampled configuration to a
//! [`MultimodalPlan`], and validates it *up front* via [`MultimodalPlan::validate`]
//! so the objective can reject an incompatible plan before any training starts.

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::ParamSet;
use crate::space::SearchSpace;
use serde::{Deserialize, Serialize};

/// How per-modality feature vectors are combined into a joint representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fusion {
    /// Concatenate all modality vectors. Works for any per-modality dimensions.
    Concat,
    /// Element-wise mean. Requires every modality to share one dimension.
    Mean,
    /// Element-wise sum. Requires every modality to share one dimension.
    Sum,
}

impl Fusion {
    /// Parse a categorical label into a fusion strategy.
    pub fn from_label(s: &str) -> Option<Fusion> {
        match s {
            "concat" => Some(Fusion::Concat),
            "mean" => Some(Fusion::Mean),
            "sum" => Some(Fusion::Sum),
            _ => None,
        }
    }

    /// Whether this strategy requires all modality dimensions to be equal.
    pub fn requires_equal_dims(&self) -> bool {
        matches!(self, Fusion::Mean | Fusion::Sum)
    }
}

/// A modality: a name plus its candidate encoders. Its projection dimension is
/// searched within `[min_dim, max_dim]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Modality {
    name: String,
    encoders: Vec<String>,
    min_dim: usize,
    max_dim: usize,
}

impl Modality {
    /// A modality named `name` with candidate `encoders`, projecting to a
    /// searched dimension in `[min_dim, max_dim]`.
    pub fn new<I, S>(name: impl Into<String>, encoders: I, min_dim: usize, max_dim: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let min_dim = min_dim.max(1);
        Modality {
            name: name.into(),
            encoders: encoders.into_iter().map(Into::into).collect(),
            min_dim,
            max_dim: max_dim.max(min_dim),
        }
    }

    /// The modality name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A searchable multimodal configuration: several modalities plus a fusion
/// strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalSpace {
    modalities: Vec<Modality>,
    fusions: Vec<Fusion>,
}

impl MultimodalSpace {
    /// A new multimodal space over `modalities`, offering all fusion strategies.
    pub fn new(modalities: Vec<Modality>) -> Self {
        MultimodalSpace {
            modalities,
            fusions: vec![Fusion::Concat, Fusion::Mean, Fusion::Sum],
        }
    }

    /// Restrict the candidate fusion strategies.
    pub fn with_fusions(mut self, fusions: Vec<Fusion>) -> Self {
        if !fusions.is_empty() {
            self.fusions = fusions;
        }
        self
    }

    /// Lower to a flat [`SearchSpace`]: per modality an `"{mod}.encoder"`
    /// categorical and a `"{mod}.dim"` integer, plus a top-level `"fusion"`
    /// categorical.
    pub fn to_search_space(&self) -> SearchSpace {
        let mut space = SearchSpace::new();
        for m in &self.modalities {
            space = space
                .add(
                    format!("{}.encoder", m.name),
                    Distribution::categorical(m.encoders.iter().cloned()),
                )
                .add(
                    format!("{}.dim", m.name),
                    Distribution::int(m.min_dim as i64, m.max_dim as i64),
                );
        }
        let labels: Vec<String> = self
            .fusions
            .iter()
            .map(|f| fusion_label(*f).into())
            .collect();
        space.add("fusion", Distribution::categorical(labels))
    }

    /// Decode a sampled [`ParamSet`] into a [`MultimodalPlan`].
    pub fn decode(&self, params: &ParamSet) -> Result<MultimodalPlan> {
        let mut encoders = Vec::with_capacity(self.modalities.len());
        for m in &self.modalities {
            let encoder = params
                .categorical(&format!("{}.encoder", m.name))?
                .to_string();
            let dim = params.int(&format!("{}.dim", m.name))? as usize;
            encoders.push(ModalityChoice {
                name: m.name.clone(),
                encoder,
                dim,
            });
        }
        let fusion = Fusion::from_label(params.categorical("fusion")?)
            .ok_or_else(|| Error::Objective("unknown fusion strategy".into()))?;
        Ok(MultimodalPlan { encoders, fusion })
    }
}

fn fusion_label(f: Fusion) -> &'static str {
    match f {
        Fusion::Concat => "concat",
        Fusion::Mean => "mean",
        Fusion::Sum => "sum",
    }
}

/// The chosen encoder and projection dimension for one modality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityChoice {
    /// The modality name.
    pub name: String,
    /// The selected encoder.
    pub encoder: String,
    /// The projection dimension.
    pub dim: usize,
}

/// A decoded multimodal configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPlan {
    /// One choice per modality.
    pub encoders: Vec<ModalityChoice>,
    /// How the projected modality features are fused.
    pub fusion: Fusion,
}

impl MultimodalPlan {
    /// Validate dimension compatibility **before scheduling** (§15): element-wise
    /// fusions require every modality to project to the same dimension. Returns an
    /// actionable error naming the offending dimensions so the objective can
    /// reject the configuration without training.
    pub fn validate(&self) -> Result<()> {
        if self.encoders.is_empty() {
            return Err(Error::Objective(
                "a multimodal plan needs ≥1 modality".into(),
            ));
        }
        if self.fusion.requires_equal_dims() {
            let d0 = self.encoders[0].dim;
            if let Some(bad) = self.encoders.iter().find(|m| m.dim != d0) {
                return Err(Error::Objective(format!(
                    "{:?} fusion needs equal modality dims, but '{}' is {} and '{}' is {}",
                    self.fusion, self.encoders[0].name, d0, bad.name, bad.dim
                )));
            }
        }
        Ok(())
    }

    /// The dimension of the fused representation, assuming [`Self::validate`]
    /// passed.
    pub fn fused_dim(&self) -> usize {
        match self.fusion {
            Fusion::Concat => self.encoders.iter().map(|m| m.dim).sum(),
            Fusion::Mean | Fusion::Sum => self.encoders.first().map(|m| m.dim).unwrap_or(0),
        }
    }
}

/// Fuse per-modality feature vectors according to `fusion`. Element-wise
/// strategies error on a length mismatch (the same constraint
/// [`MultimodalPlan::validate`] checks up front).
pub fn fuse(features: &[Vec<f32>], fusion: Fusion) -> Result<Vec<f32>> {
    if features.is_empty() {
        return Err(Error::Objective("nothing to fuse".into()));
    }
    match fusion {
        Fusion::Concat => Ok(features.iter().flat_map(|f| f.iter().copied()).collect()),
        Fusion::Mean | Fusion::Sum => {
            let d = features[0].len();
            if features.iter().any(|f| f.len() != d) {
                return Err(Error::Objective(
                    "element-wise fusion requires equal-length modality vectors".into(),
                ));
            }
            let mut out = vec![0f32; d];
            for f in features {
                for (o, v) in out.iter_mut().zip(f) {
                    *o += v;
                }
            }
            if matches!(fusion, Fusion::Mean) {
                let n = features.len() as f32;
                out.iter_mut().for_each(|o| *o /= n);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{RandomSampler, Sampler};
    use crate::trial::TrialHistory;

    fn space() -> MultimodalSpace {
        MultimodalSpace::new(vec![
            Modality::new("image", ["cnn", "vit"], 4, 8),
            Modality::new("text", ["rnn", "transformer"], 4, 8),
        ])
    }

    #[test]
    fn fuse_concat_and_mean() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert_eq!(
            fuse(&[a.clone(), b.clone()], Fusion::Concat).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            fuse(&[a.clone(), b.clone()], Fusion::Mean).unwrap(),
            vec![2.0, 3.0]
        );
        assert_eq!(fuse(&[a, b], Fusion::Sum).unwrap(), vec![4.0, 6.0]);
    }

    #[test]
    fn mean_fusion_rejects_mismatched_dims() {
        let bad = fuse(&[vec![1.0, 2.0], vec![3.0]], Fusion::Mean);
        assert!(bad.is_err());
        // Concat tolerates ragged dims.
        assert!(fuse(&[vec![1.0, 2.0], vec![3.0]], Fusion::Concat).is_ok());
    }

    #[test]
    fn plan_validates_dims_before_scheduling() {
        let ok = MultimodalPlan {
            encoders: vec![
                ModalityChoice {
                    name: "a".into(),
                    encoder: "x".into(),
                    dim: 6,
                },
                ModalityChoice {
                    name: "b".into(),
                    encoder: "y".into(),
                    dim: 6,
                },
            ],
            fusion: Fusion::Mean,
        };
        assert!(ok.validate().is_ok());
        assert_eq!(ok.fused_dim(), 6);

        let bad = MultimodalPlan {
            encoders: vec![
                ModalityChoice {
                    name: "a".into(),
                    encoder: "x".into(),
                    dim: 6,
                },
                ModalityChoice {
                    name: "b".into(),
                    encoder: "y".into(),
                    dim: 4,
                },
            ],
            fusion: Fusion::Mean,
        };
        assert!(bad.validate().is_err());
        // The same mismatch is fine under concat, and its fused dim is the sum.
        let concat = MultimodalPlan {
            fusion: Fusion::Concat,
            ..bad
        };
        assert!(concat.validate().is_ok());
        assert_eq!(concat.fused_dim(), 10);
    }

    #[test]
    fn decodes_every_sampled_plan() {
        let ms = space();
        let sp = ms.to_search_space();
        let mut sampler = RandomSampler::new(9);
        let history = TrialHistory::default();
        for _ in 0..200 {
            let plan = ms.decode(&sampler.suggest(&sp, &history)).unwrap();
            assert_eq!(plan.encoders.len(), 2);
            for m in &plan.encoders {
                assert!((4..=8).contains(&m.dim));
            }
            // Validation is total: it either passes or returns an error, never panics.
            let _ = plan.validate();
        }
    }
}
