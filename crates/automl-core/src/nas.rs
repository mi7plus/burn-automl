//! Architecture-graph search primitives for Neural Architecture Search
//! (roadmap v0.7; PRD §4.1 "architecture graphs, added in the NAS releases",
//! §9/§21).
//!
//! NAS lives *inside* the optimization engine rather than beside it. A
//! macro-architecture — a linear stack of cells, where each cell chooses one
//! operation from a shared, **reusable** palette plus a channel width and an
//! optional skip connection from the previous cell — is encoded as an ordinary
//! conditional [`SearchSpace`]. Every existing [`crate::sampler::Sampler`] then
//! searches architectures with no NAS-specific machinery, and the
//! [`crate::evolution::EvolutionarySampler`] in particular yields *evolutionary
//! NAS* for free (its Gaussian/categorical mutations are architecture mutations
//! once the graph is encoded).
//!
//! The core stays framework-agnostic: it knows only typed operation *names*
//! (`"conv3"`, `"conv5"`, `"identity"`, …). A framework adapter such as
//! `automl-burn` maps a decoded [`Architecture`] onto concrete layers.
//!
//! Two entry points into the same representation:
//! - Encode → search: [`MacroSpace::to_search_space`] then any sampler; decode
//!   each proposal with [`MacroSpace::decode`].
//! - Direct graph mutation: [`MacroSpace::mutate`] applies one architecture-level
//!   edit (op swap, width change, skip toggle, depth ±1), the explicit
//!   evolutionary operator over the graph.

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::{ParamSet, ParamValue};
use crate::space::{Condition, SearchSpace};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A reusable, named set of candidate operations shared across every cell of a
/// macro-architecture — the PRD's "reusable named subspaces, composable across
/// task adapters" (§4.1). The same palette gates every layer's op choice, so a
/// block vocabulary is declared once and reused throughout the search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpPalette {
    ops: Vec<String>,
}

impl OpPalette {
    /// Build a palette from operation names. Duplicates are dropped (first wins);
    /// order is preserved for deterministic encoding.
    pub fn new<I, S>(ops: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen = Vec::new();
        for op in ops {
            let op = op.into();
            if !seen.contains(&op) {
                seen.push(op);
            }
        }
        if seen.is_empty() {
            return Err(Error::InvalidDistribution {
                name: "op_palette".into(),
                reason: "an op palette needs at least one operation".into(),
            });
        }
        Ok(OpPalette { ops: seen })
    }

    /// The operation names, in declaration order.
    pub fn ops(&self) -> &[String] {
        &self.ops
    }
}

/// One decoded cell of an architecture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    /// The chosen operation name (an entry of the [`OpPalette`]).
    pub op: String,
    /// The cell's channel width.
    pub width: usize,
    /// Whether this cell receives a skip/residual connection from the previous
    /// cell (always `false` for the first cell).
    pub skip_from_prev: bool,
}

/// A decoded macro-architecture: an ordered stack of [`Layer`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    /// Cells from input to output.
    pub layers: Vec<Layer>,
}

impl Architecture {
    /// Number of cells (the architecture depth).
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// A short, stable, human-readable description for logs and provenance,
    /// e.g. `conv3x16 -> conv5x32(+skip) -> identity x16`.
    pub fn describe(&self) -> String {
        self.layers
            .iter()
            .map(|l| {
                let skip = if l.skip_from_prev { "(+skip)" } else { "" };
                format!("{}x{}{skip}", l.op, l.width)
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

/// A searchable macro-architecture: a variable-depth stack of cells drawn from a
/// shared [`OpPalette`], with per-cell width and skip choices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroSpace {
    palette: OpPalette,
    min_depth: usize,
    max_depth: usize,
    widths: Vec<usize>,
    allow_skip: bool,
}

impl MacroSpace {
    /// A new macro-space over `palette`, with depth in `[min_depth, max_depth]`
    /// and per-cell channel widths chosen from `widths`. Skip connections are
    /// enabled by default; disable with [`MacroSpace::without_skips`].
    pub fn new(
        palette: OpPalette,
        min_depth: usize,
        max_depth: usize,
        widths: Vec<usize>,
    ) -> Result<Self> {
        let min_depth = min_depth.max(1);
        let max_depth = max_depth.max(min_depth);
        let mut widths = widths;
        widths.retain(|&w| w > 0);
        widths.sort_unstable();
        widths.dedup();
        if widths.is_empty() {
            return Err(Error::InvalidDistribution {
                name: "macro_space.widths".into(),
                reason: "at least one positive channel width is required".into(),
            });
        }
        Ok(MacroSpace {
            palette,
            min_depth,
            max_depth,
            widths,
            allow_skip: true,
        })
    }

    /// Disable skip connections (every cell is purely sequential).
    pub fn without_skips(mut self) -> Self {
        self.allow_skip = false;
        self
    }

    /// The shared operation palette.
    pub fn palette(&self) -> &OpPalette {
        &self.palette
    }

    /// The inclusive depth bounds `(min, max)`.
    pub fn depth_bounds(&self) -> (usize, usize) {
        (self.min_depth, self.max_depth)
    }

    /// The candidate channel widths (sorted, de-duplicated).
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// Encode this macro-architecture as a conditional [`SearchSpace`]: a `depth`
    /// integer, then per-cell `op{i}` / `width{i}` / `skip{i}` parameters, each
    /// gated to be active only when `depth` reaches cell `i`. Any sampler can
    /// then search architectures; decode a proposal with [`MacroSpace::decode`].
    pub fn to_search_space(&self) -> SearchSpace {
        let mut space = SearchSpace::new().add(
            "depth",
            Distribution::int(self.min_depth as i64, self.max_depth as i64),
        );
        let width_choices: Vec<String> = self.widths.iter().map(|w| w.to_string()).collect();
        for i in 0..self.max_depth {
            // Cell `i` is active for any depth that includes it: depth in i+1..=max.
            let active_depths: Vec<ParamValue> = ((i + 1)..=self.max_depth)
                .map(|d| ParamValue::Int(d as i64))
                .collect();
            let gate = Condition::when_in("depth", active_depths);
            space = space.add_conditional(
                format!("op{i}"),
                Distribution::categorical(self.palette.ops.iter().cloned()),
                gate.clone(),
            );
            space = space.add_conditional(
                format!("width{i}"),
                Distribution::categorical(width_choices.iter().cloned()),
                gate.clone(),
            );
            // The first cell has no predecessor, so no skip parameter.
            if self.allow_skip && i > 0 {
                space = space.add_conditional(format!("skip{i}"), Distribution::boolean(), gate);
            }
        }
        space
    }

    /// Decode a sampled [`ParamSet`] (from [`MacroSpace::to_search_space`]) into a
    /// concrete [`Architecture`].
    pub fn decode(&self, params: &ParamSet) -> Result<Architecture> {
        let depth = params.int("depth")? as usize;
        let depth = depth.clamp(self.min_depth, self.max_depth);
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let op = params.categorical(&format!("op{i}"))?.to_string();
            let width: usize = params
                .categorical(&format!("width{i}"))?
                .parse()
                .map_err(|_| Error::Objective(format!("width{i} is not an integer")))?;
            let skip_from_prev = if self.allow_skip && i > 0 {
                params.boolean(&format!("skip{i}")).unwrap_or(false)
            } else {
                false
            };
            layers.push(Layer {
                op,
                width,
                skip_from_prev,
            });
        }
        Ok(Architecture { layers })
    }

    /// A uniformly-random valid architecture, for seeding an evolutionary search
    /// or a smoke test.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Architecture {
        let depth = rng.gen_range(self.min_depth..=self.max_depth);
        let layers = (0..depth)
            .map(|i| Layer {
                op: self.palette.ops[rng.gen_range(0..self.palette.ops.len())].clone(),
                width: self.widths[rng.gen_range(0..self.widths.len())],
                skip_from_prev: self.allow_skip && i > 0 && rng.gen_bool(0.5),
            })
            .collect();
        Architecture { layers }
    }

    /// Apply exactly one architecture-level mutation — the explicit evolutionary
    /// operator over the graph. One of: swap a cell's op, change a cell's width,
    /// toggle a skip connection, or grow/shrink the depth by one cell (staying
    /// within `[min_depth, max_depth]`). The result is always a valid
    /// architecture of this macro-space.
    pub fn mutate<R: Rng>(&self, arch: &Architecture, rng: &mut R) -> Architecture {
        let mut layers = arch.layers.clone();
        // Restrict depth changes to the ends of the available range.
        let can_grow = layers.len() < self.max_depth;
        let can_shrink = layers.len() > self.min_depth;

        // Choose a mutation kind, then retry-free apply.
        #[derive(Clone, Copy)]
        enum Kind {
            Op,
            Width,
            Skip,
            Grow,
            Shrink,
        }
        let mut kinds = vec![Kind::Op, Kind::Width];
        if self.allow_skip && layers.len() > 1 {
            kinds.push(Kind::Skip);
        }
        if can_grow {
            kinds.push(Kind::Grow);
        }
        if can_shrink {
            kinds.push(Kind::Shrink);
        }
        match kinds[rng.gen_range(0..kinds.len())] {
            Kind::Op => {
                let i = rng.gen_range(0..layers.len());
                layers[i].op = self.palette.ops[rng.gen_range(0..self.palette.ops.len())].clone();
            }
            Kind::Width => {
                let i = rng.gen_range(0..layers.len());
                layers[i].width = self.widths[rng.gen_range(0..self.widths.len())];
            }
            Kind::Skip => {
                // A skip only exists on cells after the first.
                let i = rng.gen_range(1..layers.len());
                layers[i].skip_from_prev = !layers[i].skip_from_prev;
            }
            Kind::Grow => {
                layers.push(Layer {
                    op: self.palette.ops[rng.gen_range(0..self.palette.ops.len())].clone(),
                    width: self.widths[rng.gen_range(0..self.widths.len())],
                    skip_from_prev: self.allow_skip && rng.gen_bool(0.5),
                });
            }
            Kind::Shrink => {
                layers.pop();
            }
        }
        Architecture { layers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{RandomSampler, Sampler};
    use crate::trial::TrialHistory;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn space() -> MacroSpace {
        let palette = OpPalette::new(["conv3", "conv5", "identity"]).unwrap();
        MacroSpace::new(palette, 2, 4, vec![8, 16, 32]).unwrap()
    }

    #[test]
    fn palette_dedups_and_rejects_empty() {
        let p = OpPalette::new(["a", "b", "a"]).unwrap();
        assert_eq!(p.ops(), &["a".to_string(), "b".to_string()]);
        assert!(OpPalette::new(Vec::<String>::new()).is_err());
    }

    #[test]
    fn encoded_space_has_gated_cells() {
        let s = space().to_search_space();
        // depth + 4*(op,width,skip) but cell 0 has no skip: 1 + (4*3 - 1) = 12.
        assert_eq!(s.len(), 12);
        assert_eq!(s.get("depth").unwrap().condition, None);
        // op0 is gated (only active once depth >= 1), and cell 0 has no skip.
        assert!(s.get("op0").unwrap().condition.is_some());
        assert!(s.get("skip0").is_none());
        assert!(s.get("skip1").unwrap().condition.is_some());
    }

    #[test]
    fn every_sampled_architecture_is_valid() {
        // A sampler over the encoded space decodes to a well-formed architecture
        // whose depth, ops and widths all respect the macro-space.
        let ms = space();
        let sp = ms.to_search_space();
        let mut sampler = RandomSampler::new(7);
        let history = TrialHistory::default();
        for _ in 0..200 {
            let params = sampler.suggest(&sp, &history);
            let arch = ms.decode(&params).unwrap();
            assert!((2..=4).contains(&arch.depth()));
            assert!(!arch.layers[0].skip_from_prev);
            for l in &arch.layers {
                assert!(ms.palette().ops().contains(&l.op));
                assert!(ms.widths().contains(&l.width));
            }
        }
    }

    #[test]
    fn mutation_stays_within_bounds() {
        let ms = space();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut arch = ms.sample(&mut rng);
        for _ in 0..500 {
            arch = ms.mutate(&arch, &mut rng);
            assert!((2..=4).contains(&arch.depth()));
            assert!(!arch.layers[0].skip_from_prev);
            for l in &arch.layers {
                assert!(ms.palette().ops().contains(&l.op));
                assert!(ms.widths().contains(&l.width));
            }
        }
    }

    #[test]
    fn mutation_changes_exactly_one_thing() {
        // Over many trials a mutation always produces a different architecture
        // (some single edit was applied).
        let ms = space();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let arch = ms.sample(&mut rng);
        let mut changed = 0;
        for _ in 0..200 {
            let m = ms.mutate(&arch, &mut rng);
            if m != arch {
                changed += 1;
            }
        }
        // An op/width re-pick can land on the same value, so a minority of
        // mutations are no-ops; the operator should still change the graph the
        // large majority of the time.
        assert!(
            changed > 140,
            "mutation rarely changed the graph: {changed}/200"
        );
    }

    #[test]
    fn describe_is_readable() {
        let arch = Architecture {
            layers: vec![
                Layer {
                    op: "conv3".into(),
                    width: 16,
                    skip_from_prev: false,
                },
                Layer {
                    op: "conv5".into(),
                    width: 32,
                    skip_from_prev: true,
                },
            ],
        };
        assert_eq!(arch.describe(), "conv3x16 -> conv5x32(+skip)");
    }
}
