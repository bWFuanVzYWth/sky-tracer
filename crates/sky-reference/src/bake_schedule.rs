//! Exact duplicate-state reuse for collapsed regions of the reference chart.
//! The dense file layout is unchanged; only expensive transport work is shared.
use crate::{
    config::BakeConfig,
    mapping::{Geometry, State},
    reference_mapping,
};

#[derive(serde::Serialize)]
pub struct WorkEstimate {
    pub dense_states: usize,
    pub phase_unique_states: usize,
    /// View collapse is verified again in GPU arithmetic before sharing work.
    pub cpu_unique_states: usize,
    pub active_states_by_radius: Vec<usize>,
    pub coordinate_cache_bytes: usize,
    pub old_statistics_bytes_per_order: usize,
    pub compact_statistics_bytes_per_order: usize,
}

impl WorkEstimate {
    pub fn from_nodes(c: &BakeConfig, nodes: &[f32]) -> Self {
        let [nr, nm, ns, nn] = c.scattering;
        let ng = (nm / 4).max(2);
        let mut result = Self {
            dense_states: c.scattering_len(),
            phase_unique_states: 0,
            cpu_unique_states: 0,
            active_states_by_radius: vec![0; nr],
            coordinate_cache_bytes: nodes.len() * 4,
            old_statistics_bytes_per_order: c.scattering_len().div_ceil(64) * 16,
            compact_statistics_bytes_per_order: c.scattering_len().div_ceil(64).div_ceil(256) * 20,
        };
        if !c.mapping.is_reference() {
            result.phase_unique_states = result.dense_states;
            result.cpu_unique_states = result.dense_states;
            result.active_states_by_radius.fill(nm * ns * nn);
            return result;
        }
        let base = nn + nr + nr * ns + nr * ns * 2 * nn;
        for (r, row) in result.active_states_by_radius.iter_mut().enumerate() {
            for si in 0..ns {
                for ground in [false, true] {
                    for ni in 0..nn {
                        let code = nodes[base + ((r * ns + si) * 2 + usize::from(ground)) * nn + ni]
                            as usize;
                        if code % nn == ni {
                            let n = if ground { ng } else { nm - ng };
                            result.phase_unique_states += n;
                            let count = if code >= nn { 1 } else { n };
                            result.cpu_unique_states += count;
                            *row += count;
                        }
                    }
                }
            }
        }
        result
    }
}

pub fn angular_nodes(g: Geometry, c: &BakeConfig) -> Vec<f32> {
    let mut nodes: Vec<_> = (0..c.scattering[3])
        .map(|i| crate::mapping::scattering_cosine(crate::mapping::unit(i, c.scattering[3])))
        .collect();
    reference_mapping::append_nodes(g, c, &mut nodes);
    if !c.mapping.is_reference() {
        return nodes;
    }
    let [nr, nm, ns, nn] = c.scattering;
    let height_base = nn;
    let solar_base = height_base + nr;
    let phase_base = solar_base + nr * ns;
    let ng = (nm / 4).max(2);
    let mut aliases = Vec::with_capacity(nr * ns * 2 * nn);
    for ri in 0..nr {
        for si in 0..ns {
            for ground in [false, true] {
                let mut canonical = 0;
                let mut previous = 2.0;
                for ni in 0..nn {
                    let nu =
                        nodes[phase_base + ((ri * ns + si) * 2 + usize::from(ground)) * nn + ni];
                    if nu != previous {
                        canonical = ni;
                        previous = nu;
                    }
                    let s = State {
                        altitude_km: nodes[height_base + ri],
                        mu: 0.0,
                        mu_s: nodes[solar_base + ri * ns + si],
                        nu,
                        ground,
                    };
                    let (lo, hi) = if ground { (0, ng - 1) } else { (ng, nm - 1) };
                    // Equal endpoint views imply a collapsed monotone interval. Verify
                    // every interior view as well: no epsilon or approximate clustering.
                    let mu = g.optical_cone_view(s, lo, nm);
                    let collapsed = g.optical_cone_view(s, hi, nm).to_bits() == mu.to_bits()
                        && (lo + 1..hi)
                            .all(|mi| g.optical_cone_view(s, mi, nm).to_bits() == mu.to_bits());
                    aliases.push((canonical + if collapsed { nn } else { 0 }) as f32);
                }
            }
        }
    }
    nodes.extend(aliases);
    // Raw u32 indices share the node buffer's storage binding. Shaders read
    // this suffix as integers, so no f32 index rounding or denormal arithmetic.
    let active_base = nodes.len();
    nodes.push(0.0);
    for ri in 0..nr {
        for mi in 0..nm {
            for si in 0..ns {
                let ground = mi < ng;
                let code_base =
                    phase_base + nr * ns * 2 * nn + ((ri * ns + si) * 2 + usize::from(ground)) * nn;
                let row = ((ri * nm + mi) * ns + si) * nn;
                for ni in 0..nn {
                    if nodes[code_base + ni] as usize % nn == ni {
                        nodes.push(f32::from_bits((row + ni) as u32));
                    }
                }
            }
        }
    }
    nodes[active_base] = f32::from_bits((nodes.len() - active_base - 1) as u32);
    nodes
}

pub fn active_work_len(c: &BakeConfig, nodes: &[f32]) -> usize {
    if !c.mapping.is_reference() {
        return c.scattering_len();
    }
    let [nr, _, ns, nn] = c.scattering;
    nodes[nn + nr + nr * ns + nr * ns * 4 * nn].to_bits() as usize
}

pub fn canonical_index(c: &BakeConfig, nodes: &[f32], index: usize) -> usize {
    if !c.mapping.is_reference() {
        return index;
    }
    let [nr, nm, ns, nn] = c.scattering;
    let ri = index / (nm * ns * nn);
    let si = index / nn % ns;
    let mi = index / (ns * nn) % nm;
    let ng = (nm / 4).max(2);
    let ground = mi < ng;
    let base = nn + nr + nr * ns + nr * ns * 2 * nn;
    let code = nodes[base + ((ri * ns + si) * 2 + usize::from(ground)) * nn + index % nn] as usize;
    let canonical_m = if code >= nn {
        if ground { 0 } else { ng }
    } else {
        mi
    };
    ((ri * nm + canonical_m) * ns + si) * nn + code % nn
}
