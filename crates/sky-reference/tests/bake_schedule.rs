//! CPU-only checks for exact work sharing. No device is created by this test.
use sky_reference::{
    bake_schedule::{WorkEstimate, angular_nodes, canonical_index},
    config::{BakeConfig, CoordinateMapping},
    mapping::{Geometry, State},
    reference_mapping,
};

#[test]
fn canonical_nodes_are_identical_and_never_reference_another_alias() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    for nm in [4, 6, 12, 32] {
        let c = BakeConfig {
            scattering: [5, nm, 17, 33],
            ..BakeConfig::reference()
        };
        c.validate(1).unwrap();
        let nodes = angular_nodes(g, &c);
        let [nr, nm, ns, nn] = c.scattering;
        let decode = |i: usize| {
            let ri = i / (nm * ns * nn);
            let mi = i / (ns * nn) % nm;
            let si = i / nn % ns;
            let ni = i % nn;
            let ground = mi < (nm / 4).max(2);
            let s = State {
                altitude_km: nodes[nn + ri],
                mu: 0.0,
                mu_s: nodes[nn + nr + ri * ns + si],
                nu: nodes[nn + nr + nr * ns + ((ri * ns + si) * 2 + usize::from(ground)) * nn + ni],
                ground,
            };
            State {
                mu: g.optical_cone_view(s, mi, nm),
                ..s
            }
        };
        let mut shared = 0;
        for i in 0..c.scattering_len() {
            let canonical = canonical_index(&c, &nodes, i);
            assert_eq!(canonical_index(&c, &nodes, canonical), canonical);
            let a = decode(i);
            let b = decode(canonical);
            assert_eq!(
                (a.altitude_km, a.mu, a.mu_s, a.nu, a.ground),
                (b.altitude_km, b.mu, b.mu_s, b.nu, b.ground),
                "{i} -> {canonical}"
            );
            // Equality of the shader's directions and ray geometry is the
            // reason density/integration work can be shared, including ground.
            assert_eq!(a.directions(), b.directions());
            assert_eq!(
                g.distance(a.altitude_km, a.mu, a.ground),
                g.distance(b.altitude_km, b.mu, b.ground)
            );
            shared += usize::from(i != canonical);
        }
        assert!(
            shared > c.scattering_len() / 4,
            "test must cover collapsed chart regions"
        );
        let estimate = WorkEstimate::from_nodes(&c, &nodes);
        let active = sky_reference::bake_schedule::active_work_len(&c, &nodes);
        assert_eq!(active, estimate.phase_unique_states);
        let list = &nodes[nodes.len() - active..];
        let mut last = None;
        for packed in list {
            let index = packed.to_bits() as usize;
            assert!(last.is_none_or(|old| index > old));
            last = Some(index);
            let canonical = canonical_index(&c, &nodes, index);
            // The list compacts exact phase duplicates. Adapter-dependent view
            // collapse remains a checked optimization within this list.
            assert_eq!(index % nn, canonical % nn);
        }
        assert_eq!(estimate.cpu_unique_states, c.scattering_len() - shared);
        assert_eq!(
            estimate.active_states_by_radius.iter().sum::<usize>(),
            estimate.cpu_unique_states
        );
        assert!(estimate.phase_unique_states >= estimate.cpu_unique_states);
        let mut display = nodes[..nn].to_vec();
        reference_mapping::append_display_nodes(g, &c, &mut display);
        assert_eq!(display, &nodes[..nn + nr]);
    }
}

#[test]
fn older_coordinate_mappings_keep_all_work_items() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    for mapping in [
        CoordinateMapping::Legacy,
        CoordinateMapping::SunAligned,
        CoordinateMapping::SunAlignedAngular,
    ] {
        let c = BakeConfig {
            mapping,
            ..BakeConfig::smoke()
        };
        let nodes = angular_nodes(g, &c);
        assert_eq!(nodes.len(), c.scattering[3]);
        for i in 0..c.scattering_len() {
            assert_eq!(canonical_index(&c, &nodes, i), i);
        }
    }
}
