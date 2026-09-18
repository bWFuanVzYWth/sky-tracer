use sky_atmosphere_lut::{
    packed::{decode_block, encode_block, quantize},
    renderer::shader_source,
};

#[test]
fn packed_roundtrip_signs_dark_values_and_word_boundaries() {
    let mut seed = 1978u32;
    for n in [16, 32, 64] {
        for _ in 0..256 {
            let values: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    std::array::from_fn(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        // Include every finite exponent and both signs, including subnormals.
                        f32::from_bits(if seed & 0x7f800000 == 0x7f800000 {
                            seed & !0x00800000
                        } else {
                            seed
                        })
                    })
                })
                .collect();
            let encoded = encode_block(&values);
            for (i, v) in values.iter().enumerate() {
                for (c, a) in v.iter().enumerate() {
                    let b = decode_block(&encoded, n, i, c);
                    assert_eq!(b.to_bits(), quantize(*a) << 12);
                    assert!(b.is_finite());
                    // Saturation at f32::MAX has a slightly looser bound.
                    if a.is_normal() {
                        assert!((a - b).abs() / a.abs() < 0.00049);
                    }
                }
            }
        }
    }
    for value in [0.0, -0.0, 1.0, -1.0, 1e-35, 1e30] {
        let values = vec![[value; 3]; 32];
        let encoded = encode_block(&values);
        assert_eq!(encoded.len(), 3);
        assert_eq!(
            decode_block(&encoded, 32, 31, 2).to_bits(),
            quantize(value) << 12
        );
    }
}

#[test]
fn packed_shader_validates_without_gpu_and_uses_packed_bindings() {
    for packed in [false, true] {
        let source = shader_source(packed);
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let entry = module
            .entry_points
            .iter()
            .position(|e| e.name == "render_spectral")
            .unwrap();
        let bindings: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(h, v)| {
                if info.get_entry_point(entry)[h].is_empty() {
                    None
                } else {
                    v.binding.as_ref().map(|b| b.binding)
                }
            })
            .collect();
        assert_eq!(bindings.contains(&5), !packed);
        assert_eq!(bindings.contains(&12), packed);
        assert_eq!(bindings.contains(&13), packed);
    }
}
