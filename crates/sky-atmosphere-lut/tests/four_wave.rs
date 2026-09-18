#[test]
fn shader_validates() {
    let source = sky_atmosphere_lut::four_wave::renderer::SHADER;
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn cached_shader_validates() {
    let source = sky_atmosphere_lut::four_wave::cached::shader_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn solar_models_validate() {
    use sky_atmosphere_lut::four_wave::{
        cached, renderer,
        solar::{SunModel, shader_for_model},
    };
    for model in [
        SunModel::Parallel,
        SunModel::PhaseAveraged,
        SunModel::Finite4,
        SunModel::Finite64,
    ] {
        for source in [renderer::SHADER.to_owned(), cached::shader_source()] {
            let source = shader_for_model(&source, model);
            let module = wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{model:?}: {}", e.emit_to_string(&source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
