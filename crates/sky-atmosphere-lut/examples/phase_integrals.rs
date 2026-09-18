//! Diagnose the deterministic angular rule against the source phase tables.
fn main() -> sky_atmosphere_lut::Result<()> {
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = sky_atmosphere_lut::model::Model::from_scene(&scene)?;
    for n in [16, 32, 64, 128] {
        let q = sky_atmosphere_lut::quadrature::sphere(n, 1);
        for i in [6, 17, 30] {
            let mut integrals = [0.0; 5];
            for (v, w) in &q {
                for (value, p) in integrals.iter_mut().zip(model.bands[i].phases(v.z)) {
                    *value += w * p;
                }
            }
            println!("n={n}, {} nm: {integrals:?}", model.bands[i].info.center_nm);
        }
    }
    Ok(())
}
