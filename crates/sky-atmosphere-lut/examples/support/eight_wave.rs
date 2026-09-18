//! Reuse the demo's searched quadrature; the medium remains the frozen teacher's.
use sky_atmosphere_lut::{Result, asset::Manifest};
use sky_unreal_atmosphere_8wave::params::SPECTRAL_SAMPLE_WAVELENGTHS_NM;

// Read the authoritative demo WGSL constants during CPU precomputation rather
// than keeping a second copy or composing different rounded RGB color bases.
fn vec3_columns(source: &str, function: &str) -> Result<Vec<[f32; 3]>> {
    let body = source
        .split(function)
        .nth(1)
        .and_then(|s| s.split("\n}").next())
        .ok_or("demo color function missing")?;
    body.split("vec3<f32>(")
        .skip(1)
        .map(|s| {
            let end = s.split(')').next().ok_or("invalid demo color column")?;
            let values = end
                .split(',')
                .map(|v| v.trim().parse::<f32>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            values
                .try_into()
                .map_err(|_| "demo color column must have three values".into())
        })
        .collect()
}

pub struct EightWave {
    pub indices: [usize; 8],
    /// Convert *band-integrated* radiance, not radiance per nm. Keep the full
    /// reference solar-to-D65 adaptation; do not re-neutralize the eight samples.
    pub rgb_from_integrated: [[f32; 3]; 8],
}
impl EightWave {
    pub fn new(m: &Manifest) -> Result<Self> {
        let shader = sky_unreal_atmosphere_8wave::COMMON_WGSL;
        let matrix = vec3_columns(shader, "fn linear_rec2020_from_spectral8(")?;
        let white = vec3_columns(shader, "fn white_balance_rec2020(")?;
        if matrix.len() != 8 || white.len() != 3 {
            return Err("unsupported demo color matrix shape".into());
        }
        let mut indices = [0; 8];
        let mut rgb_from_integrated = [[0.0; 3]; 8];
        for k in 0..8 {
            let nm = SPECTRAL_SAMPLE_WAVELENGTHS_NM[k / 4][k % 4];
            let i = m
                .bands
                .iter()
                .position(|b| b.center_nm == nm)
                .ok_or("the frozen source does not contain a demo wavelength")?;
            let width = m.bands[i].upper_nm - m.bands[i].lower_nm;
            if width <= 0.0 {
                return Err("invalid source bandwidth".into());
            }
            indices[k] = i;
            rgb_from_integrated[k] = std::array::from_fn(|c| {
                (0..3).map(|j| white[j][c] * matrix[k][j]).sum::<f32>() / width
            });
        }
        Ok(Self {
            indices,
            rgb_from_integrated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_atmosphere_lut::{config::BakeConfig, model::Model};
    #[test]
    fn searched_weights_match_the_demo_shader_and_band_units() -> Result<()> {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let scene = sky_core::data::load_scene_data(&data, 0.0, 0.0).map_err(|e| e.to_string())?;
        let model = Model::from_scene(&scene)?;
        let m = Manifest::new(&model, BakeConfig::smoke(), "CPU test".into())?;
        let eight = EightWave::new(&m)?;
        let shader = sky_unreal_atmosphere_8wave::COMMON_WGSL;
        let matrix = vec3_columns(shader, "fn linear_rec2020_from_spectral8(")?;
        let white = vec3_columns(shader, "fn white_balance_rec2020(")?;
        assert_eq!(matrix.len(), 8);
        assert_eq!(white.len(), 3);
        for (k, &i) in eight.indices.iter().enumerate() {
            let b = &m.bands[i];
            let width = b.upper_nm - b.lower_nm;
            let solar_per_nm = b.solar_irradiance_w_m2 / width;
            assert!(
                (solar_per_nm
                    - sky_unreal_atmosphere_8wave::params::SUN_SPECTRAL_IRRADIANCE[k / 4][k % 4])
                    .abs()
                    < 1e-5
            );
            for c in 0..3 {
                let shader_value = (0..3).map(|j| white[j][c] * matrix[k][j]).sum::<f32>();
                let cpu_value = eight.rgb_from_integrated[k][c] * width;
                assert!(
                    (cpu_value - shader_value).abs() < 2e-5 * shader_value.abs().max(1.0),
                    "lane={k} channel={c}: {cpu_value} vs {shader_value}"
                );
            }
        }
        Ok(())
    }
}
