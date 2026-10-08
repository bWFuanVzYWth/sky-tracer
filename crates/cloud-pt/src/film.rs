//! Sample statistics, independent of display exposure and tone mapping.
use crate::{
    Result,
    config::{Camera, RenderConfig},
    sampling::Pcg32,
    transport::{self, DensityField, TransportSettings},
};

#[derive(Clone, Debug)]
pub struct Film {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub mean: Vec<[f32; 3]>,
    /// Unbiased estimate of variance of individual samples (not mean variance).
    /// Undefined at one sample; stores zero and output metadata marks this.
    pub sample_variance: Vec<[f32; 3]>,
}
impl Film {
    pub fn validate(&self) -> Result<()> {
        let len = self
            .width
            .checked_mul(self.height)
            .ok_or("film dimensions overflow")? as usize;
        if self.samples_per_pixel == 0
            || len == 0
            || self.mean.len() != len
            || self.sample_variance.len() != len
            || self
                .mean
                .iter()
                .chain(&self.sample_variance)
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid cloud reference film".into());
        }
        Ok(())
    }
}

/// Serial f64 reference, intentionally separate from the GPU estimator. A
/// failed path aborts the image; discarding it would bias the remaining mean.
pub fn render_cpu(
    volume: &impl DensityField,
    camera: &Camera,
    settings: &TransportSettings,
    config: &RenderConfig,
    mut progress: impl FnMut(u32),
) -> Result<Film> {
    config.validate()?;
    camera.basis()?;
    settings.validate()?;
    let len = (config.width * config.height) as usize;
    let mut means = vec![[0.0f64; 3]; len];
    let mut m2 = vec![[0.0f64; 3]; len];
    for sample in 0..config.spp {
        for y in 0..config.height {
            for x in 0..config.width {
                let index = (y * config.width + x) as usize;
                let mut rng = Pcg32::for_sample(config.seed, index as u64, u64::from(sample));
                let ray = camera.ray(
                    config.width,
                    config.height,
                    f64::from(x) + rng.open01(),
                    f64::from(y) + rng.open01(),
                )?;
                let value = transport::trace_sample(volume, ray, settings, &mut rng)?
                    .radiance
                    .to_array();
                for channel in 0..3 {
                    let delta = value[channel] - means[index][channel];
                    means[index][channel] += delta / f64::from(sample + 1);
                    m2[index][channel] += delta * (value[channel] - means[index][channel]);
                }
            }
        }
        progress(sample + 1);
    }
    let film = Film {
        width: config.width,
        height: config.height,
        samples_per_pixel: config.spp,
        mean: means.into_iter().map(|v| v.map(|x| x as f32)).collect(),
        sample_variance: m2
            .into_iter()
            .map(|v| {
                v.map(|x| {
                    if config.spp > 1 {
                        (x / f64::from(config.spp - 1)) as f32
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
    };
    film.validate()?;
    Ok(film)
}
