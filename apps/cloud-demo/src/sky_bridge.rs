//! Application-level boundary lighting. The independent cloud core knows only
//! a linear RGB direction texture and a delta-sun irradiance texture.
use crate::output::EnvironmentRecord;
use bytemuck::{Pod, Zeroable};
use cloud_pt::{Result, config::Camera, transport::TransportSettings};
use glam::DVec3;
use sky_realtime::{Config, Renderer, View, Wavelengths, model::Model};
use wgpu::util::DeviceExt;

pub const ENVIRONMENT_SIZE: [u32; 2] = [1024, 512];
pub const COMPOSITE_SHADER: &str = include_str!("shaders/cloud_composite.wgsl");
const CONVERT_SHADER: &str = include_str!("shaders/cloud_environment_convert.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Normalize {
    solar: [f32; 4],
    rgb: [[f32; 4]; 4],
    scale: [f32; 4],
}
#[derive(Clone, Copy, PartialEq)]
struct Key {
    altitude: f32,
    sun: [f64; 3],
    irradiance: [f64; 3],
}

pub struct SkyBackground {
    sky: Renderer,
    _environment: wgpu::Texture,
    environment: wgpu::TextureView,
    _sunlight: wgpu::Texture,
    sunlight: wgpu::TextureView,
    uniform: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    wavelengths: Wavelengths,
    config: Config,
    medium_key: String,
    sea_level: f64,
    key: Option<Key>,
    enabled: bool,
    record: Option<EnvironmentRecord>,
}

impl SkyBackground {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &TransportSettings,
    ) -> Result<Self> {
        let model = Model::earth()?;
        let wavelengths = Wavelengths::optimized_four();
        let config = Config::balanced();
        let mut sky = Renderer::new(device, &model, &wavelengths, 0.18, config.clone())?;
        let solved = sky.rebuild(device, queue)?;
        sky.resize(device, ENVIRONMENT_SIZE);
        let (environment_texture, environment) =
            texture(device, "cloud linear sky environment", ENVIRONMENT_SIZE);
        let (sunlight_texture, sunlight) =
            texture(device, "cloud atmospheric delta sunlight", [1, 1]);
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud sky normalization"),
            contents: bytemuck::bytes_of(&Normalize::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud sky color conversion"),
            source: wgpu::ShaderSource::Wgsl(CONVERT_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud sky color conversion"),
            layout: None,
            module: &module,
            entry_point: Some("convert"),
            compilation_options: Default::default(),
            cache: None,
        });
        let transmittance = sky.transmittance_texture().create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud sky conversion resources"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sky.target_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&environment),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&sunlight),
                },
            ],
        });
        Ok(Self {
            sky,
            _environment: environment_texture,
            environment,
            _sunlight: sunlight_texture,
            sunlight,
            uniform,
            pipeline,
            group,
            wavelengths,
            config,
            medium_key: solved.medium_key,
            sea_level: settings.ground.map_or(-1000.0, |g| g.height),
            key: None,
            enabled: false,
            record: None,
        })
    }
    /// Call only after cancelling old cloud paths and waiting for their chunk.
    /// Horizontal translation/rotation reuse the full-sphere environment.
    pub fn encode(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        settings: &TransportSettings,
        enabled: bool,
    ) -> Result<bool> {
        self.enabled = enabled;
        if !enabled {
            return Ok(false);
        }
        camera.basis()?;
        settings.validate()?;
        let altitude = ((camera.origin.y - self.sea_level) / 1000.0).max(0.0) as f32;
        let sun = settings.sun_direction.normalize();
        let key = Key {
            altitude,
            sun: sun.to_array(),
            irradiance: settings.sun_irradiance.to_array(),
        };
        if self.key == Some(key) {
            return Ok(false);
        }
        let gain = lighting_gain(&self.wavelengths, settings.sun_irradiance) as f32;
        if !altitude.is_finite() || !gain.is_finite() {
            return Err("sky boundary exceeds f32 range".into());
        }
        let view = View {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            fov_y_deg: 90.0,
            altitude_km: altitude,
            sun_elevation_deg: sun.y.clamp(-1.0, 1.0).asin().to_degrees() as f32,
            sun_azimuth_deg: sun.x.atan2(sun.z).to_degrees() as f32,
        };
        self.sky.render_environment(queue, encoder, view);
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&Normalize {
                solar: self.wavelengths.solar,
                rgb: self.wavelengths.rgb.map(|v| [v[0], v[1], v[2], 0.0]),
                scale: [gain, 0.0, 0.0, 0.0],
            }),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud sky normalization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(
                ENVIRONMENT_SIZE[0].div_ceil(8),
                ENVIRONMENT_SIZE[1].div_ceil(8),
                1,
            );
        }
        self.key = Some(key);
        self.record = Some(EnvironmentRecord {
            kind: "realtime_sky_directional_boundary_v1",
            size: ENVIRONMENT_SIZE,
            altitude_km: altitude,
            sea_level_world_y: self.sea_level,
            sun_direction: key.sun,
            sun_irradiance: key.irradiance,
            realtime_medium_key: self.medium_key.clone(),
            realtime_config: self.config.clone(),
            mapping: serde_json::from_str(sky_realtime::mapping::calibration_json())?,
            wavelengths: self.wavelengths.clone(),
            reconstruction: "texel-centered equirectangular linear RGB; bilinear longitude wrap and latitude clamp",
            color_space: "linear sRGB; one gain matching cloud top-of-atmosphere solar luminance; negative converted channels clamped",
            solar_disk: "excluded from dome; atmosphere-attenuated delta-sun next-event estimator",
            approximation: "one far-field environment at camera altitude; 1 world unit = 1 metre; no atmospheric transport inside cloud paths; cloud PT estimates this frozen RGB boundary model",
        });
        Ok(true)
    }
    pub fn view(&self) -> &wgpu::TextureView {
        &self.environment
    }
    pub fn sunlight_view(&self) -> &wgpu::TextureView {
        &self.sunlight
    }
    /// Diagnostic readback only; production lighting binds the views above.
    /// Read-only access for the standalone environment GPU audit.
    #[allow(dead_code)]
    pub fn environment_texture(&self) -> &wgpu::Texture {
        &self._environment
    }
    #[allow(dead_code)]
    pub fn sunlight_texture(&self) -> &wgpu::Texture {
        &self._sunlight
    }
    pub fn metadata(&self) -> Option<EnvironmentRecord> {
        self.enabled.then(|| self.record.clone()).flatten()
    }
}

fn texture(d: &wgpu::Device, label: &str, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
    let t = d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}
fn linear_srgb(v: DVec3) -> DVec3 {
    DVec3::new(
        1.660491 * v.x - 0.5876411 * v.y - 0.0728499 * v.z,
        -0.1245505 * v.x + 1.1328999 * v.y - 0.0083494 * v.z,
        -0.0181508 * v.x - 0.1005789 * v.y + 1.1187297 * v.z,
    )
}
fn lighting_gain(w: &Wavelengths, irradiance: DVec3) -> f64 {
    let white = w
        .rgb
        .iter()
        .zip(w.solar)
        .fold(DVec3::ZERO, |sum, (rgb, e)| {
            sum + DVec3::from_array(rgb.map(f64::from)) * f64::from(e)
        });
    let luminance = DVec3::new(0.2126, 0.7152, 0.0722);
    irradiance.dot(luminance) / linear_srgb(white).dot(luminance)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_matches_top_sun_luminance() {
        let w = Wavelengths::optimized_four();
        let e = DVec3::new(2.6, 2.5, 2.3);
        let gain = lighting_gain(&w, e);
        assert!(gain > 0.012 && gain < 0.014);
        assert_eq!(lighting_gain(&w, DVec3::ZERO), 0.0);
        assert!((linear_srgb(DVec3::ONE) - DVec3::ONE).abs().max_element() < 1e-6);
    }
    #[test]
    fn viewer_shaders_validate_and_generate_spirv() {
        for source in [CONVERT_SHADER, COMPOSITE_SHADER] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap();
        }
    }
}
