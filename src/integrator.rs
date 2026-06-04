use std::error::Error;
use std::fmt;
use std::mem;
use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::atmosphere::{PHASE_BINS, SPECIES_COUNT, SceneData};
use crate::config::RenderConfig;
use crate::film::Film;
use crate::spectrum::BAND_COUNT;

const WATCHDOG_LIMIT: u32 = 1024;
const SAMPLES_PER_DISPATCH: u32 = 1;
const TILE_HEIGHT: usize = 32;

#[derive(Debug)]
pub enum RenderError {
    InvalidConfig(String),
    NoAdapter(String),
    RequestDevice(String),
    BufferMap(String),
    Poll(String),
    Watchdog,
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid render config: {message}"),
            Self::NoAdapter(message) => write!(f, "no usable wgpu adapter: {message}"),
            Self::RequestDevice(message) => write!(f, "failed to request wgpu device: {message}"),
            Self::BufferMap(message) => write!(f, "failed to map GPU readback buffer: {message}"),
            Self::Poll(message) => write!(f, "failed while waiting for GPU work: {message}"),
            Self::Watchdog => write!(f, "GPU path watchdog was reached"),
        }
    }
}

impl Error for RenderError {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuConstants {
    width: u32,
    height: u32,
    spp: u32,
    direct_light_samples: u32,
    sample_offset: u32,
    samples_this_dispatch: u32,
    tile_y: u32,
    tile_height: u32,
    atmosphere_len: u32,
    aerosol_len: u32,
    majorant_layers: u32,
    phase_bins: u32,
    seed_lo: u32,
    seed_hi: u32,
    watchdog_limit: u32,
    _pad0: u32,
    _pad1: u32,
    observer_altitude_km: f32,
    ground_radius_km: f32,
    atmosphere_radius_km: f32,
    top_altitude_km: f32,
    sun_x: f32,
    sun_y: f32,
    sun_z: f32,
    sun_angular_radius_rad: f32,
    sun_solid_angle_sr: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuBand {
    center_nm: f32,
    lower_nm: f32,
    upper_nm: f32,
    ozone_cross_section_cm2: f32,
    solar_radiance_w_m2_sr: f32,
    rayleigh_cross_section_m2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuAtmospherePoint {
    altitude_km: f32,
    temperature_k: f32,
    air_cm3: f32,
    ozone_cm3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuAerosolPoint {
    altitude_km: f32,
    mass_g_m3: [f32; SPECIES_COUNT],
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuAerosolOptics {
    scattering_km_inv_per_g_m3: f32,
    absorption_km_inv_per_g_m3: f32,
}

#[derive(Clone, Debug)]
struct PackedScene {
    constants: GpuConstants,
    bands: Vec<GpuBand>,
    atmosphere: Vec<GpuAtmospherePoint>,
    aerosol: Vec<GpuAerosolPoint>,
    aerosol_optics: Vec<GpuAerosolOptics>,
    majorants: Vec<f32>,
    phase_values: Vec<f32>,
}

pub fn render(scene: &SceneData, config: &RenderConfig) -> Result<Film, RenderError> {
    validate_config(scene, config)?;
    pollster::block_on(render_async(scene, config))
}

fn validate_config(scene: &SceneData, config: &RenderConfig) -> Result<(), RenderError> {
    if scene.bands.len() != BAND_COUNT {
        return Err(RenderError::InvalidConfig(format!(
            "expected {BAND_COUNT} spectral bands, found {}",
            scene.bands.len()
        )));
    }
    if config.width == 0 || config.height == 0 || config.spp == 0 {
        return Err(RenderError::InvalidConfig(
            "width, height, and spp must be greater than zero".to_owned(),
        ));
    }
    let pixel_count = config
        .width
        .checked_mul(config.height)
        .and_then(|x| x.checked_mul(BAND_COUNT))
        .ok_or_else(|| RenderError::InvalidConfig("film dimensions overflow".to_owned()))?;
    if pixel_count > u32::MAX as usize {
        return Err(RenderError::InvalidConfig(
            "film is too large for the GPU shader index type".to_owned(),
        ));
    }
    if scene.atmospheric_profile.is_empty() || scene.aerosol_profile.is_empty() {
        return Err(RenderError::InvalidConfig(
            "atmosphere and aerosol profiles must not be empty".to_owned(),
        ));
    }
    if scene.majorant_grid.layer_count == 0 {
        return Err(RenderError::InvalidConfig(
            "majorant grid must contain at least one layer".to_owned(),
        ));
    }
    Ok(())
}

async fn render_async(scene: &SceneData, config: &RenderConfig) -> Result<Film, RenderError> {
    let packed = PackedScene::new(scene, config)?;
    let output_len = config.width * config.height * BAND_COUNT;
    let output_size = byte_size::<f32>(output_len)?;
    let diagnostic_size = byte_size::<u32>(1)?;

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|err| RenderError::NoAdapter(err.to_string()))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sky_tracer_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|err| RenderError::RequestDevice(err.to_string()))?;

    let constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky_tracer_constants"),
        contents: bytemuck::bytes_of(&packed.constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bands_buffer = storage_buffer(&device, "sky_tracer_bands", &packed.bands);
    let atmosphere_buffer = storage_buffer(&device, "sky_tracer_atmosphere", &packed.atmosphere);
    let aerosol_buffer = storage_buffer(&device, "sky_tracer_aerosol_profile", &packed.aerosol);
    let aerosol_optics_buffer =
        storage_buffer(&device, "sky_tracer_aerosol_optics", &packed.aerosol_optics);
    let majorants_buffer = storage_buffer(&device, "sky_tracer_majorants", &packed.majorants);
    let phase_values_buffer =
        storage_buffer(&device, "sky_tracer_phase_values", &packed.phase_values);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sky_tracer_spectral_film"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let diagnostic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky_tracer_diagnostics"),
        contents: bytemuck::cast_slice(&[0_u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let output_readback =
        readback_buffer(&device, "sky_tracer_spectral_film_readback", output_size);
    let diagnostic_readback =
        readback_buffer(&device, "sky_tracer_diagnostics_readback", diagnostic_size);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky_tracer_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("sky_trace.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sky_tracer_compute_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky_tracer_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            bind_entry(0, &constants_buffer),
            bind_entry(1, &bands_buffer),
            bind_entry(2, &atmosphere_buffer),
            bind_entry(3, &aerosol_buffer),
            bind_entry(4, &aerosol_optics_buffer),
            bind_entry(5, &majorants_buffer),
            bind_entry(6, &phase_values_buffer),
            bind_entry(7, &output_buffer),
            bind_entry(8, &diagnostic_buffer),
        ],
    });

    let mut dispatched = 0_u32;
    let mut sample_offset = 0_u32;
    while sample_offset < config.spp as u32 {
        let samples_this_dispatch = SAMPLES_PER_DISPATCH.min(config.spp as u32 - sample_offset);
        for tile_y in (0..config.height).step_by(TILE_HEIGHT) {
            let tile_height = TILE_HEIGHT.min(config.height - tile_y);
            let mut constants = packed.constants;
            constants.sample_offset = sample_offset;
            constants.samples_this_dispatch = samples_this_dispatch;
            constants.tile_y = tile_y as u32;
            constants.tile_height = tile_height as u32;
            queue.write_buffer(&constants_buffer, 0, bytemuck::bytes_of(&constants));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sky_tracer_dispatch_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sky_tracer_compute_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    config.width.div_ceil(8) as u32,
                    tile_height.div_ceil(8) as u32,
                    BAND_COUNT as u32,
                );
            }
            queue.submit([encoder.finish()]);
            dispatched += 1;
            if dispatched % 64 == 0 {
                device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|err| RenderError::Poll(err.to_string()))?;
            }
        }
        sample_offset += samples_this_dispatch;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sky_tracer_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_readback, 0, output_size);
    encoder.copy_buffer_to_buffer(
        &diagnostic_buffer,
        0,
        &diagnostic_readback,
        0,
        diagnostic_size,
    );
    queue.submit([encoder.finish()]);

    let diagnostics: Vec<u32> = read_buffer(&device, &diagnostic_readback)?;
    if diagnostics.first().copied().unwrap_or(0) != 0 {
        return Err(RenderError::Watchdog);
    }

    let mut values: Vec<f32> = read_buffer(&device, &output_readback)?;
    let inv_spp = 1.0 / config.spp as f32;
    for value in &mut values {
        *value = (*value * inv_spp).max(0.0);
    }
    let mut film = Film::new(config.width, config.height, BAND_COUNT);
    for pixel in 0..config.width * config.height {
        let start = pixel * BAND_COUNT;
        film.set_pixel_spectrum(pixel, &values[start..start + BAND_COUNT]);
    }
    Ok(film)
}

impl PackedScene {
    fn new(scene: &SceneData, config: &RenderConfig) -> Result<Self, RenderError> {
        let constants = GpuConstants {
            width: config.width as u32,
            height: config.height as u32,
            spp: config.spp as u32,
            direct_light_samples: config.direct_light_samples.max(1) as u32,
            sample_offset: 0,
            samples_this_dispatch: 0,
            tile_y: 0,
            tile_height: config.height as u32,
            atmosphere_len: scene.atmospheric_profile.len() as u32,
            aerosol_len: scene.aerosol_profile.len() as u32,
            majorant_layers: scene.majorant_grid.layer_count as u32,
            phase_bins: PHASE_BINS as u32,
            seed_lo: config.seed as u32,
            seed_hi: (config.seed >> 32) as u32,
            watchdog_limit: WATCHDOG_LIMIT,
            _pad0: 0,
            _pad1: 0,
            observer_altitude_km: config.observer_altitude_km,
            ground_radius_km: scene.planet.ground_radius_km,
            atmosphere_radius_km: scene.planet.atmosphere_radius_km,
            top_altitude_km: scene.majorant_grid.top_altitude_km,
            sun_x: scene.sun.direction.x,
            sun_y: scene.sun.direction.y,
            sun_z: scene.sun.direction.z,
            sun_angular_radius_rad: scene.sun.angular_radius_rad,
            sun_solid_angle_sr: scene.sun.solid_angle_sr,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };

        let bands = scene
            .bands
            .iter()
            .enumerate()
            .map(|(band, value)| GpuBand {
                center_nm: value.center_nm,
                lower_nm: value.lower_nm,
                upper_nm: value.upper_nm,
                ozone_cross_section_cm2: value.ozone_cross_section_cm2,
                solar_radiance_w_m2_sr: scene.solar_radiance_w_m2_sr(band),
                rayleigh_cross_section_m2: scene.rayleigh_cross_sections_m2[band],
            })
            .collect();

        let atmosphere = scene
            .atmospheric_profile
            .iter()
            .map(|point| GpuAtmospherePoint {
                altitude_km: point.altitude_km,
                temperature_k: point.temperature_k,
                air_cm3: point.air_cm3,
                ozone_cm3: point.ozone_cm3,
            })
            .collect();

        let aerosol = scene
            .aerosol_profile
            .iter()
            .map(|point| GpuAerosolPoint {
                altitude_km: point.altitude_km,
                mass_g_m3: point.mass_g_m3,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            })
            .collect();

        let aerosol_optics = scene
            .aerosol_optics
            .iter()
            .flat_map(|band| {
                band.iter().map(|optics| GpuAerosolOptics {
                    scattering_km_inv_per_g_m3: optics.scattering_km_inv_per_g_m3,
                    absorption_km_inv_per_g_m3: optics.absorption_km_inv_per_g_m3,
                })
            })
            .collect();

        let expected_majorants = BAND_COUNT * scene.majorant_grid.layer_count;
        if scene.majorant_grid.values_km_inv.len() != expected_majorants {
            return Err(RenderError::InvalidConfig(format!(
                "majorant grid length mismatch: expected {expected_majorants}, found {}",
                scene.majorant_grid.values_km_inv.len()
            )));
        }

        Ok(Self {
            constants,
            bands,
            atmosphere,
            aerosol,
            aerosol_optics,
            majorants: scene.majorant_grid.values_km_inv.clone(),
            phase_values: scene.phase_table.values().to_vec(),
        })
    }
}

fn storage_buffer<T: Pod>(device: &wgpu::Device, label: &'static str, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn readback_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<Vec<T>, RenderError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| RenderError::Poll(err.to_string()))?;
    receiver
        .recv()
        .map_err(|err| RenderError::BufferMap(err.to_string()))?
        .map_err(|err| RenderError::BufferMap(err.to_string()))?;

    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn byte_size<T>(len: usize) -> Result<u64, RenderError> {
    len.checked_mul(mem::size_of::<T>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| RenderError::InvalidConfig("buffer size overflow".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_scene_matches_cpu_table_shapes() {
        let config = RenderConfig {
            width: 4,
            height: 2,
            spp: 1,
            ..RenderConfig::default()
        };
        let scene =
            crate::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0).expect("scene");
        let packed = PackedScene::new(&scene, &config).expect("packed scene");

        assert_eq!(packed.bands.len(), BAND_COUNT);
        assert_eq!(packed.atmosphere.len(), scene.atmospheric_profile.len());
        assert_eq!(packed.aerosol.len(), scene.aerosol_profile.len());
        assert_eq!(packed.aerosol_optics.len(), BAND_COUNT * SPECIES_COUNT);
        assert_eq!(
            packed.majorants.len(),
            BAND_COUNT * scene.majorant_grid.layer_count
        );
        assert_eq!(
            packed.phase_values.len(),
            SPECIES_COUNT * BAND_COUNT * PHASE_BINS
        );
        assert_eq!(
            packed.aerosol_optics[3].absorption_km_inv_per_g_m3,
            scene.aerosol_optics[0][3].absorption_km_inv_per_g_m3
        );
        assert_eq!(
            packed.majorants[scene.majorant_grid.layer_count],
            scene.majorant_grid.values_km_inv[scene.majorant_grid.layer_count]
        );
    }
}
