//! Small decode-only GPU check of the CPU-generated payload; no transport work.
use sky_atmosphere_lut::{
    Result, asset::Manifest, packed::PackedLut, renderer::shader_source, solver::GpuBaker,
};
use wgpu::util::DeviceExt;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let path = std::path::Path::new(args.get(1).ok_or("packed LUT")?);
    let m = Manifest::open(path)?;
    let lut = PackedLut::read(&m, path)?;
    let gpu = GpuBaker::new()?;
    let device = gpu.device();
    let queue = gpu.queue();
    let mut shader = shader_source(true);
    shader.push_str("\n@group(0) @binding(14) var<storage,read> indices:array<u32>;\n@group(0) @binding(15) var<storage,read_write> decoded:array<f32>;\n@compute @workgroup_size(64) fn validate_decoder(@builtin(global_invocation_id) id:vec3u) {if id.x<arrayLength(&indices) {decoded[id.x]=packed_radiance(indices[id.x]);}}\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("packed decode validation"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("packed decode validation"),
        layout: None,
        module: &module,
        entry_point: Some("validate_decoder"),
        compilation_options: Default::default(),
        cache: None,
    });
    let upload = |data: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let packed_data = upload(&lut.data);
    let block_map = upload(&lut.map);
    let mut seed = 91397u32;
    let mut samples: Vec<u32> = (0..65536)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed % m.config.scattering_len() as u32
        })
        .collect();
    samples.extend(0..512);
    samples.extend((m.config.scattering_len() - 512..m.config.scattering_len()).map(|v| v as u32));
    let indices = upload(&samples);
    let size = samples.len() as u64 * 4;
    let frame = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let entries: Vec<_> = [
        (9, &frame),
        (12, &packed_data),
        (13, &block_map),
        (14, &indices),
        (15, &output),
    ]
    .into_iter()
    .map(|(binding, b)| wgpu::BindGroupEntry {
        binding,
        resource: b.as_entire_binding(),
    })
    .collect();
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut mismatches = 0;
    for c in 0..3 {
        let mut params = [0u32; 20];
        params[2] = c as u32;
        params[17] = lut.block_texels as u32;
        queue.write_buffer(&frame, 0, bytemuck::cast_slice(&params));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(samples.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let mapped = staging.slice(..).get_mapped_range();
        for (&i, bits) in samples.iter().zip(mapped.as_chunks::<4>().0) {
            if u32::from_le_bytes(*bits) != lut.fetch(i as usize, c).to_bits() {
                mismatches += 1;
            }
        }
        drop(mapped);
        staging.unmap();
    }
    let report = serde_json::json!({"adapter":gpu.adapter_name,"samples":samples.len()*3,"bit_mismatches":mismatches,"transport_dispatched":false});
    println!("{report}");
    if let Some(out) = args.get(2) {
        std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    }
    if mismatches != 0 {
        return Err("CPU/GPU packed decoder mismatch".into());
    }
    Ok(())
}
