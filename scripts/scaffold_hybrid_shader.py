"""One-time extraction of the already validated geometry and sky-view charts.
The new crate owns the generated files; this is not part of its build process.
"""
from pathlib import Path
src=Path('crates/sky-atmosphere-lut/src/four_wave/runtime.wgsl').read_text()
out=Path('crates/sky-hybrid-atmosphere/src');out.mkdir(parents=True,exist_ok=True)
geometry=src[src.index('const PI:'):src.index('fn tau_load')]
medium=src[src.index('struct Medium'):src.index('struct Stencil')].replace('aux[','data[')
transport=src[src.index('struct Transport'):src.index('@compute @workgroup_size(8,8) fn render')]
transport=transport.replace('fn integrate(ray:vec3<f32>,sun:vec3<f32>)','fn integrate_at(ray:vec3<f32>,sun:vec3<f32>,initial_height:f32,segment:f32,steps:u32)')
transport=transport.replace('p.view.w','initial_height').replace('p.medium.z','segment').replace('p.size_steps.z','steps')
start=transport.index('    var sun_dirs:');end=transport.index('    var previous=0.0;',start)
transport=transport[:start]+'''    var solar_phase:array<vec4<f32>,5>;
    for(var k=0u;k<5u;k++){solar_phase[k]=phase(clamp(nu,-1.0,1.0),k);}
'''+transport[end:]
start=transport.index('        for(var j=0u;j<16u;j++)');end=transport.index('        let source=',start)
transport=transport[:start]+'''        for(var k=0u;k<5u;k++){direct+=c.scattering[k]*solar_phase[k];}
        direct*=sun_t(hp,dot(local_up,sun));
'''+transport[end:]
transport+='\nfn integrate(ray:vec3<f32>,sun:vec3<f32>)->Transport{return integrate_at(ray,sun,p.view.w,p.medium.z,p.size_steps.z);}\n'
render=src[src.index('@compute @workgroup_size(8,8) fn render'):]
(out/'transport.wgsl').write_text('// Stable height arithmetic and ray stepping adapted from the four-wave prototype.\n'+geometry+medium+transport+render)
cache=Path('crates/sky-atmosphere-lut/src/four_wave/cache.wgsl').read_text()
(out/'sky_view.wgsl').write_text('// Observer SkyView charts, independent of the fixed-medium solver.\nconst SKY_SIZE:vec2<u32>=vec2<u32>(256u,256u);\nconst SKY_ROWS:u32=192u;\n'+cache[cache.index('fn elevation_bounds'):])
