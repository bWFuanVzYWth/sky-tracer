// Display only. Linear RGB running means and sample statistics stay untouched.
struct Display {
    image: vec4<u32>, // image width, height, viewport width, viewport height
    exposure: vec4<f32>, // exposure EV, manual sRGB encode, padding
}
@group(0) @binding(0) var<storage, read> film_mean: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> display: Display;
struct Vertex { @builtin(position) position: vec4<f32> }
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> Vertex {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return Vertex(vec4<f32>(positions[index], 0.0, 1.0));
}
fn srgb(linear: vec3<f32>) -> vec3<f32> {
    return select(12.92 * linear, 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055, linear > vec3<f32>(0.0031308));
}
@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = min(vec2<u32>(position.xy / vec2<f32>(display.image.zw) * vec2<f32>(display.image.xy)), display.image.xy - vec2<u32>(1u));
    let linear = film_mean[xy.y * display.image.x + xy.x].xyz;
    // Reinhard is a reversible viewing aid, never part of the reference data.
    var mapped = linear / (vec3<f32>(exp2(-display.exposure.x)) + linear);
    if display.exposure.y != 0.0 { mapped = srgb(mapped); }
    return vec4<f32>(mapped, 1.0);
}
