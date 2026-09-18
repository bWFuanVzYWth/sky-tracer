//! Solar illumination variants share the same medium, transport and MS source.
//! The visible solar disk is independent of this choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, clap::ValueEnum)]
pub enum SunModel {
    Finite16,
    /// Four equal-weight beams match disk second moments in the small-angle limit.
    /// Retained finite-disk comparison mode.
    Finite4,
    /// Dense disk quadrature for convergence checks, never the real-time default.
    Finite64,
    /// Production approximation: one parallel illumination direction.
    /// The visible solar disk is still rendered at its physical angular size.
    Parallel,
    PhaseAveraged,
}

pub const DEFAULT_SUN_MODEL: SunModel = SunModel::Parallel;

pub fn shader_for_model(source: &str, model: SunModel) -> String {
    if model == SunModel::Finite16 {
        return source.to_owned();
    }
    if model == SunModel::Finite4 {
        return source
            .replace("array<vec3<f32>,16>", "array<vec3<f32>,4>")
            .replace("array<vec4<f32>,80>", "array<vec4<f32>,20>")
            .replace("array<f32,16>", "array<f32,4>")
            .replace("j<16u", "j<4u")
            .replace("nodes[j/4u]", "0.5")
            .replace("weights[j/4u]*0.25", "0.25");
    }
    if model == SunModel::Finite64 {
        return source
            .replace("array<vec3<f32>,16>", "array<vec3<f32>,64>")
            .replace("array<vec4<f32>,80>", "array<vec4<f32>,320>")
            .replace("array<f32,16>", "array<f32,64>")
            .replace("j<16u", "j<64u")
            .replace("nodes[j/4u]", "nodes[j/16u]")
            .replace("weights[j/4u]*0.25", "weights[j/16u]*0.0625")
            .replace("(f32(j%4u)+0.5)*PI*0.5", "(f32(j%16u)+0.5)*PI*0.125");
    }
    if model == SunModel::PhaseAveraged {
        return source
            .replace("var sun_dirs:array<vec3<f32>,16>;var phases:array<vec4<f32>,80>;var disk_weights:array<f32,16>;", "var solar_phase:array<vec4<f32>,5>;")
            .replace("sun_dirs[j]=sun*cm+(tx*cos(phi)+ty*sin(phi))*sm;disk_weights[j]=weights[j/4u]*0.25;", "let direction=sun*cm+(tx*cos(phi)+ty*sin(phi))*sm;let weight=weights[j/4u]*0.25;")
            .replace("phases[j*5u+k]=phase(clamp(dot(ray,sun_dirs[j]),-1.0,1.0),k);", "solar_phase[k]+=phase(clamp(dot(ray,direction),-1.0,1.0),k)*weight;")
            .replace("        for(var j=0u;j<16u;j++) {\n            var phase_weight=vec4<f32>(0.0);for(var k=0u;k<5u;k++){phase_weight+=c.scattering[k]*phases[j*5u+k];}\n            direct+=sun_t(hp,dot(local_up,sun_dirs[j]))*phase_weight*disk_weights[j];\n        }", "        for(var k=0u;k<5u;k++){direct+=c.scattering[k]*solar_phase[k];}\n        direct*=sun_t(hp,dot(local_up,sun));");
    }
    let start = source
        .find("    var helper=vec3<f32>")
        .expect("solar quadrature start");
    let end = source[start..]
        .find("    var previous=0.0;")
        .expect("solar quadrature end")
        + start;
    let mut result = String::from(&source[..start]);
    result.push_str("    sun_dirs[0]=sun;disk_weights[0]=1.0;\n    for(var k=0u;k<5u;k++){phases[k]=phase(clamp(dot(ray,sun),-1.0,1.0),k);}\n");
    result.push_str(&source[end..]);
    result.replace("for(var j=0u;j<16u;j++)", "for(var j=0u;j<1u;j++)")
}
