// Appended to bake.wgsl: use exactly the same mapping and interpolation as the solver.
struct ViewParams {
    size_band:vec4u,
    view:vec4f, // yaw, pitch, vertical FOV in radians, aspect
    sun_height:vec4f, // world Y-up sun direction, observer altitude km
    rgb_radius:vec4f, // linear RGB contribution per band, solar angular radius
    storage:vec4u, // 1: RGB channels with attenuated solar irradiance tables
}
@group(0) @binding(9) var<uniform> frame:ViewParams;
@group(0) @binding(10) var screen:texture_storage_2d<rgba32float,write>;
@group(0) @binding(11) var<storage,read_write> rgb_sum:array<vec4f>;

@compute @workgroup_size(8,8)
fn render_spectral(@builtin(global_invocation_id) id:vec3u) {
    if id.x>=frame.size_band.x || id.y>=frame.size_band.y { return; }
    let pixel=id.y*frame.size_band.x+id.x;
    let uv=(vec2f(id.xy)+vec2f(0.5))/vec2f(frame.size_band.xy);
    let yaw=frame.view.x;let pitch=frame.view.y;
    let forward=vec3f(sin(yaw)*cos(pitch),sin(pitch),cos(yaw)*cos(pitch));
    let right=vec3f(cos(yaw),0.0,-sin(yaw));let up=cross(forward,right);
    let xy=vec2f(uv.x*2.0-1.0,1.0-uv.y*2.0)*vec2f(frame.view.w,1.0)*tan(frame.view.z*0.5);
    let ray=normalize(forward+xy.x*right+xy.y*up);let sun=frame.sun_height.xyz;
    let h=frame.sun_height.w;let nu=clamp(dot(ray,sun),-1.0,1.0);
    var point=State(h,ray.y,sun.y,nu,hits_ground(h,ray.y));
    var intersects=true;
    if h>p.planet.y {
        let radius=p.planet.x+h;let top=p.planet.x+p.planet.y;
        let b=radius*ray.y;let c=(h-p.planet.y)*(radius+top);let disc=b*b-c;
        intersects=b<0.0 && disc>0.0;
        if intersects {
            let d=c/(-b+sqrt(disc));let mu=clamp((b+d)/top,-1.0,1.0);
            point=State(p.planet.y,mu,clamp((radius*sun.y+d*nu)/top,-1.0,1.0),nu,hits_ground(p.planet.y,mu));
        }
    }
    var value=0.0;
    if intersects {value=lookup(point,false);}
    let half_radius_sin=sin(frame.rgb_radius.w*0.5);
    if (!intersects || !point.ground) && dot(ray-sun,ray-sun)<=4.0*half_radius_sin*half_radius_sin {
        var solar_irradiance=p.planet.w;
        if intersects {
            if frame.storage.x==1u {solar_irradiance=tau_lookup(point.h,point.mu,false);}
            else {solar_irradiance*=sun_transmittance(point.h,point.mu);}
        }
        value+=solar_irradiance/(4.0*PI*half_radius_sin*half_radius_sin);
    }
    var rgb=value*frame.rgb_radius.xyz;
    if frame.size_band.z>0u { rgb+=rgb_sum[pixel].rgb; }
    rgb_sum[pixel]=vec4f(rgb,1.0);
    if frame.size_band.z+1u==frame.size_band.w { textureStore(screen,vec2i(id.xy),vec4f(rgb,1.0)); }
}
