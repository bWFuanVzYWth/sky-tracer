// Film already contains the boundary radiance. Unsampled pixels show the same
// directional boundary immediately; no alpha overlay or second background sum.
struct Display {
    image:vec4<u32>, viewport:vec4<f32>, exposure:vec4<f32>,
    camera_forward:vec4<f32>,camera_right:vec4<f32>,camera_up:vec4<f32>,ambient:vec4<f32>
}
@group(0) @binding(0) var<storage,read> film_mean:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> display:Display;
@group(0) @binding(2) var sky:texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) index:u32)->@builtin(position) vec4<f32>{
    var p=array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(p[index],0.0,1.0);
}
fn environment(ray:vec3<f32>)->vec3<f32>{
    let n=vec2<i32>(textureDimensions(sky));
    let u=fract(atan2(ray.x,ray.z)/(2.0*3.141592653589793)+1.0);
    let v=acos(clamp(ray.y,-1.0,1.0))/3.141592653589793;
    let xy=vec2<f32>(u,v)*vec2<f32>(n)-0.5;let base=vec2<i32>(floor(xy));let t=fract(xy);
    let x0=(base.x+n.x)%n.x;let x1=(x0+1)%n.x;
    let y0=clamp(base.y,0,n.y-1);let y1=clamp(base.y+1,0,n.y-1);
    return mix(mix(textureLoad(sky,vec2<i32>(x0,y0),0).xyz,textureLoad(sky,vec2<i32>(x1,y0),0).xyz,t.x),
        mix(textureLoad(sky,vec2<i32>(x0,y1),0).xyz,textureLoad(sky,vec2<i32>(x1,y1),0).xyz,t.x),t.y);
}
fn srgb(v:vec3<f32>)->vec3<f32>{return select(12.92*v,1.055*pow(v,vec3<f32>(1.0/2.4))-0.055,v>vec3<f32>(0.0031308));}
@fragment fn fs_main(@builtin(position) pos:vec4<f32>)->@location(0) vec4<f32>{
    let uv=clamp((pos.xy-display.viewport.xy)/display.viewport.zw,vec2<f32>(0.0),vec2<f32>(1.0));
    let xy=min(vec2<u32>(uv*vec2<f32>(display.image.xy)),display.image.xy-vec2<u32>(1u));
    let mean=film_mean[xy.y*display.image.x+xy.x];var linear=display.ambient.xyz;
    if mean.w>0.0&&display.exposure.w==0.0{linear=mean.xyz;}
    else if display.exposure.z!=0.0{
        let ndc=uv*2.0-1.0;let tan_half=display.camera_forward.w;
        let ray=normalize(display.camera_forward.xyz+display.camera_right.xyz*ndc.x*tan_half
            -display.camera_up.xyz*ndc.y*tan_half*f32(display.image.y)/f32(display.image.x));
        linear=environment(ray);
    }
    linear=max(linear,vec3<f32>(0.0));var mapped=linear/(vec3<f32>(exp2(-display.exposure.x))+linear);
    if display.exposure.y!=0.0{mapped=srgb(mapped);}
    return vec4<f32>(mapped,1.0);
}
