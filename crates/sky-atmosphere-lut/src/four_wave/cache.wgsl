// The source atlas is immutable for this fixed medium. It spans all heights,
// solar angles and outgoing directions, independently of the current camera.
@group(0) @binding(6) var ms_read:texture_2d<f32>;
@group(0) @binding(7) var ms_sampler:sampler;
@group(0) @binding(8) var<storage,read> cache_nodes:array<vec4<f32>>;
@group(0) @binding(9) var ms_write:texture_storage_2d<rgba16float,write>;
@group(0) @binding(10) var sky_write:texture_storage_2d<rgba32float,write>;
@group(0) @binding(11) var sky_read:texture_2d<f32>;
const PHASE_COUNT:u32=24u;
const CONE_COUNT:u32=8u;
const SKY_SIZE:vec2<u32>=vec2<u32>(256u,192u);
const SKY_ROWS:u32=144u;

fn phase_coord(nu:f32)->f32 {
    return 0.5*acos(clamp(nu,-1.0,1.0))/PI+0.5*pow(max((1.0-nu)*0.5,0.0),1.0/3.0);
}
fn angular_shape(h:u32,s:u32,uv:vec2<f32>)->vec4<f32> {
    let pixel=vec2<f32>(f32(s*PHASE_COUNT),f32(h*CONE_COUNT))+uv*vec2<f32>(f32(PHASE_COUNT-1u),f32(CONE_COUNT-1u))+0.5;
    return textureSampleLevel(ms_read,ms_sampler,pixel/vec2<f32>(textureDimensions(ms_read)),0.0);
}
fn indirect(h:f32,mu:f32,mu_s:f32,nu:f32,c:Medium)->vec4<f32> {
    if (p.size_steps.w&1u)==0u{return vec4<f32>(0.0);}
    // Above the aerosol layer, the six Rayleigh moments are exact and cheaper
    // than a directional atlas. The original packed lower layers are discarded.
    if h>=35.0{return indirect_sh(h,mu,mu_s,nu,c);}
    var lo=0u;var hi=p.dims.x-1u;
    while hi-lo>1u {let mid=(lo+hi)/2u;if aux[p.offsets0.y+mid].x<=h{lo=mid;}else{hi=mid;}}
    let h0=aux[p.offsets0.y+lo].x;let h1=aux[p.offsets0.y+hi].x;
    let th=clamp((h-h0)/(h1-h0),0.0,1.0);
    let solar0=solar_coord(h0,mu_s)*f32(p.dims.y-1u);let solar1=solar_coord(h1,mu_s)*f32(p.dims.y-1u);
    let s0=min(u32(solar0),p.dims.y-2u);let s1=min(u32(solar1),p.dims.y-2u);
    let t0=solar0-f32(s0);let t1=solar1-f32(s1);
    let angle=acos(clamp((mu-mu_s*nu)/sqrt(max((1.0-mu_s*mu_s)*(1.0-nu*nu),1e-20)),-1.0,1.0))/PI;
    let uv=vec2<f32>(phase_coord(nu),angle);
    let shape0=mix(angular_shape(lo,s0,uv),angular_shape(lo,s0+1u,uv),t0);
    let shape1=mix(angular_shape(hi,s1,uv),angular_shape(hi,s1+1u,uv),t1);
    let floor_value=vec4<f32>(1e-30);
    let mean0=mix(log(max(aux[lo*p.dims.y+s0],floor_value)),log(max(aux[lo*p.dims.y+s0+1u],floor_value)),t0);
    let mean1=mix(log(max(aux[hi*p.dims.y+s1],floor_value)),log(max(aux[hi*p.dims.y+s1+1u],floor_value)),t1);
    var sigma=vec4<f32>(0.0);for(var k=0u;k<5u;k++){sigma+=c.scattering[k];}
    return max(mix(shape0,shape1,th)*exp(mix(mean0,mean1,th))*sigma,vec4<f32>(0.0));
}

@compute @workgroup_size(8,8) fn build_ms(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(ms_write);if any(id.xy>=size){return;}
    let layer=id.y/CONE_COUNT;let si=id.x/PHASE_COUNT;
    let h=aux[p.offsets0.y+layer].x;let mu_s=cache_nodes[layer*p.dims.y+si].x;
    let nu=cache_nodes[p.dims.x*p.dims.y+id.x%PHASE_COUNT].x;
    let angle=PI*f32(id.y%CONE_COUNT)/f32(CONE_COUNT-1u);
    let mu=clamp(mu_s*nu+sqrt(max((1.0-mu_s*mu_s)*(1.0-nu*nu),0.0))*cos(angle),-1.0,1.0);
    let c=medium(h);var sigma=vec4<f32>(0.0);for(var k=0u;k<5u;k++){sigma+=c.scattering[k];}
    let scale=source_stencil(h,mu_s).scale;
    let value=indirect_sh(h,mu,mu_s,nu,c)/max(sigma*scale,vec4<f32>(1e-30));
    textureStore(ms_write,vec2<i32>(id.xy),value);
}

// Three vertical charts split ground, lower sky and upper sky. Duplicated
// endpoints avoid interpolation across the ground discontinuity and keep the
// lower chart continuous when the observer crosses the atmosphere top.
// A smooth CDF resolves the physical horizon, limb and solar elevation.
fn elevation_bounds(chart:u32)->vec2<f32>{
    let hor=asin(horizon(p.view.w));
    if chart==2u{return vec2<f32>(-PI*0.5,hor);}
    if chart==1u{return vec2<f32>(0.0,PI*0.5);}
    // From space, the lower sky chart spans only the atmospheric shell.
    // Mixing a finite limb texel with vacuum in log space erases the thin limb.
    if p.view.w>=p.medium.x{return vec2<f32>(hor,-acos(clamp((p.sun.w+p.medium.x)/(p.sun.w+p.view.w),0.0,1.0)));}
    return vec2<f32>(hor,0.0);
}
fn interval_cdf(e:f32,c:f32,w:f32,bounds:vec2<f32>)->f32 {
    let low=atan((bounds.x-c)/w);return (atan((e-c)/w)-low)/max(atan((bounds.y-c)/w)-low,1e-10);
}
fn elevation_coord(e:f32,chart:u32)->f32 {
    let bounds=elevation_bounds(chart);let hor=asin(horizon(p.view.w));
    var outer=hor;
    if p.view.w>p.medium.x {
        let r=p.sun.w+p.view.w;let top=p.sun.w+p.medium.x;
        outer=-acos(clamp(top/r,0.0,1.0));
    }
    let x=clamp(e,bounds.x,bounds.y);
    return clamp(0.15*(x-bounds.x)/max(bounds.y-bounds.x,1e-10)
        +0.40*interval_cdf(x,hor,1.0*PI/180.0,bounds)
        +0.35*interval_cdf(x,p.sun.y,0.7*PI/180.0,bounds)
        +0.10*interval_cdf(x,outer,0.5*PI/180.0,bounds),0.0,1.0);
}
fn elevation_from_coord(u:f32,chart:u32)->f32 {
    let bounds=elevation_bounds(chart);var lo=bounds.x;var hi=bounds.y;
    for(var i=0u;i<24u;i++){let mid=(lo+hi)*0.5;if elevation_coord(mid,chart)<u{lo=mid;}else{hi=mid;}}
    return (lo+hi)*0.5;
}
@compute @workgroup_size(8,8) fn build_sky(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=SKY_SIZE){return;}
    let lower_rows=SKY_ROWS/2u;
    var chart=0u;var start=0u;var rows=lower_rows;
    if id.y>=SKY_ROWS{chart=2u;start=SKY_ROWS;rows=SKY_SIZE.y-SKY_ROWS;}
    else if id.y>=lower_rows{chart=1u;start=lower_rows;}
    if chart==1u&&p.view.w>=p.medium.x{textureStore(sky_write,vec2<i32>(id.xy),vec4<f32>(log(vec3<f32>(1e-30)),0.0));return;}
    let y=f32(id.y-start)/f32(rows-1u);
    let e=elevation_from_coord(y,chart);
    let hit=chart==2u;
    let u=f32(id.x)/f32(SKY_SIZE.x-1u);
    let cp=1.0-2.0*u*u;
    // Offset the duplicated horizon endpoints onto their own boundary branch.
    var mu=sin(e);let hor=horizon(p.view.w);
    if hit{mu=min(mu,hor-1e-7);}else{mu=max(mu,hor+1e-7);}
    let r=sqrt(max(1.0-mu*mu,0.0));
    let ray=vec3<f32>(r*sqrt(max(1.0-cp*cp,0.0)),mu,r*cp);
    let sun=vec3<f32>(0.0,sin(p.sun.y),cos(p.sun.y));
    let result=integrate(ray,sun);var rgb=vec3<f32>(0.0);
    for(var k=0u;k<4u;k++){rgb+=result.light[k]*p.rgb[k].xyz;}
    textureStore(sky_write,vec2<i32>(id.xy),vec4<f32>(log(max(rgb,vec3<f32>(1e-30))),0.0));
}
fn sky_sample(ray:vec3<f32>)->vec3<f32>{
    if p.view.w>=p.medium.x {
        let radius=p.sun.w+p.view.w;let b=radius*ray.y;
        let c=(p.view.w-p.medium.x)*(radius+p.sun.w+p.medium.x);
        if b>=0.0||b*b<=c{return vec3<f32>(0.0);}
    }
    let cp=clamp((sin(p.sun.x)*ray.x+cos(p.sun.x)*ray.z)/max(length(ray.xz),1e-10),-1.0,1.0);
    let u=sqrt(max((1.0-cp)*0.5,0.0));let hit=ground(p.view.w,ray.y);
    let lower_rows=SKY_ROWS/2u;var chart=0u;var start=0u;var rows=lower_rows;
    if hit{chart=2u;start=SKY_ROWS;rows=SKY_SIZE.y-SKY_ROWS;}
    else if ray.y>=0.0{chart=1u;start=lower_rows;}
    let y=elevation_coord(asin(clamp(ray.y,-1.0,1.0)),chart);
    let row=f32(start)+y*f32(rows-1u);let end=start+rows-1u;
    let x=u*f32(SKY_SIZE.x-1u);
    let ix=min(u32(x),SKY_SIZE.x-2u);let iy=clamp(u32(row),start,end-1u);
    let a=mix(textureLoad(sky_read,vec2<i32>(i32(ix),i32(iy)),0).xyz,textureLoad(sky_read,vec2<i32>(i32(ix+1u),i32(iy)),0).xyz,x-f32(ix));
    let b=mix(textureLoad(sky_read,vec2<i32>(i32(ix),i32(iy+1u)),0).xyz,textureLoad(sky_read,vec2<i32>(i32(ix+1u),i32(iy+1u)),0).xyz,x-f32(ix));
    if p.view.w>=p.medium.x&&!hit&&iy==lower_rows-2u {
        // At the outer limb radiance tends to zero with the atmospheric chord,
        // not exponentially. Log interpolation into a vacuum endpoint destroys
        // the thin shell even when the sky chart ends at the correct tangent.
        let previous=sin(elevation_from_coord(f32(lower_rows-2u)/f32(lower_rows-1u),0u));
        let previous2=sin(elevation_from_coord(f32(lower_rows-3u)/f32(lower_rows-1u),0u));
        let radius=p.sun.w+p.view.w;
        let c=(p.view.w-p.medium.x)*(radius+p.sun.w+p.medium.x);
        let d=max((radius*ray.y)*(radius*ray.y)-c,0.0);
        let d0=max((radius*previous)*(radius*previous)-c,1e-20);
        let d1=max((radius*previous2)*(radius*previous2)-c,d0+1e-10);
        let earlier=mix(textureLoad(sky_read,vec2<i32>(i32(ix),i32(iy-1u)),0).xyz,textureLoad(sky_read,vec2<i32>(i32(ix+1u),i32(iy-1u)),0).xyz,x-f32(ix));
        let normalized_change=exp(earlier-a)*sqrt(d0/d1)-vec3<f32>(1.0);
        return max(exp(a)*sqrt(clamp(d/d0,0.0,1.0))*(vec3<f32>(1.0)+normalized_change*((d-d0)/(d1-d0))),vec3<f32>(0.0));
    }
    return max(exp(mix(a,b,row-f32(iy)))-vec3<f32>(1e-30),vec3<f32>(0.0));
}
fn view_transmittance(ray:vec3<f32>)->vec4<f32>{
    var h=p.view.w;var mu=ray.y;if ground(h,mu){return vec4<f32>(0.0);}
    if h>p.medium.x {
        let r=p.sun.w+h;let b=r*mu;let c=(h-p.medium.x)*(r+p.sun.w+p.medium.x);let d=b*b-c;
        if b>=0.0||d<=0.0{return vec4<f32>(1.0);}
        mu=-sqrt(d)/(p.sun.w+p.medium.x);h=p.medium.x;
    }
    return sun_t(h,mu);
}
@compute @workgroup_size(8,8) fn project_sky(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=p.size_steps.xy){return;}
    let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(p.size_steps.xy)*2.0-1.0;
    let sy=sin(p.view.x);let cy=cos(p.view.x);let sp=sin(p.view.y);let cp=cos(p.view.y);
    let forward=vec3<f32>(sy*cp,sp,cy*cp);let right=vec3<f32>(cy,0.0,-sy);let up=cross(forward,right);
    let ray=normalize(forward+right*uv.x*tan(p.view.z*0.5)*f32(p.size_steps.x)/f32(p.size_steps.y)-up*uv.y*tan(p.view.z*0.5));
    let sun=vec3<f32>(sin(p.sun.x)*cos(p.sun.y),sin(p.sun.y),cos(p.sun.x)*cos(p.sun.y));
    var rgb=sky_sample(ray);let half_radius=sin(p.sun.z*0.5);
    if (p.size_steps.w&2u)!=0u&&!ground(p.view.w,ray.y)&&dot(ray-sun,ray-sun)<=4.0*half_radius*half_radius {
        let light=view_transmittance(ray)*p.solar/(4.0*PI*half_radius*half_radius);
        for(var k=0u;k<4u;k++){rgb+=light[k]*p.rgb[k].xyz;}
    }
    textureStore(output,vec2<i32>(id.xy),vec4<f32>(rgb,1.0));
}
