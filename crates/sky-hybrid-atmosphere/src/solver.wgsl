struct Params {
    size_steps:vec4<u32>, view:vec4<f32>, sun:vec4<f32>, dims:vec4<u32>,
    offsets0:vec4<u32>, offsets1:vec4<u32>, medium:vec4<f32>, solar:vec4<f32>,
    rgb:array<vec4<f32>,4>, solve:vec4<u32>, extra:vec4<u32>,
    mapping:vec4<u32>,sky:vec4<u32>,optical_segments:array<vec4<f32>,6>,
    angular_mapping:vec4<f32>,
    importance:vec4<f32>,
    batch:vec4<u32>,
}
override SPECIALIZE:bool=false;
override FIXED_MAPPING_FLAGS:u32=63u;
override FIXED_PHASE_WEIGHT:f32=0.61;
override FIXED_CONE_WARP:f32=0.164285714;
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> data:array<vec4<f32>>;
@group(0) @binding(2) var tau_read:texture_2d<f32>;
@group(0) @binding(3) var source_read:texture_2d<f32>;
@group(0) @binding(4) var mean_read:texture_2d<f32>;
@group(0) @binding(5) var ground_read:texture_2d<f32>;
@group(0) @binding(6) var linear_sampler:sampler;
@group(0) @binding(7) var<storage,read_write> incoming:array<vec4<f32>>;
@group(0) @binding(8) var source_write:texture_storage_2d<rgba16float,write>;
@group(0) @binding(9) var mean_write:texture_storage_2d<rgba32float,write>;
@group(0) @binding(10) var ground_write:texture_storage_2d<rgba32float,write>;
@group(0) @binding(11) var tau_write:texture_storage_2d<rgba16float,write>;
@group(0) @binding(12) var output:texture_storage_2d<rgba32float,write>;
@group(0) @binding(13) var output_t:texture_storage_2d<rgba32float,write>;
@group(0) @binding(14) var sky_write:texture_storage_2d<rgba32float,write>;
@group(0) @binding(15) var sky_read:texture_2d<f32>;
@group(0) @binding(16) var<storage,read_write> directions:array<vec4<f32>>;
@group(0) @binding(17) var<storage,read_write> moments:array<vec4<f32>>;
@group(0) @binding(18) var high_read:texture_2d<f32>;
@group(0) @binding(19) var high_write:texture_storage_2d<rgba32float,write>;
@group(0) @binding(20) var<storage,read> sky_mapping:array<vec4<f32>>;
fn mapping_flags()->u32 {if SPECIALIZE{return FIXED_MAPPING_FLAGS;}return p.mapping.x;}
fn fitted_phase_weight()->f32 {if SPECIALIZE{return FIXED_PHASE_WEIGHT;}return p.angular_mapping.x;}
fn fitted_cone_warp()->f32 {if SPECIALIZE{return FIXED_CONE_WARP;}return p.angular_mapping.y;}

fn sun_t(h:f32,mu:f32)->vec4<f32>{
    if ground(h,mu){return vec4<f32>(0.0);}
    let x=clamp((mu-horizon(h))/(1.0-horizon(h)),0.0,1.0);
    let a=pow(x,1.0/3.0);var u=a;
    if (mapping_flags()&16u)==0u{let b=pow(1.0-x,1.0/3.0);u=a/max(a+b,1e-20);}
    let pixel=vec2<f32>(u*f32(p.dims.w-1u),height_coord(h)*f32(p.dims.z-1u))+0.5;
    return exp(-textureSampleLevel(tau_read,linear_sampler,pixel/vec2<f32>(f32(p.dims.w),f32(p.dims.z)),0.0));
}
fn phase_coord(nu:f32)->f32{
    let a=select(0.5,fitted_phase_weight(),(mapping_flags()&4u)!=0u);
    return a*acos(clamp(nu,-1.0,1.0))/PI+(1.0-a)*pow(max((1.0-nu)*0.5,0.0),1.0/3.0);
}
fn shape(hi:u32,si:u32,uv:vec2<f32>)->vec4<f32>{
    let pixel=vec2<f32>(f32(si*p.solve.y),f32(hi*p.solve.z))+uv*vec2<f32>(f32(p.solve.y-1u),f32(p.solve.z-1u))+0.5;
    return textureSampleLevel(source_read,linear_sampler,pixel/vec2<f32>(textureDimensions(source_read)),0.0);
}
fn log_mean(hi:u32,si:u32)->vec4<f32>{return textureLoad(mean_read,vec2<i32>(i32(si),i32(hi)),0);}
fn high_basis(mu:f32,mu_s:f32,nu:f32)->vec4<f32>{
    let y=clamp(mu,-1.0,1.0);let radial=sqrt(max(1.0-y*y,0.0));
    // Near a zenith Sun, cancellation in nu-mu*mu_s must not produce a
    // non-unit view vector. Azimuth is immaterial at the exact polar state.
    let z=clamp((nu-y*mu_s)/sqrt(max(1.0-mu_s*mu_s,1e-20)),-radial,radial);
    return vec4<f32>(1.0,y*y,z*z,y*z);
}
fn high_shape(hi:u32,si:u32,b:vec4<f32>)->vec4<f32>{
    var m:array<vec4<f32>,5>;
    for(var k=0u;k<5u;k++){m[k]=textureLoad(high_read,vec2<i32>(i32(si*5u+k),i32(hi+1u-p.offsets1.y)),0);}
    return (m[0]+m[1]*b.y+m[2]*b.z+m[3]*b.w)/max(dot(m[4],b),1e-20);
}
fn indirect(h:f32,mu:f32,mu_s:f32,nu:f32,c:Medium)->vec4<f32>{
    if (p.size_steps.w&1u)==0u{return vec4<f32>(0.0);}
    var lo=0u;var hi=p.dims.x-1u;
    while hi-lo>1u{let mid=(lo+hi)/2u;if data[mid].x<=h{lo=mid;}else{hi=mid;}}
    let h0=data[lo].x;let h1=data[hi].x;let th=clamp((h-h0)/(h1-h0),0.0,1.0);
    let elevation=asin(clamp(mu_s,-1.0,1.0));
    let s0=solar_layer(lo,elevation)*f32(p.dims.y-1u);let s1=solar_layer(hi,elevation)*f32(p.dims.y-1u);
    let i0=min(u32(s0),p.dims.y-2u);let i1=min(u32(s1),p.dims.y-2u);let t0=s0-f32(i0);let t1=s1-f32(i1);
    let brightness=exp(mix(mix(log_mean(lo,i0),log_mean(lo,i0+1u),t0),mix(log_mean(hi,i1),log_mean(hi,i1+1u),t1),th));
    if h>=35.0&&p.offsets1.y<p.dims.x{
        let b=high_basis(mu,mu_s,nu);
        let v=mix(mix(high_shape(lo,i0,b),high_shape(lo,i0+1u,b),t0),mix(high_shape(hi,i1,b),high_shape(hi,i1+1u,b),t1),th);
        return max(v*brightness*c.scattering[0],vec4<f32>(0.0));
    }
    let cp=clamp((mu-mu_s*nu)/sqrt(max((1.0-mu_s*mu_s)*(1.0-nu*nu),1e-20)),-1.0,1.0);
    var angle=0.0;
    if (mapping_flags()&8u)!=0u{let t=(1.0-cp)*0.5;angle=t+fitted_cone_warp()*t*(1.0-t)*(2.0*t-1.0);}else{angle=acos(cp)/PI;}
    let uv=vec2<f32>(phase_coord(nu),angle);
    let v=mix(mix(shape(lo,i0,uv),shape(lo,i0+1u,uv),t0),mix(shape(hi,i1,uv),shape(hi,i1+1u,uv),t1),th);
    var sigma=vec4<f32>(0.0);for(var k=0u;k<5u;k++){sigma+=c.scattering[k];}
    return max(v*brightness*sigma,vec4<f32>(0.0));
}
fn ground_light(mu_s:f32)->vec4<f32>{
    let x=solar_layer(0u,asin(clamp(mu_s,-1.0,1.0)))*f32(p.dims.y-1u);let i=min(u32(x),p.dims.y-2u);
    return mix(textureLoad(ground_read,vec2<i32>(i32(i),0),0),textureLoad(ground_read,vec2<i32>(i32(i+1u),0),0),x-f32(i));
}
@compute @workgroup_size(8,8) fn optical_depth(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.dims.w||id.y>=p.dims.z{return;}
    let h=data[p.mapping.z+id.y].x;let v=f32(id.x)/f32(p.dims.w-1u);let a=v*v*v;
    var x=a;if (mapping_flags()&16u)==0u{let b=(1.0-v)*(1.0-v)*(1.0-v);x=a/max(a+b,1e-20);}
    let mu=horizon(h)+(1.0-horizon(h))*x;let length=boundary(h,mu,false);
    let dt=length/f32(p.extra.w);var tau=vec4<f32>(0.0);
    for(var j=0u;j<p.extra.w;j++){tau+=medium(height_at(h,mu,(f32(j)+0.5)*dt)).extinction*dt;}
    textureStore(tau_write,vec2<i32>(id.xy),min(tau,vec4<f32>(80.0)));
}
@compute @workgroup_size(64) fn prepare_directions(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.solve.x||id.y>=p.batch.y*p.dims.y{return;}
    let state=data[p.extra.x+p.batch.x*p.dims.y+id.y];let h=state.x;let mu_s=state.y;
    let sun=vec3<f32>(0.0,mu_s,state.z);
    let n=p.solve.x/3u;var v:vec3<f32>;var w:f32;
    let sample_index=select(n+id.x%n,id.x,id.x<n);
    let q=data[p.extra.y+sample_index];
    if q.w==0.0{directions[id.y*p.solve.x+id.x]=vec4<f32>(0.0);return;}
    if id.x<n{
        let tangent=vec3<f32>(0.0,sun.z,-sun.y);
        v=q.x*tangent+q.y*vec3<f32>(1.0,0.0,0.0)+q.z*sun;w=q.w;
    }else{
        let hor=horizon(h);var low=-1.0;var high=hor;
        if id.x>=2u*n{low=hor;high=1.0;}
        let u=pow(q.x,p.medium.w);var mu=mix(hor,low,u);if id.x>=2u*n{mu=mix(hor,high,u);}
        let r=sqrt(max(1.0-mu*mu,0.0));v=vec3<f32>(r*q.z,mu,r*q.y);w=q.w*(high-low)*p.medium.w*pow(q.x,p.medium.w-1.0);
    }
    let d=0.0004+max(1.0-dot(v,sun),0.0);
    let sw=p.importance.x/(p.importance.x+d*d)*mix(1.0,p.importance.z,smoothstep(12.0,35.0,h));
    w*=select(1.0-sw,sw,id.x<n);
    // Only one reflection half is traced. The convolution evaluates both signs.
    directions[id.y*p.solve.x+id.x]=vec4<f32>(v,w);
}
@compute @workgroup_size(64) fn trace_incident(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.solve.x||id.y>=p.batch.y*p.dims.y{return;}
    let state_index=p.batch.x*p.dims.y+id.y;
    let state=data[p.extra.x+state_index];let q=directions[id.y*p.solve.x+id.x];
    if q.w==0.0{
        incoming[id.y*p.solve.x+id.x]=vec4<f32>(0.0);
        return;
    }
    let sun=vec3<f32>(0.0,state.y,state.z);let index=id.y*p.solve.x+id.x;
    let result=integrate_at(q.xyz,sun,state.x,0.0,p.size_steps.z);
    let light=result.light;
    incoming[index]=light;
}
@compute @workgroup_size(64) fn incident_moments(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.batch.y*p.dims.y{return;}
    let state_index=p.batch.x*p.dims.y+id.x;
    var m:array<vec4<f32>,7>;
    for(var j=0u;j<p.solve.x;j++){
        let q=directions[id.x*p.solve.x+j];let l=incoming[id.x*p.solve.x+j]*q.w;
        m[0]+=l;m[1]+=l*q.x*q.x;m[2]+=l*q.y*q.y;m[3]+=l*q.z*q.z;m[4]+=l*q.y*q.z;
        m[5]+=q.w*vec4<f32>(1.0,q.x*q.x,q.y*q.y,q.z*q.z);m[6].x+=q.w*q.y*q.z;
    }
    for(var k=0u;k<7u;k++){moments[id.x*7u+k]=m[k];}
    let hi=state_index/p.dims.y;
    if hi+1u>=p.offsets1.y{
        let scale=max(m[0]/(4.0*PI),vec4<f32>(1e-30));
        // x²=1-y²-z². Retain the exact Rayleigh quadratic in five texels.
        let packed=array<vec4<f32>,5>((m[0]+m[1])/scale,(m[2]-m[1])/scale,(m[3]-m[1])/scale,2.0*m[4]/scale,
            vec4<f32>(m[5].x+m[5].y,m[5].z-m[5].y,m[5].w-m[5].y,2.0*m[6].x));
        for(var k=0u;k<5u;k++){textureStore(high_write,vec2<i32>(i32((id.x%p.dims.y)*5u+k),i32(hi+1u-p.offsets1.y)),packed[k]);}
    }
    textureStore(mean_write,vec2<i32>(i32(state_index%p.dims.y),i32(hi)),log(max(m[0]/(4.0*PI),vec4<f32>(1e-30))));
}
@compute @workgroup_size(8,8) fn scattering_source(@builtin(global_invocation_id) id:vec3<u32>){
    let pixel=id.xy+vec2<u32>(0u,p.batch.x*p.solve.z);
    if any(pixel>=textureDimensions(source_write))||id.y>=p.batch.y*p.solve.z{return;}
    let hi=pixel.y/p.solve.z;let si=id.x/p.solve.y;let state_index=hi*p.dims.y+si;
    let local_state=(hi-p.batch.x)*p.dims.y+si;
    let state=data[p.extra.x+state_index];let h=state.x;let mu_s=state.y;
    let nu=data[p.extra.z+id.x%p.solve.y].x;let cone=data[p.mapping.w+id.y%p.solve.z];
    let sun=vec3<f32>(0.0,mu_s,state.z);let tangent=vec3<f32>(0.0,sun.z,-sun.y);
    let v=sun*nu+sqrt(max(1.0-nu*nu,0.0))*(tangent*cone.x+vec3<f32>(cone.y,0.0,0.0));
    let coeff=p.offsets0.x+hi*6u;let mi=local_state*7u;
    let ray=moments[mi]+moments[mi+1u]*v.x*v.x+moments[mi+2u]*v.y*v.y+moments[mi+3u]*v.z*v.z+2.0*moments[mi+4u]*v.y*v.z;
    let weights=moments[mi+5u];let normalization=weights.x+weights.y*v.x*v.x+weights.z*v.y*v.y+weights.w*v.z*v.z+2.0*moments[mi+6u].x*v.y*v.z;
    var source=data[coeff]*ray/max(normalization,1e-20);
    // Rayleigh is exactly degree two: no directional phase loop above aerosols.
    if h<35.0 {
        var sums:array<vec4<f32>,4>;var norms:array<vec4<f32>,4>;
        for(var j=0u;j<p.solve.x;j++){
            let q=directions[local_state*p.solve.x+j];let l=incoming[local_state*p.solve.x+j];
            let a=q.y*v.y+q.z*v.z;let b=q.x*v.x;
            // Geometry is common to all four species and spectral lanes.
            let positive=phase_stencil(clamp(a+b,-1.0,1.0));
            let negative=phase_stencil(clamp(a-b,-1.0,1.0));
            for(var species=0u;species<4u;species++){
                let weight=(tabulated_phase(positive,species)+tabulated_phase(negative,species))*(0.5*q.w);
                sums[species]+=l*weight;norms[species]+=weight;
            }
        }
        // Match the independently integrated mass of the tabulated phase.
        // This removes energy drift but not angular undersampling.
        for(var species=0u;species<4u;species++){source+=data[coeff+species+1u]*data[p.offsets1.x+species]*sums[species]/max(norms[species],vec4<f32>(1e-20));}
    }
    let sigma=data[coeff+5u];
    let mean=exp(log_mean(hi,si));
    textureStore(source_write,vec2<i32>(pixel),max(source/max(mean*sigma,vec4<f32>(1e-30)),vec4<f32>(0.0)));
}
@compute @workgroup_size(64) fn ground_irradiance(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.dims.y||p.batch.x!=0u{return;}
    let mu_s=data[p.extra.x+id.x].y;var irradiance=sun_t(0.0,mu_s)*max(mu_s,0.0)*p.solar;
    if p.solve.w>0u{for(var j=0u;j<p.solve.x;j++){
        let q=directions[id.x*p.solve.x+j];irradiance+=incoming[id.x*p.solve.x+j]*(q.w*max(q.y,0.0));
    }}
    textureStore(ground_write,vec2<i32>(i32(id.x),0),irradiance*(p.medium.y/PI));
}
