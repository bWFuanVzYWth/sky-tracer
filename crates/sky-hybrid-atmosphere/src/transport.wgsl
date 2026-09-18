// Stable height arithmetic and ray stepping adapted from the four-wave prototype.
const PI:f32=3.141592653589793;
fn horizon(h:f32)->f32 {return -sqrt(max(h*(2.0*p.sun.w+h),0.0))/(p.sun.w+h);}
fn ground(h:f32,mu:f32)->bool {return mu<0.0&&mu<=horizon(h);}
fn height_at(h:f32,mu:f32,d:f32)->f32 {
    let dh=d*(d+2.0*(p.sun.w+h)*mu);
    return max(0.0,h+dh/(sqrt(max((p.sun.w+h)*(p.sun.w+h)+dh,0.0))+p.sun.w+h));
}
fn boundary(h:f32,mu:f32,hit:bool)->f32 {
    let b=(p.sun.w+h)*mu;
    if hit {let c=h*(2.0*p.sun.w+h);return max(0.0,c/max(-b+sqrt(max(b*b-c,0.0)),1e-20));}
    let c=(p.medium.x-h)*(2.0*p.sun.w+p.medium.x+h);let root=sqrt(max(b*b+c,0.0));
    if b>0.0{return max(0.0,c/max(root+b,1e-20));}return max(0.0,-b+root);
}
fn height_coord(h:f32)->f32 {
    if (mapping_flags()&16u)!=0u {
        let x=sqrt(clamp(h,0.0,p.medium.x));var j=0u;
        for(var i=0u;i<5u;i++){if x>p.optical_segments[i].x{j=i+1u;}}
        let c=p.optical_segments[j];return clamp(x*c.y+c.z,0.0,1.0);
    }
    let scale=sqrt(0.01/p.medium.x);
    return 0.35*log(1.0+sqrt(clamp(h/p.medium.x,0.0,1.0))/scale)/log(1.0+1.0/scale)+0.65*clamp(h/p.medium.x,0.0,1.0);
}
fn atan_cdf(e:f32,c:f32,w:f32)->f32 {
    let lo=atan((-PI*0.5-c)/w);return (atan((e-c)/w)-lo)/(atan((PI*0.5-c)/w)-lo);
}
fn solar_coord(h:f32,mu:f32)->f32 {
    let e=asin(clamp(mu,-1.0,1.0));let hor=asin(horizon(h));
    // Same physical solar-elevation CDF as the frozen reference.
    let cap_scale=5.0*PI/180.0;
    let d=PI*0.5-e;
    let cap=1.0-sqrt(d/(d+cap_scale))/sqrt(PI/(PI+cap_scale));
    return clamp(0.25*(e/PI+0.5)+0.20*atan_cdf(e,hor,2.0*PI/180.0)+0.20*atan_cdf(e,hor-6.0*PI/180.0,6.0*PI/180.0)+0.15*atan_cdf(e,-hor,0.5*PI/180.0)+0.20*cap,0.0,1.0);
}
fn solar_layer(layer:u32,e:f32)->f32 {
    if (mapping_flags()&2u)==0u{return solar_coord(data[layer].x,sin(e));}
    let offset=p.mapping.y+4u*layer;let base=data[offset];let d=max(PI*0.5-e,0.0);
    var y=base.x*e+base.y-base.z*sqrt(d/(d+0.0872664626));
    for(var j=1u;j<=3u;j++){let c=data[offset+j];let x=e-c.x;y+=c.z*x/(abs(x)+c.y);}
    return clamp(y,0.0,1.0);
}
struct Medium { extinction:vec4<f32>, scattering:array<vec4<f32>,5> }
fn medium(h:f32)->Medium {
    var lo=0u;var hi=p.offsets1.z-1u;
    while hi-lo>1u {let mid=(lo+hi)/2u;if data[p.offsets0.z+7u*mid].x<=h{lo=mid;}else{hi=mid;}}
    let a=p.offsets0.z+lo*7u;let b=p.offsets0.z+hi*7u;
    let t=clamp((h-data[a].x)/(data[b].x-data[a].x),0.0,1.0);
    var result:Medium;result.extinction=mix(data[a+1u],data[b+1u],t);
    for(var j=0u;j<5u;j++){result.scattering[j]=mix(data[a+2u+j],data[b+2u+j],t);}return result;
}
struct PhaseStencil { index:u32, fraction:f32 }
fn phase_stencil(mu:f32)->PhaseStencil{
    let x=clamp(pow(max((1.0-mu)*0.5,0.0),1.0/3.0)*1024.0-0.5,0.0,1023.0);
    let i=min(u32(x),1022u);return PhaseStencil(i,x-f32(i));
}
fn tabulated_phase(stencil:PhaseStencil,species:u32)->vec4<f32>{
    let offset=p.offsets0.w+species*1024u+stencil.index;
    return mix(data[offset],data[offset+1u],stencil.fraction);
}
struct Transport { light:vec4<f32>, transmittance:vec4<f32> }
fn integrate_at(ray:vec3<f32>,sun:vec3<f32>,initial_height:f32,segment:f32,steps:u32)->Transport {
    var result:Transport;result.light=vec4<f32>(0.0);result.transmittance=vec4<f32>(1.0);
    var h=initial_height;var mu=ray.y;var mu_s=sun.y;let nu=dot(ray,sun);var entry=0.0;
    var up=vec3<f32>(0.0,1.0,0.0);
    if h>p.medium.x {
        let radius=p.sun.w+h;let b=radius*mu;let c=(h-p.medium.x)*(radius+p.sun.w+p.medium.x);let disc=b*b-c;
        if b>=0.0||disc<=0.0{return result;}entry=c/(-b+sqrt(disc));
        if segment>0.0&&entry>=segment{return result;}
        up=normalize(up*radius+ray*entry);h=p.medium.x;mu=dot(ray,up);mu_s=dot(sun,up);
    }
    let hit=ground(h,mu);let full_length=boundary(h,mu,hit);
    var length=full_length;if segment>0.0{length=min(length,max(segment-entry,0.0));}
    let count=steps;
    // Concentrate both sides of a grazing path near its minimum height.
    let b=(p.sun.w+h)*mu;let middle=clamp(-b,0.0,length);let hmin=height_at(h,mu,middle);let hend=height_at(h,mu,length);
    let scale=max(p.importance.y,1e-10);
    var before=0.0;var after=0.0;
    if p.importance.y>0.0{before=log(1.0+max(h-hmin,0.0)/scale);after=log(1.0+max(hend-hmin,0.0)/scale);}
    var split=u32(round(f32(count)*before/max(before+after,1e-20)));
    if middle<=0.0{split=0u;}else if middle>=length{split=count;}else{split=clamp(split,1u,count-1u);}
    var solar_phase:array<vec4<f32>,5>;
    let cosine=clamp(nu,-1.0,1.0);let stencil=phase_stencil(cosine);
    solar_phase[0]=vec4<f32>(3.0*(1.0+cosine*cosine)/(16.0*PI));
    for(var k=0u;k<4u;k++){solar_phase[k+1u]=tabulated_phase(stencil,k);}
    var previous=0.0;
    for(var i=1u;i<=count;i++) {
        var edge=length*f32(i)/f32(count);
        if before+after>=1e-5&&i<count {
            if i==split{edge=middle;}else{
                let incoming=i<split;var lh=after*f32(i-split)/max(f32(count-split),1.0);
                if incoming{lh=before*(1.0-f32(i)/f32(split));}
                let hh=hmin+scale*(exp(lh)-1.0);let c=(h-hh)*(2.0*p.sun.w+h+hh);let root=sqrt(max(b*b-c,0.0));
                if incoming{edge=c/max(-b+root,1e-20);}else if b>0.0{edge=-c/max(root+b,1e-20);}else{edge=-b+root;}
                edge=clamp(edge,previous,length);
            }
        }
        let dt=edge-previous;let d=(edge+previous)*0.5;previous=edge;
        let hp=height_at(h,mu,d);let local_up=(up*(p.sun.w+h)+ray*d)/(p.sun.w+hp);
        let c=medium(hp);var direct=vec4<f32>(0.0);
            for(var k=0u;k<5u;k++){direct+=c.scattering[k]*solar_phase[k];}
            direct*=sun_t(hp,dot(local_up,sun));direct*=p.solar;
        var source=direct;
        // The first iteration has an exactly zero previous volume source.
        if p.solve.w!=1u{source+=indirect(hp,dot(local_up,ray),dot(local_up,sun),nu,c);}
        let tau=c.extinction*dt;let trans=exp(-tau);
        let integral=select((vec4<f32>(1.0)-trans)/max(c.extinction,vec4<f32>(1e-30)),dt*(vec4<f32>(1.0)-tau*0.5+tau*tau/6.0),tau<vec4<f32>(0.001));
        result.light+=result.transmittance*source*integral;result.transmittance*=trans;
    }
    if hit&&length>=full_length {let end_up=(up*(p.sun.w+h)+ray*length)/p.sun.w;result.light+=result.transmittance*ground_light(dot(end_up,sun));}
    return result;
}

fn integrate(ray:vec3<f32>,sun:vec3<f32>)->Transport{return integrate_at(ray,sun,p.view.w,p.medium.z,p.size_steps.z);}
fn camera_ray(id:vec2<u32>)->vec3<f32>{
    let uv=(vec2<f32>(id)+0.5)/vec2<f32>(p.size_steps.xy)*2.0-1.0;
    let sy=sin(p.view.x);let cy=cos(p.view.x);let sp=sin(p.view.y);let cp=cos(p.view.y);
    let forward=vec3<f32>(sy*cp,sp,cy*cp);let right=vec3<f32>(cy,0.0,-sy);let up=cross(forward,right);
    return normalize(forward+right*uv.x*tan(p.view.z*0.5)*f32(p.size_steps.x)/f32(p.size_steps.y)-up*uv.y*tan(p.view.z*0.5));
}
fn world_sun()->vec3<f32>{return vec3<f32>(sin(p.sun.x)*cos(p.sun.y),sin(p.sun.y),cos(p.sun.x)*cos(p.sun.y));}
@compute @workgroup_size(8,8) fn render(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=p.size_steps.xy){return;}
    let ray=camera_ray(id.xy);let sun=world_sun();
    let result=integrate(ray,sun);var light=result.light;var rgb=vec3<f32>(0.0);
    let half_radius=sin(p.sun.z*0.5);
    if (p.size_steps.w&2u)!=0u&&p.medium.z<=0.0&&!ground(p.view.w,ray.y)&&dot(ray-sun,ray-sun)<=4.0*half_radius*half_radius {
        light+=result.transmittance*p.solar/(4.0*PI*half_radius*half_radius);
    }
    for(var i=0u;i<4u;i++){rgb+=light[i]*p.rgb[i].xyz;}
    var value=vec4<f32>(rgb,1.0);if (p.size_steps.w&4u)!=0u{value=light;}
    textureStore(output,vec2<i32>(id.xy),value);textureStore(output_t,vec2<i32>(id.xy),result.transmittance);
}
