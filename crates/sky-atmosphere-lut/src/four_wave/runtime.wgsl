struct Params {
    size_steps:vec4<u32>, view:vec4<f32>, sun:vec4<f32>, dims:vec4<u32>,
    offsets0:vec4<u32>, offsets1:vec4<u32>, medium:vec4<f32>, solar:vec4<f32>,
    rgb:array<vec4<f32>,4>,
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> packed:array<u32>;
@group(0) @binding(2) var<storage,read> aux:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> optical:array<u32>;
@group(0) @binding(4) var output:texture_storage_2d<rgba32float,write>;
@group(0) @binding(5) var output_t:texture_storage_2d<rgba32float,write>;
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
fn tau_load(i:u32)->vec4<f32>{return vec4<f32>(unpack2x16float(optical[2u*i]),unpack2x16float(optical[2u*i+1u]));}
fn sun_t(h:f32,mu:f32)->vec4<f32>{
    if ground(h,mu){return vec4<f32>(0.0);}
    let x=clamp((mu-horizon(h))/(1.0-horizon(h)),0.0,1.0);
    let a=pow(x,1.0/3.0);let b=pow(1.0-x,1.0/3.0);
    let v=a/max(a+b,1e-20)*f32(p.dims.w-1u);let u=height_coord(h)*f32(p.dims.z-1u);
    let iy=min(u32(u),p.dims.z-2u);let ix=min(u32(v),p.dims.w-2u);
    let bottom=mix(tau_load(iy*p.dims.w+ix),tau_load(iy*p.dims.w+ix+1u),v-f32(ix));
    let top=mix(tau_load((iy+1u)*p.dims.w+ix),tau_load((iy+1u)*p.dims.w+ix+1u),v-f32(ix));
    return exp(-mix(bottom,top,u-f32(iy)));
}
struct Medium { extinction:vec4<f32>, scattering:array<vec4<f32>,5> }
fn medium(h:f32)->Medium {
    var lo=0u;var hi=p.offsets1.z-1u;
    while hi-lo>1u {let mid=(lo+hi)/2u;if aux[p.offsets0.z+7u*mid].x<=h{lo=mid;}else{hi=mid;}}
    let a=p.offsets0.z+lo*7u;let b=p.offsets0.z+hi*7u;
    let t=clamp((h-aux[a].x)/(aux[b].x-aux[a].x),0.0,1.0);
    var result:Medium;result.extinction=mix(aux[a+1u],aux[b+1u],t);
    for(var j=0u;j<5u;j++){result.scattering[j]=mix(aux[a+2u+j],aux[b+2u+j],t);}return result;
}
fn phase(mu:f32,s:u32)->vec4<f32>{
    if s==0u{return vec4<f32>(3.0*(1.0+mu*mu)/(16.0*PI));}
    let x=clamp(pow(max((1.0-mu)*0.5,0.0),1.0/3.0)*1024.0-0.5,0.0,1023.0);
    let i=min(u32(x),1022u);let offset=p.offsets0.w+(s-1u)*1024u;
    return mix(aux[offset+i],aux[offset+i+1u],x-f32(i));
}
struct Stencil { address:vec4<u32>,t0:f32,t1:f32,th:f32,degree:u32,scale:vec4<f32> }
fn source_stencil(h:f32,mu_s:f32)->Stencil {
    var lo=0u;var hi=p.dims.x-1u;
    while hi-lo>1u {let mid=(lo+hi)/2u;if aux[p.offsets0.y+mid].x<=h {lo=mid;}else{hi=mid;}}
    let a=aux[p.offsets0.y+lo];let b=aux[p.offsets0.y+hi];
    let c0=solar_coord(a.x,mu_s)*f32(p.dims.y-1u);let c1=solar_coord(b.x,mu_s)*f32(p.dims.y-1u);
    let s0=min(u32(c0),p.dims.y-2u);let s1=min(u32(c1),p.dims.y-2u);
    var s:Stencil;
    s.t0=c0-f32(s0);s.t1=c1-f32(s1);s.th=clamp((h-a.x)/(b.x-a.x),0.0,1.0);
    s.degree=select(16u,2u,h>=35.0);
    let stride0=(u32(a.y)+1u)*(u32(a.y)+2u);let stride1=(u32(b.y)+1u)*(u32(b.y)+2u);
    s.address=vec4<u32>(u32(a.z)+s0*stride0,u32(a.z)+(s0+1u)*stride0,u32(b.z)+s1*stride1,u32(b.z)+(s1+1u)*stride1);
    let base0=lo*p.dims.y+s0;let base1=hi*p.dims.y+s1;
    let floor_value=vec4<f32>(1e-30);
    let v0=mix(log(max(aux[base0],floor_value)),log(max(aux[base0+1u],floor_value)),s.t0);
    let v1=mix(log(max(aux[base1],floor_value)),log(max(aux[base1+1u],floor_value)),s.t1);
    s.scale=exp(mix(v0,v1,s.th));return s;
}
fn moment(offset:u32)->vec4<f32>{
    return vec4<f32>(unpack2x16float(packed[offset]),unpack2x16float(packed[offset+1u]));
}
fn coefficient(s:Stencil,i:u32)->vec4<f32>{
    let address=s.address+vec4<u32>(2u*i);
    return mix(mix(moment(address.x),moment(address.y),s.t0),mix(moment(address.z),moment(address.w),s.t1),s.th);
}
fn indirect(h:f32,mu:f32,mu_s:f32,nu:f32,c:Medium)->vec4<f32>{
    if (p.size_steps.w&1u)==0u{return vec4<f32>(0.0);}
    let s=source_stencil(h,mu_s);
    let radial=sqrt(max(1.0-mu*mu,0.0));
    let sun_radial=sqrt(max(1.0-mu_s*mu_s,0.0));
    let cp=clamp((nu-mu*mu_s)/max(radial*sun_radial,1e-10),-1.0,1.0);
    let sp=sqrt(max(1.0-cp*cp,0.0));
    var weights:array<vec4<f32>,17>;
    for(var l=0u;l<=s.degree;l++) {
        weights[l]=vec4<f32>(0.0);
        for(var k=0u;k<5u;k++){weights[l]+=c.scattering[k]*aux[p.offsets1.x+l*5u+k];}
    }
    var result=vec4<f32>(0.0);var diagonal=1.0/sqrt(4.0*PI);var cm=1.0;var sm=0.0;
    for(var m=0u;m<=s.degree;m++) {
        if m>0u {
            if m==1u{diagonal*=sqrt(3.0)*radial;}else{diagonal*=sqrt(f32(2u*m+1u)/f32(2u*m))*radial;}
            let next=cm*cp-sm*sp;sm=sm*cp+cm*sp;cm=next;
        }
        result+=weights[m]*coefficient(s,m*(m+1u)/2u+m)*(diagonal*cm);
        var previous2=0.0;var previous=diagonal;
        for(var l=m+1u;l<=s.degree;l++) {
            var y=sqrt(f32(2u*m+3u))*mu*previous;
            if l>m+1u {let lf=f32(l);let mf=f32(m);let b=lf-1.0;y=sqrt((4.0*lf*lf-1.0)/(lf*lf-mf*mf))*(mu*previous-sqrt((b*b-mf*mf)/(4.0*b*b-1.0))*previous2);}
            result+=weights[l]*coefficient(s,l*(l+1u)/2u+m)*(y*cm);
            previous2=previous;previous=y;
        }
    }
    return max(result*s.scale,vec4<f32>(0.0));
}
fn ground_light(mu_s:f32)->vec4<f32>{let x=solar_coord(0.0,mu_s)*f32(p.offsets1.w-1u);let i=min(u32(x),p.offsets1.w-2u);return mix(aux[p.offsets1.y+i],aux[p.offsets1.y+i+1u],x-f32(i))*p.medium.y/PI;}
struct Transport { light:vec4<f32>, transmittance:vec4<f32> }
fn integrate(ray:vec3<f32>,sun:vec3<f32>)->Transport {
    var result:Transport;result.light=vec4<f32>(0.0);result.transmittance=vec4<f32>(1.0);
    var h=p.view.w;var mu=ray.y;var mu_s=sun.y;let nu=dot(ray,sun);var entry=0.0;
    var up=vec3<f32>(0.0,1.0,0.0);
    if h>p.medium.x {
        let radius=p.sun.w+h;let b=radius*mu;let c=(h-p.medium.x)*(radius+p.sun.w+p.medium.x);let disc=b*b-c;
        if b>=0.0||disc<=0.0{return result;}entry=c/(-b+sqrt(disc));
        if p.medium.z>0.0&&entry>=p.medium.z{return result;}
        up=normalize(up*radius+ray*entry);h=p.medium.x;mu=dot(ray,up);mu_s=dot(sun,up);
    }
    let hit=ground(h,mu);let full_length=boundary(h,mu,hit);
    var length=full_length;if p.medium.z>0.0{length=min(length,max(p.medium.z-entry,0.0));}
    let count=p.size_steps.z;
    // Concentrate both sides of a grazing path near its minimum height.
    let b=(p.sun.w+h)*mu;let middle=clamp(-b,0.0,length);let hmin=height_at(h,mu,middle);let hend=height_at(h,mu,length);
    let before=log(1.0+max(h-hmin,0.0)/0.25);let after=log(1.0+max(hend-hmin,0.0)/0.25);
    var split=u32(round(f32(count)*before/max(before+after,1e-20)));
    if middle<=0.0{split=0u;}else if middle>=length{split=count;}else{split=clamp(split,1u,count-1u);}
    var sun_dirs:array<vec3<f32>,16>;var phases:array<vec4<f32>,80>;var disk_weights:array<f32,16>;
    var helper=vec3<f32>(0.0,1.0,0.0);if abs(sun.y)>0.9{helper=vec3<f32>(1.0,0.0,0.0);}
    let tx=normalize(cross(helper,sun));let ty=cross(sun,tx);
    let nodes=array<f32,4>(0.069431844,0.33000948,0.66999054,0.93056816);
    let weights=array<f32,4>(0.17392742,0.32607257,0.32607257,0.17392742);
    for(var j=0u;j<16u;j++) {
        let cm=1.0-nodes[j/4u]*(1.0-cos(p.sun.z));let sm=sqrt(max(1.0-cm*cm,0.0));let phi=(f32(j%4u)+0.5)*PI*0.5;
        sun_dirs[j]=sun*cm+(tx*cos(phi)+ty*sin(phi))*sm;disk_weights[j]=weights[j/4u]*0.25;
        for(var k=0u;k<5u;k++){phases[j*5u+k]=phase(clamp(dot(ray,sun_dirs[j]),-1.0,1.0),k);}
    }
    var previous=0.0;
    for(var i=1u;i<=count;i++) {
        var edge=length*f32(i)/f32(count);
        if before+after>=1e-5&&i<count {
            if i==split{edge=middle;}else{
                let incoming=i<split;var lh=after*f32(i-split)/max(f32(count-split),1.0);
                if incoming{lh=before*(1.0-f32(i)/f32(split));}
                let hh=hmin+0.25*(exp(lh)-1.0);let c=(h-hh)*(2.0*p.sun.w+h+hh);let root=sqrt(max(b*b-c,0.0));
                if incoming{edge=c/max(-b+root,1e-20);}else if b>0.0{edge=-c/max(root+b,1e-20);}else{edge=-b+root;}
                edge=clamp(edge,previous,length);
            }
        }
        let dt=edge-previous;let d=(edge+previous)*0.5;previous=edge;
        let hp=height_at(h,mu,d);let local_up=(up*(p.sun.w+h)+ray*d)/(p.sun.w+hp);
        let c=medium(hp);var direct=vec4<f32>(0.0);
        for(var j=0u;j<16u;j++) {
            var phase_weight=vec4<f32>(0.0);for(var k=0u;k<5u;k++){phase_weight+=c.scattering[k]*phases[j*5u+k];}
            direct+=sun_t(hp,dot(local_up,sun_dirs[j]))*phase_weight*disk_weights[j];
        }
        let source=direct*p.solar+indirect(hp,dot(local_up,ray),dot(local_up,sun),nu,c);
        let tau=c.extinction*dt;let trans=exp(-tau);
        let integral=select((vec4<f32>(1.0)-trans)/max(c.extinction,vec4<f32>(1e-30)),dt*(vec4<f32>(1.0)-tau*0.5+tau*tau/6.0),tau<vec4<f32>(0.001));
        result.light+=result.transmittance*source*integral;result.transmittance*=trans;
    }
    if hit&&length>=full_length {let end_up=(up*(p.sun.w+h)+ray*length)/p.sun.w;result.light+=result.transmittance*ground_light(dot(end_up,sun));}
    return result;
}
@compute @workgroup_size(8,8) fn render(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=p.size_steps.xy){return;}
    let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(p.size_steps.xy)*2.0-1.0;
    let sy=sin(p.view.x);let cy=cos(p.view.x);let sp=sin(p.view.y);let cp=cos(p.view.y);
    let forward=vec3<f32>(sy*cp,sp,cy*cp);let right=vec3<f32>(cy,0.0,-sy);let up=cross(forward,right);
    let ray=normalize(forward+right*uv.x*tan(p.view.z*0.5)*f32(p.size_steps.x)/f32(p.size_steps.y)-up*uv.y*tan(p.view.z*0.5));
    let sun=vec3<f32>(sin(p.sun.x)*cos(p.sun.y),sin(p.sun.y),cos(p.sun.x)*cos(p.sun.y));
    let result=integrate(ray,sun);var light=result.light;var rgb=vec3<f32>(0.0);
    let half_radius=sin(p.sun.z*0.5);
    if (p.size_steps.w&2u)!=0u&&p.medium.z<=0.0&&!ground(p.view.w,ray.y)&&dot(ray-sun,ray-sun)<=4.0*half_radius*half_radius {
        light+=result.transmittance*p.solar/(4.0*PI*half_radius*half_radius);
    }
    for(var i=0u;i<4u;i++){rgb+=light[i]*p.rgb[i].xyz;}
    var value=vec4<f32>(rgb,1.0);if (p.size_steps.w&4u)!=0u{value=light;}
    textureStore(output,vec2<i32>(id.xy),value);textureStore(output_t,vec2<i32>(id.xy),result.transmittance);
}
