// Successive orders of the scalar spectral radiative transfer equation.
// All computation and storage are f32. No phase is factored out of radiance.
struct Params {
    dims: vec4u, // radius, mu, mu_s, nu
    tables: vec4u, // tau radius, tau mu, ground samples, profile samples
    integration: vec4u, // ray steps, tau steps, sphere samples, hemisphere samples
    work: vec4u, // sun samples, scattering order, dispatch base, dispatch end
    planet: vec4f, // bottom radius, top altitude, ground albedo, solar irradiance
    mapping: vec4u, // 0 legacy, 1 sun-aligned cones
}
struct Profile { a:vec4f, b:vec4f } // h, extinction, rayleigh, aerosol0; aerosol1..3, pad
struct Angular { direction_weight:vec4f, phases:vec4f, extra:vec4f }
struct State { h:f32, mu:f32, mu_s:f32, nu:f32, ground:bool }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> profile:array<Profile>;
@group(0) @binding(2) var<storage,read> node_bits:array<u32>;
@group(0) @binding(3) var<storage,read> angular:array<Angular>;
// tau, ground irradiance for this order, then per-workgroup convergence stats
@group(0) @binding(4) var<storage,read_write> auxiliary:array<f32>;
@group(0) @binding(5) var<storage,read_write> previous:array<f32>;
@group(0) @binding(6) var<storage,read_write> density:array<f32>;
@group(0) @binding(7) var<storage,read_write> next_order:array<f32>;
// accumulated radiance, then accumulated ground irradiance
@group(0) @binding(8) var<storage,read_write> accumulated:array<f32>;
const PI:f32=3.141592653589793;
const ASINH_20:f32=3.6895038689889055;
override REUSE_DUPLICATE_STATES:bool=true;
fn phase_value(i:u32)->f32 {return bitcast<f32>(node_bits[i]);}
fn work_scatter_index(work:u32)->u32 {
    if !REUSE_DUPLICATE_STATES || p.mapping.x<3u {return work;}
    let base=4096u+p.dims.w+p.dims.x+p.dims.x*p.dims.z+p.dims.x*p.dims.z*4u*p.dims.w;
    return node_bits[base+1u+work];
}

fn scatter_len()->u32 { return p.dims.x*p.dims.y*p.dims.z*p.dims.w; }
fn tau_len()->u32 { return p.tables.x*p.tables.y; }
fn ground_stats_base()->u32 {
 let rows=(scatter_len()+63u)/64u;
 return tau_len()+p.tables.z+rows*5u+((rows+255u)/256u)*5u;
}
fn unit(i:u32,n:u32)->f32 { return f32(i)/f32(n-1u); }
fn rho2(h:f32)->f32 { return h*(2.0*p.planet.x+h); }
fn reference_height_coord(h:f32)->f32 {
 let x=clamp(h/p.planet.y,0.0,1.0);let scale=sqrt(0.01/p.planet.y);
 return 0.35*log(1.0+sqrt(x)/scale)/log(1.0+1.0/scale)+0.65*x;
}
fn height(x:f32)->f32 {
    if p.mapping.x>=3u {
        if x<=0.0 {return 0.0;} if x>=1.0 {return p.planet.y;}
        var lo=0.0;var hi=sqrt(p.planet.y);
        for(var j=0u;j<24u;j++){let mid=(lo+hi)*0.5;if reference_height_coord(mid*mid)<x {lo=mid;} else {hi=mid;}}
        let root=(lo+hi)*0.5;return root*root;
    }
    let q=x*x*rho2(p.planet.y);
    return q/(sqrt(p.planet.x*p.planet.x+q)+p.planet.x);
}
fn height_coord(h:f32)->f32 { if p.mapping.x>=3u {return reference_height_coord(h);} return clamp(sqrt(max(rho2(h)/rho2(p.planet.y),0.0)),0.0,1.0); }
fn horizon(h:f32)->f32 { return -sqrt(max(rho2(h),0.0))/(p.planet.x+h); }
fn hits_ground(h:f32,mu:f32)->bool { return mu<0.0 && mu<=horizon(h); }
fn distance(h:f32,mu:f32,ground:bool)->f32 {
    let b=(p.planet.x+h)*mu;
    if ground {
        let c=rho2(h);
        let denominator=-b+sqrt(max(b*b-c,0.0));
        if denominator>0.0 { return max(c/denominator,0.0); }
        return 0.0;
    }
    let c=(p.planet.y-h)*(2.0*p.planet.x+p.planet.y+h);
    let root=sqrt(max(b*b+c,0.0));
    if b>0.0 { return max(c/(root+b),0.0); }
    return max(-b+root,0.0);
}
fn view(h:f32,i:u32,n:u32)->f32 {
    let half=n/2u; let u=unit(i%half,half); let hor=horizon(h);
    let x=u*u*u/(u*u*u+(1.0-u)*(1.0-u)*(1.0-u));
    if i<half { return hor-(1.0+hor)*x; }
    return hor+(1.0-hor)*x;
}
fn inverse_view_warp(y:f32)->f32 {
    let x=clamp(y,0.0,1.0); let a=pow(x,1.0/3.0); let b=pow(1.0-x,1.0/3.0);
    return a/(a+b);
}
fn view_coord(h:f32,mu:f32,ground:bool,n:u32)->f32 {
    let hor=horizon(h); let half=n/2u;
    if ground { return inverse_view_warp((hor-mu)/(1.0+hor))*f32(half-1u); }
    return f32(half)+inverse_view_warp((mu-hor)/(1.0-hor))*f32(half-1u);
}
fn sun_cosine(x:f32)->f32 { return clamp(sinh((2.0*x-1.0)*ASINH_20)/20.0,-1.0,1.0); }
fn sun_coord(mu:f32)->f32 { return clamp(0.5+0.5*asinh(20.0*mu)/ASINH_20,0.0,1.0); }
fn reference_atan_cdf(e:f32,center:f32,scale:f32)->f32 {
 let a=atan((-PI*0.5-center)/scale);
 return (atan((e-center)/scale)-a)/(atan((PI*0.5-center)/scale)-a);
}
fn reference_solar_coord(h:f32,mu:f32)->f32 {
 let e=asin(clamp(mu,-1.0,1.0));let hor=asin(horizon(h));
 let d=PI*0.5-e;let scale=PI/36.0;
 let cap=1.0-sqrt(d/(d+scale))/sqrt(PI/(PI+scale));
 return clamp(0.25*(e/PI+0.5)+0.20*reference_atan_cdf(e,hor,PI/90.0)
   +0.20*reference_atan_cdf(e,hor-PI/30.0,PI/30.0)
   +0.15*reference_atan_cdf(e,-hor,PI/360.0)+0.20*cap,0.0,1.0);
}
fn solar_cosine(h:f32,u:f32)->f32 {
    if p.mapping.x>=3u {
        if u<=0.0 {return -1.0;} if u>=1.0 {return 1.0;}
        var lo=-PI*0.5;var hi=PI*0.5;
        for(var j=0u;j<24u;j++){let mid=(lo+hi)*0.5;if reference_solar_coord(h,sin(mid))<u {lo=mid;} else {hi=mid;}}
        return sin((lo+hi)*0.5);
    }
    if p.mapping.x==0u { return sun_cosine(u); }
    if p.mapping.x==2u {
        let index=u*96.0;let hor=horizon(h);
        if index>62.0 {
            let theta=acos(hor+(1.0-hor)*(30.0/32.0)*(30.0/32.0));
            let x=(index-62.0)/34.0;
            return cos(theta*(1.0-x)*(1.0-x));
        }
        let x=index/32.0-1.0;
        return clamp(hor+x*abs(x)*select(1.0-hor,1.0+hor,x<0.0),-1.0,1.0);
    }
    let hor=horizon(h); let x=2.0*u-1.0;
    return clamp(hor+x*abs(x)*select(1.0-hor,1.0+hor,x<0.0),-1.0,1.0);
}
fn solar_coord(h:f32,mu:f32)->f32 {
    if p.mapping.x>=3u {return reference_solar_coord(h,mu);}
    if p.mapping.x==0u { return sun_coord(mu); }
    if p.mapping.x==2u {
        let hor=horizon(h);let delta=mu-hor;
        let x=delta/select(1.0-hor,1.0+hor,delta<0.0);
        let old=clamp(32.0+32.0*select(sqrt(abs(x)),-sqrt(abs(x)),x<0.0),0.0,64.0);
        if old<=62.0 {return old/96.0;}
        let theta=acos(hor+(1.0-hor)*(30.0/32.0)*(30.0/32.0));
        let fraction=1.0-sqrt(clamp(acos(clamp(mu,-1.0,1.0))/theta,0.0,1.0));
        return (62.0+fraction*34.0)/96.0;
    }
    let hor=horizon(h); let delta=mu-hor;
    let x=delta/select(1.0-hor,1.0+hor,delta<0.0);
    return clamp(0.5+0.5*select(sqrt(abs(x)),-sqrt(abs(x)),x<0.0),0.0,1.0);
}
fn scattering_cosine(u:f32)->f32 {
    var lo=0.0;var hi=PI;
    for(var j=0u;j<22u;j++) {
        let mid=(lo+hi)*0.5;
        if scattering_angle_coord(mid)<u {lo=mid;} else {hi=mid;}
    }
    return cos((lo+hi)*0.5);
}
fn scattering_coord(nu:f32)->f32 {
    return scattering_angle_coord(acos(clamp(nu,-1.0,1.0)));
}
fn scattering_angle_coord(theta:f32)->f32 {
    let scale=PI/90.0;
    return clamp(0.6*log(1.0+theta/scale)/log(91.0)
        +0.15*(1.0-log(1.0+(PI-theta)/scale)/log(91.0))
        +0.25*(atan((theta-PI*0.5)/scale)+atan(45.0))/(2.0*atan(45.0)),0.0,1.0);
}
// Interpolating on cones around the sun holds the phase angle fixed. Every
// generated (mu, mu_s, nu) is physically realizable, including zenith sunlight.
fn cone_horizon(h:f32,mu_s:f32,nu:f32)->vec3f {
    let center=mu_s*nu;
    let extent=sqrt(max((1.0-mu_s*mu_s)*(1.0-nu*nu),0.0));
    return vec3f(center,extent,acos(clamp((horizon(h)-center)/max(extent,1e-20),-1.0,1.0)));
}
fn cone_cosine(h:f32,mu_s:f32,nu:f32,ground:bool)->f32 {
    let alpha=acos(clamp(mu_s,-1.0,1.0));let beta=acos(horizon(h));
    var lo=max(alpha-beta,0.0);var hi=min(alpha+beta,PI);
    if ground { lo=max(beta-alpha,0.0);hi=min(2.0*PI-beta-alpha,PI); }
    return cos(clamp(acos(clamp(nu,-1.0,1.0)),lo,hi));
}
fn cone_view(h:f32,mu_s:f32,nu:f32,i:u32,n:u32)->f32 {
    if p.mapping.w==1u {
        let ng=max(n/4u,2u);let ground=i<ng;
        let s=State(h,0.0,mu_s,nu,ground);let bounds=optical_cone_bounds(s);
        var u=0.0;
        if ground {u=unit(i,ng);} else {u=unit(i-ng,n-ng);}
        let x=mix(bounds.x,bounds.y,u);
        let delta=bounds.z*x/max(1.0-x,1e-20);
        let c=cone_horizon(h,mu_s,nu);
        return clamp(horizon(h)+select(delta,-delta,ground),max(c.x-c.y,-1.0),min(c.x+c.y,1.0));
    }
    let ng=max(n/4u,2u); let c=cone_horizon(h,mu_s,nu);
    var angle=0.0;
    if i<ng { let u=unit(i,ng); angle=c.z+(PI-c.z)*u*u; }
    else { let u=unit(i-ng,n-ng); angle=c.z*(2.0*u-u*u); }
    return clamp(c.x+c.y*cos(angle),-1.0,1.0);
}
fn cone_coord(s:State,n:u32)->f32 {
    if p.mapping.w==1u {
        let ng=max(n/4u,2u);let bounds=optical_cone_bounds(s);
        let delta=abs(s.mu-horizon(s.h));let x=delta/(delta+bounds.z);
        var u=0.0;
        if abs(bounds.y-bounds.x)>1e-10 {u=clamp((x-bounds.x)/(bounds.y-bounds.x),0.0,1.0);}
        if s.ground {return u*f32(ng-1u);}
        return f32(ng)+u*f32(n-ng-1u);
    }
    let ng=max(n/4u,2u); let c=cone_horizon(s.h,s.mu_s,s.nu);
    let angle=acos(clamp((s.mu-c.x)/max(c.y,1e-20),-1.0,1.0));
    if s.ground {
        return sqrt(clamp((angle-c.z)/max(PI-c.z,1e-20),0.0,1.0))*f32(ng-1u);
    }
    return f32(ng)+(1.0-sqrt(clamp(1.0-angle/max(c.z,1e-20),0.0,1.0)))*f32(n-ng-1u);
}
fn optical_cone_bounds(s:State)->vec3f {
    let center=s.mu_s*s.nu;
    let extent=sqrt(max((1.0-s.mu_s*s.mu_s)*(1.0-s.nu*s.nu),0.0));
    let hor=horizon(s.h);
    let lo=select(max(center-extent,hor),center-extent,s.ground);
    let hi=select(center+extent,min(center+extent,hor),s.ground);
    let scale=sqrt(16.0/(p.planet.x+s.h));
    let a=abs(hi-hor);let b=abs(lo-hor);
    return vec3f(a/(a+scale),b/(b+scale),scale);
}

fn reference_radius(ri:u32)->f32 {return phase_value(4096u+p.dims.w+ri);}
fn reference_radius_coord(h:f32,linear:bool)->f32 {
 if (p.mapping.y&16777216u)==0u && !linear {return height_coord(h)*f32(p.dims.x-1u);}
 var lo=0u;var hi=p.dims.x-1u;
 while hi-lo>1u {let mid=(lo+hi)/2u;if reference_radius(mid)<=h {lo=mid;}else {hi=mid;}}
 let a=reference_radius(lo);let b=reference_radius(hi);
 if !linear && (p.mapping.y&33554432u)!=0u {
  let ca=reference_height_coord(a);let cb=reference_height_coord(b);
  return f32(lo)+clamp((reference_height_coord(h)-ca)/(cb-ca),0.0,1.0);
 }
 return f32(lo)+clamp((h-a)/(b-a),0.0,1.0);
}
fn reference_solar(ri:u32,si:u32)->f32 {return phase_value(4096u+p.dims.w+p.dims.x+ri*p.dims.z+si);}
fn reference_phase(ri:u32,si:u32,ni:u32,ground:bool)->f32 {
 let base=4096u+p.dims.w+p.dims.x+p.dims.x*p.dims.z;
 return phase_value(base+((ri*p.dims.z+si)*2u+select(0u,1u,ground))*p.dims.w+ni);
}
fn reference_bounds(s:State)->vec4f {
 let alpha=acos(clamp(s.mu_s,-1.0,1.0));let beta=acos(horizon(s.h));
 let a=abs(beta-alpha);let b=min(min(alpha+beta,2.0*PI-alpha-beta),PI);
 var lo=max(alpha-beta,0.0);var hi=min(alpha+beta,PI);
 if s.ground {lo=max(beta-alpha,0.0);hi=min(2.0*PI-beta-alpha,PI);}
 return vec4f(lo,clamp(a,lo,hi),clamp(b,lo,hi),hi);
}
fn reference_phase_cdf(theta:f32)->f32 {return 0.65*log(1.0+theta/(PI/90.0))/log(91.0)+0.35*theta/PI;}
fn reference_phase_coord(s:State)->f32 {
 let b=reference_bounds(s);let theta=clamp(acos(clamp(s.nu,-1.0,1.0)),b.x,b.w);
 let at_horizon=abs(s.mu-horizon(s.h))<1e-6;
 var k=1u;if !at_horizon && theta<b.y {k=0u;} else if !at_horizon && theta>b.z {k=2u;}
 let splits=vec4f(0.0,0.5,0.875,1.0);
 let a=reference_phase_cdf(b[k]);let z=reference_phase_cdf(b[k+1u]);
 let t=clamp((reference_phase_cdf(theta)-a)/max(z-a,1e-20),0.0,1.0);
 var fraction=sqrt(t);if k==0u {fraction=1.0-sqrt(1.0-t);} else if k==1u {fraction=acos(clamp(1.0-2.0*t,-1.0,1.0))/PI;}
 return mix(splits[k],splits[k+1u],fraction);
}
fn reference_corner(s:State,ri:u32,si:u32,ni:u32,source:bool)->f32 {
 let mc=cone_coord(s,p.dims.y);let mi=u32(floor(mc));let mt=mc-f32(mi);
 let a=((ri*p.dims.y+mi)*p.dims.z+si)*p.dims.w+ni;
 let b=((ri*p.dims.y+min(mi+1u,p.dims.y-1u))*p.dims.z+si)*p.dims.w+ni;
 if (p.mapping.y&65536u)!=0u && mt>0.0 {
  let ng=max(p.dims.y/4u,2u);let begin=select(ng,0u,s.ground);let end=select(p.dims.y-1u,ng-1u,s.ground);
  let before=((ri*p.dims.y+max(u32(max(i32(mi)-1,0)),begin))*p.dims.z+si)*p.dims.w+ni;
  let after=((ri*p.dims.y+min(mi+2u,end))*p.dims.z+si)*p.dims.w+ni;
  var v=vec4f(load_radiance(before),load_radiance(a),load_radiance(b),load_radiance(after));
  if source {v=vec4f(density[before],density[a],density[b],density[after]);}
  if before==a {v.x=2.0*v.y-v.z;}if after==b {v.w=2.0*v.z-v.y;}
  return reference_cubic(v,mt);
 }
 if source {return mix(density[a],density[b],mt);} return mix(load_radiance(a),load_radiance(b),mt);
}
fn reference_slope(a:f32,b:f32)->f32 {
 if a==0.0 || b==0.0 || (a>0.0)!=(b>0.0) {return 0.0;}
 let lo=min(abs(a),abs(b));let hi=max(abs(a),abs(b));
 return select(-1.0,1.0,a>0.0)*2.0*lo/(1.0+lo/hi);
}
fn reference_cubic(v:vec4f,t:f32)->f32 {
 let d=v.z-v.y;let a=reference_slope(v.y-v.x,d);let b=reference_slope(d,v.w-v.z);
 let t2=t*t;let t3=t2*t;
 return clamp((2.0*t3-3.0*t2+1.0)*v.y+(t3-2.0*t2+t)*a+(-2.0*t3+3.0*t2)*v.z+(t3-t2)*b,min(v.y,v.z),max(v.y,v.z));
}
fn reference_phase_lookup(s:State,ri:u32,si:u32,nc:f32,source:bool)->f32 {
 let ni=u32(floor(nc));let t=nc-f32(ni);
 let v=vec4f(reference_corner(s,ri,si,u32(max(i32(ni)-1,0)),source),
  reference_corner(s,ri,si,ni,source),reference_corner(s,ri,si,min(ni+1u,p.dims.w-1u),source),
  reference_corner(s,ri,si,min(ni+2u,p.dims.w-1u),source));
 if (p.mapping.y&256u)!=0u && all(v>=vec4f(0.0)) {
  if v.y==v.z || t<=0.0 {return v.y;}if t>=1.0 {return v.z;}
  return clamp(max(exp(reference_cubic(log(v+vec4f(1e-30)),t))-1e-30,0.0),min(v.y,v.z),max(v.y,v.z));
 }
 return reference_cubic(v,t);
}
fn reference_mix(a:f32,b:f32,t:f32)->f32 {
 if (p.mapping.y&255u)==0u || a<0.0 || b<0.0 {return mix(a,b,t);}
 if a==b || t<=0.0 {return a;} if t>=1.0 {return b;}
 return max(exp(mix(log(a+1e-30),log(b+1e-30),t))-1e-30,0.0);
}
fn reference_lookup(s:State,source:bool)->f32 {
 if p.mapping.x==4u {return ray_reference_lookup(s,source);}
 let rc=height_coord(s.h)*f32(p.dims.x-1u);let rlo=u32(floor(rc));let rt=rc-f32(rlo);
 let hor=horizon(s.h);let delta=(s.mu-hor)/select(1.0-hor,1.0+hor,s.ground);
 let extent=sqrt(max((1.0-s.mu*s.mu)*(1.0-s.mu_s*s.mu_s),0.0));
 let az=clamp((s.nu-s.mu*s.mu_s)/max(extent,1e-20),-1.0,1.0);
 var value=0.0;
 for(var r=0u;r<2u;r++) {
  let rw=select(1.0-rt,rt,r==1u);if rw<=0.0 {continue;}
  let ri=min(rlo+r,p.dims.x-1u);let h=reference_radius(ri);
  let target_hor=horizon(h);let mu=clamp(target_hor+delta*select(1.0-target_hor,1.0+target_hor,s.ground),-1.0,1.0);
  let nu=mu*s.mu_s+sqrt(max((1.0-mu*mu)*(1.0-s.mu_s*s.mu_s),0.0))*az;
  let point=State(h,mu,s.mu_s,nu,s.ground);
  let sc=solar_coord(h,s.mu_s)*f32(p.dims.z-1u);let slo=u32(floor(sc));let st=sc-f32(slo);
  let nc=reference_phase_coord(point)*f32(p.dims.w-1u);let nlo=u32(floor(nc));let nt=nc-f32(nlo);
  var solar=vec2f(0.0);
  for(var j=0u;j<2u;j++) {
   let si=min(slo+j,p.dims.z-1u);
   solar[j]=reference_phase_lookup(point,ri,si,nc,source);
  }
  value+=rw*reference_mix(solar.x,solar.y,st);
 }
 return value;
}

fn ray_reference_lookup(s:State,source:bool)->f32 {
 let local_source=source && (p.mapping.z&65536u)!=0u;
 let rc=reference_radius_coord(s.h,local_source);let rlo=u32(floor(rc));let rt=rc-f32(rlo);
 let hor=horizon(s.h);let delta=(s.mu-hor)/select(1.0-hor,1.0+hor,s.ground);
 let d=abs(asin(clamp(s.mu_s,-1.0,1.0))-asin(hor))*180.0/PI;
 let x=clamp((d-1.0)*0.5,0.0,1.0);let blend=x*x*(3.0-2.0*x);
 var radial=vec2f(0.0);
 for(var r=0u;r<2u;r++) {
  let rw=select(1.0-rt,rt,r==1u);if rw<=0.0 {continue;}
  let ri=min(rlo+r,p.dims.x-1u);let h=reference_radius(ri);let hor_i=horizon(h);
  var mu=clamp(hor_i+delta*select(1.0-hor_i,1.0+hor_i,s.ground),-1.0,1.0);
  var mus=s.mu_s;
  if 1.0-s.mu*s.mu>1e-7 {mus=clamp(mu*s.nu+sqrt(max((1.0-mu*mu)/(1.0-s.mu*s.mu),0.0))*(s.mu_s-s.mu*s.nu),-1.0,1.0);}
  var ground=s.ground;
  if local_source {mu=s.mu;mus=s.mu_s;ground=hits_ground(h,mu);}
  let point=State(h,mu,mus,s.nu,ground);
  let az=clamp((s.nu-mu*mus)/sqrt(max((1.0-mu*mu)*(1.0-mus*mus),1e-30)),-1.0,1.0);
  let sc=solar_coord(h,mus)*f32(p.dims.z-1u);let slo=u32(floor(sc));let st=sc-f32(slo);
  for(var chart=0u;chart<2u;chart++) {
   let weight=select(1.0-blend,blend,chart==1u);if weight<=0.0 {continue;}
   var solar=vec2f(0.0);
   for(var j=0u;j<2u;j++) {
    let si=min(slo+j,p.dims.z-1u);var corner=point;
    if chart==0u {
     let sm=reference_solar(ri,si);
     corner=State(h,mu,sm,mu*sm+sqrt(max((1.0-mu*mu)*(1.0-sm*sm),0.0))*az,ground);
    }
    let nc=reference_phase_coord(corner)*f32(p.dims.w-1u);
    solar[j]=reference_phase_lookup(corner,ri,si,nc,source);
   }
   radial[r]+=weight*reference_mix(solar.x,solar.y,st);
  }
 }
 if !local_source && all(radial>vec2f(0.0)) {return reference_mix(radial.x,radial.y,rt);}
 return mix(radial.x,radial.y,rt);
}

fn state(i:u32)->State {
    if p.mapping.x>=3u {
        let ri=i/(p.dims.y*p.dims.z*p.dims.w);let si=i/p.dims.w%p.dims.z;let mi=i/(p.dims.z*p.dims.w)%p.dims.y;
        let h=reference_radius(ri);let mu_s=reference_solar(ri,si);let ground=mi<max(p.dims.y/4u,2u);
        let nu=reference_phase(ri,si,i%p.dims.w,ground);return State(h,cone_view(h,mu_s,nu,mi,p.dims.y),mu_s,nu,ground);
    }
    let h=height(unit(i/(p.dims.y*p.dims.z*p.dims.w),p.dims.x));
    let mi=i/(p.dims.z*p.dims.w)%p.dims.y;
    if p.mapping.x!=0u {
        let mu_s=solar_cosine(h,unit(i/p.dims.w%p.dims.z,p.dims.z));
        let ground=mi<max(p.dims.y/4u,2u);
        let nu=cone_cosine(h,mu_s,phase_value(4096u+i%p.dims.w),ground);
        let mu=cone_view(h,mu_s,nu,mi,p.dims.y);
        return State(h,mu,mu_s,nu,ground);
    }
    let mu=view(h,mi,p.dims.y);
    let mu_s=sun_cosine(unit(i/p.dims.w%p.dims.z,p.dims.z));
    let u=unit(i%p.dims.w,p.dims.w);
    let extent=sqrt(max((1.0-mu*mu)*(1.0-mu_s*mu_s),0.0));
    let nu=clamp(1.0-2.0*u*u*u,mu*mu_s-extent,mu*mu_s+extent);
    return State(h,mu,mu_s,nu,mi<p.dims.y/2u);
}
fn advance(s:State,d:f32)->State {
    let r=p.planet.x+s.h;
    let q=rho2(s.h)+d*(2.0*r*s.mu+d);
    let h=clamp(q/(sqrt(max(p.planet.x*p.planet.x+q,0.0))+p.planet.x),0.0,p.planet.y);
    return State(h,clamp((r*s.mu+d)/(p.planet.x+h),-1.0,1.0),
        clamp((r*s.mu_s+d*s.nu)/(p.planet.x+h),-1.0,1.0),s.nu,s.ground);
}
fn coordinates(s:State)->vec4f {
    if p.mapping.x!=0u {
        return vec4f(height_coord(s.h)*f32(p.dims.x-1u),cone_coord(s,p.dims.y),
            solar_coord(s.h,s.mu_s)*f32(p.dims.z-1u),scattering_coord(s.nu)*f32(p.dims.w-1u));
    }
    return vec4f(height_coord(s.h)*f32(p.dims.x-1u),view_coord(s.h,s.mu,s.ground,p.dims.y),
        sun_coord(s.mu_s)*f32(p.dims.z-1u),pow(max((1.0-clamp(s.nu,-1.0,1.0))*0.5,0.0),1.0/3.0)*f32(p.dims.w-1u));
}
fn tensor_lookup(s:State,source:bool)->f32 {
    let c=clamp(coordinates(s),vec4f(0.0),vec4f(p.dims-vec4u(1u)));
    let lo=vec4u(floor(c)); let hi=min(lo+vec4u(1u),p.dims-vec4u(1u)); let t=c-vec4f(lo);
    var result=0.0;
    var solar_values=vec2f(0.0);
    for(var corner=0u;corner<16u;corner++) {
        var idx=0u; var weight=1.0;
        for(var a=0u;a<4u;a++) {
            let upper=(corner&(1u<<a))!=0u;
            idx=idx*p.dims[a]+select(lo[a],hi[a],upper);
            if (p.mapping.y&255u)==0u || a!=2u { weight*=select(1.0-t[a],t[a],upper); }
        }
        // An explicit branch avoids reading an unrelated buffer in each pass.
        var value=0.0;
        if source { value=density[idx]; } else { value=load_radiance(idx); }
        if (p.mapping.y&255u)==0u { result+=weight*value; }
        else { solar_values[(corner>>2u)&1u]+=weight*value; }
    }
    if (p.mapping.y&255u)==1u {
        if any(solar_values<vec2f(0.0)) { return mix(solar_values.x,solar_values.y,t.z); }
        if solar_values.x==solar_values.y || t.z<=0.0 { return solar_values.x; }
        if t.z>=1.0 { return solar_values.y; }
        return max(exp(mix(log(solar_values.x+1e-30),log(solar_values.y+1e-30),t.z))-1e-30,0.0);
    }
    return result;
}
// The cone's horizon-clipped interval depends on all three outer axes.
// Reusing one normalized cone coordinate across their corners creates a
// spurious ring when a solar cone becomes tangent to the ground horizon.
fn cone_corner(s:State,ri:u32,si:u32,ni:u32,source:bool)->f32 {
    let h=height(unit(ri,p.dims.x));
    let mu_s=solar_cosine(h,unit(si,p.dims.z));
    let nu=cone_cosine(h,mu_s,phase_value(4096u+ni),s.ground);
    let mc=cone_coord(State(h,s.mu,mu_s,nu,s.ground),p.dims.y);
    let mi=u32(floor(mc));let mh=min(mi+1u,p.dims.y-1u);let t=mc-f32(mi);
    let a=((ri*p.dims.y+mi)*p.dims.z+si)*p.dims.w+ni;
    let b=((ri*p.dims.y+mh)*p.dims.z+si)*p.dims.w+ni;
    if source {return mix(density[a],density[b],t);}
    return mix(load_radiance(a),load_radiance(b),t);
}
fn lookup(s:State,source:bool)->f32 {
    if p.mapping.x>=3u {return reference_lookup(s,source);}
    if p.mapping.x==0u {return tensor_lookup(s,source);}
    let c=clamp(vec3f(height_coord(s.h),solar_coord(s.h,s.mu_s),scattering_coord(s.nu)),vec3f(0.0),vec3f(1.0))
        *vec3f(p.dims.xzw-vec3u(1u));
    let lo=vec3u(floor(c));let hi=min(lo+vec3u(1u),p.dims.xzw-vec3u(1u));let t=c-vec3f(lo);
    var solar_values=vec2f(0.0);
    for(var si=0u;si<2u;si++) {
        for(var ri=0u;ri<2u;ri++) {
            for(var ni=0u;ni<2u;ni++) {
                let value=cone_corner(s,select(lo.x,hi.x,ri==1u),select(lo.y,hi.y,si==1u),select(lo.z,hi.z,ni==1u),source);
                solar_values[si]+=value*select(1.0-t.x,t.x,ri==1u)*select(1.0-t.z,t.z,ni==1u);
            }
        }
    }
    if (p.mapping.y&255u)==0u || any(solar_values<vec2f(0.0)) {return mix(solar_values.x,solar_values.y,t.y);}
    if solar_values.x==solar_values.y || t.y<=0.0 {return solar_values.x;}
    if t.y>=1.0 {return solar_values.y;}
    return max(exp(mix(log(solar_values.x+1e-30),log(solar_values.y+1e-30),t.y))-1e-30,0.0);
}
// The density pass samples the previous order at its own radius and solar
// zenith grid nodes. Only mu and nu vary: four fetches instead of sixteen,
// with the same corner projection and no extra approximation.
fn incident_lookup(index:u32,s:State,v:vec3f,sun:vec3f)->f32 {
    let ri=index/(p.dims.y*p.dims.z*p.dims.w);
    let si=index/p.dims.w%p.dims.z;
    let nu=clamp(dot(v,sun),-1.0,1.0);
    if p.mapping.x>=3u {
        let point=State(s.h,v.z,s.mu_s,nu,hits_ground(s.h,v.z));
        let nc=reference_phase_coord(point)*f32(p.dims.w-1u);let ni=u32(floor(nc));
        return reference_phase_lookup(point,ri,si,nc,false);
    }
    var mc=view_coord(s.h,v.z,hits_ground(s.h,v.z),p.dims.y);
    var nc=pow(max((1.0-nu)*0.5,0.0),1.0/3.0)*f32(p.dims.w-1u);
    if p.mapping.x!=0u {
        nc=scattering_coord(nu)*f32(p.dims.w-1u);
        let point=State(s.h,v.z,s.mu_s,nu,hits_ground(s.h,v.z));
        let ni=u32(floor(nc));let nh=min(ni+1u,p.dims.w-1u);
        return mix(cone_corner(point,ri,si,ni,false),cone_corner(point,ri,si,nh,false),nc-f32(ni));
    }
    let lo=vec2u(floor(vec2f(mc,nc)));
    let hi=min(lo+vec2u(1u),vec2u(p.dims.y-1u,p.dims.w-1u));
    let t=vec2f(mc,nc)-vec2f(lo);
    let a=((ri*p.dims.y+lo.x)*p.dims.z+si)*p.dims.w;
    let b=((ri*p.dims.y+hi.x)*p.dims.z+si)*p.dims.w;
    return mix(mix(load_radiance(a+lo.y),load_radiance(a+hi.y),t.y),mix(load_radiance(b+lo.y),load_radiance(b+hi.y),t.y),t.x);
}
fn load_radiance(index:u32)->f32 { return previous[index]; }
fn tau_lookup(h:f32,mu:f32,ground:bool)->f32 {
    let c=vec2f(height_coord(h)*f32(p.tables.x-1u),view_coord(h,mu,ground,p.tables.y));
    let lo=vec2u(floor(c)); let hi=min(lo+vec2u(1u),p.tables.xy-vec2u(1u)); let t=c-vec2f(lo);
    return mix(mix(auxiliary[lo.x*p.tables.y+lo.y],auxiliary[lo.x*p.tables.y+hi.y],t.y),
               mix(auxiliary[hi.x*p.tables.y+lo.y],auxiliary[hi.x*p.tables.y+hi.y],t.y),t.x);
}
fn sun_transmittance(h:f32,mu:f32)->f32 {
    if hits_ground(h,mu) { return 0.0; }
    return exp(-tau_lookup(h,mu,false));
}
fn ground_lookup(mu_s:f32)->f32 {
    let x=solar_coord(0.0,mu_s)*f32(p.tables.z-1u); let lo=u32(floor(x)); let hi=min(lo+1u,p.tables.z-1u);
    return mix(auxiliary[tau_len()+lo],auxiliary[tau_len()+hi],x-f32(lo));
}
fn coefficients(h:f32)->Profile {
    if h<=profile[0].a.x { return profile[0]; }
    if h>=profile[p.tables.w-1u].a.x { return profile[p.tables.w-1u]; }
    var lo=0u; var hi=p.tables.w-1u;
    while hi-lo>1u {
        let mid=(lo+hi)/2u;
        if profile[mid].a.x<h { lo=mid; } else { hi=mid; }
    }
    let t=clamp((h-profile[lo].a.x)/(profile[hi].a.x-profile[lo].a.x),0.0,1.0);
    return Profile(mix(profile[lo].a,profile[hi].a,t),mix(profile[lo].b,profile[hi].b,t));
}
fn aerosol_phase(species:u32,mu:f32)->f32 {
    let f=pow(max((1.0-clamp(mu,-1.0,1.0))*0.5,0.0),1.0/3.0)*1024.0-0.5;
    let lo=u32(clamp(floor(f),0.0,1023.0)); let hi=min(lo+1u,1023u);
    return mix(phase_value(species*1024u+lo),phase_value(species*1024u+hi),clamp(f-f32(lo),0.0,1.0));
}
fn phase_weight(c:Profile,mu:f32)->f32 {
    return c.a.z*3.0*(1.0+mu*mu)/(16.0*PI)+c.a.w*aerosol_phase(0u,mu)
        +c.b.x*aerosol_phase(1u,mu)+c.b.y*aerosol_phase(2u,mu)+c.b.z*aerosol_phase(3u,mu);
}
fn cached_phase_weight(c:Profile,q:Angular)->f32 {
    return dot(vec4f(c.a.zw,c.b.xy),q.phases)+c.b.z*q.extra.x;
}
fn rotate(local:vec3f,axis:vec3f)->vec3f {
    var helper=vec3f(0.0,0.0,1.0);
    if abs(axis.z)>=0.9 { helper=vec3f(1.0,0.0,0.0); }
    let x=normalize(cross(helper,axis));
    return x*local.x+cross(axis,x)*local.y+axis*local.z;
}
fn sun_direction(s:State)->vec3f {
    if p.mapping.x!=0u {return vec3f(sqrt(max(1.0-s.mu_s*s.mu_s,0.0)),0.0,s.mu_s);}
    let sin_v=sqrt(max(1.0-s.mu*s.mu,0.0));
    var sx=sqrt(max(1.0-s.mu_s*s.mu_s,0.0));
    if sin_v>0.000001 { sx=(s.nu-s.mu*s.mu_s)/sin_v; }
    return normalize(vec3f(sx,sqrt(max(1.0-s.mu_s*s.mu_s-sx*sx,0.0)),s.mu_s));
}
fn view_direction(s:State)->vec3f {
    let sin_v=sqrt(max(1.0-s.mu*s.mu,0.0));
    if p.mapping.x==0u {return vec3f(sin_v,0.0,s.mu);}
    let sin_s=sqrt(max(1.0-s.mu_s*s.mu_s,0.0));
    var x=sin_v;
    if sin_s>1e-6 {x=clamp((s.nu-s.mu*s.mu_s)/sin_s,-sin_v,sin_v);}
    return vec3f(x,sqrt(max(1.0-s.mu*s.mu-x*x,0.0)),s.mu);
}

// Smooth partition of unity between two deterministic sphere quadratures.
// These weights only place integration effort; they do not modify the phase
// function or transport. A sun-centered rule resolves the incident peak that
// an outgoing-centered rule can miss in second and higher scattering orders.
fn axis_weight(v:vec3f,axis:vec3f,other:vec3f)->f32 {
    let a=0.0004+max(1.0-dot(v,axis),0.0);
    let b=0.0004+max(1.0-dot(v,other),0.0);
    return b*b/(a*a+b*b);
}

@compute @workgroup_size(64)
fn transmittance(@builtin(global_invocation_id) id:vec3u) {
    let i=p.work.z+id.x; if i>=p.work.w { return; }
    let h=height(unit(i/p.tables.y,p.tables.x)); let mi=i%p.tables.y;
    let mu=view(h,mi,p.tables.y); let ground=mi<p.tables.y/2u;
    let s=State(h,mu,0.0,0.0,ground); let dx=distance(h,mu,ground)/f32(p.integration.y);
    var sum=0.0;
    for(var j=0u;j<p.integration.y;j++) { sum+=coefficients(advance(s,(f32(j)+0.5)*dx).h).a.y*dx; }
    auxiliary[i]=sum;
}

@compute @workgroup_size(64)
fn ground_irradiance(@builtin(global_invocation_id) id:vec3u) {
    let i=p.work.z+id.x; if i>=p.work.w { return; }
    let mu_s=solar_cosine(0.0,unit(i,p.tables.z)); let sun=vec3f(sqrt(max(1.0-mu_s*mu_s,0.0)),0.0,mu_s);
    var value=0.0;
    if p.work.y==1u {
        let offset=p.integration.z+p.integration.w;
        for(var j=0u;j<p.work.x;j++) {
            let q=angular[offset+j].direction_weight; let v=rotate(q.xyz,sun);
            value+=max(v.z,0.0)*sun_transmittance(0.0,v.z)*q.w*p.planet.w;
        }
    } else {
        for(var j=0u;j<p.integration.w;j++) {
            let q=angular[p.integration.z+j].direction_weight;
            value+=lookup(State(0.0,q.z,mu_s,dot(q.xyz,sun),false),false)*q.z*q.w;
        }
    }
    auxiliary[tau_len()+i]=value;
    let old=accumulated[scatter_len()+i];var total=old+value;
    if (p.mapping.z&256u)!=0u {
        let base=ground_stats_base()+p.tables.z;
        if p.work.y==1u {auxiliary[base+i]=value;total=value;}
        else {total=auxiliary[base+i]+value;}
    }
    accumulated[scatter_len()+i]=total;
    let ratio=abs(total-old)/max(total,p.planet.w*1e-12);
    auxiliary[ground_stats_base()+i]=select(-1.0,ratio,ratio>=0.0 && ratio<=3.402823e38);
}

@compute @workgroup_size(64)
fn scattering_density(@builtin(global_invocation_id) id:vec3u) {
    let work=p.work.z+id.x;if work>=p.work.w {return;}let i=work_scatter_index(work);
    if canonical_scatter_index(i)!=i {return;}
    let s=state(i); let c=coefficients(s.h);
    let view_dir=view_direction(s); let sun=sun_direction(s);
    var value=0.0;
    if p.work.y==1u {
        let offset=p.integration.z+p.integration.w;
        for(var j=0u;j<p.work.x;j++) {
            let q=angular[offset+j].direction_weight; let v=rotate(q.xyz,sun);
            value+=sun_transmittance(s.h,v.z)*phase_weight(c,clamp(dot(view_dir,v),-1.0,1.0))*q.w*p.planet.w;
        }
    } else {
        for(var j=0u;j<p.integration.z;j++) {
            let q=angular[j]; let v=rotate(q.direction_weight.xyz,view_dir);
            var weight=1.0;
            if (p.mapping.z&255u)==1u { weight=axis_weight(v,view_dir,sun); }
            value+=incident_lookup(i,s,v,sun)
                *cached_phase_weight(c,q)*q.direction_weight.w*weight;
            if (p.mapping.z&255u)==1u {
                let incoming=rotate(q.direction_weight.xyz,sun);
                value+=incident_lookup(i,s,incoming,sun)
                    *phase_weight(c,clamp(dot(incoming,view_dir),-1.0,1.0))
                    *q.direction_weight.w*axis_weight(incoming,sun,view_dir);
            }
        }
    }
    density[i]=value;
}

@compute @workgroup_size(64)
fn integrate_direct(@builtin(global_invocation_id) id:vec3u) {
    let work=p.work.z+id.x;if work>=p.work.w {return;}let i=work_scatter_index(work);
    if canonical_scatter_index(i)!=i {return;}
    let s=state(i); let d=distance(s.h,s.mu,s.ground); let dx=d/f32(p.integration.x);
    // Keep the finite-disc cache below the large private-array pressure cliff.
    // Every original quadrature sample is still evaluated; only summation groups change.
    var phases:array<vec4f,32>;var solar_geometry:array<vec4f,32>;
    let ray=view_direction(s);let sun=sun_direction(s);
    var value=0.0;var final_trans=1.0;
    for(var begin=0u;begin<p.work.x;begin+=32u) {
        let count=min(32u,p.work.x-begin);
        for(var q=0u;q<count;q++) {
            let sample=angular[p.integration.z+p.integration.w+begin+q].direction_weight;
            let v=rotate(sample.xyz,sun);let nu=clamp(dot(ray,v),-1.0,1.0);let w=sample.w*p.planet.w;
            phases[q]=vec4f(3.0*(1.0+nu*nu)/(16.0*PI),aerosol_phase(0u,nu),aerosol_phase(1u,nu),aerosol_phase(2u,nu))*w;
            solar_geometry[q]=vec4f(aerosol_phase(3u,nu)*w,v.z,nu,0.0);
        }
        var trans=1.0;var chunk=0.0;
        for(var j=0u;j<p.integration.x;j++) {
            let travel=(f32(j)+0.5)*dx;let point=advance(s,travel);let c=coefficients(point.h);let ext=c.a.y;let optical=ext*dx;
            var cell_weight=dx;
            if optical<0.001 { cell_weight=dx*(1.0-optical*0.5+optical*optical/6.0); }
            else { cell_weight=(1.0-exp(-optical))/ext; }
            var source=0.0;
            for(var q=0u;q<count;q++) {
                let cached=solar_geometry[q];
                let mu=((p.planet.x+s.h)*cached.y+travel*cached.z)/(p.planet.x+point.h);
                let weight=dot(vec4f(c.a.zw,c.b.xy),phases[q])+c.b.z*cached.x;
                source+=sun_transmittance(point.h,mu)*weight;
            }
            chunk+=source*trans*cell_weight;trans*=exp(-optical);
        }
        value+=chunk;final_trans=trans;
    }
    if s.ground { value+=final_trans*p.planet.z/PI*ground_lookup(advance(s,d).mu_s); }
    next_order[i]=value;
}

// A separate pipeline keeps the finite-disc private cache out of every later
// order's register allocation and out of the older solver paths.
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id:vec3u) {
    let work=p.work.z+id.x;if work>=p.work.w {return;}let i=work_scatter_index(work);
    if canonical_scatter_index(i)!=i {return;}
    let s=state(i);let d=distance(s.h,s.mu,s.ground);let dx=d/f32(p.integration.x);
    var trans=1.0;var value=0.0;
    for(var j=0u;j<p.integration.x;j++) {
        let point=advance(s,(f32(j)+0.5)*dx);let ext=coefficients(point.h).a.y;let optical=ext*dx;
        var cell_weight=dx;
        if optical<0.001 {cell_weight=dx*(1.0-optical*0.5+optical*optical/6.0);}
        else {cell_weight=(1.0-exp(-optical))/ext;}
        value+=lookup(point,true)*trans*cell_weight;trans*=exp(-optical);
    }
    if s.ground {value+=trans*p.planet.z/PI*ground_lookup(advance(s,d).mu_s);}
    if (p.mapping.z&256u)!=0u {accumulated[i]=next_order[i]+value;}
    else {next_order[i]=value;}
}

var<workgroup> reductions:array<vec4f,64>;
var<workgroup> local_increments:array<f32,64>;
@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=p.work.z+id.x;
    var v=vec4f(0.0);
    var local=0.0;
    if i<p.work.w {
        let canonical=canonical_scatter_index(i);
        var delta=next_order[canonical];var sum=accumulated[i]+delta;
        if (p.mapping.z&256u)!=0u && p.work.y>1u {
            sum=accumulated[canonical];delta=abs(sum-previous[i]);
            // Canonical radiance was completed by integrate. Only aliases are
            // written here; canonical reads cannot race with any thread's write.
            if i!=canonical {accumulated[i]=sum;}
            previous[i]=sum;
        } else {
            accumulated[i]=sum;previous[i]=delta;
        }
        // Invalid transport is surfaced in the readback diagnostic, never silently clamped.
        if !(delta>=0.0 && delta<=3.402823e38 && sum>=0.0 && sum<=3.402823e38) { v=vec4f(-1.0); }
        else { v=vec4f(delta,sum,delta,sum); }
        local=delta/max(sum,p.planet.w*1e-12);
    }
    reductions[lane]=v;local_increments[lane]=local; workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {
            let a=reductions[lane]; let b=reductions[lane+stride];
            if a.x<0.0 || b.x<0.0 { reductions[lane]=vec4f(-1.0); }
            else { reductions[lane]=vec4f(max(a.xy,b.xy),a.zw+b.zw); }
            local_increments[lane]=max(local_increments[lane],local_increments[lane+stride]);
        }
        workgroupBarrier();
    }
    if lane==0u {
        let base=tau_len()+p.tables.z+(i/64u)*5u;
        for(var a=0u;a<4u;a++) { auxiliary[base+a]=reductions[0][a]; }
        auxiliary[base+4u]=local_increments[0];
    }
}

// Cached phase duplicates are identical f32 values, not approximate clusters.
// CPU-identified collapsed view intervals are also checked in shader arithmetic:
// an endpoint that rounds differently on this adapter keeps its own evaluation.
fn canonical_scatter_index(i:u32)->u32 {
    if !REUSE_DUPLICATE_STATES {return i;}
    if p.mapping.x<3u {return i;}
    let ri=i/(p.dims.y*p.dims.z*p.dims.w);let si=i/p.dims.w%p.dims.z;
    let mi=i/(p.dims.z*p.dims.w)%p.dims.y;let ni=i%p.dims.w;
    let ng=max(p.dims.y/4u,2u);let ground=mi<ng;
    let base=4096u+p.dims.w+p.dims.x+p.dims.x*p.dims.z+p.dims.x*p.dims.z*2u*p.dims.w;
    let code=u32(phase_value(base+((ri*p.dims.z+si)*2u+select(0u,1u,ground))*p.dims.w+ni));
    var cm=mi;
    if code>=p.dims.w {
        let candidate=select(ng,0u,ground);
        let h=reference_radius(ri);let mus=reference_solar(ri,si);
        let nu=reference_phase(ri,si,ni,ground);
        if cone_view(h,mus,nu,mi,p.dims.y)==cone_view(h,mus,nu,candidate,p.dims.y) {cm=candidate;}
    }
    return ((ri*p.dims.y+cm)*p.dims.z+si)*p.dims.w+code%p.dims.w;
}

// Separate dispatch after density completes; aliases always point directly to
// independently computed nodes, so expansion never reads another alias write.
@compute @workgroup_size(64)
fn expand_density(@builtin(global_invocation_id) id:vec3u) {
    let i=p.work.z+id.x;if i>=p.work.w {return;}
    let canonical=canonical_scatter_index(i);
    if canonical!=i {density[i]=density[canonical];}
}

fn combine_statistics(a:vec4f,b:vec4f)->vec4f {
    if any(!(a>=vec4f(0.0))) || any(!(b>=vec4f(0.0)))
        || any(a>vec4f(3.402823e38)) || any(b>vec4f(3.402823e38)) {return vec4f(-1.0);}
    let sums=a.zw+b.zw;
    if any(!(sums<=vec2f(3.402823e38))) {return vec4f(-1.0);}
    return vec4f(max(a.xy,b.xy),sums);
}

// Reduce 256 first-pass records per workgroup before CPU readback. Maxima and
// invalid flags are preserved; only the diagnostic sum's addition order changes.
@compute @workgroup_size(64)
fn reduce_statistics(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32) {
    let group=(p.work.z+id.x)/64u;let rows=(scatter_len()+63u)/64u;
    let input=tau_len()+p.tables.z;var v=vec4f(0.0);var local=0.0;
    for(var k=0u;k<4u;k++) {
        let row=group*256u+lane+k*64u;
        if row<rows {
            let start=input+row*5u;
            v=combine_statistics(v,vec4f(auxiliary[start],auxiliary[start+1u],auxiliary[start+2u],auxiliary[start+3u]));
            local=max(local,auxiliary[start+4u]);
        }
    }
    if group==0u {
        for(var i=lane;i<p.tables.z;i+=64u) {
            let ratio=auxiliary[ground_stats_base()+i];
            if ratio<0.0 {v=vec4f(-1.0);}else {local=max(local,ratio);}
        }
    }
    reductions[lane]=v;local_increments[lane]=local;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {
            reductions[lane]=combine_statistics(reductions[lane],reductions[lane+stride]);
            local_increments[lane]=max(local_increments[lane],local_increments[lane+stride]);
        }
        workgroupBarrier();
    }
    if lane==0u {
        let start=input+rows*5u+group*5u;
        for(var k=0u;k<4u;k++) {auxiliary[start+k]=reductions[0][k];}
        auxiliary[start+4u]=local_increments[0];
    }
}
