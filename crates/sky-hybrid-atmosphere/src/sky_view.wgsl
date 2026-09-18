// Observer SkyView charts, independent of the fixed-medium solver.
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
    if (mapping_flags()&32u)!=0u {
        let base=sky_mapping[4u*chart];let x=clamp(e,base.z,base.w);var y=base.x*(x-base.z);
        for(var j=1u;j<=3u;j++){
            let c=sky_mapping[4u*chart+j];let d=x-c.x;
            if c.x<base.z||c.x>base.w{y+=(x-base.z)*c.z/(abs(d)+c.y);}
            else{y+=c.z*(d/(abs(d)+c.y)-c.w);}
        }
        return clamp(y,0.0,1.0);
    }
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
@compute @workgroup_size(8,8) fn build_sky(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=p.sky.xy){return;}
    let lower_rows=p.sky.z;
    var chart=0u;var start=0u;var rows=lower_rows;
    if id.y>=p.sky.w{chart=2u;start=p.sky.w;rows=p.sky.y-p.sky.w;}
    else if id.y>=lower_rows{chart=1u;start=lower_rows;rows=p.sky.w-lower_rows;}
    if chart==1u&&p.view.w>=p.medium.x{textureStore(sky_write,vec2<i32>(id.xy),vec4<f32>(log(vec3<f32>(1e-30)),0.0));return;}
    let hit=chart==2u;
    let u=f32(id.x)/f32(p.sky.x-1u);
    var a=u*u;if (mapping_flags()&32u)!=0u{a*=a;}
    let cp=1.0-2.0*a;
    // Offset the duplicated horizon endpoints onto their own boundary branch.
    var mu=sky_mapping[12u+id.y].x;let hor=horizon(p.view.w);
    if hit{mu=min(mu,hor-1e-7);}else{mu=max(mu,hor+1e-7);}
    let r=sqrt(max(1.0-mu*mu,0.0));
    let ray=vec3<f32>(r*2.0*sqrt(max(a*(1.0-a),0.0)),mu,r*cp);
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
    var u=sqrt(max((1.0-cp)*0.5,0.0));if (mapping_flags()&32u)!=0u{u=sqrt(u);}let hit=ground(p.view.w,ray.y);
    let lower_rows=p.sky.z;var chart=0u;var start=0u;var rows=lower_rows;
    if hit{chart=2u;start=p.sky.w;rows=p.sky.y-p.sky.w;}
    else if ray.y>=0.0{chart=1u;start=lower_rows;rows=p.sky.w-lower_rows;}
    let y=elevation_coord(asin(clamp(ray.y,-1.0,1.0)),chart);
    let row=f32(start)+y*f32(rows-1u);let end=start+rows-1u;
    let x=u*f32(p.sky.x-1u);
    let ix=min(u32(x),p.sky.x-2u);let iy=clamp(u32(row),start,end-1u);
    let a=mix(textureLoad(sky_read,vec2<i32>(i32(ix),i32(iy)),0).xyz,textureLoad(sky_read,vec2<i32>(i32(ix+1u),i32(iy)),0).xyz,x-f32(ix));
    let b=mix(textureLoad(sky_read,vec2<i32>(i32(ix),i32(iy+1u)),0).xyz,textureLoad(sky_read,vec2<i32>(i32(ix+1u),i32(iy+1u)),0).xyz,x-f32(ix));
    if p.view.w>=p.medium.x&&!hit&&iy==lower_rows-2u {
        // At the outer limb radiance tends to zero with the atmospheric chord,
        // not exponentially. Log interpolation into a vacuum endpoint destroys
        // the thin shell even when the sky chart ends at the correct tangent.
        let previous=sky_mapping[12u+lower_rows-2u].x;
        let previous2=sky_mapping[12u+lower_rows-3u].x;
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
    let ray=camera_ray(id.xy);let sun=world_sun();
    var rgb=sky_sample(ray);let half_radius=sin(p.sun.z*0.5);
    if (p.size_steps.w&2u)!=0u&&!ground(p.view.w,ray.y)&&dot(ray-sun,ray-sun)<=4.0*half_radius*half_radius {
        let light=view_transmittance(ray)*p.solar/(4.0*PI*half_radius*half_radius);
        for(var k=0u;k<4u;k++){rgb+=light[k]*p.rgb[k].xyz;}
    }
    textureStore(output,vec2<i32>(id.xy),vec4<f32>(rgb,1.0));
}
