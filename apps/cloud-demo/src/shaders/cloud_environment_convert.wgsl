struct Normalize { solar:vec4<f32>, rgb:array<vec4<f32>,4>, scale:vec4<f32> }
@group(0) @binding(0) var input:texture_2d<f32>;
@group(0) @binding(1) var output:texture_storage_2d<rgba32float,write>;
@group(0) @binding(2) var<uniform> norm:Normalize;
@group(0) @binding(3) var solar_t:texture_2d<f32>;
@group(0) @binding(4) var sunlight:texture_storage_2d<rgba32float,write>;
fn linear_srgb(v:vec3<f32>)->vec3<f32>{
    return max(vec3<f32>(1.660491*v.x-0.5876411*v.y-0.0728499*v.z,
        -0.1245505*v.x+1.1328999*v.y-0.0083494*v.z,
        -0.0181508*v.x-0.1005789*v.y+1.1187297*v.z)*norm.scale.x,vec3<f32>(0.0));
}
@compute @workgroup_size(8,8) fn convert(@builtin(global_invocation_id) id:vec3<u32>){
    if any(id.xy>=textureDimensions(output)){return;}
    textureStore(output,vec2<i32>(id.xy),vec4<f32>(linear_srgb(textureLoad(input,vec2<i32>(id.xy),0).xyz),1.0));
    if all(id.xy==vec2<u32>(0u)){
        let e=norm.solar*textureLoad(solar_t,vec2<i32>(0),0);var rgb=vec3<f32>(0.0);
        for(var k=0u;k<4u;k++){rgb+=e[k]*norm.rgb[k].xyz;}
        textureStore(sunlight,vec2<i32>(0),vec4<f32>(linear_srgb(rgb),1.0));
    }
}
