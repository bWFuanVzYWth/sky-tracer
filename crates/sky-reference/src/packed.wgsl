@group(0) @binding(12) var<storage,read> packed_data:array<u32>;
@group(0) @binding(13) var<storage,read> packed_blocks:array<u32>;

fn packed_radiance(index:u32)->f32 {
    let n=frame.storage.y;
    let offset=packed_blocks[index/n];
    let channel=frame.size_band.z;
    let header=packed_data[offset+channel];
    let width=header>>20u;
    let base=header&1048575u;
    if width==0u {return bitcast<f32>(base<<12u);}
    var bit=(index%n)*width;
    for(var c=0u;c<channel;c++) {bit+=n*(packed_data[offset+c]>>20u);}
    let word=offset+3u+bit/32u;
    let shift=bit%32u;
    var delta=packed_data[word]>>shift;
    if shift+width>32u {delta|=packed_data[word+1u]<<(32u-shift);}
    delta&=(1u<<width)-1u;
    return bitcast<f32>((base+delta)<<12u);
}
