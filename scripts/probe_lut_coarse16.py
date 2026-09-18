"""CPU-only nested-grid/scaled-f16 baseline. Original coordinates, fewer nodes.

Height/Sun/phase nodes are exact subsets of the frozen reference. Its view
interpolator is linear, so resampling that axis commutes with spectral-to-RGB.
decoded_*.f32 are CPU audit intermediates, not part of the GPU payload.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("source",type=Path)
p.add_argument("out",type=Path)
p.add_argument("--shape",type=int,nargs=4,default=[36,16,65,65])
p.add_argument("--bc-source",action="store_true",help="save unquantized CPU inputs for a separate BC6H experiment")
p.add_argument("--sun",type=Path,default=Path("out/lut_16mb_log12"))
a=p.parse_args()
if a.out.exists():raise ValueError("output exists")
m=json.loads((a.source/"asset.json").read_text())
nr,nv,ns,nn=m['config']['scattering']
h,v,s,n=a.shape
assert (ns-1)%(s-1)==0 and (nn-1)%(n-1)==0
assert m['config']['view_interpolation']=='linear'
height=np.asarray(m['config']['scattering_altitudes_km'],dtype=np.float32)
selected={0,nr-1}
for altitude in [0.2,1,2,11,12,35]:selected.add(int(np.argmin(abs(height-altitude))))
while len(selected)<h:
    distance=np.min(abs(np.arange(nr)[:,None]-np.asarray(sorted(selected))[None,:]),axis=1)
    selected.add(int(np.argmax(distance)))
hi=np.asarray(sorted(selected))
ng=max(v//4,2);old_ng=max(nv//4,2)
vc=np.concatenate([np.linspace(0,old_ng-1,ng,dtype=np.float32),np.linspace(old_ng,nv-1,v-ng,dtype=np.float32)])
vl=vc.astype(int);vu=np.minimum(vl+1,nv-1);vt=(vc-vl.astype(np.float32)).astype(np.float32)
si=np.arange(s)*(ns-1)//(s-1);ni=np.arange(n)*(nn-1)//(n-1)
raw=[]
for channel in range(3):
    data=np.memmap(a.source/f'channel_{channel}.bin',dtype='<f4',mode='r',
        offset=8+int(np.prod(m['config']['optical_depth']))*4,shape=(nr,nv,ns,nn))
    low=data[np.ix_(hi,vl,si,ni)];up=data[np.ix_(hi,vu,si,ni)]
    raw.append(low+(up-low)*vt[None,:,None,None])
raw=np.stack(raw,axis=-1).astype(np.float32)
peak=np.max(np.abs(raw),axis=(3,4))
exponent=np.floor(np.log2(np.maximum(peak,np.float32(1e-30)))).astype(np.float32)-np.float32(10)
scale=np.exp2(exponent).astype('<f4')
packed=(raw/scale[...,None,None]).astype('<f2')
decoded=packed.astype(np.float32)*scale[...,None,None]
if a.bc_source:decoded=raw
config=m['config'].copy();config['scattering']=a.shape;config['scattering_altitudes_km']=height[hi].tolist()
sun_metadata=json.loads((a.sun/'candidate.json').read_text())
budget=packed.nbytes+scale.nbytes+sun_metadata['sun_bytes']+131072
planned_bc_bytes=h*v*s*((n+15)//16)*17+sun_metadata['sun_bytes']+131072
assert (planned_bc_bytes if a.bc_source else budget)<=16_000_000
if a.bc_source:budget=raw.nbytes+sun_metadata['sun_bytes']+131072
a.out.mkdir(parents=True)
if not a.bc_source:
    packed.tofile(a.out/'radiance.f16');scale.tofile(a.out/'row_scale.f32')
for c in range(3):decoded[...,c].astype('<f4').tofile(a.out/f'decoded_{c}.f32')
shutil.copyfile(a.sun/'sun.f16',a.out/'sun.f16')
metadata=dict(kind='cpu_nested_grid_f16_16mb_probe_v1',source=str(a.source),source_config=m['config'],
    source_channel_checksums=m['rgb']['channel_checksums'],source_model=m['model_fingerprint_fnv1a64'],
    config=config,shape=a.shape,height_indices=hi.tolist(),radiance_bytes=packed.nbytes,scale_bytes=scale.nbytes,
    sun_bytes=sun_metadata['sun_bytes'],sun_shape=sun_metadata['sun_shape'],metadata_reserve=131072,
    gpu_payload_budget_bytes=budget,cpu_audit_intermediates='decoded_*.f32 are excluded from GPU payload',
    fitting_dtype='float32',sky_only=True,finite_segment_transport=False,gpu_integrated=False)
if a.bc_source:
    metadata['kind']='cpu_uncompressed_nested_grid_baseline'
    metadata['planned_bc_bytes']=planned_bc_bytes
(a.out/'candidate.json').write_text(json.dumps(metadata,indent=2))
print(a.shape,budget,flush=True)
