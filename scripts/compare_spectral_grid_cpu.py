"""Compare 8->4 spectral loss and LUT grid loss on exactly the same queries."""
import argparse
import json
from pathlib import Path
import numpy as np
from search_wavelengths_cpu import read_dataset

F=np.float32
p=argparse.ArgumentParser()
p.add_argument('dataset',type=Path)
p.add_argument('--grid-probe',type=Path,required=True)
p.add_argument('--eight',type=Path,required=True)
p.add_argument('--four',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise ValueError('use a new output directory')
meta,plan,data=read_dataset(a.dataset)
grid=json.loads(a.grid_probe.read_text())
eight=json.loads(a.eight.read_text());four=json.loads(a.four.read_text())
assert meta['model']==grid['model']==eight['model']==four['model']
assert meta['source_band_checksums']==grid['source_checksums']==eight['source_band_checksums']==four['source_band_checksums']
assert grid['steps']==meta['ray_steps'] and grid['sun_samples']==meta['sun_disk_samples']
rows=grid['rows']
assert len(rows)==len(plan['queries'])
assert all(all(F(point[k])==F(row['point'][k]) for k in point) for point,row in zip(plan['queries'],rows))
teacher=np.array([r['teacher'] for r in rows],dtype=F)
norm=np.linalg.norm(teacher,axis=1)
W=np.array(meta['rgb_from_integrated'],dtype=F)
direct41=data['direct']@W
original_direct=np.array([r['single'] for r in rows],dtype=F)+np.array([r['boundary'] for r in rows],dtype=F)
denom=np.maximum(norm,F(1e-30))
mask=norm>F(1e-8)
checks=dict(teacher_max_relative=float(np.max(np.linalg.norm(data['total']@W-teacher,axis=1)[mask]/denom[mask])),
            direct41_max_relative=float(np.max(np.linalg.norm(direct41-original_direct,axis=1)[mask]/denom[mask])))
assert max(checks.values())<5e-5,checks
direct8=data['direct'][:,eight['indices']]@np.array(eight['rgb_from_integrated'],dtype=F)
direct4=data['direct'][:,four['indices']]@np.array(four['rgb_from_integrated'],dtype=F)
get=lambda name,kind:np.array([r['candidates'][name][kind] for r in rows],dtype=F)
coarse_total=get('budget_rgb16','total_lut');full_total=get('full','total_lut')
coarse_ms=get('budget_rgb16','hybrid_signed');full_ms=get('full','hybrid_signed')
coarse_bf16=get('budget_rgb16','hybrid_bf16')
delta=dict(
    spectral_8_to_4=direct4-direct8,
    spectral_41_to_4=direct4-direct41,
    spectral_41_to_8=direct8-direct41,
    total_grid_coarsening=coarse_total-full_total,
    multiple_grid_coarsening=coarse_ms-full_ms,
    full_grid_split_vs_teacher=full_ms-teacher,
    bf16_multiple_vs_teacher=coarse_bf16-teacher,
    combined_bf16_multiple_8=coarse_bf16+(direct8-original_direct)-teacher,
    combined_bf16_multiple_4=coarse_bf16+(direct4-original_direct)-teacher,
)
errors={k:F(100)*np.linalg.norm(v,axis=1)/denom for k,v in delta.items()}
def stat(v):
    return dict(count=len(v),p95=float(np.quantile(v,F(.95))),p99=float(np.quantile(v,F(.99))),maximum=float(np.max(v))) if len(v) else dict(count=0)
regions=['all']+list(dict.fromkeys(r['region'] for r in rows))
summary={}
for region in regions:
    chosen=np.array([region=='all' or r['region']==region for r in rows])
    summary[region]={f'above_{floor:g}':{k:stat(v[chosen&(norm>F(floor))]) for k,v in errors.items()} for floor in [1e-8,1e-4]}
    print(region, {k:round(v['p95'],4) for k,v in summary[region]['above_1e-08'].items() if v['count']})
a.out.mkdir(parents=True)
result=dict(kind='same_query_spectral_grid_comparison_v1',queries=len(rows),dataset=str(a.dataset),grid_probe=str(a.grid_probe),
    source_band_checksums=meta['source_band_checksums'],checks=checks,eight=str(a.eight),four=str(a.four),
    grids=grid['grids'],regions=summary,
    metric='P95/P99/max of percent linear Rec.2020 vector differences, normalized by the same full reference total. Legacy grid-probe denominator; no 1e-7 floor. Positive-norm masks stated separately.',
    definitions=dict(spectral_8_to_4='D4(query)-D8(query), both direct integrations at 256 steps',
        total_grid_coarsening='coarse total RGB LUT minus full total RGB LUT, f32 before quantization',
        multiple_grid_coarsening='coarse M versus full M, identical direct query integration, f32 before quantization',
        full_grid_split_vs_teacher='full-grid split result minus original spectral teacher; includes reference direct interpolation inconsistency and nonadditivity',
        combined='bf16 coarse M plus selected-spectrum direct at 256 uniform steps; no fast runtime marching or auxiliary compression'),
    note='Same deterministic 768-query audit as earlier grid study, not a new blind set or an independent physical truth. Percentiles of differences are not additive.')
(a.out/'summary.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(a.out/'errors_percent.npz',**errors,norm=norm)
print('checks',checks)
