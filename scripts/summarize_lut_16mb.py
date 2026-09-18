"""Summarize the CPU 16 MB experiment, including targeted angular masks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path('out/lut_16mb_candidate')
metadata=json.loads((root/'candidate.json').read_text())
validation=json.loads((root/'validation.json').read_text())
queries=json.loads(Path('out/lut_v6_packed_cpu.queries.json').read_text())
colors=json.loads(Path('out/lut_v6_packed_cpu.colors.json').read_text())
reference=np.asarray([c['spectral'] for c in colors],dtype=np.float32)
candidate=np.fromfile(root/'query_rgb.f32',dtype='<f4').reshape(-1,3)
norm=np.sqrt(np.sum(reference*reference,axis=1))
error=np.sqrt(np.sum((candidate-reference)**2,axis=1))/np.maximum(norm,np.float32(1e-30))
regions={}
for scene,offset in [('noon',0),('sunset',2*8309),('blue_hour',3*8309),('orbit',7*8309)]:
    ids=np.arange(offset,offset+8309)
    e=np.deg2rad(np.asarray([queries[i]['view_elevation_deg'] for i in ids],dtype=np.float32))
    s=np.deg2rad(np.asarray([queries[i]['sun_elevation_deg'] for i in ids],dtype=np.float32))
    az=np.deg2rad(np.asarray([queries[i]['relative_azimuth_deg'] for i in ids],dtype=np.float32))
    angle=np.rad2deg(np.arccos(np.clip(np.sin(e)*np.sin(s)+np.cos(e)*np.cos(s)*np.cos(az),-1,1)))
    masks={'horizon_6deg_band':np.arange(len(ids))<4096,
           'solar_0.27_to_2deg':(angle>=.27)&(angle<2),
           'solar_10_to_30deg':(angle>=10)&(angle<30)}
    for name,mask in masks.items():
        selected=ids[mask&(norm[ids]>np.float32(1e-8))]
        if len(selected):regions[f'{scene}/{name}']=dict(samples=len(selected),
            p50_p95_p99_max=np.percentile(error[selected],[50,95,99,100]).tolist())

comparison=[]
for path in Path('out').glob('lut_16mb_*/validation.json'):
    m=json.loads(path.read_text())
    comparison.append(dict(name=path.parent.name,bytes=m['gpu_payload_budget_bytes'],**m['cases']['all']['vs_spectral']))
summary=dict(selected='lut_16mb_candidate',gpu_bytes_including_reserve=metadata['gpu_payload_budget_bytes'],
    critical_regions=regions,candidates=comparison,all_cpu=True,reference='frozen v6 spectral LUT',
    metrics_floor_rgb_norm=1e-8,finite_segment_supported=False,gpu_runtime_integrated=False)
Path('out/lut_16mb_summary.json').write_text(json.dumps(summary,indent=2))

names=['noon','afternoon','sunset','blue_hour','aircraft_twilight','stratosphere_shadow','atmosphere_edge','orbit']
labels=['Noon','47 deg Sun','Sunset','Blue hour','12 km twilight','30 km shadow','120 km edge','400 km orbit']
fig,(ax,bx)=plt.subplots(1,2,figsize=(12.6,4.8),layout='constrained')
x=np.arange(len(names))
for offset,key,label,color in [(-.18,'relative_p95','P95','#207f7a'),(.18,'relative_p99','P99','#94c0ba')]:
    values=[validation['cases'][n]['vs_spectral'][key]*100 for n in names]
    bars=ax.barh(x+offset,values,.34,label=label,color=color)
    ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
ax.set_yticks(x,labels);ax.invert_yaxis();ax.set_xlim(0,24);ax.legend(loc='lower right')
ax.set_xlabel('RGB error against the spectral teacher (%)')
ax.set_title('15.994 MB: sky accuracy by scene')
for name,label,color in [('bits_grid_source','Same grid, unquantized','#5e728f'),
                         ('candidate','Selected block codec','#207f7a'),
                         ('bc6h_linear','Dense grid + BC6H','#c08553'),
                         ('log20i8','Local PCA, rank 20','#bc9f77')]:
    values=np.fromfile(Path('out')/f'lut_16mb_{name}'/'query_rgb.f32',dtype='<f4').reshape(-1,3)
    rel=np.sqrt(np.sum((values-reference)**2,axis=1))/np.maximum(norm,np.float32(1e-30))
    rel=np.sort(rel[norm>np.float32(1e-8)])*100
    bx.plot(rel,np.arange(1,len(rel)+1)/len(rel)*100,label=label,color=color)
bx.set_xscale('log');bx.set_xlim(.01,100);bx.set_ylim(0,100);bx.set_xlabel('Relative RGB error (%)');bx.set_ylabel('Queries at or below this error (%)')
bx.set_title('The remaining tail comes mainly from grid reduction');bx.grid(alpha=.2);bx.legend(loc='lower right',fontsize=8)
fig.suptitle('Strict 16,000,000-byte LUT budget; CPU prototypes',fontsize=14)
fig.text(.5,-.04,'82,856 queries; relative statistics use 67,387 with RGB norm > 1e-8. Direct Sun and finite-segment fog are excluded from radiance metrics.',ha='center',fontsize=8)
fig.savefig('out/lut_16mb_quality.png',dpi=170,bbox_inches='tight')
print(json.dumps(regions,indent=2))
