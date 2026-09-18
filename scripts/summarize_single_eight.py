"""Compare searched eight-wave runtime integration to its frozen 41-band control."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('probe',type=Path)
a=p.parse_args()
m=json.loads(a.probe.read_text())
rows=m['rows']
rgb=lambda key:np.asarray([r[key] for r in rows],dtype=np.float32)
teacher=rgb('teacher');length=np.linalg.norm(teacher,axis=1)
s8,s41,b8,b41=[rgb(k) for k in ['single8','single41','boundary8','boundary41']]
error=lambda delta:100*np.linalg.norm(delta,axis=1)/np.maximum(length,np.float32(1e-30))

def stat(values,mask):
    v=values[mask]
    if not len(v):return {'count':0}
    return dict(count=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(np.max(v)))

metrics={
    'single_spectral_reduction':error(s8-s41),
    'boundary_spectral_reduction':error(b8-b41),
    'direct_spectral_reduction':error(s8+b8-s41-b41),
    'hybrid41_baseline':error(rgb('hybrid41')-teacher),
    'hybrid8_baseline':error(rgb('hybrid8')-teacher),
}
for i,item in enumerate(rows[0]['runtime']):
    spec=item['spec']
    name=f"{spec['steps']}_"+('point' if spec['point_sun'] else 'disk')+('_height' if spec['log_height'] else '_uniform')
    arrays={k:np.asarray([r['runtime'][i][k] for r in rows],dtype=np.float32) for k in ['direct8','direct41','hybrid8','hybrid41']}
    metrics['spectral_'+name]=error(arrays['direct8']-arrays['direct41'])
    metrics['hybrid8_'+name]=error(arrays['hybrid8']-teacher)
    metrics['hybrid41_'+name]=error(arrays['hybrid41']-teacher)

result={'source':str(a.probe),'queries':len(rows),'regions':{},
        'metric':'Percent linear Rec.2020 vector error, normalized by full teacher total radiance. This is a selected-query P95, not a dense-set P99 or physical ground truth.',
        'runtime_wavelengths_nm':m['runtime_wavelengths_nm'],'runtime_quadrature_weights_nm':m['runtime_quadrature_weights_nm']}
for region in ['all']+list(dict.fromkeys(r['region'] for r in rows)):
    chosen=np.asarray([region=='all' or r['region']==region for r in rows])
    result['regions'][region]={f'above_{floor:g}':{key:stat(value,chosen & (length>floor)) for key,value in metrics.items()} for floor in [1e-8,1e-4]}
    s=result['regions'][region]['above_1e-08']
    print(region,'count',s['single_spectral_reduction']['count'])
    for key in ['single_spectral_reduction','boundary_spectral_reduction','hybrid41_64_disk_height','hybrid8_64_disk_height']:
        if s[key]['count']:print(f"  {key:36s} P95 {s[key]['p95']:7.4f}% max {s[key]['maximum']:7.4f}%")
(a.probe.parent/'summary.json').write_text(json.dumps(result,indent=2))
