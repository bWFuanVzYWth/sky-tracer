"""Summarize a sparse, CPU single-scattering + multiple remainder experiment."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('probe', type=Path)
a = p.parse_args()
m = json.loads(a.probe.read_text())
rows = m['rows']
teacher = np.asarray([r['teacher'] for r in rows], dtype=np.float32)
norm = np.linalg.norm(teacher, axis=1)
regions = ['all'] + list(dict.fromkeys(r['region'] for r in rows))
baselines = {}
if m['full_spectral_rec2020']:
    n = len(rows) // 12
    old_indices = [j*8309 + min((2*k+1)*((82856-j*8309) if j==8 else 8309)//(2*n),((82856-j*8309) if j==8 else 8309)-1) for j in range(9) for k in range(n)]
    fresh_indices = [begin+min((2*k+1)*(end-begin)//(2*n),end-begin-1) for begin,end in [(0,4096),(4096,14491),(14491,18091)] for k in range(n)]
    # These indexes reproduce the Rust probe's deterministic query selection.
    all_points = json.loads(Path('out/lut_v6_packed_cpu.queries.json').read_text())
    fresh_points = json.loads(Path('out/lut_joint16_fresh_queries/queries.json').read_text())
    selected = [all_points[i] for i in old_indices]+[fresh_points[i] for i in fresh_indices]
    for row, point in zip(rows,selected,strict=True):
        assert all(np.float32(row['point'][k]) == np.float32(point[k]) for k in point)
    for name,path in [('joint_tt','out/lut_joint16_hybrid_floor8'),('previous_sparse16','out/lut_16mb_candidate')]:
        old = np.fromfile(Path(path)/'query_rgb.f32',dtype='<f4').reshape(-1,3)
        fresh = np.fromfile(Path(path)/'fresh_query_rgb.f32',dtype='<f4').reshape(-1,3)
        baselines[name] = np.concatenate([old[old_indices],fresh[fresh_indices]])

def stats(error, mask):
    e = error[mask]
    if not len(e):
        return {'count': 0}
    return dict(count=len(e), median=float(np.median(e)), p95=float(np.quantile(e,.95)),
                maximum=float(np.max(e)))

out = {'source': str(a.probe), 'full_spectral_rec2020': m['full_spectral_rec2020'],
       'note': 'Percent relative RGB-vector error vs frozen spectral v6. Small stratified sample: P95 and maximum, not a large-set P99 claim.',
       'regions': {}}
for region in regions:
    chosen = np.asarray([region == 'all' or r['region'] == region for r in rows])
    masks = {'above_1e-8': chosen & (norm > 1e-8), 'above_1e-4': chosen & (norm > 1e-4)}
    item = {}
    for floor, mask in masks.items():
        result = {}
        for name,v in baselines.items():
            result[name] = stats(100*np.linalg.norm(v-teacher,axis=1)/np.maximum(norm,1e-30),mask)
        for grid in m['grids']:
            methods = ['total_lut', 'hybrid_signed', 'hybrid_clamped']
            if 'hybrid_bf16' in rows[0]['candidates'][grid['name']]:
                methods += ['total_bf16', 'hybrid_bf16']
            for method in methods:
                v = np.asarray([r['candidates'][grid['name']][method] for r in rows], dtype=np.float32)
                e = 100 * np.linalg.norm(v - teacher,axis=1) / np.maximum(norm, 1e-30)
                result[grid['name'] + '/' + method] = stats(e, mask)
        direct = np.asarray([np.asarray(r['single']) + np.asarray(r['boundary']) for r in rows], dtype=np.float32)
        converged_index = next((i for i,s in enumerate(m['runtime_specs']) if isinstance(s,dict) and s['steps']==1024), None)
        converged = np.asarray([r['runtime_direct'][converged_index] for r in rows], dtype=np.float32) if converged_index is not None else direct
        for i, spec in enumerate(m['runtime_specs']):
            if isinstance(spec,dict):
                steps,point,warped = spec['steps'],spec['point_sun'],spec['log_height']
            else:
                steps,point = spec
                warped = False
            v = np.asarray([r['runtime_direct'][i] for r in rows], dtype=np.float32)
            e = 100 * np.linalg.norm(v - direct,axis=1) / np.maximum(norm,1e-30)
            key = f'march_{steps}_' + ('point' if point else 'disk') + ('_height' if warped else '')
            result[key] = stats(e, mask)
            ce = 100 * np.linalg.norm(v - converged,axis=1) / np.maximum(norm,1e-30)
            result[key+'_vs1024'] = stats(ce,mask)
            if 'hybrid_bf16' in rows[0]['candidates']['budget_rgb16']:
                base = np.asarray([r['candidates']['budget_rgb16']['hybrid_bf16'] for r in rows],dtype=np.float32)
                combined = base + v - direct
                result['hybrid_bf16_'+key] = stats(100*np.linalg.norm(combined-teacher,axis=1)/np.maximum(norm,1e-30),mask)
        if 'interpolated_direct' in rows[0]['candidates']['full']:
            sampled_direct = np.asarray([r['candidates']['full']['interpolated_direct'] for r in rows],dtype=np.float32)
            hybrid = np.asarray([r['candidates']['full']['hybrid_signed'] for r in rows],dtype=np.float32)
            result['direct_query_minus_interpolated'] = stats(100*np.linalg.norm(direct-sampled_direct,axis=1)/np.maximum(norm,1e-30),mask)
            result['nonadditive_interpolation'] = stats(100*np.linalg.norm(hybrid-direct+sampled_direct-teacher,axis=1)/np.maximum(norm,1e-30),mask)
        item[floor] = result
    item['single_fraction_median'] = float(np.median(np.asarray([r['direct_fraction'] for r in rows])[masks['above_1e-8']])) if masks['above_1e-8'].any() else None
    out['regions'][region] = item

for region, item in out['regions'].items():
    s=item['above_1e-8']
    print(region, s['budget_rgb16/total_lut']['count'])
    for k in ['budget_rgb16/total_lut','budget_rgb16/hybrid_signed','full/hybrid_signed','march_64_disk','march_32_disk','march_64_point']:
        v=s[k]
        if v['count']: print(f"  {k:30s} P95 {v['p95']:8.3f}%  max {v['maximum']:8.3f}%")
(a.probe.parent/'summary.json').write_text(json.dumps(out,indent=2))

if m['full_spectral_rec2020'] and 'budget_rgb16/hybrid_bf16' in out['regions']['all']['above_1e-8']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = ['noon_halo','blue_hour','moving_shadow','orbit_400km','upper_space']
    labels = ['Solar aureole','Blue hour','Moving shadow','Orbit 400 km','Upper / space']
    x = np.arange(len(chosen))
    fig, axes = plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for j,(name,key,color) in enumerate([
        ('Total-radiance bf16 / 13.18 MB main table','budget_rgb16/total_bf16','#78909c'),
        ('Single + remainder bf16 / same table','budget_rgb16/hybrid_bf16','#1976d2'),
        ('Previous joint TT / 15.70 MB package','joint_tt','#e69f00')]):
        y = [out['regions'][c]['above_1e-8'][key]['p95'] for c in chosen]
        axes[0].bar(x+(j-1)*.25,y,.25,label=name,color=color)
    axes[0].set_yscale('log');axes[0].set_ylim(.025,180)
    axes[0].set_ylabel('P95 relative RGB error (%)')
    axes[0].set_title('Storage experiment: high-quality direct integral, frozen v6 teacher')
    axes[0].legend(fontsize=9)
    for j,(name,key,color) in enumerate([
        ('Uniform 64 steps','march_64_disk_vs1024','#78909c'),
        ('Height warp 32 steps','march_32_disk_height_vs1024','#e69f00'),
        ('Height warp 64 steps','march_64_disk_height_vs1024','#1976d2')]):
        y=[out['regions'][c]['above_1e-8'][key]['p95'] for c in chosen]
        axes[1].bar(x+(j-1)*.25,y,.25,label=name,color=color)
    axes[1].set_yscale('log');axes[1].set_ylim(.02,20)
    axes[1].set_ylabel('P95 integration error / total sky (%)')
    axes[1].set_title('Marching experiment: finite Sun, difference from uniform 1024 steps')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('CPU feasibility: 768 selected queries, 41 spectral bands, f32',fontsize=14)
    fig.savefig(a.probe.parent/'comparison.png',dpi=160)
