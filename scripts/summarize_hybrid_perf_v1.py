"""Separate equivalent optimizations from sampling changes and convergence."""
import json
from pathlib import Path
import numpy as np
from summarize_hybrid_sh import read as read_rgb,error
from preview_wavelengths_cpu import display,heatmap,save_png

ROOT=Path('out/hybrid_perf_v1');OLD=Path('out/hybrid_mapping_v3')

def read(path):
    value=read_rgb(path)
    if not np.isfinite(value).all():raise ValueError(f'Non-finite RGB: {path}')
    return value
def compare(a,b,queries):
    return [dict(scene=s['name'],**error(read(a/(s['name']+'_source.f32')),read(b/(s['name']+'_source.f32')))) for s in queries]
def top(rows,n=4):return sorted(rows,key=lambda x:x['p95'],reverse=True)[:n]
def main():
    scenes=json.loads(Path('out/four_wave_sun_v1/queries.json').read_text())['images']
    dataset=Path('out/wavelength_search_dataset_v1')
    meta=json.loads((dataset/'dataset.json').read_text())
    bands=np.fromfile(dataset/'total.f32',dtype=np.float32).reshape(meta['array_shape']).T
    weights=np.array(meta['rgb_from_integrated'],dtype=np.float32)
    core={s['name']:bands[s['start']:s['end']]@weights for s in json.loads((dataset/'queries.json').read_text())['images']}
    teachers={s['name']:core[s['name']] if s['name'] in core else read(Path('out/full_radiance_sh_v1_dense')/(s['name']+'_rgb_teacher.f32')) for s in scenes}
    variants=[p.name for p in ROOT.iterdir() if (p/'solve.json').exists() and (p/'runs.json').exists() and not p.name.startswith('holdout_') and p.name!='sweeps']
    summary=dict(solves={v:json.loads((ROOT/v/'solve.json').read_text()) for v in variants},
        equivalent_change=compare(ROOT/'both_v2',OLD/'selected_final',scenes),
        stencil_change=compare(ROOT/'both_v2',ROOT/'both',scenes),
        cache_change=compare(ROOT/'selected_v2',ROOT/'selected_nocache',scenes),
        importance={v:compare(ROOT/v,ROOT/'q64_high_sun0',scenes) for v in ['both_v2','selected_nocache','q48_high_sun0','q64_p3','q64_p2','q64_p4','horizon1','horizon2','horizon4','sun_narrow','sun_wide']})
    holdout=json.loads((ROOT/'shadow_holdout.json').read_text())['images']
    summary['holdout']={v:compare(ROOT/('holdout_'+v),ROOT/'holdout_dense',holdout) for v in ['control','selected','dense48','dense_old']}
    summary['path_sensitivity']={v:compare(ROOT/v,OLD/'dense_check',scenes) for v in ['both','path_uniform','path_narrow','path_wide']}
    summary['frozen_reference']=[dict(scene=s['name'],kind='full spectral' if s['name'] in core else 'RGB grid',**error(read(ROOT/'final'/(s['name']+'_sky.f32')),teachers[s['name']])) for s in scenes]
    sweep=json.loads((OLD/'sweeps.json').read_text())['images']
    summary['sky_core']=[dict(scene=s['name'],**error(read(ROOT/'final'/(s['name']+'_sky.f32')),read(ROOT/'final'/(s['name']+'_source.f32')))) for s in scenes]
    summary['sky_sweeps']=[dict(scene=s['name'],**error(read(ROOT/'sweeps'/(s['name']+'_sky.f32')),read(ROOT/'sweeps'/(s['name']+'_source.f32')))) for s in sweep]
    summary['metric']='100*RGB norm difference / max(reference norm,1e-7); reference norm >1e-8; linear Rec.2020; no disk'
    summary['limits']=['Angularly dense comparison is not unbiased truth; cross-partition convergence is reported.',
        'Importance fits/choices are calibrated to this fixed atmosphere; physical changes need quality validation.',
        'Solve timings are individual measurements; GPU clock, cache and driver variability remain.']
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    visual=ROOT/'visual';visual.mkdir(exist_ok=True);gallery=[]
    for s in scenes:
        n=s['name'];teacher=teachers[n]
        baseline=read(OLD/'selected_final'/f'{n}_sky.f32');final=read(ROOT/'final'/f'{n}_sky.f32');direct=read(ROOT/'final'/f'{n}_source.f32');dense=read(ROOT/'q64_high_sun0'/f'{n}_source.f32')
        lum=teacher@np.array([.2627,.6780,.0593],dtype='f4');good=lum>1e-8;exposure=np.float32(.6/max(np.quantile(lum[good],.9),1e-7)) if good.any() else 1
        cards=[]
        for tag,label,v in [('frozen','冻结参考（含光谱与输运差异）',teacher),('before','已拟合／性能优化前',baseline),('final','当前默认／约 1.6 秒启动',final),('direct','当前直接积分',direct),('dense','64×64 角积分诊断',dense)]:
            f=f'{n}_{tag}.png';save_png(visual/f,display(v,exposure),s['width'],s['height']);cards.append(f'<figure><figcaption>{label}</figcaption><img src="{f}"></figure>')
        f=f'{n}_error.png';e=np.linalg.norm(direct-dense,axis=1)/np.maximum(np.linalg.norm(dense,axis=1),1e-7)
        save_png(visual/f,heatmap(e,1),s['width'],s['height']);cards.append(f'<figure><figcaption>角积分收敛差：黄 ≥1%，不是物理真值误差</figcaption><img src="{f}"></figure>')
        a=next(r for r in summary['importance']['both_v2'] if r['scene']==n)['p95'];b=next(r for r in summary['importance']['selected_nocache'] if r['scene']==n)['p95']
        gallery.append(dict(name=n,description=f"海拔 {s['altitude_km']} km · 太阳 {s['sun_elevation_deg']:.3f}° · 对同一加密角积分的 P95：{a:.3f}% → {b:.3f}%",cards=''.join(cards)))
    html='''<!doctype html><meta charset="utf-8"><title>大气 LUT：重要性求积与去重</title><style>body{background:#111925;color:#dae2ed;font:16px system-ui;margin:24px;line-height:1.6}h1{font-size:25px}select{font:inherit;padding:8px;background:#23354a;color:white}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{width:100%}figcaption{font-size:14px}p{max-width:1150px}@media(max-width:900px){main{grid-template-columns:repeat(2,1fr)}}</style><h1>重要性求积、重复计算与常量优化</h1><p>默认 48×176×24×12；四波长；13.00 MiB 常驻；约 1.6 秒启动。复用相位坐标，预计算固定系数，高空向地平线分配积分。单散射缓存可选，默认关闭，以节省约 495 MiB 临时显存。所有图片共享曝光，诊断加密解不代表无偏真值。</p><select id="s"></select><p id="d"></p><main id="c"></main><script>const data=DATA;let s=document.getElementById('s');data.forEach((v,i)=>s.add(new Option(v.name,i)));function show(){let v=data[s.value];document.getElementById('d').textContent=v.description;document.getElementById('c').innerHTML=v.cards;}s.onchange=show;s.value='0';show();</script>'''
    (visual/'index.html').write_text(html.replace('DATA',json.dumps(gallery,ensure_ascii=False)),encoding='utf-8')
    print('equivalent',top(summary['equivalent_change'],2))
    print('importance before',top(summary['importance']['both_v2'],3));print('importance after',top(summary['importance']['selected_nocache'],3))
    for v in summary['holdout']:print('holdout',v,top(summary['holdout'][v],2))
    print('sky',top(summary['sky_core'],2),top(summary['sky_sweeps'],2))
    print('full spectral reference',[(r['scene'],round(r['p95'],3)) for r in summary['frozen_reference'] if r['kind']=='full spectral'])
if __name__=='__main__':main()
