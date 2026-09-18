"""Image error budget for fitted coordinates, equal-budget and one-axis ablations."""
import json
from pathlib import Path
import numpy as np
from summarize_hybrid_sh import read,error
from preview_wavelengths_cpu import display,heatmap,save_png

F=np.float32
ROOT=Path('out/hybrid_mapping_v3')
def main():
    cases=json.loads(Path('out/four_wave_sun_v1/queries.json').read_text())['images']
    variants=[p.name for p in ROOT.iterdir() if (p/'runs.json').exists() and not p.name.startswith('sweeps_')]
    ds=Path('out/wavelength_search_dataset_v1');meta=json.loads((ds/'dataset.json').read_text())
    bands=np.fromfile(ds/'total.f32',dtype=F).reshape(meta['array_shape']).T
    weights=np.array(meta['rgb_from_integrated'],dtype=F)
    core={s['name']:bands[s['start']:s['end']]@weights for s in json.loads((ds/'queries.json').read_text())['images']}
    rows=[];gallery=[];visual=ROOT/'visual';visual.mkdir(exist_ok=True)
    for s in cases:
        n=s['name'];source=read(ROOT/'selected_final'/f'{n}_source.f32');sky=read(ROOT/'selected_final'/f'{n}_sky.f32')
        legacy=read(ROOT/'legacy'/f'{n}_source.f32');legacy_sky=read(ROOT/'legacy'/f'{n}_sky.f32')
        equal=read(ROOT/'fitted_initial'/f'{n}_source.f32');dense=read(ROOT/'dense'/f'{n}_source.f32')
        denser=read(ROOT/'dense_check'/f'{n}_source.f32');grid=read(ROOT/'dense_grid'/f'{n}_source.f32')
        teacher=core.get(n)
        if teacher is None:teacher=read(Path('out/full_radiance_sh_v1_dense')/f'{n}_rgb_teacher.f32')
        row=dict(scene=n,selected_sky_error=error(sky,source),legacy_sky_error=error(legacy_sky,legacy),
                 legacy_vs_dense_grid=error(legacy,grid),equal_vs_dense_grid=error(equal,grid),selected_vs_dense_grid=error(source,grid),
                 selected_vs_dense=error(source,dense),selected_vs_denser=error(source,denser),dense_convergence=error(dense,denser),
                 selected_vs_teacher=error(sky,teacher),legacy_vs_teacher=error(legacy_sky,teacher),
                 variants={v:dict(change=error(read(ROOT/v/f'{n}_source.f32'),equal),change_vs_selected=error(read(ROOT/v/f'{n}_source.f32'),source),vs_dense=error(read(ROOT/v/f'{n}_source.f32'),dense),vs_grid=error(read(ROOT/v/f'{n}_source.f32'),grid)) for v in variants})
        rows.append(row)
        lum=teacher@np.array([.2627,.6780,.0593],dtype=F);active=lum>F(1e-8)
        exposure=F(.6)/max(np.quantile(lum[active],.9),F(1e-7)) if active.any() else F(1)
        cards=[]
        for key,label,val in [('teacher','冻结参考（含光谱／输运差异）',teacher),('legacy','旧映射 / 8 cone',legacy_sky),('selected','新映射 / 12 cone',sky),('direct','新映射直接积分（无 SkyView）',source),('dense','加密诊断解（仍有积分误差）',denser)]:
            file=f'{n}_{key}.png';save_png(visual/file,display(val,exposure),s['width'],s['height'])
            cards.append(f'<figure><figcaption>{label}</figcaption><img src="{file}"></figure>')
        file=f'{n}_cache_error.png';err=np.linalg.norm(sky-source,axis=1)/np.maximum(np.linalg.norm(source,axis=1),F(1e-7))
        save_png(visual/file,heatmap(err,1),s['width'],s['height']);cards.append(f'<figure><figcaption>新 SkyView 误差：亮黄 ≥1%</figcaption><img src="{file}"></figure>')
        gallery.append(dict(name=n,description=f"海拔 {s['altitude_km']} km，太阳 {s['sun_elevation_deg']:.3f}°。SkyView P95 {row['legacy_sky_error']['p95']:.3f}% → {row['selected_sky_error']['p95']:.3f}%；对加密网格（同积分）{row['selected_vs_dense_grid']['p95']:.3f}%；加密解自身收敛差 {row['dense_convergence']['p95']:.3f}%。",cards=''.join(cards)))
    qroot=ROOT/'optical_validation';q=np.fromfile(qroot/'optical_queries.f32',dtype=F).reshape(-1,10);truth=np.exp(-q[:,6:]);valid=truth>F(1e-4)
    optical={}
    for p in qroot.glob('*_f16.f32'):
        t=np.exp(-np.fromfile(p,dtype=F).reshape(-1,4));e=100*abs(t-truth)/np.maximum(truth,F(1e-4))
        optical[p.stem]=dict(p95=float(np.percentile(e[valid],95)),p99=float(np.percentile(e[valid],99)),max=float(e[valid].max()),absolute_max=float(abs(t-truth).max()))
    sweeps=[]
    for s in json.loads((ROOT/'sweeps.json').read_text())['images']:
        n=s['name'];selected=read(ROOT/'sweeps_final'/f'{n}_source.f32');sky=read(ROOT/'sweeps_final'/f'{n}_sky.f32');dense=read(ROOT/'sweeps_dense'/f'{n}_source.f32')
        legacy=read(ROOT/'sweeps_legacy'/f'{n}_source.f32');oldsky=read(ROOT/'sweeps_legacy'/f'{n}_sky.f32')
        sweeps.append(s|dict(cache=error(sky,selected),old_cache=error(oldsky,legacy),selected_vs_dense=error(selected,dense),legacy_vs_dense=error(legacy,dense)))
    summary=dict(selected='selected_final',metric='RGB relative vector norm (%), floor 1e-7, active teacher norm >1e-8; linear Rec.2020, excludes visible disk',
                 solves={v:json.loads((ROOT/v/'solve.json').read_text()) for v in variants},rows=rows,optical=optical,sweeps=sweeps,
                 limits=['Dense quadrature is still unconverged in orbital shadow; differences are convergence probes, not ground-truth errors.',
                         'Frozen full reference is spectrally/physically different; full teacher differences cannot be attributed only to LUT allocation.',
                         'Fit and evaluation use the calibrated medium, gray albedo .18; changed atmospheres are functional tests, not a universal quality guarantee.',
                         '222 additional views test selected times/heights, not every possible trajectory.'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page='''<!doctype html><meta charset="utf-8"><title>拟合映射与精度瓶颈</title><style>body{background:#111925;color:#dae2ed;font:16px system-ui;margin:24px;line-height:1.6}h1{font-size:25px}select{font:inherit;padding:8px;background:#23354a;color:white}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{width:100%}figcaption{font-size:14px}p{max-width:1150px}@media(max-width:900px){main{grid-template-columns:repeat(2,1fr)}}</style><h1>拟合映射已接入：48 × 176 × 24 × 12</h1><p>四波长；13.00 MiB 常驻（含 SkyView）；默认 256² SkyView。旧映射与新映射等预算比较先固定 8 cone，再将剩余预算用于 12 cone。所有图共用曝光。黑暗区域的相对误差需结合亮度判断；诊断加密解并非真值。</p><select id="select"></select><p id="description"></p><main id="cards"></main><script>const data=DATA;let select=document.getElementById('select');data.forEach((s,i)=>select.add(new Option(s.name,i)));function show(){let s=data[select.value];document.getElementById('description').textContent=s.description;document.getElementById('cards').innerHTML=s.cards;}select.onchange=show;select.value='0';show();</script>'''
    (visual/'index.html').write_text(page.replace('DATA',json.dumps(gallery,ensure_ascii=False)),encoding='utf-8')
    for key in ['selected_sky_error','selected_vs_dense_grid','dense_convergence']:
        print(key,sorted([(r['scene'],round(r[key]['p95'],3)) for r in rows],key=lambda v:v[1],reverse=True)[:7])
    print('sweeps cache',sorted([(r['name'],round(r['cache']['p95'],3),round(r['old_cache']['p95'],3)) for r in sweeps],key=lambda v:v[1],reverse=True)[:12])
if __name__=='__main__':main()
