"""Performance-first preset audit, with separate grazing stress and interpolation."""
import json
from pathlib import Path
import numpy as np
from summarize_hybrid_budget_v1 import read,error
from preview_wavelengths_cpu import display,heatmap,save_png
ROOT=Path('out/hybrid_budget_v1')

def audit(folder,old,dense,queries):
    rows=[]
    for s in queries:
        n=s['name'];a=read(folder/(n+'_source.f32'));sky=read(folder/(n+'_sky.f32'))
        previous=read(old/(n+'_source.f32'));truth=read(dense/(n+'_source.f32'))
        def gradient(v):
            v=(v-truth).reshape(s['height'],s['width'],3)
            t=truth.reshape(v.shape);norm=np.linalg.norm(t,axis=2)
            valid=(norm[:,:-1]>1e-7)&(norm[:,1:]>1e-7)
            e=100*np.linalg.norm(np.diff(v,axis=1),axis=2)/np.maximum((norm[:,:-1]+norm[:,1:])*.5,1e-7)
            return float(np.percentile(e[valid],99)) if valid.any() else 0.0
        rows.append(dict(scene=n,new_vs_old=error(a,previous),old_vs_dense=error(previous,truth),new_vs_dense=error(a,truth),sky=error(sky,a),
            gradient_p99=gradient(sky),old_gradient_p99=gradient(read(old/(n+'_sky.f32')))))
    return rows

def main():
    queries=json.loads((ROOT/'all_queries.json').read_text())['images']
    grazing=json.loads((ROOT/'grazing_queries.json').read_text())['images']
    output=dict(
        all=audit(ROOT/'production',ROOT/'baseline_all',ROOT/'dense_all',queries),
        grazing=audit(ROOT/'production',ROOT/'grazing_1_baseline',ROOT/'grazing_1_dense',grazing),
        aerosol4=audit(ROOT/'grazing_4_production',ROOT/'grazing_4_baseline',ROOT/'grazing_4_dense',grazing),
        solve=json.loads((ROOT/'production/solve.json').read_text()),
        metric='Linear Rec.2020 vector relative difference, percent; norm floor 1e-7 and reference active >1e-8. Dense solves are convergence probes, not truth.',
    )
    timing=json.loads((ROOT/'production_timings.json').read_text())
    output['timings']={n:dict(zip(['median','min','max'],map(float,[np.median(v),min(v),max(v)]))) for n in ['old','production'] if (v:=[r['solve']['wall_seconds'] for r in timing if r['variant']==n])}
    ds=Path('out/wavelength_search_dataset_v1');meta=json.loads((ds/'dataset.json').read_text())
    bands=np.fromfile(ds/'total.f32',dtype=np.float32).reshape(meta['array_shape']).T;weights=np.array(meta['rgb_from_integrated'],dtype=np.float32)
    output['frozen']=[dict(scene=s['name'],**error(read(ROOT/'production'/(s['name']+'_sky.f32')),bands[s['start']:s['end']]@weights)) for s in json.loads((ds/'queries.json').read_text())['images']]
    (ROOT/'production_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    visual=ROOT/'visual';visual.mkdir(exist_ok=True);gallery=[]
    core=json.loads(Path('out/four_wave_sun_v1/queries.json').read_text())['images']
    chosen=core+[s for s in queries if s['name'].startswith('h1400_')]+[s for s in grazing if s['altitude_km'] in [.002,1.] and (s['sun_elevation_deg'] in [-8,-.1,47] or 'close' in s['name'])]
    for s in chosen:
        n=s['name'];is_grazing=n.startswith('g_');old=ROOT/('grazing_1_baseline' if is_grazing else 'baseline_all');dense=ROOT/('grazing_1_dense' if is_grazing else 'dense_all')
        truth=read(dense/(n+'_source.f32'));a=read(ROOT/'production'/(n+'_sky.f32'));b=read(old/(n+'_sky.f32'))
        lum=truth@np.array([.2627,.6780,.0593],dtype=np.float32);active=lum>1e-8;ev=np.float32(.6/max(np.quantile(lum[active],.9),1e-7)) if active.any() else 1
        cards=[]
        for tag,label,v in [('old','之前：13 MiB / 1.617 秒',b),('new','当前：7.91 MiB / 0.393 秒',a),('dense','加密诊断解（仍有偏）',truth)]:
            f=f'{n}_{tag}.png';save_png(visual/f,display(v,ev),s['width'],s['height']);cards.append(f'<figure><figcaption>{label}</figcaption><img src="{f}"></figure>')
        for tag,label,v in [('old_error','旧方案对加密解；黄 ≥3%',b),('new_error','当前对加密解；黄 ≥3%',a)]:
            f=f'{n}_{tag}.png';e=np.linalg.norm(v-truth,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-7);save_png(visual/f,heatmap(e,3),s['width'],s['height']);cards.append(f'<figure><figcaption>{label}</figcaption><img src="{f}"></figure>')
        row=next(r for r in output['grazing' if is_grazing else 'all'] if r['scene']==n)
        gallery.append(dict(name=n,description=f"海拔 {s['altitude_km']} km · 太阳 {s['sun_elevation_deg']:.3f}° · 直接积分对加密解 P95：{row['old_vs_dense']['p95']:.3f}% → {row['new_vs_dense']['p95']:.3f}% · 当前 SkyView 插值 P95：{row['sky']['p95']:.3f}%",cards=''.join(cards)))
    page='''<!doctype html><meta charset="utf-8"><title>大气：性能优先与简化审计</title><style>body{background:#111925;color:#dae2ed;font:16px system-ui;margin:24px;line-height:1.6}h1{font-size:25px}select{font:inherit;padding:8px;background:#23354a;color:white;max-width:100%}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{width:100%}figcaption{font-size:14px}p{max-width:1150px}@media(max-width:900px){main{grid-template-columns:repeat(2,1fr)}}</style><h1>性能优先：40 × 160 × 20 × 12</h1><p>四波长、7.91 MiB 常驻、32×16 统一角求积、64 步启动积分、96 步 SkyView、8 阶迭代。移除了新增高空高斯规则、分层切步与单散射缓存。以轻微质量变化换取性能；全高度采用同一套求积规则。图片共享曝光，热图显示与加密诊断的差异，不代表物理真值误差。</p><select id="s"></select><p id="d"></p><main id="c"></main><script>const data=DATA;let s=document.getElementById('s');data.forEach((v,i)=>s.add(new Option(v.name,i)));function show(){let v=data[s.value];document.getElementById('d').textContent=v.description;document.getElementById('c').innerHTML=v.cards;}s.onchange=show;s.value='0';show();</script>'''
    (visual/'index.html').write_text(page.replace('DATA',json.dumps(gallery,ensure_ascii=False)),encoding='utf-8')
    print('timings',output['timings'])
    for group in ['all','grazing','aerosol4']:
        print(group)
        for key in ['old_vs_dense','new_vs_dense','new_vs_old','sky']:
            rows=output[group];v=max(rows,key=lambda a:a[key]['p95']);print(key,v['scene'],v[key])
        print('gradient P99',max(r['old_gradient_p99'] for r in rows),max(r['gradient_p99'] for r in rows))
    print('frozen',[(r['scene'],round(r['p95'],3)) for r in output['frozen']])
if __name__=='__main__':main()
