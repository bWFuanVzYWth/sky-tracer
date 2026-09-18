"""Isolate the errors introduced by the immutable source atlas and SkyView."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from preview_wavelengths_cpu import display,heatmap,save_png

F=np.float32
base=Path('out/four_wave_validation_v1')
out=Path('out/four_wave_cache_v1')
dataset=Path('out/wavelength_search_dataset_v1')
plan=json.loads((dataset/'queries.json').read_text())
folder=out/'visual';folder.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
rows=[];sections=[]
for scene in plan['images']:
    name=scene['name'];w=scene['width'];h=scene['height']
    reference=np.fromfile(base/f'{name}_128.f32',dtype=F).reshape(-1,4)[:,:3]
    source=np.fromfile(out/'source'/f'{name}_128.f32',dtype=F).reshape(-1,4)[:,:3]
    sky=np.fromfile(out/'sky'/f'{name}_128.f32',dtype=F).reshape(-1,4)[:,:3]
    norm=np.linalg.norm(reference,axis=1);mask=norm>F(1e-8);den=np.maximum(norm,F(1e-7))
    def error(x,y):
        values=(F(100)*np.linalg.norm(x-y,axis=1)/den)[mask]
        return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(values,[50,95,99,100]))))
    row=dict(scene=name,source_vs_sh=error(source,reference),sky_vs_source=error(sky,source),combined_vs_sh=error(sky,reference),finite=bool(np.isfinite(sky).all()))
    assert row['finite'];rows.append(row)
    lum=reference@np.array([.2627,.6780,.0593],dtype=F);exposure=F(.6)/max(np.quantile(lum[lum>F(1e-8)],F(.9)),F(1e-7))
    sheet=Image.new('RGB',(1536,240),(20,24,30));draw=ImageDraw.Draw(sheet)
    for i,(key,title,values) in enumerate([
        ('old','Previous per-pixel SH / 128 steps',display(reference,exposure)),
        ('source','Fixed directional MS atlas',display(source,exposure)),
        ('sky','Fixed MS + observer SkyView',display(sky,exposure)),
        ('error','Added error: black 0%, yellow 3%',heatmap(np.linalg.norm(sky-reference,axis=1)/den,3)),
    ]):
        save_png(folder/f'{name}_{key}.png',values,w,h);sheet.paste(Image.open(folder/f'{name}_{key}.png').resize((384,192)),(i*384,48));draw.text((i*384+4,6),title,font=font,fill='white')
    draw.text((772,27),f"added P95 {row['combined_vs_sh']['p95']:.3f}%",font=font,fill='#bbc3cc')
    sheet.save(folder/f'{name}_sheet.png')
    sections.append(f'<section><h2>{name}</h2><p>缓存新增误差 P95 {row["combined_vs_sh"]["p95"]:.3f}%；P99 {row["combined_vs_sh"]["p99"]:.3f}%</p><img src="{name}_sheet.png"></section>')
    print(name, 'source P95',round(row['source_vs_sh']['p95'],3),'SkyView P95',round(row['sky_vs_source']['p95'],3),'combined',round(row['combined_vs_sh']['p95'],3),'max',round(row['combined_vs_sh']['max'],3))
(out/'quality.json').write_text(json.dumps(dict(rows=rows,metric='Compared against previous 128-step four-wave SH integrator, relative Rec2020 norm; active >1e-8, floor 1e-7'),indent=2))

sweep_plan=out/'sweep_queries.json'
if sweep_plan.exists():
    sweeps=[];previous={}
    for scene in json.loads(sweep_plan.read_text())['images']:
        name=scene['name'];group=name.split('_')[0]
        reference=np.fromfile(out/'sweep_reference'/f'{name}_128.f32',dtype=F).reshape(-1,4)[:,:3]
        sky=np.fromfile(out/'sweep_cached'/f'{name}_128.f32',dtype=F).reshape(-1,4)[:,:3]
        norm=np.linalg.norm(reference,axis=1);mask=norm>F(1e-8);den=np.maximum(norm,F(1e-7))
        residual=sky-reference
        row=dict(scene=name,error=error(sky,reference))
        if group in previous:
            row['residual_change_p95']=float(np.percentile((F(100)*np.linalg.norm(residual-previous[group],axis=1)/den)[mask],95))
        previous[group]=residual;sweeps.append(row)
        print(name,'P95',round(row['error']['p95'],3),'max',round(row['error']['max'],3),'residual change',round(row.get('residual_change_p95',0),3))
    (out/'sweeps.json').write_text(json.dumps(sweeps,indent=2))
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Four-wave cache validation</title><style>body{background:#10151d;color:#ddd;font:16px system-ui;margin:24px}img{width:100%;max-width:1536px}section{margin:28px 0}</style><h1>固定多重散射 + SkyView</h1><p>这里测的是缓存相对上一版逐像素积分新增的误差，排除了冻结参考自身的偏差。多重散射源仅加载时展开一次；相机旋转只重投影；太阳高度/观察海拔改变时更新 SkyView。每个场景使用同一曝光。</p>'+''.join(sections),encoding='utf-8')
