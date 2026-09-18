"""Same-camera audit of a fresh 4D solve and angular-only full-radiance SH.
All rendering/SH fitting is f32. Percentiles are descriptive statistics.
"""
import json
from pathlib import Path
import numpy as np
from preview_wavelengths_cpu import display, heatmap, save_png

F=np.float32
ROOT=Path('out/hybrid_v1')
SH=Path('out/full_radiance_sh_v1_dense')
SELECTED='balanced_final'
def read(path): return np.fromfile(path,dtype=F).reshape(-1,4)[:,:3]
def error(a,b):
    norm=np.linalg.norm(b,axis=1);valid=norm>F(1e-8)
    relative=F(100)*np.linalg.norm(a-b,axis=1)/np.maximum(norm,F(1e-7))
    p=np.percentile(relative[valid],[50,95,99,100]) if valid.any() else [0]*4
    return dict(zip(('p50','p95','p99','max'),map(float,p)))|{'active':int(valid.sum()),'negative_pixels':int((a<0).any(1).sum())}
def main():
    scenes=json.loads(Path('out/four_wave_sun_v1/queries.json').read_text())['images']
    variants=[p.name for p in ROOT.iterdir() if (p/'solve.json').exists() and (p/'runs.json').exists() and (p/'noon_aureole_source.f32').exists()]
    shmeta=json.loads((SH/'audit.json').read_text())
    ds=Path('out/wavelength_search_dataset_v1');meta=json.loads((ds/'dataset.json').read_text())
    bands=np.fromfile(ds/'total.f32',dtype=F).reshape(meta['array_shape']).T
    weights=np.array(meta['rgb_from_integrated'],dtype=F)
    core={s['name']:bands[s['start']:s['end']]@weights for s in json.loads((ds/'queries.json').read_text())['images']}
    summaries={v:json.loads((ROOT/v/'solve.json').read_text()) for v in variants}
    rows=[];gallery=[];visual=ROOT/'visual';visual.mkdir(exist_ok=True)
    for scene in scenes:
        name=scene['name'];rgb_teacher=read(SH/f'{name}_rgb_teacher.f32');teacher=core.get(name,rgb_teacher)
        old=read(Path('out/four_wave_sun_v1/parallel_source')/f'{name}_128.f32')
        oldsky=read(Path('out/four_wave_sun_v1/parallel_sky')/f'{name}_128.f32')
        dense=read(ROOT/'dense_final'/f'{name}_source.f32')
        hybrid=read(ROOT/SELECTED/f'{name}_source.f32');hybridsky=read(ROOT/SELECTED/f'{name}_sky.f32')
        row={'scene':name,'rgb_grid_vs_spectral':error(rgb_teacher,teacher),'old_source_vs_teacher':error(old,teacher),
            'old_sky_vs_teacher':error(oldsky,teacher),'hybrid_source_vs_teacher':error(hybrid,teacher),
            'hybrid_sky_vs_teacher':error(hybridsky,teacher),'hybrid_vs_old_source':error(hybrid,old),
            'hybrid_sky_vs_source':error(hybridsky,hybrid),'hybrid_vs_dense':error(hybrid,dense),
            'dense40_vs_dense48':error(read(ROOT/'dense_check'/f'{name}_source.f32'),dense),
            'variants':{},'sh':{},'projection_convergence':{}}
        for v in variants:
            values=read(ROOT/v/f'{name}_source.f32')
            row['variants'][v]={'vs_dense':error(values,dense),'vs_old':error(values,old),'vs_teacher':error(values,teacher)}
        for degree in shmeta['degrees']:
            for mode in ['linear','log']:
                key=f'{mode}_{degree}';values=read(SH/f'{name}_{key}.f32')
                row['sh'][key]={'vs_local_teacher':error(values,rgb_teacher),'vs_spectral_teacher':error(values,teacher)}
                row['projection_convergence'][key]=error(values,read(Path('out/full_radiance_sh_v1')/f'{name}_{key}.f32'))
        rows.append(row)
        lum=teacher@np.array([.2627,.6780,.0593],dtype=F)
        exposure=F(.6)/max(np.quantile(lum[lum>F(1e-8)],F(.9)),F(1e-7)) if (lum>F(1e-8)).any() else F(1)
        norm=np.maximum(np.linalg.norm(dense,axis=1),F(1e-7))
        cards=[]
        for key,label,values,metric in [
            ('teacher','冻结光谱参考',teacher,None),
            ('old','旧四波长 + 固定 MS / SkyView',oldsky,row['old_sky_vs_teacher']),
            ('hybrid','新 4D 混合 / SkyView',hybridsky,row['hybrid_sky_vs_teacher']),
            ('sh17','全辐亮度 log-SH 17 阶',read(SH/f'{name}_log_17.f32'),row['sh']['log_17']['vs_spectral_teacher']),
            ('sh64','全辐亮度 log-SH 64 阶',read(SH/f'{name}_log_64.f32'),row['sh']['log_64']['vs_spectral_teacher']),
            ('linear17','全辐亮度线性 SH 17 阶',read(SH/f'{name}_linear_17.f32'),row['sh']['linear_17']['vs_spectral_teacher']),
        ]:
            filename=f'{name}_{key}.png';save_png(visual/filename,display(values,exposure),scene['width'],scene['height'])
            caption=label+(f" · P95 {metric['p95']:.2f}%" if metric else '')
            cards.append(f'<figure><figcaption>{caption}</figcaption><img src="{filename}"></figure>')
        filename=f'{name}_convergence.png'
        save_png(visual/filename,heatmap(np.linalg.norm(hybrid-dense,axis=1)/norm,5),scene['width'],scene['height'])
        gallery.append({'name':name,'description':f"海拔 {scene['altitude_km']} km · 太阳 {scene['sun_elevation_deg']:.3f}° · 新求解器对更密版本 P95 {row['hybrid_vs_dense']['p95']:.2f}% · SkyView 自身 P95 {row['hybrid_sky_vs_source']['p95']:.2f}%",'cards':''.join(cards),'convergence':filename})
    summary={'selected':SELECTED,'metric':'100 * length(RGB - reference) / max(length(reference), 1e-7); active reference length > 1e-8; linear Rec.2020; no visible solar disk',
        'solves':summaries,'sh_metadata':shmeta,'rows':rows,'limitations':[
            'First nine scenes use bandwise 41-band teacher; additional scenes use RGB-grid teacher. Difference quantified on core scenes.',
            'SH independently projects each observer state, without coefficient quantization or inter-state interpolation. Budget is an estimate, not a shipping resource.',
            'Hybrid versus frozen teacher includes spectral, transport, mapping and illumination changes; not a pure compression metric.',
            'Dense solver angular quadrature is not fully converged at the dark orbital shadow; do not interpret it as unbiased truth.',
            'Static views and cache tests do not establish temporal stability across every altitude/time.']}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page='''<!doctype html><meta charset="utf-8"><title>4D 混合求解与全辐亮度球谐</title>
<style>body{margin:24px;background:#101722;color:#ddd;font:16px system-ui;line-height:1.65}h1{font-size:26px}p{max-width:1150px}select{font:inherit;background:#243247;color:#fff;padding:8px}main{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:18px}figure{margin:0}figcaption{font-size:14px}img{width:100%;image-rendering:auto}aside img{max-width:600px}a{color:#8acaff}.note{color:#becadd}@media(max-width:850px){main{grid-template-columns:1fr 1fr}}</style>
<h1>4D 混合求解与全辐亮度球谐</h1>
<p>混合方案：四波长实时单散射 + GPU 启动求解的 4D 多重散射 + SkyView。常驻含 SkyView 10.07 MiB，启动峰值载荷约 1 GiB。球谐完整保留单散射；17 阶系数估算 15.28 MiB，64 阶约 189.68 MiB。所有图使用相同曝光。</p>
<p class="note">SH 图是逐状态投影的乐观精度测试，尚未计入量化和高度／太阳插值。前 9 个场景使用逐光谱查询的教师，其余使用 RGB 网格教师。高空对参考的差异包含原参考输运偏差。太阳盘不参与比较。64 阶仍不能充分保存太阳前向峰。</p>
<select id="scene"></select><p id="description"></p><main id="cards"></main><aside><h3>新混合求解与更密版本的差异</h3><p>黑色 0%，黄色 ≥5%；仅用于采样收敛检查。更密版本的太空地影也尚未完全收敛。</p><img id="heat"></aside>
<script>const data=DATA;const select=document.getElementById('scene');data.forEach((s,i)=>select.add(new Option(s.name,i)));function show(){const s=data[select.value];document.getElementById('description').textContent=s.description;document.getElementById('cards').innerHTML=s.cards;document.getElementById('heat').src=s.convergence;}select.onchange=show;select.value='0';show();</script>'''
    (visual/'index.html').write_text(page.replace('DATA',json.dumps(gallery,ensure_ascii=False)),encoding='utf-8')
    for row in rows[:9]:
        print(row['scene'],'old/new/SH17 P95:',*[round(row[k]['p95'],3) for k in ['old_source_vs_teacher','hybrid_source_vs_teacher']],round(row['sh']['log_17']['vs_spectral_teacher']['p95'],3),'convergence',round(row['hybrid_vs_dense']['p95'],3))
if __name__=='__main__':main()
