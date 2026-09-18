"""Frozen spectral-reference quality and matched demo timings for UE8/hybrid."""
import json
from pathlib import Path
import numpy as np
from preview_wavelengths_cpu import display, heatmap, save_png

ROOT = Path('out/realtime_comparison_v1')
DATA = Path('out/wavelength_search_dataset_v1')
meta = json.loads((DATA / 'dataset.json').read_text())
scenes = json.loads((DATA / 'queries.json').read_text())['images']
spectra = np.fromfile(DATA / 'total.f32', np.float32).reshape(meta['array_shape']).T
weights = np.asarray(meta['rgb_from_integrated'], np.float32)
visual = ROOT / 'visual'
visual.mkdir(exist_ok=True)
rows, gallery = [], []


def metrics(values, reference, mask):
    norm = np.linalg.norm(reference, axis=1)
    active = mask & (norm > 1e-8)
    diff = values - reference
    relative = 100 * np.linalg.norm(diff, axis=1) / np.maximum(norm, 1e-7)
    lum = np.array([.2627, .6780, .0593])
    return dict(p95=float(np.percentile(relative[active], 95)),
                relative_l2=100 * float(np.linalg.norm(diff[active]) / np.linalg.norm(reference[active])),
                mean_luminance_bias=100 * float(np.mean(values[active] @ lum) / np.mean(reference[active] @ lum) - 1))


for s in scenes:
    name, w, h = s['name'], s['width'], s['height']
    reference = spectra[s['start']:s['end']] @ weights
    images = {}
    for key, folder in [('ue', ROOT / 'ue'), ('ue_top120', ROOT / 'ue_top120'),
                        ('hybrid', Path('out/hybrid_budget_v1/production'))]:
        images[key] = np.fromfile(folder / f'{name}_sky.f32', np.float32).reshape(-1, 4)[:, :3]
        assert images[key].shape == reference.shape and np.isfinite(images[key]).all()
    y, x = np.mgrid[:h, :w]
    yaw, pitch = np.deg2rad([s['yaw'], s['pitch']])
    forward = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    up = np.cross(forward, right)
    scale = np.tan(np.deg2rad(s['horizontal_fov']) / 2)
    rays = forward + ((2*(x+.5)/w-1)*scale)[..., None]*right + ((1-2*(y+.5)/h)*scale*h/w)[..., None]*up
    rays /= np.linalg.norm(rays, axis=2)[..., None]
    altitude = s['altitude_km']
    horizon = -np.sqrt(altitude*(2*6360+altitude))/(6360+altitude)
    masks = {'all': np.ones(w*h, bool), 'sky': (rays[..., 1] > horizon).ravel()}
    row = dict(scene=name, altitude_km=altitude, sun_elevation_deg=s['sun_elevation_deg'],
               outside_old_domain=altitude >= 100,
               metrics={k: {region: metrics(v, reference, mask) for region, mask in masks.items()} for k, v in images.items()})
    rows.append(row)
    lum = reference @ np.array([.2627, .6780, .0593])
    exposure = .6 / max(np.quantile(lum[lum > 1e-8], .9), 1e-7)
    cards = []
    for key, label, values in [('reference', '冻结 41 波段参考', reference),
                               ('ue', '旧 UE 8 波长', images['ue']),
                               ('hybrid', '当前 hybrid-4d', images['hybrid'])]:
        file = f'{name}_{key}.png'
        save_png(visual / file, display(values, exposure), w, h)
        cards.append(f'<figure><figcaption>{label}</figcaption><img src="{file}"></figure>')
    for key, label in [('ue', '旧 UE 对参考'), ('hybrid', '当前对参考')]:
        file = f'{name}_{key}_error.png'
        err = np.linalg.norm(images[key]-reference, axis=1)/np.maximum(np.linalg.norm(reference, axis=1), 1e-7)
        save_png(visual / file, heatmap(err, 10), w, h)
        cards.append(f'<figure><figcaption>{label}；黄色 ≥10%</figcaption><img src="{file}"></figure>')
    description = f"海拔 {altitude} km，太阳 {s['sun_elevation_deg']:.2f}°；天空像素 P95：UE {row['metrics']['ue']['sky']['p95']:.2f}% / hybrid {row['metrics']['hybrid']['sky']['p95']:.2f}%"
    if row['outside_old_domain']:
        description += '。超出旧方案范围：旧方案会钳制相机高度，这一项是能力差异，不能作为相同几何的误差比较。'
    gallery.append(dict(name=name, description=description, cards=''.join(cards)))

timing = json.loads((ROOT / 'timings.json').read_text())
perf = {}
for scene in ['noon', 'sunset', 'blue']:
    perf[scene] = {}
    for solver in ['unreal-8wave', 'hybrid-4d']:
        runs = [r for r in timing if r['scene'] == scene and r['solver'] == solver]
        perf[scene][solver] = {}
        for work in ['camera_change', 'sun_change', 'cached_frame']:
            times = [next(p for p in r['metadata']['performance'] if p['workload'] == work) for r in runs]
            med = [p['gpu_median_ms'] for p in times]
            perf[scene][solver][work] = dict(median=float(np.median(med)), minimum=min(med), maximum=max(med),
                                            worst_run_p95=max(p['gpu_p95_ms'] for p in times))
summary = dict(quality=rows, performance=perf,
               metric='Linear Rec.2020 vector difference; per-pixel relative P95 percent, floor 1e-7; reference norm >1e-8. Visible disk disabled in all quality images.',
               limitations=['Frozen LUT is biased, not physical truth.',
                            'Old default medium/phase/spectral/MS approximations differ; this compares complete implementations.',
                            'Old atmosphere is 100 km; top120 variant changes only shell top. Heights outside shell are clamped.',
                            'GPU demo timings use 512x384, 4 warmups + 64 measured frames, 3 alternating runs, include presentation.'])
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
page = '''<!doctype html><meta charset="utf-8"><title>旧 UE8 与当前 hybrid-4d</title><style>body{background:#111925;color:#dae2ed;font:16px system-ui;margin:24px;line-height:1.6}select{font:inherit;padding:8px;background:#23354a;color:white}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{width:100%}p{max-width:1100px}</style><h1>旧 UE8 与当前 hybrid-4d</h1><p>同一查询方向、线性 Rec.2020、无可见太阳盘；每场景统一曝光。误差参照冻结的完整光谱 LUT，它仍有偏。旧方案使用原默认物理近似和波长，当前使用四波长与各向异性多散射；差异不能归因于波长数一项。108/400 km 超出旧方案相机范围，单列为能力限制。</p><select id="s"></select><p id="d"></p><main id="c"></main><script>const data=DATA;let s=document.getElementById('s');data.forEach((v,i)=>s.add(new Option(v.name,i)));function show(){let v=data[s.value];document.getElementById('d').textContent=v.description;document.getElementById('c').innerHTML=v.cards;}s.onchange=show;s.value='0';show();</script>'''
(visual / 'index.html').write_text(page.replace('DATA', json.dumps(gallery, ensure_ascii=False)), encoding='utf-8')
for row in rows:
    print(row['scene'], {k: {region: round(v[region]['p95'], 3) for region in ['all', 'sky']} for k, v in row['metrics'].items()})
print(json.dumps(perf, indent=2))
