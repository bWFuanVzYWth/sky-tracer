"""Solar-model errors; separate direct transport from SkyView interpolation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from preview_wavelengths_cpu import display, heatmap, save_png

F = np.float32
root = Path('out/four_wave_sun_v1')
scenes = json.loads((root/'queries.json').read_text())['images']
radius = F(json.loads(Path('out/four_wave_source_v1/resource.json').read_text())['sun_radius'])
models = ['parallel', 'phase-averaged', 'finite4', 'finite16']

def read(scene, model, mode, steps=128):
    return np.fromfile(root/f'{model}_{mode}'/f'{scene["name"]}_{steps}.f32', dtype=F).reshape(-1, 4)[:, :3]

def outside_disk(scene):
    w, h = scene['width'], scene['height']
    x, y = np.meshgrid((np.arange(w, dtype=F)+F(.5))/F(w)*F(2)-F(1),
                       (np.arange(h, dtype=F)+F(.5))/F(h)*F(2)-F(1))
    pitch, yaw = np.deg2rad(F(scene['pitch'])), np.deg2rad(F(scene['yaw']))
    forward = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], dtype=F)
    right = np.array([np.cos(yaw), 0, -np.sin(yaw)], dtype=F)
    up = np.cross(forward, right)
    t = np.tan(np.deg2rad(F(scene['horizontal_fov']))*F(.5))
    ray = forward+x[:, :, None]*t*right-y[:, :, None]*t*F(h/w)*up
    ray /= np.linalg.norm(ray, axis=2)[:, :, None]
    elevation = np.deg2rad(F(scene['sun_elevation_deg']))
    sun = np.array([0, np.sin(elevation), np.cos(elevation)], dtype=F)
    return (np.sum((ray-sun)**2, axis=2) > F(4)*np.sin(radius/F(2))**2).ravel()

def errors(value, reference, outside=None):
    assert np.isfinite(value).all() and np.isfinite(reference).all()
    norm = np.linalg.norm(reference, axis=1)
    active = norm > F(1e-8)
    if outside is not None:
        active &= outside
    e = F(100)*np.linalg.norm(value-reference, axis=1)/np.maximum(norm, F(1e-7))
    percentiles = np.percentile(e[active], [50, 95, 99, 100]) if active.any() else np.zeros(4)
    result = dict(zip(['p50', 'p95', 'p99', 'max'], map(float, percentiles)))
    result['active_pixels'] = int(active.sum())
    result['relative_rms_percent'] = float(F(100)*np.linalg.norm((value-reference)[active])/max(np.linalg.norm(reference[active]), F(1e-30)))
    return result

folder = root/'visual'
folder.mkdir(exist_ok=True)
font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 15)
rows, sections = [], []
for scene in scenes:
    name = scene['name']
    ref = read(scene, 'finite64', 'source')
    ref_sky = read(scene, 'finite64', 'sky')
    outside = outside_disk(scene)
    row = dict(scene=name, source={}, sky={})
    for model in models:
        row['source'][model] = dict(all=errors(read(scene, model, 'source'), ref),
                                    outside_solar_disk=errors(read(scene, model, 'source'), ref, outside))
        row['sky'][model] = errors(read(scene, model, 'sky'), ref_sky, outside)
    row['runtime_vs_direct64'] = errors(read(scene, 'finite4', 'sky'), ref, outside)
    rows.append(row)
    print(name, 'parallel', round(row['source']['parallel']['outside_solar_disk']['p95'], 3),
          'four', round(row['source']['finite4']['outside_solar_disk']['p95'], 3),
          'four max', round(row['source']['finite4']['outside_solar_disk']['max'], 3))
    # Compare actual displayed SkyView outputs at identical exposure. Numerical
    # model-only errors above use direct integration to avoid hiding a change.
    point = read(scene, 'parallel', 'sky')
    fast = read(scene, 'finite4', 'sky')
    lum = ref_sky@np.array([.2627, .6780, .0593], dtype=F)
    positive = lum[lum > F(1e-8)]
    exposure = F(.6)/max(np.quantile(positive, F(.9)) if positive.size else F(1e-7), F(1e-7))
    den = np.maximum(np.linalg.norm(ref_sky, axis=1), F(1e-7))
    point_error = np.linalg.norm(point-ref_sky, axis=1)/den
    fast_error = np.linalg.norm(fast-ref_sky, axis=1)/den
    panels = [
        ('dense', '64-point disk / SkyView', display(ref_sky, exposure)),
        ('parallel', 'Parallel beam', display(point, exposure)),
        ('four', '4-point disk (default)', display(fast, exposure)),
        ('parallel_error', 'Parallel error: yellow = 5%', heatmap(point_error, 5)),
        ('four_error', '4-point error: yellow = 1%', heatmap(fast_error, 1)),
    ]
    sheet = Image.new('RGB', (1920, 240), (20, 24, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (key, title, values) in enumerate(panels):
        path = folder/f'{name}_{key}.png'
        save_png(path, values, scene['width'], scene['height'])
        sheet.paste(Image.open(path).resize((384, 192)), (i*384, 48))
        draw.text((i*384+5, 8), title, font=font, fill='white')
    sheet.save(folder/f'{name}_sheet.png')
    sections.append(f'<section><h2>{name}</h2><p>太阳盘外，直接积分误差 P95：平行光 {row["source"]["parallel"]["outside_solar_disk"]["p95"]:.3f}%；4 点 {row["source"]["finite4"]["outside_solar_disk"]["p95"]:.3f}%</p><img loading="lazy" src="{name}_sheet.png"></section>')

sweep_rows = []
previous = {}
for scene in json.loads((root/'sweep_queries.json').read_text())['images']:
    group = '_'.join(scene['name'].split('_')[:2])
    ref = read(scene, 'finite64', 'sky')
    outside = outside_disk(scene)
    row = dict(scene=scene['name'])
    for model in ['parallel', 'finite4']:
        value = read(scene, model, 'sky')
        residual = value-ref
        row[model] = errors(value, ref, outside)
        key = (group, model)
        if key in previous:
            row[model]['residual_change_p95'] = errors(ref+residual-previous[key], ref, outside)['p95']
        previous[key] = residual
    sweep_rows.append(row)

convergence = []
for scene in json.loads((root/'convergence_queries.json').read_text())['images']:
    dense = read(scene, 'finite64', 'source', 512)
    outside = outside_disk(scene)
    convergence.append(dict(scene=scene['name'],
        four_vs_64_at_512=errors(read(scene, 'finite4', 'source', 512), dense, outside),
        dense128_vs_dense512=errors(read(scene, 'finite64', 'source'), dense, outside)))

summary = dict(rows=rows, sweeps=sweep_rows, convergence=convergence,
    metric='Relative linear Rec2020 vector error; active reference norm >1e-8, denominator floor 1e-7. Percent units.',
    reference='64 solar samples: same four radial Gauss nodes as finite16, sixteen azimuths. Same fixed MS source; this isolates runtime solar integration, not a new full multiple-scattering solve.',
    default_model='Finite4, four equally weighted directions at half solid-angle radius; 128 steps; 256x256 SkyView')
(root/'summary.json').write_text(json.dumps(summary, indent=2))
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Solar approximation audit</title><style>body{background:#10151d;color:#ddd;font:16px system-ui;margin:24px}img{width:100%;max-width:1920px}section{margin:28px 0}p{line-height:1.6}</style><h1>太阳模型：平行光与 4 点圆盘</h1><p>四波长、128 步、固定各向异性多重散射。64 点圆盘作为本轮对照；MS 与原冻结资源相同。指标区分直接积分与 SkyView，正文 P95 排除太阳圆盘覆盖区。画面均为 SkyView 输出，曝光一致，未绘制太阳圆盘。</p><p>右侧两张误差图使用不同标尺：平行光黄色为 5%，4 点黄色为 1%。新增误差不包括四波长压缩和冻结参考自身误差。</p>'+''.join(sections), encoding='utf-8')
print('sweep max:', {m: max(r[m]['p95'] for r in sweep_rows) for m in ['parallel', 'finite4']})
print('sweep residual max:', {m: max(r[m].get('residual_change_p95', 0) for r in sweep_rows) for m in ['parallel', 'finite4']})
for row in convergence:
    print('convergence', row['scene'], 'model P95', round(row['four_vs_64_at_512']['p95'], 3),
          '128-step P95', round(row['dense128_vs_dense512']['p95'], 3))
