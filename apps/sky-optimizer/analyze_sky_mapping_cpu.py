"""CPU-only quality/cost summary for serialized SkyView coordinate audits."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    data = np.fromfile(path, dtype='<f4').reshape(-1, 4)[:, :3].astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite image: {path}')
    return data


def error(a, b):
    n = np.linalg.norm(a, axis=1)
    active = n > 1e-8
    e = np.linalg.norm(a - b, axis=1) / np.maximum(n, 1e-7)
    q = np.percentile(e[active], [50, 95, 99, 100]) if active.any() else [0] * 4
    return dict(zip(['p50', 'p95', 'p99', 'max'], (100 * np.array(q)).tolist()),
                active_pixels=int(active.sum()), maximum_absolute=float(np.linalg.norm(a-b,axis=1).max()))


def worst(rows):
    return {metric: max(rows, key=lambda r:r[metric]) for metric in ['p95', 'p99', 'max']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static', type=Path, required=True)
    p.add_argument('--cost', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen-fixed', type=Path,
                   default=ROOT/'out/realtime_dynamic_v7/quality_fixed_v1')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Choose a new output directory')
    queries = {'43':ROOT/'experiments/validation/sky_cases.json',
               'extra22':ROOT/'experiments/validation/mapping_extra_cases.json',
               'aerosol4':a.static/'aerosol4_queries.json',
               'hd14':ROOT/'experiments/validation/moving_sun_1080_queries.json'}
    rows = []
    invariants = []
    for group, path in queries.items():
        for image in json.loads(path.read_text())['images']:
            name = image['name']
            source_path = a.static/f'baseline_{group}'/f'{name}_source.f32'
            source_bytes = source_path.read_bytes()
            source = read(source_path)
            baseline_path = a.static/f'baseline_{group}'/f'{name}_sky.f32'
            baseline = read(baseline_path)
            for variant in ['baseline', '256', '224']:
                sky = read(a.static/f'{variant}_{group}'/f'{name}_sky.f32')
                rows.append(dict(group=group, scene=name, variant=variant,
                                 source_cache_error=error(source, sky),
                                 old_mapping_difference=error(baseline, sky)))
                if (a.static/f'{variant}_{group}'/f'{name}_source.f32').read_bytes() != source_bytes:
                    raise ValueError(f'Source changed in {variant}/{group}/{name}')
            frozen = a.frozen_fixed/f'steps64_{group}'
            if frozen.exists():
                for mode in ['source', 'sky']:
                    original = frozen/f'{name}_{mode}.f32'
                    current = a.static/f'baseline_{group}'/original.name
                    if original.exists():
                        exact = original.read_bytes() == current.read_bytes()
                        invariants.append(dict(group=group,scene=name,mode=mode,bit_identical=exact))
    summary = {}
    for variant in ['baseline', '256', '224']:
        summary[variant] = {}
        for group in queries:
            selected=[r for r in rows if r['variant']==variant and r['group']==group]
            summary[variant][group]={metric:worst([dict(scene=r['scene'],**r[metric]) for r in selected])
                                    for metric in ['source_cache_error','old_mapping_difference']}
    cost = {}
    for variant in ['control_ground', 'baseline', '256', '224']:
        profile=json.loads((a.cost/f'{variant}_1080_cost/profile.json').read_text())
        cost[variant]={r['trajectory']:{key:value['median_of_round_medians'] for key,value in r['metrics'].items()}
                       for r in profile['summary']}
    means={variant:{key:float(np.mean([m[key] for name,m in trajectories.items() if name!='azimuth_only']))
                    for key in ['full_gpu_ms','encode_submit_wait_cpu_ms','observer_mapping_cpu_ms']}
           for variant,trajectories in cost.items()}
    a.out.mkdir(parents=True)
    report=dict(definition='Linear Rec.2020 RGB vector norm; percent; reference norm>1e-8; denominator>=1e-7.',
                source_reference='Same solver/runtime64 direct integration at image directions. Cache error isolates SkyView interpolation; it is not full physical error against teacher.',
                static_summary=summary,static_rows=rows,frozen_baseline_invariants=invariants,
                source_images_bit_identical=True,cost=cost,cost_four_elevation_means=means,
                cost_scope='1920x1080 SkyView+projection+actual demo display shader, 120frames x3 rounds; OS present/vsync excluded; fixed-medium startup excluded.',
                gpu_initialized=False)
    # This gallery reads the actual display-attachment pixels. It does not
    # implement an approximate CPU tone curve.
    from PIL import Image, ImageDraw
    captures=json.loads((a.static/'baseline_hd_display/profile.json').read_text())['captures']
    width,height=480,270
    gallery=Image.new('RGB',(width*4, (height+32)*len(captures)+28),(20,20,24))
    draw=ImageDraw.Draw(gallery)
    for x,label in enumerate(['Original 256','Fitted 256','Fitted 224','224 display difference x32']):
        draw.text((x*width+8,8),label,fill='white')
    display_rows=[]
    for i,capture in enumerate(captures):
        arrays=[]
        for variant in ['baseline','256','224']:
            pixels=np.fromfile(a.static/f'{variant}_hd_display'/capture['display'],dtype='u1')
            arrays.append(pixels.reshape(capture['size'][1],capture['size'][0],4)[...,:3])
        for variant,pixels in zip(['256','224'],arrays[1:]):
            delta=np.max(np.abs(pixels.astype(np.int16)-arrays[0].astype(np.int16)),axis=-1)
            display_rows.append(dict(scene=capture['trajectory'],variant=variant,
                                     maximum_lsb=int(delta.max()),pixels_over_1=int((delta>1).sum()),
                                     pixels_over_2=int((delta>2).sum())))
        difference=np.clip(np.abs(arrays[2].astype(np.int16)-arrays[0].astype(np.int16))*32,0,255).astype('u1')
        for j,pixels in enumerate(arrays+[difference]):
            panel=Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS)
            y=28+i*(height+32)
            gallery.paste(panel,(j*width,y))
            draw.text((j*width+8,y+height+6),capture['trajectory'],fill='white')
    gallery.save(a.out/'actual_display_gallery.png')
    report['display_rows']=display_rows
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(means=means,all_frozen_exact=all(r['bit_identical'] for r in invariants),
                          frozen_images=len(invariants)),indent=2))


if __name__ == '__main__':
    main()
