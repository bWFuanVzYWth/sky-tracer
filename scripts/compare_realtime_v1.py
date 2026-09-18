"""Compare committed UE8 and hybrid defaults, without rebaking the reference."""
import json
import subprocess
from pathlib import Path

ROOT = Path('out/realtime_comparison_v1')
ROOT.mkdir(parents=True, exist_ok=True)


def run(args, log):
    with log.open('w', encoding='utf-8') as f:
        subprocess.run([str(v) for v in args], stdout=f, stderr=subprocess.STDOUT, check=True)


for thickness in [100, 120]:
    out = ROOT / ('ue' if thickness == 100 else 'ue_top120')
    if not (out / 'runs.json').exists():
        run(['target/release/examples/evaluate_unreal.exe', '--out', out,
             '--thickness-km', thickness], ROOT / f'ue_{thickness}.log')

scenes = [('noon', 'out/pt_noon_085/asset.json'),
          ('sunset', 'out_skyview_search/elev_000/asset.json'),
          ('blue', 'out/lut_pt_v1/elev_m06/asset.json')]
timings = []
for name, asset in scenes:
    for repeat in range(3):
        for solver in (['unreal-8wave', 'hybrid-4d'] if repeat % 2 == 0 else ['hybrid-4d', 'unreal-8wave']):
            output = ROOT / f'{name}_{solver}_{repeat}.f32'
            meta = output.with_suffix('.json')
            if not meta.exists() or not output.exists():
                run(['target/release/sky-realtime-demo.exe', '--experiment', solver,
                     '--asset', asset, '--snapshot', output, '--snapshot-linear',
                     '--benchmark-frames', 64, '--snapshot-pitch-deg', 4,
                     '--snapshot-fov-deg', 100], output.with_suffix('.log'))
            info = json.loads(meta.read_text())
            timings.append(dict(scene=name, solver=solver, repeat=repeat, metadata=info))
            print(name, solver, repeat, [(p['workload'], round(p['gpu_median_ms'], 4)) for p in info['performance']], flush=True)
(ROOT / 'timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
