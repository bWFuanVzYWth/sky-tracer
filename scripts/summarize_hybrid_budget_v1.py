"""Keep reductions, angular convergence, and final SkyView interpolation separate."""
import json
from pathlib import Path
import numpy as np
from summarize_hybrid_sh import error,read as read_rgb
ROOT=Path('out/hybrid_budget_v1')
def read(p):
    a=read_rgb(p)
    if not np.isfinite(a).all():raise ValueError(f'Nonfinite {p}')
    return a
def compare(a,b,queries,mode='source'):
    return [dict(scene=s['name'],**error(read(a/(s['name']+'_'+mode+'.f32')),read(b/(s['name']+'_'+mode+'.f32')))) for s in queries]
def main():
    scenes=json.loads(Path('out/four_wave_sun_v1/queries.json').read_text())['images'];results={}
    for p in ROOT.iterdir():
        if not (p/'runs.json').exists() or not (p/'noon_aureole_source.f32').exists():continue
        direct=compare(p,ROOT/'baseline',scenes);sky=compare(p,ROOT/'baseline',scenes,'sky')
        dense=compare(p,Path('out/hybrid_perf_v1/q64_high_sun0'),scenes)
        solve=json.loads((p/'solve.json').read_text())
        results[p.name]=dict(solve=solve,direct=direct,sky=sky,dense=dense)
        top=max(sky,key=lambda r:r['p95']);tail=max(sky,key=lambda r:r['p99'])
        print(f'{p.name:20} {solve["solve"]["wall_seconds"]:5.3f}s {solve["solve"]["resident_bytes"]/1048576:6.2f}MiB peak{solve["solve"]["peak_payload_bytes"]/1048576:7.1f} P95 {top["p95"]:6.3f}% {top["scene"]:22} P99 {tail["p99"]:6.3f}% {tail["scene"]}')
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
