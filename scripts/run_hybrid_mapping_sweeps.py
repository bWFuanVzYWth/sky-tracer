import json,subprocess
from pathlib import Path
root=Path('out/hybrid_mapping_v3')
for name,config,steps in [('legacy','legacy',128),('final','selected_final',128),('dense','dense',256)]:
    out=root/('sweeps_'+name);out.mkdir(exist_ok=True)
    if (out/'runs.json').exists():continue
    with (out/'process.log').open('w') as log:
        subprocess.run(['target/release/examples/evaluate.exe','--out',str(out),'--queries',str(root/'sweeps.json'),'--config',str(root/'configs'/(config+'.json')),'--steps',str(steps)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'sweep complete',flush=True)
