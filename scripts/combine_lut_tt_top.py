"""Package TT interior + directly encoded full-resolution atmospheric top.

Expanded data referenced by cpu_base_dir are only for CPU audit; all proposed
runtime data are copied into the output. Renderer integration is not provided.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("base",type=Path)
p.add_argument("top",type=Path)
p.add_argument("out",type=Path)
a=p.parse_args()
if a.out.exists():raise ValueError("output exists")
m=json.loads((a.base/"candidate.json").read_text())
t=json.loads((a.top/"candidate.json").read_text())
assert m["kind"]=="cpu_joint_tt_16mb_v1"
for key in ["source_model","source_channel_checksums"]:assert m[key]==t[key]
assert t["shape"]==[1,*m["shape"][1:]]
assert t["source_height_index"]==m["shape"][0]-1
budget=m["gpu_payload_budget_bytes"]+t["map_bytes"]+t["data_bytes"]
assert budget<=16_000_000,budget
a.out.mkdir(parents=True)
files=["core_a.f16","core_b.f16","core_c.f16","mean.f32","scale.f32","sun.f16"]
for name in files:shutil.copyfile(a.base/name,a.out/name)
for source,target in [("blocks.u32","top_blocks.u32"),("radiance.u32","top_radiance.u32")]:
    shutil.copyfile(a.top/source,a.out/target);files.append(target)
payload=sum((a.out/name).stat().st_size for name in files)
assert payload+131072==budget
m.update(kind="cpu_joint_tt_top_16mb_v1",cpu_base_dir=str(a.base),top_source=str(a.top),
    top_mantissa_bits=t["mantissa_bits"],top_bytes=t["map_bytes"]+t["data_bytes"],
    gpu_payload_budget_bytes=budget,payload_bytes=payload,
    payload_sha256={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in files})
m.pop("fit_node_sample_p50_p95_p99_max",None)
(a.out/"candidate.json").write_text(json.dumps(m,indent=2))
print(budget)
