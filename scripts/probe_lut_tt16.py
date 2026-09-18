"""CPU/f32 tensor-train probe with two nontrivial tensor cuts.

Index order (height, Sun, view, phase/RGB). The height identity core is implicit:
X[h,s,v,p,c] ~= sum_ab A[h,s,a] B[a,v,b] C[b,p,c].
Stored factors use f16; fitting/contractions use f32. Expanded decoded files
are CPU validation intermediates, never part of the proposed GPU payload.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.linalg import qr, svd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("source", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--rank", type=int, default=336)
p.add_argument("--tail-rank", type=int, default=192)
p.add_argument("--transform", choices=["log", "log1p"], default="log1p")
p.add_argument("--transform-floor", type=float)
p.add_argument("--scale-domain", choices=["hs", "hsv"], default="hs")
p.add_argument("--power-iterations", type=int, default=2)
p.add_argument("--sun", type=Path, default=Path("out/lut_16mb_candidate"))
a = p.parse_args()
if a.out.exists():
    raise ValueError("output exists")
m = json.loads((a.source / "asset.json").read_text())
nh, nv, ns, nn = m["config"]["scattering"]
rows, angular = nh*ns, nv*nn*3
rank, tail = a.rank, a.tail_rank
sun_meta = json.loads((a.sun / "candidate.json").read_text())
core_bytes = 2*(rows*rank+rank*nv*tail+tail*nn*3)
scale_count=rows*(nv if a.scale_domain == "hsv" else 1)
budget = core_bytes+4*(scale_count+angular)+sun_meta["sun_bytes"]+131072
assert budget <= 16_000_000, budget
a.out.mkdir(parents=True)
start = time.perf_counter()
raw = np.stack([np.memmap(a.source / f"channel_{c}.bin", dtype="<f4", mode="r",
                offset=8+int(np.prod(m["config"]["optical_depth"]))*4,
                shape=(nh,nv,ns,nn)).transpose(0,2,1,3).reshape(rows,angular//3)
                for c in range(3)], axis=-1).reshape(rows,angular)
if a.scale_domain == "hsv":
    scale=np.maximum(raw.reshape(rows,nv,nn*3).max(axis=2),np.float32(0))
    work=(np.maximum(raw.reshape(rows,nv,nn*3),0)/np.maximum(scale[:,:,None],np.float32(1e-30))).reshape(rows,angular)
else:
    scale = np.maximum(raw.max(axis=1),np.float32(0))
    work = np.maximum(raw,0)/np.maximum(scale[:,None],np.float32(1e-30))
floor = np.float32(a.transform_floor if a.transform_floor is not None else (1e-6 if a.transform == "log1p" else 1e-20))
assert np.isfinite(floor) and 0 < floor < 1
if a.transform == "log1p":
    assert np.log1p(np.float32(1)/floor) <= 20, "transform floor exceeds the recorded clamp's range"
if a.transform == "log1p":
    np.log1p(work/floor,out=work)
else:
    np.log(np.maximum(work,floor),out=work)
active = np.max(scale.reshape(rows,-1),axis=1) > np.float32(1e-20)
mean = np.mean(work[active],axis=0,dtype=np.float32)
work -= mean
work[~active] = 0
rng = np.random.default_rng(563417)
q = qr(work @ rng.standard_normal((angular,rank+16),dtype=np.float32),mode="economic",check_finite=False)[0]
z = None
for _ in range(a.power_iterations):
    z = qr(work.T @ q,mode="economic",check_finite=False)[0]
    q = qr(work @ z,mode="economic",check_finite=False)[0]
u, sigma, vt = svd(q.T @ work,full_matrices=False,check_finite=False,lapack_driver="gesdd")
core_a = (q @ u[:,:rank]).astype("<f2")
first_cut_spectrum = sigma.tolist()
print(f"first cut finished in {time.perf_counter()-start:.2f}s",flush=True)
del work,q,z,u
# Carry the singular values into the next cut. Compressing plain Vt would
# give weak and dominant first-cut modes equal weight, not TT-SVD's objective.
remaining = (sigma[:rank,None]*vt[:rank]).reshape(rank*nv,nn*3)
u2,sigma2,vt2 = svd(remaining,full_matrices=False,check_finite=False,lapack_driver="gesdd")
core_b = (u2[:,:tail]*sigma2[:tail]).reshape(rank,nv,tail).astype("<f2")
core_c = vt2[:tail].astype("<f2")
assert u2.dtype == np.float32 and vt2.dtype == np.float32
core_a.tofile(a.out / "core_a.f16")
core_b.tofile(a.out / "core_b.f16")
core_c.tofile(a.out / "core_c.f16")
mean.astype("<f4").tofile(a.out / "mean.f32")
scale.astype("<f4").tofile(a.out / "scale.f32")
shutil.copyfile(a.sun / "sun.f16",a.out / "sun.f16")
fit_seconds=time.perf_counter()-start
del vt,u2,vt2,remaining
ca,cb,cc=core_a.astype(np.float32),core_b.astype(np.float32).reshape(rank,nv*tail),core_c.astype(np.float32)
decoded=[np.memmap(a.out/f"decoded_{c}.f32",mode="w+",dtype="<f4",shape=(nh,nv,ns,nn)) for c in range(3)]
node_errors=[]
for begin in range(0,rows,384):
    end=min(begin+384,rows)
    z=(ca[begin:end] @ cb).reshape(-1,tail) @ cc
    z=z.reshape(end-begin,angular)+mean
    if a.transform == "log1p":
        z=np.expm1(np.clip(z,0,20))*floor
    else:
        z=np.exp(np.minimum(z,0))
    if a.scale_domain == "hsv":
        z=z.reshape(end-begin,nv,nn*3)
        z*=scale[begin:end,:,None]
        z=z.reshape(end-begin,angular)
    else:
        z*=scale[begin:end,None]
    original=raw[begin:end:29].reshape(-1,3)
    n=np.linalg.norm(original,axis=1)
    errors=np.linalg.norm(z[::29].reshape(-1,3)-original,axis=1)
    node_errors.append(errors[n>1e-8]/n[n>1e-8])
    z=z.reshape(end-begin,nv,nn,3)
    for i in range(begin,end):
        h,s=divmod(i,ns)
        for c in range(3):
            decoded[c][h,:,s,:]=z[i-begin,:,:,c]
for d in decoded:d.flush()
metadata=dict(kind="cpu_joint_tt_16mb_v1",source=str(a.source),source_config=m["config"],config=m["config"],
    source_channel_checksums=m["rgb"]["channel_checksums"],source_model=m["model_fingerprint_fnv1a64"],
    shape=[nh,nv,ns,nn],rank=rank,tail_rank=tail,transform=a.transform,transform_floor=float(floor),
    scale_domain=a.scale_domain,
    factorization="A[height,Sun,a] B[a,view,b] C[b,phase,RGB]; implicit height identity core",
    core_bytes=core_bytes,mean_bytes=mean.nbytes,scale_bytes=scale.nbytes,
    sequential_cut_weighting="singular values carried into the remaining tensor",
    sun_bytes=sun_meta["sun_bytes"],sun_shape=sun_meta["sun_shape"],metadata_reserve=131072,
    gpu_payload_budget_bytes=budget,first_cut_spectrum=first_cut_spectrum,second_cut_spectrum=sigma2.tolist(),
    fitting_dtype="float32",power_iterations=a.power_iterations,fit_seconds=fit_seconds,
    total_cpu_seconds=time.perf_counter()-start,
    cpu_audit_intermediates="decoded_*.f32 excluded from payload; runtime direct core contraction not benchmarked",
    gpu_integrated=False,sky_only=True,finite_segment_transport=False,
    fit_node_sample_p50_p95_p99_max=np.percentile(np.concatenate(node_errors),[50,95,99,100]).tolist())
(a.out / "candidate.json").write_text(json.dumps(metadata,indent=2))
print(json.dumps({k:metadata[k] for k in ["gpu_payload_budget_bytes","fit_seconds","total_cpu_seconds","fit_node_sample_p50_p95_p99_max"]}),flush=True)
