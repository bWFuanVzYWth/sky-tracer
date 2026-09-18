"""CPU-only local PCA candidates for a strict 16,000,000-byte GPU payload.

Fit the frozen spectral bake's RGB nodes, reconstruct individual corners before
the original nonlinear interpolation. This is an experiment, not a demo format.
All fitting and reconstruction arithmetic is float32; f16/i8 are storage only.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import map_coordinates

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--transform", choices=["sqrt", "log1p", "log"], default="sqrt")
p.add_argument("--height-groups", type=int, default=8)
p.add_argument("--sun-groups", type=int, default=4)
p.add_argument("--rank", type=int, default=12)
p.add_argument("--coefficient", choices=["f16", "i8"], default="f16")
a = p.parse_args()
if a.out.exists():
    raise ValueError("Output already exists; choose a new directory")
m = json.loads((a.source / "asset.json").read_text())
assert m["rgb"].get("packed") is None
nr, nv, ns, nn = m["config"]["scattering"]
tr, tm = m["config"]["optical_depth"]
rows, features = nr * nv * ns, nn * 3
ng = max(nv // 4, 2)
groups = a.height_groups * a.sun_groups * 2
coefficient_bytes = rows * a.rank * (2 if a.coefficient == "f16" else 1)
factor_bytes = groups * a.rank * features * 2
mean_bytes = groups * features * 4
scale_bytes = rows * 4 + groups * a.rank * 4
sun_size = [96, 1024]
sun_bytes = int(np.prod(sun_size)) * 3 * 2
# Includes reference height/Sun-coordinate caches (~80 KiB) and small uniforms.
metadata_reserve = 131072
budget = coefficient_bytes + factor_bytes + mean_bytes + scale_bytes + sun_bytes + metadata_reserve
assert budget <= 16000000, budget
a.out.mkdir(parents=True)
start = time.perf_counter()
channels = [np.memmap(a.source / f"channel_{c}.bin", dtype="<f4", mode="r",
            offset=8+tr*tm*4, shape=(rows,nn)) for c in range(3)]
row_index = np.arange(rows)
h = row_index // (nv*ns)
v = row_index // ns % nv
s = row_index % ns
group = ((h*a.height_groups//nr)*a.sun_groups + s*a.sun_groups//ns)*2 + (v>=ng)
scales = np.zeros(rows, dtype="<f4")
coefficients = np.zeros((rows,a.rank), dtype="<f2" if a.coefficient == "f16" else "i1")
bases = np.zeros((groups,a.rank,features), dtype="<f2")
means = np.zeros((groups,features), dtype="<f4")
coefficient_scales = np.ones((groups,a.rank), dtype="<f4")
node_norms, node_errors = [], []
transform_floor = np.float32(1e-6 if a.transform == "log1p" else 1e-20)

for g in range(groups):
    indices = np.flatnonzero(group==g)
    raw = np.stack([c[indices] for c in channels], axis=-1).reshape(-1,features)
    peak = np.maximum(np.max(raw,axis=1), np.float32(0))
    scales[indices] = peak
    work = np.maximum(raw,np.float32(0)) / np.maximum(peak[:,None],np.float32(1e-30))
    if a.transform == "sqrt":
        np.sqrt(work,out=work)
    elif a.transform == "log1p":
        np.log1p(work / transform_floor,out=work)
    else:
        np.log(np.maximum(work,transform_floor),out=work)
    active = peak > np.float32(1e-20)
    # Entirely black rows decode to zero via peak, regardless of PCA outputs.
    training = work[active]
    if len(training):
        mean = np.mean(training,axis=0,dtype=np.float32)
        training -= mean
        covariance = training.T @ training / np.float32(len(training))
        _, vectors = eigh(covariance, subset_by_index=[features-a.rank,features-1],check_finite=False,driver="evr")
        basis = vectors[:,::-1].T.copy()
        assert basis.dtype == np.float32
        coeff = (work-mean) @ basis.T
        # Quantize first, then measure the actual stored reconstruction.
        means[g] = mean
        bases[g] = basis.astype(np.float16)
        if a.coefficient == "i8":
            cs = np.maximum(np.max(np.abs(coeff[active]),axis=0)/np.float32(127),np.float32(1e-20))
            coefficient_scales[g] = cs
            coefficients[indices] = np.clip(np.rint(coeff/cs),-127,127).astype(np.int8)
        else:
            coefficients[indices] = coeff.astype(np.float16)
    if g%8==0:
        print(f"{a.transform} {a.coefficient} rank {a.rank}: group {g+1}/{groups}",flush=True)
    # Uniform deterministic subset of fit nodes; off-grid validation is separate.
    decoded = coefficients[indices[::13]].astype(np.float32) * coefficient_scales[g]
    decoded = decoded @ bases[g].astype(np.float32) + means[g]
    if a.transform == "sqrt":
        decoded = np.maximum(decoded,0)**2
    elif a.transform == "log1p":
        decoded = np.maximum(np.expm1(np.minimum(decoded,np.float32(20))),0)*transform_floor
    else:
        decoded = np.exp(np.minimum(decoded,np.float32(0)))
    decoded *= peak[::13,None]
    original = raw[::13].reshape(-1,3)
    decoded = decoded.reshape(-1,3)
    n = np.sqrt(np.sum(original*original,axis=1,dtype=np.float32))
    error = np.sqrt(np.sum((decoded-original)**2,axis=1,dtype=np.float32))
    lit = n>np.float32(1e-8)
    node_norms.append(n[lit]);node_errors.append(error[lit])

coefficients.tofile(a.out / "coefficients.bin")
bases.tofile(a.out / "basis.f16")
means.tofile(a.out / "mean.f32")
scales.tofile(a.out / "row_scale.f32")
coefficient_scales.tofile(a.out / "coefficient_scale.f32")

# Only sky-reaching rays are queried for a visible Sun. Keep that half of the
# original split optical table and sample its endpoints without crossing charts.
sun = np.stack([np.memmap(a.source / f"channel_{c}.bin", dtype="<f4",mode="r",offset=8,
               shape=(tr,tm))[:,tm//2:] for c in range(3)],axis=-1)
solar = np.asarray(m["rgb"]["solar_irradiance"],dtype=np.float32)
yy,xx = np.meshgrid(np.linspace(0,tr-1,sun_size[0],dtype=np.float32),
                    np.linspace(0,tm//2-1,sun_size[1],dtype=np.float32),indexing="ij")
small = np.stack([map_coordinates(sun[...,c]/solar[c],[yy,xx],order=1,prefilter=False) for c in range(3)],axis=-1).astype("<f2")
small.tofile(a.out / "sun.f16")
yy,xx = np.meshgrid(np.linspace(0,sun_size[0]-1,tr,dtype=np.float32),
                    np.linspace(0,sun_size[1]-1,tm//2,dtype=np.float32),indexing="ij")
restored = np.stack([map_coordinates(small[...,c].astype(np.float32),[yy,xx],order=1,prefilter=False) for c in range(3)],axis=-1)
sun_error = np.sqrt(np.sum((restored-sun/solar)**2,axis=-1))
sun_norm = np.sqrt(np.sum((sun/solar)**2,axis=-1))
sun_lit = sun_norm>np.float32(1e-4)
relative = np.concatenate(node_errors)/np.concatenate(node_norms)
summary = dict(kind="cpu_local_pca_16mb_probe_v1",source=str(a.source),source_config=m["config"],
    source_channel_checksums=m["rgb"]["channel_checksums"],source_model=m["model_fingerprint_fnv1a64"],
    shape=[nr,nv,ns,nn],height_groups=a.height_groups,sun_groups=a.sun_groups,groups=groups,
    rank=a.rank,coefficient=a.coefficient,transform=a.transform,transform_floor=float(transform_floor),
    coefficient_bytes=coefficient_bytes,basis_bytes=factor_bytes,mean_bytes=mean_bytes,scale_bytes=scale_bytes,
    sun_bytes=sun_bytes,metadata_reserve=metadata_reserve,gpu_payload_budget_bytes=budget,
    sun_shape=sun_size,sun_domain="sky half of original split coordinates; height unchanged",
    solar_irradiance=solar.tolist(),sky_only=True,finite_segment_transport=False,gpu_integrated=False,
    fitting_dtype="float32",cpu_seconds=time.perf_counter()-start,
    fitted_node_sample_relative_p50_p95_p99_max=np.percentile(relative,[50,95,99,100]).tolist(),
    sun_relative_p50_p95_p99_max=np.percentile(sun_error[sun_lit]/sun_norm[sun_lit],[50,95,99,100]).tolist(),
    sun_normalized_absolute_max=float(np.max(sun_error)))
(a.out / "candidate.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
