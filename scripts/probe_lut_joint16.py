"""CPU/f32 joint (height, Sun) x (view, phase, RGB) low-rank experiment.

All four reference axes retain every node. Ground/sky branches are separate.
The factors are stored, then independently reconstructed before the original
nonlinear interpolation by validate_lut_16mb_cpu. This is not a renderer asset.
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
p.add_argument("--rank", type=int, default=128)
p.add_argument("--height-groups", type=int, default=1)
p.add_argument("--sun-groups", type=int, default=1)
p.add_argument("--transform", choices=["log", "log1p", "sqrt"], default="log1p")
p.add_argument("--mean-storage", choices=["f16", "f32"], default="f32")
p.add_argument("--power-iterations", type=int, default=2)
p.add_argument("--sun", type=Path, default=Path("out/lut_16mb_candidate"))
a = p.parse_args()
if a.out.exists():
    raise ValueError("output exists")
m = json.loads((a.source / "asset.json").read_text())
nh, nv, ns, nn = m["config"]["scattering"]
ng = max(nv//4, 2)
groups = a.height_groups*a.sun_groups
rows = nh*ns
coefficient_bytes = 2*rows*a.rank*2
basis_bytes = groups*a.rank*nv*nn*3*2
mean_bytes = groups*nv*nn*3*(2 if a.mean_storage == "f16" else 4)
scale_bytes = 2*rows*4
sun_meta = json.loads((a.sun / "candidate.json").read_text())
budget = coefficient_bytes+basis_bytes+mean_bytes+scale_bytes+sun_meta["sun_bytes"]+131072
if budget > 16_000_000:
    raise ValueError(f"budget {budget}")
a.out.mkdir(parents=True)
start = time.perf_counter()
shape = (nh, nv, ns, nn)
channels = [np.memmap(a.source / f"channel_{c}.bin", mode="r", dtype="<f4",
            offset=8+int(np.prod(m["config"]["optical_depth"]))*4, shape=shape) for c in range(3)]
row_h, row_s = np.divmod(np.arange(rows), ns)
row_group = (row_h*a.height_groups//nh)*a.sun_groups+row_s*a.sun_groups//ns
floor = np.float32(1e-6 if a.transform == "log1p" else 1e-20)
diagnostics = []

for branch, (v0, v1) in enumerate([(0, ng), (ng, nv)]):
    views = v1-v0
    features = views*nn*3
    coefficients = np.zeros((rows, a.rank), dtype="<f2")
    # Feature-major factors make each per-corner dot product contiguous.
    bases = np.zeros((groups, features, a.rank), dtype="<f2")
    means = np.zeros((groups, features), dtype="<f2" if a.mean_storage == "f16" else "<f4")
    scales = np.zeros(rows, dtype="<f4")
    raw = np.stack([c[:, v0:v1].transpose(0, 2, 1, 3).reshape(rows, features//3)
                    for c in channels], axis=-1).reshape(rows, features)
    for g in range(groups):
        indices = np.flatnonzero(row_group == g)
        work = raw[indices].copy()
        peak = np.maximum(work.max(axis=1), np.float32(0))
        scales[indices] = peak
        work = np.maximum(work, np.float32(0))/np.maximum(peak[:, None], np.float32(1e-30))
        if a.transform == "log1p":
            np.log1p(work/floor, out=work)
        elif a.transform == "log":
            np.log(np.maximum(work, floor), out=work)
        else:
            np.sqrt(work, out=work)
        active = peak > np.float32(1e-20)
        if active.sum() <= a.rank:
            raise ValueError("not enough active rows for requested rank")
        mean = np.mean(work[active], axis=0, dtype=np.float32)
        work -= mean
        # Black rows are outside the fit and decode exactly via their zero scale.
        work[~active] = 0
        k = min(a.rank+16, min(work.shape))
        rng = np.random.default_rng(48271+branch*groups+g)
        q = work @ rng.standard_normal((features, k), dtype=np.float32)
        q = qr(q, mode="economic", check_finite=False)[0]
        z = None
        for _ in range(a.power_iterations):
            z = qr(work.T @ q, mode="economic", check_finite=False)[0]
            q = qr(work @ z, mode="economic", check_finite=False)[0]
        b = q.T @ work
        u, singular, vt = svd(b, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        assert u.dtype == np.float32 and vt.dtype == np.float32
        coef = (q @ u[:, :a.rank])*singular[:a.rank]
        basis = vt[:a.rank].T.copy()
        coefficients[indices] = coef.astype(np.float16)
        bases[g] = basis.astype(np.float16)
        means[g] = mean
        # Fit-node samples use rounded factors, not the unquantized SVD output.
        sample = indices[::17]
        decoded = coefficients[sample].astype(np.float32) @ bases[g].astype(np.float32).T
        decoded += means[g].astype(np.float32)
        if a.transform == "log1p":
            decoded = np.expm1(np.clip(decoded, 0, 20))*floor
        elif a.transform == "log":
            decoded = np.exp(np.minimum(decoded, 0))
        else:
            decoded = np.maximum(decoded, 0)**2
        decoded *= scales[sample, None]
        original = raw[sample].reshape(-1, 3)
        decoded = decoded.reshape(-1, 3)
        n = np.linalg.norm(original, axis=1)
        rel = np.linalg.norm(decoded-original, axis=1)[n>1e-8]/n[n>1e-8]
        item = dict(branch=branch, group=g, fit_node_sample_p95_p99=np.percentile(rel, [95, 99]).tolist(),
                    last_retained_singular=float(singular[a.rank-1]), next_singular=float(singular[a.rank]),
                    elapsed_seconds=time.perf_counter()-start)
        diagnostics.append(item)
        print(item, flush=True)
        del work, q, z, b, u, vt, coef, basis, decoded, original
    coefficients.tofile(a.out / f"coefficients_{branch}.f16")
    bases.tofile(a.out / f"basis_{branch}.f16")
    means.tofile(a.out / f"mean_{branch}.{a.mean_storage}")
    scales.tofile(a.out / f"scale_{branch}.f32")
    del raw

shutil.copyfile(a.sun / "sun.f16", a.out / "sun.f16")
metadata = dict(kind="cpu_joint_pair_svd_16mb_v1", source=str(a.source), source_config=m["config"],
    config=m["config"], source_channel_checksums=m["rgb"]["channel_checksums"],
    source_model=m["model_fingerprint_fnv1a64"], shape=list(shape), rank=a.rank,
    height_groups=a.height_groups, sun_groups=a.sun_groups, groups=groups,
    transform=a.transform, transform_floor=float(floor), mean_storage=a.mean_storage,
    factorization="sum_k A_k(height,Sun) B_k(view,phase,RGB); separate ground/sky",
    coefficient_bytes=coefficient_bytes, basis_bytes=basis_bytes, mean_bytes=mean_bytes,
    scale_bytes=scale_bytes, sun_bytes=sun_meta["sun_bytes"], sun_shape=sun_meta["sun_shape"],
    metadata_reserve=131072, gpu_payload_budget_bytes=budget, power_iterations=a.power_iterations,
    fitting_dtype="float32", cpu_seconds=time.perf_counter()-start,
    sky_only=True, finite_segment_transport=False, gpu_integrated=False, diagnostics=diagnostics)
(a.out / "candidate.json").write_text(json.dumps(metadata, indent=2))
actual = sum(f.stat().st_size for f in a.out.iterdir() if f.name != "candidate.json")
assert actual+131072 == budget
print(json.dumps({k: metadata[k] for k in ["kind", "rank", "groups", "gpu_payload_budget_bytes", "cpu_seconds"]}), flush=True)
