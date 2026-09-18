"""f32 spectral + angular factor probe, evaluated on original spectral values.

No codec or GPU performance claim: all errors are on fitted tensor nodes.
The signed spectral coefficients use ordinary linear SVD, without clamping.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("asset", type=Path)
p.add_argument("basis", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--sqrt", action="store_true", help="fit and decode in the nonnegative square-root radiance domain")
a = p.parse_args()
m = json.loads((a.asset / "asset.json").read_text())
c = m["config"]
nr, nm, ns, nn = c["scattering"]
heights = sorted(set([0, 1, 2, 4, 8, 12, nr//3, nr//2, 3*nr//4, nr-1]))
basis = np.array(json.loads(a.basis.read_text())["basis"], dtype=np.float32)[:, :12]
samples = np.empty((len(heights)*ns, nm, nn, len(m["bands"])), dtype=np.float32)
for band in range(len(m["bands"])):
    data = np.memmap(a.asset / f"band_{band:03}.bin", mode="r", dtype="<f4",
                     offset=8+int(np.prod(c["optical_depth"]))*4, shape=(nr,nm,ns,nn))
    samples[..., band] = np.stack([data[r,:,s,:] for r in heights for s in range(ns)])
working = np.sqrt(samples) if a.sqrt else samples
if a.sqrt:
    flat = working.reshape(-1, working.shape[-1])
    indices = np.random.default_rng(91837).choice(len(flat), 120000, replace=False)
    training = flat[indices]
    lengths = np.linalg.norm(training, axis=1)
    training = training[lengths > np.float32(1e-7)] / lengths[lengths > np.float32(1e-7), None]
    _, vectors = np.linalg.eigh(training.T @ training / np.float32(len(training)))
    basis = vectors[:, ::-1][:, :12]
coefficients = (working.reshape(-1, len(m["bands"])) @ basis).reshape(*samples.shape[:-1], 12)
if a.sqrt:
    del working
coefficients = coefficients.transpose(3,0,1,2).copy()
print("loaded spectra and projected 12 coefficients", flush=True)
u, sigma, vh = np.linalg.svd(coefficients, full_matrices=False)
assert u.dtype == np.float32
norm2 = np.sum(samples*samples, axis=3)
peak2 = norm2.max(axis=(1,2), keepdims=True)
lit = (norm2 > peak2*np.float32(1e-10)) & (norm2 > np.float32(1e-28))
auxiliary = (int(np.prod(c["optical_depth"]))+c["ground_sun_samples"])*4*len(m["bands"])
results = []
for spectral_rank in [8,12]:
    for angular_rank in [4,8,12,16]:
        approximation = ((u[:spectral_rank,:,:,:angular_rank]*sigma[:spectral_rank,:,None,:angular_rank])
                         @ vh[:spectral_rank,:,:angular_rank,:])
        decoded = (approximation.transpose(1,2,3,0).reshape(-1,spectral_rank)
                   @ basis[:,:spectral_rank].T).reshape(samples.shape)
        if a.sqrt:
            np.maximum(decoded,0,out=decoded)
            np.square(decoded,out=decoded)
        delta = decoded-samples
        error2 = np.sum(delta*delta,axis=3)
        relative = np.sqrt(error2[lit]/norm2[lit])
        slab_error = np.sqrt(error2.sum(axis=(1,2))/np.maximum(norm2.sum(axis=(1,2)),np.float32(1e-28)))
        size = nr*ns*spectral_rank*angular_rank*(nm+nn)*4+auxiliary+basis.shape[0]*spectral_rank*4
        results.append(dict(spectral_rank=spectral_rank,angular_rank=angular_rank,estimated_bytes=size,
            spectral_relative_l2_p95=float(np.percentile(relative,95)),
            spectral_relative_l2_p99=float(np.percentile(relative,99)),
            slab_relative_l2_p95=float(np.percentile(slab_error,95)),
            negative_fraction=float(np.mean(decoded[lit]<0))))
        print(results[-1], flush=True)
a.out.write_text(json.dumps(dict(asset=str(a.asset),basis_source="fitted sqrt spectra" if a.sqrt else str(a.basis),basis=basis.tolist(),heights=heights,dtype="float32",
    transform="sqrt; clamp negative amplitude before squaring" if a.sqrt else "linear; signed spectra retained",
    scope="spectral PCA followed by independent coefficient angular SVD; on fitted tensor nodes; no off-grid or finite-segment guarantee",
    norm_floor="exclude spectra below 1e-5 of slab peak norm or below 1e-14 absolute norm",results=results),indent=2))
