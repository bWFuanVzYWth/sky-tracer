"""Relative-energy f32 spectral PCA; report held-out LUT node errors.

The input remains the full 41-band resource. A shared basis is only a candidate
representation: it is not an eight-wavelength quadrature or a runtime encoder.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("asset", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
m = json.loads((a.asset / "asset.json").read_text())
assert all(r is not None for r in m["records"]), "complete spectrum required"
c = m["config"]
d = c["scattering"]
rng = np.random.default_rng(47021)
indices = rng.choice(int(np.prod(d)), size=min(120000, int(np.prod(d))), replace=False)
matrix = np.empty((len(indices), len(m["bands"])), dtype=np.float32)
for band in range(matrix.shape[1]):
    data = np.memmap(a.asset / f"band_{band:03}.bin", mode="r", dtype="<f4",
                    offset=8 + int(np.prod(c["optical_depth"])) * 4, shape=(int(np.prod(d)),))
    matrix[:, band] = data[indices]
norm = np.linalg.norm(matrix, axis=1)
lit = norm > np.float32(1e-14)
training = matrix[lit][::2]
training /= np.linalg.norm(training, axis=1)[:, None]
test = matrix[lit][1::2]
test_norm = np.linalg.norm(test, axis=1)
cov = training.T @ training / np.float32(len(training))
_, basis = np.linalg.eigh(cov)
basis = basis[:, ::-1]
assert basis.dtype == np.float32
results = []
for rank in [3, 4, 6, 8, 12, 16]:
    vectors = basis[:, :rank]
    approx = (test @ vectors) @ vectors.T
    relative = np.linalg.norm(approx - test, axis=1) / test_norm
    results.append(dict(rank=rank, coefficient_map_bytes=int(np.prod(d)) * rank * 4,
        relative_spectral_l2_median=float(np.median(relative)),
        relative_spectral_l2_p95=float(np.percentile(relative, 95)),
        relative_spectral_l2_p99=float(np.percentile(relative, 99)),
        relative_spectral_l2_max=float(relative.max()),
        negative_channel_fraction=float(np.mean(approx < 0))))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(dict(asset=str(a.asset), seed=47021, dtype="float32",
    training_nodes=len(training), held_out_nodes=len(test), sampling="distinct tensor indices without replacement",
    scope="held-out tensor nodes; no off-grid interpolation or finite-segment guarantee",
    basis=basis[:, :16].tolist(), results=results), indent=2))
print(json.dumps(results, indent=2))
