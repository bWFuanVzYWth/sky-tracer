"""f32 angular SVD probe on actual baked bands, not a shipping compressor.

Each selected (height, sun) slab is fit independently. Reports on-grid errors;
interpolated queries and finite-distance transport need separate validation.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("asset", type=Path)
p.add_argument("--bands", nargs="+", type=int, default=[6, 17, 30])
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
m = json.loads((a.asset / "asset.json").read_text())
c = m["config"]
nr, nm, ns, nn = c["scattering"]
heights = sorted(set([0, 1, 2, 4, 8, 12, nr // 3, nr // 2, 3 * nr // 4, nr - 1]))
results = []
for band in a.bands:
    if m["records"][band] is None:
        raise ValueError(f"Band {band} is incomplete")
    data = np.memmap(a.asset / f"band_{band:03}.bin", dtype="<f4", mode="r",
                     offset=8 + int(np.prod(c["optical_depth"])) * 4,
                     shape=(nr, nm, ns, nn))
    slabs = np.stack([data[r, :, s, :] for r in heights for s in range(ns)])
    peak = np.maximum(slabs.max(axis=(1, 2), keepdims=True), np.float32(1e-30))
    for transform in ["linear", "row_emphasis", "sqrt"]:
        scale = np.ones_like(peak)
        if transform == "row_emphasis":
            scale = np.maximum(np.sqrt(np.mean(slabs * slabs, axis=2, keepdims=True)), peak * np.float32(1e-6))
        work = np.sqrt(slabs / peak) if transform == "sqrt" else slabs / scale
        u, sigma, vh = np.linalg.svd(work, full_matrices=False)
        assert u.dtype == np.float32
        for rank in [2, 4, 8, 12, 16]:
            approx = (u[:, :, :rank] * sigma[:, None, :rank]) @ vh[:, :rank]
            if transform == "sqrt":
                approx = np.maximum(approx, 0) ** 2 * peak
            else:
                approx *= scale
            delta = approx - slabs
            lit = slabs > peak * np.float32(1e-5)
            relative = np.abs(delta[lit]) / slabs[lit]
            relative_slab_l2 = np.sqrt(np.sum(delta * delta, axis=(1, 2)) /
                                       np.maximum(np.sum(slabs * slabs, axis=(1, 2)), np.float32(1e-30)))
            radiance_bytes = nr * ns * rank * (nm + nn) * 4 * len(m["bands"])
            if transform == "sqrt":
                radiance_bytes += nr * ns * 4 * len(m["bands"])
            elif transform == "row_emphasis":
                radiance_bytes += nr * ns * nm * 4 * len(m["bands"])
            auxiliary = (int(np.prod(c["optical_depth"])) + c["ground_sun_samples"]) * 4 * len(m["bands"])
            results.append(dict(band_nm=m["bands"][band]["center_nm"], transform=transform,
                rank=rank, estimated_full_spectral_bytes=radiance_bytes + auxiliary,
                payload_compression_ratio=c["scattering"][0] * nm * ns * nn * 4 * len(m["bands"]) / radiance_bytes,
                slab_relative_l2_median=float(np.median(relative_slab_l2)),
                slab_relative_l2_p95=float(np.percentile(relative_slab_l2, 95)),
                point_relative_p95=float(np.percentile(relative, 95)),
                point_relative_p99=float(np.percentile(relative, 99)),
                negative_fraction=float(np.mean(approx < 0))))
    print(f"finished {m['bands'][band]['center_nm']} nm", flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(dict(asset=str(a.asset), config=c, heights_indices=heights,
    dtype="float32", scope="independent on-grid angular slabs, not held-out transport queries",
    relative_error_floor="exclude values below 1e-5 of each slab peak", results=results), indent=2))
