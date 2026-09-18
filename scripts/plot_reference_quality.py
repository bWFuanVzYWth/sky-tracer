"""CPU assessment figures and concise machine-readable acceptance evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("out/lut_v5_cpu_quality")
groups = json.loads((root / "groups.json").read_text())
rgb = np.fromfile(root / "dataset/samples.f32", dtype="<f4").reshape(-1, 3)
residual = json.loads((root / "residual_suspects.json").read_text())["probes"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for col, (height, bounds) in enumerate([(30, (-5.9, -5.35)), (120, (-11.2, -10.8))]):
    group = next(g for g in groups if g["kind"] == "solar" and
                 (g["altitude"], g["above"], g["azimuth"]) == (height, .1, 0))
    values = rgb[group["start"]:group["start"]+group["count"]]
    x = np.linspace(-18, 90, group["count"])
    axes[0, col].plot(x, values @ np.array([.2627, .678, .0593]), ".-", label="v5 spectral -> Rec.2020 Y")
    axes[0, col].set(xlim=bounds, ylim=(0, .6 if height == 30 else 7),
                    title=f"{height} km; 0.1 deg above horizon; solar azimuth",
                    xlabel="Solar elevation [deg]", ylabel="Linear Rec.2020 luminance")
    probes = sorted([r for r in residual if r["altitude_km"] == height and
                     r["sun_deg"] < 0 and r["nm"] == 550], key=lambda r:r["sun_deg"])
    x = np.array([r["sun_deg"] for r in probes])
    for key, label in [("stored", "Stored total L"), ("reconstructed", "Direct + K[L], CPU"), ("direct", "Direct single scattering")]:
        axes[1, col].plot(x, [r[key] for r in probes], "o-", label=label)
    axes[1, col].set(xlabel="Solar elevation [deg]", ylabel="550 nm band radiance [W m-2 sr-1]",
                    title="1024 ray steps; 2 x 64 x 128 directions; 8 x 32 solar disc")
for ax in axes.flat:
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.savefig(root / "high_altitude_solar_failure.png", dpi=170)
plt.close(fig)

high = json.loads((root / "pt_high_spp_means.json").read_text())["results"]
pt_means = {}
for region in high[0]["regions"]:
    refs = np.array([s["regions"][region]["mean_pt"] for s in high])
    lut = high[0]["regions"][region]["mean_lut"]
    pt_means[region] = dict(mean_bias_percent=float(100*(lut/refs.mean()-1)),
        pt_seed_standard_deviation_percent=float(100*refs.std(ddof=1)/refs.mean()),
        pt_mean_standard_error_percent=float(100*refs.std(ddof=1)/np.sqrt(len(refs))/refs.mean()),
        samples_per_pixel=high[0]["spp"], seeds=[h["seed"] for h in high],
        pixels=high[0]["regions"][region]["pixels"])
a = json.loads((root / "residual_256_32x64.json").read_text())["probes"]
b = json.loads((root / "residual_512_64x128.json").read_text())["probes"]
refinement = max(abs(x["relative_residual_percent"]-y["relative_residual_percent"]) for x, y in zip(a,b))
suspects = {str(nm): [r for r in residual if r["nm"] == nm] for nm in [440, 550, 680]}
summary = dict(
    decision="Suitable for pipeline/compression experiments on checked subsets; not accepted as an all-time/all-altitude high-accuracy teacher",
    high_spp_pt_regional_means=pt_means,
    max_residual_change_on_refining_256_32x64_to_512_64x128_percentage_points=refinement,
    suspect_min_residual_percent=min(r["relative_residual_percent"] for r in residual),
    suspect_max_residual_percent=max(r["relative_residual_percent"] for r in residual),
    suspects=suspects,
    note="Residual is frozen-field consistency, not total error. PT mean standard error is descriptive from only three seeds. No new GPU work.")
(root / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "suspects"}, indent=2))
