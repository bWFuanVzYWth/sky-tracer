"""Summarize measured GPU pilots and reused high-spp PT means; no rendering."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "out/lut_v6_design"
PILOTS = [
    ("v6a", "ray", "ray_full_residual_refined"),
    ("v6b", "log", "log_full_residual_refined"),
    ("v6c", "angular32", "angular32_full_residual_refined"),
    ("v6d", "fixed", "fixed_full_residual_refined"),
    ("v6e", "cubic", "cubic_full_residual_refined"),
    ("v6f", "layers", "layers_full_residual_refined"),
    ("v6g", "layers_cdf", "layers_cdf_full_residual_refined"),
]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

results = []
for name, prefix, residual in PILOTS:
    manifest_path = ROOT / f"out/lut_{name}/asset.json"
    if not manifest_path.exists():
        continue
    m = read(manifest_path)
    record = m["records"][17]
    if record is None:
        continue
    row = dict(name=name, source=str(manifest_path.parent), config=m["config"],
               orders=len(record["orders"]),
               order_seconds=sum(o["elapsed_seconds"] for o in record["orders"]),
               local_last=record["orders"][-1].get("max_local_relative_increment"),
               stopped_by_tolerance=record["stopped_by_tolerance"])
    means = FOLDER / f"{prefix}_pt_means.json"
    if means.exists():
        samples = read(means)["results"]
        row["pt_bias_percent"] = {
            k: 100 * (samples[0]["regions"][k]["mean_lut"] /
                      np.mean([s["regions"][k]["mean_pt"] for s in samples]) - 1)
            for k in ["sky", "horizon", "earth_shadow", "solar_aureole"]}
    probe_file = FOLDER / f"{residual}.json"
    if probe_file.exists():
        row["residuals"] = read(probe_file)["probes"]
    results.append(row)

benchmark = read(FOLDER / "reuse_benchmark.json")
seconds = {flag: np.mean([sum(o["elapsed_seconds"] for o in r["orders"])
                         for r in benchmark["runs"] if r["reuse"] == flag])
           for flag in [False, True]}
summary = dict(pilots=results, reuse_benchmark={
    "dense_seconds": seconds[False], "reuse_seconds": seconds[True],
    "time_reduction_percent": 100 * (1 - seconds[True] / seconds[False]),
    "different_texels": [r["different_texels"] for r in benchmark["runs"]]},
    note="550 nm diagnostics. PT: three independent 262144 spp seeds, 16x16 LUT pixel integration. Residuals are frozen-field consistency tests, not global bounds. Timings include only recorded orders, not preparation/tau/IO; pilot timings are not all controlled ABBA benchmarks.")
(FOLDER / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

complete = [r for r in results if "pt_bias_percent" in r]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
x = np.arange(len(complete))
for i, (region, color) in enumerate([("sky", "#286090"), ("horizon", "#bd562c"), ("earth_shadow", "#447e50")]):
    axes[0].bar(x + (i-1)*.25, [r["pt_bias_percent"][region] for r in complete], .24, label=region.replace("_", " "), color=color)
axes[0].set_xticks(x, [r["name"] for r in complete])
axes[0].set(title="Twilight: bias against regional PT means", ylabel="Mean radiance bias (%)")
axes[0].axhline(0, color="black", lw=.6)
axes[0].legend(fontsize=8)
for r in complete:
    if r["name"] not in ["v6a", "v6b", "v6f", "v6g"] or "residuals" not in r:
        continue
    points = [p for p in r["residuals"] if p["altitude_km"] == 120]
    axes[1].plot([p["sun_deg"] for p in points], [p["relative_residual_percent"] for p in points], "o-", label=r["name"])
axes[1].set(title="120 km, grazing Sun: frozen-field residual", xlabel="Solar elevation (degrees)", ylabel="Residual (%)")
axes[1].ticklabel_format(useOffset=False)
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle("550 nm pilot quality — not full-spectrum acceptance", fontsize=12)
fig.savefig(FOLDER / "pilot_quality.png", dpi=160)
plt.close(fig)
for r in complete:
    print(r["name"], r["orders"], round(r["order_seconds"], 2), r["pt_bias_percent"])
