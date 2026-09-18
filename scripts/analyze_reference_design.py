"""Summarize the accepted reference chart probes; no rendering or GPU work.

Inputs are emitted by `cargo run --release -p sky-atmosphere-lut --example
reference_design`. Use baked_probes.csv, never the rejected corner prototypes.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def errors(values, reference):
    norm = np.linalg.norm(reference)
    return {"relative_l2_percent": float(100*np.linalg.norm(values-reference)/norm),
            "signed_mean_percent": float(100*(values.sum()/reference.sum()-1))}


def summarize(root, old_root):
    probes = read(root / "baked_probes.csv")
    groups = {name: probes["case"] == name for name in np.unique(probes["case"])}
    groups["noon_low_horizon"] = ((probes["case"] == "ground_sweep") &
        (probes["sun_deg"] >= 60) & (probes["above_horizon_deg"] <= 3))
    groups["twilight_ground"] = ((probes["case"] == "ground_sweep") &
        (probes["sun_deg"] < 0))
    first = {name: {"samples": int(mask.sum()),
                   **{version: errors(probes[key][mask], probes["direct"][mask])
                      for version, key in [("v4", "old"), ("v5", "new")]}}
             for name, mask in groups.items()}
    profiles = {"v4": read(old_root / "dense_profiles.csv"),
                "v5": read(root / "dense_profiles.csv")}
    gradients = {}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout="constrained")
    for row, sun in enumerate([60, 80, 85]):
        metrics = {}
        for name, data in profiles.items():
            p = data[(data["sun_deg"] == sun) & (data["azimuth_deg"] == 0)]
            x, value = p["above_horizon_deg"], p["full"]
            slope = 100*np.diff(np.log(value))/np.diff(x)
            mid = (x[:-1]+x[1:])*.5
            mask = (mid[1:] >= 1) & (mid[1:] <= 10)
            jump = abs(np.diff(slope))[mask]
            metrics[name] = {"max_slope_jump": float(jump.max()),
                             "p99_slope_jump": float(np.percentile(jump, 99))}
            axes[row, 0].plot(x, value, label=name, linewidth=1.4)
            axes[row, 1].plot(mid, slope, label=name, linewidth=1.0)
        gradients[str(sun)] = metrics
        axes[row, 0].set_ylabel(f"Sun {sun} deg\n550 nm radiance")
        axes[row, 1].set_ylabel("100 d(log L)/d(deg)")
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.set_xlim(0, 10)
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Degrees above geometric horizon")
    axes[0, 0].set_title("16-order field; observer 200 m, solar azimuth")
    axes[0, 1].set_title("Derivative continuity (not radiance error)")
    fig.savefig(root / "noon_gradients_v4_v5.png", dpi=160)
    plt.close(fig)

    tau = read(root / "optical_depth.csv")
    valid = tau["tau_direct"] < 10
    trans = {name: {"max_relative_transmittance_percent": float(100*np.max(abs(
                np.exp(tau["tau_direct"][valid]-tau[key][valid])-1)))}
             for name, key in [("v4", "tau_old"), ("v5", "tau_new")]}
    phase = read(root / "phase_field.csv")
    phase_errors = {}
    for nm in np.unique(phase["nm"]):
        p = phase[phase["nm"] == nm]
        phase_errors[str(nm)] = {**errors(p["interpolated"], p["exact"]),
            "max_relative_percent": float(100*np.max(abs(p["interpolated"]/p["exact"]-1)))}
    angular = read(root / "angular_source.csv")
    keys = ["height_km", "sun_deg", "azimuth_deg", "above_horizon_deg"]
    states = np.unique(angular[keys])
    sources = []
    for state in states:
        mask = np.ones(len(angular), dtype=bool)
        for key in keys:
            mask &= angular[key] == state[key]
        p = angular[mask]
        baseline = p[(p["n_mu"] == 128) & (p["n_phi"] == 256)]["source"][0]
        sources.append({**{k: float(state[k]) for k in keys},
            "relative_percent_vs_128x256": {
                f'{r["n_mu"]}x{r["n_phi"]}': float(100*(r["source"]/baseline-1))
                for r in p}})
    output = {
        "note": "550 nm selected probes, not a global transport error bound. First-order direct baseline uses the old optical-depth table. Frozen-field angular tests do not rebake transport. PT noise is absent from these diagnostics.",
        "first_order": first, "full_field_gradient_jumps": gradients,
        "optical_depth": {"paths": len(tau), "tau_below_10": int(valid.sum()), **trans},
        "solar_disk_averaged_phase": phase_errors, "frozen_angular_source": sources,
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ["first_order", "full_field_gradient_jumps", "optical_depth"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out/lut_reference_design"))
    parser.add_argument("--baseline", type=Path, default=Path("out/lut_sampling_audit"))
    args = parser.parse_args()
    summarize(args.input, args.baseline)
