"""CPU-only fixed-atmosphere, moving-Sun candidate/trajectory preparation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are rejected')
parser.add_argument('--baseline-config', type=Path, default=ROOT / 'crates/sky-realtime/configs/balanced.json')
args = parser.parse_args()
OUT = args.out
if OUT.exists():
    raise FileExistsError('Output exists; choose a new directory to preserve frozen experiment inputs')
OUT.mkdir(parents=True)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


baseline_path = OUT / "baseline_config.json"
baseline = json.loads((baseline_path if baseline_path.exists() else args.baseline_config).read_text())
if not baseline_path.exists():
    write("baseline_config.json", baseline)
for source, target in [
    ("crates/sky-realtime/configs/mapping_fit.json", "baseline_mapping_fit.json"),
    ("crates/sky-realtime/configs/wavelengths.json", "baseline_wavelengths.json"),
]:
    if not (OUT / target).exists():
        (OUT / target).write_bytes((ROOT / source).read_bytes())
snapshot = OUT / "baseline_source"
snapshot.mkdir(exist_ok=True)
for name in ["solver.wgsl", "transport.wgsl", "sky_view.wgsl"]:
    if not (snapshot / name).exists():
        (snapshot / name).write_bytes((ROOT / "crates/sky-realtime/src" / name).read_bytes())

candidates = []
for name, size, steps, priority in [
    ("baseline", 256, 96, 0),
    ("sky224", 224, 96, 1),
    ("sky192", 192, 96, 1),
    ("steps80", 256, 80, 1),
    ("steps72", 256, 72, 1),
    ("steps64", 256, 64, 2),
    ("sky224_steps72", 224, 72, 2),
    ("sky192_steps80", 192, 80, 2),
    ("sky192_steps72", 192, 72, 3),
]:
    config = dict(baseline, sky_size=size)
    filename = f"{name}_config.json"
    if name != "baseline":
        write(filename, config)
    ratio = (size / 256) ** 2 * steps / 96
    candidates.append({
        "name": name,
        "config_path": str(filename),
        "runtime_steps": steps,
        "priority": priority,
        "changed_config_fields": {} if size == 256 else {"sky_size": size},
        "sky_chart_rows": [size * 3 // 8, size * 3 // 8, size // 4],
        "sky_texels": size * size,
        "vector_transport_steps_per_sky_update": size * size * steps,
        "wavelength_step_components": size * size * steps * 4,
        "sky_transport_work_ratio": ratio,
        "sky_transport_work_reduction_percent": (1 - ratio) * 100,
        "sky_rgba32f_bytes": size * size * 16,
        "observer_cache_rows": size,
        "cpu_inverse_bisection_steps": size * 28,
        "precomputed_source_dimensions_unchanged": True,
        "status": "CPU work estimate; no GPU timing or quality result",
    })

# A fixed camera views pure sky. The 400 km trajectory includes atmospheric
# shell and vacuum, but stays above the inner ground horizon.
trajectories = []
for name, altitude, yaw, pitch, fov, start, end, az_start, az_end in [
    ("daylight", 0.2, 0, 83, 20, 82, 88.4, 0, 0),
    ("horizon_crossing", 0.2, 0, 8.2, 30, 0.5, -0.5, 0, 0),
    ("blue_forward", 0.2, 0, 8.2, 30, -3, -9.4, 0, 0),
    ("blue_backward", 0.2, 180, 8.2, 30, -3, -9.4, 0, 0),
    ("high_shadow", 12, 0, 5.2, 30, -7, -11, 0, 0),
    ("thin_limb", 108, 0, -1.8, 30, -12, -18.4, 0, 0),
    ("orbital_limb", 400, 0, -11.2, 30, -18, -24.4, 0, 0),
    ("azimuth_crossing", 0.2, 0, 30, 90, 8, 8, -1.6, 1.6),
]:
    frames = []
    for frame in range(65):
        t = frame / 64
        frames.append({
            "frame": frame,
            "sun_elevation_deg": start * (1 - t) + end * t,
            "sun_azimuth_deg": az_start * (1 - t) + az_end * t,
        })
    trajectories.append({
        "name": name, "altitude_km": altitude, "camera_yaw_deg": yaw,
        "pitch_deg": pitch, "horizontal_fov_deg": fov,
        "frames": frames, "fixed_camera": True,
    })

def queries(width, height, selected=None):
    images = []
    for trajectory in trajectories:
        for frame in trajectory["frames"]:
            if selected is not None and (trajectory["name"], frame["frame"]) not in selected:
                continue
            images.append({
                "name": f"{trajectory['name']}_{frame['frame']:03}",
                "width": width, "height": height,
                # evaluate fixes the Sun azimuth at zero. Rotational symmetry
                # maps a moving Sun to this relative camera yaw exactly.
                "yaw": trajectory["camera_yaw_deg"] - frame["sun_azimuth_deg"],
                "pitch": trajectory["pitch_deg"],
                "horizontal_fov": trajectory["horizontal_fov_deg"],
                "sun_elevation_deg": frame["sun_elevation_deg"],
                "altitude_km": trajectory["altitude_km"],
            })
    return {"images": images}

write("trajectories.json", {
    "coordinate_interpretation": "fixed camera, continuously moving Sun; audit evaluate uses equivalent relative yaw with Sun azimuth zero",
    "frame_interval_seconds": 1 / 60,
    "trajectories": trajectories,
})
write("profile_trajectories.json", {"trajectories": [
    {"name": t["name"], "altitude_km": t["altitude_km"],
     "yaw_deg": t["camera_yaw_deg"], "pitch_deg": t["pitch_deg"],
     "fov_y_deg": math.degrees(2 * math.atan(math.tan(math.radians(t["horizontal_fov_deg"]) / 2) * 1080 / 1920)),
     "sun_elevation_start_deg": t["frames"][0]["sun_elevation_deg"],
     "sun_elevation_end_deg": t["frames"][-1]["sun_elevation_deg"],
     "sun_azimuth_start_deg": t["frames"][0]["sun_azimuth_deg"],
     "sun_azimuth_end_deg": t["frames"][-1]["sun_azimuth_deg"],
     "exposure": {"daylight": .1, "horizon_crossing": .5, "blue_forward": 30,
                  "blue_backward": 30, "high_shadow": .5, "thin_limb": .5,
                  "orbital_limb": .5, "azimuth_crossing": .5}[t["name"]]}
    for t in trajectories]})
write("queries_temporal.json", queries(320, 180))
# Preserve the actual accepted audit's 14 inputs, including the six worst
# continuous frames. They are independent of future default-budget changes.
write("queries_1080_anchors.json", json.loads((ROOT / 'experiments/validation/moving_sun_1080_queries.json').read_text()))
write("profile_1080_anchors.json", json.loads((ROOT / 'experiments/validation/moving_sun_1080_anchors.json').read_text()))
write("plan.json", {
    "objective": "Fixed atmosphere; continuously moving Sun; complete 1920x1080 pure-sky frame cost, with small smooth quality changes permitted",
    "candidates": candidates,
    "runtime_steps_are_renderer_steps_not_config_ray_steps": True,
    "source_mapping_and_four_wavelengths": "Frozen current near-quality baseline; refitting deferred until budget quality establishes a need",
    "performance": {
        "expected_complete_frame_speedup": "Measure SkyView fraction f of baseline complete frame; ideal reduction is f*(1-sky_transport_work_ratio), before CPU cache and scheduling differences",
        "projection_pixels": 1920 * 1080,
        "ordinary_projection_texture_loads": 4 * 1920 * 1080,
        "hdr_target_write_bytes": 16 * 1920 * 1080,
        "display_hdr_read_bytes": 16 * 1920 * 1080,
        "display_rgba8_write_bytes": 4 * 1920 * 1080,
        "output_resolution_io_is_unchanged_by_budget_candidates": True,
        "source_dimensions": "Lower source dimensions mainly affect startup/storage; lookup stencil remains, so do not spend first dynamic-performance budget here",
    },
    "quality": {
        "phase1": "520 continuous frames at 320x180; current 256/96 baseline, per-candidate static and temporal residual metrics",
        "phase2": "Fourteen frozen 1920x1080 sensitive/worst anchors for the selected candidates; actual display and signed residual/heatmap inspection",
        "reference": "256/96 SkyView is the current near-quality baseline. Its direct-source render helps separate changed integration from changed interpolation; neither is an unbiased oracle",
        "static": "Relative RGB norm P50/P95/P99/max, luminance and chromatic changes; report illuminated-sky subset and all sky separately",
        "temporal": "First residual difference (C[t]-C[t-1])-(B[t]-B[t-1]), normalized by max baseline norm over both frames; second residual difference for spikes; fixed per-trajectory luminance floor",
        "stripes": "Row residual means and adjacent-row differences, worst row/local heatmap; do not accept global percentiles alone",
        "acceptance": "Permit small smoothly varying residual. Reject visible coherent bands, horizon/limb discontinuities or one-frame jumps. Numerical thresholds are diagnostics until inspecting full-resolution worst images",
        "sun_disk": "Audit evaluate excludes the direct disk; profile should report complete sky/display behavior with its chosen disk setting",
    },
    "baseline_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in ["baseline_config.json", "baseline_mapping_fit.json", "baseline_wavelengths.json"]},
    "gpu_initialized": False,
})
print(f"Prepared {len(candidates)} candidates, {len(trajectories)} continuous trajectories, 520 temporal frames and 14 frozen HD anchors; CPU only.")
