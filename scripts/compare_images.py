"""Compare audit outputs in linear Rec.2020; no GPU and no display transform."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scenes = json.loads(args.queries.read_text())["images"]
    results = []
    for scene in scenes:
        for mode in ("source", "sky"):
            filename = f'{scene["name"]}_{mode}.f32'
            expected = int(scene["width"]) * int(scene["height"]) * 4
            a, b = (np.fromfile(path / filename, dtype="<f4") for path in (args.reference, args.candidate))
            if a.size != expected or b.size != expected or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Invalid dimensions or nonfinite values: {filename}")
            exact = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
            a, b = a.reshape(-1, 4)[:, :3], b.reshape(-1, 4)[:, :3]
            norm = np.linalg.norm(a, axis=1)
            delta = np.linalg.norm(b - a, axis=1)
            values = 100 * delta[norm > 1e-8] / np.maximum(norm[norm > 1e-8], 1e-7)
            stats = dict(zip(("p50", "p95", "p99", "max"), map(float, np.quantile(values, [.5, .95, .99, 1])))) if values.size else None
            results.append(dict(scene=scene["name"], mode=mode, bit_identical=exact,
                                valid_pixels=int(values.size), relative_percent=stats,
                                maximum_absolute=float(delta.max())))
    output = dict(reference=str(args.reference), candidate=str(args.candidate), queries=str(args.queries),
                  definition="linear Rec.2020 RGB norm; relative denominator >=1e-7; valid reference norm >1e-8",
                  all_bit_identical=all(row["bit_identical"] for row in results), results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"images": len(results), "all_bit_identical": output["all_bit_identical"],
                      "maximum_absolute": max(row["maximum_absolute"] for row in results)}, indent=2))

if __name__ == "__main__":
    main()
