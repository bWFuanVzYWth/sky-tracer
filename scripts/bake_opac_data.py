"""Bake libRadtran/OPAC data into CSV files for the reference path tracer.

The generated data are intentionally plain CSV so the Rust renderer can load
frozen, inspectable tables without depending on libRadtran, scipy, or
miepython at runtime. Re-run this script when changing the spectral bands.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import miepython
import numpy as np
from scipy.io import netcdf_file


SPECIES = {
    "inso": {
        "ri_file": "inso00_refr.dat",
        "rh": 0,
        "r_mode_um": 0.471,
        "sigma_g": 2.51,
        "r_min_um": 0.005,
        "r_max_um": 20.0,
    },
    "waso": {
        "ri_file": "waso50_refr.dat",
        "rh": 50,
        "r_mode_um": 0.0262,
        "sigma_g": 2.24,
        "r_min_um": 0.006,
        "r_max_um": 25.0,
    },
    "soot": {
        "ri_file": "soot00_refr.dat",
        "rh": 0,
        "r_mode_um": 0.0118,
        "sigma_g": 2.0,
        "r_min_um": 0.005,
        "r_max_um": 20.0,
    },
    "suso": {
        "ri_file": "suso50_refr.dat",
        "rh": 50,
        "r_mode_um": 0.0983,
        "sigma_g": 2.03,
        "r_min_um": 0.0073,
        "r_max_um": 30.2,
    },
}


def band_edges(n_bands: int, start_nm: float, end_nm: float):
    edges = np.linspace(start_nm, end_nm, n_bands + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges[:-1], edges[1:]


def read_columns(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=np.float64)


def interp_integral(xs: np.ndarray, ys: np.ndarray, lo: float, hi: float, samples: int = 32) -> float:
    grid = np.linspace(lo, hi, samples)
    vals = np.interp(grid, xs, ys)
    return float(np.trapezoid(vals, grid))


def ozone_cross_section(path: Path, center_nm: float) -> float:
    data = read_columns(path)
    wl = data[:, 0]
    c0 = data[:, 1]
    # The file stores SIGMA = C0 * 1e-20 cm^2 at T0. This is enough for the
    # first scalar reference pass; the profile still stores temperature for a
    # future temperature-dependent absorption evaluator.
    return float(np.interp(center_nm, wl, c0) * 1.0e-20)


def load_refractive_index(path: Path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2]


def interp_ri(target_um: float, lam_um: np.ndarray, n: np.ndarray, k: np.ndarray) -> complex:
    return complex(float(np.interp(target_um, lam_um, n)), float(np.interp(target_um, lam_um, k)))


def cube_root_mu_grid(n_bins: int) -> np.ndarray:
    u = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    return 1.0 - 2.0 * u**3


def lognormal_radii(defn: dict[str, float], n_radii: int):
    r = np.geomspace(defn["r_min_um"], defn["r_max_um"], n_radii)
    log_sigma = np.log(defn["sigma_g"])
    weights = np.exp(-(np.log(r / defn["r_mode_um"])) ** 2 / (2.0 * log_sigma**2))
    weights /= np.sum(weights)
    return r, weights


def size_dist_phase(m: complex, lam_um: float, r_um: np.ndarray, weights: np.ndarray, mu: np.ndarray):
    p_acc = np.zeros_like(mu)
    sigma_acc = 0.0
    for r_i, w_i in zip(r_um, weights):
        x = 2.0 * np.pi * r_i / lam_um
        _qext, qsca, _qback, _g = miepython.efficiencies(m, 2.0 * r_i, lam_um)
        sigma_sca = np.pi * r_i**2 * qsca
        p_r = miepython.i_unpolarized(m, x, mu, norm="one")
        p_acc += w_i * sigma_sca * p_r
        sigma_acc += w_i * sigma_sca
    return p_acc / sigma_acc


def opac_optics(opac_dir: Path, species_name: str, rh: int, center_nm: float):
    nc_path = opac_dir / "optprop" / f"{species_name}.mie.cdf"
    nc = netcdf_file(str(nc_path), "r", mmap=False)
    lam_um = np.asarray(nc.variables["wavelen"][:]).copy()
    hum = np.asarray(nc.variables["hum"][:]).copy()
    ext = np.asarray(nc.variables["ext"][:]).copy()
    ssa = np.asarray(nc.variables["ssa"][:]).copy()
    nc.close()
    rh_idx = int(np.argmin(np.abs(hum - rh)))
    lam = center_nm / 1000.0
    ext_t = float(np.interp(lam, lam_um, ext[:, rh_idx]))
    ssa_t = float(np.interp(lam, lam_um, ssa[:, rh_idx]))
    return ext_t * ssa_t, ext_t * (1.0 - ssa_t)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def bake(args):
    root = Path(args.libradtran).resolve()
    out = Path(args.out).resolve()
    opac = root / "data/aerosol/OPAC"

    centers, lows, highs = band_edges(args.bands, args.start_nm, args.end_nm)
    solar = read_columns(root / "data/solar_flux/rgb")
    solar_nm = solar[:, 0]
    solar_mw_m2_nm = solar[:, 1]
    ozone_path = root / "data/crs/crs_o3_mol_cf.dat"

    band_rows = [
        [
            i,
            f"{centers[i]:.8f}",
            f"{lows[i]:.8f}",
            f"{highs[i]:.8f}",
            # libRadtran rgb is mW/(m^2 nm); integrate over band and convert to W/m^2.
            f"{interp_integral(solar_nm, solar_mw_m2_nm, lows[i], highs[i]) * 1.0e-3:.8e}",
            f"{ozone_cross_section(ozone_path, centers[i]):.8e}",
        ]
        for i in range(args.bands)
    ]
    write_csv(out / "bands.csv", band_rows)

    atm = read_columns(root / "data/atmmod/afglus.dat")
    atm_rows = [
        [f"{row[0]:.8f}", f"{row[2]:.8e}", f"{row[3]:.8e}", f"{row[4]:.8e}"]
        for row in atm
        if 0.0 <= row[0] <= 120.0
    ]
    write_csv(out / "atmosphere_profile.csv", atm_rows)

    aerosol_rows = []
    for row in read_columns(opac / "standard_aerosol_files/continental_average.dat"):
        # source order: z, inso, waso, soot, suso
        aerosol_rows.append([f"{row[0]:.8f}", f"{row[1]:.8e}", f"{row[2]:.8e}", f"{row[3]:.8e}", f"{row[4]:.8e}"])
    write_csv(out / "aerosol_profile.csv", aerosol_rows)

    optics_rows = []
    for band_idx, center in enumerate(centers):
        for name, defn in SPECIES.items():
            sca, abs_ = opac_optics(opac, name, defn["rh"], center)
            optics_rows.append([name, band_idx, f"{sca:.8e}", f"{abs_:.8e}"])
    write_csv(out / "aerosol_optics.csv", optics_rows)

    mu = cube_root_mu_grid(args.phase_bins)
    phase_rows = []
    for name, defn in SPECIES.items():
        lam_grid, n_grid, k_grid = load_refractive_index(opac / "refractive_indices" / defn["ri_file"])
        radii, weights = lognormal_radii(defn, args.radii)
        print(f"baking phase {name}: RH={defn['rh']}%, radii={args.radii}")
        for band_idx, center in enumerate(centers):
            lam_um = center / 1000.0
            m = interp_ri(lam_um, lam_grid, n_grid, k_grid)
            phase = size_dist_phase(m, lam_um, radii, weights, mu)
            for bin_idx, value in enumerate(phase):
                phase_rows.append([name, band_idx, bin_idx, f"{float(value):.8e}"])
    write_csv(out / "mie_phase.csv", phase_rows)

    print(f"wrote baked data to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libradtran", default=r"C:\WorkSpace\cellular_automata\ref\libRadtran-2.0.6")
    parser.add_argument("--out", default="data")
    parser.add_argument("--bands", type=int, default=15)
    parser.add_argument("--phase-bins", type=int, default=1024)
    parser.add_argument("--radii", type=int, default=300)
    parser.add_argument("--start-nm", type=float, default=380.0)
    parser.add_argument("--end-nm", type=float, default=780.0)
    bake(parser.parse_args())


if __name__ == "__main__":
    main()
