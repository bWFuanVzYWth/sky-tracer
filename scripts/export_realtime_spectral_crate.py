"""Export sparse spectral constants into a realtime atmosphere crate.

The script samples the baked 10 nm OPAC/libRadtran CSV tables and rewrites the
small spectral parts of a copied realtime crate:

- src/params.rs spectral constants
- src/aerosol/{inso,waso,soot}.rs lookup tables
- src/wgsl/common.wgsl spectral-to-Rec.2020 matrix comment/body
- src/lib.rs top documentation wavelengths

Pass searched quadrature weights explicitly. Those weights are only used by the
spectral-to-Rec.2020 conversion matrix; physical atmosphere coefficients remain
sampled directly at the requested wavelength centers.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SPECIES = ["waso", "inso", "soot"]
STRUCT_NAMES = {"waso": "Waso", "inso": "Inso", "soot": "Soot"}

REC2020_FROM_XYZ_D65 = [
    [1.7166511880, -0.3556707838, -0.2533662814],
    [-0.6666843518, 1.6164812366, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.9421031212],
]


@dataclass(frozen=True)
class Band:
    index: int
    center_nm: float
    lower_nm: float
    upper_nm: float
    solar_irradiance_w_m2: float
    ozone_cross_section_cm2: float

    @property
    def width_nm(self) -> float:
        return self.upper_nm - self.lower_nm


def parse_float_list(value: str, name: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if len(values) > 4:
        raise ValueError(f"{name} supports at most 4 values, got {len(values)}")
    return values


def fmt_f32(value: float) -> str:
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    text = f"{value:.9e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    if -3 <= exponent_value < 5:
        return f"{value:.9f}".rstrip("0").rstrip(".")
    return f"{mantissa}e{exponent_value:+03d}"


def fmt_rs_array(values: list[float]) -> str:
    return "[" + ", ".join(fmt_f32(value) for value in values) + "]"


def fmt_wgsl_vec3(values: list[float]) -> str:
    if all(abs(value) <= 1.0e-20 for value in values):
        return "vec3<f32>(0.0)"
    return "vec3<f32>(" + ", ".join(fmt_f32(value) for value in values) + ")"


def read_bands(data_dir: Path) -> dict[int, Band]:
    bands = {}
    with (data_dir / "bands.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            band = Band(
                index=int(row[0]),
                center_nm=float(row[1]),
                lower_nm=float(row[2]),
                upper_nm=float(row[3]),
                solar_irradiance_w_m2=float(row[4]),
                ozone_cross_section_cm2=float(row[5]),
            )
            bands[int(round(band.center_nm))] = band
    return bands


def read_aerosol_optics(data_dir: Path) -> dict[tuple[str, int], tuple[float, float]]:
    optics = {}
    with (data_dir / "aerosol_optics.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            species = row[0]
            band_index = int(row[1])
            optics[(species, band_index)] = (float(row[2]), float(row[3]))
    return optics


def read_phase_rows(data_dir: Path, species: str, band_indices: list[int]) -> list[list[float]]:
    wanted = set(band_indices)
    by_bin = {}
    with (data_dir / "mie_phase.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if row[0] != species:
                continue
            band_index = int(row[1])
            if band_index not in wanted:
                continue
            bin_index = int(row[2])
            by_bin.setdefault(bin_index, {})[band_index] = float(row[3])
    if not by_bin:
        raise ValueError(f"no Mie phase data found for {species}")
    rows = []
    for bin_index in sorted(by_bin):
        band_values = by_bin[bin_index]
        rows.append([band_values[index] for index in band_indices])
    return rows


def read_cie(data_dir: Path) -> dict[int, tuple[float, float, float]]:
    samples = {}
    with (data_dir / "CIE_xyz_1931_2deg.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if not row or row[0].startswith("#"):
                continue
            wavelength = int(round(float(row[0])))
            samples[wavelength] = (float(row[1]), float(row[2]), float(row[3]))
    return samples


def rayleigh_cross_section_m2(wavelength_nm: float) -> float:
    return 5.8e-31 * (550.0 / wavelength_nm) ** 4


def sea_level_air_cm3(data_dir: Path) -> float:
    value = None
    with (data_dir / "atmosphere_profile.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if abs(float(row[0])) <= 1.0e-6:
                value = float(row[2])
    if value is None:
        raise ValueError("atmosphere_profile.csv does not contain altitude 0 km")
    return value


def padded(values: list[float]) -> list[float]:
    return values + [0.0] * (4 - len(values))


def replace_const_array(source: str, name: str, values: list[float]) -> str:
    pattern = re.compile(rf"pub const {name}: \[f32; 4\] = \[[^\]]*\];", re.S)
    replacement = f"pub const {name}: [f32; 4] = {fmt_rs_array(values)};"
    return pattern.sub(replacement, source, count=1)


def update_params(crate_dir: Path, wavelengths: list[float], bands: dict[int, Band], data_dir: Path) -> None:
    active_bands = [bands[int(round(wavelength))] for wavelength in wavelengths]
    solar = [band.solar_irradiance_w_m2 / band.width_nm for band in active_bands]
    ozone = [band.ozone_cross_section_cm2 * 1.0e-4 for band in active_bands]
    air_cm3 = sea_level_air_cm3(data_dir)
    molecular = [
        air_cm3 * rayleigh_cross_section_m2(wavelength) * 1.0e9 for wavelength in wavelengths
    ]

    path = crate_dir / "src" / "params.rs"
    source = path.read_text(encoding="utf-8")
    wave_text = "/".join(f"{int(round(wavelength))}" for wavelength in wavelengths)
    if len(wavelengths) < 4:
        wavelength_comment = (
            "/// Wavelength order used by every 4-channel spectral GPU constant.\n"
            "///\n"
            f"/// RGB store the {wave_text} nm samples; A is intentionally unused\n"
            "/// and kept at zero so the copied renderer can preserve the existing RGBA ABI.\n"
        )
    else:
        wavelength_comment = "/// Wavelength order used by every 4-channel spectral GPU constant.\n"
    source = re.sub(
        r"/// Wavelength order used by every 4-channel spectral GPU constant\.\n(?:///.*\n)*",
        wavelength_comment,
        source,
        count=1,
    )
    source = replace_const_array(source, "SPECTRAL_SAMPLE_WAVELENGTHS_NM", padded(wavelengths))
    source = replace_const_array(source, "SUN_SPECTRAL_IRRADIANCE", padded(solar))
    source = replace_const_array(source, "MOLECULAR_SCATTERING_BASE", padded(molecular))
    source = replace_const_array(source, "OZONE_ABSORPTION_CROSS_SECTION", padded(ozone))
    path.write_text(source, encoding="utf-8", newline="\n")


def update_lib_doc(crate_dir: Path, wavelengths: list[float], weight_strategy: str) -> None:
    path = crate_dir / "src" / "lib.rs"
    source = path.read_text(encoding="utf-8")
    wave_text = "/".join(f"{int(round(wavelength))}" for wavelength in wavelengths)
    active_count = len(wavelengths)
    if active_count == 3:
        replacement = (
            "//! This crate keeps the copied Unreal/Hillaire LUT structure but samples only\n"
            f"//! the {weight_strategy} {wave_text} nm wavelengths. The fourth RGBA channel is kept\n"
            "//! unused so the renderer can preserve the original GPU ABI."
        )
    else:
        replacement = (
            "//! This crate keeps the copied Unreal/Hillaire LUT structure but replaces the\n"
            f"//! baseline four channels with the {weight_strategy} {wave_text} nm wavelengths."
        )
    source = re.sub(
        r"//! This crate keeps the copied Unreal/Hillaire LUT structure but .*?(?=\n\npub mod aerosol;)",
        replacement,
        source,
        count=1,
        flags=re.S,
    )
    path.write_text(source, encoding="utf-8", newline="\n")


def spectral_to_rec2020_columns(wavelengths: list[float], weights: list[float], cie: dict[int, tuple[float, float, float]]) -> list[list[float]]:
    columns = []
    for wavelength, weight in zip(wavelengths, weights):
        xyz = cie[int(round(wavelength))]
        columns.append([
            sum(REC2020_FROM_XYZ_D65[row][col] * xyz[col] for col in range(3)) * weight
            for row in range(3)
        ])
    while len(columns) < 4:
        columns.append([0.0, 0.0, 0.0])
    return columns


def update_common_wgsl(crate_dir: Path, wavelengths: list[float], weights: list[float], weight_strategy: str, cie: dict[int, tuple[float, float, float]]) -> None:
    path = crate_dir / "src" / "wgsl" / "common.wgsl"
    source = path.read_text(encoding="utf-8")
    wave_text = "/".join(f"{int(round(wavelength))}" for wavelength in wavelengths)
    weight_text = "/".join(f"{weight:.6g}" for weight in weights)
    alpha_note = " The alpha channel is unused." if len(wavelengths) < 4 else ""
    columns = spectral_to_rec2020_columns(wavelengths, weights, cie)
    matrix_lines = ",\n".join(f"        {fmt_wgsl_vec3(column)}" for column in columns)
    replacement = (
        "fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {\n"
        f"    // {wave_text} nm -> scene-linear Rec.2020. The columns are direct CIE 1931\n"
        "    // 2 degree CMF samples transformed to Rec.2020 and multiplied by the\n"
        f"    // {weight_strategy} fixed-sum quadrature weights:\n"
        f"    // {weight_text} nm.{alpha_note}\n"
        "    let m = mat4x3<f32>(\n"
        f"{matrix_lines},\n"
        "    );\n"
        "    return m * l;\n"
        "}\n"
    )
    source = re.sub(
        r"fn linear_rec2020_from_spectral\(l: vec4<f32>\) -> vec3<f32> \{.*?\n\}\n\nfn white_balance_rec2020",
        replacement + "\nfn white_balance_rec2020",
        source,
        count=1,
        flags=re.S,
    )
    white_comment = (
        "fn white_balance_rec2020(rgb: vec3<f32>) -> vec3<f32> {\n"
        "    // Bradford 41-band solar-white-to-D65 adaptation expressed in Rec.2020 RGB.\n"
        "    // This matches the offline reference colorimetry instead of neutralizing\n"
        f"    // the sparse {wave_text} nm solar samples."
    )
    source = re.sub(
        r"fn white_balance_rec2020\(rgb: vec3<f32>\) -> vec3<f32> \{\n(?:    //.*\n)+",
        white_comment + "\n",
        source,
        count=1,
    )
    path.write_text(source, encoding="utf-8", newline="\n")


def render_aerosol_file(
    species: str,
    active_count: int,
    sigma_sca: list[float],
    sigma_abs: list[float],
    phase_rows: list[list[float]],
) -> str:
    struct_name = STRUCT_NAMES[species]
    lines = [
        "use crate::aerosol::{Aerosol, PHASE_LUT_COS_BINS, PHASE_LUT_WAVELENGTHS};",
        "",
        f"pub struct {struct_name};",
        "",
        "#[expect(clippy::unreadable_literal, clippy::excessive_precision)]",
        f"impl Aerosol for {struct_name} {{",
        "    const SIGMA_SCA: [f32; PHASE_LUT_WAVELENGTHS] =",
        f"        {fmt_rs_array(padded(sigma_sca))};",
        "",
        "    const SIGMA_ABS: [f32; PHASE_LUT_WAVELENGTHS] =",
        f"        {fmt_rs_array(padded(sigma_abs))};",
        "",
        "    const PHASE_LUT: &[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS] = &[",
    ]
    for row in phase_rows:
        values = row + [0.0] * (4 - active_count)
        lines.append(f"        {fmt_rs_array(values)},")
    lines.extend(["    ];", "}"])
    return "\n".join(lines) + "\n"


def update_aerosol(crate_dir: Path, wavelengths: list[float], bands: dict[int, Band], data_dir: Path) -> None:
    active_bands = [bands[int(round(wavelength))] for wavelength in wavelengths]
    band_indices = [band.index for band in active_bands]
    optics = read_aerosol_optics(data_dir)
    for species in SPECIES:
        sigma_sca = [optics[(species, index)][0] for index in band_indices]
        sigma_abs = [optics[(species, index)][1] for index in band_indices]
        phase_rows = read_phase_rows(data_dir, species, band_indices)
        source = render_aerosol_file(species, len(wavelengths), sigma_sca, sigma_abs, phase_rows)
        (crate_dir / "src" / "aerosol" / f"{species}.rs").write_text(
            source,
            encoding="utf-8",
            newline="\n",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crate-dir", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--wavelengths", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weight-strategy", default="cmf-mass")
    args = parser.parse_args()

    crate_dir = args.crate_dir.resolve()
    data_dir = args.data_dir.resolve()
    wavelengths = parse_float_list(args.wavelengths, "--wavelengths")
    weights = parse_float_list(args.weights, "--weights")
    if len(wavelengths) != len(weights):
        raise ValueError("--wavelengths and --weights must have the same length")

    bands = read_bands(data_dir)
    for wavelength in wavelengths:
        key = int(round(wavelength))
        if key not in bands or abs(bands[key].center_nm - wavelength) > 1.0e-4:
            raise ValueError(f"{wavelength} nm is not present in {data_dir / 'bands.csv'}")

    cie = read_cie(data_dir)
    for wavelength in wavelengths:
        if int(round(wavelength)) not in cie:
            raise ValueError(f"{wavelength} nm is not present in {data_dir / 'CIE_xyz_1931_2deg.csv'}")

    update_params(crate_dir, wavelengths, bands, data_dir)
    update_aerosol(crate_dir, wavelengths, bands, data_dir)
    update_common_wgsl(crate_dir, wavelengths, weights, args.weight_strategy, cie)
    update_lib_doc(crate_dir, wavelengths, args.weight_strategy)

    wave_text = "/".join(str(int(round(wavelength))) for wavelength in wavelengths)
    print(f"exported {args.weight_strategy} {wave_text} nm constants to {crate_dir}")


if __name__ == "__main__":
    main()
