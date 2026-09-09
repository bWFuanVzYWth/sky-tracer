"""从 mie_phase.csv 和 aerosol_optics.csv 提取 suso 物种数据。"""

import csv
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# 8 工程波长在 bands.csv 中的索引
LOW_IDX  = [3, 6, 10, 14]   # 410, 440, 480, 520 nm
HIGH_IDX = [16, 20, 23, 29]  # 540, 580, 610, 670 nm

def load_mie_phase(species: str) -> dict[tuple[int, int], str]:
    """返回 {(wl_idx, cos_idx): phase_str}，保留原始字符精度。"""
    data: dict[tuple[int, int], str] = {}
    with open(DATA / "mie_phase.csv") as f:
        for row in csv.reader(f):
            sp, wl, cos, val = row[0], int(row[1]), int(row[2]), row[3]
            if sp == species:
                data[(wl, cos)] = val
    return data

def load_optics(species: str) -> dict[int, tuple[str, str]]:
    """返回 {wl_idx: (sigma_sca_str, sigma_abs_str)}。"""
    data: dict[int, tuple[str, str]] = {}
    with open(DATA / "aerosol_optics.csv") as f:
        for row in csv.reader(f):
            sp, wl, sca, abs_ = row[0], int(row[1]), row[2], row[3]
            if sp == species:
                data[wl] = (sca, abs_)
    return data

def format_phase_lut(phase: dict, wl_indices: list[int]) -> list[str]:
    """格式化 PHASE_LUT 数组，保留原样精度。"""
    lines = []
    for cos in range(1024):
        vals = [phase[(wl, cos)] for wl in wl_indices]
        line = "        [" + ", ".join(vals) + "],"
        lines.append(line)
    return lines

def format_sigma(data: dict, wl_indices: list[int], key: int) -> str:
    """格式化 sigma 数组。key=0 为 sca, key=1 为 abs。"""
    vals = [data[wl][key] for wl in wl_indices]
    return "[" + ", ".join(vals) + "]"

def main():
    phase = load_mie_phase("suso")
    optics = load_optics("suso")

    # 验证数据完整性
    for wl in LOW_IDX + HIGH_IDX:
        for cos in range(1024):
            assert (wl, cos) in phase, f"missing phase: suso wl={wl} cos={cos}"
        assert wl in optics, f"missing optics: suso wl={wl}"

    out = Path("../voxel_engine/crates/unreal-atmosphere-8wave/src/aerosol_lo/suso.rs")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("use crate::aerosol::{AerosolSpecies, PHASE_LUT_COS_BINS, PHASE_LUT_WAVELENGTHS};\n\n")
        f.write("pub struct Suso;\n\n")
        f.write('#[expect(\n')
        f.write('    clippy::unreadable_literal,\n')
        f.write('    clippy::excessive_precision,\n')
        f.write('    reason = "光谱查表数据保留生成值"\n')
        f.write(')]\n')
        f.write("impl AerosolSpecies for Suso {\n")

        f.write(f"    const SIGMA_SCA: [f32; PHASE_LUT_WAVELENGTHS] = {format_sigma(optics, LOW_IDX, 0)};\n\n")
        f.write(f"    const SIGMA_ABS: [f32; PHASE_LUT_WAVELENGTHS] = {format_sigma(optics, LOW_IDX, 1)};\n\n")
        f.write("    const PHASE_LUT: &[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS] = &[\n")
        for line in format_phase_lut(phase, LOW_IDX):
            f.write(line + "\n")
        f.write("    ];\n")
        f.write("}\n")
    print(f"wrote {out}")

    out_hi = Path("../voxel_engine/crates/unreal-atmosphere-8wave/src/aerosol_hi/suso.rs")
    out_hi.parent.mkdir(parents=True, exist_ok=True)

    with open(out_hi, "w") as f:
        f.write("use crate::aerosol::{AerosolSpecies, PHASE_LUT_COS_BINS, PHASE_LUT_WAVELENGTHS};\n\n")
        f.write("pub struct Suso;\n\n")
        f.write('#[expect(\n')
        f.write('    clippy::unreadable_literal,\n')
        f.write('    clippy::excessive_precision,\n')
        f.write('    reason = "光谱查表数据保留生成值"\n')
        f.write(')]\n')
        f.write("impl AerosolSpecies for Suso {\n")

        f.write(f"    const SIGMA_SCA: [f32; PHASE_LUT_WAVELENGTHS] = {format_sigma(optics, HIGH_IDX, 0)};\n\n")
        f.write(f"    const SIGMA_ABS: [f32; PHASE_LUT_WAVELENGTHS] = {format_sigma(optics, HIGH_IDX, 1)};\n\n")
        f.write("    const PHASE_LUT: &[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS] = &[\n")
        for line in format_phase_lut(phase, HIGH_IDX):
            f.write(line + "\n")
        f.write("    ];\n")
        f.write("}\n")
    print(f"wrote {out_hi}")

if __name__ == "__main__":
    main()
