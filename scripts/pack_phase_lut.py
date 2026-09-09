"""将 4 物种 × 2 光谱组 × 1024 bins 的相位数据打包为 AoS 二进制。

布局: [PhaseEntry; 2048]
  - [0..1024]:  low 组  (waso, inso, soot, suso) × 4 波长 = 64B/entry
  - [1024..2048]: high 组 (waso, inso, soot, suso) × 4 波长 = 64B/entry

总大小: 2048 × 64B = 128KB
"""

import csv
import struct
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

SPECIES = ["waso", "inso", "soot", "suso"]
LOW_IDX  = [3, 6, 10, 14]
HIGH_IDX = [16, 20, 23, 29]
COS_BINS = 1024

def load_all_phase() -> dict[tuple[str, int, int], float]:
    data: dict[tuple[str, int, int], float] = {}
    with open(DATA / "mie_phase.csv") as f:
        for row in csv.reader(f):
            sp, wl, cos, val = row[0], int(row[1]), int(row[2]), float(row[3])
            if sp in SPECIES:
                data[(sp, wl, cos)] = val
    return data

def pack_entry(phase: dict, sp: str, wl_indices: list[int], cos: int) -> bytes:
    return struct.pack("<4f", *(phase[(sp, wl, cos)] for wl in wl_indices))

def main():
    phase = load_all_phase()
    out_bin = Path("../voxel_engine/crates/unreal-atmosphere-8wave/data/phase_lut.bin")
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    buf = bytearray()
    for group_indices in (LOW_IDX, HIGH_IDX):
        for cos in range(COS_BINS):
            for sp in SPECIES:
                buf += pack_entry(phase, sp, group_indices, cos)

    out_bin.write_bytes(buf)
    actual = len(buf)
    expected = 2048 * 4 * 4 * 4  # 2048 entries × 4 species × 4 f32
    print(f"wrote {out_bin} ({actual}B, expected {expected}B)")

if __name__ == "__main__":
    main()
