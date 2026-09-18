"""Plot recorded diagnostics; this does not perform atmosphere transport."""
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
directory = root / "out/lut_validation_v1"
reports = sorted((json.loads(p.read_text(encoding="utf-8"))
                  for p in directory.glob("solar_*deg.json")),
                 key=lambda r: r["sun_elevation_deg"])
if len(reports) != 7:
    raise RuntimeError("Expected all seven solar elevations before summarizing")
manifest = json.loads((root / "out/lut_reference_v1/asset.json").read_text())
for report in reports:
    for field in ["coordinate_mapping", "model_fingerprint_fnv1a64"]:
        if report[field] != manifest[field]:
            raise RuntimeError(f"Mixed LUT inputs in report: {field}")
    for key, expected in manifest["config"].items():
        actual = report["config"].get(key)
        # Typed f32 manifest JSON uses shortest-roundtrip decimals; the generic
        # comparison JSON can contain their expanded f64 serialization.
        equal = struct.pack("f", actual) == struct.pack("f", expected) if isinstance(expected, float) else actual == expected
        if not equal:
            raise RuntimeError(f"Mixed LUT configuration in report: {key}")
    if report["pt_transport_version"] != "wgpu-layered-midpoint-v1" or report["pt_max_orders"] is not None:
        raise RuntimeError("Expected corrected, untruncated path tracing references")
    if len(report["bands"]) != len(manifest["bands"]):
        raise RuntimeError("Incomplete spectral comparison")
size = sum(p.stat().st_size for p in (root / "out/lut_reference_v1").iterdir() if p.is_file())
rebuild_jobs = json.loads((root / "out/pt_transport_rebuild_jobs.json").read_text(encoding="utf-8-sig"))
rebuilt_count = sum(job["Status"] == "complete" for job in rebuild_jobs)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for wavelength, color in [(440, "#2274bf"), (550, "#20845a"), (680, "#bd4144")]:
    for axis, region in zip(axes, ["sky", "earth_shadow"]):
        xs, ys = [], []
        for r in reports:
            band = next(b for b in r["bands"] if b["center_nm"] == wavelength)
            s = band["regions"][region]
            if region == "earth_shadow" and r["sun_elevation_deg"] > 0:
                continue
            xs.append(r["sun_elevation_deg"])
            ys.append(100 * (s["mean_lut"] / s["mean_path_traced"] - 1))
        axis.plot(xs, ys, "o-", color=color, label=f"{wavelength} nm")
for axis, title in zip(axes, ["Whole sky: regional mean", "Anti-solar horizon: regional mean"]):
    axis.set(title=title, xlabel="Solar elevation (degrees)", ylabel="LUT / PT - 1 (%)")
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
fig.suptitle("Full spectral LUT v1 vs corrected GPU path tracer\nRegional means still contain Monte Carlo uncertainty", fontsize=12)
fig.savefig(directory / "regional_mean_bias.png", dpi=180)
plt.close(fig)

lines = ["# 离线光谱 LUT v1 验证记录", "",
         "本版可用于保留完整光谱、比较离散输运误差和研究后续表示。它仍是候选参照，蓝调时刻的网格偏差和 PT 方差尚未消除。", "",
         f"- GPU：{manifest['adapter']}。",
         f"- 完整资源：`out/lut_reference_v1`，41 波段、每波段 12 阶，实际 {size:,} bytes。全部二进制校验和已核对。",
         "- 全部输运及 LUT 为 f32，4D 表包含完整相位和地面边界辐亮度；直接太阳盘单独显示。",
         "- 统计：128×64 全景，LUT 4×4 子像素积分；−6° 为 16,384 spp，−4° 为 8,192 spp，其余为 4,096 spp。剔除太阳盘所在像素。", "",
         "## 难例结果", "",
         "逐像素 L2 含 PT 方差，不能直接解释为求解器偏差。尤其是蓝调时刻，均匀方向采样的 PT 存在明显彩色噪声；应同时看区域均值。区域定义和所有 41 波段原始结果保存在对应 JSON。", "",
         "| 太阳高度 | 波长 nm | 天空 L2 | 天空均值差 | 太阳周围 10° L2 | 地平线 L2 | 背日阴影均值差 |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    for nm in [440, 550, 680]:
        b = next(b for b in r["bands"] if b["center_nm"] == nm)
        regions = b["regions"]
        bias = lambda s: 100 * (s["mean_lut"] / s["mean_path_traced"] - 1)
        shadow = f"{bias(regions['earth_shadow']):+.1f}%" if r["sun_elevation_deg"] <= 0 else "—"
        lines.append(f"| {r['sun_elevation_deg']:g}° | {nm} | {100*regions['sky']['relative_l2']:.1f}% | {bias(regions['sky']):+.1f}% | {100*regions['near_sun']['relative_l2']:.1f}% | {100*regions['horizon']['relative_l2']:.1f}% | {shadow} |")
lines += ["", "![区域均值差](../out/lut_validation_v1/regional_mean_bias.png)", "",
          "原始光谱结果：[−6°](../out/lut_validation_v1/solar_-6deg.json)、[−4°](../out/lut_validation_v1/solar_-4deg.json)、[−2°](../out/lut_validation_v1/solar_-2deg.json)、[0°](../out/lut_validation_v1/solar_0deg.json)、[5°](../out/lut_validation_v1/solar_5deg.json)、[20°](../out/lut_validation_v1/solar_20deg.json)、[90°](../out/lut_validation_v1/solar_90deg.json)。", "",
          "## 已定位并修复的问题", "",
          "1. 天顶太阳的前向峰：单侧地平线加密会让竖直方向的 μ 节点过疏。本版采用两端加密的三次映射，开发预设天顶案例的 440/550/680 nm 全图相对 L2 从约 135%–166% 降至约 3%（这组映射回归比较使用当时的 PT；最终表格使用修正后的 PT）。",
          "2. 原 PT 掠射分层错误：5 m 前探在近切线方向不足以改变 f32 半径，误判当前层会漏算更密的层。现在搜索相邻边界，并在实际分段中点确定 majorant。源函数、相位和积分模型没有为匹配图像而加经验增益。",
          "3. 模型指纹：Rayleigh 的 powi 舍入序列可能随 debug/release 改变。改用明确乘法后，两种构建生成的整个模型 JSON 相同。", "",
          "| 高度 km；太阳 −6° | 原 PT 透射 | 修正 PT 透射 | 16,384 步 f32 直接积分 |",
          "|---:|---:|---:|---:|", "| 40 | 0.0037613 | 0.0013504 | 0.0013399 |",
          "| 50 | 0.2444458 | 0.0820999 | 0.0813605 |", "| 60 | 0.3420219 | 0.3314323 | 0.3306536 |",
          "| 70 | 0.8208008 | 0.7579117 | 0.7574292 |", "",
          "## 收敛与限制", "",
          "- 550 nm、太阳 −6°：相同参考网格的角积分从 32×64 提到 64×128，LUT 全图均值仅从 1.4173592e-4 变为 1.4178058e-4，约 0.032%。继续增加这一项收益有限。",
          "- 负太阳高度仍对太阳角度与高度插值敏感。独立首阶积分显示，在 550 nm、太阳 −6°、天顶视线，参考网格 LUT 为 4.0354e-5，直接射线积分为 3.0969e-5；在本应几乎全阴影的背日 10° 视线存在插值漏光。不能把阶数增量很小理解为整个离散解已经精确。",
          f"- 41 波段最后一阶的最大相对网格求和增量为 {max(r['orders'][-1]['relative_texel_sum_increment'] for r in manifest['records']):.6g}，默认没有触发提前终止。这不是暗区相对误差上界。",
          "- 继续研究应优先改进太阳角/阴影边界坐标与插值，随后提高对应维度分辨率。`max_asset_bytes` 默认 1 GB，可在映射改进收益饱和后提高；当前没有扩大资源预算，也未进行半精度或 RGB 压缩。", "",
          "## 复现与交互比较", "",
          "```powershell", "cargo run -p sky-realtime-demo --release -- --experiment offline-lut --lut out/lut_reference_v1 --asset out_skyview_search/elev_020/asset.json", "```", "",
          "按 1 / 2 / 3 / 4 切换 LUT、PT、绝对差异、有符号差异。改变太阳或观察高度后，需与当前参考条件一致才启用差异模式。图像差异沿用 demo 的显示变换后差异，光谱定量结果见 JSON。", "",
          "- [蓝调时刻四联图](../out/lut_validation_v1/earth_shadow_m06.png)：背日 180°，俯仰 6°，FOV 30°，曝光 +8 EV；PT 的天空视图坐标在背日方向分辨率较低，近看可见噪声和像素块。",
          "- [太阳附近四联图](../out/lut_validation_v1/near_sun_20.png)：太阳高度 20°，俯仰 20°，FOV 30°，曝光 0 EV。直接太阳盘在低分辨率 PT 图中有像素混叠，不用它判定散射 LUT 的精度。",
          "- [日落四联图](../out/lut_validation_v1/sunset_0.png)：太阳高度 0°，俯仰 2°，FOV 45°，曝光 +2 EV。",
          "- [侧向地平线四联图](../out/lut_validation_v1/horizon_0.png)：太阳高度 0°，方位 90°，俯仰 0°，FOV 30°，曝光 +2 EV。",
          "- [完整配置、格式与命令](../crates/sky-atmosphere-lut/README.md)。",
          f"- 已删除 545 张旧图，重算并校验 {rebuilt_count}/{len(rebuild_jobs)} 组资源。清单与状态：`out/pt_transport_deleted_images.txt`、`out/pt_transport_rebuild_jobs.json`。重算保留原尺寸、spp、seed 和观察条件，旧 15 波段全景升级到当前 41 波段。", "",
          "## 验证", "",
          "通过大气 LUT、sky-core、demo、PT、参考输出五个 crate 的 47 项测试，包括实际 GPU 输运与掠射透射回归。通过新 crate 与 PT 的严格 Clippy，以及 demo 的 Clippy（保留既有公共绘制函数 too_many_arguments 例外）。", ""]
(root / "reports").mkdir(exist_ok=True)
(root / "reports/offline-lut-v1.md").write_text("\n".join(lines), encoding="utf-8")
print(root / "reports/offline-lut-v1.md")
