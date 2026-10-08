# VDB 云路径追踪参考

`cloud-pt` 是与三个大气求解器互不依赖的云输运 crate。`cloud-demo` 提供渐进 GPU 窗口、CPU/GPU 离线渲染和仅使用 CPU 的资产检查、majorant 成本分析。交互窗口默认组合 `sky-realtime` 的方向天空作为背景和光照边界；云路径内部的大气输运尚未实现。

第三方编解码库通过 Cargo registry 依赖引入：`blosc-src`（启用 LZ4）、`flate2` 和 `half`。仓库没有复制第三方 Rust crate 源码。VDB 读取器是本项目针对已放入仓库的 Disney 格式实现的窄接口，保留 leaf 的 inactive 值和所有非零 uniform tiles。

## CPU 检查

以下命令不会创建 GPU adapter/device，也不会渲染图像：

```powershell
cargo run --release -p cloud-demo -- inspect
cargo run --release -p cloud-demo -- inspect --load
cargo run --release -p cloud-demo -- analyze
```

默认读取 `assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb`。`inspect` 只读头和 transform；加 `--load` 才解码密度。`analyze` 在固定的 16×8 相机射线网格上积分局部/全局 majorant，并检查相机段中点朝太阳的射线；`--rays-x`、`--rays-y` 可提高分析密度。结果是完整射线段的期望 Poisson 候选数，不能当作实际渲染耗时或帧率。

以下开发检查只使用 CPU：

```powershell
cargo check --workspace --all-targets
cargo test -p cloud-pt --lib --tests
cargo test -p cloud-demo
python scripts/check_architecture.py
```

这里的 shader 测试包括 Naga 解析/静态验证、CPU 上的 SPIR-V 代码生成和采样数学镜像。`view`、`render --backend gpu` 和 `benchmark` 会使用 GPU；与大气烘焙、其他渲染和 GPU 性能测量串行运行，避免资源竞争和计时干扰。

## 运行与性能测量

```powershell
# 渐进窗口：交互图像默认 128×72，目标仍为 1024 spp。
cargo run --release -p cloud-demo -- view
# 较短的 GPU 观看预览；输出目录必须尚不存在。
cargo run --release -p cloud-demo -- render --backend gpu --width 128 --height 72 --spp 64 --out out/cloud-preview
# f64 CPU 参考；保守高密度云可能需要很长时间。
cargo run --release -p cloud-demo -- render --backend cpu --width 333 --height 180 --spp 4096 --out out/cloud-cpu-4096
```

窗口提供侧栏参数、进度、暂停、重置与保存。`W/A/S/D` 水平移动，`Q/E` 降低/升高，`Shift` 加速；右键拖动转向，左键拖动绕目标旋转，滚轮缩放。`Space` 暂停，`R` 恢复初始相机和太阳；`I/K` 改太阳高度，`J/L` 改太阳方位；`[`/`]` 改显示曝光，侧栏 `Save` 或 `Ctrl+S` 请求快照，`Esc` 退出。UI 捕获鼠标或键盘时不会操作相机，失焦清除按键。太阳方位统一为 0° 朝 +Z、90° 朝 +X。

交互图像默认 128×72，目标 1024 spp；离线 `render` 默认仍为 640×360/1024 spp，显式尺寸或 scene 文件优先。画布按实际图像比例留边，窗口缩放不改变参考图像分辨率。侧栏支持曝光、太阳方向、云密度/反照率/HG g、相机位置/FOV/移动速度、实时天空和局部地面开关。曝光和窗口尺寸变化保留累积；相机、物理参数、光照和目标 spp 改动会在当前小工作块完成后清空旧累积。

预览采用全帧持久路径状态和固定像素置换，每轮分散推进全幅像素，不等一块区域的长路径完成才处理下一块。默认每像素并行推进 4 个独立样本，完成的连续样本前缀立即按原索引次序累积；`--sample-batch-size 1` 可选择逐样本推进，预览接受 1–4，0 自动选择 4。每个工作块仍只处理最多 4096 个路径槽、每槽至多 128 个状态转换；长路径的 RNG、射线和输运状态保留。128×72、4 样本的全帧状态约 10.125 MiB；设备容纳不了指定分辨率时明确报错。离线 `new` 保留原批次池调度，估计器与完整输出不受调度方式影响。

窗口通过异步回读检查完成状态，最多一个提交在飞，展示上限约 30 Hz。默认计算预算 80%，按 timestamp 测得的 GPU 时间安排必要的休息，扣除已经发生的回读/轮询等待，避免把 CPU 延迟再放大成空闲。每个新样本批次及相机/场景重置都从一个工作块开始；依据最近 8 块的最大耗时逐步合并短块，默认上限 4，目标整组约 12 ms。单块的工作上限不变。没有 timestamp 的设备保持每提交一块、不额外猜测休息时间。失焦、隐藏或最小化后停止新增计算。`Save/Ctrl+S` 等待当前整幅批次完成后异步捕获并在 CPU 线程输出；即使暂停，明确请求保存也会继续有界计算至该边界。混合 spp 的中间图像不会导出；重复保存请求合并。GPU/路径错误使窗口暂停并保留错误标题，详细信息写入 `out/cloud-demo-view-error-*.log`；失效设备被丢弃。

侧栏可调整计算预算，并在 Throughput（最多 4 块）与 Responsive（每提交 1 块）之间切换；这两项不清空样本。命令行 `--gpu-budget-percent 10..100` 和 `--work-group-size 1..8` 可覆盖初始值，后者是自适应上限。GPU 回读完成后使用新的时间值判断提交和展示是否到期，让无需休息的下一组在同一轮提交。侧栏显示云 compute 时间、工作块最大时间和计算 duty；duty 是计时 compute / 从第一次提交开始的墙钟窗口，包含等待、展示和休息，不能视为 SM 占用率或整机全部 GPU 活动。

退出时 `CLOUD_VIEW_METRICS` 行归档实际分组分布、GPU 与提交到回读的墙钟耗时、CPU 编码、预算休息、完整/当前批次路径计数。`python apps/cloud-demo/tools/profile_viewer.py run --nvml --rounds 3 --out out/cloud-view-profile` 可串行运行有限窗口并归档二进制/资产指纹与日志；这是显式 GPU 操作。`compare` 和脚本测试只使用 CPU。可选 NVML 数据是 `nvidia-smi` 的设备活动采样，包含启动时间和其他进程；缺失 timestamp 不记作 0。

`view --exit-after-seconds 5 --smoke-frames 4` 提供有限窗口诊断，达到任一条件即退出，诊断模式禁用参考导出。这些参数用于在独占 GPU 时隙中做小图窗口检查；程序不修改驱动或 Windows 超时设置。

`view --exit-after-seconds 5 --capture-preview out/cloud-ui.png` 在同一呈现提交中异步捕获包含 UI 的 PNG；它只用于预览，不是参考数据。只有 `--capture-preview` 时写完 PNG 自动退出。目标路径必须是新的 PNG，提前达到退出条件而未捕获会明确报错，不同步等待 GPU。

## 实时天空边界

应用层 `SkyBackground` 使用与实时天空 demo 相同的 balanced 求解、256² SkyView 与 64 步，生成 1024×512 的全方向线性环境。云核心只接受通用方向 RGB 与 1×1 太阳照度纹理，仍不依赖任何大气核心。按照云相机的真实基向量查询，保留原始相机的 roll，避免两套相机屏幕右轴相反造成镜像。

世界单位约定为米，默认海平面 y=−1000；默认相机高度约 0.9175 km。海平面基准在启动时固定，不随局部地面开关改变。天空从线性 Rec.2020 转为线性 sRGB；天空与经过大气衰减的太阳照度采用同一个增益，匹配原云太阳照度的亮度尺度，最后统一曝光、Reinhard 和一次 sRGB 编码。环境不含直接太阳盘，太阳由 delta 光源 NEE 计入。未完成采样的像素立即显示方向天空；已经采样的 film 自身含背景透射，不再次叠加天空。

默认交互场景关闭原无限局部平面，使用天空的远处球面地面，避免 0° 平面地平线与约 −0.97° 大气地平线及日落遮挡冲突。显式 `--scene` 保留其地面配置，侧栏可重新打开局部平面。离线 `render`/`benchmark` 保留原常量天空和原场景地面。

这个方向环境是在相机海拔生成后冻结的远场边界，云路径内部不积分空间变化的大气。云 PT 对所选择的冻结 RGB 边界模型保持原估计方式；它不是完整云/大气联合场景的无偏解。快照 `asset.json` 记录环境布局、海平面/高度、太阳、实时配置/映射/波长和求解输入指纹。

可选参数包括 `--vdb`、`--grid`、`--density-scale`、`--g`、`--albedo`、`--no-ground`、`--sun-elevation`、`--sun-azimuth`、`--spp`、`--seed`、`--sample-batch-size`。`--scene experiments/clouds/disney-eighth.json` 可指定完整且可保存的 JSON 配置，命令行参数覆盖 JSON 中相应项。渲染时 `--global-majorant` 和 `--no-shadow-roulette` 是消融开关，改变提议效率而不改变密度或物理模型。`benchmark` 自行选择提议/阴影轮盘组合。

```powershell
# 默认轮换 global/spatial 两种 majorant，均启用阴影 roulette；文件必须尚不存在。
cargo run --release -p cloud-demo -- benchmark --width 64 --height 36 --spp 32 --rounds 3 --out out/cloud-benchmark.json
```

设备支持 timestamp query 时记录每个有界工作块的完整路径采样和按序累积 compute 的 GPU 毫秒，并汇总完整逻辑批次；还记录工作块数量及最大 GPU 块耗时。`encode_submit_diagnostics_ms` 含编码、提交和进度/诊断回读，`wall_including_timestamp_reads_ms` 额外包含 timestamp 回读；均排除资产读取、管线创建和 EXR 输出。预热最多 2 spp。不同提议方案会走不同随机轨迹，需要足够 spp 和多轮测量。还输出各 RGB 通道的平均单次样本方差及方差×每 spp GPU 时间，供粗略比较同场景的成本；同一随机流的重复计时轮次不增加独立质量样本。

`--include-no-shadow-roulette` 才会额外加入关闭阴影轮盘的两个慢消融方案。厚云的暗阴影可能产生超过诊断预算的合法工作量；当前 Disney 场景的 global/noRR 已出现 watchdog 失败和一次设备丢失，二者分别记录，设备丢失原因尚未确定。每项测量前先写入 pending checkpoint；诊断失败项标为 invalid，不导出部分图像、不参与有效性能排名，后续轮次跳过同一失败随机流。设备丢失或 GPU/validation panic 保存 checkpoint 后终止整个 benchmark，丢弃该 device。

## 全帧预览与天空验证

并行样本与合并提交的实际 GPU 检查覆盖 4×2×4 spp 真实云：每像素 4 条路径、每组 1/4/8 块的最终均值和样本方差均与冻结参考逐位一致。128×72 零密度检查覆盖重置、部分样本拒绝导出、完整 4 样本批次及最后 3 样本批次，最终 7 spp 精确匹配环境且方差为零；整组时间戳端点与单块端点一致。报告见本机 `out/cloud_grouped_preview_qa_v1/summary.json`。固定组长的真云短探针测到 1/4/8 块约 11.8/44.1/97.6 ms，因此窗口不固定合并 4/8 块，而是冷启动单块并按测量自适应。

RTX 4090 上，同一最终程序、128×72 Disney 场景/实时天空/固定种子/目标 1024 spp，分别串行运行 3 次 30 秒。以下为中位数；控制组也使用新的 80% 预算与事件循环修正，仅把并行样本和合并上限设为 1：

| 指标 | 单样本 / 单块控制组 | 默认 4 样本 / 自适应最多 4 块 |
| --- | ---: | ---: |
| 完成路径 / 计算窗口秒 | 673.4 | 1343.7 |
| 完成路径 / 请求运行秒（包含启动） | 613.6 | 1224.0 |
| 云 compute duty | 50.1% | 73.8% |
| NVML 设备活动（包含启动） | 58.4% | 76.3% |
| 展示 / 秒 | 26.6 | 25.0 |

默认方案的完成路径速度约为 2.00 倍；三轮最大单块/整组耗时分别不超过 14.43 ms，均正常退出。控制组三轮完成 18350–18425 条路径，默认完成 36720–36724 条；后者在同一 4 样本批次内仍有少量长路径未结束，整幅 spp 仍为 0。完成路径计数包含尚未达到某像素连续前缀的后续样本，不能等同于整幅 spp、收敛质量或完整参考的长期速度。此处没有截断尾部，也未采用增加单路径工作量或工作槽数量的方案。报告、物理输入、二进制指纹与原始时钟/温度日志见本机 `out/cloud_utilization_v1/candidate_final_repeats` 和 `control_final_repeats`。

最终程序另外完成一次 15 秒的异步 UI PNG 诊断，正常退出，无参考导出。侧栏预算、统计与画布排版已目视检查；`out/cloud_utilization_v1/final_ui.png` 是达到 20 个工作组后捕获的早期图，未完成路径仍显示方向天空，不能用它判断收敛云形。15 秒结束时完成 35887 条路径，compute duty 约 74.6%，单块最大约 13.3 ms。

`new_preview` 的实际 GPU 验证覆盖 4×2×4 spp 真实云，与冻结旧调度的均值、样本方差逐位一致；128×72 零密度场景首个 4096 路径槽分布覆盖所有行，三次提交完成一遍全帧，两个 spp 的均值精确等于已知常量环境、方差为零。inflight/partial 环境切换、目标 spp、reset、重复完成和旧 epoch 均有检查。报告见本机 `out/cloud_preview_qa_v1/summary.json`。

实际天空组合验证检查了默认海拔下太阳 +0.2°/−0.2°/−1.2° 的直射衰减（前两可见，后者零）、背景实际相机基向量对齐、一次曝光与已累积 film 不重复加天空。128×72 零密度全帧逐像素完成；64×36 真实云完成 4 spp 并导出完整参考，最慢 GPU 工作块约 2.62 ms。该小图仍有像素格与 Monte Carlo 噪声；它验证组合和完成条件，不声明高清质量。报告与图像见本机 `out/cloud_sky_environment_qa_v1`。

带侧栏的默认窗口、零密度窗口均已完成有限实际运行、异步 UI PNG 捕获与正常退出，无 GPU 错误或 reference 导出。最终默认窗口 5 秒完成 73 次展示、180 个工作块；较晚截图中当前样本已有 6153/9216 个像素完成，但整幅仍为 0 spp，不能把该早期图当作收敛云图。零密度的 2 spp 测试截图写完后自动正常退出。相机数学、太阳方位、UI 输入捕获与图像比例的 CPU 测试已通过；没有使用 OS 自动输入来声称所有交互动作都做了运行时测试。最终日志和截图见本机 `out/cloud_sky_viewer_qa_v2/summary.json`，先前早期检查保留于 `out/cloud_sky_viewer_qa_v1`。

更大的单路径工作预算也做了短探针。512/256 次状态转换保持小图数值，但默认 128×72 真实云的第一个工作块分别约 39.6/21.9 ms，均未采用。交互仍保留 128 次转换，优先控制单次占用时长。不能把只测一个块的探针称作持续吞吐或 P99 检查。

## 前一版有界入口与窗口验证

下面是加入全帧预览、实时天空和 UI 之前的记录。GPU 每个路径槽推进至多 128 个状态转换。在 RTX 4090 上，4×2×4 spp 的最大工作块为 2.629 ms；8×4×32 spp 完成 3306 个工作块，GPU 块 P99 为 1.659 ms、最大 3.276 ms，参考追踪墙钟 1.661 s（含进度/timestamp 回读与 checkpoint，排除初始化及 EXR 输出）。两组结果与 32 转换预算的最终均值、样本方差逐位一致；与旧整路径实现有少数浮点末位差异。完整批次、reset、旧 epoch、重复完成和混合 spp 拒绝导出检查见 [有界核心汇总](../out/cloud_bounded_qa_v1/final_summary.json)。

最终 128 转换版本的实际默认窗口在 5 秒诊断中完成 105 次展示、181 个工作块，正常退出，没有 GPU/路径错误。此时尚未完成整幅样本批次，诊断不导出参考图像。另用预算 1 注入 watchdog 错误，错误日志生成后窗口继续保留约 4.1 s，直到预定诊断时限退出。测试期间还发现并修复了 Win32 在申请退出后进入无限 Wait 的调度问题；该失败及强制结束记录保留，修复后单独重测成功。日志与冻结可执行文件信息见 [窗口验证汇总](../out/cloud_viewer_qa_v1/summary.json)。这些是有限运行的验证记录，不代表所有资产或长期运行的时间上限。

## 历史质量与性能检查

以下记录来自改为可续传有界工作块之前的整样本 GPU 入口，不能用作新入口的当前大图耗时；新入口的小图与窗口检查见上节。2026-10-08 在 NVIDIA GeForce RTX 4090 上，以默认 Disney eighth VDB、相机、密度倍率、光照和地面运行。64×36、32 spp、3 轮轮换的 6 项测量全部有效，自动批次为 15/15/2 spp：

| 提议方案（启用阴影 roulette） | GPU 总计中位数（ms） | 编码/提交/诊断墙钟中位数（ms） | 平均样本方差 × GPU ms / spp（R/G/B） |
| --- | ---: | ---: | --- |
| global | 4272.119 | 4273.206 | 107.83 / 99.93 / 85.79 |
| spatial | 3906.796 | 3907.439 | 94.15 / 87.24 / 74.87 |

该场景 spatial 的 GPU 时间降低 8.55%，32 spp 的方差×时间指标降低约 12.7%。该方差指标有有限样本噪声，三轮重复的是相同随机流，只用于计时重复。原始记录为 [cloud_gpu_benchmark_safe_v1.json](../out/cloud_gpu_benchmark_safe_v1.json)，汇总为 [cloud_gpu_safe_summary_v1.json](../out/cloud_gpu_safe_summary_v1.json)。

旧入口的小图批次验证覆盖 8×4×32 spp 的 single↔auto，以及 8×4×1024 spp 的旧 single↔auto：解码后的所有均值、样本方差 f32 bits 均为 0 mismatch，失败路径或像素未被丢弃。8×4×1024 的 CLI 渲染墙钟从此前 186.09 s 到 auto 的 2.63 s，包含设备/管线初始化、回读和输出，且运行时刻及缓存状态不同；这些是历史小图数据，当前有界入口和大图吞吐需独立测量。

同模型 8×4×1024 的 CPU f64 与 GPU f32 独立随机流比较中，96 个 RGB 分量的最大联合标准误差倍数为 1.872，全部小于 3。联合 SE 为 sqrt(varCPU/nCPU + varGPU/nGPU)；重尾与多重比较限制有限样本解释。另一个 8×4×4 的解析测试关闭太阳和地面、保留反照率 1：全部均值精确等于常量天空，方差全为 0。逐位、CPU 对照及解析检查见 [quality_status.json](../out/cloud_batch_qa_v1/quality_status.json)。

128×72×64 spp 的默认 spatial/roulette [观看预览](../out/cloud_gpu_preview_v1/preview.png) 用时 23.60 s，云轮廓和明暗结构可见，仍有明显 Monte Carlo 噪声。该图用于预览；[输出目录](../out/cloud_gpu_preview_v1) 保留线性 EXR、样本方差、标准误差和完整元数据。默认目标仍为 1024 spp。

## 参考模型与统计

默认相机、RGB 光照、密度倍率 4、HG `g=0.877`、单次散射反照率 1、地面 y=−1000/反照率 0.2 取自资产的 Mitsuba 示例。太阳为定向 delta 光源，环境为常量 RGB。采用 VDB 原始整点采样值的三线性重建；体积边界涵盖其有限插值支撑，不额外裁剪成示例 XML 的 cube。选择不同分辨率 VDB 就选择了不同密度模型，程序不会自动重采样或降级。

delta tracking 采样真实消光事件，碰撞后乘散射反照率并采样 HG；太阳只由 ratio tracking 可见性的 next-event estimation 计入，常量环境由路径逃逸计入。Lambert 地面采用余弦采样。没有硬散射深度、确定性的低透过率终止或贡献裁剪；路径 Russian roulette 允许存活概率达到 1，适用于保守云。阴影估计每 16 个候选、权重低于 1e−4 时，以 1/2 存活概率做 roulette，存活权重乘 2；跨 cell 保持累计权重与计数。它减少暗阴影工作量，代价是额外方差，条件均值保持不变。`--event-limit` 是诊断预算，超过预算会使整个结果报错，不能靠丢弃失败路径生成参考图像。

这些估计器在精确算术下对所声明的密度/材质/光照模型无偏。CPU 使用 f64 输运及 Welford 累积，GPU 使用 f32，距离步进使用补偿表示避免微小正步长停滞；随机数为伪随机。有限精度、有限样本噪声和模型差异仍存在，不声称逐像素复现 Hyperion。输运方法可对照 [PBRT 的透过率推导](https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance) 与 [体散射积分器](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators)。

每次保存包含：

- `radiance.exr`：线性 RGB 样本均值，无曝光/色调映射。
- `sample_variance.exr`：单次样本的无偏方差估计，分母 n−1；n>1 才有。
- `standard_error.exr`：均值的标准误差估计 sqrt(variance/n)；n>1 才有。
- `preview.png`：只用于观看的曝光/Reinhard/sRGB 图。
- `asset.json`：源路径与字节数、密度统计、相机、光学和采样配置、后端精度、实际 spp、是否完成请求、归属说明。

没有自动根据同一批样本的方差提前终止，避免把停止规则引入参考均值。可以用独立预实验确定所需 spp，再按固定 spp 正式渲染。GPU 样本计数上限为 2²⁴；更长运行使用 CPU 后端。

## 稀疏加速与格式边界

原始密度以 8³ float leaf 和紧凑的 8/128/4096 uniform tile 保存，支持负坐标。GPU 精确展开小 8³ tile 到 brick，避免每次密度查询遍历 Disney 资产的大量小 tiles；大 tiles 保持紧凑。三线性查询在八个角点同属于一个 leaf 时只查一次哈希，跨 leaf 时仍查询各角点，插值结果不变。

8³ cell 的保守局部 majorant 包括相邻 leaf 的插值支撑；GPU 提议另外扩张数值 halo 和边界，覆盖 f32 舍入误差，密度本身不变。DDA 划分完整射线并跳过密度严格为零的区间。每个非空段以自身的速率进行指数采样，跨段依靠 Poisson 过程的无记忆性重新采样，不引入步长积分近似。CPU 几何测试覆盖负坐标、精确边界、多轴同时跨越和有限终点，解析 slab 测试覆盖 local/global 两种提议与 Beer–Lambert 一致性。GPU 大图的实际吞吐和长路径驱动行为需要独立测量；有限的冒烟测试不能替代收敛检查。

离线 `--sample-batch-size` 指每个逻辑批次的 spp；`0` 按整幅图像目标约 32,768 条路径自动选择，最多 1024 spp/逻辑批次，`1` 使用单样本逻辑批次。实际提交最多使用 4096 个路径槽，每槽推进至多 128 个状态转换，再保存状态继续；离线路径池占用最多 1.125 MiB。交互窗口使用上文的全帧状态池与 1–4 样本批次。没有通过限制散射深度、截断贡献或丢弃长路径控制 GPU 工作量。

每个像素按原来的样本索引顺序提交已完成的连续前缀，逐个更新 Welford 均值和方差，让长路径未完成时其他像素也可提前显示；完成整幅逻辑批次后才增加 host spp。中途像素的 spp 可以不同，因此导出必须等完整批次边界，且逐像素校验样本计数。批次保留 RNG 的 pixel/sample 索引、追踪算法和目标样本数，不做无序浮点求和；640×360 默认选择 1 spp/逻辑批次，同样使用有界续传。

GPU 的远地面反弹使用补偿的 world slab、进入点和距离，再进入共享 index 坐标链，避免远起点减法吞掉薄云段。静态索引/相机/平移的精度预算为 2¹⁸ voxel，动态远起点预算为 2⁴⁰ voxel；不支持的精度范围会报错而非悄悄漏云。解析镜像覆盖近地到 2⁴⁰ 的射线。CPU 使用相同 world→index 查询链划分边界，零长度交界只调整单元归属，保持距离不变。

读取器支持 OpenVDB 文件版本 222–224、`Tree_float_5_4_3`（可保存为 half）、正 uniform scale 和 uniform scale+translation、未压缩/active mask/ZIP/Blosc LZ4。其他树、grid instances、非均匀 transform、非零无限背景或不支持的压缩会明确报错。不是通用 OpenVDB 替代品。完整高分辨率数据可能需要大量内存，GPU 各 buffer 会检查设备绑定上限；超限时需显式选择较低分辨率资产。

资产保存在忽略跟踪的 `assets/` 中；没有纳入 Git。Disney 数据集采用 CC BY-SA 3.0，原始照片来自 Kevin Udy / Colorado Clouds Blog，输出元数据保留归属。分发资产或衍生结果请遵循其 [官方数据集许可与说明](https://www.disneyanimation.com/resources/clouds/)。
