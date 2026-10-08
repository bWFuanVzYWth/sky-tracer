# VDB 云路径追踪参考

`cloud-pt` 是与三个大气求解器互不依赖的云输运 crate。`cloud-demo` 提供渐进 GPU 窗口、CPU/GPU 离线渲染和仅使用 CPU 的资产检查、majorant 成本分析。云与大气的耦合尚未实现。

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

窗口左键拖动绕目标旋转、滚轮缩放；`Space` 暂停，`R` 恢复初始相机和太阳；`I/K` 改太阳高度、`J/L` 改太阳方位；`[`/`]` 改显示曝光，`S` 请求快照，`Esc` 退出。交互默认图像为 128×72，离线 `render` 默认仍为 640×360；两者目标均为 1024 spp，显式尺寸或 scene 文件优先。相机和太阳变化在当前小工作块完成后清空累积；曝光、窗口尺寸变化保留线性结果。窗口缩放不改变参考图像分辨率。

窗口通过异步回读检查完成状态，最多一个提交在飞，展示上限约 30 Hz。每个计算块完成后至少主动空闲 16 ms，空闲时间还取完成墙钟的 3 倍，以保守控制 GPU 占用；失焦、隐藏或最小化后停止新增计算。长路径的 RNG、射线和输运状态保留到下一个工作块，目标 spp 和无偏估计器保持不变。`S` 等待当前整幅逻辑批次完成后异步捕获并在 CPU 线程输出；即使已暂停，明确请求保存也会让有界计算继续至该边界，混合 spp 的中间图像不会导出。重复保存请求会合并。GPU/路径错误使窗口暂停并保留错误标题，详细信息写入 `out/cloud-demo-view-error-*.log`；失效设备被丢弃。

`view --exit-after-seconds 5 --smoke-frames 4` 提供有限窗口诊断，达到任一条件即退出，诊断模式禁用参考导出。这些参数用于在独占 GPU 时隙中做小图窗口检查；程序不修改驱动或 Windows 超时设置。

可选参数包括 `--vdb`、`--grid`、`--density-scale`、`--g`、`--albedo`、`--no-ground`、`--sun-elevation`、`--sun-azimuth`、`--spp`、`--seed`、`--sample-batch-size`。`--scene experiments/clouds/disney-eighth.json` 可指定完整且可保存的 JSON 配置，命令行参数覆盖 JSON 中相应项。渲染时 `--global-majorant` 和 `--no-shadow-roulette` 是消融开关，改变提议效率而不改变密度或物理模型。`benchmark` 自行选择提议/阴影轮盘组合。

```powershell
# 默认轮换 global/spatial 两种 majorant，均启用阴影 roulette；文件必须尚不存在。
cargo run --release -p cloud-demo -- benchmark --width 64 --height 36 --spp 32 --rounds 3 --out out/cloud-benchmark.json
```

设备支持 timestamp query 时记录每个有界工作块的完整路径采样和按序累积 compute 的 GPU 毫秒，并汇总完整逻辑批次；还记录工作块数量及最大 GPU 块耗时。`encode_submit_diagnostics_ms` 含编码、提交和进度/诊断回读，`wall_including_timestamp_reads_ms` 额外包含 timestamp 回读；均排除资产读取、管线创建和 EXR 输出。预热最多 2 spp。不同提议方案会走不同随机轨迹，需要足够 spp 和多轮测量。还输出各 RGB 通道的平均单次样本方差及方差×每 spp GPU 时间，供粗略比较同场景的成本；同一随机流的重复计时轮次不增加独立质量样本。

`--include-no-shadow-roulette` 才会额外加入关闭阴影轮盘的两个慢消融方案。厚云的暗阴影可能产生超过诊断预算的合法工作量；当前 Disney 场景的 global/noRR 已出现 watchdog 失败和一次设备丢失，二者分别记录，设备丢失原因尚未确定。每项测量前先写入 pending checkpoint；诊断失败项标为 invalid，不导出部分图像、不参与有效性能排名，后续轮次跳过同一失败随机流。设备丢失或 GPU/validation panic 保存 checkpoint 后终止整个 benchmark，丢弃该 device。

## 有界入口与窗口验证

当前 GPU 每个路径槽推进至多 128 个状态转换。在 RTX 4090 上，4×2×4 spp 的最大工作块为 2.629 ms；8×4×32 spp 完成 3306 个工作块，GPU 块 P99 为 1.659 ms、最大 3.276 ms，参考追踪墙钟 1.661 s（含进度/timestamp 回读与 checkpoint，排除初始化及 EXR 输出）。两组结果与 32 转换预算的最终均值、样本方差逐位一致；与旧整路径实现有少数浮点末位差异。完整批次、reset、旧 epoch、重复完成和混合 spp 拒绝导出检查见 [有界核心汇总](../out/cloud_bounded_qa_v1/final_summary.json)。

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

`--sample-batch-size` 指每个逻辑批次的 spp；`0` 按整幅图像目标约 32,768 条路径自动选择，最多 1024 spp/逻辑批次，`1` 使用单样本逻辑批次。实际提交最多使用 4096 个路径槽，每槽推进至多 128 个状态转换，再保存状态继续；路径状态占用最多 1.125 MiB。没有通过限制散射深度、截断贡献或丢弃长路径控制 GPU 工作量。

每个像素按原来的样本索引顺序提交已完成的连续前缀，逐个更新 Welford 均值和方差，让长路径未完成时其他像素也可提前显示；完成整幅逻辑批次后才增加 host spp。中途像素的 spp 可以不同，因此导出必须等完整批次边界，且逐像素校验样本计数。批次保留 RNG 的 pixel/sample 索引、追踪算法和目标样本数，不做无序浮点求和；640×360 默认选择 1 spp/逻辑批次，同样使用有界续传。

GPU 的远地面反弹使用补偿的 world slab、进入点和距离，再进入共享 index 坐标链，避免远起点减法吞掉薄云段。静态索引/相机/平移的精度预算为 2¹⁸ voxel，动态远起点预算为 2⁴⁰ voxel；不支持的精度范围会报错而非悄悄漏云。解析镜像覆盖近地到 2⁴⁰ 的射线。CPU 使用相同 world→index 查询链划分边界，零长度交界只调整单元归属，保持距离不变。

读取器支持 OpenVDB 文件版本 222–224、`Tree_float_5_4_3`（可保存为 half）、正 uniform scale 和 uniform scale+translation、未压缩/active mask/ZIP/Blosc LZ4。其他树、grid instances、非均匀 transform、非零无限背景或不支持的压缩会明确报错。不是通用 OpenVDB 替代品。完整高分辨率数据可能需要大量内存，GPU 各 buffer 会检查设备绑定上限；超限时需显式选择较低分辨率资产。

资产保存在忽略跟踪的 `assets/` 中；没有纳入 Git。Disney 数据集采用 CC BY-SA 3.0，原始照片来自 Kevin Udy / Colorado Clouds Blog，输出元数据保留归属。分发资产或衍生结果请遵循其 [官方数据集许可与说明](https://www.disneyanimation.com/resources/clouds/)。
