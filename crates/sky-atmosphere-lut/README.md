# sky-atmosphere-lut

离线光谱大气预计算：**f32 / wgpu、41 个带内积分波段、完整相位、多重散射与 Lambert 地面反射**。复用 `sky-core` 的太阳、Rayleigh、臭氧、四种气溶胶和 1024-bin 相位表。参考输运保留 f32 光谱精度，完成后可导出线性 Rec.2020，再用 CPU 压缩显示资源；相位不在运行时后乘。

v6 完整烘焙已完成，质量迭代暂时冻结。当前 demo 默认使用约 **470 MB** 的 CPU 打包版本，保留全部网格及 f32 指数范围；编码新增 RGB 误差在本轮非网格查询中 P99 约 **0.0128%**。格式、BC6H 对比及限制见 [v6 压缩报告](../../reports/lut-compression-v6.md)。

逐阶框架参考 [Bruneton 的实现与推导](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html)。当前 `reference` 使用高度混合、太阳双地平线和拓扑对齐相位坐标，配置与实测结果见[v5 参考分配报告](../../reports/reference-lut-v5.md)。v5 完整 41 波段已完成，但 [CPU 质量验收](../../reports/reference-lut-v5-quality.md) 发现高空掠日和暮光残差，**暂不作为全域生产数据的最终精度基准**。旧版本的三方案对照和压缩探索见[v3 验证报告](../../reports/offline-lut-v3.md)。

本轮候选见 [v6 精度与性能报告](../../reports/reference-lut-v6.md)：沿射线直接积分首阶、太阳地平线坐标图、对数相位插值、气溶胶分层高度节点与局部源项插值。工作表派发和太阳盘分块减少计算成本。v6 尚未通过全域精度验收，近地蓝调地平线仍有稳定偏差。`reference` 预设仍保留 v5 网格以便复现；新候选使用下方显式 JSON 配置。

## 使用

太阳周围大圆环的前期修正见[诊断报告](../../reports/sun-ring-diagnosis.md)，v4 的后续问题见[采样审计](../../reports/lut-sampling-audit-v4.md)。v5 把球锥与地平线相切的位置对齐到固定索引，并使用单调三次相位插值。旧资源仍按旧映射读取，不能通过修改 manifest 获得修正。

在 workspace 根目录执行：

```powershell
cargo run --release -p sky-atmosphere-lut -- plan --config crates/sky-atmosphere-lut/configs/reference-v6-candidate.json --analyze-work
# 新建时省略 --resume；已有本轮部分资源时使用 --resume。
cargo run --release -p sky-atmosphere-lut -- bake --config crates/sky-atmosphere-lut/configs/reference-v6-candidate.json --out out/lut_reference_v6 --resume
# 完整烘焙结束后：
cargo run --release -p sky-atmosphere-lut -- export-rgb out/lut_reference_v6 --out out/lut_reference_v6_rgb
cargo run --release -p sky-atmosphere-lut -- inspect out/lut_reference_v6_rgb --verify
# 上述 v6 资源已存在时，接下来仅需 CPU；目标目录必须尚不存在。
cargo run --release -p sky-atmosphere-lut -- compress-rgb out/lut_reference_v6_rgb --out out/lut_reference_v6_packed16 --block-texels 16
cargo run --release -p sky-realtime-demo -- --experiment offline-lut --asset out/pt_noon_085/asset.json
```

demo 的 `--lut` 默认选择 `out/lut_reference_v6_packed16`。`--lut-budget-mib` 默认 1024，在创建 GPU 设备前检查载荷；它不包括帧缓冲等额外占用，0 可关闭检查。显式使用 1.54 GB RGB 或 21 GB 光谱源时需相应调整预算，颜色导出误差优先用 CPU 合成检查。1/2/3/4 切换 LUT、PT、绝对差异、有符号差异。Page Up/Down 切换参考资源；太阳和观察高度不匹配时禁用差异比较。介质参数已烘焙。库及 GPU 查表支持太空观察者：先将射线裁剪至大气入口，再查询边界辐亮度，真空不额外占用 LUT。

`--preset reference`（CLI 默认）使用 `BakeConfig::reference()` 的 v5 分配；旧 reference 可通过库的 `BakeConfig::reference_v4()` 取得。`development` 和库的 `BakeConfig::default()` 保留 65 节点的旧基线，适合诊断对照；`smoke` 只检查流程。`--config path.json` 接受完整 `BakeConfig`，未知字段报错。`bake --band-index 17` 可先烘焙 550 nm，后以相同配置 `--resume` 补齐。部分资源不能载入 demo。恢复会验证求解器版本、配置、模型指纹和已完成波段校验和。旧版本可读取，但不同版本不能续烘焙到同一资源中。RGB 资源不能追加光谱烘焙。

| 配置 | reference | development | smoke |
|---|---:|---:|---:|
| 高度 × 球锥位置 × 太阳高度 × 散射角 | 80 × 32 × 193 × 257 | 32 × 24 × 65 × 256 | 4 × 12 × 6 × 8 |
| 光学厚度表 | 384 × 4096 | 64 × 512 | 12 × 48 |
| 地面太阳节点 | 1024 | 256 | 16 |
| 射线 / 消光积分步数 | 256 / 2048 | 128 / 512 | 24 / 64 |
| 每个球面积分规则 | 16 × 32 | 16 × 32 | 4 × 8 |
| 有限太阳盘积分 | 4 × 16 | 2 × 8 | 2 × 4 |
| 交互阶数 | 16 | 8 | 3 |

两个球面积分规则分别朝向出射方向和太阳，通过和为 1 的权重组合，reference 每个源点共 1024 个入射样本。每次 dispatch 默认 65536 texel；较慢 GPU 可降低此值，积分精度不变。

完整 reference 光谱烘焙中间资源约 **21.08 GB**，reference 预算 **22,000,000,000 bytes**，包含元数据预留。Rec.2020 f32 输出约 **1.80 GB**，其中约 257.95 MB 是后续线段研究所需的光谱光学厚度；当前 RGB 显示不上传这部分。运行时辐亮度和太阳表约 1.54 GB，另需坐标缓存和帧缓存。每个辐亮度通道约 507.91 MB，GPU 需支持对应 storage binding 大小。这是参考表，最终生产表应通过重采样和压缩另行合成。预算不包含 PT 图和报告。

## 坐标与输运

- 高度使用近地对数与均匀高度的混合 CDF，兼顾亚米近地层和高空。短路径几何使用高度和有理化根，避免地球尺度相减。
- 太阳高度使用全域、几何地平线、暮光、反向地平线和天顶五项连续密度混合，覆盖地面和太空入口的变化。
- 方向在太阳球锥上变化。天空/地面分配 75%/25% 节点，边界不互相混合。跨高度保留视线到地平线的归一化距离和相对方位角，保护地平线及天顶端点。
- 球锥位置采用 `delta/(delta+sqrt(16/(R+h)))`，`delta=abs(mu-mu_horizon)`，单位 km；向长光程变化快的地平线加密。它与 256×256 skyview 的坐标轴含义不同，不能按单轴节点数直接比较。
- 相位坐标按前部、跨地平线和后部三个区间分配 128/96/32 个间隔，分界跟随当前高度与太阳角。内部结合前向加密与端点加密。实际输入相位的太阳附近误差有独立测试。
- 相位轴采用无过冲的单调三次插值，太阳轴在正辐亮度的对数域插值，其他轴线性插值，存储仍为线性 f32。这是有偏、非线性的近似，联合加密时趋向同一连续函数；对数插值抑制亮侧对地球阴影的污染。
- 先固定太阳方向，再重建视线，避免 f32 近竖直方向重建太阳时放大误差。

CPU 和 GPU 使用同样的映射，renderer 直接复用 baker 的 WGSL。缺少 v2 开关的旧元数据仍按 v1 坐标和线性插值读取。

v6 的 `scattering_altitudes_km` 只覆盖散射高度轴，包含零高度与大气顶，光学厚度表沿用独立坐标。`source_mapping=local_linear_height` 只改变局部体源项查询；边界辐亮度仍使用射线坐标。`height_interpolation=reference_cdf` 在新增高度节点间保持原参考 CDF；`view_interpolation=monotone_cubic` 和 `iteration_scheme=fixed_point` 是显式研究选项，本轮全谱候选没有启用。

`iteration_scheme=orders` 下，`L[n]` 表示恰好 n 次交互，体散射和地面反射各计一次。每阶先计算地面辐照度与体散射密度，再沿视线积分并累加。首阶对有限太阳盘积分；后续阶对上一阶辐亮度积分。没有经验性的多重散射增益或各向同性闭合。

`fixed_point` 选项直接迭代 `Lnew = L1 + K[Lprevious]`；首阶包含直射地面边界，后续只加间接部分。它用于比较非线性插值下的迭代差异，不使用经验性多重散射增益。

默认保存 `sum(L[1..N])`，单位 **W m⁻² sr⁻¹，每波段已积分**；不得再次乘相位或波段宽度。地面表保存匹配的 `E[0..N-1]`。直接可见太阳盘由透射表另行添加。

RGB 导出将这些波段积分为固定太阳 D65 白平衡的线性 Rec.2020，运行时仅累加三个通道。对数 / 单调三次插值与光谱积分不交换，非节点查询的误差需单独验证；太阳对数插值遇到负颜色通道时回退到有符号线性插值。太阳盘表存储 `sum(weight * solar_irradiance * exp(-tau))`，避免将 RGB 值误当光学厚度取指数。

## CPU 合成生产数据

`synthesize` 从完整光谱源生成任意查询点的线性 Rec.2020，不创建 GPU 设备。每个查询的几何模板只计算一次，按波段读取和校验数据，最多使用 16 个 CPU 线程处理查询，最后以 f32 补偿求和积分颜色。不要先导出大 RGB 表再把它当光谱参考，否则非线性插值与颜色投影的顺序误差会进入合成数据。

`points.json` 是数组，每项四个字段：`altitude_km`、`sun_elevation_deg`、`view_elevation_deg`、`relative_azimuth_deg`。海拔非负，两个高度角范围 [-90,90]，方位角相对太阳。太空射线自动裁剪；输出包含天空散射和地面，不包含直接可见太阳盘。

```powershell
cargo run --release -p sky-atmosphere-lut -- synthesize out/lut_reference_v5c --queries points.json --out out/production_samples
```

输出目录包含按查询顺序排列的 RGB 三元组 `samples.f32`（小端 f32）、`queries.json` 和带输入配置 / 指纹 / 波段校验和的 `dataset.json`。源必须完整且为光谱资源；目标目录必须尚不存在。CPU 重采样和后续 CPU 压缩不依赖烘焙 GPU，BC6H / 低秩拟合仍需独立测量误差与耗时。

`compress-rgb` 是已实现的保守显示格式：符号、f32 指数、11 位尾数的整数差值打包和相同块共享，支持 16/32/64 点块。`blocks.bin` 是 u32 字偏移表，`radiance.bin` 是可随机访问的变长块，`sun.bin` 保留三个原始 f32 太阳表。每块三个 u32 头分别包含 20 位最小编码及高位位宽，之后是三个连续通道位流；所有文件均为小端，记录独立校验和。GPU 解码节点后才插值，不展开完整 RGB 网格。`inspect --verify` 校验文件、块边界、索引和解码有限性。原始 `spectral_tau.bin` 保留在源资源中；压缩格式尚未实现有限线段的视距雾/体积光。

射线使用中点介质、源函数和单个区间的常系数解析权重。网格、角度/射线/太阳盘积分及阶数截断都有误差；只加阶数不能消除其他偏差。联合加密才趋向同一输入模型，最终受 f32、41 个输入波段和原相位表限制。每阶增量及 `relative_order_tolerance` 是诊断，绝非严格误差上界。提前停止须连续两阶通过全局峰值、总量及所有体/地面节点的局部增量检查，局部分母下限为太阳带辐照度 × 1e-12。v6 候选最多 24 阶、最少 8 阶，按波段提前结束；没有强制所有波段计算 24 阶。

## 比较

PT 当前标记 `wgpu-layered-surface-v2`，包含掠射分层和近地体顶点修正。旧 PT 可在地下漏入日照；demo 拒绝旧输运版本的差异比较，仅重建 RGB 不会更新标记。

```powershell
cargo run --release -p sky-atmosphere-lut --features reference -- compare out/lut_reference_v3 --sun-elevation-deg=-6 --width 128 --height 64 --spp 16384 --pixel-samples 8 --out out/lut_validation_v3/solar_m06.json
cargo run --release -p sky-realtime-demo -- --asset out_skyview_search/elev_020/asset.json --snapshot out/lut_validation_v3/near_sun_20.f32 --snapshot-linear --snapshot-pitch-deg 20 --snapshot-fov-deg 30 --benchmark-frames 24
```

`compare` 支持 `--band-index`、`--pt-max-orders`、`--seed`，报告原始光谱误差，分天空、太阳附近、光晕、地平线、背日阴影区域；剔除直接太阳盘像素。`--pixel-samples N` 为 LUT 每轴 N 个子像素。统计按像素未加权，包含 PT 噪声。

snapshot 默认使用 LUT；显式 `--experiment unreal-8wave` 可导出 UE 对照。加 `--snapshot-linear` 输出四个 panel 连续排列的线性 Rec.2020 RGBA f32 与 JSON，绕过曝光和显示变换，同时输出同名 PNG 预览。PNG 走真实 demo 显示管线。`--benchmark-frames N` 在资源载入后以 GPU 时间戳分别测量相机更新、太阳更新与缓存帧；避免与其它 GPU 任务并发测量。

`scripts/render_sky_comparisons.ps1` 批量导出，支持 `-Lut`、`-OutputDir`、`-Solvers` 和 `-Plan`。默认使用 v4 RGB 并写入 `out/lut_validation_v4`。`scripts/summarize_sky_comparison.py` 统计线性误差，排除太阳盘污染的 PT 源纹素。PT 全零的区域相对误差无定义，不能把零次有效采样当作无辐亮度。

独立诊断示例包括 `single_scattering_check`、`phase_integrals`、`order_convergence`、`ground_irradiance_check`。CPU 积分仅用于检查，资源输运全部由 GPU 求解。

当前资源提供到边界的辐亮度和消光表；有限线段散射及动态遮挡体积光的压缩需求另外评估。不能把两次有误差的天空查询直接相减，就宣称得到了高精度短视距雾。

## 文件与检查

`asset.json` 记录映射、单位、配置、输入指纹、波段、GPU 和逐阶诊断。`band_NNN.bin` 为 8 字节 `SKYLUT01`，之后依次为小端 f32 光学厚度、总辐亮度、地面辐照度；4D 索引最后一维连续。`.part` 完成后提交波段，再更新 manifest。FNV-1a 用于损坏检测，非安全签名。

RGB kind 为 `rec2020_atmosphere_4d_phase_inclusive_v1`。三个 `channel_N.bin` 各含 8 字节 `SKYRGB01`，之后是透射后的太阳辐照度与辐亮度。`spectral_tau.bin` 按原输入波段顺序存储光学厚度，全部采用小端 f32。manifest 保存颜色变换、原波段来源及四个校验和；所有文件完成后才写入 manifest。`inspect --verify` 会验证所有通道和保留的光谱表。

```powershell
cargo test --release -p sky-atmosphere-lut -p sky-spectral-path-tracer -p sky-realtime-demo --all-features
cargo clippy -p sky-atmosphere-lut -p sky-spectral-path-tracer --all-targets --all-features --no-deps -- -D warnings
```

测试实际运行 GPU，覆盖解析输运极限、有限太阳盘、地平线、实际前向峰、光谱隔离、CPU/GPU 查询（含太空）、资源损坏、掠射透射和蓝调地面反射。没有 GPU 会明确失败。
