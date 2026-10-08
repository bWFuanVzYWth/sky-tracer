# 研究与复现指南

仓库保存可运行的方法、当前参数和能指导下一步的结论。详细运行日志、图集和大资源属于实验产物，放在 `out/`；不为每一轮调参增加一篇永久报告。

## 依赖和数据所有权

三个大气核心只依赖通用外部库，彼此没有 Cargo 依赖，也不依赖 `sky-assets`。它们各自有 `physics/`、`data/` 和求解器代码。数据是普通文件副本，不是符号链接、共享生成目录或启动时从另一算法复制。源代码相似不代表必须抽公共层：独立修改与交叉检查比消除这些重复更重要。

`sky-assets` 只规定应用交换的图像清单。参考 LUT 的存储格式、插值和打包解码归参考核心；实时的映射、波长标定、GPU shader 和缓存失效规则归实时核心；PT 的碰撞采样和随机数归 PT 核心。demo 的 shader 只做展示和颜色变换。

应用可以依赖多个核心，但只组合公开 API，分别构造输入。`sky-baker reference compare` 会检查独立输入表与参考指纹，拒绝把介质不同的结果当作同一问题对照。文件交换是研究工具的显式输入，不成为实时核心对参考核心的隐藏依赖。

`cloud-demo` 在应用层组合 `sky-realtime::Renderer::render_environment` 与 `cloud-pt::ProgressiveRenderer::set_environment`。前者求解大气并输出方向光和太阳透射率，后者只读取通用线性 RGB 纹理；色彩/照度换算和画面投影归应用。云核心仍不依赖大气核心或其数据，SkyView CDF 和大气积分仍归实时核心。这个接口定义冻结的远场边界，不提供云内大气的联合路径追踪。

优化器是 Python CPU 应用：它实现拟合和实验统计，不实现大气输运。光谱数据生成的单散射积分位于参考核心 `spectral_dataset` / `direct`，命令行只负责路径和参数。这样搜索目标可以改，天空的物理定义仍有明确归属。

## 独立数据的来源和演化

每个核心的数据包含大气/气溶胶剖面、光谱带信息、光学系数、Mie 相位和 CIE 1931 2° 色度表。当前三个副本来自同一组冻结输入，参考模型指纹为 `18e1e068af347c04`。CIE 表与太阳白点适配共同决定最终色度，比较时不能只匹配波长名称。

原始 libRadtran / OPAC 到 CSV 的生成程序保留在 [`sky-pt/tools/generate_tables.py`](../crates/sky-pt/tools/generate_tables.py)。它需要外部原始数据及 NumPy、SciPy、miepython，不是构建时依赖。修改物理输入应在对应核心内生成、检查和提交；需要三算法对照时显式更新各自副本并记录新指纹。不要让构建脚本自动同步三份输入，否则一次错误会同时污染所有参照。

当前数据文件本身是可复现运行的输入。原始外部数据库及其生成环境未全部随仓库分发，不能据此声称可以无外部依赖重建每一个物理系数。

## 最小检查

在仓库根目录执行。GPU 测试需要可用适配器，构建本身不需要预先生成 `out/`。

```powershell
python scripts/check_architecture.py
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --lib --bins --tests
cargo build --release --workspace

# 43 个固定相机，保存逐像素积分和 SkyView 两种结果。
target/release/sky-audit.exe evaluate --queries experiments/validation/sky_cases.json --out out/check-sky
# 缓存失效、介质更新、真空和有限段组合。
target/release/sky-audit.exe validate --out out/check-sky/invariants.json
```

更广覆盖使用 `sky_sweep.json`（425 视图）或 `grazing.json`（112 视图）。后者可以加 `--aerosol-scale 4` 检查高密度长路径。它们是已用于开发的回归集，不能声称为新的盲测。新方案还需增加未参与选择的太阳/海拔轨迹。

比较两次输出，不要比较经过不同曝光的 PNG：

```powershell
python scripts/compare_images.py out/baseline out/check-sky --queries experiments/validation/sky_cases.json --out out/check-sky/difference.json
```

该工具要求两边文件齐全，检查有限值，分别统计 source / sky。报告逐图 P95/P99/最大差，并保留是否逐位一致。它不运行 GPU，也不把较密解自动标为真值。

同轮比较优化前后的求解成本和不同缓存失效事件，不含设备/管线创建和回读；输出文件必须尚不存在：

```powershell
target/release/sky-audit.exe benchmark --configs crates/sky-realtime/configs/convolution_baseline.json crates/sky-realtime/configs/balanced.json --startup-repeats 6 --runtime-rounds 3 --warmup-frames 4 --frames 64 --out out/realtime-cost.json
```

startup 保存各阶段 GPU 时间与整个 rebuild 墙钟；runtime 分别测逐像素积分、太阳变化、海拔变化、仅相机变化和相同输入缓存复用。runtime 的 GPU 区间只包含核心 compute，不含 demo 的显示 pass。输入不变时没有 dispatch，计时值仅是 timestamp 标记开销，不能报告成一次实际天空计算。

固定大气、连续移动太阳的完整 1080p 纯天空成本使用 `profile`；默认包含 SkyView、投影、太阳盘和原有显示 shader。CPU 坐标构造及含编码/提交/等待的整帧墙钟另列，诊断回读在计时外；不含 UI 与 vsync。逐个运行，避免 GPU 竞争：

```powershell
target/release/sky-audit.exe profile --steps 96 --out out/sky1080-baseline
# 未指定 --steps 时，与 demo 一样使用 64 步默认。
target/release/sky-audit.exe profile --out out/sky1080-default
# 14 个真正 1080p 的敏感姿态；中点 capture 采用与开发验收相同曝光。
target/release/sky-audit.exe profile --trajectories experiments/validation/moving_sun_1080_anchors.json --frames 3 --warmup-frames 1 --rounds 1 --quality-frames 1 --out out/sky1080-anchors
```

`inputs.json` 保存配置、代码/shader 指纹、实际步数、相机、曝光和适配器；`profile.json` 保存 GPU 分阶段、CPU 时间和逐帧样本。线性与实际 SDR capture 由独立的非计时重放生成。默认性能轨迹相机全为纯天空；`moving_sun_sensitive.json` 另覆盖暮光、地影和 108/400 km 薄边缘。14 个敏感姿态包含从连续序列中发现的最坏帧，属于开发回归而非盲测。

连续误差验收可在 CPU 生成冻结配置与 8×65 查询，然后显式对照旧 96 步与新 64 步：

```powershell
python apps/sky-optimizer/prepare_moving_sun_candidates.py --out out/sky-motion-plan
target/release/sky-audit.exe evaluate --config out/sky-motion-plan/baseline_config.json --steps 96 --queries out/sky-motion-plan/queries_temporal.json --out out/sky-motion-baseline
target/release/sky-audit.exe evaluate --config out/sky-motion-plan/baseline_config.json --steps 64 --queries out/sky-motion-plan/queries_temporal.json --out out/sky-motion-default
python apps/sky-optimizer/compare_sky_sequences_cpu.py out/sky-motion-baseline out/sky-motion-default --queries out/sky-motion-plan/queries_temporal.json --out out/sky-motion-difference
```

比较工具同时报告静态相对差、候选减基线的一阶/二阶时间残差、相邻行残差及最坏帧热图。时间残差先扣除基线本身的自然变化，不能把太阳移动造成的明暗变化全归咎于候选。线性图以每轨迹固定亮度 floor 区分亮天空和近真空；数值尾部需要结合固定曝光、原尺寸显示图审查。工具也支持两组相同 `profile` capture，额外比较实际 SDR 色阶，省略 `--queries` 即可。

## 烘焙与参考查询

```powershell
# 无 GPU 工作；先了解预算。
target/release/sky-baker.exe reference plan --analyze-work
# 小表验证实际 GPU 和序列化。
target/release/sky-baker.exe reference bake --preset smoke --out out/reference-smoke
target/release/sky-baker.exe reference inspect out/reference-smoke --verify
# 完整研究配置；首次创建不加 --resume。
target/release/sky-baker.exe reference bake --out out/reference
# 中断后仅恢复同模型、同配置的已校验资产。
target/release/sky-baker.exe reference bake --out out/reference --resume
```

`--config` 可以提供完整 JSON，`--band-index` 用于单波段收敛实验。完整默认配置可能需要较大的显存和长时间，优先先烘焙少量代表波段。新教师表尺寸上界约 19.001 GB，含容器头和元数据，硬上限为十进制 20 GB；`plan` 会先拒绝超限配置。旧数据继续按 manifest 的映射和版本解释，旧二进制容器保留只读兼容；新烘焙及续烘使用 safetensors，旧表迁移需在新目录重新烘焙。不要通过改 manifest 把旧数据伪装成新算法输出。

采样次数的独立扫描入口，使用新的 JSON 输出路径：

```powershell
target/release/sky-baker.exe reference sampling-study --out out/teacher-study.json --scattering 40 16 49 65 --bands 450 550 650 --warmup --repeats 2 --only baseline balanced_logheight_192
```

命令先求较密积分对照，再测指定候选，保存每阶墙钟、固定物理查询的辐亮度、差异分位数和暗部绝对差。`--only` 支持分别扫描 ray、angular、sun、tau 和 orders；`--optical-depth` 可改变共享辅助表网格。完整参考的 GPU 工作集远大于这些小表，不得直接把秒数按状态数量外推。取舍与已测参数在[数值精度与数据分配](numerics.md#采样预算与烘焙时间)中。

CPU 合成与参考显示转换：

```powershell
# queries 为 SamplePoint 数组，单位 km / degree。
target/release/sky-baker.exe reference synthesize out/reference --queries experiments/samples.json --out out/samples
target/release/sky-baker.exe reference export-rgb out/reference --out out/reference-rgb
target/release/sky-baker.exe reference compress-rgb out/reference-rgb --out out/reference-packed
```

`experiments/samples.json` 是使用者提供的数组，例如：

```json
[{"altitude_km":0.2,"sun_elevation_deg":-6,"view_elevation_deg":1,"relative_azimuth_deg":180}]
```

合成先逐光谱插值再积分 RGB。先把整个节点表转为 RGB 再做非线性插值一般不等价，这项误差与压缩量化误差要分开测。参考显示包适合天空和地面边界查询，不提供通用有限视距雾。

## 光谱拟合

```powershell
python -m pip install -r apps/sky-optimizer/requirements.txt
python apps/sky-optimizer/main.py queries out/fit-queries.json
# CPU 生成 Ltotal / single / boundary 光谱，不重烘焙；全量查询可能较慢。
target/release/sky-baker.exe reference spectra out/reference --queries out/fit-queries.json --out out/fit-dataset
# 穷举三/四波长，附等距八波长控制组；无须先搜八波长。
python apps/sky-optimizer/main.py counts out/fit-dataset --out out/fit-counts
# 可选的多起点八波长搜索，保留不同区域取舍的候选。
python apps/sky-optimizer/main.py search out/fit-dataset --free-span --out out/fit-eight
python apps/sky-optimizer/main.py preview out/fit-counts --out out/fit-gallery --heat-max-percent 10
```

`counts --base out/fit-eight` 可加入八波长搜索结果作比较。输出候选 JSON、全部可行三/四节点组合的损失/权重、验证指标和图集。候选通过独立图像与运行时检查后，才把所选参数显式纳入实时核心自己的配置；工具不会自动覆盖默认波长。

所有输出命令应使用新的目录。参考的 `--resume` 是检查过模型、配置和校验和的专用恢复路径，不能用来混合不同实验。

v7 教师的波长重新拟合保留原有训练/验证点，先把不参与拟合的视觉图缩小以控制 CPU 工作量，再穷举 10,660 个三波长和 101,270 个四波长组合。四波长仍选当前节点，权重变化不足 0.001%，所以保持已验证的默认权重。没有把更密实时解当作物理真值。

## 从现有教师拟合 SkyView 坐标

重新分配节点先隔离插值误差。`export_sky_mapping_cpu` 只在 CPU 查询已有教师，无设备、无重烘焙、无重复单散射积分。184 条密集高度/太阳/方位曲线约 80 万点，完整 41 波段 RGB 导出本机耗时 35.47 s；四波段代理可先研究，但正式拟合采用完整光谱。训练与留出按物理海拔/太阳姿态划分，同一姿态的方位不跨划分。

```powershell
python apps/sky-optimizer/fit_sky_coordinates_cpu.py prepare --out out/sky-coordinates-queries
cargo run --release -p sky-audit --example export_sky_mapping_cpu -- --source out/teacher_reference_safetensors_v7 --queries out/sky-coordinates-queries/queries.json --out out/sky-coordinates-teacher --threads 8 --full-spectrum
python apps/sky-optimizer/fit_sky_coordinates_cpu.py fit --dataset out/sky-coordinates-teacher --sizes 256 224 --sky-only --seed 20261008 --iterations 320 --out out/sky-coordinates-fit
```

只调整现有 `low/upper/space` 的四个权重和三个宽度，保留 softsign CDF 公式、地面映射及源场坐标。天空/地面端点分别从各自边界一侧查询；大气外的数学外切点设为精确真空，最后一个 cell 与 shader 的弦长外推一致。报告明确区分近真空的绝对误差 floor 与亮天空相对差；它只评价固定方位上的垂直插值，不能替代真实四波长积分、水平插值与显示质量检查。

候选不用覆盖生产 JSON。`evaluate/profile/benchmark --mapping CANDIDATE.json` 在任何映射初始化前加载、验证并冻结整份校准；同一进程内不可切换，避免已有 GPU 场和 CPU/GPU 坐标不一致。输出记录实际校准内容。完整天空成本、静态回归与独立运动轨迹分开执行，GPU 不并行：

```powershell
python apps/sky-optimizer/audit_sky_mapping_gpu.py --phase static --candidate-dir out/sky-coordinates-fit --out out/sky-coordinates-static
python apps/sky-optimizer/audit_sky_mapping_gpu.py --phase cost --candidate-dir out/sky-coordinates-fit --out out/sky-coordinates-cost
# 只对完成静态选择的候选使用这组独立轨迹；它不参加拟合或挑选参数。
python apps/sky-optimizer/audit_sky_mapping_gpu.py --phase motion --candidate 256 --candidate-dir out/sky-coordinates-fit --out out/sky-coordinates-motion
```

运动留出见 `moving_sun_holdout.json`。每个 GPU 子进程完成后才启动下一个；失败或超时会停整批。旧数值点只有稀疏的方向线，不能直接认证 256²→224² 的细节。实际 GPU 比较还要求 source/optical 载荷不变，检查整幅 HDR、实际 SDR、相邻行和时间残差。

runner 默认使用 `experiments/validation/sky_mapping_candidates_v7/` 中冻结的旧映射、配置、波长和候选；`--candidate-dir` 则审查上面的新拟合产物。43 个标准与 22 个额外查询已版本化，冷复现不需要先前的 `out` 回归目录。`--baseline-mapping/--config/--wavelengths` 可以显式替换实验输入，输出记录实际输入。不要在生产默认变更后把新默认当成旧基线。

```powershell
python apps/sky-optimizer/analyze_sky_mapping_cpu.py --static out/sky-coordinates-static --cost out/sky-coordinates-cost --out out/sky-coordinates-report
python apps/sky-optimizer/compare_sky_sequences_cpu.py out/sky-coordinates-motion/baseline_holdout out/sky-coordinates-motion/256_holdout --allow-sky-mapping-change --out out/sky-coordinates-motion-diff
```

多散射源需要单独处理。沿一条观察射线的 `total−single−boundary` 不是局部源；完整教师的方向辐亮度与已知局部散射系数/相函数可以通过角向卷积恢复该源，再得到归一化源、均值与高空矩。也可以恢复当前实时场的完整导出和 CPU 采样器，做网格误差诊断。本轮先优化更新频率高的 SkyView，保留源场的高度/太阳/相位/绕轴坐标；没有把终端辐亮度差伪装成源场数据。

## 产物约定

|产物|内容和解释|
|---|---|
|PT `asset.json` + EXR|带内积分光谱、线性 sRGB 图像、白点与输运版本；PNG 只是预览|
|参考 `asset.json` + `band_NNN.safetensors`|模型指纹、坐标/求解配置、完整文件校验和与求解记录；F32 张量 `radiance[height,view,sun,phase]`、`optical_depth[height,view]`、`ground_irradiance[sun]`，末轴最快|
|RGB / 压缩 LUT safetensors|RGB channel 存 `solar_irradiance` / `radiance`，`spectral_tau` 保留全波段光学厚度；压缩包 `blocks` / `radiance` 存 U32 编码，`sun` 存 F32 太阳表，GPU 解码方式保持不变|
|RGB 合成 `dataset.json` + `samples.f32`|小端 f32，查询顺序的线性 Rec.2020 三分量；无可见太阳盘|
|光谱拟合数据|`total/single/boundary.f32` 为波段优先的带内积分量；`queries.json` 保存训练/验证/图像划分|
|audit 图像|`<scene>_source.f32` / `_sky.f32`，小端 f32 RGBA；相机在查询 JSON 中|
|demo 线性快照|四面板 RGBA：求解、PT、绝对差、有符号差；JSON 保存颜色空间、视角与 GPU 时序|

demo 快照示例：

```powershell
target/release/sky-demo.exe --asset out/pt_noon_085/asset.json --snapshot out/demo-check.f32 --snapshot-linear --benchmark-frames 64
```

PT 对照必须来自正确输运版本。以往 PT 的地面半球处理、层状跟踪等错误说明：参考程序同样需要解析极限、地面边界和掠角测试，不能只凭“蒙特卡洛”三个字相信输出。

## 如何继续做实验

在所属核心内实现输运或坐标变化，应用只暴露有解释的参数。优先使用小而具体的配置和实验脚本，不引入提前设计的通用求解框架。需要全新方法时，可以先建独立实验目录与局部 manifest；达到稳定功能后再决定是否成为 workspace 的新核心。

一次实验至少记录输入数据指纹、配置、shader/代码版本、适配器、查询集以及指标定义。先保存基线，再改一个主要因素；比较时避免并行占用 GPU。结果分为“实际测量”“尺寸估算”“理论推测”，不要混写。没有稳定收益的分支应删掉，结论写回相关主题文档。

本机既有证据可在 `out/hybrid_budget_v1`、`out/realtime_comparison_v1`、`out/wavelength_counts_v1` 中查到，但这些目录不随源码分发。关键参数、场景与结论已纳入当前树；若缺少旧产物，就按上述入口建立新基线，不把失效的本机路径当成必需依赖。
