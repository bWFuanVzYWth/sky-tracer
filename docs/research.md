# 研究与复现指南

仓库保存可运行的方法、当前参数和能指导下一步的结论。详细运行日志、图集和大资源属于实验产物，放在 `out/`；不为每一轮调参增加一篇永久报告。

## 依赖和数据所有权

三个大气核心只依赖通用外部库，彼此没有 Cargo 依赖，也不依赖 `sky-assets`。它们各自有 `physics/`、`data/` 和求解器代码。数据是普通文件副本，不是符号链接、共享生成目录或启动时从另一算法复制。源代码相似不代表必须抽公共层：独立修改与交叉检查比消除这些重复更重要。

`sky-assets` 只规定应用交换的图像清单。参考 LUT 的存储格式、插值和打包解码归参考核心；实时的映射、波长标定、GPU shader 和缓存失效规则归实时核心；PT 的碰撞采样和随机数归 PT 核心。demo 的 shader 只做展示和颜色变换。

应用可以依赖多个核心，但只组合公开 API，分别构造输入。`sky-baker reference compare` 会检查独立输入表与参考指纹，拒绝把介质不同的结果当作同一问题对照。文件交换是研究工具的显式输入，不成为实时核心对参考核心的隐藏依赖。

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

`--config` 可以提供完整 JSON，`--band-index` 用于单波段收敛实验。完整默认配置可能需要较大的显存和长时间，优先先烘焙少量代表波段。旧数据继续按 manifest 的映射和版本解释；不要通过改 manifest 把旧数据伪装成新算法输出。

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

## 产物约定

|产物|内容和解释|
|---|---|
|PT `asset.json` + EXR|带内积分光谱、线性 sRGB 图像、白点与输运版本；PNG 只是预览|
|参考 LUT manifest + band 文件|模型指纹、坐标/求解配置、逐带校验和与求解记录|
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
