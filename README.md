# Sky Tracer：大气光传输研究

这是一个研究型 Rust / wgpu 项目。目标是在固定大气和地面参数下，渲染任意太阳位置、观察海拔和方向的天空，并为有限视距的雾提供光传输接口。GPU 输运使用 **f32**。

项目保留三个独立求解器：路径追踪检查物理输运，完整光谱 LUT 提供低噪声研究数据，四波长实时求解器负责交互显示。参考数据有偏，实时结果也有近似；两者都不能自动当作真值。

## 从哪里开始

|文档|回答的问题|
|---|---|
|[光如何形成天空](docs/atmosphere.md)|散射、光谱、四维状态是什么？三个算法与参考项目有什么关系？|
|[数值精度与数据分配](docs/numerics.md)|为什么出现圆环、地平线和地影误差？怎样拟合坐标、波长并检验结果？|
|[实时方案与压缩取舍](docs/realtime.md)|当前配置、性能、质量边界，以及 BC6H、低秩和球谐的实验结论。|
|[研究与复现指南](docs/research.md)|代码边界、数据格式、实验流程和可运行命令。|

不熟悉图形学时，先读第一篇，再运行 demo。其余文档按关注点查阅。

## 代码布局

```text
crates/
  sky-pt/          独立光谱路径追踪器；自有物理表、采样器、GPU 输运
  sky-reference/   独立光谱参考 LUT；烘焙、查询、CPU 数据合成、资源编码
  sky-realtime/    独立四波长实时算法；启动求解、SkyView、有限段 L/T
  sky-assets/      应用之间的图像清单格式；不含介质或输运
apps/
  sky-baker/       PT 图像与参考资源的命令行烘焙入口
  sky-demo/        实时显示、参考 LUT 显示、PT 对照和性能测量
  sky-audit/       固定场景评估、缓存与输运恒等式检查
  sky-optimizer/   Python / CPU 波长搜索、候选比较和图集
experiments/
  validation/     已纳入版本管理的相机与掠角回归场景
scripts/          架构检查与通用二进制图像对比
```

三个核心 crate **互不依赖，也不读取彼此的数据目录**。它们各自拥有物理表、色度数据和参数；现在表的数值相同，之后可以独立改动。对比由应用显式组织。应用只负责参数、I/O、展示和实验统计，不实现大气散射积分或 LUT 坐标解码。

## 快速运行

需要支持 Rust 2024 edition 的工具链，以及 wgpu 可用的显卡。已验证环境是 Windows、Vulkan、RTX 4090；尚未声明其他 GPU 的性能或一致性。

在仓库根目录执行：

```powershell
cargo build --release --workspace
# 无需先烘焙资源，默认使用新的实时算法。
cargo run --release -p sky-demo
# 可选：加载已有 PT 图像作对照。
cargo run --release -p sky-demo -- --asset out/pt_noon_085/asset.json
# 很小的 PT 图像，用于检查管线，不是质量参照。
cargo run --release -p sky-baker -- pt --width 32 --height 16 --spp 64 --out out/pt-smoke
# 只检查完整参考的资源预算，不启动烘焙。
cargo run --release -p sky-baker -- reference plan
```

完整参考约 **21.08 GB**，不是首次运行的前提。实时算法不加载这份资源，常驻计算载荷约 **7.910 MiB**。已有本机资源继续可读；使用参考显示时显式选择：

```powershell
cargo run --release -p sky-demo -- --experiment reference --lut out/lut_reference_v6_packed16
```

每个命令支持 `--help`。数据产物写入忽略跟踪的 `out/`，不要把个人机器的输出路径作为算法的运行依赖。旧实时实现和一次性实验已从当前树移除，需要考古时可查阅 Git 历史中的 `4c02d7c`；已有输出资源未删除。
