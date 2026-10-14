# 高危长尾 ADS 测试

本目录负责第三步：把第二条分支生成的高危场景条件送入第一条分支的交通世界模型，再运行
被测 ADS。

当前示例 ADS 使用 `HighwayEnv` 的 IDM 控制器。评测工具也可接入其他满足同一逐步动作接口的
ADS；IDM 不是本文的交通世界模型。

## 当前主链

```text
EVT 筛选尾部事件
        ↓
Flow 采样 M、C0、K、D
        ↓
Diffusion 在车辆尺寸条件下重建 149 帧连续软轨迹
        ↓
HiQR + 在线响应进行闭环仿真
        ↓
HighwayEnv 中运行被测 ADS
        ↓
R_SEI、EVT 分数、碰撞、最小间距和越界
```

本目录保留两种互不替代的测试空间：

- `world_subset_idm.yaml`：从尾部联合条件分布采样完整新场景，用于高危测试；
- `world_factual_k_gt.yaml`：固定测试集真实 `(M,C0,K_GT,D_GT)`，用于分析事实精度和响应能力。

高危生成不能读取 `K_GT`，否则会把测试集真实未来当成生成结果；事实评测必须保留 `K_GT`，
否则无法判断世界模型对真实轨迹的重建误差。两条路径共用 Diffusion、HiQR 和逐帧在线响应。

Diffusion 的轨迹是软参考，不是闭环车辆必须遵守的命令。NPC 受 ADS 或其他车辆干扰后，
可以通过跟驰和换道决策偏离参考；不能因为参考已经包含换道而禁止重新决策。
被测自车仍由 ADS 控制，NPC 的响应模型不接管自车。
默认换道决策的验证选择和限制见世界模型 README。更换在线响应策略后，原先的闭环危险度
和失效概率必须重算；不能仅修改模型名称后继续沿用旧结果。
当前默认组件已经统一到共享 Train 的 HiQR、标定 IDM、校准 MOBIL和学习型换道路径，
并启用参考失效后的实际状态续接。选择清单在世界模型的 `artifacts/selection.json`。

世界随机变量会显式保存初始条件、位置结点、扩散噪声、驾驶人参数和响应噪声。相同随机变量
可以逐点重放，也可以让多个 ADS 在同一世界中分别运行。

`world_subset_idm.yaml` 通过 `paths.world_model_config` 读取
`hierarchical_world_model/config.yaml`。后一个配置同时指定 Flow、Diffusion 和 HiQR
权重。`c0_base_latent` 为 26 维，只决定 `C0`；车辆尺寸从相同占位模式的尾部经验分布采样。
子集模拟的 `mutation_blocks` 还明确包含 `diffusion_noise`：每次场景条件或扩散噪声变化，
都会重新执行 Diffusion，再进入 HiQR 闭环。因此这个入口不是“只采样 Flow 后直接仿真”。

## 两种评测

- Monte Carlo：独立抽取尾部条件世界，统计给定 ADS 下的危险和碰撞；
- 子集模拟：逐级保留更危险的世界，用于寻找尾部中的更极端失效。

两者输出的 `probability` 都以 EVT 尾部世界为总体，即
`P(闭环危险度达到目标 | EVT 尾部场景分布)`。它不是全部自然驾驶中的无条件失效概率，不能直接
用来表示现实道路事故率。

EVT 先确定 Flow 的 2,964 条目标事件；闭环运行后再计算 `R_SEI`，确认生成场景在被测 ADS 下是否仍然危险。

尾部 Flow 只提高高危场景的采样比例，不保证每次闭环运行都超过 EVT 门槛，因为 ADS 和背景车
在线响应会改变实际轨迹。因此最终是否属于高危事件，以闭环后的危险度为准。

## 当前状态

当前 Diffusion 不输入尺寸；真实或经验采样的尺寸用于车辆几何和危险度。
世界模型 Test 一致性检查见 `results/hierarchical_world_model/factual_test/verification.json`，
不代表生成场景的失效概率已经得到可靠估计。

尾部 Flow 的部分分布指标仍有偏差，报告保留这些结果；单个指标超线不直接禁止使用。
入口核对 EVT-tail 来源和已发布组件是否匹配，不再要求干净 Git、发布标签或 SHA 清单。
是否适合测试，依据初始占用、连续轨迹、实际执行和尾部关系判断。
当前没有更新后完整主线的 Monte Carlo 或子集模拟结论，旧概率不能继续代表新模型。

## 运行

开发检查：

```bash
conda activate tread
python -m idm_ams.scripts.run_world_models_idm \
  --development --models factual_hiqr --estimators subset
```

固定 `K_GT` 的测试集事实评测：

```bash
python idm_ams/scripts/run_empirical_test_sweep.py
```

结果会写到：

```text
results/idm_ams/hierarchical_world_model/evt_tail_with_dimensions/
├── subset/
└── monte_carlo/
```

运行记录配置、组件来源和实际版本；选定参数后不在 Test 上重选。
运行成功不等于概率已可靠，更不等于现实道路事故率；分布偏差和估计范围需随结果说明。
