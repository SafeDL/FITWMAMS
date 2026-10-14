# 分层交通世界模型

将背景车软计划变成实际交通运动，并在交互发生时调整动作。车辆通过 highway-env 以 25 Hz 执行，不逐帧回填真值位置。

## 方法主链

Diffusion 软计划 → HiQR 联合动作 → NPC 响应与计划进度协调 → highway-env 执行。

- `src/model.py`、`src/hiqr/`：结合实际历史、车辆关系和软计划，生成各车动作。
- `src/stochastic_drivers/`：IDM 跟驰、MOBIL 换道选择及换道轨迹；每车采样的驾驶参数在一次运行中保持不变。自主道路实验与正式响应层共用这套实现。
- `src/reference.py`、`src/highway.py`：交互后按实际进度调整计划时序，再执行动作。
- `src/shared_evaluation.py`：事实执行指标及按录像聚类的统计；NPC 分布和事件统计统一在 `src/npc_experiments/`。

NPC 标定采用“自然行为学习 → 闭环分布优化”，比较速度、跟驰交互、实际换道率和横向运动，加速度及风险指标作为辅助诊断。实现位于 `src/npc_experiments/`。

ADS 运行时不预先划分回放车和响应车：所有 NPC 先由 HiQR 跟随软计划，再按实际邻车、预计净间距和已发生的加减速逐帧决定是否接管。纵向制动与横向换道分别协调；未受影响车辆保留 HiQR 动作，自主换道仅在响应需要时启动。交互后按实际空间进度调整参考；不可达的旧纵向时序被弃用，动作平滑退出响应，已完成的自主换道不会被原计划拉回。

## 数据与验证

- **事实重建**：完整 Test 的 10,151 个场景，以真实初态、道路、尺寸和 `K_GT` 经 Diffusion 构造计划，检查最终执行误差。
- **NPC 自主交通**：全部车辆由 NPC 策略驾驶，不使用软计划或 ADS。覆盖 60 段录像、120 个道路方向，分别连续运行 5.96 秒和 120 秒，每方向各重复 16 次。
- **ADS 应用**：EVT → Flow → Diffusion 生成测试场景，自车交给 ADS，背景车由世界模型执行。

HiQR 和当前 NPC 均为 Train 拟合、Val 选择；覆盖全部录像的 NPC 评测不代表全量重拟合。训练与标定输入放在[共享数据目录](../results/highd_shared_training_data/README.md)，权重和发布参数放在结果目录。

## 入口与结果

项目根目录运行，环境为 `conda activate tread`：

```bash
python hierarchical_world_model/scripts/train.py --stage base
python hierarchical_world_model/scripts/train.py --stage stochastic_heads
python hierarchical_world_model/scripts/publish.py
python hierarchical_world_model/scripts/evaluate.py
python hierarchical_world_model/scripts/verify.py
```

NPC 标定与评测使用两个入口；全部设置集中在 [config.yaml](config.yaml)：

```bash
python hierarchical_world_model/scripts/calibrate_drivers.py
python hierarchical_world_model/scripts/evaluate_drivers.py
```

指标、PNG 和原速 GIF 见[结果说明](../results/hierarchical_world_model/README.md)。评测完成后合并分片，保留可复算统计。

ADS 接口为 `HierarchicalWorldSampler`、`rollout_world`；可选 `response_gate` 不由默认世界模型加载。
