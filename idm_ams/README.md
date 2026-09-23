# IDM–AMS 自动驾驶长尾评测

本目录维护“固定 ADS 控制器 + 行为世界模型”的稀有失效评测协议。它是评测工具，
不是 IDM 模型、不是新的世界模型，也不属于论文方法复现目录。

评测以 HighwayEnv 的原生 `IDMVehicle` 作为固定被测 ADS，在同一个 149 帧、25 Hz、
全背景车辆范围内运行背景交通模型，通过 EVT 风险分数定义最终失效事件，并用
Adaptive Multilevel Splitting（AMS，代码中采用 subset simulation + pCN proposal）
估计低概率长尾风险。

## 当前适配器

- `factual_hiqr`：Flow → Diffusion → 冻结事实 HiQR。该名称刻意不使用
  “hierarchical”或“CIH-WM”，因为当前 AMS 执行链尚未接入通过验收的 CIH 自主响应层。
- `trafficbots`：TrafficBots V1.5-HighD 外部行为世界模型基线。

两者的概率空间不同。固定 `K_GT` 的事实 HiQR 结果不能与 TrafficBots 的 full-prior
结果直接排名；比较器会在 test-space 不一致时拒绝生成可发表结论。

现有 TrafficBots checkpoint 按历史 `same_rear` 排除协议训练，因此默认 suite 禁用该
适配器，运行时也会拒绝把它伪装成全背景车正式结果。完成全背景车重训和验收后才可启用。

所有新评测统一使用 `highd_all_background`，包括 `same_rear`。EVT 模型必须来自
`results/highd_natural_driving_evt/`，不得混用历史 same-rear-excluded 工件。

## 目录

```text
idm_ams/
├── configs/                  # 评测协议与模型适配器配置
├── src/                      # AMS、IDM 执行、模型注册和结果校验
├── scripts/                  # 运行、审计和回放入口
└── tests/                    # 算法及协议契约测试

results/idm_ams/   # 新运行产生的结果
```

## 当前结果状态

当前没有新的正式 AMS 全量结果。旧 `IDM_subset/results/` 来自已经废止的模型命名、背景车
范围和概率空间，未迁移到新目录，避免被误认为当前结论。恢复后的算法、随机状态、模型注册、
比较和回放代码已通过 20 个单元/契约测试；`pytest` 的通过信息不等同于一次 AMS 实验。

下一次实际运行会分别写入：

- `results/idm_ams/factual_hiqr/fixed_k_gt/subset/`：事实 HiQR 的子集模拟结果；
- `results/idm_ams/factual_hiqr/fixed_k_gt/monte_carlo/`：独立 Monte Carlo 对照；
- `results/idm_ams/factual_hiqr/fixed_k_gt/test_sweep/`：完整 held-out context 诊断；
- `results/idm_ams/comparisons/`：协议一致时的跨模型比较。

这些目录由运行命令创建，不预置空目录或复制旧结果。

## 运行

```bash
conda activate tread
python -m idm_ams.scripts.run_world_models_idm \
  --development --models factual_hiqr --estimators subset
```

`--development` 允许在脏工作树中进行工程验证，并强制把结果标记为非正式。正式运行要求
clean worktree、冻结 checkpoint、匹配的哈希和全背景 EVT 契约。

核心 AMS 实现在 `src/world_subset_simulation.py`。每一层保存显式随机世界、阈值、
接受率和数值有效性；最终层的碰撞占比是条件尾部分布诊断，不能解释成无条件碰撞率。

## 当前限制

恢复本评测代码不等于当前 CIH-WM 已通过 ADS 长尾验收。现有 CIH 响应候选仍因因果方向和
剂量排序门槛失败而未晋升。只有在高精度事实层与经验证的自主交互策略合并、并为其定义
可重放随机状态后，才能增加正式的 `cih_world_model` AMS 适配器。
