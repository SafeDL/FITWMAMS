# 交通世界模型

执行 Flow／Diffusion 生成的软计划，并在交通条件改变时让 NPC 自主响应：

**Flow 场景条件 → Diffusion 软计划 → 前馈反馈跟踪 → 计划适用性一次接管 → NPC 与 HighwayEnv 执行。**

桥接判断原计划是否需要调整，NPC 决定怎样响应。合理响应后暴露的 ADS 失败应保留；桥接参数不按事故数量选择。

## 跟踪与接管

跟踪器用 21 帧、0.04 秒步长的三次 Savitzky–Golay 局部多项式求参考速度与加速度。纵向使用加速度前馈和位置、速度反馈；横向使用参考偏航率、航向与位置反馈，两轴频率均为 1 s⁻¹。动作经统一动力学执行。

桥接每步预测未来 1.5 秒。未接管 NPC 使用自己的剩余计划与当前跟踪偏差；ADS 和已响应 NPC 使用实际历史外推。每对车辆配对计算 16 组完整经验误差轨迹，保留单条误差内部的时间与方向关联，不需要 ADS 的未来动作。

| 首次接管理由 | 判断方式 |
| --- | --- |
| 计划占用冲突 | 经验轨迹下发生车身相交；跟驰接管后车，计划换道接管主动进入冲突区域的 NPC。 |
| 跟驰条件变化 | 已建立的近处领车加速，继续本车计划仍使间距明显扩大。 |
| 通道释放 | 原领车实际离开本车通道，而计划尚未开始相应加速或换道。 |
| 参考到期 | 计划耗尽后交给 NPC，单独记录。 |

当前冲突支持度门限是 `1/16`，一组完整预测冲突即可满足该项条件；它不是多数共识或事故概率。两种非碰撞分数按 Train 尺度归一化，共用门限。计划中已有的响应进入预测，无关邻车离开不触发通道释放。

每辆 NPC 最多接管一次，之后自主运行到回合结束。交接保留驾驶特征、实际历史、随机状态和已开始的换道。桥接与 NPC 共用车辆关系：真实领车使用正净间距，并排与切入对象保留有符号相对位置，避免虚假极小跟驰间距。

## 代码入口

| 位置 | 用途 |
| --- | --- |
| `src/composition.py`、`src/planner.py` | Flow／Diffusion 组合、冻结计划生成与缓存 |
| `src/tracking.py`、`src/highway.py`、`src/execution.py` | 跟踪、动力学和 ADS 闭环 |
| `src/plan_applicability.py`、`src/takeover.py` | 预测评分、一次交权与机动承接 |
| `src/scene_geometry.py`、`src/motion.py`、`src/error_trajectories.py` | 共享几何、短历史外推和经验误差库 |
| `src/stochastic_drivers/` | NPC 跟驰、换道及原始数据提取 |
| `src/npc_experiments/`、`src/bridge_evaluation.py` | NPC 标定、自主交通评测及桥接实验 |

在项目根目录运行，参数见 [config.yaml](config.yaml)：

```bash
conda activate tread
python hierarchical_world_model/scripts/calibrate_error_trajectories.py
python hierarchical_world_model/scripts/calibrate_drivers.py
python hierarchical_world_model/scripts/calibrate_bridge.py
```

三个入口依次更新完整 Train 经验误差库、纵向响应修正和 Train／Val 门限；IDM 人群、MOBIL、换道模型、生成模型与跟踪器保持冻结。误差窗口暂存在 `bridge/.cache/error_trajectories/`，模型与报告保存成功后清理。

```bash
python hierarchical_world_model/scripts/evaluate_bridge.py
python hierarchical_world_model/scripts/evaluate_bridge_interventions.py
python hierarchical_world_model/scripts/render_bridge_playbacks.py
```

分别评估自然 Test、脚本干预机制和生成配对 GIF。干预实验改变主车实际控制，不能当作真实 ADS 的评测成绩。独立 NPC 交通统计使用 `scripts/evaluate_drivers.py`；IDM／DQN 的接入入口见 [ads](../ads/README.md)。

`HierarchicalWorldSampler.create_world(..., controller="auto")` 与 ADS `FrozenWorldFactory` 加载相同发布参数；`controller=None` 执行纯跟踪。所有权、首次理由、机动及 NPC 内部状态均进入诊断和快照。

## 方法与证据

预测冲突后干预回放可参考 [LitSim](https://arxiv.org/abs/2403.04299)，相关车辆的交互响应可参考 [InterSim](https://arxiv.org/abs/2210.14413)。本实现的重点是完整经验轨迹支持的计划适用性，以及车辆关系、机动和状态一致的一次交接。跟踪、IDM 与 MOBIL 是采用的基础方法。

当前已有完整自然 Test 及针对性干预证据，最新版生成场景 ADS 全量对照尚未完成。数据范围、历史诊断和原始报告统一见[结果说明](../results/hierarchical_world_model/README.md)，具体指标以报告为准。
