# IDM_subset：多世界模型的 HighwayEnv IDM 长尾评测

本目录提供一个共同的执行和风险协议，并维护两个背景交通世界模型适配器：

- `hierarchical`：本文的 Flow → Diffusion → HiQR 分层世界模型；
- `trafficbots`：TrafficBots V1.5-HighD 外部基线。

二者都使用原生 HighwayEnv `IDMVehicle`、149 步/25 Hz 执行、同一 EVT 模型和
`highd_follower_excluded_v1` 作用域。背景车均按 `[acceleration, yaw_rate]` 推进。
当前正式链是分层模型的 `empirical_test_fixed_k_gt` 重建型测试：从 held-out highD
序列保留 `C0`、地图和真值 soft plan，仅采样响应随机性。TrafficBots 没有 `K` 输入，
只能运行自己的 causal full-prior 测试；它不能与 fixed-K 结果比较，也不会出现在默认
套件或论文主表中。

所有当前测试统一采用 `highd_follower_excluded_v1` 作用域：训练和既有权重保持不变，
但在模型推理与仿真开始前将 `same_rear` 清零并置为 invalid。碰撞、最小间距、SEI/EVT、
AMS/Monte Carlo 和回放均继承同一个 `initial_valid`，不得仅在结果表中事后过滤。
自然驾驶 EVT 必须来自 `results/highd_natural_driving_evt_same_rear_excluded/`；旧的
all-slot EVT 与结果不可混用。

## 配置和环境

统一套件配置是：

```text
IDM_subset/configs/world_models_idm.yaml
├── hierarchical -> configs/world_subset_idm.yaml
└── trafficbots  -> configs/trafficbots_subset_idm.yaml
```

独立环境不会修改项目主 `tread` 环境：

```bash
conda env create -f IDM_subset/environment-world-models-idm.yml
conda activate world-models-idm
```

正式 runner 要求 clean worktree 并验证模型 release manifest。开发中脏工作树会被明确
阻止，不能产生可误用的“正式”概率。

若需在尚未冻结的代码上做工程实验，可显式运行：

```bash
python IDM_subset/scripts/run_world_models_idm.py --development --estimators subset
```

该模式不会削弱正式入口；summary 会写入 `formal=false`、commit、dirty 状态和配置哈希。

## 执行、比较和可视化

执行默认启用的模型：

```bash
python IDM_subset/scripts/run_world_models_idm.py --estimators subset
```

全量 held-out context 的一次固定-CRN 诊断使用：

```bash
python IDM_subset/scripts/run_empirical_test_sweep.py
```

重放 fixed-K 测试空间中的多样化失效案例：

```bash
python IDM_subset/scripts/render_subset_playbacks.py \
  --model hierarchical --selection test_sweep_diverse --overwrite
```

## 结果组织

```text
IDM_subset/results/hierarchical/fixed_k_gt/
├── subset/                 # AMS development run
├── test_sweep/             # every held-out context, one fixed CRN draw
└── acceptance.json
```

每个 playback manifest 保存模型 provenance、统一视觉契约和对应的
`world_exogenous_state`，可精确回放。AMS 最终种群是条件尾部分布；其 collision fraction
不能解释为无条件碰撞概率。

## 概率空间

分层模型：

```text
Omega_H = (j, z_diff, z_scene_response, z_agent_response)
j ~ Uniform(held-out test contexts); retain (C0_j, M_j, K_GT,j)
(C0_j, M_j, K_GT,j) -> Diffusion -> HiQR -> HighwayEnv + IDM -> risk -> EVT score
```

TrafficBots standalone causal prior：

```text
Omega_TB = (u_M, z_C0, Z_personality, u_destination)
S0 + Z + G(S0,u_destination) -> TrafficBots -> HighwayEnv + IDM -> risk -> EVT score
```

两种测试各自使用相同定义的失效事件：

```text
S_EVT(Y_IDM(Omega)) >= S_EVT(z_100)
```
