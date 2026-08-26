# IDM_subset：HighwayEnv 中的 IDM 长尾估计

正式执行契约为：Flow、Diffusion 和 HiQR 只提供场景、运动先验和背景车动作；
IDM ego 与背景车都必须由本地 `HighwayEnv` 在同一条道路上推进。IDM 使用原生
`IDMVehicle`；背景车使用与 HiQR 训练一致的单轮 `[acceleration, yaw_rate]` 动力学，
同时复用 HighwayEnv 的道路、碰撞与 snapshot。`HighwayEnvTraffic` 已通过动作映射、
积分一致性、IDM 接入、因果时序及分支精确重放测试。

当前估计对象为：

```text
Omega = (u_M, z_C0, z_K, z_diff, z_scene_response, z_agent_response)
Omega ~ p(M)p(C0|M)p(K|C0,M)p(z_diff)p(z_response)
Flow -> Diffusion -> HiQR background actions -> HighwayEnv + IDM -> risk -> EVT score
```

失效事件固定为人类 EVT 模型的 100-event return level：

```text
S_EVT(Y_IDM(Omega)) >= S_EVT(z_100)
```

`world_subset_simulation.py` 的 pCN/离散场景 mutation 可继续复用；正式 runner
还必须保存 HighwayEnv 车辆状态、动作转换和 snapshot，才能保证完整重放。

## 运行

唯一维护配置为 `IDM_subset/configs/world_subset_idm.yaml`；脚本目录不再保存配置文件。

```bash
conda activate tread
python IDM_subset/scripts/run_subset_idm.py
python IDM_subset/scripts/run_monte_carlo_idm.py
```

正式结果位于 `IDM_subset/results/`，且使用同一套 Flow、Diffusion、
HiQR、IDM、EVT 和 HighwayEnv 工件。该目录是可再生运行输出，不提交到 Git；请以
本次运行写出的 summary、top-cases 和 playback manifest 为准，而不是在文档中维护
容易过期的固定概率数字。

此前 `current_world_idm/` 的 AMS、Monte Carlo 和碰撞 GIF 来自自定义运动学
`ClosedLoopWorld`，已确认不符合正式执行契约并移除，不能用于论文或策略风险比较。

```text
IDM_subset/results/subset/                         # 正式 AMS
IDM_subset/results/monte_carlo/                    # 正式独立 Monte Carlo
IDM_subset/results/subset/playbacks/               # AMS 最终种群的 HighwayEnv 回放
IDM_subset/results/risk_calibration_diagnostic/    # 独立 highD 风险校准诊断播放
IDM_subset/results/acceptance.json
```

`subset/playbacks/` 是从 `world_subset_top_cases.json` 指向的最终 AMS 粒子重新在
HighwayEnv 执行得到的碰撞回放；它与 subset 结果使用相同的 `world_exogenous_state`。
`risk_calibration_diagnostic/` 中的风险校准播放不是 subset 碰撞样本的复现：它来自固定的 highD 测试窗口，
用于检查观测人类控制与 IDM 反事实控制的差异；subset 的碰撞样本则保存在
`subset/failure_cases/`，二者不共享样本 ID。

渲染正式 AMS 回放：

```bash
python IDM_subset/scripts/render_subset_playbacks.py
```

脚本职责保持单一：

| 脚本 | 用途 |
| --- | --- |
| `run_subset_idm.py` | 在 HighwayEnv 中运行正式 pCN/AMS subset simulation |
| `run_monte_carlo_idm.py` | 在相同世界先验和执行后端下运行独立 Monte Carlo |
| `render_subset_playbacks.py` | 将 AMS 最终种群的代表性失败粒子重新执行并渲染为 GIF |
| `assess_current_world_idm.py` | 检查两种估计的阈值、工件、重放和置信区间一致性 |

正式两种估计必须使用相同的 Flow、Diffusion、HiQR、HighwayEnv、EVT、IDM policy、
failure threshold 和 `hiqr_vehicle_dynamics_contract`；各 summary 还应写入提交、工件哈希和动作映射审计。
