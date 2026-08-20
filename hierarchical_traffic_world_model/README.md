# 扩散引导 HiQR 分层交通世界模型

本模块实现统一的三级接口：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)
\rightarrow p_\theta(\tau_{\rm soft}\mid C_0,M,K,z_{\rm diff})
\rightarrow \pi_\phi(a_{\rm bg}\mid H_t,\tau_{\rm soft},z_{\rm HiQR}).
\]

Flow 与冻结扩散共享 `C0(40)+M(6)+K(72)=118` 维物理契约。`K` 是三时刻、六背景槽位的状态结点，不再拆成额外行为符号。扩散生成六车联合 soft plan；HiQR 每 0.04 s 根据已实现历史重对齐预览，并只提交下一帧动作。EVT 只提供外部人类风险标尺，不进入模型损失。

短程层复用 HiQR 的历史—关系查询编码器、车道几何、只更新已观测状态的递归滤波器、1 s 场景潜变量、连续相关的车辆潜变量和联合 jerk-knot 解码。分层模型不再设计两套 START/ROLL 生成网络：首次调用在 `filter_state` 不存在时初始化信念状态，之后每 0.04 s 用已实现历史递推更新。soft-plan 位置、速度、参考控制与当前偏差作为 preview token 同时进入 prior 和 decoder；解码器预测短计划但只提交第一帧，最终控制满足 `j_bg=j_soft+Δj_HiQR`，不存在输出端 gate 或硬轨迹回放。

`scenario_seed` 控制 `z_scenario=(u_M,z_C0,z_K)`；`motion_seed` 独立寻址扩散噪声、场景潜变量和车辆潜变量。snapshot/restore 保存滤波状态、慢潜变量和随机流位置。模型从不接收未来 ego 动作，干预只能从下一次重规划开始生效。

## 当前证据

固定 full 配置完成后，响应层使用全部 72,771/13,133 条 recording 隔离训练/验证序列训练，并在全部 10,151 条测试序列上评价：

- 25 Hz 闭环 ADE/FDE 为 `0.0345/0.0260 m`，P95 为 `0.0781 m`；冻结扩散为 `0.0255/0.0003 m`。扩散 FDE 接近零是因为条件 K 含末端结点，不能解释为 C0-only 预测能力。
- 固定条件 16 样本的 energy score 为 `0.0264 m`，平均轨迹/末端成对距离为 `0.0256/0.0142 m`。四象限同协议消融中，随机层相对完全确定性模型的 proper energy 改善和分布退化均记录在 `randomness_ablation.json`；该结果不应与单次重建误差混为一谈。
- 制动/加速方向成功率均为 `0.921`；far/near 为 `0.087/0.097`。当前结果的剂量单调性为 `0.764/0.663`，自然响应 P10-P90 覆盖率仍为 `0.0`，因此干预目标尚未达到最初声明的严格门槛，不能把旧的 `1.0`/`0.75` 数值继续当作当前结果。生产 checkpoint 的车辆噪声相关性为 `0.999`；`0.995` 仅是诊断候选。
- 响应尺度和运动随机幅度只在 validation 上选择；锁定后，全部 10,151 条 test 的事实与干预门槛、以及固定 1,024 条 test 子集的四象限随机性门槛均通过。

当前证据支持将其视为现行数据驱动范式下的交通世界模型，但不证明任意 ADS 策略下的反事实正确性。详见 `results/hierarchical_traffic_world_model/experiment_report.md`。

### 三目标实验设计（可复现到高可视化）

- 事实/事件保真（`evaluate_world_model`）：基于完整 10,151 条 highD test 做 `ADE/FDE/P95`，并按 EVT 标签与完整语义 cut-in 分层报告；同时给出 open-loop / no-long-horizon / fixed-history 消融（对应 `figures/three_objective_evaluation.png` 与 `figures/event_fidelity.png`）。
- 分布随机性（`evaluate_randomness_ablation`）：固定 1,024 条 split，在每种条件下采 `16` 条样本，报告 `energy score`、`mean pairwise trajectory distance`、`terminal pairwise distance`、速度/加速度与 jerk 的分布一致性（对齐 SceneDiffuser 的多样本指标思路）。
- 干预有效性（`evaluate_world_model`）：三档制动/加速/横移，使用共同随机数，报告方向正确率、剂量单调性、局部性比、时滞、分布 Wasserstein 距离与自然响应覆盖率。

最新评测 schema 还会保存 149 帧事实误差曲线、8 个 highD-adapted（非官方 WOSAC）运动/交互
直方图、三档干预的 `0.2/0.4/0.8 s` 剂量响应及完整 near/far 时域曲线。运行
`visualize_world_model.py` 后生成 `factual_temporal_error.png`、
`highd_adapted_distribution_realism.png`、`quality_diversity_tradeoff.png` 和
`intervention_dose_and_locality.png`。旧 schema 的图表不得与这些新指标混用；必须重新运行
`evaluate_world_model.py` 后再可视化。

对应论文图：`figures/trajectory_reconstruction.png`、`figures/three_objective_evaluation.png`、`figures/natural_response_calibration.png`，以及
`playbacks/reconstruction_*.gif`。

## 运行

```bash
conda run -n tread python hierarchical_traffic_world_model/scripts/train_world_model.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_world_model.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_randomness_ablation.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_sampling_hierarchy.py
conda run -n tread python hierarchical_traffic_world_model/scripts/visualize_world_model.py
conda run -n tread python hierarchical_traffic_world_model/scripts/build_report.py
conda run -n tread python hierarchical_traffic_world_model/scripts/verify_world_model.py
```

正式 checkpoint、随机性消融、全量评价、校准记录和论文图统一保存在 `results/hierarchical_traffic_world_model/` 根目录。
