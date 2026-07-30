# QR-WM Training Architecture

## 1. 运行边界

QR-WM 使用 highD 固定的 `[ego, six background slots]` 场景张量。ROLL 输入为真实已发生的交通历史、道路 polyline/topology 与 ADS 控制；START 输入仅为 Flow 的当前 C0、B0 与道路，不伪造历史帧。不输入交通灯，也不加载 RAMP/FIRM checkpoint。训练监督为未来背景状态和由 highD action 投影得到的背景控制。

Flow 不参与主训练的密度优化，但其冻结 76-D `C0+B0` schema 是正式 START 条件。训练、验证和测试均读取按同一 schema 生成的只读 B0 sidecar，以免 Flow 推理与训练条件不一致。

## 2. 联合模型

1. **Relation-aware scene encoder**：历史状态经多头 temporal attention 编码；两层 relation-aware multi-head self-attention 对车辆关系使用可学习 pairwise bias；随后每个 agent 对 agent/map token 做多头 cross-attention。
2. **Persistent scene memory**：唯一持续状态满足
   \[
   m_t=f(m_{t-1},S_t,U_{t-1},a_{t-1}^{ego}).
   \]
   它保存交通交互、既有计划和 ego 干预响应。QR-WM 没有独立 world memory。
3. **Behavior prior**：CVAE prior 由 scene、memory 和 agent context 给出；训练期 posterior 可读取未来监督，推理期只使用 prior。START 的 B0 先投影为 behavior seed，原始 B0 不进入 ROLL。
4. **Joint agent-time refiner**：直接处理
   \[
   U_t\in\mathbb{R}^{H\times N\times2},\quad U=[a,\dot\psi].
   \]
   每一层依次在时间维、车辆维做多头注意力，再对 scene memory 与 map tokens 做 cross-attention；由 ADS controls 积分而来的 `[B,H,6]` ego future state 作为不可修改 token 参与时间和车辆注意力。

## 3. START 与 ROLL

**START**：Flow 或 sidecar 提供 `(C0,B0,map)`。项目共享 START adapter 将 Flow 相对背景速度恢复为 `v_bg=v_ego+Δv_bg`；专用 START encoder 只编码 C0 与 map。`B0` 通过平滑 first-second action projection 初始化第一个背景控制 buffer、behavior seed 与 scene memory；首段计划采用从 `start_anchor_mix` 衰减至零的凸组合，而非控制相加。

**ROLL**：之后模型仅推进
\[
(S_t,m_t,B_t^{plan},a_t^{ego})\rightarrow(S_{t+1},m_{t+1},B_{t+1}^{plan}).
\]
每 0.2 秒执行 buffer 前五帧背景控制，接受 ADS 的五帧 ego controls，移位未执行背景 buffer，并在尾部追加五帧新控制。原始 B0 不会再次作为函数输入或状态读取。

公开的 Flow 接口为：

```python
model.rollout_from_flow(
    flow_condition, slot_valid=slot_mask,
    map_polylines=map_polylines, map_polyline_valid=map_valid,
    lane_graph_edges=lane_edges, ego_future_controls=ads_controls,
)
```

其中 `ads_controls` 必须是 `[B,T,2]`；它与背景 `U` 分离，背景模块不能修改 ego 控制。该离线接口以动力学更新 ego；高D replay 仅由 `rollout_reconstruction` 用于训练和独立重建评测。

在线接口为 `QRWorldModelEnvironment.reset_from_flow(C0,B0,metadata)`、`observe()` 和 `step(ego_action)`。`ego_action` 是下一个 0.2 s 的 `[5,2]` 控制块；内部一秒条件尾部按最后一个控制零阶保持，并在每个 response 重规划。

## 4. Buffer mask 与去噪精炼

`B_t^plan` 始终表示**未执行的背景控制**，不表示未来状态或轨迹。每一步输出：

- `executed_control_masks`：已经送入动力学的前五帧，不属于下一 buffer；
- `carried`：从上一 buffer 保留的未执行区；
- `appended`：新生成的尾部五帧；
- `refinable`/`valid`：可由联合 refiner 修改的有效背景未来控制。

历史观测、已执行控制和所有 ego 控制均不可修改。训练期在有效 `refinable` 区域采样噪声等级：
\[
\widetilde U=U+\sigma_k\epsilon,
\qquad R_\theta(\widetilde U,S,m,k)\rightarrow U.
\]
模型使用 noise-level embedding 和一次共享 joint refiner 计算 denoising loss；实际 rollout 固定零噪声，仅执行 1--2 次 amortized refinement，不实现长扩散链。

## 5. 训练、检查点与评测

训练预算仍为 8+12+20=40 epoch。每批随机平衡 50% START 与 50% ROLL；START 路径额外用首秒状态汇总 `B̂0` 计算 `L1(B̂0,B0)`。TensorBoard 记录 batch/epoch loss、START/ROLL loss、B0 summary、FDE、joint plan、denoising 与 buffer-mask 标量。checkpoint 标识为 `query_refine_world_model`，带 Flow schema hash、START encoder 与 ego-state-token 合同。旧 QR checkpoint 不兼容；当前没有可加载 QR checkpoint，必须完成重训后才可评测。

标准重建评测使用显式 highD ego replay；Flow 组合评测使用相同的 8x4 协议，但只评估生成分布，并输出包含 slot mask、primary slot、event structure、`event_structure_log_prob`、`conditional_log_prob` 与 `log_prob` 的逐样本 audit。当前架构完成 40-epoch 训练后，才可加入统一长尾比较。
