# QR-WM Training Architecture

## 1. 运行边界

QR-WM 使用 highD 固定的 `[ego, six background slots]` 场景张量。ROLL 输入为真实已发生的交通历史和道路 polyline/topology；START 输入仅为 Flow 的当前 C0、B0 与道路，不伪造历史帧。不输入交通灯、ADS 动作或 ego 未来序列，也不加载 RAMP/FIRM checkpoint。训练监督为未来背景状态和由 highD action 投影得到的背景动作。

Flow 不参与主训练的密度优化，但其冻结 76-D `C0+B0` schema 是正式 START 条件。训练、验证和测试均读取按同一 schema 生成的只读 B0 sidecar，以免 Flow 推理与训练条件不一致。

## 2. 联合模型

1. **Relation-aware scene encoder**：历史状态经多头 temporal attention 编码；两层 relation-aware multi-head self-attention 对车辆关系使用可学习 pairwise bias；随后每个 agent 对 agent/map token 做多头 cross-attention。
2. **Persistent scene memory**：唯一持续状态满足
   \[
   m_t=f(m_{t-1},S_t,U_{t-1},\Delta S_t).
   \]
   它保存交通交互、既有背景动作和已观测状态变化。QR-WM 没有独立 world memory。
3. **Behavior prior**：CVAE prior 由 scene、memory 和 agent context 给出；训练期 posterior 可读取未来监督，推理期只使用 prior。START 的 B0 先投影为 behavior seed，原始 B0 不进入 ROLL。
4. **Joint agent-time refiner**：直接处理
   \[
   U_t\in\mathbb{R}^{H\times N\times2},\quad U=[a,\dot\psi].
   \]
   每一层依次在时间维、车辆维做多头注意力，再对 scene memory 与 map tokens 做 cross-attention；由当前/历史 ego 状态编码得到的 token 作为不可修改 agent token 参与时间和车辆注意力，不携带 ego 动作或未来状态。

## 3. START 与 ROLL

**START**：Flow 或 sidecar 提供 `(C0,B0,map)`。项目共享 START adapter 将 Flow 相对背景速度恢复为 `v_bg=v_ego+Δv_bg`；专用 START encoder 只编码 C0 与 map。`B0` 通过平滑 first-second action projection 初始化第一段背景车辆未来动作序列、behavior seed 与 scene memory；首段动作采用从 `start_anchor_mix` 衰减至零的凸组合，而非动作相加。

**ROLL**：之后模型仅推进
\[
(S_t,m_t,A_t^{bg})\rightarrow(S_{t+1},m_{t+1},A_{t+1}^{bg}).
\]
每 0.2 秒执行背景车辆未来动作序列的前五帧，移位未执行动作，并在尾部追加五帧新动作。ADS 在环境外推进 ego；QR 仅在下一周期使用新观测到的 ego 状态。原始 B0 不会再次作为函数输入或状态读取。

公开的 Flow 在线接口为：

```python
environment.reset_from_flow(C0, B0, metadata)
environment.step(ego_state)
```

其中 `ego_state` 是当前观测到的 `[6]` 状态；ADS 控制不进入 QR。`rollout_reconstruction` 只在每次背景动作生成后回放日志 ego 状态，用于训练和独立重建评测。

在线接口为 `QRWorldModelEnvironment.reset_from_flow(C0,B0,metadata)`、`observe()` 和 `step(ego_state)`。`ego_state` 是每 0.2 s 提供的当前 `[6]` 状态；QR 只依据已发生状态重规划。

## 4. Buffer mask 与确定性精炼

`A_t^{bg}` 始终表示**未执行的背景车辆未来动作序列**，不表示未来状态或轨迹。每一步输出：

- `executed_background_action_masks`：已经送入动力学的前五帧，不属于下一动作序列；
- `carried`：从上一动作序列保留的未执行区；
- `appended`：新生成的尾部五帧；
- `refinable`/`valid`：可由联合 refiner 修改的有效背景未来动作。

历史观测与已执行背景动作均不可修改；ego 不属于背景动作序列。先由 carried 与 appended 区组成预动作序列，再以共享 joint refiner 执行两次确定性残差精炼：
\[
A_t^{refined}=A_t^{pre}-R_\theta(A_t^{pre},H_t,M,m_t,z,\mathrm{masks}).
\]
模型的随机性只来自 CVAE behavior latent；精炼不采样噪声、不使用噪声等级，也不承担多模态生成职责。

## 5. 训练、检查点与评测

训练预算仍为 8+12+20=40 epoch。每批随机平衡 50% START 与 50% ROLL；START 路径额外用首秒状态汇总 `B̂0` 计算 `L1(B̂0,B0)`。TensorBoard 记录 batch/epoch loss、START/ROLL loss、B0 summary、FDE、联合动作序列与 mask 标量。checkpoint 标识为 `query_refine_world_model`，带 Flow schema hash、START encoder 与 ego-state-only 合同。旧 QR checkpoint 不兼容；当前没有可加载 QR checkpoint，必须完成重训后才可评测。

标准重建评测使用显式 highD ego replay；Flow 组合评测使用相同的 8x4 协议，但只评估生成分布，并输出包含 slot mask、primary slot、event structure、`event_structure_log_prob`、`conditional_log_prob` 与 `log_prob` 的逐样本 audit。当前架构完成 40-epoch 训练后，才可加入统一长尾比较。
