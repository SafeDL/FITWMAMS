# TrafficBots V1.5 → highD 外部基线适配指南

> 目标仓库：`SafeDL/FITWMAMS`  
> 上游基线：`zhejz/TrafficBotsV1.5`  
> 用途：将 TrafficBots V1.5 作为 FITWMAMS 中“分层世界模型”的**外部闭环交通世界模型基线**，在完全相同的 highD 数据、划分、滚动时域和评测协议下训练与评测。

---

## 0. 最终建议：做“协议适配”，不要重写 TrafficBots 核心模型

适配时应保持 TrafficBots V1.5 的方法身份：

- 保留 HPTR / KNARPE 的 pairwise-relative 场景编码；
- 保留 destination + CVAE personality latent；
- 保留一步动作预测 + 显式动力学 + 自回归 rollout；
- 保留标准高斯 personality prior、posterior CVAE、free nats、10% prior rollout；
- 保留 agent-level scheduled teacher forcing；
- 保留连续 Gaussian action head 与 deterministic mean-action rollout；
- 保留“所有 agent 共享一个 policy”的设定。

只替换/改造以下接口：

1. WOMD/WOSAC 数据输入 → FITWMAMS 的 canonical highD sequence cache；
2. 10 Hz 时间协议 → highD 25 Hz；
3. Waymo 地图 / traffic light schema → highD 直线车道地图 + 全无效 traffic-light token；
4. WOSAC/WOMD Lightning wrapper、metrics、submission → highD 统一训练/评测 wrapper；
5. ego 从 TrafficBots 控制对象改为**外部控制对象**，但仍作为场景中的可观测 agent；
6. WOSAC 的 128→32 最低碰撞筛选**不得用于主结果**。

这样得到的模型建议命名为：

```text
TrafficBotsV1.5-HighD
```

论文中建议写成：

> “TrafficBots V1.5 was adapted to the same highD state, map, temporal, and closed-loop evaluation protocol as our model, while preserving its HPTR backbone, CVAE personality latent, destination conditioning, action head, and autoregressive policy rollout.”

---

# 1. 基线公平性协议

## 1.1 必须与本文模型共享的内容

TrafficBotsV1.5-HighD 必须直接读取 FITWMAMS 已生成并审计的 canonical highD sequence cache，而不是重新从 highD 原始 CSV 独立切分数据。

共享：

```text
sequence_id
agent_states
agent_valid
map_polylines
map_polyline_valid
lane_graph_edges
split_index
is_evt_tail
actions_highd
```

固定协议：

```text
FPS                  = 25 Hz
dt                   = 0.04 s
S0 index             = ANCHOR_INDEX = 24
real states           = S0 ... S149 = 150 states
closed-loop steps     = 149 transitions
rollout duration      = 5.96 s
real agents           = ego + 6 background = 7
canonical map         = 8 polylines × 8 nodes
```

TrafficBots 必须直接通过 canonical cache loader 读取数据：

```python
from world_model.src.core.sequential_dataset import load_sequential_dataset
```

并按 cache 的 `split_index` 划分 train/val/test。不得调用
`diffusion.src.data.load_data_bundle()`，也不得读取 Flow dataset、Flow 映射、
soft plan 或 EVT 特征；`is_evt_tail` 只可作为报告分层标签。

**禁止再随机生成 train/val/test split。** `split_index` 是论文不同 world model 之间最重要的公平性边界。

---

## 1.2 TrafficBots 不得获得本文模型之外的额外未来信息

主实验中 TrafficBots 输入只能包含：

```text
当前/已实现 agent history
当前有效性 mask
highD lane polylines
自身预测的 destination
自身 CVAE personality prior sample
```

不得输入：

```text
Flow 的 C0 density / latent
Flow 采样出的 trajectory knots K
background diffusion 的 soft_reference
未来 2 s / 4 s / 5.96 s trajectory constraints
EVT tail label
GT future destination（测试阶段）
GT future action
GT future ego trajectory（作为模型输入）
```

其中 GT future 可用于：

- 训练 CVAE posterior；
- 训练 destination predictor；
- 计算监督损失；

但不能作为测试时 causal input。

---

## 1.3 ego 的统一处理

FITWMAMS 的世界模型主要预测背景交通参与者对 ego 的闭环响应。因此 TrafficBots 基线也必须采用同一角色定义：

```text
ego：外部控制，但持续作为场景中的 agent token
background agents：TrafficBots 控制
```

训练时：

```text
TrafficBots 预测 ego action → 丢弃
ego next state → 使用 logged highD ego action / state 覆盖
```

评测时：

```text
factual      → logged ego control
intervention → brake / accelerate / left intervention
ADS test     → ADS policy output
```

因此，TrafficBots 的 background policy 能看到 ego 执行后的新状态并在下一步响应，但不能提前读取未来 ego action。

---

# 2. 推荐目录结构

FITWMAMS 当前已经保留：

```text
ref_code/TrafficBotsV1.5-main/
```

建议**不要直接修改这一份上游快照**。将其作为 provenance/reference 保持不可变。

新增：

```text
FITWMAMS/
├── ref_code/
│   └── TrafficBotsV1.5-main/       # 原始上游，保持不动
│
├── baselines/
│   └── trafficbots_highd/
│       ├── README.md
│       ├── NOTICE.md
│       ├── config/
│       │   └── highd.yaml
│       ├── src/
│       │   ├── __init__.py
│       │   ├── data.py
│       │   ├── preprocess.py
│       │   ├── dynamics.py
│       │   ├── module.py
│       │   ├── rollout.py
│       │   ├── evaluation.py
│       │   ├── stochastic_eval.py
│       │   ├── intervention_eval.py
│       │   └── adapter.py
│       ├── scripts/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── evaluate_stochastic.py
│       │   └── evaluate_intervention.py
│       └── tests/
│           ├── test_data_contract.py
│           ├── test_causal_rollout.py
│           ├── test_dynamics_dt.py
│           ├── test_no_future_leakage.py
│           └── test_reproducibility.py
│
└── results/
    └── baselines/
        └── trafficbots_highd/
```

`NOTICE.md` 至少记录：

```text
Upstream: https://github.com/zhejz/TrafficBotsV1.5
Upstream license: CC BY-NC 4.0
Adaptation: highD data/time/evaluation protocol only
```

保留复制代码文件中的上游版权/许可证头。

---

# 3. highD → TrafficBots 数据适配

## 3.1 原项目 canonical state

FITWMAMS 中每个 agent 的统一状态为：

```text
[x, y, vx, vy, ax, ay]
```

canonical cache：

```python
states = arrays["agent_states"][row, ANCHOR_INDEX:ANCHOR_INDEX + 150]
valid  = arrays["agent_valid"][row, ANCHOR_INDEX:ANCHOR_INDEX + 150]

# states: [150, 7, 6]
# valid : [150, 7]
```

转换为 TrafficBots agent-major layout：

```python
states = np.swapaxes(states, 0, 1)  # [7, 150, 6]
valid  = np.swapaxes(valid, 0, 1)   # [7, 150]
```

---

## 3.2 pose 与 motion

TrafficBots 的 agent representation 使用：

```text
pose   = [x, y, yaw]
motion = [speed, acceleration, yaw_rate]
```

优先复用 FITWMAMS 中已经定义好的物理转换：

```python
from world_model.src.core.dynamics import KinematicTrafficDynamics
```

转换：

```python
xy = states[..., 0:2]
velocity = states[..., 2:4]
cart_acc = states[..., 4:6]

speed = np.linalg.norm(velocity, axis=-1)
yaw = np.arctan2(velocity[..., 1], np.maximum(velocity[..., 0], 1.0e-4))

controls = KinematicTrafficDynamics.controls_from_highd_actions(
    torch.from_numpy(cart_acc),
    torch.from_numpy(states),
).numpy()

acc = controls[..., 0]
yaw_rate = controls[..., 1]
```

然后构建：

```python
agent_pos[..., :2] = xy
agent_pos[..., 2] = 0.0
agent_vel = velocity
agent_spd = speed[..., None]
agent_acc = acc[..., None]
agent_yaw_bbox = yaw[..., None]
agent_yaw_rate = yaw_rate[..., None]
```

所有 invalid state 在进入网络前归零，并保留单独 `valid` mask。

---

## 3.3 agent type

highD 当前任务只有车辆。因此保持 TrafficBots 原始 3 类接口，但所有真实 agent 均设：

```python
agent_type = np.zeros((N_AGENT_PAD, 3), dtype=bool)
agent_type[:7, 0] = valid[:7, 0]  # S0-existing Vehicle only
```

不要把 type dimension 改为 1；原始 TrafficBots 多处默认 `[veh, ped, cyc]` 三类表示，保持 3 维能显著减少核心代码改动。

---

## 3.4 agent role

建议：

```python
agent_role = np.zeros((N_AGENT_PAD, 3), dtype=bool)
agent_role[0, 0] = valid[0, 0]        # SDC / ego
agent_role[:7, 2] = valid[:7, 0]      # S0-existing predict slots
```

正式 highD 评测应显式使用 background mask：

```python
background_mask = valid[:, 0, 1:7]
```

不要依赖 Waymo 的 `interest/predict` role 完成核心评测。训练的
reconstruction/KL/destination loss 仍覆盖所有有效 vehicle（包含 ego），与
第 14 节的最终协议一致。

---

## 3.5 agent size

canonical state 不需要 vehicle size，而 TrafficBots encoder 原始接口中包含 size。

主实验建议采用**固定常量**，而不是给 TrafficBots 额外提供每辆车的 raw metadata：

```text
所有真实车辆使用同一个 train-only 固定 size vector；
所有 invalid padded agent size = 0。
```

可以选择：

1. 从 highD train split 统计 length/width 中位数，冻结后写入 config；或
2. 使用统一归一化 constant。

不要在 test 阶段读取额外未来或 per-sample metadata 以避免信息不对等。

---

# 4. highD 地图适配到 HPTR MapEncoder

FITWMAMS canonical map：

```text
map_polylines:       [8, 8, 6]
map_polyline_valid:  [8, 8]
```

其中当前 highD adapter 的 polyline point 已包含：

```text
[x, y, dir_x, dir_y, lane_width, marker]
```

TrafficBots 需要：

```text
map/valid
map/type
map/pos
map/dir
map/boundary
```

转换：

```python
map_pos = np.zeros((N_MAP_PAD, 8, 3), np.float32)
map_dir = np.zeros((N_MAP_PAD, 8, 3), np.float32)
map_valid = np.zeros((N_MAP_PAD, 8), bool)
map_type = np.zeros((N_MAP_PAD, 11), bool)

map_pos[:8, :, :2] = map_polylines[..., :2]
map_dir[:8, :, :2] = map_polylines[..., 2:4]
map_valid[:8] = map_polyline_valid

real_lane = map_valid[:8].any(-1)
map_type[:8, 0] = real_lane  # Waymo-compatible FREEWAY type
```

保留 `map/type` 的 11 维，是因为原始 `NaviPredictor` 对 map type 有固定的 Waymo-style mask；使用 index 0 的 `FREEWAY` 对 highD 是合理的语义映射，同时无需修改 `navigation.py`。

`map/boundary`：

```python
xy = map_pos[map_valid, :2]
margin = 20.0
boundary = np.array([
    xy[:, 0].min() - margin,
    xy[:, 0].max() + margin,
    xy[:, 1].min() - margin,
    xy[:, 1].max() + margin,
], dtype=np.float32)
```

---

# 5. destination：在 highD 中解释为“目标车道”

TrafficBots 的 destination 原本是 horizon-independent 的 map polyline index。

highD 没有复杂路口 route，因此适配后将 destination 定义为：

> **highD lane-destination surrogate**：agent 在该 5.96 s segment 最后一个有效状态最近的、且与 S0 行驶方向兼容的有效 FREEWAY lane polyline。

`map/type` 对每条有效 lane center 使用 11 维 `FREEWAY=0` one-hot；padding
token 必须全零且 invalid。destination 的候选 mask 只包含有效 FREEWAY lanes，
并要求其最近节点切向与 S0 agent heading 的内积非负。该 mask 必须同时用于
CE、argmax 和 categorical sample，以排除反向 carriageway。

训练 GT：

```python
def final_lane_destination(agent_states, agent_valid, map_pos, map_valid):
    # agent_states: [T, 6]
    # agent_valid : [T]
    t_last = np.flatnonzero(agent_valid)[-1]
    xy = agent_states[t_last, :2]

    delta = map_pos[:, :, :2] - xy[None, None]
    dist2 = np.sum(delta * delta, axis=-1)
    dist2[~map_valid] = np.inf
    return int(np.unravel_index(np.argmin(dist2), dist2.shape)[0])
```

### 关键数据泄漏规则

训练：

```text
policy rollout 可使用 GT destination
NaviPredictor 用 GT destination 做 CE/NLL 监督
```

测试：

```text
禁止 GT destination
```

确定性评测：

```text
predicted destination = categorical argmax
```

随机评测：

```text
predicted destination = categorical sample
```

可以额外报告 `TrafficBots-oracle-dest` 作为诊断上界，但不得作为论文主 baseline。

---

# 6. highD 没有交通灯：不要删除整个 HPTR 分支，使用全无效 dummy token

TrafficBots V1.5 的 core model 强依赖 `tl_encoder` 接口。如果直接删除 traffic-light branch，会触及较多核心代码并削弱“外部基线”的可追溯性。

推荐做法：

```text
保留 TL encoder 结构
所有 highD traffic-light tokens 全 invalid
w_tl_state = 0
```

例如 pad 8 个 dummy TL：

```python
N_TL_PAD = 8

tl_valid = np.zeros((N_TL_PAD, 150), dtype=bool)
tl_state = np.zeros((N_TL_PAD, 150, 5), dtype=bool)
tl_pos = np.zeros((N_TL_PAD, 3), np.float32)
tl_dir = np.zeros((N_TL_PAD, 3), np.float32)
tl_dir[:, 0] = 1.0
```

配置：

```yaml
pre_processing:
  scene_centric:
    tl_mode: stop

training_metrics:
  w_tl_state: 0.0
```

原始 KNARPE attention 对“所有 target 都 invalid”的情况会将输出置 0，因此全无效 TL token 不会向 agent feature 注入伪造语义。

---

# 7. KNN 数量必须修改：原始 helper 不允许 K ≥ token 数量

上游 `get_tgt_knn_idx()` 有硬约束：

```python
assert 0 < n_tgt_knn < n_tgt
```

而 highD 只有：

```text
7 real agents
8 real map polylines
0 traffic lights
```

不能继续使用 Waymo 配置中的 32 / 64 / 25。

## 推荐：只做 invalid padding，不改变真实 highD 内容

```text
N_AGENT_PAD = 8   # 7 real + 1 invalid
N_MAP_PAD   = 16  # 8 real + 8 invalid
N_TL_PAD    = 8   # all invalid
n_tgt_knn   = 8
```

推荐 K：

```yaml
model:
  n_tgt_knn: 8

  ag_encoder:
    k_tgt_knn_ag2mp: 1.0     # 8
    k_tgt_knn_ag2ag: 0.875   # 7
    k_tgt_knn_ag2tl: 0.875   # 7, but all invalid

  tl_encoder:
    k_tgt_knn_tl2mp: 1.0     # 8
    k_tgt_knn_tl2tl: 0.875   # 7
```

这样：

- map self-attention 可以覆盖全部 8 条真实 lane polyline；
- agent self-attention 对每个真实 agent 可覆盖自己和另外 6 个真实 agent；
- dummy token 全部通过 validity mask 屏蔽。

`dist_limit` 主实验建议继续使用上游值 `500`，避免引入额外的距离阈值调参。

---

# 8. 10 Hz → 25 Hz：必须修改 Dynamics.dt

上游 `src/utils/dynamics.py` 中：

```python
self.dt = 0.1
```

是硬编码。

必须改成可配置：

```python
class Dynamics:
    def __init__(
        self,
        veh,
        ped,
        cyc,
        navi_mode,
        use_veh_dynamics_for_all=False,
        dt=0.1,
    ):
        self.dt = float(dt)
        ...
```

highD config：

```yaml
dynamics:
  dt: 0.04
  use_veh_dynamics_for_all: true
  veh:
    _target_: utils.dynamics.MultiPathPP
    max_acc: 5.0
    max_yaw_rate: 1.5
```

因为 highD 全部是 vehicle，`use_veh_dynamics_for_all: true` 能移除无效的 cyclist/pedestrian branch，但不改变车辆动力学。

### 主结果不要直接换成本文自己的 dynamics

建议：

- **主 baseline**：TrafficBots 原始 `MultiPathPP`，只将 dt 改为 0.04；
- **附加公平性消融**：`TrafficBots-CommonDynamics` 使用 FITWMAMS 的 `KinematicTrafficDynamics`。

这样可以区分：

```text
模型生成能力差异
vs.
动力学积分器差异
```

---

# 9. policy temporal window：保留 released TrafficBots 的 11-step window

TrafficBots V1.5 公开配置与论文均使用：

```text
11-step stacked history
```

highD 适配保留这个离散模型契约，不按物理时长扩展为 25 帧；否则会改变
HPTR policy 的已发布结构：

```yaml
model:
  policy_history_steps: 11
```

FITWMAMS canonical evaluator 初始时只有 `S0` 是有效 observed state，之前 24 个 compatibility frames 是 invalid。

因此：

```text
第一步输入：S0
第二步输入：S0 + generated S1
...
达到 11 帧后：始终使用最近 11 个 realized states
```

不要把 24 个 invalid prefix 当作真实历史填入 PointNet。

也不要为了模仿 Waymo 的 1 s warm start 而把 `S1...S24` GT 输入给模型；那会改变本文的 149-step causal rollout 任务。

---

# 10. CVAE posterior temporal downsampling 必须单独改

上游 `LatentEncoder` 使用：

```python
temporal_down_sample_rate = 5
assert (n_step - 1) % temporal_down_sample_rate == 0
```

Waymo：

```text
0.1 s × 5 = 0.5 s latent posterior sampling interval
```

highD：150 state points，`149` 是质数，不能直接使用 12/13 作为 stride。

## 推荐改为“固定数量的时间索引”

新增参数：

```yaml
latent_encoder:
  latent_dim: 16
  latent_num_temporal_samples: 13
```

原因：

```text
5.96 s / 0.5 s ≈ 11.92 intervals
→ 包含首尾约 13 temporal samples
```

建议代码：

```python
def temporal_indices(n_step: int, n_sample: int, device):
    if n_step <= n_sample:
        return torch.arange(n_step, device=device)
    return torch.linspace(
        0,
        n_step - 1,
        steps=n_sample,
        device=device,
    ).round().long()
```

在 posterior forward：

```python
idx = temporal_indices(
    ag_valid.shape[-1],
    self.latent_num_temporal_samples,
    ag_valid.device,
)

ag_valid = ag_valid.index_select(2, idx)
ag_motion = ag_motion.index_select(2, idx)
ag_pose = ag_pose.index_select(2, idx)
tl_state = tl_state.index_select(2, idx)
```

同时 `__init__` 中 posterior 的 `ag_encoder/tl_encoder.temp_window_size` 设置为 `13`。

保留：

```yaml
latent_post:
  dist_type: diag_gaus

latent_prior:
  dist_type: std_gaus

training_metrics:
  kl_free_nats: 1.0
  kl_balance_scale: 0.2
```

不要把 personality prior 改为本文自己的 latent flow，否则不再是 TrafficBots 基线。

---

# 11. highD 数据输出 contract

建议 `TrafficBotsHighDDataset.__getitem__()` 输出接近原始 `SceneCentricPreProcessing` 所需的字段：

```text
agent/valid
agent/pos
agent/vel
agent/spd
agent/acc
agent/yaw_bbox
agent/yaw_rate
agent/type
agent/role
agent/size
agent/goal
agent/dest

map/valid
map/type
map/pos
map/dir
map/boundary

tl_stop/valid
tl_stop/state
tl_stop/pos
tl_stop/dir
```

验证/测试为了兼容原始 preprocessor 的 `history/` 分支，可额外输出：

```text
history/agent/*    = S0 only
history/tl_stop/*  = dummy S0 only
```

但更推荐在新 highD wrapper 中直接控制前缀逻辑，不让数据 schema 为 Waymo validation API 做无意义复制。

---

# 12. 不要继续使用 `WaymoMotion` 作为 highD Lightning wrapper

不建议在：

```text
src/pl_modules/waymo_motion.py
```

里堆积 highD `if` 分支。

它强耦合：

```text
WOMDMetrics
WOSACMetrics
WOMDPostProcessing
WOSACPostProcessing
SubWOMD
SubWOSAC
TrafficRuleChecker 的 Waymo map semantics
Waymo scenario metadata
```

应新建：

```text
baselines/trafficbots_highd/src/module.py
```

类：

```python
class HighDTrafficBotsModule(LightningModule):
    ...
```

只复用上游：

```python
from models.traffic_bots import TrafficBots
from models.metrics.loss import BalancedKL
from utils.dynamics import MultiPathPP
```

以及 HPTR、navigation、latent、action head 等 core modules。

---

# 13. 推荐训练流程

## 13.1 每个 batch

```text
1. 从 canonical highD sequence cache 读取 150 states
2. 构造 highD→TrafficBots batch
3. encode map once
4. precompute fully-invalid TL tokens
5. posterior personality q(z | GT trajectory)
6. prior personality p(z) = N(0, I)
7. 90% episode：posterior z
   10% episode：prior z
8. destination predictor 预测 lane-polyline distribution
9. policy 从 S0 开始 autoregressive rollout 149 transitions
10. 每步丢弃 ego prediction，并用 logged ego next state 覆盖
11. scheduled teacher forcing 仅作用于 background agents
12. 累积 reconstruction + KL + destination loss
```

---

## 13.2 teacher forcing

highD canonical active background slots在 S0 后保持稳定，因此：

```yaml
teacher_forcing_training:
  step_spawn_agent: 0
  step_warm_start: 0
  step_horizon: 0
  prob_forcing_agent: 0.30
  prob_forcing_agent_decrease_per_epoch: 0.10
  prob_scheduled_sampling: 0.0
  gt_sdc: true
```

解释：

- step 0 是初始化真实状态；
- ego 始终 external/logged；
- 初期约 30% background agent 整条轨迹 teacher-forced；
- 比例按上游逻辑逐 epoch 降到 0。

### 不要把 `step_training_start=10` 机械改成 25

原 Waymo 配置中前 10 step 与 1 s observed/warm-start window 对齐。

本项目主任务从 S0 就开始闭环预测，因此 highD wrapper 应从第一个预测 transition 产生 reconstruction loss：

```yaml
loss_start_step: 1
```

否则在 `training_detach_model_input=True` 时，前 1 s 预测几乎没有有效训练梯度。

---

# 14. reconstruction / KL / destination loss

保持上游损失结构：

```math
L = L_rec + L_KL + L_dest
```

background reconstruction：

```math
L_rec =
0.1 L_{SmoothL1}(x,y)
+ 10 \cdot \frac{1-\cos(\hat\psi-\psi)}{2}
+ 0.1 L_{SmoothL1}(v)
```

KL：

```text
BalancedKL
kl_balance_scale = 0.2
free_nats = 1.0
weight = 1.0
```

destination：

```text
cross entropy / negative log likelihood of target lane-polyline index
weight = 1.0
```

traffic light：

```text
weight = 0
```

collision：

```text
weight = 0
```

### ego loss mask

主任务只评价 background world response；但为保持上游共享 policy 的训练语义，
reconstruction、KL 与 destination NLL 对**所有有效 vehicle（包含 ego）**计算。
训练 rollout 中 ego 下一状态可以 GT override；正式评测仅汇报 background 指标。

---

# 15. 训练时动作与动力学

保持 V1.5：

```text
training_deterministic_action = True
```

即使用：

```python
action = action_dist.mean
```

再通过可微动力学得到 next state。

不要训练阶段每步随机 sample action，否则同时引入 personality randomness 和 action-noise randomness，会偏离上游 V1.5 的训练定义。

---

# 16. 推荐 highD 配置骨架

```yaml
experiment:
  name: trafficbots_v15_highd
  seed: 20260814

paths:
  sequence_cache_dir: results/highd_shared_training_data/highd_sequence_cache
  output_dir: results/baselines/trafficbots_highd

data:
  fps: 25.0
  dt_s: 0.04
  anchor_index: 24
  state_points: 150
  rollout_steps: 149
  real_agents: 7
  padded_agents: 8
  real_map_polylines: 8
  padded_map_polylines: 16
  map_nodes: 8
  padded_traffic_lights: 8

model:
  hidden_dim: 128
  pairwise_relative: true
  policy_history_steps: 11
  n_tgt_knn: 8
  dist_limit: 500.0

  tf_cfg:
    d_model: 128
    n_head: 4
    k_feedforward: 4
    dropout_p: 0.1
    bias: true
    activation: relu
    out_layernorm: false
    apply_q_rpe: false

  mp_encoder:
    n_layer_tf: 8

  tl_encoder:
    n_layer_tf: 4
    k_tgt_knn_tl2tl: 0.875
    k_tgt_knn_tl2mp: 1.0

  ag_encoder:
    n_layer_tf: 4
    k_tgt_knn_ag2mp: 1.0
    k_tgt_knn_ag2tl: 0.875
    k_tgt_knn_ag2ag: 0.875

  latent_encoder:
    latent_dim: 16
    latent_num_temporal_samples: 13
    latent_post:
      dist_type: diag_gaus
    latent_prior:
      dist_type: std_gaus

  navi_mode: dest

  action_head:
    log_std: -2
    branch_type: true

pre_processing:
  tl_mode: stop
  navi_mode: dest
  dropout_p_history: 0.1

teacher_forcing:
  step_spawn_agent: 0
  step_warm_start: 0
  prob_forcing_agent: 0.30
  prob_forcing_agent_decrease_per_epoch: 0.10
  gt_sdc: true

training:
  p_rollout_prior: 0.10
  deterministic_action: true
  detach_model_input: true
  loss_start_step: 1

loss:
  pos_weight: 0.1
  yaw_weight: 10.0
  speed_weight: 0.1
  kl_weight: 1.0
  kl_balance_scale: 0.2
  kl_free_nats: 1.0
  destination_weight: 1.0
  traffic_light_weight: 0.0
  collision_weight: 0.0

optimizer:
  name: AdamW
  lr: 2.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

evaluation:
  factual_deterministic: true
  stochastic_samples: 16
  collision_filtering: false
```

除 highD 必需字段外，优先沿用 TrafficBots V1.5 上游默认超参数，而不是按本文模型重新调参。

---

# 17. 确定性 factual evaluation

主 deterministic baseline 建议：

```text
personality z = prior mean = 0
destination   = argmax p(g | S0, map)
action        = Gaussian mean
ego           = logged ego controls through the same external kinematic dynamics
background    = TrafficBots autoregressive rollout
```

输入：

```python
S0 = agent_states[:, ANCHOR_INDEX]
```

滚动：

```text
149 × 0.04 s
```

输出统一转换回 FITWMAMS state：

```text
[x, y, vx, vy, ax, ay]
```

其中：

```python
vx = speed * cos(yaw)
vy = speed * sin(yaw)

ax = acc * cos(yaw) - speed * yaw_rate * sin(yaw)
ay = acc * sin(yaw) + speed * yaw_rate * cos(yaw)
```

建议直接输出：

```python
Rollout(
    states=[B,149,7,6],
    background_actions=[B,149,6,2],
    ego_actions=[B,149,2],
    reference_actions=...,
)
```

使其能够复用 `hierarchical_world_model/src/evaluation.py` 中的同一套 highD metrics。

---

# 18. 必须与本文模型共享的 factual metrics

使用相同 test rows 与 active mask，至少报告：

```text
ADE_m
FDE_m
P50_displacement_error_m
P90_displacement_error_m
P95_displacement_error_m
P99_displacement_error_m
speed_MAE_mps
```

并复用同样的 test strata：

```text
all_natural
evt_labelled
semantic_cutin
```

还应绘制/保存同一套 temporal drift 曲线：

```text
ADE(t)
P95 displacement error(t)
speed MAE(t)
```

这样论文中可以直接比较：

```text
HiQR / hierarchical model
vs.
TrafficBotsV1.5-HighD
```

而不是比较两个完全不同 benchmark 的指标。

---

# 19. stochastic evaluation

TrafficBots 的随机性主要来自：

```text
z_i ~ N(0, I)
g_i ~ p(g_i | history, map)
```

给定 `(z, g)` 后使用 action mean，rollout 近似确定。

建议与本文随机评测保持相同：

```yaml
stochastic_samples: 16
```

每个 condition 采样 16 次，直接复用 FITWMAMS 的 distribution metrics：

```text
sample_mean_ADE
min_ADE
sample_mean_FDE
min_FDE
energy_score
mean_pairwise_trajectory_distance
terminal_pairwise_distance
KS / Wasserstein of speed
KS / Wasserstein of ax, ay
KS / Wasserstein of jerk
KS / Wasserstein of yaw_rate / yaw_acceleration
nearest-object-distance distribution
gap distribution
TTC distribution
collision incidence
```

---

# 20. 严禁主结果使用 WOSAC 的“128 选 32 最低碰撞”

上游 TrafficBots V1.5 为提高 WOSAC leaderboard collision metric：

```text
sample 128 scenarios
select 32 scenarios with the least collisions
```

在 FITWMAMS 长尾测试研究中，这个步骤必须关闭：

```yaml
collision_filtering: false
```

原因：

```text
它改变基础生成分布；
它主动删除危险尾部；
它会使 world-model risk evaluation 产生安全偏置；
它不能用于 failure probability / long-tail comparison。
```

如果希望复现论文 WOSAC-style trick，只能放在附录：

```text
TrafficBotsV1.5-HighD-WOSACFiltered
```

并明确与主 baseline 分开。

---

# 21. ego intervention / causal-response evaluation

TrafficBots 原始 dynamics 已有 `player_override` 设计，因此非常适合做本文的干预对比。

对同一个 highD test condition：

```text
factual
brake
a ccelerate
left
```

使用同一组 personality / destination random numbers：

```text
Common Random Numbers (CRN)
```

也就是：

```python
z_factual == z_intervention
g_factual == g_intervention
```

只改变 ego action。

建议报告与 hierarchical model 相同的：

```text
background displacement response
reaction latency
longitudinal response
a lateral response
monotonic dose response
locality
nearest-object distance
TTC / gap change
```

这比单纯 ADE 更能体现 TrafficBots 作为闭环世界模型的能力。

---

# 22. 与 AMS / long-tail test 的可选二阶段接口

如果后续希望 TrafficBots 不只做 offline external baseline，而是进入相同的 AMS/branching test，可增加：

```text
TrafficBotsHighDPolicyAdapter
```

必须支持：

```python
reset(initial_state, map)
step(ego_action)
snapshot()
restore(snapshot)
sample_exogenous(seed)
```

snapshot 至少包含：

```text
current dynamics state
agent validity
last 11 agent history
sampled personality z
sampled/predicted destination g
navigation feature cache
map token cache
rollout time index
random seed / RNG state
```

这样 AMS branch 才能从同一中间世界状态严格复制继续滚动。

但建议先完成 offline highD baseline；AMS 集成放第二阶段，避免一次修改过多。

---

# 23. 推荐需要改/不改的上游文件

| 上游文件 | highD 处理 | 建议 |
|---|---|---|
| `src/models/map_encoder.py` | 无 | 保持不变 |
| `src/models/agent_encoder.py` | 无 | 保持不变 |
| `src/models/modules/attention_rpe.py` | 无 | 保持不变 |
| `src/models/modules/transformer_rpe.py` | 无 | 保持不变 |
| `src/models/modules/action_head.py` | 无 | 保持不变 |
| `src/models/navigation.py` | 通过 11-D FREEWAY map type 适配 | 保持不变 |
| `src/models/traffic_bots.py` | 只由新 wrapper 调用 | 尽量不改 |
| `src/models/latent_encoder.py` | 150-step posterior temporal sampling | **小改** |
| `src/utils/dynamics.py` | `dt=0.1` → configurable | **必须改/包装** |
| `src/data_modules/data_h5_womd.py` | WOMD schema | 不改，新增 `data_highd.py` |
| `src/data_modules/scene_centric.py` | 可复用 | 优先复用 |
| `src/pl_modules/waymo_motion.py` | Waymo API 强耦合 | 不改，新增 `highd_motion.py` |
| WOMD/WOSAC metrics/post-processing | 不适用 | 完全绕过 |

核心原则：

> **改 benchmark adapter，不改 model hypothesis。**

---

# 24. 单元测试与审计

## 24.1 数据 contract

```python
assert states.shape == (150, 7, 6)
assert padded_agent_valid.shape == (8, 150)
assert map_valid.shape == (16, 8)
assert tl_valid.shape == (8, 150)
assert not tl_valid.any()
```

---

## 24.2 split identity

对每个 split：

```python
assert set(trafficbots_sequence_ids) == set(hierarchical_model_sequence_ids)
```

并保存 hash：

```text
train_sequence_id_sha256
val_sequence_id_sha256
test_sequence_id_sha256
```

---

## 24.3 无未来泄漏

测试：修改 `S1...S149` GT future，但保持 S0 不变，在 deterministic test 初始化后：

```text
prior personality distribution 不变
predicted destination distribution 不变
first-step action distribution 不变
```

训练 posterior 允许变化。

---

## 24.4 ego override

检查：

```text
TrafficBots 自己预测的 ego action 改变
→ 最终执行 ego state 不应改变
```

执行 ego state 只取决于 logged/ADS/intervention control。

---

## 24.5 dt

恒速无 yaw case：

```text
25 steps ≈ 1.0 s
```

检查位移是否约等于：

```math
\Delta x = v \times 1.0s
```

避免遗漏 `0.1 → 0.04` 后产生 2.5 倍时间尺度错误。

---

## 24.6 deterministic replay

固定：

```text
z
destination
model checkpoint
ego actions
```

重复运行两次应满足：

```python
np.testing.assert_allclose(states_run1, states_run2)
```

---

## 24.7 stochastic CRN

干预成对测试应保存：

```text
seed
z sample
destination sample
```

确保 factual/intervention 对只改变 ego control。

---

# 25. 训练和评测命令建议

```bash
python -m baselines.trafficbots_highd.scripts.train \
  --config baselines/trafficbots_highd/config/highd.yaml
```

```bash
python -m baselines.trafficbots_highd.scripts.evaluate \
  --config baselines/trafficbots_highd/config/highd.yaml \
  --checkpoint results/baselines/trafficbots_highd/checkpoints/best.pt
```

```bash
python -m baselines.trafficbots_highd.scripts.evaluate_stochastic \
  --config baselines/trafficbots_highd/config/highd.yaml \
  --samples 16
```

```bash
python -m baselines.trafficbots_highd.scripts.evaluate_intervention \
  --config baselines/trafficbots_highd/config/highd.yaml
```

---

# 26. 结果目录建议

```text
results/baselines/trafficbots_highd/
├── manifest.json
├── config_resolved.yaml
├── checkpoints/
│   ├── best.pt
│   └── final.pt
├── training_history.json
├── factual_metrics.json
├── factual_temporal_metrics.json
├── factual_strata_metrics.json
├── stochastic_metrics.json
├── intervention_metrics.json
└── samples/
```

`manifest.json` 至少保存：

```text
upstream TrafficBots commit
FITWMAMS commit
canonical sequence manifest hash
train/val/test sequence IDs hash
highD fps/dt
rollout steps
model parameter count
training seed
checkpoint hash
whether collision filtering is disabled
whether GT destination is disabled at test
```

---

# 27. 论文中推荐的对比表

## 27.1 Factual reconstruction

| Model | ADE ↓ | FDE ↓ | P95 ↓ | Speed MAE ↓ |
|---|---:|---:|---:|---:|
| TrafficBotsV1.5-HighD |  |  |  |  |
| Proposed hierarchical WM |  |  |  |  |

同时按：

```text
All Natural
EVT-labelled
Semantic Cut-in
```

报告。

---

## 27.2 Distributional realism

| Model | Energy Score ↓ | Pairwise Diversity ↑ | Speed W1 ↓ | ax W1 ↓ | ay W1 ↓ | TTC W1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TrafficBotsV1.5-HighD | | | | | | |
| Proposed hierarchical WM | | | | | | |

---

## 27.3 Closed-loop causal response

| Model | Brake Response | Accel Response | Lateral Response | Dose Monotonicity | Locality |
|---|---:|---:|---:|---:|---:|
| TrafficBotsV1.5-HighD | | | | | |
| Proposed hierarchical WM | | | | | |

这一组实验能够直接说明本文“分层 world state / persistent behavior / trajectory refinement”相较 TrafficBots 的优势，而不是只证明 open-loop ADE 更小。

---

# 28. 建议的消融版本

主论文只需要一个正式 TrafficBots 外部 baseline：

```text
TrafficBotsV1.5-HighD
```

如果篇幅允许，可添加两个诊断版本：

```text
TrafficBotsV1.5-HighD-OracleDest
TrafficBotsV1.5-HighD-CommonDynamics
```

不要把以下版本作为正式 baseline：

```text
TrafficBots + Flow knots
TrafficBots + diffusion soft plan
TrafficBots + EVT labels
TrafficBots + 128→32 collision filtering
```

因为它们已经改变 TrafficBots 原方法或破坏风险分布。

---

# 29. 实施优先级

## Phase A：先得到严格可比较的 deterministic baseline

```text
highD dataset adapter
map adapter
dummy TL
dt=0.04
external ego
149-step rollout
destination predictor
z=0 deterministic evaluation
shared factual metrics
```

验收后再进入 Phase B。

## Phase B：恢复完整 CVAE 随机性

```text
posterior temporal sampler
free nats
10% prior rollout
stochastic destination
16-sample distribution evaluation
```

## Phase C：闭环干预

```text
brake / accelerate / left
CRN
causal response metrics
```

## Phase D：可选 AMS adapter

```text
snapshot / restore
branch replay
rare-event testing
```

这种分阶段方式最容易定位“数据问题、物理时间问题、模型问题和随机性问题”。

---

# 30. 最终方法学定义

适配后的 TrafficBotsV1.5-HighD 应严格定义为：

```math
z_i \sim \mathcal N(0,I),
\qquad
g_i \sim p_\psi(g_i\mid S_0,M),
```

```math
a_{i,t}
= \mu_\theta\big(
H_{t-10:t}, M, g_i,z_i
\big),
```

```math
S_{t+1}
=F_{\mathrm{TB}}(S_t,a_t;\Delta t=0.04),
```

其中：

```text
M：与本文完全相同的 highD lane-polylines
H：最多最近 11 个已经实现的状态
z：TrafficBots CVAE personality
 g：highD 目标车道 destination
F_TB：TrafficBots MultiPathPP dynamics
```

对 ego：

```math
S^{ego}_{t+1}
=F_{ego}(S^{ego}_t,a^{external}_{ego,t}),
```

TrafficBots 对 ego 的预测 action 被丢弃。

该定义保留了 TrafficBots 的核心建模思想，同时将数据、物理时间和评测协议严格对齐到 FITWMAMS。

---

# 31. 最重要的五条实现约束

1. **同一 canonical highD cache、同一 split、同一 149-step horizon。**
2. **测试时不能使用 GT destination 或任何 future trajectory constraint。**
3. **ego 必须 external-control；background 才是 world-model 输出。**
4. **必须把 dynamics dt 从 0.1 改成 0.04。**
5. **主实验必须关闭 128→32 least-collision filtering。**

只要这五条守住，TrafficBotsV1.5-HighD 就能作为本文可信且方法身份清晰的外部对比基线。
