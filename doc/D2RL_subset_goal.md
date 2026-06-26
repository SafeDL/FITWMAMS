# Codex Goal: 在 TREAD 中新增 D2RL_subset，用于基于强化学习 ADS 策略的长尾安全评估

## 0. 任务背景

参考代码：

- RL / RL policy reference: `https://github.com/nagasriramnani/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-RL`
- 本文代码：`https://github.com/SafeDL/TREAD`

本文当前的 `IDM_subset/` 目录已经实现了在长尾测试分布下的自动驾驶安全评估流程。现在需要在仓库根目录下新增一个文件夹，用于评估新的基于强化学习的 highway-env 自动驾驶策略。该策略主要来自 RL 参考代码中的 PPO / A2C / Behavior Cloning 等 learned ADS policy。

由于本文的实验核心是 **car-following** 和 **cut-in** 两类高速公路交互场景，新增实验必须保证与现有 `IDM_subset/` 管线公平可比：

1. 不重新训练或更改扩散模型；
2. 不更改初始场景参数分布；
3. 不更改 EVT/GPD 风险阈值；
4. 不更改子集模拟算法和默认超参数；
5. 只替换被测 ADS policy，即将原有默认 ADS 策略替换为 RL 或者是 RL 策略。

代码必须在系统的conda环境中运行:
conda activate tread

如项目中有多个自动驾驶策略可以参考,则先实现其中一个;

---

## 1. 总目标

参照现有 `IDM_subset/` 目录的脚本、配置和数据接口，实现基于 RL/RL policy 的长尾安全评估。该模块应能够在相同长尾测试分布下，对 learned ADS policy 进行闭环仿真，并通过子集模拟估计其越过人类驾驶校准风险阈值的概率。

最终需要支持两个场景：

```text
car_following
cut_in
```

每个场景至少应输出：

```text
p_hat              # ADS 在长尾测试分布下超过风险阈值的概率估计
rse / se           # 相对标准误或标准误
threshold          # 使用的 EVT/GPD 人类驾驶风险阈值
return_mileage     # 风险重现里程或等价 exposure-normalized risk metric
num_evaluations    # 实际闭环仿真调用次数
subset_levels      # 子集模拟分层阈值序列
acceptance_rate    # MCMC 接受率，如现有 subset 支持
seed               # 随机种子
policy_name        # RL-PPO / RL-A2C / RL-BC 等
```

---

## 3. 必须复用的现有 subset 组件

实现时应优先复用现有 `IDM_subset/` 目录中的代码，不要重写已有功能。需要从 `IDM_subset/` 中查找并复用以下模块或等价脚本：

1. 长尾测试分布采样器；
2. 扩散模型加载与 DDIM/DDPM 采样函数；
3. 初始场景参数恢复函数；
4. car-following 和 cut-in 场景 reset / rollout 逻辑；
5. 风险变量计算函数；
6. EVT/GPD 阈值读取函数；
7. 子集模拟主循环；
8. MCMC proposal 与接受率统计；
9. Monte Carlo baseline 评估函数；
10. 结果保存与日志格式。

新增文件夹不应改变 `IDM_subset/` 原有实验结果。所有改动应局限在新增目录或必要的轻量接口抽象中。

---

## 4. 公平评估原则

新增的字段驾驶策略评估必须满足以下公平性约束。

### 4.1 扩散模型保持一致

不得重新训练扩散模型。必须使用与现有 `IDM_subset/` 实验相同的：

```text
model checkpoint
normalization statistics
condition/context encoder
latent dimension
sampling steps
DDIM/DDPM setting
random seed
trajectory horizon
```

如果现有 `IDM_subset/` 中分别有 car-following 和 cut-in 的扩散模型，则新实现必须按场景分别加载对应模型。

### 4.2 初始场景参数保持一致
自动驾驶的闭环测试必须使用与原 `IDM_subset/` 相同的初始场景参数，包括但不限于：

```text
ego 初始速度
target/front vehicle 初始速度
relative distance / gap
relative speed
lane index
cut-in vehicle lateral position
cut-in vehicle lateral velocity
scenario duration
simulation frequency
policy frequency
vehicle size / dynamics constraints
```

如果原 `IDM_subset/` 使用 Copula 或 empirical tail context 采样初始条件，则 RL policy必须复用同一个 context sampler。

### 4.3 风险变量和阈值保持一致

不得修改以下内容：

```text
Y_following / Y_long 的定义
Y_cutin 的定义
collision / near-collision / TTC / THW / DRAC 等风险分量
风险聚合权重
EVT/GPD 参数
human-calibrated threshold x_e^*
return mileage 映射方式
```

RL/RL policy 只影响闭环响应轨迹，不影响风险评价标尺。

### 4.4 子集模拟设置保持一致

默认使用与原 `subset/` 完全一致的子集模拟设置：

```text
N                 # 每层样本数
p0                # 每层保留比例
max_levels        # 最大层数
proposal_sigma    # MCMC proposal 标准差
burn_in           # 如有
chain_length      # 如有
random_seed
failure_threshold
score_function
stopping rule
```

若为了调试允许覆盖参数，必须在结果 JSON 中完整记录，并在 README 中说明这些参数会影响公平比较。

### 4.5 同一批测试样本比较

为了可比性，建议增加 `--sample_cache` 参数。第一次运行时保存长尾测试样本及 latent/context：

```text
results/cache/{event}_seed{seed}_samples.npz
```

后续不同 policy 使用同一份缓存进行评估，避免因采样差异造成不公平比较。

---

## 5. RL/RL policy 集成要求

### 5.1 支持的 policy 类型

至少支持以下一种策略：

```text
RL-PPO
```

建议预留接口支持：

```text
RL-A2C
RL-Behavior-Cloning
RL-PPO-perturbed
```

配置示例：

```yaml
policy:
  name: d2rl_ppo
  backend: stable_baselines3
  checkpoint_path: external_models/d2rl/highway_ppo/model.zip
  deterministic: true
  longitudinal_only: true
  lane_change_handling: map_to_idle
```

### 5.2 策略加载器

实现：

```text
D2RL_subset/policies/d2rl_policy.py
```

建议接口：

```python
class D2RLPolicy:
    def __init__(self, checkpoint_path, policy_type="ppo", deterministic=True, config=None):
        ...

    def reset(self):
        ...

    def act(self, observation):
        """Return action compatible with the TREAD/highway-env rollout loop."""
        ...
```

要求：

1. 能加载 RL 参考仓库中的 PPO/A2C checkpoint；
2. 能处理 stable-baselines3 `.zip` 模型；
3. 如果 checkpoint 不存在，应给出清晰错误信息，而不是静默失败；
4. 支持 deterministic evaluation；
5. evaluation 中不得继续训练 policy。

### 5.3 观测适配器

RL 参考代码通常使用 highway-env 的 5×5 kinematics observation，即 ego vehicle 加周围若干车辆的位置/速度特征。TREAD 中的 `subset` 场景观测格式可能不同，因此需要实现：

```text
D2RL_subset/policies/observation_adapter.py
```

建议接口：

```python
def convert_tread_obs_to_d2rl_obs(tread_obs, env, config):
    """Convert TREAD/highway-env observation to RL expected observation."""
    ...
```

要求：

1. 保持特征顺序与 RL 训练时一致；
2. 保持归一化方式与 RL 训练时一致；
3. 处理车辆数量不足时的 padding；
4. 处理车辆数量过多时的 nearest-neighbor selection；
5. 在 `tests/test_observation_adapter.py` 中检查输出 shape。

### 5.4 动作适配器

RL 参考代码一般使用 highway-env 的 5 个离散高层动作：

```text
LANE_LEFT
IDLE
LANE_RIGHT
FASTER
SLOWER
```

但本文 car-following 和 cut-in 安全评估主要关注纵向响应。若允许 learned policy 主动换道，可能会改变场景语义，使结果不再与原始 car-following/cut-in 风险评估完全可比。因此必须提供两种评估模式：

#### 模式 A：longitudinal_only=true（默认，推荐）

将横向动作映射为 `IDLE`：

```text
LANE_LEFT  -> IDLE
LANE_RIGHT -> IDLE
FASTER     -> FASTER
SLOWER     -> SLOWER
IDLE       -> IDLE
```

该模式用于公平评估 RL 策略在 car-following 和 cut-in 场景中的纵向安全响应。

#### 模式 B：longitudinal_only=false（可选）

允许策略执行完整离散动作。该模式可作为补充分析，但不得作为与原始 `subset` 主结果比较的默认模式。

实现：

```text
D2RL_subset/policies/action_adapter.py
```

并在结果中记录：

```text
longitudinal_only
lane_change_handling
```

---

## 6. 评估脚本要求

评估脚本流程：

1. 读取原 `subset` 配置；
2. 读取 RL policy 配置；
3. 加载对应场景的扩散模型、归一化器、Copula/context sampler 和 EVT 阈值；
4. 创建与原 `subset` 相同的 highway-env/TREAD 场景；
5. 加载 RL policy；
6. 将 RL policy 注入 rollout/evaluate 函数；
7. 调用原 `subset` 子集模拟函数；
8. 保存结果 JSON、逐样本 CSV/NPZ 和日志。


---

## 12. 不应做的事情

实现过程中不要做以下事情：

1. 不要重新训练扩散模型；
2. 不要重新拟合 EVT/GPD；
3. 不要改变风险变量定义；
4. 不要默认允许 RL 主动换道参与主结果；
5. 不要修改原 `subset/` 结果；
6. 不要将 RL reward 用作测试风险指标；
7. 不要在评估过程中继续训练 RL policy；
8. 不要用随机策略作为主对照；
9. 不要把 RL 自身训练环境的普通 highway-v0 分布与本文长尾测试分布混淆。

---

## 13. README.md 应包含的内容

`D2RL_subset/README.md` 至少包含：

1. 实验目的；
2. 与原 `subset/` 的关系；
3. RL checkpoint 放置路径；
4. 安装依赖说明；
5. car-following 运行命令；
6. cut-in 运行命令；
7. 如何启用/禁用 `longitudinal_only`；
8. 输出文件解释；
9. 与 baseline policy 比较的方法；
10. 公平性约束说明。

---


