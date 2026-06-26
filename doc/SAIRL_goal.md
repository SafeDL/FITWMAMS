# 目标：在长尾测试分布下评估 SAIRL 策略的安全性

本项目旨在在现有 TREAD 项目中新增一个 `SAIRL_subset` 目录，用于在相同的长尾测试分布和子集模拟设置下评估 **Safety‑Aware Adversarial Inverse Reinforcement Learning (SAIRL)** 策略的安全性。该评估应与现有 `IDM_subset` 目录中的实验保持公平可比，复用相同的扩散模型、初始场景参数、阈值定义以及子集模拟超参数。

系统的运行环境在:
conda activate tread

## 背景说明

- TREAD 项目中的 `IDM_subset` 目录实现了针对 **跟驰 (car‑following)** 和 **切入 (cut‑in)** 两类场景的长尾测试分布生成、子集模拟概率估计以及风险重现里程计算。该管线包含：通过极值理论定义风险阈值，使用扩散模型生成长尾测试样本，以及应用子集模拟 (subset simulation) 估计稀有事件概率等模块。被估计的是highway-env的内置IDM策略.
- `Safe_imitation_learning` 仓库提供了 SAIRL 算法的实现和预训练模型。SAIRL 在 highway‑env==1.2.0 环境中加入了安全约束，训练得到一个能主动规避碰撞的策略【130502817091876†L274-L305】。

## 总目标

在 TREAD 项目的根目录下创建 `SAIRL_subset` 文件夹，拷贝并改造 `IDM_subset` 目录中的脚本，使之能够加载 SAIRL 策略并在相同的长尾测试分布与子集模拟参数下完成跟驰和切入场景的安全评估。评估结果应包括稀有事件越阈概率 \(p_{ADS,e}\)、风险重现里程 \(\lambda_{ADS,e}\) 及其置信区间，便于与基线策略做横向比较。

## 任务拆解

1. **复制基线脚本**：在 `SAIRL_subset` 中复制 `IDM_subset` 目录下与长尾测试有关的所有脚本和模块，例如长尾分布生成器、子集模拟实现、风险度量函数、环境配置等。保持目录结构一致（如 `configs/`、`utils/`、`diffusion/` 等），便于后续代码的复用与修改。被测试的ADS脚本也应该在`SAIRL_subset` 中复制和实现.

2. **实现 SAIRL 策略加载**：新增 `sairl_policy.py`，实现加载 SAIRL 预训练模型的功能。
   - 在 `Safe_imitation_learning` 仓库的 `trained_models` 目录中选取合适的 checkpoint，并在该文件中提供一个默认的 `checkpoint_path` 参数；如没有模型，则留出占位符并在文档中说明需要用户提供。
   - 使用 TensorFlow 1.x API 恢复模型参数【130502817091876†L274-L305】。并需要将策略迁移到 PyTorch，可在此文件中定义网络结构并从 TensorFlow 权重中读取变量。
   - 定义 `SAIRLPolicy` 类，提供 `__init__(self, checkpoint_path)`、`reset(self)` 和 `act(self, observation)` 方法，使其能在每个时间步返回与 `highway-env` 动作空间兼容的控制信号（例如连续加速度或离散动作）。

3. **编写策略适配器**：为了让子集模拟管线能调用 SAIRL 策略，定义一个简单的包装函数，例如：

```python
from sairl_policy import SAIRLPolicy

def policy_fn(obs, state):
    """返回 SAIRL 在观测 obs 下的动作。state 用于内部 RNN 状态，可忽略或返回 None。"""
    return sairl_policy.act(obs), state
```

4. **保持环境兼容**：TREAD 的 `IDM_subset` 管线可能使用 gymnasium/`highway-env` 较新版本。SAIRL 代码依赖于 `highway-env==1.2.0`【130502817091876†L274-L305】。如果版本不一致，需要参考代码并迁移在本工程的highway-env

5. **使用相同的长尾分布和阈值**：不要重新训练扩散模型或重新拟合 GPD 阈值。直接复用 `IDM_subset` 目录中训练好的扩散生成器和配置文件来生成长尾测试样本，并读取已经确定的风险阈值 \(x_e^*\)。统一随机种子，确保 SAIRL 与基线策略在同一批测试样本上评估。

6. **复用子集模拟设置**：在新脚本中使用与基线相同的子集模拟超参数，例如：
   - 初始样本数 `N`（例如 200）。
   - 存活概率 `p0`（例如 0.1 或 0.2）。
   - 提议分布标准差 `sigma_p`。
   - MCMC 样本跳变策略和接受准则。
   根据 `IDM_subset` 脚本中定义的接口填入参数，确保评估的方差/置信区间计算方式一致。

7. **编写评估脚本**：在各事件入口中实现 SAIRL 评估流程：
   - car-following 使用 `SAIRL_subset/scripts/run_subset_following.py`；
   - cut-in 使用 `SAIRL_subset/scripts/run_subset_cutin.py`；
   - 解析命令行参数：`--checkpoint_path`、`--seed`、子集模拟超参数等；
   - 加载扩散模型和阈值配置；
   - 根据脚本对应事件创建相应的长尾测试分布生成器；
   - 实例化 SAIRL 策略和对应的 `highway-env` 环境；
   - 调用子集模拟函数，估计 SAIRL 策略在该事件下的越阈概率和风险强度；
   - 输出估计值、标准误或置信区间，并将结果写入 `SAIRL_subset/results/` 下的 `sairl_{event}_result.json`。

   - 与基线结果比较时需保持相同随机种子和参数设置。


## 输出要求

- `SAIRL_subset` 目录完整存在于仓库根目录，包含复制的基线脚本、SAIRL 策略加载器和评估脚本。
- 本文件 `SAIRL_goal.md` 清楚描述了任务目标、设计思路、具体步骤和使用方法，便于开发者按此实现。
