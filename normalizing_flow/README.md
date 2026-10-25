# 长尾场景条件 Flow

将 EVT 筛出的长尾场景建成可采样的条件分布，为世界模型提供初始交通状态和稀疏位置结点：

```text
p(M) · p(C0 | M) · p(K | C0,M)
```

- `M`：六个背景车槽位是否有车。
- `C0`：自车初始速度及背景车相对位置、速度，共 26 维。
- `K`：六辆背景车在 2、4、5.96 秒的位置结点，共 36 维。
- `D`：七辆车的长宽，按 `M` 从尾部经验分布采样，不进入 Flow 损失。

最终模型使用全部尾部事件拟合目标分布，另用录像级五折检查折外表现。采样结果 `(M,C0,K,D)` 交给 [Diffusion](../diffusion/README.md) 构造软计划。

| 代码 | 用途 |
| --- | --- |
| `src/data.py`、`src/features.py` | 共享场景条件、录像划分和初态提取 |
| `src/constraints.py`、`src/transforms.py` | 位置结点与模型坐标转换 |
| `src/scenario.py`、`src/model.py` | 分布模型、采样和权重加载 |
| `src/train.py`、`src/evaluation.py`、`src/metrics.py` | 全量拟合、五折评测和分布检查 |
| `tests/` | 条件分解、归一化与物理投影检查 |

运行接口直接使用 `DirectScenarioFlow.sample_scenarios()`；固定初态时使用 `sample_constraints()`；配对测试可通过 `sample_constraints_from_base_randomness()` 提供相同随机输入。世界模型统一调用这些方法。

在项目根目录运行，参数见 [config.yaml](config.yaml)：

```bash
conda activate tread
python normalizing_flow/scripts/prepare.py
python normalizing_flow/scripts/train.py
python normalizing_flow/scripts/evaluate.py
python normalizing_flow/scripts/verify.py
```

自然分布使用 [natural_config.yaml](natural_config.yaml)，运行 `python normalizing_flow/scripts/train_natural.py`。预处理、车辆模式与尺寸支持均使用 Train＋Val，训练轮数取自配置的 `training.epochs`，权重和训练报告保存到 `results/natural_flow/`。

`prepare.py` 生成[全量共享场景条件](../results/highd_shared_training_data/README.md)，训练时只选择 EVT 尾部。当前正式权重、固定种子样本、拟合与五折评测见[结果说明](../results/flow/README.md)，数值以原始报告为准。
