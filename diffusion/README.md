# 背景车轨迹扩散模型

根据初始状态 `C0`、车辆存在标记 `M` 和稀疏位置结点 `K`，生成最多六辆背景车未来 5.96 秒的位置软计划，共 149 个未来位置点。

过程是：PCHIP 保形插值构造参考 → 扩散模型生成位置残差 → 合成软计划。条件共 68 维，不包含车辆尺寸或自车未来轨迹。软计划由[世界模型](../hierarchical_world_model/README.md)的确定性跟踪器执行，需要调整时一次性交给 NPC 策略；自车由 ADS 或回放控制。

训练和标准化使用全部自然驾驶 Train，Val 选择权重，Test 评估，空车位的输入置零。数据直接读取[共享数据](../results/highd_shared_training_data/README.md)，本目录不复制轨迹。自然 Test 给定真实结点，衡量结点之间的重建精度；生成场景则使用 Flow 采样的结点。

| 代码 | 用途 |
| --- | --- |
| `src/data.py` | 共享数据读取、条件编码、参考与残差转换 |
| `src/model.py` | 条件编码器、去噪模型和采样 |
| `src/train.py`、`src/evaluation.py` | 训练、权重加载与评测 |
| `tests/` | 条件、数据划分和轨迹转换检查 |

参数见 [`config.yaml`](config.yaml)，在项目根目录依次运行：

```bash
conda activate tread
python diffusion/scripts/prepare.py
python diffusion/scripts/train.py
python diffusion/scripts/evaluate.py
python diffusion/scripts/verify.py
```

`prepare.py` 生成标准化参数与切入评测索引；中断后可用 `train.py --resume` 续训。当前正式权重与完整 Test 评测见[结果说明](../results/diffusion/README.md)，动作执行评测见[世界模型结果](../results/hierarchical_world_model/README.md)。数值以对应报告为准。
