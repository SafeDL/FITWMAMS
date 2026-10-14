# 长尾场景条件 Flow

将 EVT 筛出的长尾场景建成可连续采样的条件分布，为 ADS 测试提供初始交通状态和稀疏位置结点：

```text
p(M) · p(C0 | M) · p(K | C0,M)
```

- `M`：六个背景车槽位是否有车。
- `C0`：自车初始速度及背景车相对位置、速度，共 26 维。
- `K`：六辆背景车在 2、4、5.96 秒的位置结点，共 36 维。
- `D`：七辆车的长宽，按 `M` 从尾部经验分布采样，不进入 Flow 损失。

最终模型使用全部 2,964 条尾部事件拟合，不再切分这批目标分布；录像级五折用于单独检查折外表现。采样器输出 `(M,C0,K,D)`，再交给 Diffusion 构造软计划，不直接输出车辆动作。

## 运行

在项目根目录运行，参数见 [config.yaml](config.yaml)：

```bash
conda activate tread
python normalizing_flow/scripts/prepare.py
python normalizing_flow/scripts/train.py
python normalizing_flow/scripts/evaluate.py
python normalizing_flow/scripts/verify.py
```

`prepare.py` 生成全量共享场景条件，训练时只选择 EVT 尾部。权重、生成样本、拟合与五折评测见[结果说明](../results/flow/README.md)。
