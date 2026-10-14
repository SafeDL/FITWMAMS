# 背景车轨迹扩散模型

根据初始状态 `C0`、车辆存在标记 `M` 和位置结点 `K`，生成最多六辆背景车未来 5.96 秒的位置软计划（149 个未来位置点）。

主链为：PCHIP 保形插值参考 → 条件扩散生成位置残差 → 合成软计划。条件共 68 维，不包含车辆尺寸 `D` 或自车未来轨迹；软计划交给 HiQR 与世界模型执行，自车由 ADS 或回放控制。

使用全部自然驾驶 Train 训练和标准化，空槽特征置零，Val 选择权重，Test 评估；划分为 72,771 / 13,133 / 10,151 条。Test 输入真实结点，验证的是给定结点后的轨迹重建，不是仅凭初始状态预测未来。

## 运行

参数见 [`config.yaml`](config.yaml)，在项目根目录依次运行：

```bash
conda activate tread
python diffusion/scripts/prepare.py
python diffusion/scripts/train.py
python diffusion/scripts/evaluate.py
python diffusion/scripts/verify.py
```

`prepare.py` 同时生成标准化参数与切入评测索引；中断后可用 `train.py --resume` 续训。权重、评测与 PNG 见[结果说明](../results/diffusion/README.md)，实际动作执行结果见[世界模型](../results/hierarchical_world_model/README.md)。
