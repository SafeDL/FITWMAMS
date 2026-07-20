# highD EVT-tail 归一化流实验报告

## 选定基线

最终选定配置：

```text
normalizing_flow/scripts/configs/highd_tail_flow_best.yaml
```

最终输出目录：

```text
results/highd_tail_flow_best/
```

最终 checkpoint：

```text
results/highd_tail_flow_best/checkpoints/best_tail_conditional_maf.pt
```

选定模型保持严格的 EVT-tail 目标分布。采样时对 `(mask_pattern, primary_slot)`
使用精确配额，并采用 latent 采样温度 `1.0295`。没有加入 full-natural 或
near-tail 行。

## 训练链路

```text
density_fit:             基础 strict-tail 加权密度拟合，max_epochs 240
likelihood_refinement:   从 density_fit 恢复，lr 1e-5，epochs 180
final_refinement:        从 likelihood_refinement 恢复，lr 5e-7，在 230 epochs 停止
ultra_low_lr_probe:      从选定 checkpoint 尝试 lr 1e-7；val-NLL 未提升
```

checkpoint 选择使用未加权的 strict-tail validation NLL。选定 checkpoint 将
validation NLL 从 `-88.3093` 提升到 `-88.3726`；最后的 `1e-7` 续训没有继续提升。

## 最终指标

```text
conditional rq-spline MAF    train -147.0504  val -88.3726  test -106.0621
GMM                          train -125.6069  val -43.3915  test  -65.9791
Gaussian                     train    1.4248  val  16.2438  test    6.7505
Copula                       train   30.3386  val  36.5070  test   32.6475
Unconditional RealNVP        train   78.3462  val  81.3035  test   79.1450
```

完整 EVT-tail 2209 vs 2209 复现：

```text
mean per-feature KS:          0.0928
mean Wasserstein:             0.3126
Pearson corr MAE:             0.0419
mask occupancy L1:            0.0000
primary-slot occupancy L1:    0.0000
invalid_rate:                 0.0000
overlap_rate:                 0.0000
negative_gap_rate:            0.0000
semantic_error_rate:          0.0000
sampling rejection_rate:      0.1421
```

按 slot 统计的 mean KS：

```text
same_front  0.0390
same_rear   0.0726
left_front  0.1392
left_rear   0.1455
right_front 0.0518
right_rear  0.1082
```

## 与上一轮 Refinement Baseline 对比

上一轮选定 baseline 在这里仅作为指标参考；当前保留的 baseline 是
`results/highd_tail_flow_best/`。

```text
test NLL:                  -105.9444 -> -106.0621
mean per-feature KS:          0.0946 -> 0.0928
mean Wasserstein:             0.3186 -> 0.3126
Pearson corr MAE:             0.0420 -> 0.0419
mask occupancy L1:            0.0000 -> 0.0000
primary-slot occupancy L1:    0.0000 -> 0.0000
sampling rejection_rate:      0.1408 -> 0.1421
```

按 slot 统计的 KS 变化如下：

```text
same_front  0.0388 -> 0.0390
same_rear   0.0746 -> 0.0726
left_front  0.1418 -> 0.1392
left_rear   0.1478 -> 0.1455
right_front 0.0526 -> 0.0518
right_rear  0.1119 -> 0.1082
```

选定 baseline 改善了密度和分布指标，但 rejection rate 小幅上升。rejection
之后的物理有效性仍保持完整。

## 温度校准

对选定 checkpoint，相关温度扫描结果为：

```text
temperature 1.0290: mean KS 0.0929, mean W 0.3130, corr 0.0419, rejection 0.1421
temperature 1.0295: mean KS 0.0928, mean W 0.3126, corr 0.0419, rejection 0.1421
temperature 1.0335: mean KS 0.0937, mean W 0.3122, corr 0.0415, rejection 0.1431
temperature 1.0345: mean KS 0.0935, mean W 0.3115, corr 0.0415, rejection 0.1431
```

选择温度 `1.0295`，因为它给出了最佳 mean KS，同时相比上一轮 refinement
baseline 仍改善 test NLL、Wasserstein 和相关性。更高温度会降低
Wasserstein/correlation 误差，但会恶化 KS 和 rejection。

## 被拒绝候选

```text
previous_refinement_baseline + temperature 1.02:
  test NLL -105.9444, mean KS 0.0946, mean W 0.3186, corr 0.0420, rejection 0.1408
likelihood_refinement_checkpoint + temperature 1.031:
  test NLL -105.9911, mean KS 0.0931, mean W 0.3124, corr 0.0419, rejection 0.1428
final_refinement + temperature 1.0345:
  test NLL -106.0621, mean KS 0.0935, mean W 0.3115, corr 0.0415, rejection 0.1431
ultra_low_lr_probe:
  validation NLL 未超过 -88.3726
additional_low_lr_probe:
  validation NLL 未超过上一轮 refinement baseline
alternate_seed_long_run:
  test NLL -109.6735, mean KS 0.1078, mean W 0.3787, corr 0.0433, rejection 0.1199
```

尽管 `alternate_seed_long_run` 的 NLL 明显更好，但生成的 strict-tail 分布指标更差，
尤其是 `left_rear`，因此被拒绝。

## 剩余问题

当前最大的 KS 误差仍来自横向加速度和 side-slot 一秒动作摘要。纵向相对距离仍主导
Wasserstein 误差。下一步应重点处理 `left_rear` 和 longitudinal-gap 变换，同时保持
精确配额采样和物理有效性不变。
