# highD 场景条件 Flow

本模块在唯一的 96,055 条完整性筛查后自然驾驶序列上学习

\[
p(M)p(C_0\mid M)p(K\mid C_0,M).
\]

- `M` 是六背景槽位 mask，使用训练集经验 PMF。
- `C0` 是无未来泄漏的 40 维初始状态，使用条件 RQ-spline MAF。
- `K[6,12]` 是六车在 2、4、5.96 s 的物理状态结点，使用一个 72 维条件 RQ-spline MAF。

没有行为分类器、分阶段计划 Flow、`primary_slot` 或连续量经验 donor。纵向和横向模式只从采样后的 `K` 派生用于诊断，不属于生成链。EVT 仅作为外部人类风险标尺和分层标签，不筛选 Flow 数据，也不进入损失。

公共接口为：

- `sample_scenarios(n, scenario_seed)`：采样 `M/C0/K`；
- `sample_constraints(c0, slot_mask, n, scenario_seed)`：固定 `C0/M` 采样 `K`；
- `log_prob(c0, slot_mask, k)`：返回 `mask_log_prob`、`c0_log_prob`、`k_log_prob` 与严格求和的 `joint_log_prob`。

无效槽位在密度空间使用固定标准高斯参考坐标；投影前坐标用于严格概率计算，物理投影后的 `C0/K` 用于仿真。`scenario_seed` 确定地寻址 `z_scenario=(u_M,z_C0,z_K)`。

## 正式结果

recording-level split 为 72,771/13,133/10,151。模型在第 53 epoch 取得最佳验证联合 NLL `87.614`，第 65 epoch 早停。测试集结果为：

- 联合 NLL `92.589`，其中 mask/C0/K 为 `2.379/47.717/42.493`；
- K NLL 优于无条件 Gaussian `101.827` 和 GMM8 `94.831`；
- C0/K 平均 KS 为 `0.0842/0.1243`，K 相关性 MAE 为 `0.0301`；
- 派生行为模式平均 TV 为 `0.0409`；
- 投影后 C0、结点速度和纵向位移物理合法率均为 100%。

所有预设总体质量门槛均通过，但平均 KS 会掩盖少数后车纵向结点的较大偏差（单项最高约 `0.414`），不能据此声称每个条件边际都可靠。完整结果、采样和图见 `results/highd_natural_driving_flow/`。扩散仍接收 `C0(40)+M(6)+K(72)=118` 维物理契约，因此冻结扩散模型不需要重训。

```bash
conda run -n tread python normalizing_flow/scripts/prepare_highd_natural_driving_flow_dataset.py
conda run -n tread python normalizing_flow/scripts/train_highd_natural_driving_flow.py
conda run -n tread python normalizing_flow/scripts/evaluate_highd_natural_driving_flow.py
conda run -n tread python normalizing_flow/scripts/verify_highd_natural_driving_flow.py
```
