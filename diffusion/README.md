# highD 长时程约束条件扩散模型

本模块用于回答一个明确的问题：给定无未来泄漏的 40 维场景初始条件、六槽位 mask，以及背景车在 2 s、4 s、5.96 s 的稀疏状态约束，能否生成平滑、物理合理且具有条件随机性的六车联合轨迹。

扩散接口只有三类条件，共 118 维：

- 40 维 `C0`；
- 6 维背景槽位 mask；
- 每辆背景车 12 维长时程约束，即三个时刻的相对 `dx, dy, dvx, dvy`。

条件中没有未来 ego 状态。三个背景状态结点属于显式的 oracle 长时程约束，因此这里评估的是“长时程约束下的轨迹重建”，不是仅凭 `C0` 预测未来。尤其是末端状态已被条件锚定，FDE 必须连同这一条件定义解释。

## 模型

稀疏状态结点通过三次 Hermite 插值得到物理参考。条件扩散模型不直接预测动作，也不硬执行参考轨迹，而是生成参考上的 `[149, 6, 2]` 平滑位置残差。固定的三阶 Savitzky–Golay 基（41 帧窗口）同时用于监督目标和解码，速度与加速度由同一平滑轨迹求导，避免“位置精度高但数值微分产生不合理 jerk”的伪改进。

去噪器是紧凑的因子化时空 Transformer：交替执行单车时间注意力、同帧车辆交互注意力和条件交叉注意力。`motion_seed` 控制扩散噪声；相同条件和 seed 可逐点复现，不同 seed 用于衡量同条件运动多样性。

## 输入与 Flow 组合边界

118 维条件按固定顺序组成：

- `0:40`：标准化后的物理 `C0`。前 4 维是 ego 的 `vx, vy, ax, ay`；随后每个固定槽位 6 维 `dx, dy, dvx, dvy, ax, ay`。槽位顺序为 `same_front, same_rear, left_front, left_rear, right_front, right_rear`。
- `40:46`：六个槽位的存在性 mask。
- `46:118`：六车各 12 维状态结点，即 2.00、4.00、5.96 s 时相对自身初始状态的 `dx, dy, dvx, dvy`。

这三类条件统一记为 `(C0,M,K)`。Flow 的 `sample_scenarios()` 输出物理 40 维 `C0`、mask `M`、`[6,12]` 长时程状态结点 `K` 和缺失槽位的参考坐标；`prepare_flow_condition()` 可将任一样本直接转换为模型所需的 118 维条件。外部场景生成器也可以通过 `prepare_external_condition()` 提供相同物理量。缺失槽位在 Flow 密度中使用标准高斯参考测度，因此外部调用还必须给出可复现的 `inactive_seed`，或者传入已有的标准化无效坐标。

长时程结点由单个 72 维条件 RQ-spline MAF 一次生成；行为模式只从生成后的 `K` 派生用于统计，不属于生成链，也不使用连续经验 donor。

对外只需要下面的统一条件模型：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M).
\]

当前 Flow 直接建模 `p(M)p(C0|M)p(K|C0,M)`，与扩散条件逐点兼容。仍需区分接口可组合性与实验结论：扩散模型正式精度来自 highD 日志提供的 oracle 状态结点；Flow→扩散生成只评价密度、物理合法性和分布，不与任意单条日志强行计算 ADE。

## 数据与评估边界

- 唯一自然驾驶数据：96,055 条完整性筛查后的 highD 六秒序列；
- recording-level split：72,771/13,133/10,151；
- 训练最多 50 epoch，patience 6，只按验证损失选择 checkpoint；
- 最终测试覆盖完整 10,151 条测试序列，并分别报告随机样本、样本均值、最优样本、零潜变量、动作分布、物理合法性、EVT 与换道子集。

设计审计位于 `results/background_diffusion/design_audit/`，用于验证当前稀疏状态结点和运动基底的设计依据。正式结果只保留当前 118 维模型。
最佳 checkpoint 仅保存 EMA 推理权重、模型配置和数据契约；优化器状态只在训练未完成时作为临时恢复文件存在，训练正常结束后自动删除。

## 当前正式结果

全量训练覆盖 72,771 条训练序列和 13,133 条验证序列，第 50 轮取得最佳验证损失 0.06064；训练未早停，但第 30 轮后收益已明显减小。完整 10,151 条测试序列、每条件 4 次采样、50 步 DDIM 的结果为：

- ensemble ADE/FDE：0.02637/0.00463 m；
- 单次随机样本平均 ADE/FDE：0.03089/0.00906 m；
- `vx/vy` MAE：0.04054/0.01028 m/s；
- 严格 crossing-aligned cut-in（122 条序列、567 个窗口）ensemble ADE/FDE：0.02876/0.02684 m；
- 所有非有限值、负纵向速度、加速度与 jerk 阈值越界率均为 0。

事实重建已超过旧协议 0.059/0.103 m 的历史参照，但两者数据协议不同，而且本模型显式条件化末端状态，不能把低 FDE 解释成 C0-only 预测能力。四次采样的同条件平均成对轨迹距离仅 0.02515 m，说明模型有可复现随机性但多样性仍弱；加速度边际接近自然数据，jerk 的 KS 仍偏高。当前结论因此限定为“可靠的长时程约束轨迹重建器”，不宣称已经是充分多模态或反事实正确的 ADS 闭环世界模型。

“5 cm 内”是总体与绝大多数序列意义上的结果，不是逐条绝对上界。单次随机样本的逐序列 ADE 中位数、P90、P95、P99 分别为 0.02993、0.04068、0.04605、0.06239 m；96.75% 的序列低于 5 cm，最大值为 0.334 m。Hermite 参考本身的 ADE 已为 0.02532 m，说明高精度主要来自显式状态结点；扩散层负责拟合参考上的平滑残差并提供受控随机性，而不是独自从 C0 推断整段未来。

```bash
conda run -n tread python diffusion/scripts/prepare_background_diffusion_data.py
conda run -n tread python diffusion/scripts/prepare_cutin_cohort.py
conda run -n tread python diffusion/scripts/train_background_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_background_diffusion.py
conda run -n tread python diffusion/scripts/visualize_background_diffusion_testset.py
conda run -n tread python diffusion/scripts/verify_background_diffusion_results.py
```
