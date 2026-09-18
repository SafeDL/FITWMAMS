# HDP-HSMM 分布歧义与迁移偏离

Wu et al. (2025) 第 5.2 节写明选择 Poisson observation distribution 与 Gaussian state-duration
distribution；但该文的观测 `s, v, Δv` 为连续且可负，duration 为正整数。该组合没有给出可执行的
变量转换或参数化，也没有可核对的作者推断代码。

因此本目录不会把有限 HMM/HSMM 伪称为论文 HDP-HSMM。Stage A 的明确适配为：

- train-only 标准化后的连续 `[gap, speed, closing_speed]` 使用对角 Gaussian emission；
- duration 使用截断的正整数 Poisson；
- 最多三个显式状态，禁止自转移，hard-EM/Viterbi 训练；
- 所有离线标签只用于回顾性上界和参数拟合，不能流入 online controller。

`stage_a_report.json` 会记录 truncated tail mass、各状态时长与样本数。获得论文作者的分布澄清或
完整实现前，`paper_exact_segmentation=blocked_ambiguity` 必须保持不变。
