# IDM 长尾评估：当前交通世界概率语义

本文以 IDM 作为固定 ADS policy，并在当前完整交通世界先验下估计安全关键事件概率：

```text
(M, C0, K) ~ p(M)p(C0|M)p(K|C0,M)
z_diff, z_response ~ standard base distributions
Flow -> Diffusion Soft Plan -> 25 Hz HiQR background policy -> HighwayEnv + IDM ego
Y = safety-envelope trajectory risk
failure = S_EVT(Y) >= S_EVT(z_100)
```

因此估计量是自然交通世界中的
`P_world(IDM safety-critical event)`，不再是旧 tail-context 条件概率。

## Subset simulation

每个 subset level 保留最高风险的 `p0` 样本。后续条件链对高斯 Flow/Diffusion/
response block 使用 pCN，对离散场景 block 使用独立 uniform refresh。proposal 相对
世界先验可逆，故只在其风险分数维持高于当前层阈值时接受。估计量为：

```text
P_hat = p0^(L - 1) * mean(final_score >= EVT_threshold)
```

每个最终高风险案例必须保存完整 `WorldExogenousState`、HighwayEnv 车辆状态和全部正式
工件哈希，才可逐点重放。

## Independent Monte Carlo

Monte Carlo 从同一完整世界先验独立采样，使用同一 HighwayEnv、IDM、EVT 模型和 final
checkpoint。
它只用于验证 subset estimate、报告抽样不确定性及展示效率差异；不定义另一套概率语义。

旧 following/cut-in 数值使用旧 tail-context 与旧 diffusion，不能作为本文当前方法的结果。
正式 HighwayEnv 估计使用 512 粒子 AMS 和 2,000 场景独立 Monte Carlo：
`P_AMS=3.4766%`（近似 95% CI 3.064%--3.889%），`P_MC=4.3000%`（95% CI
3.411%--5.189%），区间相交。IDM 使用原生 HighwayEnv 动力学，HiQR 背景车使用与训练
一致的单轮动作契约，并在同一 HighwayEnv 道路与碰撞系统中执行。此前 `current_world_idm`
的自定义运动学数值已移除，不是本文正式结果。
