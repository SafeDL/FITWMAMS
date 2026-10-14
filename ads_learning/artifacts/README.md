# 当前学习型 ADS 策略

唯一维护的策略方法是 DQN，验证集选中的训练时长为 10,000 步。
`policies/seed*/policy.zip` 保存三个独立训练种子的策略；同目录
`replay_buffer.pkl` 是与该 checkpoint 配对的续训状态。种子号是实验条件，
不是版本号。历史选策指标见
[`archive/results/paper_benchmark/selection.json`](../../archive/results/paper_benchmark/selection.json)；
该结果没有在当前世界权重上重跑，不作为当前主链结论。
新训练权重写入本模块，评测 CSV/JSON 写入 `results/`。
